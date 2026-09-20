"""Official BRACS FTP acquisition and patient metadata audit; credentials are ephemeral."""

import argparse
import csv
import getpass
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from ftplib import FTP
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/BRACS"
HOST = "histoimage.na.icar.cnr.it"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SPLITS = {"Training": "train", "Validation": "val", "Testing": "test"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_summary(path):
    with zipfile.ZipFile(path) as archive:
        strings = [
            "".join(node.itertext()) for node in ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(".//m:sheetData/m:row", NS):
        cells = {}
        for cell in row:
            value = cell.find("m:v", NS)
            if value is None:
                continue
            column = re.sub(r"\d", "", cell.attrib["r"])
            cells[column] = (
                strings[int(value.text)] if cell.get("t") == "s" else value.text
            ).strip()
        if not cells.get("A", "").startswith("BRACS_"):
            continue
        if not cells.get("B") or cells.get("E") not in SPLITS:
            raise ValueError(f"Missing patient or unknown split for {cells['A']}")
        rows.append(
            {
                "slide_id": cells["A"],
                "patient_id": cells["B"],
                "roi_count": int(float(cells.get("C", "0"))),
                "wsi_label": cells["D"],
                "official_split": SPLITS[cells["E"]],
            }
        )
    if not rows or len({row["slide_id"] for row in rows}) != len(rows):
        raise ValueError("Empty summary or duplicate slide identifiers")
    return rows


def audit_summary(rows):
    groups = defaultdict(set)
    roi_groups = defaultdict(set)
    for row in rows:
        groups[row["patient_id"]].add(row["official_split"])
        if row["roi_count"]:
            roi_groups[row["patient_id"]].add(row["official_split"])
    overlap = {patient: sorted(splits) for patient, splits in groups.items() if len(splits) > 1}
    roi_overlap = {p: sorted(s) for p, s in roi_groups.items() if len(s) > 1}
    report = {
        "source": "BRACS.xlsx/WSI_Information",
        "source_sha256": digest(RAW / "BRACS.xlsx"),
        "observed_slides": len(rows),
        "observed_patients": len(groups),
        "roi_total_in_summary": sum(row["roi_count"] for row in rows),
        "patient_overlap_all_slides": overlap,
        "patient_overlap_roi_slides": roi_overlap,
        "status": "pass" if not overlap else "requires_review_before_training",
        "scope": "summary_metadata_only; FTP image inventory not fully reconciled",
    }
    with (ROOT / "data/bracs_slide_metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "reports/bracs_metadata_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def retrieve(ftp, remote, target, max_bytes: int | None = 64 * 1024 * 1024):
    ftp.voidcmd("TYPE I")
    size = ftp.size(remote)
    if size is None or (max_bytes is not None and size > max_bytes):
        raise ValueError("Missing size or file exceeds configured transfer limit")
    url = f"ftp://{HOST}{remote}"
    receipt = target.with_name(target.name + ".receipt.json")
    if target.exists():
        if not receipt.exists():
            raise ValueError("Existing file lacks receipt")
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record["sha256"] != digest(target) or target.stat().st_size != size:
            raise ValueError("Cached file differs from receipt or server size")
        if record["source_url"] != url:
            raise ValueError("Cached file has a different source")
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with partial.open("wb") as stream:
        ftp.retrbinary("RETR " + remote, stream.write, blocksize=256 * 1024)
    if partial.stat().st_size != size:
        raise ValueError("Incomplete FTP transfer")
    partial.replace(target)
    record = {
        "source_url": url,
        "sha256": digest(target),
        "bytes": size,
        "acquired_at": datetime.now(UTC).isoformat(),
        "license": "non-commercial research; official portal terms supplied by user",
        "checksum_kind": "local_digest_and_server_size_not_provider_digest",
    }
    receipt.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def inventory_rois(ftp):
    """Inventory public BRACS RoI directories without downloading image pixels."""
    rows = []
    for split in ("train", "val"):
        base = f"/BRACS_RoI/latest_version/{split}"
        for remote_category in sorted(ftp.nlst(base)):
            category = Path(remote_category).name
            listing = []
            ftp.retrlines(f"LIST {base}/{category}", listing.append)
            for line in listing:
                parts = line.split(maxsplit=8)
                if not line.startswith("-") or len(parts) != 9:
                    continue
                name = parts[8]
                match = re.fullmatch(r"(BRACS_\d+)_(N|PB|UDH|FEA|ADH|DCIS|IC)_\d+\.png", name)
                if not match:
                    raise ValueError(f"Unexpected RoI filename: {name}")
                rows.append(
                    {
                        "split": split,
                        "category": category,
                        "slide_id": match[1],
                        "roi_label": match[2],
                        "image_id": name,
                        "bytes": int(parts[4]),
                        "remote_path": f"{base}/{category}/{name}",
                    }
                )
    if not rows:
        raise ValueError("Empty BRACS RoI FTP inventory")
    return rows


def write_inventory(rows):
    """Write a reproducible inventory and summary before a full acquisition."""
    target = ROOT / "data/bracs_roi_inventory.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    by_split = defaultdict(int)
    by_class = defaultdict(int)
    for row in rows:
        by_split[row["split"]] += 1
        by_class[row["roi_label"]] += 1
    report = {
        "scope": "FTP_metadata_inventory_only; no_images_downloaded",
        "images": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "by_split": dict(sorted(by_split.items())),
        "by_class": dict(sorted(by_class.items())),
        "inventory": target.relative_to(ROOT).as_posix(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (ROOT / "reports/bracs_roi_inventory.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def download_full_rois(ftp, rows, max_total_gib: float):
    """Download the inventoried development RoIs with per-file receipts and resume."""
    total = sum(row["bytes"] for row in rows)
    allowed = int(max_total_gib * 1024**3)
    if total > allowed:
        raise ValueError(f"Inventory is {total} bytes, exceeding --max-total-gib={max_total_gib}")
    summary = {row["slide_id"]: row for row in read_summary(RAW / "BRACS.xlsx")}
    records = []
    for index, row in enumerate(rows, start=1):
        metadata = summary.get(row["slide_id"])
        if metadata is None or metadata["official_split"] != row["split"]:
            raise ValueError(f"FTP inventory disagrees with summary for {row['image_id']}")
        target = RAW / "RoI" / row["split"] / row["category"] / row["image_id"]
        receipt = retrieve(ftp, row["remote_path"], target, max_bytes=None)
        with Image.open(target) as image:
            image.verify()
        with Image.open(target) as image:
            width, height = image.size
        records.append(
            {
                **metadata,
                "image_id": row["image_id"],
                "roi_label": row["roi_label"],
                "path": target.relative_to(ROOT).as_posix(),
                "width": width,
                "height": height,
                "bytes": row["bytes"],
                "scope": "full_development_acquisition; no_model_training",
                **receipt,
            }
        )
        if index % 25 == 0 or index == len(rows):
            print(f"Verified {index}/{len(rows)} BRACS RoI", flush=True)
    manifest = ROOT / "data/bracs_full_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Completed {len(records)} verified BRACS RoI: {manifest.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true")
    action.add_argument("--inventory", action="store_true")
    action.add_argument("--download-full", action="store_true")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--max-file-mib", type=int, default=16)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-total-gib", type=float, default=55.0)
    args = parser.parse_args()
    if not 1 <= args.per_class <= 10:
        raise ValueError("Pilot supports 1..10 images per class")
    if not 1 <= args.max_file_mib <= 64:
        raise ValueError("Pilot file limit must be between 1 and 64 MiB")
    RAW.mkdir(parents=True, exist_ok=True)
    if not args.download and not args.inventory and not args.download_full:
        audit_summary(read_summary(RAW / "BRACS.xlsx"))
        return
    username = os.environ.get("BRACS_FTP_USER") or input("FTP username: ")
    password = os.environ.get("BRACS_FTP_PASSWORD") or getpass.getpass("FTP password: ")
    records = []
    selection = []
    with FTP() as ftp:
        ftp.connect(HOST, timeout=90)
        ftp.login(username, password)
        if args.inventory:
            write_inventory(inventory_rois(ftp))
            return
        retrieve(ftp, "/BRACS.xlsx", RAW / "BRACS.xlsx")
        rows = read_summary(RAW / "BRACS.xlsx")
        audit_summary(rows)
        if args.download_full:
            inventory_path = ROOT / "data/bracs_roi_inventory.csv"
            if not inventory_path.exists():
                raise ValueError("Run --inventory before --download-full")
            with inventory_path.open(encoding="utf-8", newline="") as stream:
                inventory = list(csv.DictReader(stream))
            for row in inventory:
                row["bytes"] = int(row["bytes"])
            download_full_rois(ftp, inventory, args.max_total_gib)
            return
        slides = {row["slide_id"]: row for row in rows}
        base = f"/BRACS_RoI/latest_version/{args.split}"
        ftp.cwd(base)
        classes = sorted(ftp.nlst())
        if classes != ["0_N", "1_PB", "2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC"]:
            raise ValueError("Unexpected FTP class directories")
        for category in classes:
            ftp.cwd(base + "/" + category)
            listing = []
            ftp.retrlines("LIST", listing.append)
            entries = [line.split(maxsplit=8) for line in listing if line.startswith("-")]
            if any(len(entry) != 9 for entry in entries):
                raise ValueError("Unsupported FTP directory listing")
            eligible = [e[8] for e in entries if int(e[4]) <= args.max_file_mib * 1024 * 1024]
            files = sorted(eligible)[: args.per_class]
            if len(files) != args.per_class:
                raise ValueError("Insufficient files under pilot size limit")
            selection.append(
                {
                    "category": category,
                    "listed_files": len(entries),
                    "selected_files": files,
                    "max_file_mib": args.max_file_mib,
                    "skipped_large_files": [e[8] for e in entries if e[8] not in eligible],
                    "scope": "lexical bounded acquisition sample; not study exclusions",
                }
            )
            for name in files:
                match = re.fullmatch(r"(BRACS_\d+)_(N|PB|UDH|FEA|ADH|DCIS|IC)_\d+\.png", name)
                if not match or match[2] != category.split("_", 1)[1]:
                    raise ValueError("Unexpected image filename or class")
                metadata = slides[match[1]]
                if metadata["official_split"] != args.split:
                    raise ValueError("FTP partition disagrees with source summary")
                path = RAW / "RoI" / args.split / category / name
                print("Downloading " + name, flush=True)
                receipt = retrieve(ftp, base + "/" + category + "/" + name, path)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                records.append(
                    {
                        **metadata,
                        "image_id": name,
                        "roi_label": match[2],
                        "path": path.relative_to(ROOT).as_posix(),
                        "width": width,
                        "height": height,
                        "scope": "acquisition_smoke_only",
                        **receipt,
                    }
                )
    manifest = ROOT / "data/bracs_smoke_manifest.csv"
    existing = []
    if manifest.exists():
        with manifest.open(encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
    combined = {row["image_id"]: row for row in existing}
    combined.update({row["image_id"]: row for row in records})
    all_records = list(combined.values())
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(all_records)
    (ROOT / "reports/bracs_acquisition_plan.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(
        f"Completed: {len(records)} selected RoI; {len(all_records)} total pilot RoI; "
        "no model trained; no test images opened."
    )


if __name__ == "__main__":
    main()
