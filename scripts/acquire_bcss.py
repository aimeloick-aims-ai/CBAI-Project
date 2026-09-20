"""Acquire bounded public BCSS regions through the authors' documented API."""

import argparse
import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
API = "https://demo.kitware.com/histomicstk/api/v1"
FOLDER = "5bbdeba3e629140048d017bb"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url, target, max_bytes=32 * 1024 * 1024, budget=None):
    """Cache only verified complete files; preserve existing originals."""
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_name(target.name + ".receipt.json")
    if target.exists():
        if not receipt.exists():
            raise ValueError(f"Existing file has no acquisition receipt: {target}")
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record["source_url"] != url or record["sha256"] != sha256(target):
            raise ValueError(f"Source/checksum mismatch: {target}")
        return record
    request = urllib.request.Request(url, headers={"User-Agent": "GCIA-acquisition/0.1"})
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as stream:
        observed = 0
        while chunk := response.read(256 * 1024):
            observed += len(chunk)
            if budget is not None:
                budget[0] -= len(chunk)
                if budget[0] < 0:
                    raise ValueError("Additional download budget exhausted")
            if observed > max_bytes:
                raise ValueError("Response exceeds the bounded region download limit")
            stream.write(chunk)
        length = response.headers.get("Content-Length")
        if length and observed != int(length):
            raise ValueError("Incomplete HTTP response")
    if not observed:
        raise ValueError("Empty HTTP response")
    partial.replace(target)
    record = {
        "source_url": url,
        "sha256": sha256(target),
        "bytes": observed,
        "acquired_at": datetime.now(UTC).isoformat(),
        "checksum_kind": "locally_computed_not_provider_digest",
    }
    receipt.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bcss_acquisition.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-additional-mib", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "data/bcss_smoke_manifest.csv")
    args = parser.parse_args()
    if args.max_additional_mib <= 0:
        raise ValueError("Download budget must be positive")
    budget = [args.max_additional_mib * 1024**2]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    limit = args.limit if args.limit is not None else config["limit"]
    if not 1 <= limit <= 151 or config["mpp"] <= 0:
        raise ValueError("Expected 1..151 regions and positive MPP")
    raw = (ROOT / config["raw_directory"]).resolve()
    if not raw.is_relative_to(ROOT / "data/raw"):
        raise ValueError("Raw directory must stay in project data/raw")
    commit = config["metadata_commit"]
    base = f"https://raw.githubusercontent.com/PathologyDataScience/BCSS/{commit}"
    for filename in ("roiBounds.csv", "gtruth_codes.tsv"):
        fetch(f"{base}/meta/{filename}", raw / "metadata" / filename, budget=budget)
    fetch(f"{base}/README.md", raw / "metadata/official_README.txt", budget=budget)
    fetch(
        f"{API}/item?folderId={FOLDER}&limit=1000",
        raw / "metadata/server_items.json",
        budget=budget,
    )
    items = json.loads((raw / "metadata/server_items.json").read_text())
    with (raw / "metadata/roiBounds.csv").open(encoding="utf-8", newline="") as stream:
        regions = list(csv.DictReader(stream))[:limit]
    manifest = []
    for row in regions:
        roi_id = row[""]
        if not re.fullmatch(r"[A-Za-z0-9-]+", roi_id):
            raise ValueError("Unexpected region identifier")
        patient_id = roi_id[:12]
        matches = [item for item in items if item["name"].startswith(patient_id + "-")]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous or absent slide for {roi_id}")
        item = matches[0]
        bounds = {key: int(float(row[key])) for key in ("xmin", "xmax", "ymin", "ymax")}
        query = urllib.parse.urlencode(
            {
                "left": bounds["xmin"],
                "right": bounds["xmax"],
                "top": bounds["ymin"],
                "bottom": bounds["ymax"],
                "mm_x": config["mpp"] / 1000,
                "mm_y": config["mpp"] / 1000,
                "encoding": "PNG",
            }
        )
        fetch(f"{API}/item/{item['_id']}/tiles", raw / "metadata" / f"{roi_id}.json", budget=budget)
        sources = {
            "image": f"{API}/item/{item['_id']}/tiles/region?{query}",
            "mask": row["mask_link"],
        }
        for kind, url in sources.items():
            path = raw / ("images" if kind == "image" else "masks_native") / f"{roi_id}.png"
            print(f"Acquisition {kind}: {roi_id}", flush=True)
            record = fetch(url, path, budget=budget)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            manifest.append(
                {
                    "dataset": "BCSS",
                    "scope": "acquisition_smoke_only",
                    "roi_id": roi_id,
                    "patient_id": patient_id,
                    "slide_id": item["name"],
                    "kind": kind,
                    "path": path.relative_to(ROOT).as_posix(),
                    "width": width,
                    "height": height,
                    "requested_mpp": config["mpp"] if kind == "image" else None,
                    "bounds_base_pixels": json.dumps(bounds, sort_keys=True),
                    "metadata_commit": commit,
                    "license": "CC0-1.0",
                    **record,
                }
            )
    output = args.output
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Completed: {len(regions)} regions; {output}", flush=True)


if __name__ == "__main__":
    main()
