"""Acquire the approved 40-image manifest, without expanding its scope."""

import csv
import getpass
import json
import os
from collections import Counter
from ftplib import FTP

from bracs_data import HOST, RAW, ROOT, digest, retrieve
from PIL import Image


def main():
    proposal = ROOT / "reports/bracs_binary_plan/proposed_manifest.csv"
    with proposal.open(newline="") as f:
        rows = list(csv.DictReader(f))
    expected = Counter(
        {
            ("train", "N"): 10,
            ("train", "IC"): 10,
            ("validation", "N"): 5,
            ("validation", "IC"): 5,
            ("internal_test", "N"): 5,
            ("internal_test", "IC"): 5,
        }
    )
    if Counter((r["proposed_split"], r["roi_label"]) for r in rows) != expected:
        raise ValueError("Unexpected manifest allocation")
    if len({r["patient_id"] for r in rows}) != 40:
        raise ValueError("Patient overlap")
    cap = 250_000_000
    pending = sum(
        int(r["bytes"])
        for r in rows
        if not (RAW / "RoI" / r["split"] / r["category"] / r["image_id"]).exists()
    )
    if pending > cap:
        raise ValueError("Additional acquisition exceeds 250 MB hard ceiling")
    user = os.environ.get("BRACS_FTP_USER") or input("FTP username: ")
    password = os.environ.get("BRACS_FTP_PASSWORD") or getpass.getpass("FTP password: ")
    records = []
    with FTP() as ftp:
        ftp.connect(HOST, timeout=90)
        ftp.login(user, password)
        for i, row in enumerate(rows):
            path = (RAW / "RoI" / row["split"] / row["category"] / row["image_id"]).resolve()
            if not path.is_relative_to(RAW.resolve()):
                raise ValueError("Path outside raw directory")
            ftp.voidcmd("TYPE I")
            if ftp.size(row["remote_path"]) != int(row["bytes"]):
                raise ValueError("Server size changed since fixed inventory")
            record = retrieve(ftp, row["remote_path"], path, max_bytes=16 * 1024**2)
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                width, height = im.size
            records.append(
                {
                    **row,
                    "path": path.relative_to(ROOT).as_posix(),
                    "width": width,
                    "height": height,
                    "sha256": record["sha256"],
                }
            )
            print(f"{i + 1}/40 verified: {row['image_id']}", flush=True)
    output = ROOT / "data/bracs_binary_manifest.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    (ROOT / "reports/bracs_binary_plan/acquisition.json").write_text(
        json.dumps(
            {
                "images": len(records),
                "patients": 40,
                "verified_bytes": sum(int(r["bytes"]) for r in rows),
                "proposal_sha256": digest(proposal),
                "manifest_sha256": digest(output),
                "test_status": "integrity_verified_only_no_model_evaluation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
