"""Inventory verified local downloads, including metadata; never scan external storage."""

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / "data/raw"
    records = []
    for receipt in sorted(raw.rglob("*.receipt.json")):
        path = receipt.with_name(receipt.name.removesuffix(".receipt.json"))
        record = json.loads(receipt.read_text(encoding="utf-8"))
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Acquisition checksum/size mismatch: {path}")
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                image.load()
        records.append(
            {
                "dataset": path.relative_to(raw).parts[0],
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": observed,
                "source_url": record["source_url"],
            }
        )
    if not records:
        raise ValueError("No completed acquisitions")
    with (ROOT / "data/checksums.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    partials = [p.relative_to(ROOT).as_posix() for p in raw.rglob("*.part")]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "verified_files": len(records),
        "verified_bytes": sum(r["bytes"] for r in records),
        "partial_files": partials,
        "status": "pass" if not partials else "incomplete_downloads",
        "datasets": {
            name: sum(r["dataset"] == name for r in records)
            for name in sorted({r["dataset"] for r in records})
        },
        "scope": "locally acquired files only; not full cohort completeness",
    }
    (ROOT / "reports/acquisition_checks.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return int(bool(partials))


if __name__ == "__main__":
    raise SystemExit(main())
