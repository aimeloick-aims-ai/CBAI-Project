"""Prepare size-bounded acquisition proposals without transferring image data."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    metadata = {r["slide_id"]: r for r in read(ROOT / "data/bracs_slide_metadata.csv")}
    groups = defaultdict(list)
    all_rows = []
    for r in read(ROOT / "data/bracs_roi_inventory.csv"):
        if r["roi_label"] not in ("N", "IC"):
            continue
        p = metadata[r["slide_id"]]["patient_id"]
        path = ROOT / "data/raw/BRACS/RoI" / r["split"] / r["category"] / r["image_id"]
        r = {**r, "patient_id": p, "bytes": int(r["bytes"]), "local": path.exists()}
        all_rows.append(r)
        if not r["local"]:
            groups[p].append(r)
    # Round-robin patients gives new patient coverage priority over repeated regions.
    for rows in groups.values():
        rows.sort(
            key=lambda r: hashlib.sha256(("GCIA-expand:" + r["image_id"]).encode()).hexdigest()
        )
    ordering = sorted(
        groups, key=lambda p: hashlib.sha256(("GCIA-expand:" + p).encode()).hexdigest()
    )
    out = ROOT / "reports/bracs_expansion"
    out.mkdir(parents=True, exist_ok=True)
    plans = []
    for name, budget in [
        ("500MB", 500_000_000),
        ("1GB", 1_000_000_000),
        ("5GB", 5_000_000_000),
        ("all", sum(r["bytes"] for r in all_rows)),
    ]:
        remaining = budget
        selected = []
        for depth in range(max(map(len, groups.values()), default=0)):
            for p in ordering:
                if depth >= len(groups[p]):
                    continue
                row = groups[p][depth]
                if row["bytes"] <= remaining:
                    selected.append(row)
                    remaining -= row["bytes"]
        if selected:
            with (out / f"{name}_proposal.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=selected[0])
                writer.writeheader()
                writer.writerows(selected)
        plans.append(
            {
                "budget": name,
                "new_images": len(selected),
                "additional_bytes": budget - remaining,
                "patients_in_selected_files": len({r["patient_id"] for r in selected}),
                "normal": sum(r["roi_label"] == "N" for r in selected),
                "invasive": sum(r["roi_label"] == "IC" for r in selected),
            }
        )
    result = {
        "scope": "metadata_only_no_download",
        "available_images": len(all_rows),
        "available_patients": len({r["patient_id"] for r in all_rows}),
        "total_bytes": sum(r["bytes"] for r in all_rows),
        "proposals": plans,
        "limits": [
            "Inventory contains official train/val only.",
            "Budget filtering biases file sizes; no model-score based selection.",
            "Historical test patients remain exposed; acquisition does not define a new test.",
        ],
    }
    (out / "options.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
