"""Offline patient-level inventory and bounded binary BRACS acquisition proposal."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    metadata = {r["slide_id"]: r for r in read_csv(ROOT / "data/bracs_slide_metadata.csv")}
    inventory = read_csv(ROOT / "data/bracs_roi_inventory.csv")
    seen_patients = set()
    # Conservatively treat every locally acquired RoI patient as previously exposed.
    for path in (ROOT / "data/raw/BRACS/RoI").rglob("*.png"):
        slide = "_".join(path.stem.split("_")[:2])
        if slide not in metadata:
            raise ValueError(f"Unknown local slide: {slide}")
        seen_patients.add(metadata[slide]["patient_id"])
    groups = defaultdict(list)
    patient_splits = defaultdict(set)
    for row in inventory:
        meta = metadata[row["slide_id"]]
        if row["split"] != meta["official_split"]:
            raise ValueError("Inventory/metadata partition mismatch")
        patient_splits[meta["patient_id"]].add(row["split"])
        if row["roi_label"] in ("N", "IC"):
            groups[(row["split"], row["roi_label"])].append(
                {**row, "patient_id": meta["patient_id"], "bytes": int(row["bytes"])}
            )
    overlaps = {p: sorted(s) for p, s in patient_splits.items() if len(s) > 1}
    if overlaps:
        raise ValueError(f"Patient leakage in RoI inventory: {overlaps}")
    counts = []
    for (split, label), rows in sorted(groups.items()):
        patients = {r["patient_id"] for r in rows}
        counts.append(
            {
                "official_split": split,
                "label": label,
                "roi": len(rows),
                "patients": len(patients),
                "unexposed_patients": len(patients - seen_patients),
                "total_bytes": sum(r["bytes"] for r in rows),
            }
        )
    # Proposal only: fixed patient hash order, single small ROI/patient, disjoint groups.
    # Internal validation and internal test come from official train; official val is excluded.
    # No claim that this is the official BRACS test, which has not been inventoried.
    candidates = defaultdict(dict)
    for label in ("N", "IC"):
        for row in groups[("train", label)]:
            if row["bytes"] > 16 * 1024**2:
                continue
            patient = row["patient_id"]
            old = candidates[label].get(patient)
            if old is None or (row["bytes"], row["image_id"]) < (old["bytes"], old["image_id"]):
                candidates[label][patient] = row
    selected, used = [], set()
    unmet = []
    # Reserve unexposed test patients first, without consulting image pixels or model scores.
    for split, quota in [("internal_test", 5), ("validation", 5), ("train", 10)]:
        for label in ("N", "IC"):
            eligible = [
                r
                for p, r in candidates[label].items()
                if p not in used and (split == "train" or p not in seen_patients)
            ]
            eligible.sort(
                key=lambda r: hashlib.sha256(
                    ("GCIA-binary-v1:" + r["patient_id"]).encode()
                ).hexdigest()
            )
            chosen = eligible[:quota]
            if len(chosen) < quota:
                unmet.append(f"{split}/{label}: {len(chosen)} of {quota} available")
            for row in chosen:
                used.add(row["patient_id"])
                path = (
                    ROOT / "data/raw/BRACS/RoI" / row["split"] / row["category"] / row["image_id"]
                )
                selected.append({**row, "proposed_split": split, "already_local": path.exists()})
    budget = 150 * 1024**2
    assert len(selected) == len({r["patient_id"] for r in selected})
    assert not {r["patient_id"] for r in selected if r["proposed_split"] != "train"} & seen_patients
    extra = sum(r["bytes"] for r in selected if not r["already_local"])
    summary = {
        "scope": "offline_inventory_and_proposal_no_download_no_training",
        "inventory_scope": "official train and val only; official test not inventoried",
        "counts": counts,
        "previously_exposed_patients": sorted(seen_patients),
        "patient_overlap_official_roi_partitions": overlaps,
        "proposed_patients": len(used),
        "proposed_images": len(selected),
        "additional_bytes": extra,
        "additional_mib": round(extra / 1024**2, 2),
        "budget_mib": 150,
        "within_budget": extra <= budget,
        "unmet_quotas": unmet,
        "selection": "one <=16MiB smallest RoI/patient; patient hash order; N then IC; test reserved first",
        "limits": [
            "Smallest-file sampling biases size/morphology; not a representative cohort.",
            "Mixed-label patients assigned only once; labels belong to RoIs, not patients.",
            "Internal test is not official test; ten patients cannot support strong precision.",
            "Normal vs invasive RoI is not comprehensive breast diagnosis.",
            "This is a proposed design, not a power-based sample-size calculation.",
        ],
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / "data/bracs_roi_inventory.csv", ROOT / "data/bracs_slide_metadata.csv"]
        },
    }
    out = ROOT / "reports/bracs_binary_plan"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "proposed_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selected[0])
        writer.writeheader()
        writer.writerows(selected)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
