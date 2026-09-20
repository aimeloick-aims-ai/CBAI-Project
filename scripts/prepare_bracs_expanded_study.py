"""Freeze patient-disjoint expansion splits, without reading new image pixels."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    previous = read(ROOT / "data/bracs_binary_manifest.csv")
    extension = read(ROOT / "data/bracs_expansion_manifest.csv")
    summary = json.loads((ROOT / "reports/bracs_expansion/completed_summary.json").read_text())
    test = set(summary["new_patient_ids_relative_to_previous_local_roi"])
    validation = {r["patient_id"] for r in previous if r["proposed_split"] == "validation"}
    assert not test & validation
    patients = defaultdict(list)
    for row in previous + extension:
        patients[row["patient_id"]].append(row)
    rows = []
    for pid, candidates in sorted(patients.items()):
        # One RoI per patient; smallest-file rule matches the prior pilot.
        chosen = min(candidates, key=lambda r: (int(r["bytes"]), r["image_id"]))
        split = "internal_test" if pid in test else "validation" if pid in validation else "train"
        rows.append(
            {
                k: chosen[k]
                for k in ("patient_id", "image_id", "roi_label", "path", "sha256", "bytes")
            }
            | {"proposed_split": split}
        )
    config = json.loads((ROOT / "configs/bracs_binary_protocol.json").read_text())
    config["splits"] = {
        s: sum(r["proposed_split"] == s for r in rows)
        for s in ("train", "validation", "internal_test")
    }
    config["custom_patient_split"] = (
        "22 previously unacquired patients reserved; original validation patients retained; historical test may be training now"
    )
    config["scope"] = (
        "expanded size-biased binary pilot, distinct new-patient test; not comparable head-to-head with old test"
    )
    config["selection_of_roi"] = (
        "smallest available N/IC image per patient, including mixed-label patients; label is regional"
    )
    config["class_counts"] = {
        s: {
            label: sum(r["proposed_split"] == s and r["roi_label"] == label for r in rows)
            for label in ("N", "IC")
        }
        for s in config["splits"]
    }
    path = ROOT / "data/bracs_expanded_study_manifest.csv"
    config_path = ROOT / "configs/bracs_expanded_protocol.json"
    if path.exists() or config_path.exists():
        raise FileExistsError("Preserve fixed split")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    config["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
