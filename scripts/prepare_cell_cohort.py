"""Build an exploratory HoVer-Net cell-graph cohort from the existing BRACS split."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tiatoolbox.models.architecture import get_pretrained_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graphs.cells import cell_graph
from src.segmentation.hovernet import infer_tiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.crop_size < 32:
        parser.error("crop-size must be at least 32")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Preserve cohort outputs; use a new directory")
    receipt = json.loads(args.weights.with_suffix(".receipt.json").read_text())
    if hashlib.sha256(args.weights.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    source = ROOT / "data/bracs_expanded_study_manifest.csv"
    rows = list(csv.DictReader(source.open()))
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("Patient duplication")
    if any(r["roi_label"] not in {"N", "IC"} for r in rows):
        raise ValueError("Expected binary N/IC labels")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if args.resume:
        previous = json.loads(protocol_path.read_text())
        if any(
            previous[k] != v
            for k, v in {
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "weight_sha256": receipt["sha256"],
                "crop_size": args.crop_size,
            }.items()
        ):
            raise ValueError("Resume protocol mismatch")
        (args.output / "resume_amendment.json").write_text(
            json.dumps(
                {
                    "reason": "Prior run stopped at fewer than two retained nuclei",
                    "policy": "Preserve crop/model; explicitly record unusable patients; do not invent nodes",
                    "original_protocol_sha256": hashlib.sha256(
                        protocol_path.read_bytes()
                    ).hexdigest(),
                },
                indent=2,
            )
        )
    else:
        protocol_path.write_text(
            json.dumps(
                {
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "weight_sha256": receipt["sha256"],
                    "crop_size": args.crop_size,
                    "mpp": None,
                    "radius_pixels": 50,
                    "scope": "exploratory candidate cell graphs; segmentation unvalidated; historical patient test already explored",
                    "failure_policy": "fail rather than silently omit patients",
                },
                indent=2,
            )
        )
    torch.set_num_threads(2)
    model, _ = get_pretrained_model("hovernet_fast-pannuke", pretrained_weights=args.weights)
    model.eval().requires_grad_(False)
    manifest, quality, excluded = [], [], []
    for row in rows:
        path = ROOT / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Image checksum mismatch")
        with Image.open(path) as image:
            w, h = image.size
            x, y = max(0, (w - args.crop_size) // 2), max(0, (h - args.crop_size) // 2)
            rgb = np.asarray(
                image.crop((x, y, min(w, x + args.crop_size), min(h, y + args.crop_size))).convert(
                    "RGB"
                )
            )
        pid = row["patient_id"]
        mask_path = args.output / f"{pid}_instances.npy"
        instances = (
            np.load(mask_path, allow_pickle=False)
            if args.resume and mask_path.exists()
            else infer_tiled(model, rgb)
        )
        np.save(mask_path, instances, allow_pickle=False)
        _, sizes = np.unique(instances[instances > 0], return_counts=True)
        if int(np.sum(sizes >= 4)) < 2:
            excluded.append(
                {
                    "patient_id": pid,
                    "split": row["proposed_split"],
                    "label": row["roi_label"],
                    "reason": "fewer_than_two_retained_nuclei",
                }
            )
            print(f"Patient {pid}: unusable, explicitly recorded", flush=True)
            continue
        graph = cell_graph(rgb, instances, radius=50)
        graph.source_image_sha256 = row["sha256"]
        graph.crop_origin_xy = [x, y]
        pid = row["patient_id"]
        graph_path = args.output / f"{pid}.pt"
        if not (args.resume and graph_path.exists()):
            torch.save(graph.to_dict(), graph_path)
        else:
            stored = torch.load(graph_path, weights_only=True)
            if (
                stored["source_image_sha256"] != row["sha256"]
                or not torch.equal(stored["x"], graph.x)
                or not torch.equal(stored["edge_index"], graph.edge_index)
            ):
                raise ValueError("Existing graph does not match source/mask")
        np.save(args.output / f"{pid}_instances.npy", instances, allow_pickle=False)
        degree = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)
        quality.append(
            {
                "patient_id": pid,
                "nodes": graph.num_nodes,
                "undirected_edges": graph.num_edges // 2,
                "isolated_nodes": int((degree == 0).sum()),
                "crop_width": rgb.shape[1],
                "crop_height": rgb.shape[0],
                "coordinate_units": graph.coordinate_units,
                "segmentation_validated_on_BRACS": False,
            }
        )
        manifest.append(
            {
                "patient_id": pid,
                "split": {"internal_test": "test"}.get(
                    row["proposed_split"], row["proposed_split"]
                ),
                "label": int(row["roi_label"] == "IC"),
                "graph_path": graph_path.name,
                "sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
            }
        )
        print(f"Patient {pid}: {graph.num_nodes} candidate cells", flush=True)
    with (args.output / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (args.output / "quality.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(quality[0]))
        writer.writeheader()
        writer.writerows(quality)
    with (args.output / "excluded.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "split", "label", "reason"])
        writer.writeheader()
        writer.writerows(excluded)
    (args.output / "completion.json").write_text(
        json.dumps(
            {
                "source_patients": len(rows),
                "usable": len(manifest),
                "excluded": len(excluded),
                "all_patients_accounted_for": len(manifest) + len(excluded) == len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
