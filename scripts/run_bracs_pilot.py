"""Run a bounded real-data BRACS pilot, explicitly unsuitable for publication claims."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import read_bracs_smoke_manifest
from src.data.splits import assert_no_patient_overlap, label_index
from src.evaluation.metrics import accuracy
from src.graphs.build import make_graph
from src.models.gnn import GraphClassifier
from src.segmentation.superpixels import image_patch_nodes


def graphs_by_split():
    """Build deterministic patch graphs from the small authorized BRACS sample."""
    records = read_bracs_smoke_manifest()
    assert_no_patient_overlap(records)
    labels = label_index(records)
    groups = {"train": [], "val": []}
    for record in records:
        if record.split not in groups:
            continue
        nodes = image_patch_nodes(record.path)
        groups[record.split].append(
            make_graph(nodes.features, nodes.coordinates, labels[record.label])
        )
    if not groups["train"] or not groups["val"]:
        raise ValueError("Pilot requires non-empty official train and validation RoI")
    return groups, labels


def fit_and_evaluate(
    architecture: str,
    train_batch,
    validation_batch,
    classes: int,
    seed: int,
    epochs: int,
) -> dict[str, float | int]:
    torch.manual_seed(seed)
    model = GraphClassifier(architecture, train_batch.num_node_features, 16, classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_batch.x, train_batch.edge_index, train_batch.batch)
        torch.nn.functional.cross_entropy(logits, train_batch.y).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_predictions = model(train_batch.x, train_batch.edge_index, train_batch.batch).argmax(
            dim=1
        )
        validation_predictions = model(
            validation_batch.x, validation_batch.edge_index, validation_batch.batch
        ).argmax(dim=1)
    return {
        "seed": seed,
        "train_accuracy": accuracy(train_batch.y.tolist(), train_predictions.tolist()),
        "validation_accuracy": accuracy(
            validation_batch.y.tolist(), validation_predictions.tolist()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bracs_pilot_results.json")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    groups, labels = graphs_by_split()
    train_batch = next(iter(DataLoader(groups["train"], batch_size=len(groups["train"]))))
    validation_batch = next(iter(DataLoader(groups["val"], batch_size=len(groups["val"]))))
    seeds = [11, 23, 37, 53, 71]
    architectures = {}
    for architecture in ("GCN", "GraphSAGE", "GATv2"):
        runs = [
            fit_and_evaluate(
                architecture, train_batch, validation_batch, len(labels), seed, args.epochs
            )
            for seed in seeds
        ]
        values = [run["validation_accuracy"] for run in runs]
        architectures[architecture] = {
            "runs": runs,
            "validation_accuracy_mean": sum(values) / len(values),
            "validation_accuracy_min": min(values),
            "validation_accuracy_max": max(values),
        }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "exploratory_real_data_pilot_not_scientific_final_result",
        "data": {
            "dataset": "BRACS",
            "train_roi": len(groups["train"]),
            "validation_roi": len(groups["val"]),
            "classes": labels,
            "patient_split_checked": True,
        },
        "protocol": {"epochs": args.epochs, "seeds": seeds, "node_builder": "rgb_patch_grid"},
        "architectures": architectures,
        "unmet_requirements": [
            "full_cohort",
            "validated_nuclear_segmentation",
            "independent_concept_annotations",
            "intervention_validity_review",
            "locked_BACH_external_evaluation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
