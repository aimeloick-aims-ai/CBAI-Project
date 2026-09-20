"""Generate label-coded synthetic fixtures. NOT BACH or external validation.

Software tests only; labels explicitly determine node counts and features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]

TOY_CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]


def generate_synthetic_cell_graph(case_id: str, label_idx: int, seed: int = 42) -> Data:
    """Generate an artificial label-coded graph; not extracted from any image."""
    g = torch.Generator().manual_seed(seed)
    num_nodes = 30 + (label_idx * 5)  # Invasive tumors have higher cell density
    in_channels = 8

    # Features: area, perimeter, eccentricity, solidity, mean_intensity, std_intensity, local_density, neighbor_dist_avg
    x = torch.randn(num_nodes, in_channels, generator=g)
    # Shift features slightly according to diagnosis
    x[:, 0] += label_idx * 0.5  # Nuclear size increases with malignancy grade
    x[:, 6] += label_idx * 0.4  # Cell density increases with malignancy

    pos = torch.rand(num_nodes, 2, generator=g) * 500.0
    dist = torch.cdist(pos, pos)
    _, knn_indices = torch.topk(dist, k=6, largest=False, dim=-1)

    sources = torch.arange(num_nodes).repeat_interleave(5)
    targets = knn_indices[:, 1:].reshape(-1)
    edge_index = torch.stack([sources, targets], dim=0)

    y = torch.tensor([label_idx], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        pos=pos,
        y=y,
        case_id=case_id,
        diagnosis=TOY_CLASSES[label_idx],
        dataset="SYNTHETIC_LABEL_CODED_FIXTURE",
    )


def prepare_synthetic_cohort(output_dir: Path, num_samples_per_class: int = 5) -> dict:
    """Save synthetic fixtures for software checks only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "cell_graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    sample_id = 0

    for label_idx, cls_name in enumerate(TOY_CLASSES):
        for i in range(num_samples_per_class):
            sample_id += 1
            case_id = f"toy_{cls_name.lower()}_{i + 1:02d}"
            graph = generate_synthetic_cell_graph(case_id, label_idx, seed=2000 + sample_id)

            graph_path = graphs_dir / f"{case_id}.pt"
            torch.save(graph, graph_path)

            try:
                rel_path = str(graph_path.relative_to(ROOT))
            except ValueError:
                rel_path = str(graph_path)

            manifest.append(
                {
                    "case_id": case_id,
                    "diagnosis": cls_name,
                    "label": label_idx,
                    "num_cells": graph.num_nodes,
                    "num_edges": graph.edge_index.shape[1],
                    "filepath": rel_path,
                }
            )

    summary = {
        "dataset_name": "Synthetic label-coded fixture; NOT external validation",
        "num_classes": len(TOY_CLASSES),
        "classes": TOY_CLASSES,
        "total_cases": len(manifest),
        "manifest": manifest,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/toy_demo/output")
    parser.add_argument("--num-samples-per-class", type=int, default=5)
    args = parser.parse_args()

    print(f"Preparing SYNTHETIC FIXTURES in: {args.output}")
    summary = prepare_synthetic_cohort(args.output, args.num_samples_per_class)
    print(f"Synthetic generation complete; no real data acquired. Graphs: {summary['total_cases']}")


if __name__ == "__main__":
    main()
