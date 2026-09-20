"""Run a tiny end-to-end technical pipeline on authorized pilot RoIs.

The output is an engineering report. It proves that manifests, patient checks,
image decoding, graph construction, model forward passes, gradients and a
bounded feature edit are wired together. It is not a scientific result.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.loader import DataLoader

from src.data.manifest import ROOT, read_bracs_smoke_manifest
from src.data.splits import assert_no_patient_overlap, label_index
from src.graphs.build import make_graph
from src.interventions.feature_edits import clamp_feature_shift
from src.models.gnn import GraphClassifier
from src.segmentation.superpixels import image_patch_nodes


def build_graphs(limit: int | None = None):
    records = read_bracs_smoke_manifest()
    assert_no_patient_overlap(records)
    labels = label_index(records)
    selected = records[:limit] if limit else records
    graphs = []
    for record in selected:
        nodes = image_patch_nodes(record.path)
        graph = make_graph(nodes.features, nodes.coordinates, labels[record.label], k=4)
        graph.image_id = record.image_id
        graph.patient_id = record.patient_id
        graph.label_name = record.label
        graphs.append(graph)
    return graphs, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase2_smoke.json")
    parser.add_argument("--limit", type=int, default=7)
    args = parser.parse_args()
    graphs, labels = build_graphs(limit=args.limit)
    if not graphs:
        raise ValueError("No pilot graphs available")
    loader = DataLoader(graphs, batch_size=len(graphs))
    batch = next(iter(loader))
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "engineering_smoke_only_not_scientific_validation",
        "graphs": len(graphs),
        "nodes_per_graph": int(graphs[0].num_nodes),
        "node_features": int(graphs[0].num_node_features),
        "labels": labels,
        "architectures": {},
        "gates": {
            "patient_split_checked": True,
            "hovernet_validated_on_nucls": False,
            "concept_annotations_available": False,
            "bach_external_test_opened": False,
        },
    }
    for architecture in ["GCN", "GraphSAGE", "GATv2"]:
        torch.manual_seed(11)
        model = GraphClassifier(architecture, batch.num_node_features, 16, len(labels))
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        loss.backward()
        edited_x = clamp_feature_shift(batch.x, feature=0, delta=0.05)
        edited_logits = model(edited_x, batch.edge_index, batch.batch)
        report["architectures"][architecture] = {
            "forward_shape": list(logits.shape),
            "loss_finite": bool(torch.isfinite(loss)),
            "gradient_available": any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ),
            "feature_edit_delta_l1": float((edited_logits - logits).abs().sum().detach()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
