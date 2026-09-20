"""Run ConceptCellGNNExplainer & Counterfactual Interventions on Cell Graphs."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interventions.cell_xai import (
    CellGNNExplainer,
    CellGraphInterventions,
    ConceptCellAligner,
)
from src.models.gnn import GraphClassifier


def create_synthetic_cell_graph(num_nodes: int = 30, num_features: int = 8, seed: int = 42) -> Data:
    """Generate a synthetic cell graph for testing/exploratory audit execution."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, num_features, generator=g)

    # 5-NN spatial graph simulation
    pos = torch.rand(num_nodes, 2, generator=g) * 500.0  # micrometer coordinates
    dist = torch.cdist(pos, pos)
    _, knn_indices = torch.topk(dist, k=6, largest=False, dim=-1)

    sources = torch.arange(num_nodes).repeat_interleave(5)
    targets = knn_indices[:, 1:].reshape(-1)
    edge_index = torch.stack([sources, targets], dim=0)

    feature_names = [
        "area_um2",
        "perimeter_um",
        "eccentricity",
        "solidity",
        "mean_intensity",
        "std_intensity",
        "local_density",
        "neighbor_dist_avg",
    ]
    return Data(
        x=x,
        edge_index=edge_index,
        pos=pos,
        feature_names=feature_names,
        coordinate_units="micrometers",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="GCN", choices=["GCN", "GraphSAGE", "GATv2"])
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cell_graph_xai")
    parser.add_argument("--num-graphs", type=int, default=5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Executing Cell Graph XAI audit with architecture: {args.architecture}")

    torch.manual_seed(42)
    in_channels = 8
    model = GraphClassifier(
        args.architecture, in_channels=in_channels, hidden_channels=16, out_channels=2
    )
    model.eval()

    explainer = CellGNNExplainer(model, epochs=50, learning_rate=0.05)
    aligner = ConceptCellAligner(
        feature_names=[
            "area_um2",
            "perimeter_um",
            "eccentricity",
            "solidity",
            "mean_intensity",
            "std_intensity",
            "local_density",
            "neighbor_dist_avg",
        ]
    )

    explanation_records = []
    alignment_records = []
    fidelity_scores = []

    for idx in range(args.num_graphs):
        data = create_synthetic_cell_graph(num_nodes=35, num_features=in_channels, seed=100 + idx)
        batch = torch.zeros(data.num_nodes, dtype=torch.long)

        # 1. Topological explanation
        exp_res = explainer.explain_graph(data.x, data.edge_index, batch)

        # 2. Concept alignment
        corrs = aligner.compute_feature_correlations(exp_res.node_mask, data.x)

        # 3. Interventions
        fid_res = CellGraphInterventions.evaluate_cell_intervention_fidelity(
            model,
            data.x,
            data.edge_index,
            batch,
            exp_res.top_node_indices,
            top_k=5,
            random_control_count=15,
        )

        fidelity_scores.append(fid_res.fidelity_score)

        explanation_records.append(
            {
                "graph_id": f"cell_graph_{idx}",
                "top_node_idx": int(exp_res.top_node_indices[0].item()),
                "top_node_importance": float(exp_res.node_mask[exp_res.top_node_indices[0]].item()),
                "target_prob_drop": fid_res.target_prob_drop,
                "control_prob_drop_mean": fid_res.control_prob_drop_mean,
                "fidelity_score": fid_res.fidelity_score,
            }
        )

        for feat_name, corr_val in corrs.items():
            alignment_records.append(
                {
                    "graph_id": f"cell_graph_{idx}",
                    "feature_name": feat_name,
                    "correlation_with_importance": corr_val,
                }
            )

    # Save explanations CSV
    with (args.output / "explanations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(explanation_records[0].keys()))
        writer.writeheader()
        writer.writerows(explanation_records)

    # Save alignments CSV
    with (args.output / "alignments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(alignment_records[0].keys()))
        writer.writeheader()
        writer.writerows(alignment_records)

    summary = {
        "architecture": args.architecture,
        "num_graphs_evaluated": args.num_graphs,
        "mean_fidelity_score": float(torch.tensor(fidelity_scores).mean().item()),
        "status": "completed",
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
