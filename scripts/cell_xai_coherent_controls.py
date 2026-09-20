"""Supplementary spatially coherent, per-node norm-matched GCIA controls."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_cell_xai import Ensemble, enlarge_nuclei, interval
from src.models.gnn import GraphClassifier


def main():
    torch.set_num_threads(2)
    out = ROOT / "reports/cell_xai"
    main_summary = json.loads((out / "summary.json").read_text())
    with (out / "coherent_control_protocol.json").open("x") as f:
        json.dump(
            {
                "scope": "supplementary robustness analysis after initial audit; not tuning or replacing initial results",
                "seed": 2027,
                "repeats": 20,
                "rule": "one isotropic direction shared across all nuclei per repeat; per-node standardized L2 norms match the target edit",
                "factors": [0.95, 1.05],
                "all_architectures": True,
            },
            f,
            indent=2,
        )
    manifest = list(csv.DictReader((ROOT / "reports/cell_cohort/manifest.csv").open()))
    target_rows = list(csv.DictReader((out / "interventions.csv").open()))
    rows, summary = [], {}
    rng = np.random.default_rng(2027)
    for architecture in main_summary["results"]:
        models = []
        for seed in (11, 23, 37):
            state = torch.load(
                ROOT / f"reports/cell_gnn_comparison/{architecture}_{seed}.pt", weights_only=True
            )
            model = GraphClassifier(architecture, 8, 16, 2)
            model.load_state_dict(state["weights"])
            models.append(model.eval().requires_grad_(False))
        wrapper = Ensemble(models)
        mean, scale = state["mean"], state["scale"]
        for r in manifest:
            if r["split"] != "test":
                continue
            g = Data(
                **torch.load(ROOT / "reports/cell_cohort" / r["graph_path"], weights_only=True)
            )
            x = (g.x - mean) / scale
            baseline = float(wrapper(x, g.edge_index)[0])
            for factor in [0.95, 1.05]:
                norm = (((enlarge_nuclei(g.x, factor) - mean) / scale) - x).norm(
                    dim=1, keepdim=True
                )
                for repeat in range(20):
                    direction = torch.tensor(rng.normal(size=(1, 8)), dtype=x.dtype)
                    edited = x + norm * direction / direction.norm()
                    rows.append(
                        {
                            "architecture": architecture,
                            "patient": r["patient_id"],
                            "factor": factor,
                            "repeat": repeat,
                            "probability_IC_change": float(wrapper(edited, g.edge_index)[0])
                            - baseline,
                        }
                    )
        summary[architecture] = {}
        for factor in [0.95, 1.05]:
            patients = [r["patient_id"] for r in manifest if r["split"] == "test"]
            random_means, target_means = [], []
            for pid in patients:
                random_means.append(
                    np.mean(
                        [
                            abs(r["probability_IC_change"])
                            for r in rows
                            if r["architecture"] == architecture
                            and r["patient"] == pid
                            and r["factor"] == factor
                        ]
                    )
                )
                target_means.append(
                    next(
                        abs(float(r["probability_IC_change"]))
                        for r in target_rows
                        if r["architecture"] == architecture
                        and r["patient"] == pid
                        and float(r["area_factor"]) == factor
                        and r["method"] == "gcia_morphology"
                    )
                )
            summary[architecture][str(factor)] = {
                "coherent_control_mean_absolute_change": float(np.mean(random_means)),
                "target_minus_coherent_control": interval(np.array(target_means) - random_means),
            }
    with (out / "coherent_controls.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "coherent_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
