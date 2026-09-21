"""Train frozen-model mechanism benchmark; never access histology test data."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.synthetic.data import make_split
from src.models import StudyModel
from src.interventions import intervene
from src.probes import fit_probe
from src.tcav import concept_direction, positive_fraction


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    config_path = ROOT / "configs/synthetic.json"
    config = json.loads(config_path.read_text())
    out = ROOT / "results/synthetic_run"
    out.mkdir(parents=True, exist_ok=False)
    sources = [config_path, Path(__file__), Path(__file__).with_name("data.py"), *sorted((ROOT / "src").glob("*.py"))]
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "config": config,
                "source_sha256": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
                "scope": "finite factorial software/mechanism benchmark, no histology; generator truth does not establish learned model use",
                "inference": "graph replicates share factorial states; report descriptive per-seed results, no patient/population inference",
                "control": "paired irrelevant-concept edit (graph identity); this negative control is NOT norm-matched; no selective functional-reliance claim",
                "erasure": "CAV orthogonal projection, NOT LEACE; train-only direction and mean",
                "tcav": "fraction of positive derivatives of baseline predicted-class logit along positively oriented CAV; descriptive, no significance test",
            },
            indent=2,
        )
    )
    splits = {
        name: make_split(n, 400 + j, name)
        for j, (name, n) in enumerate(config["split_repetitions"].items())
    }
    batches = {name: Batch.from_data_list(graphs) for name, graphs in splits.items()}
    torch.save(splits, out / "graphs.pt")
    scores, probes, recovery, raw = [], [], [], []
    for architecture in config["architectures"]:
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            model = StudyModel(architecture, config["hidden"], config["layers"])
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            train, val, test = (batches[name] for name in ("train", "validation", "test"))
            best_loss, best, best_epoch = float("inf"), None, None
            for epoch in range(config["epochs"]):
                model.train()
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    model(train.x, train.edge_index, train.batch), train.y
                )
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    vloss = float(
                        torch.nn.functional.cross_entropy(
                            model(val.x, val.edge_index, val.batch), val.y
                        )
                    )
                if vloss < best_loss:
                    best_loss, best_epoch = vloss, epoch + 1
                    best = {k: value.clone() for k, value in model.state_dict().items()}
            model.load_state_dict(best)
            model.eval().requires_grad_(False)
            torch.save(
                {
                    "weights": best,
                    "architecture": architecture,
                    "seed": seed,
                    "validation_epoch": best_epoch,
                },
                out / f"{architecture}_{seed}.pt",
            )
            with torch.no_grad():
                hs_train = model.representations(train.x, train.edge_index, train.batch)
                hs_test = model.representations(test.x, test.edge_index, test.batch)
                probabilities = model.head(hs_test[-1]).softmax(-1)
                predicted = probabilities.argmax(-1)
            accuracy = float((predicted == test.y).float().mean())
            scores.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "accuracy": accuracy,
                    "validation_epoch": best_epoch,
                }
            )
            np.savez_compressed(
                out / f"{architecture}_{seed}_representations.npz",
                **{f"train_layer_{i}": h.numpy() for i, h in enumerate(hs_train)},
                **{f"test_layer_{i}": h.numpy() for i, h in enumerate(hs_test)},
            )
            for c, concept in enumerate(config["concepts"]):
                ytrain, ytest = train.concepts[:, c].numpy(), test.concepts[:, c].numpy()
                final_auc, final_probe = None, None
                for layer, (htrain, htest) in enumerate(zip(hs_train, hs_test, strict=True)):
                    probe = fit_probe(htrain.numpy(), ytrain, seed)
                    auc = roc_auc_score(ytest, probe.predict_proba(htest.numpy())[:, 1])
                    probes.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "concept": concept,
                            "layer": layer,
                            "auc": float(auc),
                        }
                    )
                    final_auc, final_probe = float(auc), probe
                direction, norm = concept_direction(final_probe)
                tcav = positive_fraction(model.head, predicted, direction, norm)
                center = hs_train[-1].mean(0)
                erased = (
                    hs_test[-1] - ((hs_test[-1] - center) @ direction)[:, None] * direction[None]
                )
                with torch.no_grad():
                    erased_p = model.head(erased).softmax(-1)
                    modified = Batch.from_data_list([intervene(g, c) for g in splits["test"]])
                    after = model(modified.x, modified.edge_index, modified.batch).softmax(-1)
                # Total variation is sensitive to any output redistribution, not only the winning class.
                effect = 0.5 * (after - probabilities).abs().sum(-1)
                erase_effect = 0.5 * (erased_p - probabilities).abs().sum(-1)
                for i, g in enumerate(splits["test"]):
                    raw.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "sample": g.sample_id,
                            "concept": concept,
                            "total_variation": float(effect[i]),
                            "erasure_tv": float(erase_effect[i]),
                        }
                    )
                encoded = final_auc >= config["probe_auc_threshold"]
                sensitive = float(effect.mean()) > config["model_absolute_effect_margin"]
                generator_used = c in (0, 2)
                recovery.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "concept": concept,
                        "generator_uses_concept": generator_used,
                        "heldout_probe_auc": final_auc,
                        "encoded_screen": encoded,
                        "model_effect_mean_tv": float(effect.mean()),
                        "model_effect_max_tv": float(effect.max()),
                        "sensitive_screen": sensitive,
                        "matches_generator_dependence": sensitive == generator_used,
                        "tcav_positive_fraction": tcav,
                        "orthogonal_cav_erasure_mean_tv": float(erase_effect.mean()),
                        "interpretation": "descriptive_screen_not_equivalence_or_clinical_validation",
                    }
                )
            print(f"{architecture} seed={seed} accuracy={accuracy:.3f}", flush=True)
    write_csv(out / "classification.csv", scores)
    write_csv(out / "layerwise_probes.csv", probes)
    write_csv(out / "ground_truth_recovery.csv", recovery)
    write_csv(out / "interventions.csv", raw)
    metrics = []
    for architecture in config["architectures"]:
        rows = [r for r in recovery if r["architecture"] == architecture]
        positive = [r for r in rows if r["generator_uses_concept"]]
        negative = [r for r in rows if not r["generator_uses_concept"]]
        metrics.append(
            {
                "architecture": architecture,
                "sensitivity_to_generator_dependence": float(
                    np.mean([r["sensitive_screen"] for r in positive])
                ),
                "specificity_to_generator_dependence": float(
                    np.mean([not r["sensitive_screen"] for r in negative])
                ),
                "screen_errors": sum(not r["matches_generator_dependence"] for r in rows),
            }
        )
    write_csv(out / "recovery_metrics.csv", metrics)
    summary = {
        "scores": scores,
        "recovery": metrics,
        "progression_gate_passed": False,
        "missing": [
            "matched nontrivial intervention controls",
            "equivalence procedure",
            "GNNExplainer benchmark",
            "LEACE implementation",
            "independent randomized-model/label sanity checks",
        ],
        "warning": "Do not equate generator label dependence with learned-model dependence; descriptive threshold is not equivalence proof.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "README.md").write_text(
        "# Synthetic results\n\nSee classification.csv, layerwise_probes.csv and ground_truth_recovery.csv. "
        "This is a descriptive factorial benchmark, not clinical validation. "
        "The progression gate remains false until matched controls, equivalence and sanity checks are complete.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
