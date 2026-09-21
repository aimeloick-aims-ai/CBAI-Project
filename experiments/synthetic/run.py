"""Canonical binary synthetic experiment and shared frozen-model training loop."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_model, load_config, save_stage, verify_software
from src.graph import synthetic_split
from src.models import StudyModel


def train_model(architecture, seed, splits, config):
    torch.manual_seed(seed)
    model = StudyModel(
        architecture,
        config["hidden"],
        config["layers"],
        input_dim=splits["train"][0].num_node_features,
        classes=2,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    train = Batch.from_data_list(splits["train"])
    validation = Batch.from_data_list(splits["validation"])
    best_loss, best, best_epoch = float("inf"), None, None
    # Full-batch optimization avoids splitting factorial pairs; evaluation is batched.
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
            val_loss = float(
                torch.nn.functional.cross_entropy(
                    model(validation.x, validation.edge_index, validation.batch), validation.y
                )
            )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    model.eval().requires_grad_(False)
    digest = hashlib.sha256()
    for name, value in sorted(best.items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return model, {
        "selected_epoch": best_epoch,
        "weights_sha256": digest.hexdigest(),
        "selection": "lowest validation cross entropy; test not used for selection",
    }


def execute(stage, splits, metadata, config):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    nfeatures = splits["train"][0].num_node_features
    mean, scale = torch.zeros(nfeatures), torch.ones(nfeatures)
    if stage == "bracs":
        features = torch.cat([g.x for g in splits["train"]])
        mean = features.mean(0)
        scale = features.std(0, unbiased=False).clamp_min(config["numerical_tolerance"])
        splits = {name: [g.clone() for g in graphs] for name, graphs in splits.items()}
        for graphs in splits.values():
            for graph in graphs:
                graph.x = (graph.x - mean) / scale
    rows, models = [], {}
    for architecture in config["models"]:
        for seed in config["seeds"]:
            model, training = train_model(architecture, seed, splits, config)
            torch.manual_seed(seed)
            random_model = (
                StudyModel(
                    architecture, config["hidden"], config["layers"], input_dim=nfeatures, classes=2
                )
                .eval()
                .requires_grad_(False)
            )
            rows.extend(
                evaluate_model(model, random_model, splits, stage, seed, config, mean, scale)
            )
            models[f"{architecture}_{seed}"] = training
            rows.append(
                {
                    "stage": stage,
                    "dataset": stage,
                    "model": architecture,
                    "seed": seed,
                    "analysis": "classification",
                    "record_type": "training",
                    "metric": "selected_epoch",
                    "value": training["selected_epoch"],
                }
            )
    metadata.update(
        {
            "models": models,
            "train_feature_mean": mean.tolist(),
            "train_feature_scale": scale.tolist(),
            "split_sizes": {s: len(gs) for s, gs in splits.items()},
        }
    )
    save_stage(ROOT, stage, rows, metadata, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        raise SystemExit(0 if verify_software(ROOT) else 1)
    config = load_config(ROOT)
    synthetic = config["synthetic"]
    splits = {
        name: synthetic_split(n, synthetic["split_seeds"][name], name, synthetic)
        for name, n in synthetic["split_repetitions"].items()
    }
    metadata = {
        "generator_ground_truth": synthetic["label_rule"],
        "expected_before_training": {
            "C_used": "encoded and used",
            "C_unused": "encoded but not in label rule",
            "C_relational": "GraphSAGE can use relations; DeepSets cannot",
        },
        "independence_unit": "replicate, not factorial variant",
        "physical_scale_known": False,
        "split_identifiers": {s: sorted({g.patient_id for g in gs}) for s, gs in splits.items()},
        "data_tensor_sha256": {
            s: hashlib.sha256(
                b"".join(
                    g.x.numpy().tobytes() + g.edge_index.numpy().tobytes() + g.y.numpy().tobytes()
                    for g in gs
                )
            ).hexdigest()
            for s, gs in splits.items()
        },
    }
    execute("synthetic", splits, metadata, config)


if __name__ == "__main__":
    main()
