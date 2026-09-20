"""Small positive-control study of mask-grounded tissue substitution, no downloads."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_extended_xai import save_csv
from src.graphs.build import make_graph


def patch_features(patch):
    """RGB statistics plus intensity distribution and horizontal/vertical texture.

    These are texture descriptors, not segmented nuclear morphology.
    """
    rgb = np.r_[patch.mean((0, 1)), patch.std((0, 1))]
    gray = patch.mean(2)
    hist = np.histogram(gray, bins=8, range=(0, 1))[0].astype(float)
    hist /= hist.sum()
    texture = np.r_[
        np.quantile(gray, [0.1, 0.25, 0.5, 0.75, 0.9]),
        hist,
        np.abs(np.diff(gray, axis=0)).mean(),
        np.abs(np.diff(gray, axis=1)).mean(),
    ]
    return np.r_[rgb, texture].astype("float32")


def mask_fractions(mask):
    valid = (mask != 0) & (mask != 7)
    fractions = np.array([(mask == code).sum() for code in range(22)], dtype=float)
    fractions[[0, 7]] = 0
    return fractions / max(valid.sum(), 1), valid.mean()


def load_patients(config, manifest=None):
    with (manifest or ROOT / "data/bcss_extended_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = {}
    for row in rows:
        path = ROOT / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Source checksum mismatch")
        pair = pairs.setdefault(row["patient_id"], {})
        if row["kind"] in pair:
            raise ValueError("Duplicate region per patient")
        pair[row["kind"]] = row
    patients = []
    size = config["patch_pixels"]
    for pid, pair in sorted(pairs.items()):
        if float(pair["image"]["requested_mpp"]) != config["pixel_spacing_um"]:
            raise ValueError("Unexpected requested pixel spacing")
        with Image.open(ROOT / pair["image"]["path"]) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        with Image.open(ROOT / pair["mask"]["path"]) as im:
            mask = np.array(im.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST))
        patches, fractions, valid, coords = [], [], [], []
        for y in range(0, rgb.shape[0] - size + 1, size):
            for x in range(0, rgb.shape[1] - size + 1, size):
                patches.append(rgb[y : y + size, x : x + size].copy())
                frac, coverage = mask_fractions(mask[y : y + size, x : x + size])
                fractions.append(frac)
                valid.append(coverage >= config["minimum_annotated_fraction"])
                coords.append([x / size, y / size])
        if len(patches) < 5 or not any(valid):
            raise ValueError(f"Insufficient patches for {pid}")
        patients.append(
            {
                "patient": pid,
                "patches": patches,
                "fractions": np.array(fractions),
                "valid": np.array(valid),
                "coords": np.array(coords),
                "features": np.stack([patch_features(p) for p in patches]),
            }
        )
    return patients


class FractionGCN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = GCNConv(channels, 16)
        self.second = GCNConv(16, 16)
        self.head = nn.Linear(16, 1)

    def forward(self, g):
        h = self.first(g.x, g.edge_index).relu()
        h = self.second(h, g.edge_index).relu()
        return self.head(h).squeeze(-1).sigmoid()


def substitutions(patient, config):
    """Choose recipients/donors by mask and pixel amplitude, never model scores."""
    frac, valid = patient["fractions"], patient["valid"]
    tumor = np.flatnonzero(valid & (frac[:, 1] >= config["pure_tissue_threshold"]))
    stroma = np.flatnonzero(valid & (frac[:, 2] >= config["pure_tissue_threshold"]))
    if len(tumor) < 2 or len(stroma) == 0:
        return []
    rows = []
    for recipient in tumor[: config["maximum_recipients_per_patient"]]:
        p = patient["patches"][recipient]
        amplitudes = np.array([float(np.sqrt(((q - p) ** 2).mean())) for q in patient["patches"]])
        target = int(stroma[np.argmin(amplitudes[stroma])])
        controls = tumor[tumor != recipient]
        control = int(controls[np.argmin(np.abs(amplitudes[controls] - amplitudes[target]))])
        mismatch = abs(amplitudes[target] - amplitudes[control]) / max(amplitudes[target], 1e-8)
        other = [i for i in range(22) if i not in (0, 1, 2, 7)]
        drift = float(np.abs(frac[target, other] - frac[recipient, other]).max())
        control_drift = float(np.abs(frac[control] - frac[recipient]).max())
        rows.append(
            {
                "recipient": int(recipient),
                "target_donor": target,
                "control_donor": control,
                "target_pixel_rms": amplitudes[target],
                "control_pixel_rms": amplitudes[control],
                "relative_amplitude_mismatch": mismatch,
                "other_tissue_drift": drift,
                "control_tissue_drift": control_drift,
                "numerical_checks_pass": bool(
                    mismatch <= config["amplitude_match_relative_tolerance"]
                    and drift <= config["non_target_fraction_tolerance"]
                    and control_drift <= config["non_target_fraction_tolerance"]
                ),
            }
        )
    return rows


def main():
    torch.set_num_threads(2)
    config_path = ROOT / "configs/focused_protocol.json"
    config = json.loads(config_path.read_text())
    patients = load_patients(config)
    out = ROOT / "reports/focused_study"
    out.mkdir(parents=True, exist_ok=True)
    measures, edits, coverage = [], [], []
    for p in patients:
        selected = substitutions(p, config)
        coverage.append(
            {
                "patient": p["patient"],
                "patches": len(p["valid"]),
                "valid_patches": int(p["valid"].sum()),
                "candidate_edits": len(selected),
                "edits_passing_numeric_checks": sum(r["numerical_checks_pass"] for r in selected),
            }
        )
    for test, patient in enumerate(patients):
        fit = [p for i, p in enumerate(patients) if i != test]
        for feature_set in config["feature_sets"]:
            channels = 6 if feature_set == "rgb" else patient["features"].shape[1]
            training = np.concatenate([p["features"][p["valid"], :channels] for p in fit])
            mean, scale = training.mean(0), training.std(0).clip(1e-6)

            def build(p, channels=channels, mean=mean, scale=scale):
                g = make_graph((p["features"][:, :channels] - mean) / scale, p["coords"], 0)
                g.target = torch.tensor(p["fractions"][:, 1], dtype=torch.float32)
                g.valid = torch.tensor(p["valid"])
                return g

            train = Batch.from_data_list([build(p) for p in fit])
            g = build(patient)
            baseline = float(train.target[train.valid].mean())
            for seed in config["seeds"]:
                torch.manual_seed(seed)
                model = FractionGCN(channels)
                opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
                for _ in range(config["epochs"]):
                    opt.zero_grad()
                    prediction = model(train)
                    # Equal weight per training patient, not per patch.
                    losses = [
                        (
                            prediction[(train.batch == j) & train.valid]
                            - train.target[(train.batch == j) & train.valid]
                        )
                        .square()
                        .mean()
                        for j in range(len(fit))
                    ]
                    torch.stack(losses).mean().backward()
                    opt.step()
                model.eval().requires_grad_(False)
                before = model(g)
                observed = float(g.target[g.valid].mean())
                predicted = float(before[g.valid].mean())
                measures.append(
                    {
                        "patient": patient["patient"],
                        "feature_set": feature_set,
                        "seed": seed,
                        "patch_mse": float((before[g.valid] - g.target[g.valid]).square().mean()),
                        "constant_patch_mse": float((baseline - g.target[g.valid]).square().mean()),
                        "observed_fraction": observed,
                        "predicted_fraction": predicted,
                        "region_absolute_error": abs(predicted - observed),
                        "constant_region_error": abs(baseline - observed),
                    }
                )
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "mean": torch.tensor(mean),
                        "scale": torch.tensor(scale),
                    },
                    out / f"{test}_{feature_set}_{seed}.pt",
                )
                for edit in substitutions(patient, config):
                    for arm in ("target", "control"):
                        donor = edit[f"{arm}_donor"]
                        recipient = edit["recipient"]
                        altered = g.clone()
                        # Fixed independent per-patch descriptors: replacing this feature row
                        # is exactly equivalent to transplanting the donor RGB patch.
                        altered.x[recipient] = g.x[donor]
                        after = float(model(altered)[g.valid].mean())
                        delta = float((g.target[donor] - g.target[recipient]) / g.valid.sum())
                        edits.append(
                            {
                                "patient": patient["patient"],
                                "feature_set": feature_set,
                                "seed": seed,
                                "arm": arm,
                                **edit,
                                "observed_fraction_change": delta,
                                "predicted_fraction_change": after - predicted,
                                "change_absolute_error": abs(after - predicted - delta),
                            }
                        )
        print(f"Patient {patient['patient']}: completed", flush=True)
    save_csv(out / "prediction.csv", measures)
    save_csv(out / "coverage.csv", coverage)
    if edits:
        save_csv(out / "substitutions.csv", edits)
    report = {
        "scope": config["scope"],
        "patients": len(patients),
        "fits": len(measures),
        "intervention_rows": len(edits),
        "coverage": coverage,
        "histological_realism_validated": False,
        "interpretation": "Direct concept prediction is a positive control, not evidence of concept use in diagnosis.",
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (config_path, ROOT / "data/bcss_extended_manifest.csv", Path(__file__))
        },
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
