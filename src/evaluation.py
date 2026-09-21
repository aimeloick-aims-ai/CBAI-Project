"""Conservative interpretation of independently validated concept interventions."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditEvidence:
    independent_annotations: bool
    held_out_probe_passed: bool
    intervention_selective: bool
    plausibility_validated: bool
    matched_controls_valid: bool
    effect_lower: float
    effect_upper: float
    equivalence_margin: float
    margin_fixed_before_evaluation: bool


def classify_evidence(evidence: AuditEvidence) -> str:
    """An uncertain effect is not evidence of non-use; labels are audit-specific."""
    if not all(
        math.isfinite(x)
        for x in (evidence.effect_lower, evidence.effect_upper, evidence.equivalence_margin)
    ):
        raise ValueError("Effect bounds and margin must be finite")
    if evidence.effect_lower > evidence.effect_upper or evidence.equivalence_margin <= 0:
        raise ValueError("Invalid interval or equivalence margin")
    if not evidence.independent_annotations:
        return "concept_unvalidated"
    if not evidence.held_out_probe_passed:
        return "encoding_not_established"
    if not (
        evidence.intervention_selective
        and evidence.plausibility_validated
        and evidence.matched_controls_valid
    ):
        return "intervention_inconclusive"
    if not evidence.margin_fixed_before_evaluation:
        return "effect_threshold_not_prespecified"
    margin = evidence.equivalence_margin
    if evidence.effect_lower > margin:
        return "encoded_with_supported_target_specific_effect"
    if evidence.effect_lower > -margin and evidence.effect_upper < margin:
        return "encoded_with_negligible_excess_effect_under_tested_intervention"
    if evidence.effect_upper < -margin:
        return "encoded_with_reverse_target_control_effect"
    return "encoded_with_uncertain_effect"


# Canonical evaluation, uncertainty and reporting. Historical AuditEvidence stays compatible.
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch_geometric.data import Batch

from src.interventions import matched_interventions
from src.probes import fit_probe
from src.tcav import heldout_sensitivity


def load_config(root):
    with (Path(root) / "configs/study.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cluster_interval(values, groups, config, seed):
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    means = np.array([values[groups == group].mean() for group in np.unique(groups)])
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), (config["bootstrap_repetitions"], len(means)))].mean(
        1
    )
    tail = (1 - config["confidence"]) / 2
    return float(means.mean()), *np.quantile(draws, [tail, 1 - tail]).tolist()


def grouped_permutation(labels, groups, rng):
    """Permute entire equal-size patient blocks, never individual ROIs.

    For factorial synthetic clusters, a random XOR flips the binary concept for
    an entire replicate; exchanging identical balanced blocks would be a no-op.
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    unique = np.unique(groups)
    blocks = [np.flatnonzero(groups == group) for group in unique]
    result = labels.copy()
    if all(len(b) == 1 for b in blocks):
        return labels[rng.permutation(len(labels))]
    if all(set(np.unique(labels[b])) == {0, 1} for b in blocks):
        for block in blocks:
            if rng.integers(2):
                result[block] = 1 - labels[block]
        return result
    # General repeated-ROI cohorts: exchange only equal-size patient blocks.
    for size in sorted({len(b) for b in blocks}):
        subset = [b for b in blocks if len(b) == size]
        for recipient, donor in zip(subset, rng.permutation(len(subset)), strict=True):
            result[recipient] = labels[subset[donor]]
    return result


def probe_metrics(probe, h, labels, continuous):
    if continuous:
        predicted = probe.predict(h)
        return {
            "r2": float(r2_score(labels, predicted)),
            "mae": float(mean_absolute_error(labels, predicted)),
        }
    return {
        "auroc": float(roc_auc_score(labels, probe.predict_proba(h)[:, 1])),
        "balanced_accuracy": float(balanced_accuracy_score(labels, probe.predict(h))),
    }


def probe_analysis(
    htrain, ytrain, htest, ytest, train_groups, test_groups, seed, config, continuous
):
    """Public held-out interface: fits all scaling and concept directions on train only."""
    if set(train_groups) & set(test_groups):
        raise ValueError("Probe train/test patient overlap")
    probe = fit_probe(htrain, ytrain, seed, continuous=continuous, config=config["probe"])
    metrics = probe_metrics(probe, htest, ytest, continuous)
    primary = "r2" if continuous else "auroc"
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(config["probe"]["permutations"]):
        permuted = grouped_permutation(ytrain, train_groups, rng)
        shuffled = fit_probe(htrain, permuted, seed, continuous=continuous, config=config["probe"])
        # The null must also break held-out feature/label association. Shuffling
        # training labels alone can leave a random coefficient aligned with a
        # perfectly encoded concept and is not an association permutation test.
        permuted_test = grouped_permutation(ytest, test_groups, rng)
        null.append(probe_metrics(shuffled, htest, permuted_test, continuous)[primary])
    pvalue = (1 + sum(v >= metrics[primary] for v in null)) / (1 + len(null))
    return probe, metrics, float(np.mean(null)), pvalue


def probabilities(model, graphs, batch_size):
    result = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size])
            result.append(model(batch.x, batch.edge_index, batch.batch).softmax(-1))
    return torch.cat(result)


def representations(model, graphs, batch_size):
    layers = [[] for _ in range(len(model.layers) + 1)]
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size])
            for dest, h in zip(
                layers, model.representations(batch.x, batch.edge_index, batch.batch), strict=True
            ):
                dest.append(h)
    return [torch.cat(h) for h in layers]


def effect_interpretation(valid, excess_low, excess_high, target_high, config):
    margin = config["equivalence_margin"]
    if not valid:
        return "inconclusive: intervention validity failed"
    if excess_low > margin:
        return "model-level functional reliance under the tested intervention"
    # Small excess alone is not non-use: the target itself must be negligible.
    if -margin < excess_low and excess_high < margin and target_high < margin:
        return "negligible effect under the tested intervention"
    return "inconclusive"


def evaluate_model(model, random_model, splits, stage, seed, config, mean, scale):
    train, test = splits["train"], splits["test"]
    groups = [g.patient_id for g in test]
    train_groups = [g.patient_id for g in train]
    base = {
        "stage": stage,
        "dataset": "synthetic" if stage == "synthetic" else "BRACS",
        "model": model.architecture,
        "seed": seed,
        "record_type": "aggregate",
    }
    rows = []
    p = probabilities(model, test, config["batch_size"])
    labels = np.array([int(g.y) for g in test])
    for metric, value in {
        "accuracy": accuracy_score(labels, p.argmax(-1)),
        "balanced_accuracy": balanced_accuracy_score(labels, p.argmax(-1)),
        "auroc": roc_auc_score(labels, p[:, 1]),
    }.items():
        rows.append(
            {
                **base,
                "analysis": "classification",
                "metric": metric,
                "value": float(value),
                "n_graphs": len(test),
                "n_groups": len(set(groups)),
            }
        )
    random_p = probabilities(random_model, test, config["batch_size"])
    rows.append(
        {
            **base,
            "analysis": "sanity_check",
            "metric": "randomized_model_accuracy",
            "value": float(accuracy_score(labels, random_p.argmax(-1))),
        }
    )
    htrain = representations(model, train, config["batch_size"])
    htest = representations(model, test, config["batch_size"])
    continuous = stage == "bracs"
    for index, concept in enumerate(config[stage]["concepts"]):
        cbase = {**base, "concept": concept}
        ytrain = np.array([float(g.concepts[0, index]) for g in train])
        ytest = np.array([float(g.concepts[0, index]) for g in test])
        for layer, (ht, he) in enumerate(zip(htrain, htest, strict=True)):
            probe, scores, null, pvalue = probe_analysis(
                ht.numpy(),
                ytrain,
                he.numpy(),
                ytest,
                train_groups,
                groups,
                seed,
                config,
                continuous,
            )
            for metric, value in scores.items():
                rows.append(
                    {
                        **cbase,
                        "analysis": "probe",
                        "layer": layer,
                        "probe_metric": metric,
                        "probe_value": value,
                        "permutation_p": pvalue,
                    }
                )
            rows.append(
                {
                    **cbase,
                    "analysis": "sanity_check",
                    "layer": layer,
                    "metric": "shuffled_concept_probe_" + ("r2" if continuous else "auroc"),
                    "value": null,
                    "permutation_p": pvalue,
                }
            )
        primary = "r2" if continuous else "auroc"
        threshold = config["probe"]["continuous_threshold" if continuous else "binary_threshold"]
        encoded = scores[primary] >= threshold and pvalue <= config["probe"]["alpha"]
        tcav = heldout_sensitivity(model, htest[-1], probe, config["numerical_tolerance"])
        final = {
            **cbase,
            "layer": len(model.layers),
            "probe_metric": primary,
            "probe_value": scores[primary],
            "permutation_p": pvalue,
            "encoded": encoded,
            "tcav_value": tcav,
        }
        rows.append({**final, "analysis": "tcav"})
        rng = np.random.default_rng(seed + index)
        targets, controls, validity, audits = [], [], [], []
        for graph in test:
            target, matched, valid, audit = matched_interventions(
                graph, index, stage, config, rng, raw_scale=scale, raw_mean=mean
            )
            targets.append(target)
            controls.extend(matched)
            validity.append(valid)
            audits.append(audit)
        target_p = probabilities(model, targets, config["batch_size"])
        control_p = probabilities(model, controls, config["batch_size"]).reshape(
            len(test), config["matched_controls"], -1
        )
        target = (target_p - p).abs().sum(-1).numpy() / 2
        control = (control_p - p[:, None]).abs().sum(-1).mean(-1).numpy() / 2
        excess = target - control
        effect, low, high = cluster_interval(excess, groups, config, seed)
        target_mean, target_low, target_high = cluster_interval(target, groups, config, seed)
        control_mean, control_low, control_high = cluster_interval(control, groups, config, seed)
        valid = all(validity)
        interpretation = effect_interpretation(valid, low, high, target_high, config)
        rows.append(
            {
                **final,
                "analysis": "gcia",
                "target_effect": target_mean,
                "control_effect": control_mean,
                "gcia_excess": effect,
                "ci_low": low,
                "ci_high": high,
                "target_ci_low": target_low,
                "target_ci_high": target_high,
                "control_ci_low": control_low,
                "control_ci_high": control_high,
                "intervention_valid": valid,
                "valid_fraction": float(np.mean(validity)),
                "invalid_reason": "; ".join(
                    sorted({a.get("invalid_reason", "") for a in audits} - {""})
                ),
                "interpretation": interpretation,
                "n_graphs": len(test),
                "n_groups": len(set(groups)),
                "budget": float(np.mean([a["budget"] for a in audits])),
                "control_budget": float(np.mean([a["control_budget"] for a in audits])),
                "concept_shift": float(np.mean([a["concept_shift"] for a in audits])),
                "control_concept_shift": float(
                    np.mean([a["control_concept_shift"] for a in audits])
                ),
            }
        )
        for i, graph in enumerate(test):
            rows.append(
                {
                    **cbase,
                    "analysis": "gcia",
                    "record_type": "observation",
                    "sample_id": graph.sample_id,
                    "patient_id": graph.patient_id,
                    "target_effect": float(target[i]),
                    "control_effect": float(control[i]),
                    "gcia_excess": float(excess[i]),
                    "intervention_valid": validity[i],
                    **audits[i],
                }
            )
        random_target = probabilities(random_model, targets, config["batch_size"])
        random_control = probabilities(random_model, controls, config["batch_size"]).reshape(
            len(test), config["matched_controls"], -1
        )
        rt = (random_target - random_p).abs().sum(-1).numpy() / 2
        rc = (random_control - random_p[:, None]).abs().sum(-1).mean(-1).numpy() / 2
        re, rl, rh = cluster_interval(rt - rc, groups, config, seed)
        rows.append(
            {
                **cbase,
                "analysis": "sanity_check",
                "metric": "randomized_model_gcia_excess",
                "value": re,
                "ci_low": rl,
                "ci_high": rh,
                "target_effect": float(rt.mean()),
                "control_effect": float(rc.mean()),
                "interpretation": "random networks may also respond; sensitivity alone does not establish learned task reliance",
            }
        )
        print(
            f"{stage} {model.architecture} seed={seed} {concept}: probe={scores[primary]:.3f}, "
            f"TV={target_mean:.4f}, excess={effect:.4f} [{low:.4f}, {high:.4f}], {interpretation}",
            flush=True,
        )
    return rows


def source_hashes(root):
    paths = [
        *sorted((root / "src").glob("*.py")),
        root / "experiments/synthetic/run.py",
        root / "experiments/bracs/run.py",
        root / "configs/study.yaml",
        root / "tests/test_gcia.py",
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save_stage(root, stage, rows, metadata, config):
    output = root / "results"
    output.mkdir(exist_ok=True)
    summary = output / "summary.csv"
    old = []
    if summary.exists():
        with summary.open(newline="", encoding="utf-8") as handle:
            old = [r for r in csv.DictReader(handle) if r["stage"] != stage]
    required = [
        "stage",
        "analysis",
        "record_type",
        "dataset",
        "model",
        "seed",
        "concept",
        "layer",
        "probe_metric",
        "probe_value",
        "tcav_value",
        "target_effect",
        "control_effect",
        "gcia_excess",
        "ci_low",
        "ci_high",
        "intervention_valid",
        "interpretation",
    ]
    all_rows = old + rows
    columns = required + sorted(set().union(*(r.keys() for r in all_rows)) - set(required))
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    path = output / "run_manifest.json"
    manifest = (
        json.loads(path.read_text())
        if path.exists()
        else {"stages": {}, "tests_result": "not yet run"}
    )
    manifest.update(
        {
            "git_commit_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, text=True
                ).strip()
            ),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "config": config,
            "source_hashes": source_hashes(root),
            "seeds": config["seeds"],
            "package_versions": {
                p: importlib.metadata.version(p)
                for p in [
                    "torch",
                    "torch-geometric",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "PyYAML",
                    "pytest",
                    "ruff",
                ]
            },
            "missing_evidence": config["bracs"]["missing_evidence"],
            "effect": "total variation; target minus mean of matched controls; repeated controls averaged within graph",
            "implementation_correction": "An initial synthetic execution exposed an invalid association null: train-only label shuffling can leave a random direction aligned with the true held-out concept. Final results rerun independent grouped train AND held-out label permutations. No architecture, optimizer, data, intervention, margin or threshold changed. A regression test covers exact visible encoding.",
            "uncertainty": "percentile bootstrap of patient/independent-replicate means; pointwise intervals, exploratory across concepts/layers/seeds",
            "equivalence": "margin fixed in study.yaml before evaluation; both excess CI within margin and target upper CI below margin required",
            "tcav": "final pooled layer only, train-fitted increasing-concept direction; fraction of positive class-1 probability derivatives",
            "controls": "feature: same coordinate, node count and L2 magnitude, balanced signs; relation: same replaced-edge count, degree and connected cycle, original mixing fraction",
            "limitations": [
                "Synthetic feature controls change within-graph variability and can leave generator support.",
                "BRACS area-only edits do not co-edit axes or image pixels; technical feature reliance, not validated morphological intervention.",
                "Random-model response does not establish learned task reliance.",
                "No biological causality claim.",
            ],
        }
    )
    manifest["stages"][stage] = {
        **metadata,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes(root),
        "config": config,
        "status": "completed",
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(root)


def write_report(root):
    with (root / "results/summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads((root / "results/run_manifest.json").read_text())
    effects = [r for r in rows if r["analysis"] == "gcia" and r["record_type"] == "aggregate"]

    def truth(value):
        return str(value).lower() == "true"

    def matches(stage, concept, model=None):
        return [
            r
            for r in effects
            if r["stage"] == stage
            and r["concept"] == concept
            and (model is None or r["model"] == model)
        ]

    def count(rs, predicate):
        return f"{sum(predicate(r) for r in rs)}/{len(rs)} model-seed runs"

    used = matches("synthetic", "C_used")
    unused = matches("synthetic", "C_unused")
    relational = matches("synthetic", "C_relational", "GraphSAGE")
    noedge = matches("synthetic", "C_relational", "DeepSets")
    reliance = lambda r: r["interpretation"].startswith("model-level")
    negligible = lambda r: r["interpretation"].startswith("negligible")
    lines = [
        "# Canonical GCIA study",
        "",
        "Question: does concept decodability imply functional reliance in histopathology GNNs?",
        "",
        (
            "**Scope: development/exploratory; no biological causality.** The only supported dependence wording is "
            "model-level functional reliance under the tested intervention. No test-based tuning was performed."
        ),
        "",
        (
            "Generator ground truth: binary OR of C_used and C_relational with independent label flips. "
            "C_unused is visible but absent from the label function. Eight factorial variants share a replicate; "
            "paired relations have identical features and node counts. This construction is not evidence of learned use."
        ),
        "",
        "## Answers to the ten study questions",
        "",
        "1. **Encoded and used:** "
        + count(used, lambda r: truth(r["encoded"]) and reliance(r))
        + " meet both criteria.",
        "2. **Encoded but unused:** "
        + count(unused, lambda r: truth(r["encoded"]) and negligible(r))
        + " support negligible effect under the tested intervention; other runs are inconclusive.",
        "3. **Relational use:** "
        + count(relational, reliance)
        + " GraphSAGE runs show excess effects above the prespecified margin.",
        "4. **No-edge control:** "
        + count(noedge, lambda r: abs(float(r["target_effect"])) == 0)
        + " DeepSets runs have exactly zero edge-only response.",
        "5. **Valid BRACS concept:** measured mean segmented nuclear area in pixel squared units, a technical positive control already explicit in node features. No independent histological concept is validated.",
        "6. **BRACS encoding:** "
        + count(matches("bracs", "nuclear_area_pixels"), lambda r: truth(r["encoded"]))
        + " pass held-out R2 and patient-group permutation criteria.",
        "7. **BRACS TCAV:** see final-layer values below. Values are descriptive directional sensitivities, not significance or reliance tests.",
        "8. **BRACS GCIA:** "
        + count(matches("bracs", "nuclear_area_pixels"), reliance)
        + " show supported excess area-coordinate response; see validity and confidence intervals below.",
        "9. **Unsupported:** Concept 2 and Concept 3 are BLOCKED: requires independent concept annotations; relational histology also requires trustworthy cell/compartment labels. MPP and independent BRACS segmentation validation are missing.",
        "10. **Exploratory:** historical BRACS test patients were already explored; there is no untouched formally locked confirmatory test set.",
        "",
        "## Frozen-model results",
        "",
        "| Stage | Model | Seed | Concept | Probe | Encoded | TCAV | Target TV | Control TV | Excess [CI] | Valid | Interpretation |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---|---|---|",
    ]
    for r in effects:
        lines.append(
            f"| {r['stage']} | {r['model']} | {r['seed']} | {r['concept']} | {float(r['probe_value']):.3f} | {r['encoded']} | {float(r['tcav_value']):.3f} | {float(r['target_effect']):.4f} | {float(r['control_effect']):.4f} | {float(r['gcia_excess']):.4f} [{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}] | {r['intervention_valid']} | {r['interpretation']} |"
        )
    lines += [
        "",
        (
            "Probe is AUROC for synthetic binary concepts and R2 for continuous BRACS area. "
            "Balanced accuracy and MAE, all layerwise probes, permutation nulls, randomized-model checks "
            "and patient/replicate observations are in summary.csv."
        ),
        "",
        "## Classification",
        "",
        "| Stage | Model | Seed | Accuracy | Balanced accuracy | AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stage in ("synthetic", "bracs"):
        for model in manifest["config"]["models"]:
            for seed in manifest["config"]["seeds"]:
                selected = {
                    r["metric"]: float(r["value"])
                    for r in rows
                    if r["stage"] == stage
                    and r["model"] == model
                    and str(r["seed"]) == str(seed)
                    and r["analysis"] == "classification"
                }
                if selected:
                    lines.append(
                        f"| {stage} | {model} | {seed} | {selected['accuracy']:.3f} | {selected['balanced_accuracy']:.3f} | {selected['auroc']:.3f} |"
                    )
    demonstrated = any(truth(r["encoded"]) and negligible(r) for r in unused)
    random_positive = [
        r
        for r in rows
        if r.get("metric") == "randomized_model_gcia_excess"
        and float(r["ci_low"]) > manifest["config"]["equivalence_margin"]
    ]
    random_warning = "; ".join(
        f"{r['stage']} {r['model']} seed {r['seed']} {r['concept']}: "
        f"{float(r['value']):.4f} [{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}]"
        for r in random_positive
    )
    failures = sorted(
        {
            f"{r['stage']} patient/replicate {r['patient_id']}: {r.get('invalid_reason', '')}"
            for r in rows
            if r["record_type"] == "observation" and r["intervention_valid"] == "False"
        }
    )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "Randomized-model sanity checks exceeding the excess-effect margin: "
        + (random_warning or "none")
        + ". These are failures of interpreting positive GCIA alone as learned task reliance; the model responds even before training.",
        "",
        "Intervention validity failures (all retained in the analysis): "
        + ("; ".join(failures) if failures else "none"),
        "",
        (
            "The synthetic learned-model results demonstrate decodable != necessarily functionally used for at least one run."
            if demonstrated
            else "The current learned-model results do NOT demonstrate encoded-but-negligible use; generator truth alone cannot rescue this failure."
        ),
        (
            "Recovery must be judged per seed; the table exposes all negative and inconclusive findings. "
            "A small excess by itself is insufficient for negligible use: the target-effect upper interval must also lie below the fixed margin."
        ),
        "",
        (
            "GCIA intervals resample patients (BRACS) or independent paired replicates (synthetic), after averaging controls within graph. "
            "They are pointwise exploratory intervals, not multiplicity-adjusted confirmatory evidence. "
            "Probe null fits use permuted training labels and are scored against independently permuted held-out labels; both permutations respect patient blocks. Synthetic binary labels are flipped jointly within replicate. No held-out labels enter fitting."
        ),
        "",
        (
            "Matched feature controls preserve the edited column, node count and L2 budget but redistribute signs; they change within-graph variability. "
            "Synthetic relation controls preserve mixing fraction, exact replaced-edge budget, degrees, edge count, no loops/duplicates, and connectivity. "
            "Synthetic graphs have no physical coordinates, so edge-length matching is unavailable."
        ),
        "",
        (
            "BRACS uses existing graphs without resegmentation. Measured area is not an independent pathology annotation. "
            "Area-only edits leave correlated axes and pixels unchanged; biological plausibility is unvalidated. "
            "Any supported result is conditional on this numerical feature intervention. Population and histological relation interventions were not run."
        ),
        "",
        (
            "TCAV estimates local class-1 probability sensitivity along a train-fitted direction, only at the final pooled layer. "
            "An increasing concept can be associated with the negative class; a low positive fraction is not proof of insensitivity. NaN means a zero-length CAV, not a measured zero response."
        ),
        "",
        (
            "Randomized networks may respond to interventions and encode input features. Their checks are reported without requiring zero sensitivity. "
            "Shuffled-concept null probe values and permutation p-values must be considered with the encoding threshold."
        ),
        "",
        "## Missing evidence and next step",
        "",
    ]
    lines += ["- " + item for item in manifest["missing_evidence"]]
    lines += [
        "",
        (
            "Next scientific step: audit the synthetic feature controls for departure from generator support and preregister a revised control protocol for a new benchmark run; do not retune this run. "
            "For BRACS, independently validate nuclear segmentation and obtain patient-linked expert concept annotations before locking an untouched patient-disjoint confirmatory cohort."
        ),
        "",
        "## Software verification",
        "",
        "The final run corrects the permutation null to shuffle labels independently by group in both train and held-out partitions. This is an implementation correction, with unchanged training, interventions and decision thresholds; a regression test covers the error.",
        "",
        "```json",
        json.dumps(manifest.get("tests_result"), indent=2),
        "```",
        "",
        (
            "Historical pilot artifacts remain unchanged in GCIA_archive/legacy and results/synthetic_v1. "
            "Only summary.csv, report.md and run_manifest.json are canonical full-study outputs."
        ),
    ]
    (root / "results/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_software(root):
    checks = []
    commands = [
        (
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "src",
                "experiments/synthetic",
                "experiments/bracs",
                "tests",
            ],
            root,
        ),
        (
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--check",
                "src",
                "experiments/synthetic",
                "experiments/bracs",
                "tests",
            ],
            root,
        ),
        ([sys.executable, "-m", "pytest", "-q"], root),
        ([sys.executable, "-m", "pytest", "-q", "tests/test_gcia.py"], root),
        (
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            root.parent / "GCIA_archive/legacy",
        ),
    ]
    for command, cwd in commands:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        checks.append(
            {
                "command": command,
                "cwd": str(cwd),
                "returncode": result.returncode,
                "output": result.stdout + result.stderr,
            }
        )
        print(result.stdout + result.stderr, flush=True)
    path = root / "results/run_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"stages": {}}
    manifest["tests_result"] = {
        "checks": checks,
        "passed": all(c["returncode"] == 0 for c in checks),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes(root),
    }
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if (root / "results/summary.csv").exists():
        write_report(root)
    return manifest["tests_result"]["passed"]
