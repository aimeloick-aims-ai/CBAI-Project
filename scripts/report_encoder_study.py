"""Summarize the fixed experiment, retaining all outcomes and patient-level results."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "reports/encoder_study"
    with (out / "prediction.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["features"], row["model"])].append(row)
    lines = [
        "# Comparaison avec encodeur histopathologique gele",
        "",
        "Experience exploratoire executee une seule fois selon configs/encoder_protocol.json.",
        "Six patients de developpement deja explores ; deux nouveaux patients reserves.",
        "Aucun reglage choisi sur leurs scores. Chaque condition est rapportee.",
        "",
        "## Audit visuel",
        "",
        "La planche alignment.png a ete examinee : pas de decalage grossier apparent a cette",
        "echelle. Les contours regionaux sont compatibles visuellement avec l'image.",
        "Cette inspection technique ne valide ni chaque annotation ni les interventions",
        "par un pathologiste. Noir = autres classes ou zones ignorees, pas forcement fond.",
        "",
        "## Resultats",
        "",
        "Moyennes sur patients et trois graines ; plus faible est meilleur.",
        "La fraction regionale est la moyenne des patches valides, pas celle de toute la lame.",
        "",
        "| Ensemble | Caracteristiques | Modele | MSE patches | MAE region | MSE constante | MAE constante |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for (split, features, model), group in sorted(groups.items()):
        values = [
            np.mean([float(r[key]) for r in group])
            for key in ("patch_mse", "region_mae", "constant_patch_mse", "constant_region_mae")
        ]
        lines.append(
            f"| {split} | {features} | {model} | " + " | ".join(f"{v:.4f}" for v in values) + " |"
        )
    lines += [
        "",
        "## Nouveaux patients separement",
        "",
        "| Patient | Caracteristiques | Modele | MSE moyenne |",
        "|---|---|---|---:|",
    ]
    patient_groups = defaultdict(list)
    for row in rows:
        if row["split"] == "held_out":
            patient_groups[(row["patient"], row["features"], row["model"])].append(
                float(row["patch_mse"])
            )
    for (patient, features, model), values in sorted(patient_groups.items()):
        lines.append(f"| {patient} | {features} | {model} | {np.mean(values):.4f} |")
    lines += [
        "",
        "## Portee et sources",
        "",
        "PCam preentraine MobileNetV3-Small : poids officiels TIACentre, empreinte verifiee,",
        "3 833 181 octets. RGB /255, patches 96x96 au pas physique demande de 1 micrometre.",
        "Le backbone reste gele. MLP et GCN utilisent les memes representations.",
        "Reference constante : moyenne par patient des fractions tumorales du train.",
        "La normalisation est apprise uniquement sur les patients du train.",
        "",
        "Poids et licence CC0 documentes par [TIAToolbox](https://tia-toolbox.readthedocs.io/en/stable/pretrained.html).",
        "Lymph nodes PCam et tissu mammaire BCSS sont des domaines distincts.",
        "Les patches de 96 pixels et la reference constante ponderee par patient different",
        "de certains essais precedents : la comparaison juste est celle de ce tableau.",
        "",
        "Deux nouveaux patients ne suffisent pas a une validation populationnelle.",
        "C'est une prediction directe de concept, pas une preuve de son utilisation diagnostique.",
        "Aucun test de significativite, choix du meilleur seed ou validation GCIA ne sont revendiques.",
        "",
        "Reproduction : scripts/run_encoder_study.py ; refus decrasement des resultats existants.",
        "Synthese sans reentrainement : scripts/report_encoder_study.py.",
        "Les checkpoints, normalisations, embeddings, protocole et empreintes sont conserves ici.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, ((patient, features, model), values) in enumerate(sorted(patient_groups.items())):
        ax.scatter(j, np.mean(values), color="navy")
    ax.set_xticks(
        range(len(patient_groups)), ["\n".join(k) for k in sorted(patient_groups)], fontsize=7
    )
    ax.set_ylabel("Patch MSE (mean over 3 seeds)")
    ax.set_title("Two held-out patients; descriptive results, no confidence intervals")
    fig.tight_layout()
    fig.savefig(out / "held_out_results.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines[:25]))


if __name__ == "__main__":
    main()
