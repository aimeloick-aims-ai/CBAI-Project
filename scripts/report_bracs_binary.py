"""Render the locked binary pilot results without reopening or retraining models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "reports/bracs_binary"
    r = json.loads((out / "results.json").read_text())
    t = r["internal_test"]
    lines = [
        "# Pilote BRACS : normal versus carcinome invasif",
        "",
        "40 RoI de 40 patients distincts : 20 train, 10 validation et 10 test interne.",
        "Chaque groupe comporte autant de regions normales que de regions invasives.",
        "Les trois graines sont assemblees par moyenne des probabilites ; seuil fixe a 0,5.",
        "",
        "## Selection sur validation",
        "",
        "| Modele | Balanced accuracy | AUC |",
        "|---|---:|---:|",
    ]
    for name, scores in r["validation"].items():
        lines.append(f"| {name} | {scores['balanced_accuracy']:.3f} | {scores['auc']:.3f} |")
    lines += [
        "",
        f"Modele retenu : **{r['selected_model']}**. En cas d egalite, preference fixee au modele simple.",
        "Selection sauvegardee avant decodage des images de test.",
        "",
        "## Evaluation unique du modele retenu sur le test interne",
        "",
        f"- Patients : {t['patients']}.",
        f"- Accuracy : {t['accuracy']:.1%}.",
        f"- Balanced accuracy : {t['balanced_accuracy']:.1%}.",
        f"- AUC : {t['auc']:.3f}.",
        "- Reference constante : balanced accuracy 50 %.",
        "",
        "Matrice : lignes = classes reelles, colonnes = classes predites, ordre N puis IC.",
        f"`{t['confusion_matrix_N_IC']}`",
        "",
        "## Portee",
        "",
        "Les patients sont distincts entre groupes. Le test est interne, issu de la",
        "partition officielle train : ce n est ni le test officiel BRACS ni BACH.",
        "La selection des petites images introduit un biais ; dix sujets de test donnent",
        "une estimation tres imprecise. Un changement de prediction change l accuracy de 10 points.",
        "La distinction de ces deux extremes ne couvre pas les lesions intermediaires.",
        "Aucun resultat clinique ni validation interventionnelle GCIA ne sont revendiques.",
        "",
        "PCam est gele, mais son domaine et son echelle different des RoI BRACS.",
        "Chaque image est redimensionnee a 768x768 puis divisee en 64 patches 96x96.",
        "L echelle physique des RoI BRACS n est pas etablie ; cette limitation est conservee.",
        "",
        "## Tracabilite",
        "",
        "Protocoles : configs/bracs_binary_protocol.json, selection.json, test_opened.json.",
        "Predictions individuelles : predictions.csv. Poids et normalisations : *.pt.",
        "Acquisition : data/bracs_binary_manifest.csv et rapports dans reports/bracs_binary_plan.",
        "Le script run_bracs_binary.py refuse de rouvrir le test apres cette execution.",
        "Ce test est maintenant examine et ne doit plus servir a optimiser la methode.",
        "Regenerer ce rapport sans entrainement : scripts/report_bracs_binary.py.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    matrix = np.array(t["confusion_matrix_N_IC"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["Normal", "Invasive"])
    ax.set_yticks([0, 1], ["Normal", "Invasive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed RoI label")
    ax.set_title("Internal test: 10 patients")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(json.dumps(t, indent=2))


if __name__ == "__main__":
    main()
