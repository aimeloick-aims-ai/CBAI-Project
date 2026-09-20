"""Describe the fixed expanded experiment without retraining or selecting results."""

import json
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "reports/bracs_expanded"
    result = json.loads((folder / "results.json").read_text())
    test = result["internal_test"]
    n, p, z = test["patients"], test["accuracy"], 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    matrix = test["confusion_matrix_N_IC"]
    lines = [
        "# BRACS : expérience binaire élargie",
        "",
        "Étude pilote N (normal) contre IC (carcinome invasif), une région par patient.",
        "34 patients d'apprentissage, 10 de validation et 22 nouveaux patients de test interne.",
        "Encodeur PCam gelé, 120 époques, graines 11/23/37 et moyenne des probabilités.",
        "",
        "## Validation",
        "",
        "| Modèle | Balanced accuracy | AUC |",
        "|---|---:|---:|",
    ]
    for name, scores in result["validation"].items():
        lines.append(f"| {name} | {scores['balanced_accuracy']:.1%} | {scores['auc']:.3f} |")
    lines += [
        "",
        f"Modèle retenu avant ouverture du test : **{result['selected_model']}**.",
        "",
        "## Test interne",
        "",
        f"- Exactitude : {p:.1%} ; intervalle de Wilson à 95 % : [{center - half:.1%}, {center + half:.1%}].",
        f"- Balanced accuracy : {test['balanced_accuracy']:.1%}.",
        f"- AUC : {test['auc']:.3f}.",
        f"- Matrice de confusion (lignes réelles, colonnes prédites ; ordre N, IC) : {matrix}.",
        "",
        "## Portée scientifique",
        "",
        "Ce résultat teste un socle de classification, pas la contribution interventionnelle GCIA.",
        "Les graphes représentent des patches, pas des cellules segmentées. Les expériences",
        "multi-cohortes, l'ancrage pathologique des concepts et la fidélité des interventions",
        "ne sont pas établis par cette expérience. L'abstract complet n'est donc pas validé.",
        "",
        "Le recrutement privilégie les petits fichiers et le test ne contient que 22 patients.",
        "La résolution physique est inconnue après redimensionnement. Il ne s'agit pas du test",
        "officiel BRACS. Certains anciens patients de test sont maintenant en apprentissage ;",
        "le nouveau test est distinct et ne permet pas de comparaison directe avec l'ancien score.",
        "Aucun ajustement ne doit être guidé par ce nouveau test désormais consulté.",
        "",
        "Traçabilité : results.json, selection.json, predictions.csv et test_opened.json",
        "dans ce dossier ; protocole configs/bracs_expanded_protocol.json.",
    ]
    (folder / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
