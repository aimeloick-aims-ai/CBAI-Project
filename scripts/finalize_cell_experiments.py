"""Resume interrupted cohort, train fixed architectures, report; fail visibly on errors."""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main():
    status = REPORTS / "finalization_status.json"

    def update(stage, **details):
        count = len(list((REPORTS / "cell_cohort").glob("*.pt")))
        progress = {
            "stage": stage,
            "updated_utc": datetime.now(UTC).isoformat(),
            "constructed_graphs": count,
            "evaluated_images": 14,
            **details,
        }
        status.write_text(json.dumps(progress, indent=2))
        (REPORTS / "segmentation_progress.json").write_text(json.dumps(progress, indent=2))
        (REPORTS / "segmentation_progress.html").write_text(
            f"""<!doctype html><html lang="fr"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>GCIA progression</title><body style="font:18px sans-serif;margin:3em;line-height:1.6"><h1>Expériences cellulaires</h1><p>Étape : {stage}</p><p>Segmentation : 14/14 images évaluées. Graphes enregistrés : {count} sur 66 patients à traiter. Les exclusions seront consignées séparément.</p><p>Dernière mise à jour UTC : {progress["updated_utc"]}</p><ul><li><a href="segmentation_evaluation/SEGMENTATION.md">Résultats de segmentation terminés</a></li><li><a href="finalization_status.json">État détaillé et éventuelle erreur</a></li><li><a href="CELL_EXPERIMENTS.md">Bilan cellulaire, disponible en fin de calcul</a></li></ul><p>La fin de ce calcul ne signifie pas que la validation scientifique complète est acquise.</p></body></html>""",
            encoding="utf-8",
        )

    steps = [
        (
            "building_remaining_graphs",
            [
                "scripts/prepare_cell_cohort.py",
                "--weights",
                "data/pretrained/hovernet_fast-pannuke.pth",
                "--output",
                "reports/cell_cohort",
                "--resume",
            ],
        ),
        ("reporting_segmentation", ["scripts/report_segmentation_and_graphs.py"]),
        (
            "training_cell_gnns",
            [
                "scripts/train_cell_gnns.py",
                "--manifest",
                "reports/cell_cohort/manifest.csv",
                "--output",
                "reports/cell_gnn_comparison",
            ],
        ),
    ]
    for stage, command in steps:
        update(stage, log=f"reports/finalize_{stage}.log")
        with (REPORTS / f"finalize_{stage}.log").open("w") as log:
            result = subprocess.Popen(
                [sys.executable, *command],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while True:
                try:
                    result.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    update(stage, log=f"reports/finalize_{stage}.log")
        if result.returncode:
            update(
                "failed",
                failed_step=stage,
                exit_code=result.returncode,
                log=f"reports/finalize_{stage}.log",
            )
            return
    trained = json.loads((REPORTS / "cell_gnn_comparison/results.json").read_text())
    coverage = json.loads((REPORTS / "cell_cohort/completion.json").read_text())
    lines = [
        "# Bilan des expériences cellulaires",
        "",
        f"Patients traités : {coverage['source_patients']} ; graphes exploitables : {coverage['usable']} ; exclusions explicites : {coverage['excluded']}.",
        "",
        "Trois architectures, trois graines chacune, 120 époques fixes. Aucun réglage sur le test.",
        "Le test est historique et déjà exploré. Les graphes restent candidats, à échelle physique inconnue.",
        "",
        "| Architecture | Exactitude | Balanced accuracy | AUC | Patients de test |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in trained["test"].items():
        lines.append(
            f"| {name} | {result['accuracy']:.1%} | {result['balanced_accuracy']:.1%} | {result['auc']:.3f} | {result['patients']} |"
        )
    lines += [
        "",
        "## Ce que cette exécution termine",
        "",
        "Évaluation des 14 images MoNuSeg, traitement de tous les patients BRACS disponibles, exclusions tracées, comparaison des trois GNN cellulaires et sauvegarde des poids/prédictions.",
        "",
        "## Ce qui n'est pas démontré",
        "",
        "La validation de segmentation BRACS, la validation externe BACH, les audits GCIA des nouveaux GNN cellulaires et la revue anatomopathologique ne sont pas achevés par cette exécution. Les audits XAI antérieurs concernent les modèles de patches. Aucune réussite clinique n'est déduite automatiquement des scores.",
        "",
        "Voir `segmentation_evaluation/RESULTS.md`, `cell_cohort/excluded.csv` et `cell_gnn_comparison/results.json`.",
    ]
    (REPORTS / "CELL_EXPERIMENTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update(
        "cell_experiments_complete",
        report="reports/CELL_EXPERIMENTS.md",
        whole_research_project_complete=False,
    )


if __name__ == "__main__":
    main()
