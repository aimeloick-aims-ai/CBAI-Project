"""Refresh the completed cell experiment dashboard without rerunning experiments."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / "reports"
    state = json.loads((reports / "finalization_status.json").read_text())
    coverage = json.loads((reports / "cell_cohort/completion.json").read_text())
    if state["stage"] != "cell_experiments_complete" or not coverage["all_patients_accounted_for"]:
        raise ValueError("The experiment has not completed; do not show a completed dashboard")
    state.update(coverage)
    (reports / "segmentation_progress.json").write_text(json.dumps(state, indent=2))
    page = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<title>GCIA — expériences cellulaires terminées</title>
<style>body{{font:18px sans-serif;max-width:850px;margin:3em auto;line-height:1.6}}</style>
<h1>Expériences cellulaires terminées</h1>
<p><strong>{coverage["source_patients"]}/{coverage["source_patients"]} patients traités :
{coverage["usable"]} graphes exploitables et {coverage["excluded"]} exclusion documentée.</strong></p>
<p>Le compteur de graphes ne doit pas atteindre 66 : un crop ne contenait pas deux noyaux retenus.
Aucun patient n'est encore en attente dans cette cohorte.</p>
<p>Les 14 images MoNuSeg sont évaluées. L'entraînement de GCN, GraphSAGE et GATv2 est terminé.</p>
<ul><li><a href="CELL_EXPERIMENTS.md">Résultats des trois GNN</a></li>
<li><a href="cell_cohort/excluded.csv">Patient exclu et motif</a></li>
<li><a href="segmentation_evaluation/RESULTS.md">Évaluation de segmentation</a></li></ul>
<p>Restent pour le projet scientifique : audits XAI des modèles cellulaires, validation externe BACH,
validation de segmentation BRACS et revue anatomopathologique. Les graphes restent candidats.</p></html>"""
    (reports / "segmentation_progress.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
