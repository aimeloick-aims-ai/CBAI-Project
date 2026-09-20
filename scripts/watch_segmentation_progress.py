"""Maintain a local progress page without model calls; no inference or downloads."""

import csv
import html
import json
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main():
    deadline = time.monotonic() + 4 * 3600
    while True:
        metrics = REPORTS / "segmentation_evaluation/metrics.csv"
        try:
            rows = list(csv.DictReader(metrics.open())) if metrics.exists() else []
        except (PermissionError, OSError):
            time.sleep(5)
            continue
        evaluated = len({r.get("image") for r in rows if r.get("image")})
        graphs = len(list((REPORTS / "cell_cohort").glob("*.pt")))
        complete = (REPORTS / "segmentation_evaluation/RESULTS.md").exists()
        stage = (
            "complete"
            if complete
            else (
                "building_graphs"
                if (REPORTS / "segmentation_evaluation/summary.json").exists()
                else "evaluating_segmentation"
            )
        )
        expired = time.monotonic() >= deadline
        if expired and not complete:
            stage = "monitoring_timeout_check_logs"
        status = {
            "status": stage,
            "evaluated_images": evaluated,
            "expected_images": 14,
            "constructed_graphs": graphs,
            "expected_graphs": 66,
            "updated_utc": datetime.now(UTC).isoformat(),
            "scientific_validation_of_BRACS": False,
        }
        (REPORTS / "segmentation_progress.json").write_text(json.dumps(status, indent=2))
        body = f"""<!doctype html><html lang="fr"><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>Segmentation et graphes — progression</title><style>body{{font:18px sans-serif;max-width:850px;margin:3em auto;line-height:1.6}} img{{max-width:100%}}</style>
<h1>Évaluation de segmentation et graphes</h1>
<p>État : <strong>{html.escape(stage)}</strong></p>
<p>Images évaluées : {evaluated}/14 — Graphes enregistrés : {graphs}/66.</p>
<p>Les graphes restent candidats : la segmentation BRACS n'est pas directement validée par MoNuSeg.</p>
<ul><li><a href="segmentation_evaluation/metrics.csv">Scores détaillés disponibles</a></li>
<li><a href="segmentation_evaluation/RESULTS.md">Rapport final (disponible après calcul)</a></li>
<li><a href="cell_cohort/manifest.csv">Manifeste des graphes (disponible après construction)</a></li>
<li><a href="segmentation_evaluation_run.log">Journal d'évaluation</a></li>
<li><a href="cell_cohort_run.log">Journal de construction</a></li></ul>
<p>Mise à jour UTC : {status["updated_utc"]}</p></html>"""
        (REPORTS / "segmentation_progress.html").write_text(body, encoding="utf-8")
        if complete or expired:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
