"""Pathologist Review & Clinical Validation Framework for GCIA.

Generates structured clinical evaluation sheets, visual overlays of cell graphs
(before vs after intervention), and standardized JSON forms for pathologist scoring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class ClinicalScoringForm:
    case_id: str
    patient_id: str
    target_diagnosis: str
    fidelity_score: float
    histological_plausibility_score: int  # 1 (unrealistic artifact) to 5 (fully plausible)
    cell_morphology_alignment: int  # 1 to 5
    counterfactual_validity: int  # 1 to 5
    pathologist_comments: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class PathologistReviewGenerator:
    """Exports structured visual review dossiers for anatomopathologist evaluation."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_review_sheet(
        self,
        case_id: str,
        patient_id: str,
        predicted_class: str,
        confidence: float,
        node_positions: torch.Tensor,
        node_importance: torch.Tensor,
        top_k_indices: list[int],
        fidelity_score: float,
    ) -> Path:
        """Renders a standalone interactive HTML review sheet for clinical scoring."""
        pos_np = node_positions.detach().cpu().numpy()
        imp_np = node_importance.detach().cpu().numpy()

        # Build SVG visualization of cell graph
        svg_circles = []
        min_x, max_x = pos_np[:, 0].min(), pos_np[:, 0].max()
        min_y, max_y = pos_np[:, 1].min(), pos_np[:, 1].max()
        width, height = 500, 500

        # Normalize coordinates to 500x500 viewport
        range_x = max(max_x - min_x, 1e-5)
        range_y = max(max_y - min_y, 1e-5)

        for idx, (px, py) in enumerate(pos_np):
            cx = 40 + ((px - min_x) / range_x) * 420
            cy = 40 + ((py - min_y) / range_y) * 420
            imp = float(imp_np[idx])
            is_top = idx in top_k_indices

            color = (
                f"rgba(220, 38, 38, {0.3 + 0.7 * imp})"
                if is_top
                else f"rgba(59, 130, 246, {0.3 + 0.7 * imp})"
            )
            r = 8 if is_top else 5
            stroke = "#991b1b" if is_top else "#1e40af"
            stroke_width = 2 if is_top else 1

            svg_circles.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"><title>Cell {idx}: importance {imp:.2f}</title></circle>'
            )

        svg_content = f"""<svg width="{width}" height="{height}" style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px;">
            {"".join(svg_circles)}
        </svg>"""

        html_content = rf"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Formulaire d'Évaluation Pathologique GCIA - {case_id}</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 20px; color: #1e293b; background: #f1f5f9; }}
        .card {{ background: white; border-radius: 12px; padding: 24px; max-width: 800px; margin: 0 auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        h1 {{ font-size: 1.5rem; color: #0f172a; margin-top: 0; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.875rem; background: #e0f2fe; color: #0369a1; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }}
        label {{ font-weight: 600; display: block; margin-top: 12px; font-size: 0.9rem; }}
        input[type="number"], textarea {{ width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; margin-top: 4px; box-sizing: border-box; }}
        .score-info {{ font-size: 0.8rem; color: #64748b; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>GCIA Audit - Validation Clinique Histologique</h1>
        <p><strong>Cas ID :</strong> {case_id} &nbsp;|&nbsp; <strong>Patient :</strong> {patient_id}</p>
        <p><span class="badge">Diagnostic prédit : {predicted_class} (Confiance : {confidence:.1%})</span></p>
        <p><strong>Score de Fidélité GCIA ($\Delta P$) :</strong> {fidelity_score:.3f}</p>
        
        <div class="grid">
            <div>
                <h3>Emplacement des Noyaux & Subgraph Explainer</h3>
                {svg_content}
                <p class="score-info">🔴 Noyaux cibles ablations (Top K) | 🔵 Noyaux voisins</p>
            </div>
            <div>
                <h3>Formulaire de Validation Clinique</h3>
                <form id="review-form">
                    <label>Plausibilité Histologique (1 à 5)</label>
                    <input type="number" name="histological_plausibility" min="1" max="5" value="4">
                    <span class="score-info">1 = Artefact non plausible, 5 = Altération histologique réaliste</span>
                    
                    <label>Alignement Morphologique (1 à 5)</label>
                    <input type="number" name="morphology_alignment" min="1" max="5" value="4">
                    <span class="score-info">1 = Incohérent avec les critères atypiques, 5 = Parfaitement aligné</span>
                    
                    <label>Validité du Contre-factuel (1 à 5)</label>
                    <input type="number" name="counterfactual_validity" min="1" max="5" value="4">
                    
                    <label>Commentaires du Pathologiste</label>
                    <textarea name="comments" rows="4" placeholder="Observations histologiques sur l'ablation des noyaux..."></textarea>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
"""
        out_path = self.output_dir / f"review_{case_id}.html"
        out_path.write_text(html_content, encoding="utf-8")

        # Save template JSON scoring form
        template_form = ClinicalScoringForm(
            case_id=case_id,
            patient_id=patient_id,
            target_diagnosis=predicted_class,
            fidelity_score=fidelity_score,
            histological_plausibility_score=4,
            cell_morphology_alignment=4,
            counterfactual_validity=4,
            pathologist_comments="Ablation des noyaux atypiques réduit la plausibilité tumorale.",
        )
        (self.output_dir / f"scoring_{case_id}.json").write_text(
            template_form.to_json(), encoding="utf-8"
        )

        return out_path
