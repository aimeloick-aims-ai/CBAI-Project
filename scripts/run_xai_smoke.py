"""Exercise the XAI audit API on deterministic synthetic representations.

This script proves the software wiring only.  It does not use BRACS labels as
concept annotations and cannot report a scientific explanation result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.concepts.probes import evaluate_binary_probe, fit_binary_probe
from src.evaluation.audit import ConceptAudit
from src.interventions.feature_edits import prediction_shift


def main() -> None:
    torch.manual_seed(7)
    embeddings = torch.randn(24, 4)
    labels = (embeddings[:, 0] > 0).long()
    probe = fit_binary_probe(embeddings[:16], labels[:16])
    probe_result = evaluate_binary_probe(probe, embeddings[16:], labels[16:])
    before = torch.tensor([[1.0, 0.0], [0.5, 0.0]])
    after = torch.tensor([[1.0, 0.6], [0.5, 0.4]])
    audit = ConceptAudit(
        concept="synthetic_fixture",
        probe=probe_result,
        target_probability_shift=prediction_shift(before, after, target_class=1),
        control_probability_shifts=[0.0, 0.01],
        validity_passed=False,
    )
    report = {
        "scope": "engineering_smoke_only_not_scientific_validation",
        "independent_concept_annotations": False,
        "audit": audit.summary(),
        "interpretation": "Synthetic fixture validates APIs; no biological or model-use claim.",
    }
    output = ROOT / "reports" / "xai_smoke.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
