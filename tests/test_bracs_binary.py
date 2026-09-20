import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from scripts.run_bracs_binary import BinaryModel, check_manifest, metrics
from src.graphs.build import make_graph


def test_manifest_rejects_shared_patient():
    rows = []
    for split, count in [("train", 10), ("validation", 5), ("internal_test", 5)]:
        for label in ("N", "IC"):
            for i in range(count):
                rows.append(
                    {
                        "patient_id": f"{split}_{label}_{i}",
                        "proposed_split": split,
                        "roi_label": label,
                    }
                )
    check_manifest(rows)
    rows[-1]["patient_id"] = rows[0]["patient_id"]
    with pytest.raises(ValueError, match="distinct"):
        check_manifest(rows)


def test_metrics_fixed_threshold_and_confusion():
    result = metrics([0, 0, 1, 1], [0.2, 0.6, 0.7, 0.8])
    assert result["balanced_accuracy"] == 0.75
    assert result["confusion_matrix_N_IC"] == [[1, 1], [0, 2]]


def test_models_produce_one_logit_per_image_and_gradients():
    rng = np.random.default_rng(1)
    g = make_graph(rng.normal(size=(5, 6)), np.arange(5).reshape(-1, 1), 0)
    batch = Batch.from_data_list([g, g.clone()])
    for kind in ("mean_mlp", "gcn"):
        model = BinaryModel(kind, 6)
        logits = model(batch)
        assert logits.shape == (2,)
        logits.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
