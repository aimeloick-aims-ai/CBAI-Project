import numpy as np
import torch

from scripts.run_real_xai import features, graph, ridge_probe, variants


def test_integer_coordinates_and_unique_graph_edges():
    g = graph(np.zeros((256, 256, 3), dtype=np.float32))
    assert g.x.shape == (64, 6)
    assert g.edge_index.shape == torch.unique(g.edge_index, dim=1).shape
    assert torch.isfinite(g.x).all()


def test_ridge_generalizes_known_linear_signal():
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=(100, 3)), rng.normal(size=(30, 3))
    assert ridge_probe(a, a[:, 0], b, b[:, 0]) < 0.01


def test_controls_match_budget_and_do_not_mutate_input():
    rgb = np.random.default_rng(1).random((256, 256, 3)).astype("float32")
    original = rgb.copy()
    mask = np.full((256, 256), 2, dtype=np.uint8)
    mask[:64] = 1
    rows = list(variants(rgb, mask, 11))
    for level in ("pixels", "attributes", "edges"):
        selected = [r for r in rows if r[0] == level]
        assert len(selected) == 11
        assert len({r[2] for r in selected}) == 1
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_allclose(features(rgb), graph(rgb).x.numpy())
