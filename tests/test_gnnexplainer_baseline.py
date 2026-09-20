import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer

from scripts.run_focused_study import FractionGCN
from scripts.run_gnnexplainer_baseline import RegionalModel, deletion_prediction
from src.graphs.build import make_graph


def test_regional_wrapper_and_noop_deletion():
    torch.manual_seed(1)
    g = make_graph(np.ones((5, 3)), np.arange(5).reshape(-1, 1), 0)
    valid = torch.tensor([True, False, True, True, True])
    core = FractionGCN(3).eval().requires_grad_(False)
    model = RegionalModel(core)
    expected = core(g)[valid].mean()
    assert torch.allclose(model(g.x, g.edge_index, valid)[0], expected)
    before = g.x.clone()
    assert abs(deletion_prediction(model, g.x, g.edge_index, valid, []) - expected) < 1e-7
    assert torch.equal(before, g.x)


def test_explainer_preserves_weights_and_removes_temporary_masks():
    torch.manual_seed(11)
    g = make_graph(np.random.default_rng(11).normal(size=(5, 3)), np.arange(5).reshape(-1, 1), 0)
    valid = torch.ones(5, dtype=torch.bool)
    core = FractionGCN(3).eval().requires_grad_(False)
    model = RegionalModel(core)
    baseline = model(g.x, g.edge_index, valid).detach().clone()
    weights = {k: v.clone() for k, v in core.state_dict().items()}
    explain = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=5),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config={"mode": "regression", "task_level": "graph", "return_type": "raw"},
    )
    result = explain(g.x, g.edge_index, valid=valid)
    assert result.node_mask.shape == (5, 1)
    assert torch.isfinite(result.edge_mask).all()
    assert torch.allclose(baseline, model(g.x, g.edge_index, valid))
    assert all(torch.equal(weights[k], v) for k, v in core.state_dict().items())
