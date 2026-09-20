import numpy as np
import torch
from torch_geometric.data import Batch

from scripts.run_extended_xai import saliency_order, tissue_targets
from scripts.run_real_xai import LayerGNN, graph


def test_tissue_fraction_ignores_unannotated_pixels():
    mask = np.ones((256, 256), dtype=np.uint8)
    mask[:16, :32] = 0
    fraction, valid = tissue_targets(mask, 1)
    assert fraction[0] == 1
    assert valid[0]
    mask[:17, :32] = 7
    _, valid = tissue_targets(mask, 1)
    assert not valid[0]


def test_saliency_does_not_change_frozen_weights_or_graph():
    torch.manual_seed(3)
    model = LayerGNN("GCN", 7).eval().requires_grad_(False)
    g = Batch.from_data_list([graph(np.random.default_rng(1).random((256, 256, 3)))])
    original = g.x.clone()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    order = saliency_order(model, g, 0)
    assert sorted(order.tolist()) == list(range(64))
    assert torch.equal(original, g.x)
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
