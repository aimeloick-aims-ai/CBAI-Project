import numpy as np
import torch

from experiments.full_study.synthetic import StudyModel, intervene, make_split


def test_factorial_generator_and_selective_edits():
    data = make_split(1, 3, "unit")
    assert len({tuple(g.concepts.flatten().tolist()) for g in data}) == 16
    for g in data:
        for c in range(4):
            edited = intervene(g, c)
            delta = (edited.concepts != g.concepts).flatten()
            assert int(delta.sum()) == 1 and delta[c]
            assert int(edited.y != g.y) == int(c in (0, 2))
            if c == 2:
                assert torch.equal(g.x, edited.x)
                assert not torch.equal(g.edge_index, edited.edge_index)
                assert torch.equal(
                    torch.bincount(g.edge_index[0], minlength=12),
                    torch.bincount(edited.edge_index[0], minlength=12),
                )
                assert not (edited.edge_index[0] == edited.edge_index[1]).any()
                assert len(set(map(tuple, edited.edge_index.T.tolist()))) == 24
            if c == 3:
                assert torch.equal(g.x, edited.x) and torch.equal(g.edge_index, edited.edge_index)


def test_deepsets_invariant_to_relational_intervention():
    torch.manual_seed(3)
    g = make_split(1, 8, "unit")[0]
    altered = intervene(g, 2)
    model = StudyModel("DeepSets").eval()
    batch = torch.zeros(12, dtype=torch.long)
    assert torch.equal(model(g.x, g.edge_index, batch), model(altered.x, altered.edge_index, batch))
    assert len(model.representations(g.x, g.edge_index, batch)) == 4
    assert np.all(np.bincount(g.edge_index[0].numpy(), minlength=12) == 2)
