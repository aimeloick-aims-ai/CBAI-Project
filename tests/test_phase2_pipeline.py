import pytest
import torch

from src.data.manifest import RoiRecord
from src.data.splits import assert_no_patient_overlap
from src.graphs.build import knn_edges, make_graph
from src.interventions.feature_edits import clamp_feature_shift


def test_patient_overlap_is_rejected(tmp_path):
    records = [
        RoiRecord("D", "p1", "a", "N", "train", tmp_path / "a.png", 1, 1),
        RoiRecord("D", "p1", "b", "IC", "val", tmp_path / "b.png", 1, 1),
    ]
    with pytest.raises(ValueError, match="Patient overlap"):
        assert_no_patient_overlap(records)


def test_knn_edges_are_bidirectional():
    coords = torch.tensor([[0.0], [1.0], [3.0]]).numpy()
    edge_index = knn_edges(coords, k=1)
    edges = {tuple(edge) for edge in edge_index.t().tolist()}
    assert (0, 1) in edges
    assert (1, 0) in edges


def test_make_graph_and_feature_edit():
    features = torch.rand(4, 6).numpy()
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).numpy()
    graph = make_graph(features, coords, y=2, k=2)
    edited = clamp_feature_shift(graph.x, feature=0, delta=10)
    assert graph.x.shape == (4, 6)
    assert edited[:, 0].max() <= 1
