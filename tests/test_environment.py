"""Integration checks on explicit software fixtures, not generated clinical data."""

from pathlib import Path

import pytest
import yaml

from scripts.check_environment import check_gnn, check_slide

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("architecture", ["GCN", "GraphSAGE", "GATv2"])
def test_graph_forward_and_input_gradient(architecture):
    config = yaml.safe_load((ROOT / "configs/smoke_test.yaml").read_text(encoding="utf-8"))
    assert check_gnn(architecture, config)["status"] == "pass"


def test_openslide_decodes_tiled_pixels():
    assert check_slide()["status"] == "pass"
