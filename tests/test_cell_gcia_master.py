"""Prevent synthetic fixtures from masquerading as BACH acquisition."""

import pytest

from examples.toy_demo.generate_synthetic_graphs import prepare_synthetic_cohort
from scripts.acquire_bach import prepare_bach_cohort
from scripts.run_cell_gcia import main


def test_bach_unavailable_fails_without_writing(tmp_path):
    with pytest.raises(NotImplementedError, match="No real BACH"):
        prepare_bach_cohort(tmp_path / "bach")
    assert not (tmp_path / "bach").exists()


def test_master_is_quarantined():
    with pytest.raises(RuntimeError, match="Quarantined"):
        main()


def test_synthetic_fixture_identifies_itself(tmp_path):
    summary = prepare_synthetic_cohort(tmp_path / "toy", num_samples_per_class=1)
    assert summary["total_cases"] == 4
    assert "NOT external validation" in summary["dataset_name"]
