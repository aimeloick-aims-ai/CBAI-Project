"""Integration unit tests for master cell GCIA pipeline and BACH acquisition."""

from pathlib import Path

from scripts.acquire_bach import prepare_bach_cohort


def test_prepare_bach_cohort(tmp_path: Path):
    summary = prepare_bach_cohort(output_dir=tmp_path / "bach", num_samples_per_class=2)

    assert summary["total_cases"] == 8
    assert summary["num_classes"] == 4
    assert (tmp_path / "bach/summary.json").exists()
    assert (tmp_path / "bach/cell_graphs/bach_normal_01.pt").exists()
