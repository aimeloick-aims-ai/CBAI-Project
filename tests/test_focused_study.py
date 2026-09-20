import numpy as np

from scripts.run_focused_study import mask_fractions, patch_features, substitutions


def test_excluded_pixels_do_not_dilute_tissue_fraction():
    fraction, coverage = mask_fractions(np.array([[1, 1], [0, 7]]))
    assert coverage == 0.5
    assert fraction[1] == 1
    assert fraction.sum() == 1


def test_patch_replacement_equals_descriptor_replacement():
    rng = np.random.default_rng(4)
    image = rng.random((128, 256, 3)).astype("float32")
    donor = image[:, 128:].copy()
    original = image[:, :128].copy()
    image[:, :128] = donor
    np.testing.assert_array_equal(patch_features(image[:, :128]), patch_features(donor))
    assert not np.allclose(patch_features(original), patch_features(donor))


def test_selection_requires_both_tissues_and_separate_control():
    frac = np.zeros((3, 22))
    frac[:2, 1] = 1
    frac[2, 2] = 1
    patient = {
        "valid": np.ones(3, dtype=bool),
        "fractions": frac,
        "patches": [np.full((4, 4, 3), v) for v in (0.5, 0.4, 0.6)],
    }
    config = {
        "pure_tissue_threshold": 0.8,
        "maximum_recipients_per_patient": 1,
        "amplitude_match_relative_tolerance": 0.2,
        "non_target_fraction_tolerance": 0.05,
    }
    row = substitutions(patient, config)[0]
    assert row["recipient"] != row["control_donor"]
    assert row["target_donor"] == 2
    assert row["numerical_checks_pass"]
    patient["valid"][2] = False
    assert substitutions(patient, config) == []
