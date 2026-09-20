"""Numerical checks on explicitly synthetic annotation arrays."""

import numpy as np
import pytest

from scripts.audit_bcss import area_fractions


def test_ignored_pixels_do_not_bias_area_denominator():
    mask = np.array([[0, 7, 1], [1, 2, 2]], dtype=np.uint8)
    assert area_fractions(mask, [0, 1, 2, 7], [0, 7]) == {1: 0.5, 2: 0.5}


def test_unknown_labels_are_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        area_fractions(np.array([[255]]), [0, 1], [0])


def test_empty_annotation_is_not_reported_as_zero_tumor():
    with pytest.raises(ValueError, match="No evaluable"):
        area_fractions(np.zeros((2, 2), dtype=np.uint8), [0, 1], [0])
