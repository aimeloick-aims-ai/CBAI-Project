"""Nuclear instance contracts and an unvalidated classical integration baseline."""

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def validate_instances(mask, shape=None):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError("Instance map must be a nonnegative 2D integer array; zero is background")
    if shape is not None and mask.shape != tuple(shape):
        raise ValueError("Instance map and image must have identical pixel dimensions")
    return mask


def watershed_baseline(rgb, min_area=12, min_distance=3):
    """Return candidate nuclei, not clinically validated segmentation or cell types."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB")
    if min_area < 1 or min_distance < 1:
        raise ValueError("Area and peak distance must be positive")
    hematoxylin = rgb2hed(rgb)[..., 0]
    if np.ptp(hematoxylin) < 1e-8:
        return np.zeros(rgb.shape[:2], dtype=np.int32)
    foreground = remove_small_objects(hematoxylin > threshold_otsu(hematoxylin), min_size=min_area)
    distance = ndimage.distance_transform_edt(foreground)
    peaks = peak_local_max(
        distance, min_distance=min_distance, labels=foreground, exclude_border=False
    )
    markers = np.zeros(foreground.shape, dtype=np.int32)
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    # Ensure thin foreground components without a detected maximum still receive a marker.
    components = label(foreground)
    next_id = int(markers.max()) + 1
    for component in range(1, int(components.max()) + 1):
        region = components == component
        if not np.any(markers[region]):
            index = np.argmax(np.where(region, distance, -1))
            markers.flat[index] = next_id
            next_id += 1
    return watershed(-distance, markers, mask=foreground).astype(np.int32)


def instance_metrics(predicted, reference, iou_threshold=0.5):
    """One-to-one detection and panoptic metrics with a strict IoU threshold."""
    reference = validate_instances(reference)
    predicted = validate_instances(predicted, reference.shape)
    if not 0 < iou_threshold < 1:
        raise ValueError("IoU threshold must be between zero and one")
    pids, pflat = np.unique(predicted.ravel(), return_inverse=True)
    rids, rflat = np.unique(reference.ravel(), return_inverse=True)
    if len(pids) * len(rids) > 25_000_000:
        raise ValueError(
            "Too many instances for dense matching; evaluate smaller annotated regions"
        )
    # Insert background even when the entire image is foreground.
    pn, rn = pids != 0, rids != 0
    counts = np.bincount(pflat * len(rids) + rflat, minlength=len(pids) * len(rids)).reshape(
        len(pids), len(rids)
    )
    intersection = counts[pn][:, rn].astype(float)
    union = counts.sum(1)[pn, None] + counts.sum(0)[None, rn] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    eligible = iou > iou_threshold
    if iou.size:
        # Maximize match count first, then IoU; low-IoU matches are never counted.
        a, b = linear_sum_assignment(-(eligible * (min(iou.shape) + 1) + iou * eligible))
        accepted = eligible[a, b]
        tp = int(accepted.sum())
        matched_iou = float(iou[a[accepted], b[accepted]].sum())
    else:
        tp, matched_iou = 0, 0.0
    fp, fn = int(pn.sum()) - tp, int(rn.sum()) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    fg_p, fg_r = predicted > 0, reference > 0
    fg_denom = int(fg_p.sum() + fg_r.sum())
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "detection_f1": tp / denom if denom else None,
        "segmentation_quality": matched_iou / tp if tp else None,
        "panoptic_quality": matched_iou / denom if denom else None,
        "foreground_dice": float(2 * np.sum(fg_p & fg_r) / fg_denom) if fg_denom else None,
        "iou_threshold_strict": iou_threshold,
    }
