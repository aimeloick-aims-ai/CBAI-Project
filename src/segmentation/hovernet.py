"""Native-resolution HoVer-Net fast inference, with global instance postprocessing."""

import numpy as np
import torch


def infer_tiled(model, rgb, *, device="cpu", output_size=164):
    """Stitch central output maps, then postprocess once to avoid duplicate tile IDs.

    No physical-scale resampling is implicit. Tile-boundary quality must be evaluated.
    Input pixels are RGB uint8, matching HoVer-Net's internal /255 normalization.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 RGB")
    if output_size < 1 or output_size > 256 or (256 - output_size) % 2:
        raise ValueError("Invalid central output size")
    height, width = rgb.shape[:2]
    halo = (256 - output_size) // 2
    rows, cols = (height + output_size - 1) // output_size, (width + output_size - 1) // output_size
    padded = np.pad(
        rgb,
        (
            (halo, halo + rows * output_size - height),
            (halo, halo + cols * output_size - width),
            (0, 0),
        ),
        mode="edge",
    )
    assembled = None
    for y in range(rows):
        for x in range(cols):
            top, left = y * output_size, x * output_size
            tile = torch.from_numpy(padded[top : top + 256, left : left + 256].copy())[None]
            maps = model.infer_batch(model, tile, device=device)
            if any(v.shape[1:3] != (output_size, output_size) for v in maps):
                raise ValueError(
                    "Unexpected model output dimensions; check fast model configuration"
                )
            if assembled is None:
                assembled = [
                    np.zeros(
                        (rows * output_size, cols * output_size, v.shape[-1]), dtype=np.float32
                    )
                    for v in maps
                ]
            for dest, value in zip(assembled, maps, strict=True):
                dest[top : top + output_size, left : left + output_size] = value[0]
    result = model.postproc([v[:height, :width] for v in assembled])
    instances = np.asarray(result[0]["predictions"])
    if instances.shape != (height, width):
        raise ValueError("Postprocessed instance map is misaligned")
    return instances.astype(np.int32)


def segment_pretrained(rgb, weights, *, device="cpu"):
    from tiatoolbox.models.architecture import get_pretrained_model

    model, _ = get_pretrained_model("hovernet_fast-pannuke", pretrained_weights=weights)
    model.to(device).eval().requires_grad_(False)
    return infer_tiled(model, rgb, device=device)
