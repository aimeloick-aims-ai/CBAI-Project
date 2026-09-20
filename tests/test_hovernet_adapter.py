import numpy as np

from src.segmentation.hovernet import infer_tiled


class IdentityCentralModel:
    @staticmethod
    def infer_batch(model, tile, *, device):
        return (tile.numpy()[:, 46:210, 46:210, :1],)

    def postproc(self, maps):
        return ({"predictions": maps[0][..., 0]},)


def test_tiled_inference_preserves_alignment_on_nonmultiple_dimensions():
    image = np.random.default_rng(5).integers(0, 256, (179, 331, 3), dtype=np.uint8)
    result = infer_tiled(IdentityCentralModel(), image)
    assert np.array_equal(result, image[..., 0])


def test_tiled_inference_preserves_tiny_image_borders():
    image = np.full((1, 2, 3), 128, dtype=np.uint8)
    assert np.array_equal(infer_tiled(IdentityCentralModel(), image), image[..., 0])
