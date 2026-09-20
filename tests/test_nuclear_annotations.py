import numpy as np

from scripts.evaluate_nuclei import xml_instances


def test_polygon_rasterizer_uses_xy_without_transposition(tmp_path):
    path = tmp_path / "nuclei.xml"
    path.write_text(
        '<Annotations><Region><Vertices><Vertex X="6" Y="1"/><Vertex X="8" Y="1"/><Vertex X="8" Y="3"/><Vertex X="6" Y="3"/></Vertices></Region></Annotations>'
    )
    mask, overlap = xml_instances(path, (12, 14))
    expected = np.zeros((12, 14), dtype=np.int32)
    expected[1:4, 6:9] = 1
    assert np.array_equal(mask, expected)
    assert overlap == 0
