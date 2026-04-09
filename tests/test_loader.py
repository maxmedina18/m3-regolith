import numpy as np

from lunar_m3.dev.synthetic import generate_synthetic_cube


def test_synthetic_cube_shapes() -> None:
    cube = generate_synthetic_cube(rows=5, cols=7, bands=10, seed=1)
    assert cube.data.shape == (5, 7, 10)
    assert cube.wavelengths.shape == (10,)
    assert hasattr(cube, "synthetic_labels")
    labels = getattr(cube, "synthetic_labels")
    assert labels.shape == (5, 7)
    assert np.issubdtype(labels.dtype, np.integer)
