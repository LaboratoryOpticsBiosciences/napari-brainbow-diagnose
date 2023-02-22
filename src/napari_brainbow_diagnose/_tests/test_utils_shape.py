import numpy as np

from napari_brainbow_diagnose._utils_shape import (
    flat_data_points_to_image,
    image_to_flat_data_points,
)


def test_image_to_flat_data_points():
    a = np.random.rand(10, 10, 3)
    channel_axis = 2
    b = image_to_flat_data_points(a, channel_axis)
    c = flat_data_points_to_image(b, a.shape, channel_axis)
    assert (a == c).all()
