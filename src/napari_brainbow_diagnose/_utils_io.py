import numpy as np


def empty_brainbow_image():
    """Returns an empty brainbow image. With shape (3, 1, 1, 1)
    corresponding to (C, Z, Y, X)"""
    return np.random.random((3, 2, 2, 2))


def get_brainbow_image_from_layers(
    red_layer=None, green_layer=None, blue_layer=None
):
    if red_layer is None or green_layer is None or blue_layer is None:
        return empty_brainbow_image()
    else:
        return np.array(
            [
                red_layer.data,
                green_layer.data,
                blue_layer.data,
            ]
        )
