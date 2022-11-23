"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy


def make_rgb_cube_data():
    """Generates an rgb cube"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    rgb_size = 256
    step = numpy.linspace(0, 255, rgb_size)
    rgb_cube = numpy.meshgrid(step, step, step)
    return [
        (rgb_cube[0], {"colormap": "red", "blending": "additive"}),
        (rgb_cube[1], {"colormap": "green", "blending": "additive"}),
        (rgb_cube[2], {"colormap": "blue", "blending": "additive"}),
    ]
