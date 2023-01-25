"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy
import pooch
from tifffile import imread

from . import __version__  # The version string of your project

BRIAN = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("napari_brainbow_diagnose"),
    # The remote data is on Github
    base_url="https://github.com/LaboratoryOpticsBiosciences/"
    "napari-brainbow-diagnose/raw/{version}/data/",
    version=f"v{__version__}",
    # If this is a development version,
    # get the data from the "add_examples" branch
    version_dev="main",
    registry={
        "chroms_cortex_sample.tif": "sha256:"
        "eb6dd1e670f214ffaa1beb0a57552fb269b07c456e06fda3af3ec5fe69d8e531",
    },
)


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


def fetch_chroms_data():
    """
    Load the chroms data sample as a tifffile
    """
    fname = BRIAN.fetch("chroms_cortex_sample.tif")
    data = imread(fname)
    return data


def load_chroms_data_sample():
    """Load chroms data sample from the data folder"""
    chroms_data = fetch_chroms_data()
    return [
        (
            chroms_data[:, 0, :, :],
            {"colormap": "red", "blending": "additive", "name": "red channel"},
        ),
        (
            chroms_data[:, 1, :, :],
            {
                "colormap": "green",
                "blending": "additive",
                "name": "green channel",
            },
        ),
        (
            chroms_data[:, 2, :, :],
            {
                "colormap": "blue",
                "blending": "additive",
                "name": "blue channel",
            },
        ),
    ]
