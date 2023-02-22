from typing import TYPE_CHECKING

import numpy as np
from magicgui import magic_factory

from ._utils_io import get_brainbow_image_from_layers

if TYPE_CHECKING:
    import napari

FLOAT_ROUND_DECIMAL = 2  # float number is too long to display


def get_brainbow_selection(brainbow, selection_mask):
    """Returns a brainbow image with only the selected region.

    Parameters
    ----------
    brainbow : np.ndarray
        Brainbow image with shape (C, Z, Y, X)
    selection_mask : np.ndarray
        Selection mask with shape (Z, Y, X)
        All the selected voxels should be True or any integer > 0.

    Returns
    -------
    np.ndarray
        Array with shape (N, C) with N the number of selected voxels.
    """
    selection = brainbow[:, selection_mask.astype(bool)]
    return selection.T


def get_mean_brainbow_selection(voxels):
    """Returns the mean of the selected voxels.

    Parameters
    ----------
    voxels : np.ndarray
        Array with shape (N, C) with N the number of selected voxels.

    Returns
    -------
    np.ndarray
        Array with shape (C,) with the mean of the selected voxels.
    """
    return np.mean(voxels, axis=0)


def get_median_brainbow_selection(voxels):
    """Returns the median of the selected voxels.

    Parameters
    ----------
    voxels : np.ndarray
        Array with shape (N, C) with N the number of selected voxels.

    Returns
    -------
    np.ndarray
        Array with shape (C,) with the median of the selected voxels.
    """
    return np.median(voxels, axis=0)


def get_selection_metrics(red_layer, green_layer, blue_layer, selection_layer):
    """Returns the mean and median of the selected voxels.

    Parameters
    ----------
    red_layer : napari.layers.Image
        Red channel layer.
    green_layer : napari.layers.Image
        Green channel layer.
    blue_layer : napari.layers.Image
        Blue channel layer.
    selection_layer : napari.layers.Labels
        Selection layer.

    Returns
    -------
    np.ndarray
        Array with shape (C,) with the mean of the selected voxels.
    np.ndarray
        Array with shape (C,) with the median of the selected voxels.
    """
    brainbow = get_brainbow_image_from_layers(
        red_layer, green_layer, blue_layer
    )
    selection = get_brainbow_selection(brainbow, selection_layer.data)

    mean = get_mean_brainbow_selection(selection)
    median = get_median_brainbow_selection(selection)

    return mean, median


@magic_factory(call_button="run")
def tooltip_pointer_widget(
    red_layer: "napari.layers.Image",
    green_layer: "napari.layers.Image",
    blue_layer: "napari.layers.Image",
    selection_layer: "napari.layers.Labels",
):
    """
    User interface to show a tooltip on the footprint selected.
    """

    mean, median = get_selection_metrics(
        red_layer, green_layer, blue_layer, selection_layer
    )

    mean_r, mean_g, mean_b = mean
    median_r, median_g, median_b = median

    table = {
        "mean_r": np.array([mean_r]),
        "mean_g": np.array([mean_g]),
        "mean_b": np.array([mean_b]),
        "median_r": np.array([median_r]),
        "median_g": np.array([median_g]),
        "median_b": np.array([median_b]),
    }

    # (tmp) store the raw `table` output from regionprop
    selection_layer._metadata = table
    # Round floating numbers, because float64 is too long too display and harms
    # readability
    for k, v in table.items():
        if v.dtype == "float":
            table[k] = v.round(FLOAT_ROUND_DECIMAL)

    # Append dummy 0 label
    for k in table.keys():
        table[k] = np.insert(table[k], 0, 0)
    # Set `features` property. It will be copied to `properties` property too
    # and displayed in the tooltip.
    selection_layer.features = table
