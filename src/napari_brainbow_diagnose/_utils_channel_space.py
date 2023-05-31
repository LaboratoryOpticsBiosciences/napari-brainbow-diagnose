import math
import ternary
from ternary.helpers import simplex_iterator

import numpy as np
from numpy import tan, sqrt, pi, sin, cos
import skimage.color as skc

from ._utils_shape import get_2D_wheel_coordinate, image_to_flat_data_points

# this files allows conversion between different color spaces


def rgb_to_hsv(rgb: np.ndarray, channel_axis: int = -1) -> np.ndarray:
    """Convert RGB to HSV.

    Parameters
    ----------
    rgb : array-like
        RGB values.
    channel_axis : int, optional
        The axis of the color channels, by default -1

    Returns
    -------
    hsv : array-like
        HSV values.
    """
    return skc.rgb2hsv(rgb, channel_axis=channel_axis)


# TODO ajouter un paramètre pour choisir l'axe des canaux
def get_channels_ranges(a: np.ndarray) -> np.ndarray:
    """
    Returns an array containing the minimum and maximum values of
    each channel in the input array.

    Parameters
    ----------
    a : np.ndarray
        The input array of n points with c channels (n*c) as shape.

    Returns
    -------
    np.ndarray
        The (channel*2) array containing the minimum and maximum values
        of each channel.
    """
    ranges = np.array([np.amin(a, axis=0), np.amax(a, axis=0)]).T
    return ranges


def image_mask_of_wheel_selection(
    image: np.ndarray, wheel_selection: np.ndarray
):
    """
    Returns a boolean mask of the wheel selection.

    Parameters
    ----------
    image : np.ndarray
        The image to select from.
        Shape: (c, z, y, x) or (c, y, x)
    wheel_selection : np.ndarray
        The wheel selection to apply.

    Returns
    -------
    np.ndarray
        The boolean mask of the wheel selection. Same shape as the image.

    """

    hsv = rgb_to_hsv(image, channel_axis=0)
    hs = hsv[0:2]

    vector_points = image_to_flat_data_points(hs, 0)

    wheel_positions = get_2D_wheel_coordinate(
        vector_points[..., 0], vector_points[..., 1]
    )
    wheel_positions = wheel_positions * (len(wheel_selection) - 1)
    wheel_positions = wheel_positions.astype(int)

    selected_points = wheel_selection[(wheel_positions[0], wheel_positions[1])]

    selection_mask = selected_points.reshape(image.shape[1:])
    selection_mask = selection_mask.astype(bool)

    return selection_mask


def image_selection_to_wheel_selection(image, selection, wheel_diameter):
    """Create a wheel of a given diameter and selection.

    Parameters
    ----------
    image : np.ndarray
        The image to be converted to a wheel.
    selection : np.ndarray
        The selection in the image to be converted to a wheel.
    wheel_diameter : int
        The diameter of the wheel.

    Returns
    -------
    np.ndarray
        The wheel with each bin containing the density of the
        corresponding selection in the image.
    """

    # mask image by selection
    masked_image = image[:, selection == True].T

    # get hue and saturation
    hs = rgb_to_hsv(masked_image, channel_axis=1)[..., :2]

    # flatten hue and saturation
    hs = image_to_flat_data_points(hs, 0)

    # convert angles and radii to wheel coordinates
    pos = get_2D_wheel_coordinate(hs[0], hs[1]) * (wheel_diameter - 1)
    pos = pos.astype(int)

    # create empty wheel figure
    wheel = np.zeros((wheel_diameter, wheel_diameter))

    # populate wheel by summing the density of each bin
    # at the corresponding coordinates
    np.add.at(wheel, (pos[0], pos[1]), 1)

    return wheel


def hue_saturation_color_wheel(size: int = 100) -> np.ndarray:
    """
    Returns a numpy array representing a color wheel of hue and
    saturation values.

    Parameters
    ----------
    size : int, optional
        The size of the color wheel, defaults to 100.

    Returns
    -------
    np.ndarray
        The color wheel of hue and saturation values.
    """
    wheel = np.zeros((size, size, 3)).astype("int")
    radius = size / 2.0
    cx, cy = size / 2, size / 2

    for x in range(size):
        for y in range(size):
            rx = x - cx
            ry = y - cy
            s = (rx**2.0 + ry**2.0) ** 0.5 / radius
            h = ((math.atan2(ry, rx) / math.pi) + 1.0) / 2.0
            if s <= 1:
                rgb = skc.hsv2rgb([h, s, 1]) * 255
                wheel[size - 1 - x, size - 1 - y] = rgb
    return wheel

def maxwell_hue_space(size: int = 100):
    """
    Returns a dictionary representing a barycentric color triangle

    Parameters
    ----------
    size : int, optional
        The size of the color triangle, defaults to 100.

    Returns
    -------
    python dictionary
        The color triangle of rgb values in maxwell's hue space.
    """

    def color_point(x, y, z, scale = size):
        r = x / float(scale)
        g = y / float(scale)
        b = z / float(scale)
        return (r, g, b, 1.)

    d = dict()
    for (i, j, k) in simplex_iterator(size):
        d[(i, j, k)] = color_point(i, j, k, size)
    return d

def maxwell_hue_empty(size: int = 100):
    d = dict()
    for (i, j, k) in simplex_iterator(size):
        d[(i, j, k)] = 2
    return d

def spherical_coordinates_color_wheel(size: int = 100):
    """
    Returns a numpy array representing a color wheel of polar and
    azimuth angle values.

    Parameters
    ----------
    size : int, optional
        The size of the color wheel, defaults to 100.

    Returns
    -------
    np.ndarray
        The color wheel of polar and azimuth angle values.
    """

    wheel = np.zeros((size,size,3)).astype("float")
    for i in range(size):
        for j in range(size):
            theta, phi = pi/2*float(i)/float(size), pi/2*float(j)/float(size)
            r = max(0, min(1, sin(theta) * cos(phi)))
            g = max(0, min(1, sin(theta) * sin(phi)))
            b = max(0, min(1, cos(theta)))
            wheel[-i-1,j] = np.array([r, g, b])
    return wheel