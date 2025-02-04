import math

import numpy as np
import skimage.color as skc
from numpy import cos, pi, sin, sqrt
from ternary.helpers import simplex_iterator

from ._utils_shape import image_to_flat_data_points

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


def get_2D_wheel_coordinate(angles: np.ndarray, radii: np.ndarray):
    """Create a wheel of a given radius and size.

    Parameters
    ----------
    angles : np.ndarray
        List of angles between 0 and 1.
    radii : np.ndarray
        List of radii. 0 is in the center, 1 is on the edge.

    Returns
    -------
    np.ndarray
        The coordinates of each point on the 2D wheel.
        The coordinates are in the range [0, 1] for both axes.
    """
    angles = np.array(angles) * 2 * np.pi
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    pos = (1 / 2) * np.array([1 + x, 1 + y])
    return pos


def rgb_to_maxwell_triangle(r, g, b):
    """
    Given coordinates in rgb color space, returns x and y coordinates
    of the Maxwell triangle.
    """
    s = r + g + b
    r = r / s
    g = g / s
    b = b / s
    x = (r - b) / np.sqrt(3)
    y = g
    return x, y


def rgb_to_spherical(r, g, b):
    radius = np.linalg.norm([r, g, b])
    theta = np.arctan2(sqrt(r**2 + g**2), b)
    phi = np.arctan2(g, r)

    # convert rad to degree
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)
    return radius, theta, phi


def spherical_to_rgb(radius, theta, phi):
    r = radius * np.sin(theta) * np.cos(phi)
    g = radius * np.sin(theta) * np.sin(phi)
    b = radius * np.cos(theta)
    return r, g, b


def calculate_brightness(r, g, b):
    """
    Given standardized values (from 0 to 1) of rgb return brightness
    """
    return (1 / 2) * (
        np.maximum(np.maximum(r, g), b) + np.minimum(np.minimum(r, g), b)
    )


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
    image: np.ndarray, wheel_selection: np.ndarray, value_threshold: float
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
    value_threshold : float
        The value threshold to apply to the image voxel.
        Only voxels with a value above the threshold will be selected.

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

    print(selection_mask.shape, (hsv[2] < value_threshold).shape)
    selection_mask[hsv[2] < value_threshold] = False

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
    masked_image = image[:, selection].T

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
    dict
        The color triangle of rgb values in maxwell's hue space.
    """

    def color_point(x, y, z, scale=size):
        r = x / float(scale)
        g = y / float(scale)
        b = z / float(scale)
        return (r, g, b, 1.0)

    d = dict()
    for (i, j, k) in simplex_iterator(size):
        d[(i, j, k)] = color_point(i, j, k, size)
    return d


def maxwell_hue_empty(size: int = 100):
    """
    Returns an empty maxwell_hue_space()
    Note: every density value is defaulted to 1 for eventually passing
    a log-norm on the densities
    """

    d = dict()
    for (i, j, k) in simplex_iterator(size):
        d[(i, j, k)] = 1
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

    wheel = np.zeros((size, size, 3)).astype("float")
    for i in range(size):
        for j in range(size):
            theta, phi = pi / 2 * float(i) / float(size), pi / 2 * float(
                j
            ) / float(size)
            r = sin(theta) * cos(phi)
            g = sin(theta) * sin(phi)
            b = cos(theta)

            # r = cos(theta)
            # g = sin(theta)
            # b = cos(pi/2 - phi)
            wheel[-i - 1, j] = np.array([r, g, b])
    return wheel


def hue_saturation_metric(x, y, w_h=1, w_s=1):
    """Custom metric to compute the distance between two points
    in hue-saturation space. As hue is a circular variable,
    the distance is computed as the circular distance between
    the two hues. The saturation distance is computed as the linear
    distance between the two saturations.

    In mathematical terms, the distance is computed as:
    sqrt(w_h * (1 - cos(h1 - h2)) + w_s * (s1 - s2)^2)

    Parameters
    ----------
    x : np.ndarray
        first point in hue-saturation space (h, s)
    y : np.ndarray
        second point in hue-saturation space (h, s)
    w_h : int, optional
        weight for hue distance, by default 1
    w_s : int, optional
        weight for saturation distance, by default

    Returns
    -------
    float
        distance between the two points in hue-saturation space
    """
    h1, s1 = x
    h2, s2 = y

    # Circular distance for hue
    hue_diff = abs(h1 - h2) % (2 * np.pi)
    hue_diff = min(hue_diff, 2 * np.pi - hue_diff) ** 2

    # Linear distance for saturation
    saturation_diff = (s1 - s2) ** 2

    # Combined weighted distance
    return np.sqrt(w_h * hue_diff + w_s * saturation_diff)


def hue_saturation_wheel_metric(x, y, w_h=1, w_s=1):
    """Custom metric to compute the distance between two points
    in hue-saturation wheel space.
    It will first compute the final position of the points on the wheel
    and then compute the distance between them using a weighted cartesian
    distance.
    """
    x_pos = get_2D_wheel_coordinate(x[0], x[1])
    y_pos = get_2D_wheel_coordinate(y[0], y[1])

    return np.linalg.norm(x_pos - y_pos)
