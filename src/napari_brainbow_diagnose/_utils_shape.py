import numpy as np


def image_to_flat_data_points(a: np.ndarray, channel_axis: int) -> np.ndarray:
    """
    Flattens a numpy array by moving the specified channel index to
    the first axis and flattening the array.

    Parameters
    ----------
    a : np.ndarray
        The input numpy array to be flattened.
    channel_axis : int
        The index of the channel to be moved to the first axis.

    Returns
    -------
    np.ndarray
        The flattened (n*channel) array.
    """
    channel_first = np.moveaxis(a, channel_axis, 0)
    flat_data_points = channel_first.reshape(channel_first.shape[0], -1).T
    return flat_data_points


def flat_data_points_to_image(
    a: np.ndarray, shape: tuple, channel_axis: int
) -> np.ndarray:
    """
    Reshapes a flattened array to the specified shape.

    Parameters
    ----------
    a : np.ndarray
        The flattened array.
    shape : tuple
        The desired shape of the output array.
    channel_axis : int
        The index of the channel to be moved to the first axis.

    Returns
    -------
    np.ndarray
        The reshaped array.
    """
    shape = np.array(shape)
    shape[[0, channel_axis]] = shape[
        [channel_axis, 0]
    ]  # put channel axis first
    channel_first = a.T.reshape(shape)
    channel_last = np.moveaxis(channel_first, 0, channel_axis)
    return channel_last


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
