from typing import List, Union

import numpy as np


def generate_ellipse_coordinates(
    center: np.ndarray,
    radius: Union[int, list] = 1,
) -> tuple:
    """
    Generate coordinates of an ellipse with a given center and radius.

    Parameters
    ----------
    center : np.ndarray
        point in space where the ellipse is centered with shape (d,)
        where d is the dimension of the space
    radius : Union[int, list], optional
        radius of the ellipse. It can be a scalar or a vector with the same
        dimension as the center. If it is a scalar, the same radius is used
        for all dimensions, by default 1.

    Returns
    -------
    tuple
        tuple of arrays with the coordinates of the points in the ellipse.
        In shape (d, n) where d is the dimension of the space and n is the
        number of points in the ellipse.
    """

    if isinstance(radius, int):
        radius = [radius] * len(center)

    ranges = [
        np.arange(int(center[i] - radius[i]), int(center[i] + radius[i]) + 1)
        for i in range(len(center))
    ]
    grids = np.meshgrid(*ranges, indexing="ij")

    distances = np.array(
        [
            ((grids[d] - center[d]) ** 2) / radius[d] ** 2
            for d in range(len(center))
        ]
    )

    distances = np.sum(distances, axis=0) <= 1
    distances = distances.flatten()

    coordinates = np.column_stack([coord.flatten() for coord in grids])
    coordinates = coordinates[distances]

    return tuple(coordinates.T)


def get_mean_intensity_ellipse(
    img: np.ndarray,
    points: np.ndarray,
    radius: Union[int, List[int]] = 1,
) -> np.ndarray:
    """
    Calculate the mean intensity of an ellipse around a point in an image.

    Parameters
    ----------
    img : np.ndarray
        image to calculate the mean intensity of the ellipse.
    points : np.ndarray
        points to calculate the mean intensity of the ellipse.
        the points should be in the format of (n, d) where n is the number of
        points and d is the number of dimensions. The number of dimensions
        should correspond to the number of dimensions in the image.
    radius : Union[int, list[int]]
        radius of the ellipse to calculate the mean intensity of.
        - if a int is provided, the radius will be the same for all dimensions.
        - if a list is provided, the radius will be different for each
            dimensions. (default is 1)

        For the channel dimension, you can use the parameter 'channel_axis' to
        specify the channel axis. The default is None, which means that the
        function will assume that the image does not have a channel axis.
        (See brainbow_decorator for more information.)

    Returns
    -------
    np.ndarray
        mean intensity of the ellipse around the points.
    """

    assert img.ndim == points.shape[1], (
        "The number of dimensions in the image should be "
        "the same as the number of dimensions in the points."
    )

    point_intensity_means = []
    for i in range(len(points)):
        # check point is inside the image
        assert all(
            [0 <= points[i][j] < img.shape[j] for j in range(img.ndim)]
        ), f"The point {points[i]} is outside the image of shape {img.shape}."

        idx = generate_ellipse_coordinates(points[i], radius)

        # check if the coordinates are within the image.
        idx = tuple(
            [np.clip(idx[i], 0, img.shape[i] - 1) for i in range(img.ndim)]
        )

        selected = img[idx]
        point_intensity_means.append(np.mean(selected))

    return np.array(point_intensity_means)
