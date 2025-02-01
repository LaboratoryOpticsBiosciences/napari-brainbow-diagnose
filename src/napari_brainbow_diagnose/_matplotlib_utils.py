import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, hsv_to_rgb
from scipy.interpolate import griddata
from sklearn.neighbors import KernelDensity

from ._utils_channel_space import (
    get_2D_wheel_coordinate,
    hue_saturation_metric,
)


def meshgrid_polar_coordinates(n_angles: int = 360, n_radii: int = 100):
    # Create a meshgrid for polar coordinates
    radii = np.linspace(0, 1, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    R, Theta = np.meshgrid(radii, angles)
    grid_points = np.vstack([Theta.ravel(), R.ravel()])

    return grid_points, R, Theta


def hue_saturation_polar_plot(
    ax: "matplotlib.axes.Axes",
    n_angles: int = 360,
    n_radii: int = 100,
    alpha: float = 1.0,
):

    _, R, Theta = meshgrid_polar_coordinates()

    # Convert polar coordinates to HSV values
    H = Theta / (2 * np.pi)  # Hue (angle normalized to [0, 1])
    S = R  # Saturation (radius)
    V = np.ones_like(H)  # Value (constant at maximum brightness)

    # Stack HSV channels and convert to RGB
    HSV = np.dstack((H, S, V))

    RGB = hsv_to_rgb(HSV)
    alpha = np.ones_like(H) * alpha
    RGB = np.dstack((RGB, alpha))

    # Plot the color wheel using pcolormesh in polar projection
    ax.pcolormesh(Theta, R, RGB)

    # name the axis

    return ax


def scatter_polar_plot(
    ax: "matplotlib.axes.Axes",
    theta: np.ndarray,
    r: np.ndarray,
    scatter: bool = True,
    theta_r_histogram: bool = False,
    wheel_histogram: bool = False,
    log_scale: bool = False,
    kernel_density: bool = False,
    contour: bool = False,
    color_bg: bool = False,
    n_angles: int = 360,
    n_radii: int = 100,
    point_color: str = None,  # if none, color is computed from theta / r
    point_size=5,
    alpha=0.5,
):

    if color_bg:
        ax = hue_saturation_polar_plot(ax, alpha=alpha)

    if theta_r_histogram:
        grid_points, R, Theta = meshgrid_polar_coordinates(n_angles, n_radii)

        hist, _, _ = np.histogram2d(
            theta, r, bins=(n_angles, n_radii), range=[[0, 2 * np.pi], [0, 1]]
        )

        if log_scale:
            norm = LogNorm()
        else:
            norm = None
        h = ax.pcolormesh(Theta, R, hist, norm=norm)

    if wheel_histogram:
        ax, h = plot_histogram_wheel(
            ax, theta, r, log_scale=log_scale, bins=(n_angles, n_radii)
        )

    if wheel_histogram or theta_r_histogram:
        label = "Number of points"
        plt.colorbar(
            h,
            ax=ax,
            orientation="horizontal",
            label=label,
            extend="max",
            fraction=0.046,
            pad=0.04,
        )

    if contour or kernel_density:
        # Compute the density
        grid_points, R, Theta = meshgrid_polar_coordinates(n_angles, n_radii)

        kde = KernelDensity(
            bandwidth="scott",
            kernel="gaussian",
            metric="pyfunc",
            metric_params={"func": hue_saturation_metric},
        )
        kde.fit(np.vstack([theta, r]).T)
        density = kde.score_samples(grid_points.T)

        if not log_scale:
            density = np.exp(density)

        density = density.reshape(R.shape)

        label = "Number of points"
        if kernel_density:
            h = ax.pcolormesh(
                Theta, R, density
            )  # , norm=LogNorm(0.1, max_density))

        if contour:
            h = ax.contourf(Theta, R, density, levels=10, cmap="viridis")

        plt.colorbar(
            h,
            ax=ax,
            orientation="horizontal",
            label=label,
        )
    if scatter:
        if point_color is None:
            color = np.zeros((len(theta), 3))
            color[:, 0] = theta / (2 * np.pi)
            color[:, 1] = r
            color[:, 2] = 1
            rgb = hsv_to_rgb(color)
            ax.scatter(theta, r, c=rgb, s=point_size, alpha=alpha)
        else:
            ax.scatter(theta, r, c=point_color, s=point_size, alpha=alpha)

    return ax


def plot_histogram_wheel(
    ax: "matplotlib.axes.Axes",
    theta: np.ndarray,
    r: np.ndarray,
    bins: tuple = (360, 100),
    log_scale: bool = False,
):
    x_wheel, y_wheel = get_2D_wheel_coordinate(theta, r)

    x_bins = np.linspace(-1, 1, bins[0] + 1)
    y_bins = np.linspace(-1, 1, bins[1] + 1)

    hist, x_edges, y_edges = np.histogram2d(
        x_wheel, y_wheel, bins=[x_bins, y_bins]
    )

    # Get bin centers (instead of edges)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    Xc, Yc = np.meshgrid(x_centers, y_centers)  # Grid of bin centers

    # Flatten for interpolation
    cartesian_points = np.array([Xc.ravel(), Yc.ravel()]).T
    hist_values = hist.T.ravel()  # Flatten histogram values

    theta_bins = np.linspace(0, 2 * np.pi, bins[0] + 1)
    r_bins = np.linspace(0, 1, bins[1] + 1)

    T, R = np.meshgrid(theta_bins, r_bins)  # Polar mesh grid

    # Convert polar grid to Cartesian for interpolation
    x_polar, y_polar = get_2D_wheel_coordinate(T.flatten(), R.flatten())

    C = griddata(
        cartesian_points, hist_values, (x_polar, y_polar), method="nearest"
    )
    C = C.reshape(R.shape)  # Reshape back to grid shape

    if log_scale:
        norm = LogNorm()
    else:
        norm = None

    h = ax.pcolormesh(T, R, C, norm=norm)

    return ax, h
