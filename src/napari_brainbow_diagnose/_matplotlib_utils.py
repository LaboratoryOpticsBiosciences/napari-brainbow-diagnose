import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, hsv_to_rgb
from sklearn.neighbors import KernelDensity

from ._utils_channel_space import hue_saturation_metric


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
    histogram: bool = False,
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

    if histogram:
        grid_points, R, Theta = meshgrid_polar_coordinates(n_angles, n_radii)

        hist, _, _ = np.histogram2d(
            theta, r, bins=(n_angles, n_radii), range=[[0, 2 * np.pi], [0, 1]]
        )

        label = "Number of points"
        if log_scale:
            max_ = np.max(hist)
            hist[hist <= 0] = np.nan
            hist = np.log(hist, where=hist != np.nan)
            h = ax.pcolormesh(Theta, R, hist, norm=LogNorm(0.1, max_))
            label += " in log scale"
        else:
            h = ax.pcolormesh(Theta, R, hist)

        # add colorbar to axis

        plt.colorbar(
            h, ax=ax, orientation="horizontal", label=label, extend="max"
        )

    if contour or kernel_density:
        # Compute the density
        grid_points, R, Theta = meshgrid_polar_coordinates(n_angles, n_radii)
        # Scott's rule of thumb for bandwidth
        bandwith = len(theta) ** (-1.0 / (2 + 4))  # d=2
        print(bandwith)
        kde = KernelDensity(
            bandwidth=bandwith,
            kernel="gaussian",
            metric="pyfunc",
            metric_params={"func": hue_saturation_metric},
        )
        kde.fit(np.vstack([theta, r]).T)
        density = kde.score_samples(np.vstack([Theta.ravel(), R.ravel()]).T)

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
