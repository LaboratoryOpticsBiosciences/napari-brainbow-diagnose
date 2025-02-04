import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ternary
from matplotlib.colors import LogNorm, hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from sklearn.neighbors import KernelDensity
from ternary.helpers import simplex_iterator

from ._utils_channel_space import (
    get_2D_wheel_coordinate,
    hue_saturation_metric,
    hue_saturation_wheel_metric,
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

    _, R, Theta = meshgrid_polar_coordinates(n_angles, n_radii)

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
    point_color: str = None,
    color_bg: bool = False,
    theta_r_histogram: bool = False,
    wheel_histogram: bool = False,
    log_scale: bool = False,
    kernel_density: bool = False,
    kernel_metric: str = "hue_saturation_wheel_metric",
    contour: bool = False,
    n_angles: int = 360,
    n_radii: int = 100,
    point_size=5,
    alpha=0.5,
):
    """
    Plot a scatter plot in polar coordinatesor its histogram,
    density or contour.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Your matplotlib axe.
    theta : np.ndarray
        angles in radians.
    r : np.ndarray
        saturation values between 0 and 1.
    scatter : bool, optional
        If True, scatter plot is displayed, by default True
    point_color : str, optional
        If not None, the color of the points is set to this value.
        If None, the color is computed from the theta and r values
        by default None
    color_bg : bool, optional
        If True, the background is colored with the hue and saturation,
        by default False
    theta_r_histogram : bool, optional
        If True, a histogram of the theta and r values is displayed,
        by default False
    wheel_histogram : bool, optional
        If True, a histogram of the wheel values is displayed, by default False
    log_scale : bool, optional
        If True, the color bar is displayed in log scale, by default False
    kernel_density : bool, optional
        If True, the kernel density is displayed, by default False
    kernel_metric : str, optional
        The metric used for the kernel density.
        Custom options:
        - "hue_saturation_metric": Distance in hue-saturation plane space
        - "hue_saturation_wheel_metric": distance once projected on the wheel.
        All other metrics from sklearn.neighbors.KernelDensity are available.
        by default "hue_saturation_wheel_metric".
    contour : bool, optional
        If True, the contour plot is displayed, by default False
    n_angles : int, optional
        Number of bins for the angles, by default 360
    n_radii : int, optional
        Number of bins for the radii, by default 100
    alpha : float, optional
        Transparency of all the plots, by default 0.5

    Returns
    -------
    matplotlib.axes.Axes
        Your matplotlib axe with the plot.
    """

    if color_bg:
        ax = hue_saturation_polar_plot(ax, alpha=alpha)

    if theta_r_histogram:
        grid_points, R, Theta = meshgrid_polar_coordinates(n_angles, n_radii)

        hist, _, _ = np.histogram2d(
            theta, r, bins=(n_angles, n_radii), range=[[0, 2 * np.pi], [0, 1]]
        )
        hist = hist / np.sum(hist)

        if log_scale:
            norm = LogNorm(0.001, 1)
        else:
            norm = None
        h = ax.pcolormesh(Theta, R, hist, norm=norm)

    if wheel_histogram:

        ax, h = plot_histogram_wheel(
            ax, theta, r, log_scale=log_scale, bins=(n_angles, n_radii)
        )

    if wheel_histogram or theta_r_histogram:
        label = "density"
        plt.colorbar(
            h,
            ax=ax,
            orientation="vertical",
            label=label,
            fraction=0.046,
            pad=0.04,
        )

    if contour or kernel_density:
        # Compute the density
        grid_points, R, Theta = meshgrid_polar_coordinates(n_angles, n_radii)

        if kernel_metric == "hue_saturation_wheel_metric":
            metric = "pyfunc"
            metric_params_func = {"func": hue_saturation_wheel_metric}
        elif kernel_metric == "hue_saturation_metric":
            metric = "pyfunc"
            metric_params_func = {"func": hue_saturation_metric}
        else:
            metric = kernel_metric
            metric_params_func = None
        kde = KernelDensity(
            bandwidth="scott",
            kernel="gaussian",
            metric=metric,
            metric_params=metric_params_func,
            rtol=10,  # force higher rtol for faster computation
        )

        kde.fit(np.vstack([theta, r]).T)
        density = kde.score_samples(grid_points.T)

        if not log_scale:
            density = np.exp(density)

        density = density.reshape(R.shape)

        label = "density"
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

    # density of hist
    hist_values = hist_values / np.sum(hist_values)

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
        norm = LogNorm(0.001, 1)
    else:
        norm = None

    h = ax.pcolormesh(T, R, C, norm=norm)

    return ax, h


def resume_polar_plot(theta, r, title=None, n_angles=36, n_radii=10):

    # same as above, but with two figures side by side that share
    # the same colorbar
    fig, axs = plt.subplots(
        1,
        2,
        subplot_kw={"projection": "polar"},
        figsize=(10, 5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    axs[0] = scatter_polar_plot(
        axs[1],
        theta,
        r,
        color_bg=True,
        n_angles=n_angles,
        n_radii=n_radii,
        point_size=1,
        alpha=1,
        point_color="black",
        scatter=True,
    )

    axs[1] = scatter_polar_plot(
        axs[0],
        theta,
        r,
        wheel_histogram=True,
        log_scale=True,
        n_angles=n_angles,
        n_radii=n_radii,
        point_color="black",
        scatter=False,
    )

    axs[0].text(-0.1, 1.1, "a)", transform=axs[0].transAxes, fontsize=16)
    axs[1].text(-0.1, 1.1, "b)", transform=axs[1].transAxes, fontsize=16)

    if title:
        fig.suptitle(title, y=1.05)
    else:
        fig.suptitle(
            f"Polar scatter plot with histogram (total {len(theta)} points)",
            y=1.05,
        )


# Maxwel triangle plot


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

    def color_point(x, y, z, scale):
        r = x / float(scale)
        g = y / float(scale)
        b = z / float(scale)
        max_ = max(r, g, b)
        r = r / max_
        g = g / max_
        b = b / max_
        return (r, g, b, 1)

    d = dict()
    for (i, j, k) in simplex_iterator(size):
        d[(i, j, k)] = color_point(i, j, k, size)
    return d


def color_point(hist, x, y, z):
    color = hist[int(x), int(y), int(z)]
    return color


def generate_heatmap_data(hist, scale=99):
    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = color_point(hist, i, j, k)
    return d


def create_maxwell_triangle(
    data: np.ndarray,
    ax=None,
    dark_mode=False,
    unmixing_triangle: np.ndarray = None,
    scale: int = 100,
    fontsize: int = 10,
    point_color: str = None,
    point_size: int = 5,
    point_alpha: float = 1,
    heatmap: bool = False,
    labels: np.ndarray = None,
    cluster_colors: dict = None,
    title: str = None,
) -> tuple:
    """
    Create a Maxwell triangle plot of the data to visualize
    the color distribution.

    Args:
        data (np.ndarray): The data to plot. The data should be
            in RGB format and the sum of the values should be 1.
        dark_mode (bool, optional): Use dark mode. Defaults to False.
        unmixing_triangle (np.ndarray, optional):
            The unmixing triangle to plot.
            Defaults to None.
        scale (int, optional): The scale of the plot. Defaults to 100.
        fontsize (int, optional): The fontsize of the labels. Defaults to 10.
        point_size (int, optional): The size of the points. Defaults to 2.
        point_alpha (float, optional): The alpha of the points.
            Defaults to 0.5.
        heatmap (bool, optional): Use a heatmap. Defaults to False. Use `scale`
            to set the number of bins.
        labels (np.ndarray, optional): The labels of the data.
            Defaults to None.
        cluster_colors (dict, optional): The colors of the clusters.
            Defaults to None.
        title (str, optional): The title of the plot. Defaults to None.
            If None, the title will be the number of points.
    """

    if dark_mode:
        plt.style.use("dark_background")
        c_boundary = "whitesmoke"
        c_gridlines = "white"
        axes_colors = {"l": "w", "r": "w", "b": "w"}
        c_triangle = "gray"
        set_background_color = "black"
    else:
        plt.style.use("default")
        c_boundary = "black"
        c_gridlines = "black"
        axes_colors = {"l": "black", "r": "black", "b": "black"}
        c_triangle = "black"
        set_background_color = "white"

    # Boundary and Gridlines
    figure, tax = ternary.figure(ax=ax, scale=scale)

    # Draw Boundary and Gridlines
    tax.boundary(linewidth=1.0, c=c_boundary)
    tax.gridlines(color=c_gridlines, multiple=10, alpha=0.4)

    # Set ticks
    tax.ticks(
        axis="lbr",
        linewidth=1,
        multiple=10,
        offset=0.025,
        axes_colors=axes_colors,
    )

    center = np.array((1 / 3, 1 / 3, 1 / 3)) * scale
    p1 = np.array((0, 2 / 3, 1 / 3)) * scale
    p2 = np.array((1 / 3, 2 / 3, 0)) * scale
    p3 = np.array((2 / 3, 0, 1 / 3)) * scale
    p4 = np.array((1 / 3, 0, 2 / 3)) * scale
    p5 = np.array((0, 1 / 3, 2 / 3)) * scale
    p6 = np.array((2 / 3, 1 / 3, 0)) * scale
    tax.line(
        center, p1, linewidth=1, color=c_gridlines, linestyle="--", alpha=0.5
    )
    tax.line(
        center, p2, linewidth=1, color=c_gridlines, linestyle="--", alpha=0.5
    )
    tax.line(
        center, p3, linewidth=1, color=c_gridlines, linestyle="--", alpha=0.5
    )
    tax.line(
        center, p4, linewidth=1, color=c_gridlines, linestyle="--", alpha=0.5
    )
    tax.line(
        center, p5, linewidth=1, color=c_gridlines, linestyle="--", alpha=0.5
    )
    tax.line(
        center, p6, linewidth=1, color=c_gridlines, linestyle="--", alpha=0.5
    )
    # Remove default Matplotlib Axes
    tax.get_axes().axis("off")

    # Draw the unmixing triangle
    if unmixing_triangle is not None:
        tax.line(
            unmixing_triangle[:, 0] * scale,
            unmixing_triangle[:, 1] * scale,
            color=c_triangle,
            linestyle=":",
        )
        tax.line(
            unmixing_triangle[:, 0] * scale,
            unmixing_triangle[:, 2] * scale,
            color=c_triangle,
            linestyle=":",
        )
        tax.line(
            unmixing_triangle[:, 1] * scale,
            unmixing_triangle[:, 2] * scale,
            color=c_triangle,
            linestyle=":",
        )

    if heatmap:
        import matplotlib.colors as colors

        # create the heatmap data
        hist, _ = np.histogramdd(
            data, bins=scale + 1, range=[[0, 1], [0, 1], [0, 1]], density=True
        )
        # hist_log = np.log(hist, where=hist>0)
        # hist_log[hist_log <= 0] = np.nan
        hist = hist / np.sum(hist)
        d = generate_heatmap_data(hist, scale)
        heat = tax.heatmap(d, colorbar=False)

        # add the colorbar after to avoid modifing the figure heatmap size.
        divider = make_axes_locatable(tax.get_axes())
        cax = divider.append_axes("right", size="4%", pad=0.1)
        _ = figure.colorbar(
            heat,
            cax=cax,
            # shrink the colorbar
            fraction=0.046,
            pad=0.04,
            orientation="vertical",
            norm=colors.LogNorm(0.001, 1),
            label="density",
        )
    # Plot the data
    elif cluster_colors:
        # Plot points with colors corresponding to their labels
        for label in cluster_colors:
            cluster_data = data[labels == cluster_colors[label]]
            if len(cluster_data) > 0:
                tax.scatter(
                    cluster_data * scale,
                    color=label,
                    s=point_size,
                    alpha=point_alpha,
                )
    else:

        map_ = maxwell_hue_space(100)
        tax.heatmap(
            map_,
            use_rgba=True,
            colorbar=False,
        )
        if point_color is not None:
            tax.scatter(
                data * scale,
                color=point_color,
                s=point_size,
                alpha=point_alpha,
            )
        else:
            tax.scatter(
                data * scale, color=data, s=point_size, alpha=point_alpha
            )
        tax.set_background_color(set_background_color)

    # Set Axis labels and Title
    # put the title at the bottom
    if title is not None:
        tax.set_title(title, fontsize=fontsize * 1.5, y=-0.15)
    else:
        tax.set_title(
            f"{len(data)} points ternary plot",
            fontsize=fontsize * 1.5,
            y=-0.15,
        )
    tax.left_axis_label("blue", fontsize=fontsize, offset=0.15)
    tax.right_axis_label("green", fontsize=fontsize, offset=0.15)
    tax.bottom_axis_label("red", fontsize=fontsize, offset=0.05)

    # ensure that the triangle is equilateral
    figure.set_size_inches(10, 10 * np.cos(30 * np.pi / 180))

    return figure, tax


def resume_ternary(data: np.ndarray):
    # Create a figure with two subplots
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 5), sharey=True, sharex=True
    )  # , layout="constrained")

    # Ensure both subplots have an equal aspect ratio
    for ax in axes:
        ax.set_aspect("equal")

    # Create second ternary plot
    _ = create_maxwell_triangle(
        data,
        point_size=1,
        ax=axes[0],
        title="",
        scale=100,
        point_alpha=0.1,
        point_color="black",
        heatmap=False,
    )

    # Create first ternary plot
    _ = create_maxwell_triangle(
        data,
        ax=axes[1],
        title="",
        scale=100,
        point_alpha=1,
        point_color="black",
        heatmap=True,
    )

    # Adjust layout

    # add a) and b) to the subplots at their left corner
    axes[0].text(-0.1, 1.1, "a)", transform=axes[0].transAxes, fontsize=16)
    axes[1].text(-0.1, 1.05, "b)", transform=axes[1].transAxes, fontsize=16)

    plt.subplots_adjust(wspace=0.3)  # Adjust spacing
