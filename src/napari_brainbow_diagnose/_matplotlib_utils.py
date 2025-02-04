import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ternary
from matplotlib.colors import LogNorm, hsv_to_rgb, rgb_to_hsv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from sklearn.neighbors import KernelDensity
from ternary.helpers import simplex_iterator

from ._utils_channel_space import (
    get_2D_wheel_coordinate,
    hue_saturation_metric,
    hue_saturation_wheel_metric,
    rgb_to_spherical,
    spherical_to_rgb,
)


def meshgrid_polar_coordinates(n_angles: int = 360, n_radii: int = 100):
    # Create a meshgrid for polar coordinates
    radii = np.linspace(0, 1, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    R, Theta = np.meshgrid(radii, angles)
    grid_points = np.vstack([Theta.ravel(), R.ravel()])

    return grid_points, R, Theta


def background_hue_saturation_plot(
    ax: "matplotlib.axes.Axes",
    bins=(360, 100),
    alpha: float = 1.0,
):

    _, R, Theta = meshgrid_polar_coordinates(n_angles=bins[0], n_radii=bins[1])

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

    return ax


def background_hue_value_plot(ax, bins, alpha=1):

    _, Y, X = meshgrid_polar_coordinates(n_angles=bins[0], n_radii=bins[1])

    # Convert polar coordinates to HSV values
    H = X / (2 * np.pi)  # Hue (angle normalized to [0, 1])
    S = np.ones_like(H) * 1  # Saturation (constant at maximum brightness)
    V = Y  # Value (radius)

    # Stack HSV channels and convert to RGB
    HSV = np.dstack((H, S, V))

    RGB = hsv_to_rgb(HSV)
    alpha = np.ones_like(H) * alpha
    RGB = np.dstack((RGB, alpha))

    # Plot the color wheel using pcolormesh in polar projection
    ax.pcolormesh(X, Y, RGB)

    return ax


def background_spherical_plot(ax, bins=(90, 90), alpha=1):

    x = np.linspace(0, 1, bins[0])
    y = np.linspace(0, 1, bins[1])

    X, Y = np.meshgrid(x, y, indexing="ij")

    THETA = np.pi / 2 * X  # 0 to pi/2
    PHI = np.pi / 2 * Y  # 0 to pi/2
    R = np.ones_like(THETA)

    RGB = np.array(spherical_to_rgb(R, THETA, PHI)).T
    alpha = np.ones_like(RGB[..., 0]) * alpha
    RGB = np.dstack((RGB, alpha))

    X = np.swapaxes(X, 0, 1)
    Y = np.swapaxes(Y, 0, 1)

    ax.pcolormesh(X, Y, RGB)

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
    bins: tuple[int, int] = (360, 100),
    point_size=5,
    alpha=1,
    show_colorbar=False,
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
    bins : tuple[int,int], optional
        The number of bins for the histogram, by default (360, 100)
    point_size : int, optional
        The size of the points, by default 5
    alpha : float, optional
        Transparency of all the plots, by default 1
    show_colorbar : bool, optional
        If True, the colorbar is displayed, by default True

    Returns
    -------
    matplotlib.axes.Axes
        Your matplotlib axe with the plot.
    """

    if color_bg:
        ax = background_hue_saturation_plot(ax, alpha=alpha)

    if theta_r_histogram:
        grid_points, R, Theta = meshgrid_polar_coordinates(
            n_angles=bins[0], n_radii=bins[1]
        )

        hist, _, _ = np.histogram2d(
            theta, r, bins=bins, range=[[0, 2 * np.pi], [0, 1]]
        )
        hist = hist / np.sum(hist)

        if log_scale:
            norm = LogNorm(0.001, 1)
        else:
            norm = None
        h = ax.pcolormesh(Theta, R, hist, norm=norm)

    if wheel_histogram:

        ax, h = plot_histogram_wheel(
            ax, theta, r, log_scale=log_scale, bins=(bins[0], bins[1])
        )

    if show_colorbar and (theta_r_histogram or wheel_histogram):
        label = "density"
        plt.colorbar(
            h,
            ax=ax,
            orientation="vertical",
            label=label,
            fraction=0.046,
            pad=0.04,
        )
    elif show_colorbar:
        print("Colorbar is only available for histograms")

    if contour or kernel_density:
        # Compute the density
        grid_points, R, Theta = meshgrid_polar_coordinates(
            n_angles=bins[0], n_radii=bins[1]
        )

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

    # pretty axes
    ax.set_xlabel("Hue")
    # ax.set_ylabel("Saturation")

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0", "90", "180", "270"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])

    # label_position=ax.get_rlabel_position()
    # ax.text(np.radians(label_position+10),ax.get_rmax()/2.,'Saturation',
    #         rotation=label_position,ha='center',va='center', fontsize=20)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 1)

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(20)

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


def resume_polar_plot(theta, r, title=None, bins=(360, 100)):

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
        axs[0],
        theta,
        r,
        color_bg=True,
        bins=bins,
        point_size=1,
        alpha=1,
        point_color="black",
        scatter=True,
        show_colorbar=False,
    )

    axs[1] = scatter_polar_plot(
        axs[1],
        theta,
        r,
        wheel_histogram=True,
        log_scale=True,
        bins=bins,
        point_color="black",
        scatter=False,
        show_colorbar=True,
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
    fontsize: int = 20,
    background: bool = False,
    point_color: str = None,
    point_size: int = 5,
    point_alpha: float = 1,
    heatmap: bool = False,
    labels: np.ndarray = None,
    cluster_colors: dict = None,
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
    """

    if dark_mode:
        # plt.style.use("dark_background")
        c_boundary = "whitesmoke"
        c_gridlines = "white"
        # axes_colors = {"l": "w", "r": "w", "b": "w"}
        c_triangle = "gray"
        set_background_color = "black"
    else:
        # plt.style.use("default") # removed to respect scienceplot style
        c_boundary = "black"
        c_gridlines = "black"
        # axes_colors = {"l": "black", "r": "black", "b": "black"}
        c_triangle = "black"
        set_background_color = "white"

    # Boundary and Gridlines
    ax.set_aspect("equal", adjustable="box", anchor="C")
    figure, tax = ternary.figure(ax=ax, scale=scale)

    # Draw Boundary and Gridlines
    tax.boundary(linewidth=1.0, c=c_boundary)
    tax.gridlines(color=c_gridlines, multiple=50, alpha=0.4)

    # Set ticks
    tax.ticks(
        multiple=50,
        offset=0.025,
        fontsize=fontsize,
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
        cax = divider.append_axes("right", size="4%", pad=0.2)
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

        # avoid the right margin
        # figure.subplots_adjust(right=0.8)

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
        if background:
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

    tax.left_axis_label("blue", fontsize=fontsize, offset=0.15)
    tax.right_axis_label("green", fontsize=fontsize, offset=0.15)
    tax.bottom_axis_label("red", fontsize=fontsize, offset=0.05)

    # ensure that the triangle is equilateral
    # figure.set_size_inches(10, 10 * np.cos(30 * np.pi / 180))

    return figure, tax


def resume_ternary(data: np.ndarray):
    # Create a figure with two subplots
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 5), sharey=True, sharex=True
    )  # , layout="constrained")

    # Ensure both subplots have an equal aspect ratio
    for ax in axes:
        ax.set_aspect("equal", adjustable="box", anchor="SE")

    # Create second ternary plot
    _ = create_maxwell_triangle(
        data,
        point_size=1,
        ax=axes[0],
        scale=100,
        point_alpha=0.1,
        point_color="black",
        heatmap=False,
        background=True,
    )

    # Create first ternary plot
    _ = create_maxwell_triangle(
        data,
        ax=axes[1],
        scale=100,
        point_alpha=1,
        point_color="black",
        heatmap=True,
    )

    # Adjust layout

    # add a) and b) to the subplots at their left corner
    axes[0].text(-0.1, 1.1, "a)", transform=axes[0].transAxes, fontsize=16)
    axes[1].text(-0.1, 1.1, "b)", transform=axes[1].transAxes, fontsize=16)

    plt.subplots_adjust(wspace=0.3)  # Adjust spacing


def cartesian_brainbow_plot(
    ax: "matplotlib.axes.Axes",
    x: np.ndarray,
    y: np.ndarray,
    scatter: bool = True,
    point_color: str = None,
    histogram: bool = False,
    log_scale: bool = False,
    kernel_density: bool = False,
    kernel_metric: str = None,
    contour: bool = False,
    bins: tuple[int, int] = (100, 100),
    point_size=5,
    alpha=0.5,
    polar=True,
):
    """
    Plot a scatter plot in cartesian coordinates or its histogram,

    Returns
    -------
    matplotlib.axes.Axes
        Your matplotlib axe with the plot.
    """

    if histogram:
        grid_points, Y, X = meshgrid_polar_coordinates(bins[0], bins[1])

        if not polar:
            Y = Y / np.max(Y)
            X = X / np.max(X)

        hist, _, _ = np.histogram2d(x, y, bins=bins)
        hist = hist / np.sum(hist)

        if log_scale:
            norm = LogNorm(0.001, 1)
        else:
            norm = None
        h = ax.pcolormesh(X, Y, hist, norm=norm)

        label = "density"
        # colorbar(h)

        # divider = make_axes_locatable(ax)
        # cax1 = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(h, cax=cax1)
        # plt.colorbar(
        #     h,
        #     ax=ax,
        #     orientation="vertical",
        #     label=label,
        #     fraction=0.046,
        #     pad=0.04,
        # )

    if contour or kernel_density:
        # Compute the density
        grid_points, Y, X = meshgrid_polar_coordinates(bins[0], bins[1])

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

        kde.fit(np.vstack([x, y]).T)
        density = kde.score_samples(grid_points.T)

        if not log_scale:
            density = np.exp(density)

        density = density.reshape(Y.shape)

        label = "density"
        if kernel_density:
            h = ax.pcolormesh(
                X, Y, density
            )  # , norm=LogNorm(0.1, max_density))

        if contour:
            h = ax.contourf(X, Y, density, levels=10, cmap="viridis")

        plt.colorbar(
            h,
            ax=ax,
            orientation="horizontal",
            label=label,
        )
    if scatter:
        if point_color is None:
            ax.scatter(x, y, c="black", s=point_size, alpha=alpha)
        else:
            ax.scatter(x, y, c=point_color, s=point_size, alpha=alpha)

    return ax


def hue_saturation_plot(
    ax: "matplotlib.axes.Axes",
    hue: np.ndarray,
    saturation: np.ndarray,
    scatter: bool = True,
    background: bool = True,
    point_color: str = None,
    histogram: bool = False,
    log_scale: bool = False,
    kernel_density: bool = False,
    kernel_metric: str = None,
    contour: bool = False,
    bins: tuple[int, int] = (360, 100),
    point_size=5,
    alpha=0.5,
) -> "matplotlib.axes.Axes":

    # ax.set_aspect(f"{2 * np.pi / 1}", adjustable="box", anchor="C")

    if background:
        ax = background_hue_saturation_plot(ax=ax, bins=bins)

    if point_color is None:
        hsv = np.zeros((len(hue), 3))
        hsv[:, 0] = hue / (2 * np.pi)
        hsv[:, 1] = saturation
        hsv[:, 2] = 1

        point_color = hsv_to_rgb(hsv)

    ax = cartesian_brainbow_plot(
        ax=ax,
        x=hue,
        y=saturation,
        scatter=scatter,
        point_color=point_color,
        histogram=histogram,
        log_scale=log_scale,
        kernel_density=kernel_density,
        kernel_metric=kernel_metric,
        contour=contour,
        bins=bins,
        point_size=point_size,
        alpha=alpha,
    )

    ax.set_aspect(f"{2 * np.pi / 1}", adjustable="box", anchor="C")

    # pretty axes
    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")

    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", "180", "360"])

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 1)

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(20)

    return ax


def hue_value_plot(
    ax: "matplotlib.axes.Axes",
    hue: np.ndarray,
    value: np.ndarray,
    scatter: bool = True,
    background: bool = True,
    point_color: str = None,
    histogram: bool = False,
    log_scale: bool = False,
    kernel_density: bool = False,
    kernel_metric: str = None,
    contour: bool = False,
    bins: tuple[int, int] = (360, 100),
    point_size=5,
    alpha=0.5,
) -> "matplotlib.axes.Axes":

    if background:
        ax = background_hue_value_plot(ax=ax, bins=bins)

    if point_color is None:
        hsv = np.zeros((len(hue), 3))
        hsv[:, 0] = hue / (2 * np.pi)
        hsv[:, 1] = 1
        hsv[:, 2] = value

        point_color = hsv_to_rgb(hsv)

    ax = cartesian_brainbow_plot(
        ax=ax,
        x=hue,
        y=value,
        scatter=scatter,
        point_color=point_color,
        histogram=histogram,
        log_scale=log_scale,
        kernel_density=kernel_density,
        kernel_metric=kernel_metric,
        contour=contour,
        bins=bins,
        point_size=point_size,
        alpha=alpha,
    )

    # pretty axes
    ax.set_xlabel("Hue")
    ax.set_ylabel("Value")

    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", "180", "360"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 1)

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(20)

    ax.set_aspect(f"{2 * np.pi / 1}", adjustable="box", anchor="C")

    # ax.set_aspect("", adjustable="datalim")

    return ax


def spherical_plot(
    ax: "matplotlib.axes.Axes",
    theta: np.ndarray,
    phi: np.ndarray,
    scatter: bool = True,
    background: bool = True,
    point_color: str = None,
    histogram: bool = False,
    log_scale: bool = False,
    kernel_density: bool = False,
    kernel_metric: str = None,
    contour: bool = False,
    bins: tuple[int, int] = (360, 100),
    point_size=5,
    alpha=0.5,
) -> "matplotlib.axes.Axes":

    if background:
        ax = background_spherical_plot(ax=ax, bins=bins)

    if point_color is None:
        point_color = np.zeros((len(theta), 3))
        r, g, b = spherical_to_rgb(
            np.ones_like(theta), np.pi / 2 * theta, np.pi / 2 * phi
        )
        point_color[:, 0] = r
        point_color[:, 1] = g
        point_color[:, 2] = b

    ax = cartesian_brainbow_plot(
        ax=ax,
        x=theta,
        y=phi,
        scatter=scatter,
        point_color=point_color,
        histogram=histogram,
        log_scale=log_scale,
        kernel_density=kernel_density,
        kernel_metric=kernel_metric,
        contour=contour,
        bins=bins,
        point_size=point_size,
        alpha=alpha,
        polar=False,
    )

    # pretty axes
    ax.set_xlabel("Theta")
    ax.set_ylabel("Phi")

    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "45", "90"])

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "45", "90"])

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(20)

    ax.set_aspect("equal", adjustable="box")

    return ax


def plot_all(rgb):

    # Create a figure with two subplots
    fig, axes = plt.subplots(5, 2, figsize=(20, 40), layout="constrained")
    # Ensure both subplots have an equal aspect ratio

    hsv = rgb_to_hsv(rgb)

    theta = hsv[:, 0] * 2 * np.pi
    r = hsv[:, 1]

    _, theta2, phi = rgb_to_spherical(rgb[:, 0], rgb[:, 1], rgb[:, 2])
    maxwell_data = rgb / rgb.sum(axis=1)[:, None]

    axes[1][0].axis("off")

    axes[1][0] = plt.subplot(5, 2, 3, projection="polar")
    scatter_polar_plot(
        axes[1][0],
        theta,
        r,
        color_bg=True,
        point_color="black",
        scatter=True,
        point_size=0.1,
    )

    axes[1][1].axis("off")

    axes[1][1] = plt.subplot(5, 2, 4, projection="polar")
    scatter_polar_plot(
        axes[1][1],
        theta,
        r,
        color_bg=False,
        scatter=False,
        wheel_histogram=True,
        log_scale=True,
        bins=(50, 50),
        point_size=0.1,
    )

    hue_value_plot(
        axes[4][0], theta, r, point_color="white", point_size=1, alpha=0.5
    )
    hue_value_plot(
        axes[4][1],
        theta,
        r,
        scatter=False,
        background=False,
        histogram=True,
        log_scale=True,
        bins=(30, 30),
    )

    hue_saturation_plot(
        axes[3][0], theta, r, point_color="black", point_size=1, alpha=0.5
    )
    hue_saturation_plot(
        axes[3][1],
        theta,
        r,
        scatter=False,
        background=False,
        histogram=True,
        log_scale=True,
        bins=(30, 30),
    )

    spherical_plot(
        axes[2][0],
        theta2 / 90,
        phi / 90,
        point_color="black",
        point_size=1,
        alpha=0.5,
    )
    spherical_plot(
        axes[2][1],
        theta2 / 90,
        phi / 90,
        scatter=False,
        background=False,
        histogram=True,
        log_scale=True,
        bins=(30, 30),
    )

    create_maxwell_triangle(
        maxwell_data,
        point_size=1,
        ax=axes[0][0],
        point_color="black",
        background=True,
    )

    create_maxwell_triangle(
        maxwell_data,
        point_size=1,
        ax=axes[0][1],
        point_color="black",
        heatmap=True,
        scale=100,
    )

    for i, ax in enumerate(axes.flatten()):
        ax.text(
            -0.2,
            1.2,
            f"{chr(97+i)})",
            transform=ax.transAxes,
            fontsize=20,
            va="top",
        )

    return fig, axes
