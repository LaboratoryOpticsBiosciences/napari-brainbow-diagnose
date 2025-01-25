import copy
from typing import TYPE_CHECKING

import matplotlib.cm
import matplotlib.colors as colors
import numpy as np
import ternary
from matplotlib import path
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector

from napari_brainbow_diagnose._utils_shape import (
    get_2D_wheel_coordinate,
    image_to_flat_data_points,
)

from ._utils_channel_space import (
    hue_saturation_color_wheel,
    image_selection_to_wheel_selection,
    maxwell_hue_empty,
    maxwell_hue_space,
    rgb_to_hsv,
    spherical_coordinates_color_wheel,
)

if TYPE_CHECKING:
    pass


def get_channel_density(
    data_points: np.ndarray,
    bins: int,
    density: bool = True,
):
    """Compute the density of each channel in a data points array.

    Parameters
    ----------
    data_points : np.ndarray
        The data points to compute the density of.
    bins : int
        The number of bins to use for the histogram.
    density : bool, optional
        Whether to normalize the histogram, by default True.

    Returns
    -------
    np.ndarray
        The density of each channel.
    """
    range = [(0, 1) for _ in bins]
    density, _ = np.histogramdd(
        data_points,
        bins=bins,
        range=range,
        density=density,
    )
    return density


def channels_vector_to_density_wheel(
    channel_vector: np.ndarray, bins: np.ndarray, wheel_diameter: int
):
    """Create a wheel of a given radius and size.

    Parameters
    ----------
    channel_vector : np.ndarray
        The channel vector to create the wheel from.
        Must be of shape (N, channel).
    bins : np.ndarray
        The number of bins to use for the histogram.
    wheel_diameter : int
        The diameter of the wheel to create.

    Returns
    -------
    np.ndarray
        The wheel of the given diameter.
    """

    # compute density of each channel
    density = get_channel_density(channel_vector, bins=bins, density=True)
    flatten_density = density.flatten()

    # create a list of index coordinate for each bin
    angles = np.linspace(0, 1, density.shape[0])
    radii = np.linspace(0, 1, density.shape[1])
    angles, radii = np.array([(x, y) for x in angles for y in radii]).T

    # convert angles and radii to wheel coordinates
    pos = get_2D_wheel_coordinate(angles, radii) * (wheel_diameter - 1)
    pos = pos.astype(int)

    # create empty wheel figure
    wheel = np.zeros((wheel_diameter, wheel_diameter))

    # populate wheel by summing the density of each bin
    # at the corresponding coordinates
    np.add.at(wheel, (pos[0], pos[1]), flatten_density)

    return wheel


def channels_vector_to_density_triangle(
    channel_vector: np.ndarray, size: int, log_scale: bool
):
    """Create a density triangle of a given size.

    Parameters
    ----------
    channel_vector : np.ndarray
        The channel vector to create the wheel from.
        Must be of shape (N, channel).
    size : int
        The side length of the triangle.
    log_scale: bool
        Manual computation of a normed density for visualization.
        If true, the density values are passed through a logarithm.

    Returns
    -------
    dict
        The triangle of given side length; a dictionary
        (as represented with ternary's simplex_iterator object)
    """

    # create empty triangle figure
    density_triangle = maxwell_hue_empty(size=size)

    # RGB to Barycentric
    # standardize eaach channel to [0,1]
    channel_vector = channel_vector - np.min(
        channel_vector, axis=0, keepdims=True
    )
    channel_vector = channel_vector / np.max(
        channel_vector, axis=0, keepdims=True
    )

    # divide each (r,g,b) by (r+g+b) so that r+g+b = 1
    channel_vector = (
        size * channel_vector / np.sum(channel_vector, axis=1, keepdims=True)
    ).astype(int)
    # correct rounding errors (changing last coordinate s.t. r+g+b = size)
    channel_vector[:, 2] = size - np.sum(channel_vector[:, :2], axis=1)

    # populate triangular density by summing the density of each bin
    # at the corresponding barycentric coordinates
    for (r, g, b) in channel_vector:
        density_triangle[(r, g, b)] += 1
    if log_scale:
        density_triangle = {k: np.log(v) for k, v in density_triangle.items()}
    else:
        density_triangle = {k: v for k, v in density_triangle.items()}
    return density_triangle


def channels_vector_to_density_spherical(
    channel_vector: np.ndarray, bins: np.ndarray
):
    """Create a density array of given size (size = bins x bins).

    Parameters
    ----------
    channel_vector : np.ndarray
        The channel vector to create the density array from.
        Must be of shape (N, channel).
    bins : np.ndarray
        The number of bins to use for the histogram.

    Returns
    -------
    np.ndarray
        The histogram with given number of bins on each side of the plot
    """

    # create empty wheel figure
    density_arr = np.zeros((bins + 1, bins + 1))

    # RGB to Barycentric
    # standardize eaach channel to [0,1]
    channel_vector = channel_vector - np.min(
        channel_vector, axis=0, keepdims=True
    )
    channel_vector = channel_vector / np.max(
        channel_vector, axis=0, keepdims=True
    )

    # populate wheel by summing the density of each bin
    # at the corresponding coordinates
    for (r, g, b) in channel_vector:
        theta, phi = round(2 * bins * (np.arctan(b / r)) / np.pi), round(
            2 * bins * (np.arctan(g / np.sqrt(r**2 + b**2))) / np.pi
        )
        density_arr[theta, phi] += 1
    return density_arr


class DensityFigure(FigureCanvas):
    def __init__(
        self,
        image: np.ndarray,
        density_resolution: list = [3600, 1000],
        figure_size: int = 100,
        channels_ranges: np.ndarray = None,
        channel_axis: int = 1,
        log_scale: bool = True,
        cmap: str = "gray",
        value_threshold: float = 0.0,
        color_space: str = "HSV",
    ):
        self.fig = Figure(figsize=(figure_size, figure_size))
        super().__init__(self.fig)
        self.figure_size = figure_size
        self.density_resolution = density_resolution
        self.channels_ranges = channels_ranges
        self.channel_axis = channel_axis
        self.log_scale = log_scale
        self.cmap = cmap
        self.value_threshold = value_threshold
        self.color_space = color_space

        self.create_color_wheel(self.color_space)
        self.image = image
        self.density_barycentric = None
        self.density_spherical = None

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value
        self.update_density()  # callback to update figure

    @property
    def density_wheel(self):
        return self._density_wheel

    @density_wheel.setter
    def density_wheel(self, value):
        self._density_wheel = value
        self.update_figure()  # callback to update figure

    @property
    def selection_mask(self) -> np.ndarray:
        """Returns the selection mask of the density wheel."""
        if not hasattr(self, "_selection_mask"):
            self.init_selection_mask()
        return self._selection_mask

    @selection_mask.setter
    def selection_mask(self, val: np.ndarray):
        self._selection_mask = val

    def init_selection_mask(self):
        """Initializes the selection mask of the density wheel.
        with all zeros."""
        self.selection_mask = np.zeros((self.figure_size, self.figure_size))
        return self.selection_mask

    def update_density(self):
        vector_points = image_to_flat_data_points(
            self.image, self.channel_axis
        )
        if self.color_space == "HSV":
            hsv = rgb_to_hsv(vector_points, channel_axis=1)
            hsv = hsv[hsv[..., 2] > self.value_threshold]
            hs_points = hsv[:, 0:2]
            self.density_wheel = channels_vector_to_density_wheel(
                hs_points, [360, 100], self.figure_size
            )
        elif self.color_space == "Barycentric":
            self.density_wheel = (
                channels_vector_to_density_triangle(
                    vector_points, self.figure_size, self.log_scale
                )
                if self.density_barycentric is None
                else self.density_barycentric
            )

        elif self.color_space == "Spherical":
            self.density_wheel = (
                channels_vector_to_density_spherical(
                    vector_points, self.figure_size
                )
                if self.density_spherical is None
                else self.density_spherical
            )

    def create_color_wheel(self, color_space):
        if color_space == "HSV":
            self.color_wheel = hue_saturation_color_wheel(self.figure_size)
        elif color_space == "Barycentric":
            self.color_wheel = maxwell_hue_space(self.figure_size)
        elif color_space == "Spherical":
            self.color_wheel = spherical_coordinates_color_wheel(
                self.figure_size
            )

    def update_figure(self):
        """Populates the density figure with the density and color wheel."""
        if self.color_space == "HSV":
            self.fig.clear()
            self.fig.set_facecolor("#262930")  # match napari background color

            norm = None
            if self.log_scale:
                norm = colors.LogNorm()

            # copy colormap and set bad values to black
            mycmap = copy.copy(matplotlib.cm.get_cmap(self.cmap))
            mycmap.set_bad(mycmap(0))

            # create colormap for selection mask
            mask_cmap = colors.LinearSegmentedColormap.from_list(
                "colormap", [[0, 0, 0, 0], [0, 0, 0]]
            )
            mask_cmap = colors.LinearSegmentedColormap.from_list(
                "colormap", [[0, 0, 0, 1], [0, 0, 0, 0]]
            )

            # density plot
            density_ax = self.fig.add_subplot(121)
            density_ax.set_facecolor(
                "#262930"
            )  # match napari background color
            density_ax.set_title("hue/saturation density", color="white")
            self.msk_density_wheel = density_ax.imshow(
                self.density_wheel,
                interpolation="nearest",
                norm=norm,
                cmap=mycmap,
            )
            density_ax.set_xlim([0, self.figure_size - 1])
            density_ax.set_ylim([0, self.figure_size - 1])
            density_ax.set_aspect("equal")
            density_ax.set_axis_off()

            # color wheel plot
            color_wheel_ax = self.fig.add_subplot(122)
            color_wheel_ax.set_facecolor(
                "#262930"
            )  # match napari background color
            color_wheel_ax.set_title("hue/saturation wheel", color="white")
            self.msk_color_wheel = color_wheel_ax.imshow(
                self.color_wheel,
            )

            self.msk_selection_mask = color_wheel_ax.imshow(
                self.selection_mask,
                vmax=1,
                alpha=0.5,
                cmap=mask_cmap,
            )
            color_wheel_ax.set_xlim([0, self.figure_size - 1])
            color_wheel_ax.set_ylim([0, self.figure_size - 1])
            color_wheel_ax.set_axis_off()

            # lasso selectors
            self.lasso_density = LassoSelector(
                density_ax, self.onselect, button=1
            )
            self.lasso_color_wheel = LassoSelector(
                color_wheel_ax, self.onselect, button=1
            )
            self.fig.canvas.draw_idle()

        elif self.color_space == "Barycentric":
            self.fig.clear()
            self.fig.set_facecolor("#262930")  # match napari background color

            # copy colormap and set bad values to black
            mycmap = copy.copy(matplotlib.cm.get_cmap(self.cmap))
            mycmap.set_bad(mycmap(0))

            # create colormap for selection mask
            mask_cmap = colors.LinearSegmentedColormap.from_list(
                "colormap", [[0, 0, 0, 0], [0, 0, 0]]
            )
            mask_cmap = colors.LinearSegmentedColormap.from_list(
                "colormap", [[0, 0, 0, 1], [0, 0, 0, 0]]
            )

            # density plot
            density_ax = self.fig.add_subplot(121)
            density_ax.set_facecolor(
                "#262930"
            )  # match napari background color
            density_ax.set_title("barycentric density", color="white")
            _, tax = ternary.figure(ax=density_ax, scale=self.figure_size)

            self.msk_density_wheel = tax.heatmap(
                self.density_wheel,
                style="hexagonal",
                use_rgba=False,
                colorbar=False,
                cmap=mycmap,
            )
            density_ax.set_xlim([0, self.figure_size - 1])
            density_ax.set_ylim([0, self.figure_size - 1])
            density_ax.set_aspect("equal")
            density_ax.set_axis_off()

            # color wheel plot
            color_triangle_ax = self.fig.add_subplot(122)
            _, tax = ternary.figure(
                ax=color_triangle_ax, scale=self.figure_size
            )
            color_triangle_ax.set_facecolor(
                "#262930"
            )  # match napari background color
            color_triangle_ax.set_title("barycentric triangle", color="white")
            self.msk_color_wheel = tax.heatmap(
                self.color_wheel,
                style="hexagonal",
                use_rgba=True,
                colorbar=False,
            )

            color_triangle_ax.set_xlim([0, self.figure_size - 1])
            color_triangle_ax.set_ylim([0, self.figure_size - 1])
            color_triangle_ax.set_aspect("equal")
            color_triangle_ax.set_axis_off()

            tax.clear_matplotlib_ticks()
            tax.get_axes().axis("off")

            # lasso selectors
            self.lasso_density = LassoSelector(
                density_ax, self.onselect, button=1
            )
            self.lasso_color_triangle = LassoSelector(
                color_triangle_ax, self.onselect, button=1
            )
            self.fig.canvas.draw_idle()

        elif self.color_space == "Spherical":
            self.fig.clear()
            self.fig.set_facecolor("#262930")  # match napari background color

            norm = None
            if self.log_scale:
                norm = colors.LogNorm()

            # copy colormap and set bad values to black
            mycmap = copy.copy(matplotlib.cm.get_cmap(self.cmap))
            mycmap.set_bad(mycmap(0))

            # create colormap for selection mask
            mask_cmap = colors.LinearSegmentedColormap.from_list(
                "colormap", [[0, 0, 0, 0], [0, 0, 0]]
            )
            mask_cmap = colors.LinearSegmentedColormap.from_list(
                "colormap", [[0, 0, 0, 1], [0, 0, 0, 0]]
            )

            # density plot
            density_ax = self.fig.add_subplot(121)
            density_ax.set_facecolor(
                "#262930"
            )  # match napari background color
            density_ax.set_title(
                "azimuth vs. polar angle densities", color="white"
            )
            self.msk_density_wheel = density_ax.imshow(
                self.density_wheel,
                interpolation="nearest",
                norm=norm,
                cmap=mycmap,
            )
            density_ax.set_xlim([0, self.figure_size - 1])
            density_ax.set_ylim([0, self.figure_size - 1])
            density_ax.set_aspect("equal")
            density_ax.set_axis_off()

            # color wheel plot
            color_wheel_ax = self.fig.add_subplot(122)
            color_wheel_ax.set_facecolor(
                "#262930"
            )  # match napari background color
            color_wheel_ax.set_title(
                "azimuth vs. polar angle colormap", color="white"
            )
            self.msk_color_wheel = color_wheel_ax.imshow(
                self.color_wheel,
            )

            self.msk_selection_mask = color_wheel_ax.imshow(
                self.selection_mask,
                vmax=1,
                alpha=0.5,
                cmap=mask_cmap,
            )
            # color_wheel_ax.invert_yaxis()
            color_wheel_ax.set_xlim([0, self.figure_size - 1])
            color_wheel_ax.set_ylim([0, self.figure_size - 1])
            color_wheel_ax.set_axis_off()

            # lasso selectors
            self.lasso_density = LassoSelector(
                density_ax, self.onselect, button=1
            )
            self.lasso_color_wheel = LassoSelector(
                color_wheel_ax, self.onselect, button=1
            )
            self.fig.canvas.draw_idle()

    def update_cmap(self, cmap: str):
        """Updates the colormap of the density wheel figure."""
        self.cmap = cmap
        self.update_figure()

    def update_log_scale(self, log_scale: bool):
        """Updates the log scale of the density wheel figure."""
        self.log_scale = log_scale

        norm = None
        if self.log_scale:
            norm = colors.LogNorm()

        if self.color_space in ["HSV", "Spherical"]:
            self.msk_density_wheel.set_norm(norm)
        elif self.color_space == "Barycentric":
            self.update_density()
        self.fig.canvas.draw_idle()

    def update_color_space(self, color_space: str):
        """Updates the color space of the density figure"""
        self.color_space = color_space
        if color_space == "HSV":
            self.color_wheel = hue_saturation_color_wheel(self.figure_size)
        elif color_space == "Barycentric":
            self.color_wheel = maxwell_hue_space(self.figure_size)
        elif color_space == "Spherical":
            self.color_wheel = spherical_coordinates_color_wheel(
                self.figure_size
            )
        self.update_density()

    def update_density_figure_parameters(
        self,
        figure_size: int,
        density_log_scale: bool,
        cmap: str,
        value_threshold: float,
    ):
        """Updates the density figure parameters."""
        self.figure_size = figure_size
        self.log_scale = density_log_scale
        self.cmap = cmap
        self.value_threshold = value_threshold

        self.update_color_space(color_space=self.color_space)
        self.update_lasso_pixel()

    def update_lasso_pixel(self):
        # Pixel coordinates for the density wheel
        pix = np.arange(self.figure_size)
        xv, yv = np.meshgrid(pix, pix)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T
        self.init_selection_mask()

    def update_selection_mask(self, array, indices):
        """Updates the selection mask with the indices of
        the selected pixels."""
        lin = np.arange(array.size)
        newArray = array.flatten()
        newArray[lin[indices]] = 1
        return newArray.reshape(array.shape)

    def onselect(self, verts):
        """Callback function for the lasso selector."""
        if len(verts) == 2:
            self.init_selection_mask()
        else:
            p = path.Path(verts)
            ind = p.contains_points(self.pix, radius=1)
            self.selection_mask = self.update_selection_mask(
                self.selection_mask, ind
            )

        self.msk_selection_mask.set_data(self.selection_mask)
        self.fig.canvas.draw_idle()

    def update_mask_on_wheel(self, image, mask):
        mask = image_selection_to_wheel_selection(
            image, mask, self.figure_size
        )
        mask = mask > 0
        self.msk_selection_mask.set_data(mask)
        self.selection_mask = mask
        self.fig.canvas.draw_idle()
