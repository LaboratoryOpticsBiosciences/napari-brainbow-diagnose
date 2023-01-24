import copy
import math
from typing import TYPE_CHECKING

import matplotlib.cm
import matplotlib.colors as colors
import numpy as np
from matplotlib import path
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from qtpy.QtWidgets import QVBoxLayout, QWidget
from skimage.color import hsv2rgb, rgb2hsv

from ._utils_widget import (
    brainbow_layers_selector,
    create_selection_mask,
    density_figure_parameters,
    density_resolution_widget,
)

if TYPE_CHECKING:
    pass


def flatten_data_points(a: np.ndarray, channel_index: int) -> np.ndarray:
    """
    Flattens a numpy array by moving the specified channel index to
    the first axis and flattening the array.

    Parameters
    ----------
    a : np.ndarray
        The input numpy array to be flattened.
    channel_index : int
        The index of the channel to be moved to the first axis.

    Returns
    -------
    np.ndarray
        The flattened (n*channel) array.
    """
    channel_first = np.moveaxis(a, channel_index, 0)
    flat_data_points = channel_first.reshape(channel_first.shape[0], -1).T
    return flat_data_points


def hue_saturation_to_wheel_position(h, s, size):
    """ """
    rx = s * np.cos(h * 2 * np.pi)
    ry = s * np.sin(h * 2 * np.pi)
    pos = (size / 2) * np.array([1 + rx, 1 + ry]) - 1
    return pos.astype(int)


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


def density(a: np.ndarray, bins: int) -> np.ndarray:
    """
    Returns the density of the input array using histogramdd.

    Parameters
    ----------
    a : np.ndarray
        The input array to calculate density for.
    bins : int
        The number of bins to use in the histogramdd calculation.

    Returns
    -------
    np.ndarray
        The density of the input array.
    """
    density, _ = np.histogramdd(
        a, bins=bins, range=((0, 1), (0, 1)), density=True
    )
    return density


def hue_saturation_density_wheel(
    hue_saturation_array: np.ndarray, size: int = 50
) -> np.ndarray:
    """
    Returns a placeholder array with the density of hue and saturation values.

    Parameters
    ----------
    hue_saturation_array : np.ndarray
        The array of hue and saturation values.
    size : int, optional
        The size of the placeholder array, defaults to 50.

    Returns
    -------
    np.ndarray
        The placeholder array with the density of hue and saturation values.
    """
    placeholder = np.zeros((size, size))
    radius = size / 2.0
    cx, cy = size / 2, size / 2
    for x in range(size):
        for y in range(size):
            rx = x - cx
            ry = y - cy
            s = (rx**2.0 + ry**2.0) ** 0.5 / radius
            h = ((math.atan2(ry, rx) / math.pi) + 1.0) / 2.0
            if s <= 1:
                h = int(h * size - 1)
                s = int(s * size - 1)
                placeholder[x, y] = hue_saturation_array[h, s]
    return placeholder


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
                rgb = hsv2rgb([h, s, 1]) * 255
                wheel[x, y] = rgb

    return wheel


def create_density_wheel(
    image: np.ndarray,
    size: int,
    channels_ranges: np.ndarray = None,
    channel_index: int = 1,
) -> np.ndarray:
    """
    Creates a density wheel of an image by flattening the data points,
    converting them to hsv, keeping only the hue and saturation values,
    and creating a density wheel image of those values.

    Parameters
    ----------
    image : np.ndarray
        The input image to create the density wheel for.
    size : int
        The size of the density wheel.
    channels_ranges : np.ndarray, optional
        The range of values for each channel, defaults to None.
    channel_index : int, optional
        The index of the channel, defaults to 1.

    Returns
    -------
    np.ndarray
        The density wheel of the input image.
    """
    data_points = flatten_data_points(image, channel_index)

    if not channels_ranges:  # TODO CLIP Channel range?
        channels_ranges = get_channels_ranges(data_points)

    hsv = rgb2hsv(data_points)
    # keep only hue and saturation as value is independant in brainbow
    hs = hsv[:, :-1]

    histogram = density(hs, size)
    hs_density_wheel = hue_saturation_density_wheel(histogram, size)

    return hs_density_wheel


class DensityFigure(FigureCanvas):
    def __init__(
        self,
        image: np.ndarray,
        size: int = 100,
        channels_ranges: np.ndarray = None,
        channel_index: int = 1,
        log_scale: bool = True,
        cmap: str = "gray",
    ):
        """
        Creates an instance of DensityFigure with a density wheel and color
        wheel of the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image to create the density wheel and color wheel
        density_resolution : int
            The resolution of the density wheel and thus color wheel.
        channels_ranges : np.ndarray, optional
            The range of values for each channel, defaults to None.
        channel_index : int, optional
            The index of the channel, defaults to 1.
        log_scale : bool, optional
            Whether to use a log scale for the density wheel, defaults to True.
        cmap : str, optional
            The colormap to use for the density wheel, defaults to "gray".
        """
        self.fig = Figure(figsize=(size, size))
        super().__init__(self.fig)
        self.image = image
        self.density_resolution = size
        self.channels_ranges = channels_ranges
        self.channel_index = channel_index
        self.log_scale = log_scale
        self.cmap = cmap

        self.update_brainbow_image(self.image)

    def update_density_parameters(
        self, density_resolution, density_log_scale, cmap
    ):
        """Updates the density parameters of the density wheel figure."""
        self.density_resolution = density_resolution
        self.log_scale = density_log_scale
        self.cmap = cmap
        self.fig.canvas.draw_idle()

    def update_brainbow_image(self, brainbow_image: np.ndarray):
        """Updates the brainbow image of the density wheel figure."""
        self.init_selection_mask()
        self.image = brainbow_image
        self.density_wheel = create_density_wheel(
            self.image,
            self.density_resolution,
            self.channels_ranges,
            self.channel_index,
        )

        # Pixel coordinates for the density wheel
        pix = np.arange(self.density_resolution)
        xv, yv = np.meshgrid(pix, pix)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        # Create the color wheel figure
        self.color_wheel = hue_saturation_color_wheel(self.density_resolution)

        self.populate_density_figure()

    def update_cmap(self, cmap: str):
        """Updates the colormap of the density wheel figure."""
        self.cmap = cmap
        self.populate_density_figure()

    def update_log_scale(self, log_scale: bool):
        """Updates the log scale of the density wheel figure."""
        self.log_scale = log_scale

        norm = None
        if self.log_scale:
            norm = colors.LogNorm()

        self.msk_density_wheel.set_norm(norm)
        self.fig.canvas.draw_idle()

    @property
    def density_wheel(self) -> np.ndarray:
        return self._density_wheel

    @density_wheel.setter
    def density_wheel(self, val: np.ndarray):
        """Sets the density wheel of the density wheel figure."""
        self._density_wheel = val

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
        self.selection_mask = np.zeros(
            (self.density_resolution, self.density_resolution)
        )
        return self.selection_mask

    def populate_density_figure(self):
        """Populates the density figure with the density and color wheel."""
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
        density_ax.set_facecolor("#262930")  # match napari background color
        density_ax.set_title("hue/saturation density", color="white")
        self.msk_density_wheel = density_ax.imshow(
            self.density_wheel,
            interpolation="nearest",
            norm=norm,
            cmap=mycmap,
        )
        density_ax.set_xlim([0, self.density_resolution - 1])
        density_ax.set_ylim([0, self.density_resolution - 1])
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
        color_wheel_ax.set_xlim([0, self.density_resolution - 1])
        color_wheel_ax.set_ylim([0, self.density_resolution - 1])
        color_wheel_ax.set_axis_off()

        # lasso selectors
        self.lasso_density = LassoSelector(density_ax, self.onselect, button=1)
        self.lasso_color_wheel = LassoSelector(
            color_wheel_ax, self.onselect, button=1
        )

        self.fig.canvas.draw_idle()

    def update_density_figure_parameters(
        self, density_resolution: int, density_log_scale: bool, cmap: str
    ):
        """Updates the density figure parameters."""
        self.density_resolution = density_resolution
        self.log_scale = density_log_scale
        self.cmap = cmap

        self.color_wheel = hue_saturation_color_wheel(self.density_resolution)

        self.update_brainbow_image(self.image)

        self.populate_density_figure()
        self.fig.canvas.draw_idle()

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


class DensityWidget(QWidget):
    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer

        img = (
            self.empty_brainbow_image()
        )  # should be (C, Z, Y, X) or (C, Y, X)
        self.density_figure = DensityFigure(
            img, 100, log_scale=True, cmap="gray", channel_index=0
        )

        self.brainbow_layers_selector = brainbow_layers_selector()
        self.density_resolution_widget = density_resolution_widget()
        self.density_figure_parameters = density_figure_parameters()
        self.selection_mask_creator = create_selection_mask()
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.brainbow_layers_selector.native)
        self.layout().addWidget(self.density_resolution_widget.native)
        self.layout().addWidget(self.density_figure)
        self.layout().addWidget(self.density_figure_parameters.native)
        self.layout().addWidget(self.selection_mask_creator.native)

        self.density_resolution_widget.call_button.clicked.connect(
            self.update_brainbow_image
        )
        self.density_figure_parameters.cmap.changed.connect(
            self.update_cmap_density
        )
        self.density_figure_parameters.density_log_scale.changed.connect(
            self.update_log_density
        )
        self.density_figure_parameters.density_log_scale.changed.connect(
            self.update_log_density
        )
        self.selection_mask_creator.call_button.clicked.connect(
            self.update_selection_mask
        )

    def update_selection_mask(self):
        img = self.get_brainbow_image_from_layers()
        # img = np.moveaxis(img, 0, -1) # set channel last
        hsv = rgb2hsv(img, channel_axis=0)
        hs = hsv[:-1]
        resolution = self.density_resolution_widget.density_resolution.value
        wheel_pos = hue_saturation_to_wheel_position(hs[0], hs[1], resolution)
        points = flatten_data_points(wheel_pos, 0)

        # must be inversed because of the way the color wheel is plotted
        mask_corrected = self.density_figure.selection_mask[::-1, ::-1]

        points_selected = mask_corrected[(points[:, 0], points[:, 1])]

        selection_mask = points_selected.reshape(img.shape[1:])

        selection_mask = selection_mask.astype(bool)

        self.viewer.add_labels(selection_mask, name="selection_mask")

    def update_brainbow_image(self):
        density_resolution = (
            self.density_resolution_widget.density_resolution.value
        )
        density_log_scale = (
            self.density_figure_parameters.density_log_scale.value
        )
        cmap = self.density_figure_parameters.cmap.value
        self.density_figure.update_density_figure_parameters(
            density_resolution, density_log_scale, cmap
        )
        self.density_figure.update_brainbow_image(
            self.get_brainbow_image_from_layers()
        )

    def update_cmap_density(self):
        cmap = self.density_figure_parameters.cmap.value
        self.density_figure.update_cmap(cmap)

    def update_log_density(self):
        log_scale = self.density_figure_parameters.density_log_scale.value
        self.density_figure.update_log_scale(log_scale)

    def empty_brainbow_image(self):
        """Returns an empty brainbow image. With shape (3, 1, 1, 1)
        corresponding to (C, Z, Y, X)"""
        return np.random.random((3, 1, 1, 1))

    def get_brainbow_image_from_layers(self):
        red_layer = self.brainbow_layers_selector.red_layer
        green_layer = self.brainbow_layers_selector.green_layer
        blue_layer = self.brainbow_layers_selector.blue_layer
        if red_layer is None or green_layer is None or blue_layer is None:
            return self.empty_brainbow_image()
        else:
            return np.array(
                [
                    red_layer.value.data,
                    green_layer.value.data,
                    blue_layer.value.data,
                ]
            )
