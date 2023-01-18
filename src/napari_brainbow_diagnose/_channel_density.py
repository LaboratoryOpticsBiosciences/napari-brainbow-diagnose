import copy
import math
from typing import TYPE_CHECKING

import matplotlib.cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import path
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.widgets import LassoSelector
from qtpy.QtWidgets import QHBoxLayout, QWidget
from skimage.color import hsv2rgb, rgb2hsv

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

    if not channels_ranges:
        channels_ranges = get_channels_ranges(data_points)

    hsv = rgb2hsv(data_points)
    # keep only hue and saturation as value is independant in brainbow
    hs = hsv[:, :-1]

    histogram = density(hs, size)
    hs_density_wheel = hue_saturation_density_wheel(histogram, size)

    return hs_density_wheel


class DensityFigure:
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
        size : int
            The size of the density wheel and color wheel.
        channels_ranges : np.ndarray, optional
            The range of values for each channel, defaults to None.
        channel_index : int, optional
            The index of the channel, defaults to 1.
        """
        self.image = image
        self.size = size
        self.log_scale = log_scale
        self.init_selection_mask()
        self.density_wheel = create_density_wheel(
            image, self.size, channels_ranges, channel_index
        )
        self.color_wheel = hue_saturation_color_wheel(self.size)

        # create Qt widget with density and color wheel
        self.create_density_figure(log_scale, cmap)

        # Pixel coordinates
        pix = np.arange(self.size)
        xv, yv = np.meshgrid(pix, pix)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T
        # placeholder to draw lasso selection

        # add interactivity to select pixel in both subplot
        # i don't understand why it doesn't when defined here
        LassoSelector(self.mpl_widget.figure.axes[0], self.onselect, button=1)
        LassoSelector(self.mpl_widget.figure.axes[0], self.onselect, button=1)

    @property
    def density_wheel(self) -> np.ndarray:
        return self._density_wheel

    @density_wheel.setter
    def density_wheel(self, val: np.ndarray):
        self._density_wheel = val

    @property
    def selection_mask(self) -> np.ndarray:
        if not hasattr(self, "_selection_mask"):
            self.init_selection_mask()
        return self._selection_mask

    @selection_mask.setter
    def selection_mask(self, val: np.ndarray):
        self._selection_mask = val

    def init_selection_mask(self):
        self.selection_mask = np.zeros((self.size, self.size))
        return self.selection_mask

    def create_density_figure(
        self, log_scale: bool = True, cmap: str = "gray"
    ):
        self.fig = plt.figure()
        self.mpl_widget = FigureCanvas(self.fig)  # Qt widget to give to napari

        self.fig.set_facecolor("#262930")  # match napari background color

        norm = None
        if log_scale:
            norm = colors.LogNorm()

        mycmap = copy.copy(matplotlib.cm.get_cmap(cmap))
        mycmap.set_bad(mycmap(0))

        mask_cmap = colors.LinearSegmentedColormap.from_list(
            "colormap", [[0, 0, 0, 0], [0, 0, 0]]
        )
        mask_cmap = colors.LinearSegmentedColormap.from_list(
            "colormap", [[0, 0, 0, 1], [0, 0, 0, 0]]
        )

        # density plot
        self.density_ax = self.fig.add_subplot(121)
        self.density_ax.set_facecolor(
            "#262930"
        )  # match napari background color
        self.density_ax.set_title("hue/saturation density", color="white")
        self.msk_density_wheel = self.density_ax.imshow(
            self.density_wheel,
            origin="lower",
            # vmax=1,
            interpolation="nearest",
            norm=norm,
            cmap=mycmap,
        )
        self.density_ax.set_xlim([0, self.size - 1])
        self.density_ax.set_ylim([0, self.size - 1])
        self.density_ax.set_aspect("equal")
        self.density_ax.set_axis_off()
        # color wheel plot
        self.color_wheel_ax = self.fig.add_subplot(122)
        self.color_wheel_ax.set_facecolor(
            "#262930"
        )  # match napari background color
        self.color_wheel_ax.set_title("hue/saturation wheel", color="white")
        self.msk_color_wheel = self.color_wheel_ax.imshow(
            self.color_wheel, origin="lower", vmax=1, interpolation="nearest"
        )
        self.msk_selection_mask = self.color_wheel_ax.imshow(
            self.selection_mask,
            origin="lower",
            vmax=1,
            interpolation="nearest",
            alpha=0.5,
            cmap=mask_cmap,
        )
        self.color_wheel_ax.set_xlim([0, self.size - 1])
        self.color_wheel_ax.set_ylim([0, self.size - 1])
        self.color_wheel_ax.set_axis_off()

        return self.mpl_widget

    def update_selection_mask(self, array, indices):
        lin = np.arange(array.size)
        newArray = array.flatten()
        newArray[lin[indices]] = 1
        return newArray.reshape(array.shape)

    def onselect(self, verts):
        if len(verts) == 2:
            self.init_selection_mask()
        else:
            p = path.Path(verts)
            ind = p.contains_points(self.pix, radius=1)
            self.selection_mask = self.update_selection_mask(
                self.selection_mask, ind
            )

        self.msk_selection_mask.set_data(self.selection_mask)
        # self.msk_selection_mask.set_data(self.selection_mask)
        self.fig.canvas.draw_idle()


class DensityWidget(QWidget):
    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer

        # widget for density figure
        image = self.viewer.layers[0].data
        density_figure = DensityFigure(image, 100, log_scale=True, cmap="gray")

        LassoSelector(
            density_figure.density_ax, density_figure.onselect, button=1
        )
        LassoSelector(
            density_figure.color_wheel_ax, density_figure.onselect, button=1
        )

        # napari_viewer.window.add_dock_widget(density_figure.mpl_widget)
        print("hello")
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(density_figure.mpl_widget)
