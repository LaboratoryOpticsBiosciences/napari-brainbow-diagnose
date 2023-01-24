from typing import TYPE_CHECKING

from magicgui import magic_factory
from napari.layers import Image
from napari.utils.colormaps import matplotlib_colormaps

if TYPE_CHECKING:
    pass


@magic_factory(
    auto_call=True,
    cmap={"choices": matplotlib_colormaps.keys()},
)
def density_figure_parameters(
    density_log_scale: bool = True,
    cmap: str = "gray",
):
    """Parameters for the density figure."""
    ...


@magic_factory(
    call_button="Create mask from selected highlighted zones",
)
def create_selection_mask():
    ...


@magic_factory(
    auto_call=True,
    call_button="Compute brainbow image density",
    density_resolution={
        "label": "Resolution for hue and saturation wheel figure"
    },
)
def density_resolution_widget(
    density_resolution: int = 100,
):
    """Parameters for density histogram figure."""
    ...


@magic_factory(
    auto_call=True,
    red_layer={"label": "Select Red layer"},
    green_layer={"label": "Select Green layer"},
    blue_layer={"label": "Select Blue layer"},
)
def brainbow_layers_selector(
    red_layer: Image,
    green_layer: Image,
    blue_layer: Image,
):
    """Select the layers to be used for the brainbow image."""
    ...
