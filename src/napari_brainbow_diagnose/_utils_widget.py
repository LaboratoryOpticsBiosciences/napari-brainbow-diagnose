from typing import TYPE_CHECKING

from magicgui import magic_factory
from napari.layers import Image, Labels
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
    call_button="Get wheel mask applied to image",
)
def wheel_mask_to_image_mask():
    ...


@magic_factory(
    call_button="Get image mask applied to wheel",
    selection_mask={"label": "Select selection layer"},
)
def image_mask_to_wheel(selection_mask: Labels):
    ...


@magic_factory(
    auto_call=True,
    call_button="Compute brainbow image density",
    density_resolution={
        "label": "Resolution for hue and saturation wheel figure",
    },
    value_threshold={
        "label": "Minimum voxel Value in HSV space for density calculation",
        "widget_type": "FloatSlider",
        "min": 0,
        "max": 1,
    },
)
def density_resolution_widget(
    density_resolution: int = 100,
    value_threshold: float = 0,
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


def layers_event_callback_connector(
    layers_events,
    layer_dropdown,
):
    """Connects a callback function to a layer dropdown widget.
    The callback function is called when the list of layers changes.
    """
    layers_events.inserted.connect(layer_dropdown.reset_choices)
    layers_events.removed.connect(layer_dropdown.reset_choices)
    layers_events.reordered.connect(layer_dropdown.reset_choices)
    return
