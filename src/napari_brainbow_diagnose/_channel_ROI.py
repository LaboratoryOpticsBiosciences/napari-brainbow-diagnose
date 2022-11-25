from typing import TYPE_CHECKING

from magicgui import magic_factory

if TYPE_CHECKING:
    import napari


@magic_factory
def channel_ROI_widget(img_layer: "napari.layers.Image"):
    """
    User interface to select a region of interest in the channel space.
    """
    print(f"you have selected {img_layer}")
