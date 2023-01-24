from typing import TYPE_CHECKING

from magicgui import magic_factory

if TYPE_CHECKING:
    import napari


@magic_factory
def channel_space_widget(img_layer: "napari.layers.Image"):
    """
    User interface to show a cube with each pixel projected according to
    its intensity in each channel.
    """
    print(f"you have selected {img_layer}")
