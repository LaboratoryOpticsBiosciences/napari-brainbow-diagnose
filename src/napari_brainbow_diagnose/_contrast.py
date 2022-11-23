from typing import TYPE_CHECKING

from magicgui import magic_factory

if TYPE_CHECKING:
    import napari


@magic_factory
def contrast_widget(img_layer: "napari.layers.Image"):
    """
    User interface to show the contrast andintensity histogram of
    the img_layer
    """
    print(f"you have selected {img_layer}")
