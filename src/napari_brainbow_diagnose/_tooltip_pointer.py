from typing import TYPE_CHECKING

from magicgui import magic_factory

if TYPE_CHECKING:
    import napari


@magic_factory
def tooltip_pointer_widget(img_layer: "napari.layers.Image"):
    """
    User interface to show a tooltip on the footprint selected.
    """
    print(f"you have selected {img_layer}")
