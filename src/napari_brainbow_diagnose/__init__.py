try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._channel_ROI import channel_ROI_widget
from ._channel_space import channel_space_widget
from ._contrast import contrast_widget
from ._reader import napari_get_reader
from ._sample_data import make_rgb_cube_data
from ._widget import ExampleQWidget, example_magic_widget
from ._writer import write_multiple, write_single_image

__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "make_rgb_cube_data",
    "ExampleQWidget",
    "example_magic_widget",
    "contrast_widget",
    "channel_space_widget",
    "channel_ROI_widget",
)
