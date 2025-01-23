try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._read_csv_widget import add_features_widget, read_csv_widget
from ._sample_data import (
    fetch_chroms_data,
    load_chroms_data_sample,
    make_rgb_cube_data,
)
from ._tooltip_pointer import tooltip_pointer_widget
from ._widget import (
    DiagnoseWidget,
    compute_all_channel_space,
    create_rgb_features_widget,
)

__all__ = (
    "tooltip_pointer_widget",
    "make_rgb_cube_data",
    "fetch_chroms_data",
    "load_chroms_data_sample",
    "DiagnoseWidget",
    "create_rgb_features_widget",
    "compute_all_channel_space",
    "read_csv_widget",
    "add_features_widget",
)
