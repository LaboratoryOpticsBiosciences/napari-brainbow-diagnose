from typing import TYPE_CHECKING, Dict

import numpy as np
from magicgui import magic_factory

if TYPE_CHECKING:
    import napari

FLOAT_ROUND_DECIMAL = 2  # float number is too long to display


@magic_factory(call_button="run")
def tooltip_pointer_widget(
    img_layer: "napari.layers.Image",
    label_layer: "napari.layers.Labels",
):
    """
    User interface to show a tooltip on the footprint selected.
    """
    print(f"you have selected {img_layer}")
    print(f"you have selected {label_layer}")
    # for i in np.unique(label_layer.data)[1:]:
    from skimage.measure import regionprops_table as sk_regionprops_table

    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    properties = ["label", "area", "intensity_mean"]
    image = img_layer.data
    labels = label_layer.data
    table: Dict[str, np.ndarray] = sk_regionprops_table(
        labels,
        intensity_image=image,
        properties=properties,
        # extra_properties=extra_properties  # list[callbacks]
    )
    # (tmp) store the raw `table` output from regionprop
    label_layer._metadata = table
    # Round floating numbers, because float64 is too long too display and harms
    # readability
    for k, v in table.items():
        if v.dtype == "float":
            table[k] = v.round(FLOAT_ROUND_DECIMAL)
    # Append dummy 0 label
    for k in table.keys():
        table[k] = np.insert(table[k], 0, 0)
    # Set `features` property. It will be copied to `properties` property too
    # and displayed in the tooltip.
    label_layer.features = table
