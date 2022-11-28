from typing import TYPE_CHECKING

from magicgui import magic_factory
from skimage.color import rgb2hsv
from typing import List
import numpy as np
import numpy as np

if TYPE_CHECKING:
    import napari


@magic_factory(
    call_button="Filter selected layers",
)
def channel_ROI_widget(
    red_channel: "napari.layers.Image",
    green_channel: "napari.layers.Image",
    blue_channel: "napari.layers.Image",
    r: int=255,
    g: int=255,
    b: int=0,
    radius_angle: float=0.2,
    ) -> "napari.types.LayerDataTuple":
    """
    User interface to select a region of interest in the channel space.
    """
    rgb_img = np.moveaxis(np.array([red_channel.data, green_channel.data, blue_channel.data]), 0, -1)
    rgb_vector = np.array([r, g, b])

    # for each pixel in rgb_img get its angle with rgb_vector
    # flatten the array except color channel
    flat_rgb = rgb_img.reshape((-1, rgb_img.shape[-1]))
    # normalize vector and image
    flat_rgb = np.divide(flat_rgb, np.linalg.norm(flat_rgb, axis=1)[:, np.newaxis])
    rgb_vector = rgb_vector / np.linalg.norm(rgb_vector)

    # get angle between the vector and the image
    angle = np.arccos(np.clip(np.dot(flat_rgb, rgb_vector),  -1.0, 1.0))
    
    # reshape the image
    angle = angle.reshape(red_channel.data.shape)

    # create cone mask
    cone = np.zeros(angle.shape)
    cone[angle<radius_angle] = 1

    return (
            cone,
            {
                'name':'bb_red_filtered',
                'blending': 'additive',
            },
            'image'
        )