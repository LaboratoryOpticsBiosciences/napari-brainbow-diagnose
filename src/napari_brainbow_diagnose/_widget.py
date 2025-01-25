# Object to create a widget for the plugin to be displayed in napari
# with all the necessary buttons and sliders of the different subwidgets.

import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ._density import DensityFigure
from ._utils_channel_space import (
    get_2D_wheel_coordinate,
    image_mask_of_wheel_selection,
    rgb_to_hsv,
    rgb_to_maxwell_triangle,
    rgb_to_spherical,
)
from ._utils_image import get_mean_intensity_ellipse
from ._utils_io import empty_brainbow_image, get_brainbow_image_from_layers
from ._utils_widget import (
    brainbow_layers_selector,
    density_figure_parameters,
    density_resolution_widget,
    image_mask_to_wheel,
    layers_event_callback_connector,
    wheel_mask_to_image_mask,
)


def create_rgb_features_widget(
    red_layer: "napari.layers.Image",
    green_layer: "napari.layers.Image",
    blue_layer: "napari.layers.Image",
    point_layer: "napari.layers.Points",
    ellipse_radius: str = "1,1,1",
    compute_channel_space: bool = True,
) -> None:
    # check if the radius is a list of integers
    try:
        ellipse_radius = [int(r) for r in ellipse_radius.split(",") if r != ""]
    except ValueError:
        raise ValueError(
            "The radius should be a list of integers separated by commas."
        )

    assert len(ellipse_radius) == red_layer.data.ndim, (
        "The number of dimensions in the radius should be the same as the "
        "number of dimensions in the image."
    )

    means = []
    means.append(
        get_mean_intensity_ellipse(
            red_layer.data, point_layer.data, radius=ellipse_radius
        )
    )
    means.append(
        get_mean_intensity_ellipse(
            green_layer.data, point_layer.data, radius=ellipse_radius
        )
    )
    means.append(
        get_mean_intensity_ellipse(
            blue_layer.data, point_layer.data, radius=ellipse_radius
        )
    )
    means = np.array(means).T

    point_layer.features = pd.DataFrame(means, columns=["R", "G", "B"])
    point_layer.features["SELECTED_CLUSTER"] = 0

    def update_manual_selection_cluster(selected):
        # reset selected cluster in features
        point_layer.features["SELECTED_CLUSTER"] = 0
        point_layer.features.loc[selected, "SELECTED_CLUSTER"] = 1

    point_layer.selected_data.events.items_changed.connect(
        update_manual_selection_cluster
    )

    if compute_channel_space:
        compute_all_channel_space(point_layer=point_layer)


def compute_all_channel_space(
    point_layer: "napari.layers.Points",
) -> None:

    # check that point_layer features has the RGB columns
    if not all([c in point_layer.features.columns for c in ["R", "G", "B"]]):
        raise ValueError(
            "The point layer should have the columns 'R', 'G', 'B'."
        )

    rgb = point_layer.features[["R", "G", "B"]].values
    hsv = rgb_to_hsv(rgb)

    wheel_pos = get_2D_wheel_coordinate(hsv[:, 0], hsv[:, 1])
    wheel_x, wheel_y = wheel_pos[0], wheel_pos[1]

    maxwell_x, maxwell_y = rgb_to_maxwell_triangle(
        rgb[:, 0], rgb[:, 1], rgb[:, 2]
    )
    radius, theta, phi = rgb_to_spherical(rgb[:, 0], rgb[:, 1], rgb[:, 2])

    point_layer.features["H"] = hsv[:, 0]
    point_layer.features["S"] = hsv[:, 1]
    point_layer.features["V"] = hsv[:, 2]
    point_layer.features["wheel_x"] = wheel_x
    point_layer.features["wheel_y"] = wheel_y
    point_layer.features["X_maxwell"] = maxwell_x
    point_layer.features["Y_maxwell"] = maxwell_y
    point_layer.features["spherical_radius"] = radius
    point_layer.features["spherical_theta"] = theta
    point_layer.features["spherical_phi"] = phi


class DiagnoseWidget(QWidget):
    def __init__(self, napari_viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer

        # Create default data
        self.empty_data()

        # Brainbow loader widget
        self.brainbow_layers_selector = brainbow_layers_selector()

        # Parameter density Widget
        self.density_resolution_widget = density_resolution_widget()

        # Wheel Widget
        self.density_figure = DensityFigure(
            self.brainbow_image, channel_axis=0
        )

        # Wheel parameters Widget
        self.density_figure_parameters = density_figure_parameters()

        # Mask selection on wheel Widget
        self.wheel_mask_to_image_mask = wheel_mask_to_image_mask()

        # Mask selection on image Widget
        self.image_mask_to_wheel = image_mask_to_wheel()

        # Tooltip Widget

        # Create layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.brainbow_layers_selector.native)
        self.layout().addWidget(self.density_resolution_widget.native)
        self.layout().addWidget(self.density_figure)
        self.layout().addWidget(self.density_figure_parameters.native)
        self.layout().addWidget(self.wheel_mask_to_image_mask.native)
        self.layout().addWidget(self.image_mask_to_wheel.native)

        # Create callback functions
        layers_events = self.viewer.window._qt_viewer.viewer.layers.events

        layers_event_callback_connector(
            layers_events, self.brainbow_layers_selector
        )
        layers_event_callback_connector(
            layers_events, self.image_mask_to_wheel
        )

        self.density_resolution_widget.call_button.clicked.connect(
            self.update_density_wheel
        )
        self.wheel_mask_to_image_mask.call_button.clicked.connect(
            self.create_mask_on_image
        )
        self.image_mask_to_wheel.call_button.clicked.connect(
            self.create_mask_on_wheel
        )

        self.density_figure_parameters.color_space.changed.connect(
            self.update_color_space
        )
        self.density_figure_parameters.cmap.changed.connect(
            self.update_cmap_density
        )
        self.density_figure_parameters.density_log_scale.changed.connect(
            self.update_log_density
        )

    def create_mask_on_image(self):
        channels = get_brainbow_image_from_layers(
            self.brainbow_layers_selector.red_layer.value,
            self.brainbow_layers_selector.green_layer.value,
            self.brainbow_layers_selector.blue_layer.value,
        )

        # must be inversed because of the way the color wheel is plotted
        mask_corrected = self.density_figure.selection_mask[::-1, ::-1].astype(
            bool
        )
        mask_corrected = self.density_figure.selection_mask.astype(bool)

        value_threshold = self.density_resolution_widget.value_threshold.value

        mask_on_image = image_mask_of_wheel_selection(
            channels, mask_corrected, value_threshold
        )

        self.viewer.add_labels(mask_on_image, name="mask_on_image")

    def create_mask_on_wheel(self):
        mask = self.image_mask_to_wheel.selection_mask.value.data
        image = get_brainbow_image_from_layers(
            self.brainbow_layers_selector.red_layer.value,
            self.brainbow_layers_selector.green_layer.value,
            self.brainbow_layers_selector.blue_layer.value,
        )
        self.density_figure.update_mask_on_wheel(image, mask)

    def empty_data(self):
        self.brainbow_image = empty_brainbow_image()

    def update_density_wheel(self):
        density_figure_resolution = (
            self.density_resolution_widget.density_resolution.value
        )
        value_threshold = self.density_resolution_widget.value_threshold.value
        density_log_scale = (
            self.density_figure_parameters.density_log_scale.value
        )
        cmap = self.density_figure_parameters.cmap.value
        self.density_figure.update_density_figure_parameters(
            density_figure_resolution, density_log_scale, cmap, value_threshold
        )
        self.density_figure.image = get_brainbow_image_from_layers(
            self.brainbow_layers_selector.red_layer.value,
            self.brainbow_layers_selector.green_layer.value,
            self.brainbow_layers_selector.blue_layer.value,
        )

    def update_log_density(self):
        log_scale = self.density_figure_parameters.density_log_scale.value
        self.density_figure.update_log_scale(log_scale)

    def update_cmap_density(self):
        cmap = self.density_figure_parameters.cmap.value
        self.density_figure.update_cmap(cmap)

    def update_color_space(self):
        color_space = self.density_figure_parameters.color_space.value
        self.density_figure.update_color_space(color_space)
