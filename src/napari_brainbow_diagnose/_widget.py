# Object to create a widget for the plugin to be displayed in napari
# with all the necessary buttons and sliders of the different subwidgets.

import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ._density import DensityFigure
from ._utils_channel_space import image_mask_of_wheel_selection
from ._utils_widget import (
    brainbow_layers_selector,
    density_figure_parameters,
    density_resolution_widget,
    image_mask_to_wheel,
    layers_event_callback_connector,
    wheel_mask_to_image_mask,
)


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
            self.update_brainbow_image
        )
        self.wheel_mask_to_image_mask.call_button.clicked.connect(
            self.create_mask_on_image
        )
        self.image_mask_to_wheel.call_button.clicked.connect(
            self.create_mask_on_wheel
        )

        self.density_figure_parameters.cmap.changed.connect(
            self.update_cmap_density
        )
        self.density_figure_parameters.density_log_scale.changed.connect(
            self.update_log_density
        )

    def create_mask_on_image(self):
        channels = self.get_brainbow_image_from_layers()

        # must be inversed because of the way the color wheel is plotted
        mask_corrected = self.density_figure.selection_mask[::-1, ::-1].astype(
            bool
        )
        mask_corrected = self.density_figure.selection_mask.astype(bool)
        mask_on_image = image_mask_of_wheel_selection(channels, mask_corrected)

        self.viewer.add_labels(mask_on_image, name="mask_on_image")

    def create_mask_on_wheel(self):
        mask = self.image_mask_to_wheel.selection_mask.value.data
        image = self.get_brainbow_image_from_layers()
        self.density_figure.update_mask_on_wheel(image, mask)

    def empty_data(self):
        self.brainbow_image = self.empty_brainbow_image()

    def empty_brainbow_image(self):
        """Returns an empty brainbow image. With shape (3, 1, 1, 1)
        corresponding to (C, Z, Y, X)"""
        return np.random.random((3, 2, 2, 2))

    def update_brainbow_image(self):
        density_resolution = (
            self.density_resolution_widget.density_resolution.value
        )
        density_log_scale = (
            self.density_figure_parameters.density_log_scale.value
        )
        cmap = self.density_figure_parameters.cmap.value
        self.density_figure.update_density_figure_parameters(
            density_resolution, density_log_scale, cmap
        )
        self.density_figure.image = self.get_brainbow_image_from_layers()

    def get_brainbow_image_from_layers(self):
        red_layer = self.brainbow_layers_selector.red_layer
        green_layer = self.brainbow_layers_selector.green_layer
        blue_layer = self.brainbow_layers_selector.blue_layer
        if red_layer is None or green_layer is None or blue_layer is None:
            return self.empty_brainbow_image()
        else:
            return np.array(
                [
                    red_layer.value.data,
                    green_layer.value.data,
                    blue_layer.value.data,
                ]
            )

    def update_log_density(self):
        log_scale = self.density_figure_parameters.density_log_scale.value
        self.density_figure.update_log_scale(log_scale)

    def update_cmap_density(self):
        cmap = self.density_figure_parameters.cmap.value
        self.density_figure.update_cmap(cmap)
