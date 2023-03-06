# Object to create a widget for the plugin to be displayed in napari
# with all the necessary buttons and sliders of the different subwidgets.

from qtpy.QtWidgets import QVBoxLayout, QWidget

from ._density import DensityFigure
from ._utils_channel_space import image_mask_of_wheel_selection
from ._utils_io import empty_brainbow_image, get_brainbow_image_from_layers
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
            self.update_density_wheel
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
        mask_on_image = image_mask_of_wheel_selection(channels, mask_corrected)

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
