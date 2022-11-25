import numpy as np

from napari_brainbow_diagnose import (
    channel_ROI_widget,
    channel_space_widget,
    contrast_widget,
    tooltip_pointer_widget,
)


def test_channel_ROI_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = channel_ROI_widget()

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0])

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == f"you have selected {layer}\n"


def test_contrast_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = contrast_widget()

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0])

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == f"you have selected {layer}\n"


def test_channel_space_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = channel_space_widget()

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0])

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == f"you have selected {layer}\n"


def test_tooltip_pointer_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = tooltip_pointer_widget()

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0])

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == f"you have selected {layer}\n"
