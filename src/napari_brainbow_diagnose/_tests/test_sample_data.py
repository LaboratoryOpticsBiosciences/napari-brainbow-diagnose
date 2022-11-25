from napari_brainbow_diagnose import make_rgb_cube_data


def test_make_rgb_cube_data(make_napari_viewer, capsys):
    viewer = make_napari_viewer()

    # this time, our widget will be a MagicFactory or FunctionGui instance
    loader = make_rgb_cube_data()

    # if we "call" this object, it'll execute our function
    loader(viewer)

    # read captured output and check that it's as we expected
    assert len(viewer.layers) == 3
