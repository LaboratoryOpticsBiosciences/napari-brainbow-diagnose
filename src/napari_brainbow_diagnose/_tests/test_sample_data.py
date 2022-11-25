from napari_brainbow_diagnose import make_rgb_cube_data


def test_make_rgb_cube_data(make_napari_viewer, capsys):

    # call the sample
    layers = make_rgb_cube_data()

    # read captured output and check that it's as we expected
    assert len(layers) == 3
