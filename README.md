# napari-brainbow-diagnose

[![License BSD-3](https://img.shields.io/pypi/l/napari-brainbow-diagnose.svg?color=green)](https://github.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-brainbow-diagnose.svg?color=green)](https://pypi.org/project/napari-brainbow-diagnose)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-brainbow-diagnose.svg?color=green)](https://python.org)
[![tests](https://github.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/workflows/tests/badge.svg)](https://github.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/actions)
[![codecov](https://codecov.io/gh/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/branch/main/graph/badge.svg)](https://codecov.io/gh/LaboratoryOpticsBiosciences/napari-brainbow-diagnose)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-brainbow-diagnose)](https://napari-hub.org/plugins/napari-brainbow-diagnose)

Explore image in channel coordinate space.
Brainbow dataset have unique features that need to be addressed by specialized tools. This plugin aims at visualize and diagnose brainbow dataset.
In particular we want to interact with the distribution in the channel space. This plugin allows you to visualize the distribution of the channel ratio in the image and to select pixel in the image and see where they are in the channel coordinate space. You can also use this plugin along with the [`napari-cluster-plotter` plugin](https://github.com/BiAPoL/napari-clusters-plotter?tab=readme-ov-file#installation) plugin to visualize the distribution of the channel ratio of every point object in the image.

![demo_gif](https://raw.githubusercontent.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/main/docs/demo_napari-brainbow-diagnose.gif)

## Available Channel space transformation

The following channel spaces are available:

- Cartesian RGB
- Hue-Saturation-Value planes [illustration (b)(f)](https://en.wikipedia.org/wiki/File:Hsl-hsv_models.svg)
- Hue-Saturation wheel [illustration (g)](https://en.wikipedia.org/wiki/File:Hsl-hsv_models.svg)
- Maxwell triangle (ternary plot) [illustration](https://en.wikipedia.org/wiki/Ternary_plot)
- Spherical coordinates (Theta, Phi and Radius) [illustration](https://en.wikipedia.org/wiki/Spherical_coordinate_system)

## Example Notebook

You can use this plugin to visualize channel space of !
- every voxel in the image (see [demo notebook](docs/demo.ipynb))
- every object (aka center point) in the image (see [demo notebook](docs/cluster_plotter_compatibility.ipynb)). To use this notebook you need to install [`napari-cluster-plotter` plugin](https://github.com/BiAPoL/napari-clusters-plotter?tab=readme-ov-file#installation).
Find all menus under `Plugins > napari-brainbow-diagnose > Diagnose Brainbow Image`

## Example Datasets

If you want to use your dataset, you have to format it such as each channel is in one distinct `napari.Layers`
You can open test dataset to try this plugin in `File > Open Sample > napari-brainbow-diagnose`.

- The RGB Cube is an array with shape (3x256x256x256) cube : Great to check how the plugin work when all color are represented
- ChroMS Cortex Sample is an array with shape (3x256x256x256) #Hugo : Real life brainbow image (Cortex E18 Emx1Cre) !

Once you have your layers you can use the dropdown and select the corresponding layer. It is advised to match the `red, green, blue` order so the ratio you see on the napari viewer corresponds to the Hue-Saturation Wheel of the plugin.

## Example using every voxel in the image

### Get Channel Ratio Density of the image

When you click on `Compute brainbow image density` you will populate the Hue-Saturation density Wheel.
This should allow you to quickly see which ratio is more present in your image. You can see the corresponding ratio according to the "HS Color wheel" on the right.
For example here on this screenshot we can see that:

- there is a high number of non saturated red-only ratio. (2)
- there is not a high number of non saturated magenta ratio. (3)

![ratio](https://raw.githubusercontent.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/main/docs/ratio_view.png)

### Create a selection of ratio on the channel coordinate system and apply it on the original image


![ratio](https://raw.githubusercontent.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/main/docs/wheel_to_image_selection.gif)

### Create a selection of pixel in the image and show where they are in the channel coordinate system

![ratio](https://raw.githubusercontent.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/main/docs/image_to_wheel_selection.gif)


## Installation

You can install `napari-brainbow-diagnose` via [pip]:

    pip install napari-brainbow-diagnose



To install latest development version :

    pip install git+https://github.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-brainbow-diagnose" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->
