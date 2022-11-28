# napari-brainbow-diagnose

[![License BSD-3](https://img.shields.io/pypi/l/napari-brainbow-diagnose.svg?color=green)](https://github.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-brainbow-diagnose.svg?color=green)](https://pypi.org/project/napari-brainbow-diagnose)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-brainbow-diagnose.svg?color=green)](https://python.org)
[![tests](https://github.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/workflows/tests/badge.svg)](https://github.com/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/actions)
[![codecov](https://codecov.io/gh/LaboratoryOpticsBiosciences/napari-brainbow-diagnose/branch/main/graph/badge.svg)](https://codecov.io/gh/LaboratoryOpticsBiosciences/napari-brainbow-diagnose)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-brainbow-diagnose)](https://napari-hub.org/plugins/napari-brainbow-diagnose)

Visualize and Diagnose brainbow dataset in color space.

## Pitch
Brainbow dataset have unique features that need to be addressed by specialized tools. This plugin aims at visualize and diagnose brainbow dataset.
In particular we want to interact with the distribution of the dataset in the channel space.

## Dev Installation

```bash
conda create -n napari-brainbow-diagnose python=3.9
conda activate napari-brainbow-diagnose
conda install pip
git clone git@github.com:LaboratoryOpticsBiosciences/napari-brainbow-diagnose.git
cd napari-brainbow-diagnose
pip install -e .[testing]
pre-commit install
```

Do not forget to run pytest `pytest .` before you commit to check your code.

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
