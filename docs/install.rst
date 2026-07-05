============
Installation
============

System dependency
-----------------

``cstrees`` uses `pygraphviz <https://pygraphviz.github.io/>`_ for graph
visualisation, which requires graphviz to be installed on your system before
installing the package.

On Debian/Ubuntu::

    $ sudo apt install graphviz libgraphviz-dev pkg-config

On macOS (Homebrew)::

    $ brew install graphviz

On Windows, install graphviz from https://graphviz.org/download/ and ensure
it is on your ``PATH``.

Package installation
--------------------

Install ``cstrees`` from PyPI::

    $ pip install cstrees

Development installation
------------------------

The development environment uses `devenv <https://devenv.sh>`_, which also
installs `uv <https://docs.astral.sh/uv/>`_. With devenv installed, run from
the repository root::

    $ devenv shell

This activates the environment and runs ``uv sync --all-extras``, installing
all dependencies including the optional experiment extras.
