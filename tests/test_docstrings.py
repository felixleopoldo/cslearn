"""Run the ``>>>`` examples in the package docstrings as doctests.

``pytest --doctest-modules src/cslearn`` does the same thing; this module makes
the docstring examples part of the default ``pytest tests/`` run as well.
"""

import doctest
import importlib

import pytest

MODULES = [
    "cslearn.cstree",
    "cslearn.dependence",
    "cslearn.learning",
    "cslearn.scoring",
    "cslearn.stage",
    "cslearn.ldag",
    "cslearn.evaluate",
    "cslearn.examp_datasets",
]


@pytest.mark.documentation
@pytest.mark.parametrize("module_name", MODULES)
def test_docstring_examples(module_name):
    module = importlib.import_module(module_name)
    result = doctest.testmod(module, verbose=False, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0, f"{result.failed} doctest failure(s) in {module_name}"
