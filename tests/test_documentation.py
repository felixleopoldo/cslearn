"""Execute the ``python`` code blocks in ``README.md`` so the quick-start example
stays runnable.

The package's own ``>>>`` docstring examples are covered separately by
``test_docstrings.py``.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

_BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)


@pytest.mark.documentation
@pytest.mark.parametrize("code", _BLOCKS, ids=[f"readme_block_{i}" for i in range(len(_BLOCKS))])
def test_readme_python_block_runs(code):
    exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})


def test_readme_has_a_python_block():
    assert _BLOCKS, "README.md has no ```python code block to test"
