"""Canonical legend labels for the method column in shd.csv/kl_div.csv/time.csv.

The same baseline is written with different casing/wording by different
producer scripts (estimate_cstree.py, shd.py, shd_dag.py, kl_div.py, and the
shd_to_true_* rules in the Snakefile), so plots built from different CSVs show
the same method under different legend text. This module normalizes that at
plot time without touching the already-committed CSVs or the scripts that
produced them.
"""

import re

_BASE_LABELS = {
    "pc": "PC",
    "pc+dag": "PC",
    "grasp": "GRaSP",
    "grasp+dag": "GRaSP",
    "cslearn+pc": "CSlearn+PC",
    "cslearn+grasp": "CSlearn+GRaSP",
    "bos": "BOS",
    "bhc": "GRaSP+BHC",
    "grasp+bhc": "GRaSP+BHC",
}

_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<suffix>mle|map)\)$", re.IGNORECASE)

# Phase-1-only methods with no CSlearn context-specific refinement -- shown in
# every figure as a fixed reference, never as "the paper's own method". Not
# distinguished by a "baseline" word in the label (that cost legend width
# every figure needed back once legends moved beside/within narrower
# columns); the distinction is carried by the method name itself plus the
# surrounding prose.
BASELINE_METHODS = {"PC", "GRaSP"}


def baseline_method_label(method: str, mle_suffix: bool = False) -> str:
    """Canonical legend label for a bare method name.

    `mle_suffix` states a baseline's estimator explicitly (" (MLE)") --
    only meaningful when the figure/legend also shows a real MLE-vs-MAP split
    for some other method (kl_divergence_2a/2b), where the baseline appears
    unchanged in both, so it must say which estimator that unchanging value
    actually is; otherwise it reads as "we also computed a MAP version of the
    baseline", which is false.
    """
    if method in BASELINE_METHODS:
        suffix = " (MLE)" if mle_suffix else ""
        return f"{method}{suffix}"
    return method


def canonical_method_label(raw: str) -> str:
    """Map a raw method string to its canonical legend label.

    Recognizes an optional trailing "(mle)"/"(map)" suffix (case-insensitive)
    and reattaches it as " (MLE)"/" (MAP)" on the canonical base label.
    """
    match = _SUFFIX_RE.match(raw.strip())
    base, suffix = (match["base"], match["suffix"].upper()) if match else (raw.strip(), None)

    try:
        canonical = _BASE_LABELS[base.lower()]
    except KeyError:
        raise ValueError(f"Unrecognized method label: {raw!r}") from None

    return f"{canonical} ({suffix})" if suffix else canonical
