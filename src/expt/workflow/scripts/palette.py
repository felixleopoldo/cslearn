"""Shared color/hatch/sizing logic for the boxplot scripts.

Colors are assigned per method (not per hue slot), so the same method always
reads as the same color family across every figure it appears in. Within a
method, n is encoded as a light-to-dark shade (more data = darker). Hatching
is reserved for the one place it carries real meaning: MAP vs MLE parameter
estimates in the KL plots.
"""

import re

import matplotlib.colors as mcolors
import seaborn as sns

METHOD_COLORS = {
    "CSlearn+PC": "#009E73",
    "PC": "#D55E00",
    "CSlearn+GRaSP": "#0072B2",
    "GRaSP": "#E69F00",
    "BOS": "#CC79A7",
    "GRaSP+BHC": "#4D4D4D",
}

# Base colors for the sensitivity plots' beta-misspecification arms.
BETA_COLORS = {
    1: "#0072B2",  # over-spec
    2: "#4D4D4D",  # correct
    3: "#D55E00",  # under-spec
}

MAP_HATCH = "///"

_HUE_RE = re.compile(r"^(?P<method>.+?)(?: \((?P<suffix>MLE|MAP)\))?, n=(?P<n>\d+)$")


def _shades(base_hex, count):
    """`count` shades of base_hex, from a light tint (smallest group) to the
    full base color (largest group)."""
    base = mcolors.to_rgb(base_hex)
    white = (1.0, 1.0, 1.0)
    if count == 1:
        return [base]
    min_t = 0.35
    ts = [min_t + (1 - min_t) * i / (count - 1) for i in range(count)]
    return [tuple((1 - t) * w + t * b for w, b in zip(white, base)) for t in ts]


def sorted_hues(hues):
    """Sort hue labels by method (in METHOD_COLORS order), then n ascending,
    then MLE before MAP."""

    def key(h):
        m = _HUE_RE.match(h)
        method, suffix, n = m["method"], m["suffix"] or "", int(m["n"])
        method_rank = list(METHOD_COLORS).index(method) if method in METHOD_COLORS else len(METHOD_COLORS)
        suffix_rank = {"": 0, "MLE": 0, "MAP": 1}[suffix]
        return (method_rank, n, suffix_rank)

    return sorted(hues, key=key)


def method_hue_style(hues):
    """Palette + hatch dicts for hue labels of the form
    "<method>[ (MLE|MAP)], n=<n>": color by method (shaded by n), hatch only
    to mark MAP vs MLE.
    """
    parsed = {h: _HUE_RE.match(h).groupdict() for h in hues}
    ns_by_method = {}
    for p in parsed.values():
        ns_by_method.setdefault(p["method"], set()).add(int(p["n"]))

    palette, hatches = {}, {}
    for method, ns in ns_by_method.items():
        shade_map = dict(zip(sorted(ns), _shades(METHOD_COLORS[method], len(ns))))
        for h, p in parsed.items():
            if p["method"] != method:
                continue
            palette[h] = shade_map[int(p["n"])]
            hatches[h] = MAP_HATCH if p["suffix"] == "MAP" else ""
    return palette, hatches


def beta_hue_style(label_map, ns):
    """Palette dict for the sensitivity plots' beta-misspecification hues:
    color by beta arm (shaded by n); no hatching needed since color alone
    already distinguishes the 3 arms."""
    palette = {}
    for beta, label in label_map.items():
        shade_map = dict(zip(sorted(ns), _shades(BETA_COLORS[beta], len(ns))))
        for n, color in shade_map.items():
            palette[f"{label}, n={n}"] = color
    return palette


def style_figure(g, n_x, n_hue):
    """Common figure sizing/legend styling so boxes have room to breathe and
    the legend doesn't crowd the plot area.

    Sizes the axes panel from n_x/n_hue alone; the legend is placed outside
    the axes and its width is left to `savefig(..., bbox_inches="tight")`
    (see the caller), since legend label length varies a lot between figures
    (e.g. the sensitivity plots' beta labels) and a fixed guess either wastes
    space or clips the legend.
    """
    axes_width = max(6.0, 0.28 * n_x * n_hue + 1.0)
    g.figure.set_size_inches(axes_width, 5.5)
    g.legend_.set_title("")
    g.legend_.set_frame_on(False)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 1))
