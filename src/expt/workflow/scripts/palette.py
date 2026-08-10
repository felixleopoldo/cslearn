"""Shared color/sizing logic for the boxplot scripts.

Grayscale printing only preserves lightness, so lightness is spent on the
dimension that actually needs to survive print: method identity (the
comparison every one of these figures makes). Each method has one fixed base
color, chosen so its luma is well separated from every other method's, used
everywhere that method appears across every figure.

n is encoded as a shade of that same color: the method's base color is used
unmodified for the *largest* n (most data = most saturated/confident), and
lightens toward (not reaching) white as n shrinks. The lightening amount is
capped per-method, per-figure, at the base luma of the next-lighter method
that's actually present in that hue set (with a safety margin) -- so a
method's shaded range never drifts into a neighboring method's territory and
they stay distinguishable in true grayscale. The method with the lightest
base color in a given figure has no lighter neighbor to avoid, so it's capped
only by staying short of pure white (which would be invisible on paper).
"""

import re

import matplotlib.colors as mcolors
import seaborn as sns

METHOD_COLORS = {
    "CSlearn+PC": "#f4d36e",
    "PC": "#ff7f40",
    "CSlearn+GRaSP": "#32a588",
    "GRaSP": "#8c3f72",
    "BOS": "#193f65",
    "GRaSP+BHC": "#190f22",
}

# Base colors for the sensitivity plots' beta-misspecification arms.
BETA_COLORS = {
    1: "#66b2ff",  # over-spec
    2: "#737373",  # correct
    3: "#50230c",  # under-spec
}

WHITE_LUMA_CAP = 0.93
LUMA_MARGIN = 0.03

_HUE_RE = re.compile(r"^(?P<method>.+?), n=(?P<n>\d+)$")


def _luma(hexcolor):
    r, g, b = mcolors.to_rgb(hexcolor)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _light_caps(present_colors):
    """For each key in `present_colors` (method -> base hex), the max luma
    its shades may lighten to: the next-lighter present color's base luma
    (minus a margin), or WHITE_LUMA_CAP for the lightest one present."""
    ordered = sorted(present_colors, key=lambda k: _luma(present_colors[k]))
    caps = {}
    for i, key in enumerate(ordered):
        if i + 1 < len(ordered):
            caps[key] = _luma(present_colors[ordered[i + 1]]) - LUMA_MARGIN
        else:
            caps[key] = WHITE_LUMA_CAP
    return caps


def _shades(base_hex, count, light_cap):
    """`count` shades of base_hex, from the base color itself (darkest,
    largest n) lightening toward `light_cap` (lightest, smallest n)."""
    base = mcolors.to_rgb(base_hex)
    white = (1.0, 1.0, 1.0)
    base_luma = _luma(base_hex)
    t_max = 0.0 if base_luma >= 1 else max(0.0, min(1.0, (light_cap - base_luma) / (1 - base_luma)))
    shades = []
    for i in range(count):
        t = t_max * (count - 1 - i) / (count - 1) if count > 1 else 0.0
        shades.append(tuple((1 - t) * b + t * w for b, w in zip(base, white)))
    return shades


def sorted_hues(hues):
    """Sort hue labels by method (in METHOD_COLORS order), then n ascending."""

    def key(h):
        m = _HUE_RE.match(h)
        method, n = m["method"], int(m["n"])
        method_rank = list(METHOD_COLORS).index(method) if method in METHOD_COLORS else len(METHOD_COLORS)
        return (method_rank, n)

    return sorted(hues, key=key)


def method_hue_style(hues):
    """Palette dict for hue labels of the form "<method>, n=<n>": color by
    method (fixed base), shaded by n-rank within the luma headroom actually
    available among the methods present in `hues`."""
    parsed = {h: _HUE_RE.match(h).groupdict() for h in hues}
    ns_by_method = {}
    for p in parsed.values():
        ns_by_method.setdefault(p["method"], set()).add(int(p["n"]))

    present_colors = {m: METHOD_COLORS[m] for m in ns_by_method}
    caps = _light_caps(present_colors)

    palette = {}
    for method, ns in ns_by_method.items():
        n_rank = {n: i for i, n in enumerate(sorted(ns))}
        shades = _shades(METHOD_COLORS[method], len(ns), caps[method])
        for h, p in parsed.items():
            if p["method"] != method:
                continue
            palette[h] = shades[n_rank[int(p["n"])]]
    return palette


def beta_hue_style(label_map, ns):
    """Palette dict for the sensitivity plots' beta-misspecification hues:
    color by beta arm (fixed base), shaded by n-rank."""
    caps = _light_caps(BETA_COLORS)
    palette = {}
    for beta, label in label_map.items():
        shades = _shades(BETA_COLORS[beta], len(ns), caps[beta])
        for i, n in enumerate(sorted(ns)):
            palette[f"{label}, n={n}"] = shades[i]
    return palette


def style_figure(g, width=2.9, height=None, ncol=1, n_hue=None):
    """Common figure sizing/legend styling. Pins the axes close to the
    manuscript's actual print width (~2.75in for a 0.475\\linewidth
    two-up subfigure) rather than growing with hue count, since fonts are
    fixed in points and don't rescale with the canvas: a wider source PDF
    just gets shrunk more by LaTeX, making text smaller, not bigger.

    The legend goes below the axes in a single column by default: multiple
    columns made the legend wider than the (narrow) axes, which made
    `bbox_inches="tight"` blow the whole saved PDF out past the intended
    print width -- the opposite of the point of pinning `width` above. A
    single column is guaranteed to stay within the axes width regardless of
    label length, at the cost of a taller figure, which `height` accounts
    for based on how many legend rows there'll be.
    """
    if height is None:
        height = 2.3 + 0.16 * (n_hue or 1) * ncol
    g.figure.set_size_inches(width, height)
    g.legend_.set_title("")
    g.legend_.set_frame_on(False)
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.18), ncol=ncol)
