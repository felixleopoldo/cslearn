import re

import pandas as pd
from labels import canonical_method_label
from palette import (
    METHOD_COLORS,
    METHOD_MARKERS,
    add_shared_legend,
    draw_lines,
    grow_to_fit,
    legend_height_in,
    linestyles_for,
    reserve_legend_margin,
    set_plot_style,
    sorted_methods,
    square_grid,
)

PANEL_SIZE = 1.59  # 4 panels x 1.59in = 6.36in, near the ceiling \textwidth (6.396431in) allows
HUE_NCOL = 4

_SUFFIX_RE = re.compile(r"^(?P<base>.+) \((?P<suffix>MLE|MAP)\)$")


def split_suffix(label):
    m = _SUFFIX_RE.match(label)
    return (m["base"], m["suffix"]) if m else (label, None)


sources = [("PC", snakemake.input.a), ("GRaSP", snakemake.input.b)]

# One panel per (phase-1 method, estimator) combination, 1x4. Methods with
# no MAP variant (the plain PC/GRaSP baselines) act as a fixed reference and
# appear in both the MLE and MAP panel of their column.
panels = []
all_methods, all_ns = set(), set()
for phase1, path in sources:
    df = pd.read_csv(path)
    df["method"] = df["method"].map(canonical_method_label)
    df["n"] = df["n"].astype(int)
    df["_base_method"], df["_param"] = zip(*df["method"].map(split_suffix))
    all_methods.update(df["_base_method"].unique())
    all_ns.update(df["n"].unique())

    methods_with_map = set(df.loc[df["_param"] == "MAP", "_base_method"].unique())
    for param in ["MLE", "MAP"]:
        subset = df[(df["_param"] == param) | (~df["_base_method"].isin(methods_with_map))]
        panels.append((f"{phase1}, {param}", subset))

linestyle_map = linestyles_for(all_ns)
hue_entries = [(m, METHOD_COLORS[m], METHOD_MARKERS[m]) for m in sorted_methods(all_methods)]
style_entries = [(f"n={n}", linestyle_map[n]) for n in sorted(all_ns)]

set_plot_style()
fig, axes = square_grid(1, 4, panel_size=PANEL_SIZE)

# Letter each panel (a-d): with no per-panel LaTeX subfigure, this is how
# manuscript prose points at one specific panel.
for letter, ax, (title, subset) in zip("abcd", axes.flat, panels):
    draw_lines(ax, subset, "p", "kl_div", "_base_method", "n", METHOD_COLORS, METHOD_MARKERS, linestyle_map)
    ax.set_yscale("log")
    ax.set_title(f"({letter}) {title}")

fig.supxlabel("number of variables ($p$)")
# set_ylabel on just the leftmost panel, not fig.supylabel: matplotlib
# centers a per-axes ylabel on that axes' own box natively, whereas
# supylabel centers on the whole figure canvas -- with a legend margin
# reserved above, those aren't the same point, so supylabel read as
# vertically offset from the actual panels.
axes[0].set_ylabel("KL-divergence")

# grow_to_fit before reserving legend margin, so panels grow into unused
# space first rather than reserving margin against an already-starved row.
grow_to_fit(fig, axes[0])

# Margin sized from the legend's actual content (row counts) and the row's
# now-final (post-growth) height, not a guess -- an earlier guessed
# fraction left the legend overlapping the panels.
legend_h = legend_height_in(len(hue_entries), HUE_NCOL, len(style_entries))
frac = legend_h / (fig.get_figheight() + legend_h)
reserve_legend_margin(fig, "above", frac)

add_shared_legend(fig, hue_entries, style_entries, loc="above", frac=frac, hue_ncol=HUE_NCOL)

fig.savefig(snakemake.output[0])
