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

PANEL_SIZE = 1.59  # matches figure 2's 1x4-with-legend-above sizing (near the \textwidth ceiling)
HUE_NCOL = 6  # try all 6 methods on one row -- _fit_legend backs off to fewer columns on its
# own if this doesn't actually fit the rendered width, so requesting 6 is safe either way

panels = [("a", snakemake.input.a), ("b", snakemake.input.b), ("c", snakemake.input.c), ("d", snakemake.input.scale)]
PANEL_TITLES = {
    "a": "(a) vs. PC",
    "b": "(b) vs. GRaSP",
    "c": "(c) vs. staged trees",
    "d": "(d) scalable",
}

dfs = {}
all_methods, all_ns = set(), set()
for letter, path in panels:
    df = pd.read_csv(path)
    df["method"] = df["method"].map(canonical_method_label)
    df["n"] = df["n"].astype(int)
    dfs[letter] = df
    all_methods.update(df["method"].unique())
    all_ns.update(df["n"].unique())

linestyle_map = linestyles_for(all_ns)
hue_entries = [(m, METHOD_COLORS[m], METHOD_MARKERS[m]) for m in sorted_methods(all_methods)]
style_entries = [(f"n={n}", linestyle_map[n]) for n in sorted(all_ns)]

set_plot_style()
# sharex=False, sharey=False: panels (a)-(c) cover p=5-20 with SHD in the
# tens, but (d) is a scalability sweep to p=500 with SHD in the thousands --
# a genuinely different domain on both axes, not a wider view of the same one.
fig, axes = square_grid(1, 4, panel_size=PANEL_SIZE, sharex=False, sharey=False)

for ax, (letter, _) in zip(axes.flat, panels):
    draw_lines(ax, dfs[letter], "p", "shd", "method", "n", METHOD_COLORS, METHOD_MARKERS, linestyle_map)
    ax.set_title(PANEL_TITLES[letter], loc="left")

# (a)-(c) are mutually comparable, just not to (d) -- link them post hoc
# so only those three share a y-scale.
axes[1].sharey(axes[0])
axes[2].sharey(axes[0])
for ax in axes[:3]:
    ax.relim()
    ax.autoscale(axis="y")

# label_outer (inside square_grid) hides y-tick-labels by grid position, not
# by actual y-scale sharing, so it wrongly hid (d)'s labels too -- restore them.
axes[1].tick_params(labelleft=False)
axes[2].tick_params(labelleft=False)
axes[3].tick_params(labelleft=True)

fig.supxlabel("number of variables ($p$)")
axes[0].set_ylabel("SHD (LDAG)")

# grow_to_fit before reserving legend margin, so panels grow into unused
# space first rather than reserving margin against an already-starved row.
grow_to_fit(fig, axes[0])

# Margin sized from the legend's actual content (row counts) and the row's
# now-final (post-growth) height, not a guess.
legend_h = legend_height_in(len(hue_entries), HUE_NCOL, len(style_entries))
frac = legend_h / (fig.get_figheight() + legend_h)
reserve_legend_margin(fig, "above", frac)

add_shared_legend(fig, hue_entries, style_entries, loc="above", frac=frac, hue_ncol=HUE_NCOL)

fig.savefig(snakemake.output[0])
