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

PANEL_SIZE = 1.59  # 4 panels x 1.59in = 6.36in vs. \textwidth's exact 6.396431in (verified via
# a scratch `\the\textwidth` compile, not the earlier approximation) -- 0.036in/2.6pt margin,
# close to the practical ceiling a 1x4 row can use at this page width; matches figure 2, the
# other 1x4-with-legend-above figure. See CLAUDE.md's sizing note.
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
# sharex=False: panels (a)-(c) cover p=5-20 (accuracy comparison) but panel
# (d) covers p up to 500 (scalability sweep) -- a genuinely different
# x-domain, not just a wider view of the same one. Sharing squeezed (a)-(c)'s
# data into a sliver near the axis origin of a 0-500 axis instead of letting
# each panel use its own natural range. sharey=False for the same reason on
# the y-axis: (a)-(c)'s SHD values top out around 50-80, but (d)'s scale-up
# to p=500 reaches into the thousands (~15-25x larger) -- sharing one y-axis
# across all 4 would squash (a)-(c)'s variation into a sliver near zero.
fig, axes = square_grid(1, 4, panel_size=PANEL_SIZE, sharex=False, sharey=False)

for ax, (letter, _) in zip(axes.flat, panels):
    draw_lines(ax, dfs[letter], "p", "shd", "method", "n", METHOD_COLORS, METHOD_MARKERS, linestyle_map)
    ax.set_title(PANEL_TITLES[letter], loc="left")

# (a)-(c) share a common y-axis (their magnitudes are comparable to each
# other, just not to (d)'s) -- link them post hoc rather than in
# square_grid, so only these three panels' scales are tied together and
# (d) keeps its own.
axes[1].sharey(axes[0])
axes[2].sharey(axes[0])
for ax in axes[:3]:
    ax.relim()
    ax.autoscale(axis="y")

# label_outer (already called once, inside square_grid) hides y-tick-labels
# by grid *position* (every column but the first), not by whether an axis
# is actually sharing a y-scale -- with sharey=False at creation time that
# incorrectly hid (d)'s labels too, even though it has its own independent
# scale and needs them to be readable at all. (a)/(c) still need hiding
# (now genuinely redundant, since they share (a)'s scale); (d) needs its
# own labels restored.
axes[1].tick_params(labelleft=False)
axes[2].tick_params(labelleft=False)
axes[3].tick_params(labelleft=True)

fig.supxlabel("number of variables ($p$)")
axes[0].set_ylabel("SHD (LDAG)")

# Let the panels grow into whatever horizontal space square_grid's box
# height starvation was leaving unused (see grow_to_fit) before reserving
# room for the legend -- reserving the legend margin first, against the
# too-small starved row, was locking in the waste instead of recovering it.
grow_to_fit(fig, axes[0])

# Margin sized from the legend's actual content (row counts) and the row's
# now-final (post-growth) height, not a guess.
legend_h = legend_height_in(len(hue_entries), HUE_NCOL, len(style_entries))
frac = legend_h / (fig.get_figheight() + legend_h)
reserve_legend_margin(fig, "above", frac)

add_shared_legend(fig, hue_entries, style_entries, loc="above", frac=frac, hue_ncol=HUE_NCOL)

fig.savefig(snakemake.output[0])
