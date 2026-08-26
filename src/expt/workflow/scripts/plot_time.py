import pandas as pd
from labels import canonical_method_label
from palette import (
    METHOD_COLORS,
    METHOD_MARKERS,
    add_shared_legend,
    draw_lines,
    grow_to_fit,
    linestyles_for,
    reserve_legend_margin,
    set_plot_style,
    sorted_methods,
    square_grid,
)

panels = [snakemake.input.a, snakemake.input.b]
PANEL_TITLES = ["(a) all methods", "(b) scalable methods"]

dfs = []
all_methods, all_ns = set(), set()
for path in panels:
    df = pd.read_csv(path)
    df["method"] = df["method"].map(canonical_method_label)
    df["n"] = df["n"].astype(int)
    dfs.append(df)
    all_methods.update(df["method"].unique())
    all_ns.update(df["n"].unique())

linestyle_map = linestyles_for(all_ns)

set_plot_style()
# sharex=False: panel (a) covers p=5-20 (all methods) but panel (b) covers
# p up to 500 (scalable methods only) -- a genuinely different x-domain, not
# a wider view of the same one. Sharing squeezed (a)'s data into a sliver
# near the axis origin of a 0-500 axis instead of letting it use its own
# natural range.
fig, axes = square_grid(1, 2, panel_size=2.1, sharex=False)

for title, ax, df in zip(PANEL_TITLES, axes, dfs):
    draw_lines(ax, df, "p", "total_time", "method", "n", METHOD_COLORS, METHOD_MARKERS, linestyle_map)
    ax.set_yscale("log")
    ax.set_title(title, loc="left")

fig.supxlabel("number of variables ($p$)")
axes[0].set_ylabel("runtime (seconds)")

# Let the panels grow into whatever vertical space square_grid's box height
# starvation was leaving unused (see grow_to_fit) before reserving room for
# the legend. A "beside" legend eats figure *width*, so unlike the "above"
# figures, frac here doesn't need recomputing after growth -- it's a width
# fraction, untouched by growing height only.
grow_to_fit(fig, axes[0])
FRAC = 0.30
reserve_legend_margin(fig, "beside", FRAC)

hue_entries = [(m, METHOD_COLORS[m], METHOD_MARKERS[m]) for m in sorted_methods(all_methods)]
style_entries = [(f"n={n}", linestyle_map[n]) for n in sorted(all_ns)]
add_shared_legend(fig, hue_entries, style_entries, loc="beside", frac=FRAC, hue_ncol=1)

fig.savefig(snakemake.output[0])
