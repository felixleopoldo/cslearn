import pandas as pd
from labels import canonical_method_label
from palette import (
    METHOD_COLORS,
    METHOD_MARKERS,
    add_shared_legend,
    draw_lines,
    grow_to_fit,
    linestyles_for,
    log_yaxis,
    reserve_legend_margin,
    set_plot_style,
    sorted_methods,
    square_grid,
)

PANEL_SIZE = 1.59  # matches figures 2 and 5's panel size

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
# sharex=False: panel (a) covers p=5-20, panel (b) up to 500 -- a genuinely
# different x-domain, not a wider view of the same one.
fig, axes = square_grid(1, 2, panel_size=PANEL_SIZE, sharex=False)

for title, ax, df in zip(PANEL_TITLES, axes, dfs):
    draw_lines(ax, df, "p", "total_time", "method", "n", METHOD_COLORS, METHOD_MARKERS, linestyle_map)
    log_yaxis(ax)
    ax.set_title(title, loc="left")

fig.supxlabel("number of variables ($p$)")
axes[0].set_ylabel("time (seconds)")

# A "beside" legend's frac is a width fraction, untouched by grow_to_fit
# (which only grows height), so it doesn't need recomputing after growth.
grow_to_fit(fig, axes[0])
FRAC = 0.32
reserve_legend_margin(fig, "beside", FRAC)

hue_entries = [(m, METHOD_COLORS[m], METHOD_MARKERS[m]) for m in sorted_methods(all_methods)]
style_entries = [(f"n={n}", linestyle_map[n]) for n in sorted(all_ns)]
add_shared_legend(fig, hue_entries, style_entries, loc="beside", frac=FRAC, hue_ncol=1)

fig.savefig(snakemake.output[0])
