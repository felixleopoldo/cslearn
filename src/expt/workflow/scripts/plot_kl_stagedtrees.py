import pandas as pd
from labels import canonical_method_label
from palette import (
    METHOD_COLORS,
    METHOD_MARKERS,
    _split_suffix,
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

plot_data = pd.read_csv(snakemake.input[0])
plot_data["method"] = plot_data["method"].map(canonical_method_label).map(_split_suffix)
plot_data["n"] = plot_data["n"].astype(int)

methods = sorted_methods(plot_data["method"].unique())
ns = sorted(plot_data["n"].unique())
linestyle_map = linestyles_for(ns)

set_plot_style()
fig, axes = square_grid(1, 1, panel_size=PANEL_SIZE)
ax = axes[0]

draw_lines(ax, plot_data, "p", "kl_div", "method", "n", METHOD_COLORS, METHOD_MARKERS, linestyle_map)
log_yaxis(ax)

fig.supxlabel("number of variables ($p$)")
ax.set_ylabel("KL divergence")

# grow_to_fit only grows height, which helps a multi-column grid (figures 2
# and 5) borrow width back via label_outer once height stops binding; this
# is a 1x1 grid, so it has no such column to borrow from and the y-tick
# labels alone cap the box below PANEL_SIZE. Widen first so grow_to_fit has
# a real width to grow into.
fig.canvas.draw()
fig.set_figwidth(fig.get_figwidth() + 0.65)

# A "beside" legend's frac is a width fraction, untouched by grow_to_fit
# (which only grows height), so it doesn't need recomputing after growth.
grow_to_fit(fig, ax)
FRAC = 0.5
reserve_legend_margin(fig, "beside", FRAC)

hue_entries = [(m, METHOD_COLORS[m], METHOD_MARKERS[m]) for m in methods]
style_entries = [(f"n={n}", linestyle_map[n]) for n in ns]
add_shared_legend(fig, hue_entries, style_entries, loc="beside", frac=FRAC, hue_ncol=1)

fig.savefig(snakemake.output[0])
