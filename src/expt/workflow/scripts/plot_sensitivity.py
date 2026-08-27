import pandas as pd
from palette import (
    BETA_COLORS,
    BETA_LABELS,
    BETA_MARKERS,
    add_shared_legend,
    draw_lines,
    grow_to_fit,
    linestyles_for,
    reserve_legend_margin,
    set_plot_style,
    square_grid,
)

PANEL_SIZE = 1.59  # matches figures 2 and 5's panel size

kl_data = pd.read_csv(snakemake.input.kl)
shd_data = pd.read_csv(snakemake.input.shd)
kl_data["n"] = kl_data["n"].astype(int)
shd_data["n"] = shd_data["n"].astype(int)

all_betas = sorted(set(kl_data["true_max_cvar"].unique()) | set(shd_data["true_max_cvar"].unique()))
all_ns = sorted(set(kl_data["n"].unique()) | set(shd_data["n"].unique()))
linestyle_map = linestyles_for(all_ns)

set_plot_style()
# KL divergence and SHD are different metrics, so sharey=False keeps each
# panel's own y-tick labels rather than treating one as redundant.
fig, axes = square_grid(1, 2, panel_size=PANEL_SIZE, sharey=False)

draw_lines(axes[0], kl_data, "p", "kl_div", "true_max_cvar", "n", BETA_COLORS, BETA_MARKERS, linestyle_map)
axes[0].set_yscale("log")
axes[0].set_ylabel("KL divergence")
# Titles name what each metric means, not just repeat the y-axis label.
axes[0].set_title("(a) distributional accuracy", loc="left")

draw_lines(axes[1], shd_data, "p", "shd", "true_max_cvar", "n", BETA_COLORS, BETA_MARKERS, linestyle_map)
axes[1].set_ylabel("SHD (LDAG)")
axes[1].set_title("(b) structural accuracy", loc="left")

fig.supxlabel("number of variables ($p$)")

# A "beside" legend's frac is a width fraction, untouched by grow_to_fit
# (which only grows height). Both panels show their own y-tick labels
# (sharey=False), so either panel's box grows identically to the other's.
grow_to_fit(fig, axes[0])
FRAC = 0.22
reserve_legend_margin(fig, "beside", FRAC)

hue_entries = [(BETA_LABELS[b], BETA_COLORS[b], BETA_MARKERS[b]) for b in all_betas]
style_entries = [(f"n={n}", linestyle_map[n]) for n in all_ns]
add_shared_legend(fig, hue_entries, style_entries, loc="beside", frac=FRAC, hue_ncol=1)

fig.savefig(snakemake.output[0])
