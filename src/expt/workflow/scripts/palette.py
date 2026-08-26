"""Shared color/marker/linestyle logic and line-drawing for the experiment figures.

The print edition is black and white (JCGS production requirement); only the
online edition is color. So color alone can never carry a distinction the
figure needs a b&w reader to make. Every figure here compares two things at
once -- a *method* (or, for the sensitivity figures, a beta-misspecification
arm) and a sample size *n* -- so each gets its own non-color channel: method
identity is color *and* marker shape (marker survives grayscale on its own),
n is linestyle. A reader with the print copy can still tell every series
apart from marker shape and dash pattern alone; color is a bonus for the
online reader, not a required channel.
"""

import math
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from labels import canonical_method_label

# Generated with qualpal (github.com/jolars/qualpal), not hand-picked: a
# farthest-point search over CIEDE2000 distance in LCh(ab) space, so each
# color is maximally distinct from the others (not just "looks different to
# me") -- also asked to stay distinct after simulating protanopia/
# deuteranopia/tritanopia (full severity for protan/deutan, partial for
# tritan, the least common of the three) and against this project's actual
# plot background (seaborn "darkgrid"'s #EAEAF2, not white -- a color that
# reads fine on white can wash out against it). `background` alone doesn't
# enforce a hard minimum, though -- it's one more point in the same joint
# optimization as the colors' mutual separation, so a first pass (l: 0.25-
# 0.65) left a couple of colors (a yellow among them) only ~29-34 CIEDE2000
# units from the background, visibly closer than the ~43 minimum the colors
# kept from each other. Capping lightness well below the background's own
# (l: 0.1-0.4, deliberately darker rather than just avoiding yellow
# specifically, since any light color has the same wash-out problem against
# a light background regardless of hue) raised that worst case to ~32 for
# the method colors and ~37 for the beta colors below, at a small cost to
# mutual separation (43->37). Run once, offline, and hardcoded here rather
# than left as a runtime dependency -- these are fixed colors, not something
# that needs regenerating per plot, and adding qualpal to the experiment
# container (`workflow/envs/cslearn-expt.def`) for a one-time color choice
# isn't worth the image rebuild/republish:
#   from qualpal import Qualpal
#   Qualpal(
#       colorspace={"h": (0, 360), "c": (0.5, 0.9), "l": (0.1, 0.4)},
#       space="lchab", background="#EAEAF2",
#       cvd={"protan": 1.0, "deutan": 1.0, "tritan": 0.5},
#   ).generate(6)  # or .generate(3) for BETA_COLORS below
METHOD_COLORS = {
    "CSlearn+PC": "#b8420a",
    "PC": "#0caebd",
    "CSlearn+GRaSP": "#394d04",
    "GRaSP": "#acae1c",
    "BOS": "#b20aa6",
    "GRaSP+BHC": "#051b31",
}

METHOD_MARKERS = {
    "CSlearn+PC": "o",
    "PC": "s",
    "CSlearn+GRaSP": "^",
    "GRaSP": "D",
    "BOS": "v",
    "GRaSP+BHC": "P",
}

# Base colors/markers for the sensitivity plots' beta-misspecification arms.
# Also qualpal-generated (see METHOD_COLORS above; same params, generate(3)).
BETA_COLORS = {
    1: "#a6108e",  # over-spec
    2: "#0a2a19",  # correct
    3: "#acae1c",  # under-spec
}

BETA_MARKERS = {
    1: "^",
    2: "o",
    3: "v",
}

# Sensitivity legend/plot labels -- deliberately just the beta value, not the
# over/correct/under-spec framing. That explanation belongs once in the
# section's prose, not repeated in every legend entry; keeping it there also
# frees the legend to sit beside the panels instead of spanning full width.
# Mathtext ($\beta$), not a literal Unicode "β": `set_plot_style` switches
# regular text to Computer Modern's bundled cmr10.ttf, which only has ~130
# basic Latin/punctuation glyphs and no Greek letters -- this is the only
# non-ASCII text any figure renders, and mathtext's Computer Modern math
# italic (cmmi10, also bundled) has beta, matching how the manuscript itself
# renders it ($\beta$ in the .tex source).
BETA_LABELS = {
    1: r"$\beta=1$",
    2: r"$\beta=2$",
    3: r"$\beta=3$",
}

# Cycled by n-rank (ascending) within whatever n-values a given figure
# actually shows; up to 5 distinct dash patterns before repeating.
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]

_SUFFIX_RE = re.compile(r"^(?P<base>.+) \((?P<suffix>MLE|MAP)\)$")


def _split_suffix(label):
    """Strip a trailing " (MLE)"/" (MAP)" estimator suffix, if present."""
    m = _SUFFIX_RE.match(label)
    return m["base"] if m else label


def linestyles_for(values):
    """{value: linestyle} for sorted unique `values` (typically n), by rank."""
    ordered = sorted(set(values))
    return {v: LINESTYLES[i % len(LINESTYLES)] for i, v in enumerate(ordered)}


def sorted_methods(methods):
    """Order `methods` by METHOD_COLORS's canonical order, unknowns last."""
    order = list(METHOD_COLORS)
    return sorted(set(methods), key=lambda m: order.index(m) if m in order else len(order))


def set_plot_style():
    """Shared seaborn/matplotlib setup for every panel and legend script.

    Explicit point sizes, not a font_scale multiplier: JCGS requires figure
    text of at least 10pt, and every combined figure here is included in the
    manuscript at its native size (no `\\includegraphics[width=...]`
    downscale -- see `square_grid`), so whatever size is set here is what
    ends up on the printed page. One size (10pt, the minimum itself) for
    every text element -- axis labels/titles, tick labels, legend text --
    rather than a previous mix (11-11.5pt labels/titles vs. 10.5pt ticks/
    legend) that gave axis titles a visibly different, larger size than the
    rest of each figure's text for no reason tied to readability.

    Font family matters just as much as point size for how big text *looks*:
    matplotlib defaults to DejaVu Sans, whose x-height is 0.547x its em size
    (measured directly from the glyph outlines), vs. 0.431x for Computer
    Modern Roman -- the font family the manuscript's body text (Latin
    Modern, metric-compatible with Computer Modern) is built on. At this
    figure text's 10pt vs. the body's 12pt, DejaVu's x-height (10*0.547=
    5.47) is actually *larger* than Latin Modern's (12*0.431=5.17): "10pt"
    in one font family isn't visually comparable to "12pt" in another, only
    matching families makes the nominal size difference show up as an
    actual size difference. `cmr10` (plus its math companions `cmmi10`/
    `cmsy10`/`cmex10` for mathtext, e.g. axis labels' "$p$") are Computer
    Modern and ship as part of matplotlib's own package data -- no new
    dependency, unlike `qualpal` in the palette above. `pdf.fonttype=42`
    embeds them as proper CID TrueType rather than matplotlib's default
    Type 3, which some journal production pipelines reject in camera-ready
    figures.

    `axes.titlesize`/`axes.labelsize`/etc. above don't cover every text
    element matplotlib draws -- `fig.supxlabel`/`fig.supylabel` (every
    figure's shared axis label, its most prominent text) use a *different*
    rcParam, `figure.labelsize`, which defaults to the relative keyword
    "large" (1.2x `font.size`) if left unset -- verified by dumping actual
    `Tf` (set-font-size) operators from a rendered PDF's content stream and
    finding literal `12 Tf`, not `10 Tf`. That's almost certainly why the
    figures read as the same size as this manuscript's 12pt body text
    despite every other rcParam here saying 10. Silently-relative keywords
    are the recurring failure mode here: this same content-stream check also
    caught `8.33 Tf` (matplotlib's "small" keyword, 0.833x base) from a
    handful of `ax.set_title(..., fontsize="small")` calls in the panel
    scripts that bypassed `axes.titlesize` the same way. Trust rendered
    output over rcParam values when verifying this -- an rcParam can be set
    correctly and still not be the one actually governing a given text
    element.
    """
    sns.set_style("darkgrid")
    sns.set_style({"legend.frameon": False})
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.labelsize": 10,
            "figure.titlesize": 10,
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "pdf.fonttype": 42,
        }
    )


def draw_lines(ax, data, x, y, hue_col, style_col, color_map, marker_map, linestyle_map):
    """Draw one line per (hue_col, style_col) group: median `y` against `x`,
    with an IQR error bar at each point.

    Manual per-group plotting, rather than a single `sns.lineplot` call,
    because seaborn ties markers to `style`, not `hue` -- there's no
    built-in way to get marker-by-method and dashes-by-n independently,
    which is exactly the pair of channels the b&w requirement needs kept
    apart (see module docstring).
    """
    hue_order = sorted(
        data[hue_col].unique(), key=lambda h: list(color_map).index(h) if h in color_map else len(color_map)
    )
    for hue_val in hue_order:
        group = data[data[hue_col] == hue_val]
        for style_val in sorted(group[style_col].unique()):
            sub = group[group[style_col] == style_val]
            agg = sub.groupby(x)[y].agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
            agg = agg.sort_index()
            yerr = [agg["median"] - agg["q25"], agg["q75"] - agg["median"]]
            ax.errorbar(
                agg.index,
                agg["median"],
                yerr=yerr,
                color=color_map[hue_val],
                marker=marker_map[hue_val],
                linestyle=linestyle_map[style_val],
                linewidth=1.2,
                markersize=4.5,
                capsize=2,
                elinewidth=0.8,
            )


def method_group_entries(csv_paths, mle_suffix=False):
    """Union the methods and n-values present across one or more CSVs (e.g.
    time_3a + time_3b, sharing one legend) -- returns (methods, ns), each
    sorted, ready for `add_shared_legend`. An MLE/MAP estimator suffix
    on `method`, if present, is stripped first: that's a row facet in
    plot_kl.py, not a hue, so it never belongs in a legend."""
    methods, ns = set(), set()
    for path in csv_paths:
        df = pd.read_csv(path)
        methods.update(df["method"].map(canonical_method_label).map(_split_suffix))
        ns.update(df["n"].astype(int))
    return sorted_methods(methods), sorted(ns)


LEGEND_ROW_H_IN = 0.30
LEGEND_GROUP_GAP_IN = 0.03
LEGEND_PAD_IN = 0.10


def legend_height_in(n_hue_entries, hue_ncol, n_style_entries, style_ncol=None):
    """Height in inches a two-part `add_shared_legend` call will need:
    ceil(n_hue_entries/hue_ncol) rows for the hue grid, plus
    ceil(n_style_entries/style_ncol) rows for the style row (usually 1,
    since `style_ncol` defaults to fitting every style entry on one line),
    each `LEGEND_ROW_H_IN` tall, with a small gap between the two groups
    and a little padding.

    Used to size `square_grid`'s `legend_margin` from the actual entry
    counts instead of a guessed fraction -- figure 2's legend previously
    overlapped its panels because its margin (a guess) was sized for about
    half the rows the legend actually needed.
    """
    n_hue_rows = math.ceil(n_hue_entries / hue_ncol)
    n_style_rows = math.ceil(n_style_entries / (style_ncol or n_style_entries))
    return LEGEND_ROW_H_IN * (n_hue_rows + n_style_rows) + LEGEND_GROUP_GAP_IN + LEGEND_PAD_IN


def square_grid(nrows, ncols, panel_size=1.5, sharex=True, sharey=True):
    """Create an `nrows` x `ncols` grid of square axes (`panel_size` inches
    per side -- though see `grow_to_fit` below for why the *actual* box
    size a caller ends up with is usually bigger than this), sharing the y
    axis (by default -- pass `sharey=False` when panels plot different
    metrics, e.g. figure 6's KL divergence vs. SHD, which can't share a
    y-axis label or scale) and the x axis (by default -- pass
    `sharex=False` when panels span genuinely different x-domains, e.g.
    figures 4/5's small-p accuracy panels vs. their large-p scalability
    panel: sharing forced the small-p panels' data into a sliver near the
    axis origin instead of using their own range), hiding every tick label
    `label_outer` would otherwise consider redundant given whichever axes
    are actually shared. Multi-panel figures used to duplicate their x/y
    axis labels and tick labels on every panel independently, since each
    panel was a separately-generated PDF with no shared state; assembling
    the whole figure in one call fixes that structurally instead of
    patching each panel script.

    Legend space is *not* reserved here -- see `grow_to_fit` and
    `reserve_legend_margin`, called after this figure's content (lines,
    titles, tick customization) is drawn, for why reserving it this early
    used to waste space that the panels could otherwise have grown into.
    """
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size * ncols, panel_size * nrows),
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)  # always a flat array, regardless of nrows/ncols -- 1x1 is the common edge case
    for ax in axes:
        ax.set_box_aspect(1)
        ax.label_outer()
    return fig, axes


def grow_to_fit(fig, ref_ax, tol=0.003, step_in=0.3, max_iter=12):
    """Grow `fig`'s height only -- not width -- until `ref_ax`'s box stops
    getting wider, then back off the last (unproductive) step.

    `square_grid`'s `ax.set_box_aspect(1)` forces box width to equal box
    height. The figure height `square_grid` sets is exactly `panel_size *
    nrows`, with no room reserved for the x-tick-labels/`fig.supxlabel`
    that sit beneath the row -- that space gets carved *out of*
    `panel_size` instead of added on top of it (the mirror image of the
    mistake `legend_margin` used to make before this existed, and which
    `reserve_legend_margin` below now avoids the same way). The result: box
    height comes out smaller than `panel_size`, and box_aspect drags box
    *width* down to match -- even though a row's columns typically have
    unused horizontal space (interior columns need no y-axis-label margin,
    thanks to `label_outer`), which just sits there as dead space between
    panels instead of being absorbed into any panel's box. Measured on
    figure 5's actual data: box width came out 1.047in against a requested
    `panel_size` of 1.59in -- an over 50% shortfall, all of it recoverable
    from space the row already had.

    Growing height lets the box widen into that space, since only height
    was the starved/binding dimension -- width (the figure's total,
    already tuned against `\\textwidth`, see CLAUDE.md's sizing note) is
    never touched, so this can't push a figure over the page-width budget
    the way naively enlarging `panel_size` itself would. Tick/axis-label
    space is set in points, not as a fraction of figure size, so it's
    ~constant across iterations -- this typically converges in 2 steps.
    Call this after all panel content (lines, titles, tick customization,
    including any post-hoc `sharey`/`tick_params` calls) is drawn, since
    only then are the actual label extents known, and *before* reserving
    legend margin (`reserve_legend_margin`), so the width used to
    determine how much unused space exists is the row's own, not inflated
    by a legend band.
    """
    fig.canvas.draw()
    prev_w = ref_ax.get_position().width * fig.get_figwidth()
    prev_h = fig.get_figheight()
    for _ in range(max_iter):
        fig.set_figheight(fig.get_figheight() + step_in)
        fig.canvas.draw()
        w = ref_ax.get_position().width * fig.get_figwidth()
        if w - prev_w < tol:
            fig.set_figheight(prev_h)  # last step didn't help -- revert the wasted growth
            # constrained_layout needs a second draw to fully reconverge after
            # a size change -- one draw() alone leaves stale internal state
            # from the just-reverted size, which for a panel that needed *no*
            # growth at all (the revert lands back on the pristine original
            # height, e.g. a single-panel figure with no wasted gutters to
            # reclaim) showed up as the x-axis label's bounding box extending
            # below y=0 -- clipped off the bottom of the saved PDF entirely.
            fig.canvas.draw()
            fig.canvas.draw()
            return
        prev_w, prev_h = w, fig.get_figheight()


def reserve_legend_margin(fig, loc, frac):
    """Grow `fig` to add space for `add_shared_legend`, and reserve it via
    the constrained-layout engine's `rect` -- without this, a legend added
    afterwards would either overlap the grid or get clipped at the figure
    edge, since nothing outside `rect` is protected space as far as
    constrained_layout is concerned. The margin is *added* on top of
    `fig`'s current size, not carved out of it -- carving it out would
    shrink the panel area below what `grow_to_fit` (called first) already
    settled on, the same authored-size-doesn't-match-intent mistake this
    whole redesign exists to stop making.

    Call after `grow_to_fit`, using a `frac` computed relative to `fig`'s
    *current* (post-growth) size -- e.g. `legend_h / (fig.get_figheight() +
    legend_h)` for an "above" legend -- not relative to the original
    `panel_size`, since growth already changed what fraction of the final
    figure the legend's fixed absolute height needs to be.
    """
    width, height = fig.get_figwidth(), fig.get_figheight()
    if loc == "above":
        height /= 1 - frac
    elif loc == "beside":
        width /= 1 - frac
    else:
        raise ValueError(f"Unknown legend loc: {loc!r}")
    fig.set_size_inches(width, height)
    rect = {"above": (0.0, 0.0, 1.0, 1.0 - frac), "beside": (0.0, 0.0, 1.0 - frac, 1.0)}[loc]
    fig.get_layout_engine().set(rect=rect)
    # Two draws, not one -- the same constrained_layout reconvergence quirk
    # documented in grow_to_fit's revert branch: a single draw after a size
    # change can leave stale internal state (observed as a bottom-edge label
    # extending below y=0, clipped at save time). add_shared_legend draws
    # again on its own right after this, which alone wasn't enough for a
    # single-panel figure with no further legend-fitting draws to mask it.
    fig.canvas.draw()
    fig.canvas.draw()


def _fit_legend(fig, handles, labels, max_ncol, max_width_in, x, y, anchor_loc="upper left", **kwargs):
    """Place a legend at up to `max_ncol` columns, anchored at figure-
    fraction `(x, y)` via `anchor_loc` (e.g. "upper left" to grow rightward
    from `x`, "upper center" to grow symmetrically around `x`), backing off
    to fewer columns (and thus more automatically-wrapped rows -- matplotlib
    wraps a legend's entries into ceil(count/ncol) rows on its own once ncol
    is fixed) until its actual rendered width fits `max_width_in`, or
    there's only one column left to try.

    A legend's rendered width is set by its content, not by the parent
    figure's nominal size -- and isn't clipped to it either, so measuring
    via a canvas draw (rather than trusting the requested `max_width_in`)
    is required, not optional: a naive `ncol=len(handles)` (e.g. one row per
    unique n) silently overflowed a narrow `max_width_in` by 3-4x before
    this existed. That overflow isn't necessarily "easy to spot" the way it
    might sound: a "beside" legend growing rightward from its left edge
    overflows off the *right edge of the canvas entirely* (there's no axes
    there to visibly collide with), and since `add_shared_legend` doesn't
    save with `bbox_inches="tight"`, anything past the canvas edge is
    silently clipped at save time, not just badly laid out -- this is
    exactly how figure 3's legend text ended up cut off: one label
    ("CSlearn+GRaSP") didn't fit in its reserved width even at `ncol=1`,
    the floor this loop can back off to, and nothing checked that floor
    case against `max_width_in` the way `add_shared_legend`'s height check
    already does for the vertical case. The anchor point doesn't affect
    the width measurement (`bbox.width` is anchor-agnostic), so `anchor_loc`
    is a free choice independent of the fitting logic.
    """
    ncol = max(1, min(max_ncol, len(handles)))
    while True:
        legend = fig.legend(handles, labels, loc=anchor_loc, bbox_to_anchor=(x, y), ncol=ncol, **kwargs)
        fig.canvas.draw()
        bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        width_in = bbox.width * fig.get_figwidth()
        if width_in <= max_width_in:
            return legend, bbox
        if ncol == 1:
            # Can't back off any further -- some single label is wider than
            # max_width_in on its own. Catch it here, at generation time,
            # rather than as silently clipped text in the saved PDF (see
            # this function's docstring: this is exactly how figure 3's
            # legend went unnoticed).
            raise ValueError(
                f"_fit_legend: content needs {width_in:.2f}in but only {max_width_in:.2f}in is reserved, "
                f"even at ncol=1 -- widen the legend's frac/max_width_in, or shorten its longest label."
            )
        legend.remove()
        ncol -= 1


def add_shared_legend(fig, hue_entries, style_entries, loc, frac, hue_ncol=1, fontsize=None):
    """Draw a two-part legend (method/beta-arm grid + n/linestyle row)
    directly into `fig`'s margin reserved by `square_grid`'s
    `legend_margin=(loc, frac)` -- same `loc`/`frac` values as passed there.

    Two physically separate `fig.legend` calls, rather than one legend
    listing every method x n combination: that only made sense when n was
    encoded as a color shade of its method, so every combination needed its
    own swatch. Now that marker/linestyle carry method/n independently (see
    module docstring), each dimension only needs to be listed once -- which
    is also what lets a figure like kl-div 2a+2b share a *single* legend
    instead of each panel needing its own (the n/linestyle mapping is
    identical across every panel in a figure; repeating it per panel, as
    the previous per-panel-legend design did, was pure duplication).

    `hue_entries`: [(label, color, marker), ...].
    `style_entries`: [(label, linestyle), ...].
    `hue_ncol`: requested columns for the method/beta-arm grid -- e.g. 3 for
    a 6-entry legend to render as 2 rows x 3 cols. `_fit_legend` may still
    reduce it (and separately choose columns for `style_entries`) if that
    doesn't fit the reserved margin.
    """
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    if loc == "above":
        x0, y0, x1, y1 = 0.0, 1.0 - frac, 1.0, 1.0
        # Centered, not left-anchored: a legend flush against the left edge
        # of a full-width "above" band reads as an accident, not a choice,
        # especially once each row's width no longer fills that band (e.g.
        # a 4-entry method row that's much narrower than the figure). Each
        # row centers independently below, since the two rows are rarely
        # the same width.
        anchor_loc, legend_x = "upper center", (x0 + x1) / 2
    elif loc == "beside":
        x0, y0, x1, y1 = 1.0 - frac, 0.0, 1.0, 1.0
        # Left-anchored: reads naturally in a narrow sidebar column, and
        # wasn't flagged as needing to change.
        anchor_loc, legend_x = "upper left", x0
    else:
        raise ValueError(f"Unknown legend loc: {loc!r}")
    max_width_in = (x1 - x0) * fig_w

    hue_handles = [
        plt.Line2D([0], [0], color=color, marker=marker, linestyle="-", markersize=5, linewidth=1.2)
        for _, color, marker in hue_entries
    ]
    hue_labels = [label for label, _, _ in hue_entries]

    style_handles = [
        plt.Line2D([0], [0], color="0.3", marker="None", linestyle=linestyle, linewidth=1.4)
        for _, linestyle in style_entries
    ]
    style_labels = [label for label, _ in style_entries]

    # handlelength=1.484 (em) = 14.84pt at this legend's 10pt font -- not a
    # round number, a deliberately exact one: matplotlib's default dash
    # periods at this row's linewidth=1.4 are dashed=7.42pt, dotted=3.71pt,
    # dashdot=14.84pt (clean 1:2:4 ratios), so 14.84pt is simultaneously 2
    # full dashed periods, 4 full dotted periods, and 1 full dashdot period
    # -- every non-solid linestyle these legends actually use ends cleanly
    # (mid-gap, or partway through a genuinely long segment), with no
    # trailing stub. The previous 1.6em (16pt) was 2.156 dashed periods,
    # leaving a 1.16pt stub into a new dash cycle -- short enough to read as
    # a stray dot instead of a third dash ("- - ." instead of "- - -").
    kwargs = dict(
        frameon=False, fontsize=fontsize, handlelength=1.484, handletextpad=0.4, columnspacing=0.8, labelspacing=0.4
    )
    vgap_in = LEGEND_GROUP_GAP_IN  # keep in sync with legend_height_in's margin-sizing estimate

    hue_legend, hue_bbox = _fit_legend(
        fig, hue_handles, hue_labels, hue_ncol, max_width_in, x=legend_x, y=y1, anchor_loc=anchor_loc, **kwargs
    )
    _, style_bbox = _fit_legend(
        fig,
        style_handles,
        style_labels,
        len(style_entries),
        max_width_in,
        x=legend_x,
        y=hue_bbox.y0 - vgap_in / fig_h,
        anchor_loc=anchor_loc,
        **kwargs,
    )

    # Catch a margin sized too small for its content at generation time
    # rather than needing a human to spot it in the rendered PDF -- this is
    # exactly how figure 2's legend overlapping its panels went unnoticed
    # (the margin was a guess; nothing checked it against what got drawn).
    if style_bbox.y0 < y0 - 1e-6:
        overflow_in = (y0 - style_bbox.y0) * fig_h
        raise ValueError(
            f"add_shared_legend: legend needs {overflow_in:.2f}in more than its reserved margin "
            f"(loc={loc!r}, frac={frac}) -- it will overlap the axes. Widen `frac` (see "
            f"`legend_height_in` for a content-derived value) or reduce hue_ncol/entry count."
        )
