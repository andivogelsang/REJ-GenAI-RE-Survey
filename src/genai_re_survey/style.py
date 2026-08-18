import matplotlib as mpl


# If you also use seaborn elsewhere, you can still apply this rc setup.
def apply_paper_style(
    font_family="STIXGeneral",
    title_size=16,
    label_size=14,
    tick_size=12,
    legend_size=12,
    edge_lw=0.8,
    grid_alpha=0.15
):
    mpl.rcParams.update({
        "font.family": font_family,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.linewidth": edge_lw,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.alpha": grid_alpha,
        "savefig.bbox": "tight",
    })


# --- Shared layout constants for horizontal bar charts ---
BAR_HEIGHT      = 0.35
ROW_SPACING     = 0.50
PER_TASK_HEIGHT = 0.40
MIN_FIG_HEIGHT  = 4.0


def figure_height_for_rows(n_rows: int,
                           per_row: float = PER_TASK_HEIGHT,
                           min_height: float = MIN_FIG_HEIGHT) -> float:
    return max(min_height, n_rows * per_row)


# --- Shared color choices (color-blind & print friendly) ---
COLORS = {
    "yes": "#0072B2",   # blue (Okabe-Ito)
    "no":  "#D55E00",   # vermillion (Okabe-Ito)

    # Ordered scales (center-out) for diverging plot
    # Usefulness: blues (light -> dark)
    "use_blues": ["#DCEBFA", "#B9D7F4", "#86B9E8", "#4E97D9", "#1F6FB5"],
    # Harmfulness: oranges (light -> dark)
    "harm_oranges": ["#FBE6C5", "#F6C98A", "#F0A34F", "#E07A1F", "#B85A0A"],

    # Ordered scale (low -> high) for GenAI usage-frequency / experience-
    # duration charts. Distinct hue from use_blues/harm_oranges so it isn't
    # mistaken for the usefulness/harmfulness encoding.
    "frequency_purples": ["#F2F0F7", "#CBC9E2", "#9E9AC8", "#756BB1", "#54278F"],
    # Catch-all category (e.g. tool-usage frequency's free-text "Other").
    "neutral_other": "#999999",
}

# Round-distinguishing styling for grouped/paired bars (comparison plots).
# Category color (yes/no, use/harm) stays the encoding for *meaning*; round
# is encoded orthogonally via bar height/offset (see plotting.py) plus a
# lighter fill + dashed edge for the second round, so a black-and-white or
# colorblind reading still separates "round 1" from "round 2" bars.
ROUND_STYLES = {
    "round1": {"alpha": 1.0, "edgecolor": "black", "linestyle": "solid"},
    "round2": {"alpha": 0.55, "edgecolor": "black", "linestyle": "dashed"},
}
