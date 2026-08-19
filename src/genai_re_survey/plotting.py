"""Plotting functions shared by the round-1, round-2, and comparison reports.

Each public function takes `rounds: list[tuple[str, pd.DataFrame]]` instead
of a single DataFrame. With exactly one round, it dispatches to a private
`_..._single()` implementation that is untouched from the original
Survey_RE_and_AI.ipynb logic (just referencing style.py/labels.py constants
instead of notebook-global ones) — this is what guarantees round-1 output
stays byte-identical to the published REFSQ figures. With two rounds, bars
are drawn grouped/paired per item: the row band each item occupies is split
across rounds, using the same color per category (Yes/No, usefulness/
harmfulness) and a round-specific alpha/edge style from style.ROUND_STYLES
so round 1 vs. round 2 stays distinguishable without changing what a color
means.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from . import labels, style

USEFULNESS_LABELS = [
    'Extremely useful', 'Very useful', 'Moderately useful',
    'Slightly useful', 'Not useful at all'
]
HARMFULNESS_LABELS = [
    'Extremely harmful', 'Very harmful', 'Moderately harmful',
    'Slightly harmful', 'Not harmful at all'
]

# From center (near 0%) outward
USE_CENTER_OUT = ['Not useful at all', 'Slightly useful', 'Moderately useful', 'Very useful', 'Extremely useful']
HARM_CENTER_OUT = ['Not harmful at all', 'Slightly harmful', 'Moderately harmful', 'Very harmful', 'Extremely harmful']


def _percent_counts(series: pd.Series, categories: list[str], exclude="I don't know") -> pd.Series:
    series = series[series != exclude]
    counts = series.value_counts().reindex(categories).fillna(0)
    return (counts / counts.sum() * 100) if counts.sum() > 0 else counts


def _round_offsets(n_rounds: int, base_height: float) -> tuple[float, list[float]]:
    """(bar_height, y-offsets), one offset per round.

    With one round, bar_height == base_height and offset == 0 — the exact
    single-round layout. With N>1 rounds, the row band is split into N
    equal sub-bars centered on the row.
    """
    if n_rounds == 1:
        return base_height, [0.0]
    bar_height = base_height / n_rounds
    offsets = [(i - (n_rounds - 1) / 2) * bar_height for i in range(n_rounds)]
    return bar_height, offsets


def _round_style(round_label: str, n_rounds: int) -> dict:
    # Only differentiate rounds visually when there's actually more than one
    # in the same chart — a standalone round-2 report shouldn't render
    # faded/dashed just because that's how it looks *next to* round 1.
    if n_rounds == 1:
        return {"alpha": 1.0, "edgecolor": "black", "linestyle": "solid"}
    key = round_label.lower().replace(" ", "")
    return style.ROUND_STYLES.get(key, {"alpha": 1.0, "edgecolor": "black", "linestyle": "solid"})


# ---------------------------------------------------------------------------
# Significance markers — shared by every comparison (multi-round) chart.
# `significant_items` is None when significance wasn't computed for a chart
# (e.g. single-round reports, or a family reports.py doesn't test); passing
# an explicit (possibly empty) set marks the chart as "tested" and shows the
# caption regardless of whether anything reached significance.
# ---------------------------------------------------------------------------

_SIGNIFICANCE_CAPTION = '* significant difference vs. the other round (q < 0.05, FDR-corrected within family)'


def _mark_significant(label: str, item: str, significant_items: set[str] | None) -> str:
    if significant_items is not None and item in significant_items:
        return f"{label} *"
    return label


def _finalize_significance_marks(ax, item_order: list[str], significant_items: set[str] | None) -> None:
    """Bold the y-tick labels already suffixed via `_mark_significant`, and
    add a caption explaining the convention. No-op if significance wasn't
    computed for this chart.

    The caption is appended to the xlabel (as a second line) rather than
    placed as a floating `ax.text` annotation below the axes — a floating
    annotation's position is in axes-fraction coordinates and isn't reliably
    accounted for by `tight_layout`/`bbox_inches="tight"`, so it tended to
    collide with the xlabel itself. A multi-line xlabel is laid out and
    measured by matplotlib like any other label, so it can't overlap.
    """
    if significant_items is None:
        return
    for text, item in zip(ax.get_yticklabels(), item_order):
        if item in significant_items:
            text.set_fontweight('bold')
    ax.set_xlabel(f"{ax.get_xlabel()}\n{_SIGNIFICANCE_CAPTION}")


# ---------------------------------------------------------------------------
# Categorical percentage (stacked horizontal bar) — Yes/No charts (usage,
# skills) and ordinal single-choice charts (tool-usage frequency, GenAI
# experience) all share this: one or more items, each a 100%-stacked bar
# over a fixed, ordered set of categories.
#
# `rounds` dataframes must be indexed by a *category-free* item key (e.g.
# "Elicitation", not "Elicitation (n=64)") — the multi-round path aligns
# rounds on that index, so baking a round-specific n into it would make
# every item look round-unique and silently break pairing. Per-item,
# per-round n instead travels in a dedicated 'n' column and is rendered
# into the tick label (single round) or combined into it per round
# (comparison), so it never has to participate in alignment.
# ---------------------------------------------------------------------------

def _plot_stacked_percentage_barh_single(
    plot_df: pd.DataFrame,   # index = item key (no n); columns = categories + 'n'
    categories: list[str],
    colors: list[str],
    title: str,
    xlabel: str,
    savepath: str,
):
    n_tasks = len(plot_df)
    fig_height = style.figure_height_for_rows(n_tasks)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y_pos = np.arange(n_tasks) * style.ROW_SPACING

    left_vals = np.zeros(n_tasks)
    for cat, color in zip(categories, colors):
        vals = plot_df[cat].to_numpy()
        ax.barh(y_pos, vals, left=left_vals, color=color,
                height=style.BAR_HEIGHT, edgecolor='black', linewidth=0.8, label=cat)
        left_vals += vals

    # In-bar labels (percent, no decimals)
    for i, row in enumerate(plot_df[categories].to_numpy()):
        x_left = 0.0
        for width_val in row:
            if width_val > 0:
                ax.text(x_left + width_val / 2, y_pos[i], f"{width_val:.0f}%",
                        ha='center', va='center', fontsize=14, color='white')
                x_left += width_val

    y_labels = [f"{item} (n={int(n)})" for item, n in zip(plot_df.index, plot_df['n'])]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_title(title)

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=min(len(categories), 3),
        frameon=False,
        title=None
    )

    plt.tight_layout()
    plt.savefig(savepath, format=savepath.split('.')[-1], bbox_inches="tight")
    plt.show()


def plot_stacked_percentage_barh(
    rounds: list[tuple[str, pd.DataFrame]],
    categories: list[str],
    colors: list[str],
    title: str,
    savepath: str,
    xlabel: str = 'Percentage of Respondents',
    significant_items: set[str] | None = None,
):
    if len(rounds) == 1:
        _, df = rounds[0]
        return _plot_stacked_percentage_barh_single(df, categories, colors, title, xlabel, savepath)

    n_rounds = len(rounds)

    item_order = list(rounds[0][1].index)
    for _, df in rounds[1:]:
        item_order += [item for item in df.index if item not in item_order]
    n_tasks = len(item_order)

    fig_height = style.figure_height_for_rows(n_tasks)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    base_y = np.arange(n_tasks) * style.ROW_SPACING
    bar_height, offsets = _round_offsets(n_rounds, style.BAR_HEIGHT)

    round_handles = []
    n_by_item = {item: {} for item in item_order}
    for r_idx, (round_label, df) in enumerate(rounds):
        df = df.reindex(item_order)
        y_pos = base_y + offsets[r_idx]
        rstyle = _round_style(round_label, n_rounds)
        left_vals = np.zeros(n_tasks)
        for cat, color in zip(categories, colors):
            vals = df[cat].fillna(0.0).to_numpy()
            ax.barh(y_pos, vals, left=left_vals, color=color,
                    height=bar_height, linewidth=0.8,
                    label=(cat if r_idx == 0 else None), **rstyle)
            left_vals += vals
        round_handles.append(Patch(facecolor='0.6', label=round_label, **rstyle))
        for item, n in zip(item_order, df['n']):
            n_by_item[item][round_label] = 0 if pd.isna(n) else int(n)

    y_labels = [
        _mark_significant(
            f"{item} (" + ", ".join(f"{rl} n={n_by_item[item].get(rl, 0)}" for rl, _ in rounds) + ")",
            item, significant_items,
        )
        for item in item_order
    ]
    ax.set_yticks(base_y)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_title(title)

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    _finalize_significance_marks(ax, item_order, significant_items)

    handles, legend_labels = ax.get_legend_handles_labels()
    handles += round_handles
    legend_labels += [h.get_label() for h in round_handles]
    ax.legend(
        handles, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=min(len(categories) + n_rounds, 4),
        frameon=False,
        title=None
    )

    plt.tight_layout()
    plt.savefig(savepath, format=savepath.split('.')[-1], bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# "Percentage of Yes" horizontal bar (prevention / threats / training
# charts). Both single-round and multi-round comparison show percentage of
# that round's respondents (with n given alongside), so single-round and
# comparison figures for the same block read on the same axis.
# ---------------------------------------------------------------------------

def _count_column(s: pd.Series) -> int:
    non_null = s.dropna()
    unique_vals = set(non_null.unique())
    # If values other than Yes/No exist -> count all non-NaN
    if not unique_vals.issubset({'Yes', 'No'}):
        return non_null.shape[0]
    # Otherwise -> count only "Yes"
    return (s == 'Yes').sum()


def _plot_yes_counts_barh_single(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    label_func,
    savepath: str
):
    counts = df.apply(_count_column).sort_values(ascending=False)

    # Global respondent count (at least one non-null across all columns)
    n_responses = df.dropna(how="all").shape[0]
    pct = (counts / n_responses * 100) if n_responses else counts.astype(float)

    labels_ = [label_func(col) for col in counts.index]

    n = len(labels_)
    fig_height = style.figure_height_for_rows(n)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    # shrink bar area, enlarge label area
    fig.subplots_adjust(left=0.5, right=0.82)
    y_pos = np.arange(n) * style.ROW_SPACING

    ax.barh(
        y=y_pos,
        width=pct.values,
        height=style.BAR_HEIGHT,
        color="white",
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_)
    ax.set_xlabel(f'Percentage of Respondents (n={n_responses})')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)

    plt.savefig(savepath, format=savepath.split('.')[-1])
    plt.show()


def plot_yes_counts_barh(
    rounds: list[tuple[str, pd.DataFrame]],
    title: str,
    ylabel: str,
    label_func,
    savepath: str,
    significant_items: set[str] | None = None,
):
    if len(rounds) == 1:
        _, df = rounds[0]
        return _plot_yes_counts_barh_single(df, title, ylabel, label_func, savepath)

    # Rounds differ in total respondents (e.g. round 1 n=150 vs round 2
    # n=79), so raw "yes" counts aren't comparable side by side — convert
    # each round to a percentage of that round's own respondent count.
    n_rounds = len(rounds)
    per_round = []  # (round_label, {item_label: pct_of_round}, n_responses)
    for round_label, df in rounds:
        raw_counts = df.apply(_count_column)
        n_responses = df.dropna(how="all").shape[0]
        by_label = {
            label_func(col): (float(v) / n_responses * 100 if n_responses else 0.0)
            for col, v in raw_counts.items()
        }
        per_round.append((round_label, by_label, n_responses))

    anchor_label, anchor_counts, _ = per_round[0]
    item_order = sorted(anchor_counts, key=lambda k: anchor_counts[k], reverse=True)
    for _, counts, _ in per_round[1:]:
        extras = [k for k in counts if k not in item_order]
        item_order += sorted(extras, key=lambda k: counts[k], reverse=True)

    n = len(item_order)
    fig_height = style.figure_height_for_rows(n)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.subplots_adjust(left=0.5, right=0.82)
    base_y = np.arange(n) * style.ROW_SPACING
    bar_height, offsets = _round_offsets(n_rounds, style.BAR_HEIGHT)

    round_handles = []
    for r_idx, (round_label, counts, n_responses) in enumerate(per_round):
        values = np.array([counts.get(item, 0.0) for item in item_order])
        y_pos = base_y + offsets[r_idx]
        rstyle = _round_style(round_label, n_rounds)
        ax.barh(y=y_pos, width=values, height=bar_height, color="white",
                linewidth=0.8, **rstyle)
        round_handles.append(Patch(facecolor="white", label=f"{round_label} (n={n_responses})", **rstyle))

    ax.set_yticks(base_y)
    ax.set_yticklabels([_mark_significant(item, item, significant_items) for item in item_order])
    ax.set_xlabel('Percentage of Respondents')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=round_handles, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), ncol=n_rounds, frameon=False)

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    _finalize_significance_marks(ax, item_order, significant_items)

    plt.savefig(savepath, format=savepath.split('.')[-1], bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Diverging usefulness / harmfulness (stacked horizontal, center-out)
# ---------------------------------------------------------------------------

def _plot_diverging_usefulness_harmfulness_single(
    df: pd.DataFrame,
    title_prefix: str,
    scale1_key: str = "Scale 1",
    scale2_key: str = "Scale 2",
    exclude_value: str = "I don't know",
    width: float = 14.0,
    show_legend: bool = True,
    savepath: str | None = None
):
    use_cols = [c for c in df.columns if scale1_key in c]
    harm_cols = [c for c in df.columns if scale2_key in c]

    use_data = {labels.extract_task_name(c): _percent_counts(df[c], USEFULNESS_LABELS, exclude_value) for c in use_cols}
    harm_data = {labels.extract_task_name(c): _percent_counts(df[c], HARMFULNESS_LABELS, exclude_value) for c in harm_cols}

    use_df = pd.DataFrame(use_data).T
    harm_df = pd.DataFrame(harm_data).T

    common_tasks = use_df.index.intersection(harm_df.index)
    use_df, harm_df = use_df.loc[common_tasks], harm_df.loc[common_tasks]

    useful_share = use_df[['Extremely useful', 'Very useful']].sum(axis=1)
    order_idx = useful_share.sort_values(ascending=False).index
    use_df, harm_df = use_df.loc[order_idx], harm_df.loc[order_idx]

    n_tasks = len(use_df)
    fig_height = style.figure_height_for_rows(n_tasks)
    fig, ax = plt.subplots(figsize=(width, fig_height))

    # shrink bar area, enlarge label area
    fig.subplots_adjust(left=0.5, right=0.82)

    y_pos = np.arange(n_tasks) * style.ROW_SPACING

    use_colors = style.COLORS["use_blues"]
    harm_colors = style.COLORS["harm_oranges"]

    hatch_use = "///"
    hatch_harm = "..."

    neg_cum = np.zeros(n_tasks)
    for cat, color in zip(USE_CENTER_OUT, use_colors):
        vals = use_df[cat].to_numpy()
        ax.barh(
            y_pos, -vals, left=-neg_cum,
            color=color, height=style.BAR_HEIGHT,
            edgecolor='black', linewidth=0.8,
            hatch=hatch_use
        )
        neg_cum += vals

    pos_cum = np.zeros(n_tasks)
    for cat, color in zip(HARM_CENTER_OUT, harm_colors):
        vals = harm_df[cat].to_numpy()
        ax.barh(
            y_pos, vals, left=pos_cum,
            color=color, height=style.BAR_HEIGHT,
            edgecolor='black', linewidth=0.8,
            hatch=hatch_harm
        )
        pos_cum += vals

    # Per-item valid counts (exclude "I don't know"). Usefulness and
    # harmfulness are separate questions per task and are answered by
    # different numbers of respondents, so each gets its own n.
    def _valid_counts(cols: list[str]) -> dict[str, int]:
        return {
            labels.extract_task_name(c): int(df[c][~df[c].isin([exclude_value])].notna().sum())
            for c in cols
        }

    use_valid_counts = _valid_counts(use_cols)
    harm_valid_counts = _valid_counts(harm_cols)

    y_labels = [
        f"{task} (use n={use_valid_counts.get(task, 0)}, harm n={harm_valid_counts.get(task, 0)})"
        for task in use_df.index
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel('Percentage of Respondents')
    ax.axvline(0, linewidth=1.2, color='black')  # slightly stronger centerline
    ax.set_xlim(-100, 100)
    ax.set_xticks(np.arange(-100, 101, 20))
    ax.set_xticklabels([str(abs(int(x))) for x in ax.get_xticks()])
    ax.invert_yaxis()

    # Side annotations (critical for B/W print + caption ambiguity)
    ax.text(0.01, 1.02, "← Usefulness", transform=ax.transAxes, ha="left", va="bottom")
    ax.text(0.99, 1.02, "Harmfulness →", transform=ax.transAxes, ha="right", va="bottom")

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)

    if show_legend:
        use_handles = [
            Patch(facecolor=c, edgecolor='black', linewidth=0.8, hatch=hatch_use, label=f"Usefulness: {l}")
            for l, c in zip(USE_CENTER_OUT, use_colors)
        ]
        harm_handles = [
            Patch(facecolor=c, edgecolor='black', linewidth=0.8, hatch=hatch_harm, label=f"Harmfulness: {l}")
            for l, c in zip(HARM_CENTER_OUT, harm_colors)
        ]

        legend_handles = use_handles + harm_handles
        ax.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc='lower center',
            bbox_to_anchor=(0.5, 1.12),
            ncol=2,
            frameon=False,
            title=None,
            fontsize=9
        )

    ax.set_title(title_prefix)

    if savepath:
        plt.savefig(savepath, format=savepath.split('.')[-1], bbox_inches="tight")
    plt.show()


def plot_diverging_usefulness_harmfulness(
    rounds: list[tuple[str, pd.DataFrame]],
    title_prefix: str,
    scale1_key: str = "Scale 1",
    scale2_key: str = "Scale 2",
    exclude_value: str = "I don't know",
    width: float = 14.0,
    show_legend: bool = True,
    savepath: str | None = None,
    significant_use: set[str] | None = None,
    significant_harm: set[str] | None = None,
):
    if len(rounds) == 1:
        _, df = rounds[0]
        return _plot_diverging_usefulness_harmfulness_single(
            df, title_prefix, scale1_key, scale2_key, exclude_value, width, show_legend, savepath
        )

    n_rounds = len(rounds)
    use_colors = style.COLORS["use_blues"]
    harm_colors = style.COLORS["harm_oranges"]
    hatch_use = "///"
    hatch_harm = "..."

    per_round = []  # (round_label, use_df, harm_df, use_valid_counts, harm_valid_counts)
    for round_label, df in rounds:
        use_cols = [c for c in df.columns if scale1_key in c]
        harm_cols = [c for c in df.columns if scale2_key in c]
        use_data = {labels.extract_task_name(c): _percent_counts(df[c], USEFULNESS_LABELS, exclude_value) for c in use_cols}
        harm_data = {labels.extract_task_name(c): _percent_counts(df[c], HARMFULNESS_LABELS, exclude_value) for c in harm_cols}
        use_df = pd.DataFrame(use_data).T
        harm_df = pd.DataFrame(harm_data).T
        common = use_df.index.intersection(harm_df.index)

        def _valid_counts(cols: list[str], df=df) -> dict[str, int]:
            return {
                labels.extract_task_name(c): int(df[c][~df[c].isin([exclude_value])].notna().sum())
                for c in cols
            }

        per_round.append((
            round_label, use_df.loc[common], harm_df.loc[common],
            _valid_counts(use_cols), _valid_counts(harm_cols),
        ))

    # Task order: anchor round's usefulness share, then any later-round-only
    # tasks appended (sorted by their own usefulness share).
    _, anchor_use, _, _, _ = per_round[0]
    anchor_share = anchor_use[['Extremely useful', 'Very useful']].sum(axis=1)
    task_order = list(anchor_share.sort_values(ascending=False).index)
    for _, use_df, _, _, _ in per_round[1:]:
        extra = [t for t in use_df.index if t not in task_order]
        if extra:
            extra_share = use_df.loc[extra, ['Extremely useful', 'Very useful']].sum(axis=1)
            task_order += list(extra_share.sort_values(ascending=False).index)

    n_tasks = len(task_order)
    fig_height = style.figure_height_for_rows(n_tasks)
    fig, ax = plt.subplots(figsize=(width, fig_height))
    fig.subplots_adjust(left=0.5, right=0.82)
    base_y = np.arange(n_tasks) * style.ROW_SPACING
    bar_height, offsets = _round_offsets(n_rounds, style.BAR_HEIGHT)

    round_handles = []
    for r_idx, (round_label, use_df, harm_df, _, _) in enumerate(per_round):
        use_df = use_df.reindex(task_order).fillna(0.0)
        harm_df = harm_df.reindex(task_order).fillna(0.0)
        y_pos = base_y + offsets[r_idx]
        rstyle = _round_style(round_label, n_rounds)

        neg_cum = np.zeros(n_tasks)
        for cat, color in zip(USE_CENTER_OUT, use_colors):
            vals = use_df[cat].to_numpy()
            ax.barh(y_pos, -vals, left=-neg_cum, color=color, height=bar_height,
                    linewidth=0.8, hatch=hatch_use, **rstyle)
            neg_cum += vals

        pos_cum = np.zeros(n_tasks)
        for cat, color in zip(HARM_CENTER_OUT, harm_colors):
            vals = harm_df[cat].to_numpy()
            ax.barh(y_pos, vals, left=pos_cum, color=color, height=bar_height,
                    linewidth=0.8, hatch=hatch_harm, **rstyle)
            pos_cum += vals

        round_handles.append(Patch(facecolor='0.8', label=round_label, **rstyle))

    def _use_harm_suffix(task: str) -> str:
        parts = []
        if significant_use is not None and task in significant_use:
            parts.append('Usefulness*')
        if significant_harm is not None and task in significant_harm:
            parts.append('Harmfulness*')
        return f" [{', '.join(parts)}]" if parts else ""

    y_labels = [
        f"{task} (" + "; ".join(
            f"{round_label}: use n={use_valid.get(task, 0)}, harm n={harm_valid.get(task, 0)}"
            for round_label, _, _, use_valid, harm_valid in per_round
        ) + ")" + _use_harm_suffix(task)
        for task in task_order
    ]
    ax.set_yticks(base_y)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel('Percentage of Respondents')
    ax.axvline(0, linewidth=1.2, color='black')
    ax.set_xlim(-100, 100)
    ax.set_xticks(np.arange(-100, 101, 20))
    ax.set_xticklabels([str(abs(int(x))) for x in ax.get_xticks()])
    ax.invert_yaxis()

    ax.text(0.01, 1.02, "← Usefulness", transform=ax.transAxes, ha="left", va="bottom")
    ax.text(0.99, 1.02, "Harmfulness →", transform=ax.transAxes, ha="right", va="bottom")

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)

    if significant_use is not None or significant_harm is not None:
        sig_either = (significant_use or set()) | (significant_harm or set())
        for text, task in zip(ax.get_yticklabels(), task_order):
            if task in sig_either:
                text.set_fontweight('bold')
        # Appended to the xlabel (not a floating ax.text annotation) so it's
        # laid out and measured like any other label instead of risking
        # overlap with the xlabel itself — see _finalize_significance_marks.
        ax.set_xlabel(f"{ax.get_xlabel()}\n{_SIGNIFICANCE_CAPTION}")

    if show_legend:
        use_handles = [
            Patch(facecolor=c, edgecolor='black', linewidth=0.8, hatch=hatch_use, label=f"Usefulness: {l}")
            for l, c in zip(USE_CENTER_OUT, use_colors)
        ]
        harm_handles = [
            Patch(facecolor=c, edgecolor='black', linewidth=0.8, hatch=hatch_harm, label=f"Harmfulness: {l}")
            for l, c in zip(HARM_CENTER_OUT, harm_colors)
        ]
        legend_handles = use_handles + harm_handles + round_handles
        ax.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc='lower center',
            bbox_to_anchor=(0.5, 1.14),
            ncol=2,
            frameon=False,
            title=None,
            fontsize=9,
        )

    ax.set_title(title_prefix)

    if savepath:
        plt.savefig(savepath, format=savepath.split('.')[-1], bbox_inches="tight")
    plt.show()
