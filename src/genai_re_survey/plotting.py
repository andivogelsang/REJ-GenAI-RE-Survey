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
# Yes/No percentage (stacked horizontal bar)
# ---------------------------------------------------------------------------

def _plot_yesno_percentage_barh_single(
    plot_df_percentage: pd.DataFrame,   # index already set to labels with (n=…)
    title: str,
    savepath: str
):
    n_tasks = len(plot_df_percentage)
    fig_height = style.figure_height_for_rows(n_tasks)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y_pos = np.arange(n_tasks) * style.ROW_SPACING

    categories = ['Yes', 'No']
    cat_colors = [style.COLORS['yes'], style.COLORS['no']]

    left_vals = np.zeros(n_tasks)
    for cat, color in zip(categories, cat_colors):
        vals = plot_df_percentage[cat].to_numpy()
        ax.barh(y_pos, vals, left=left_vals, color=color,
                height=style.BAR_HEIGHT, edgecolor='black', linewidth=0.8, label=cat)
        left_vals += vals

    # In-bar labels (percent, no decimals)
    for i, row in enumerate(plot_df_percentage[categories].to_numpy()):
        x_left = 0.0
        for width_val in row:
            if width_val > 0:
                ax.text(x_left + width_val / 2, y_pos[i], f"{width_val:.0f}%",
                        ha='center', va='center', fontsize=14, color='white')
                x_left += width_val

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df_percentage.index)
    ax.set_xlabel('Percentage of Respondents')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_title(title)

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        title=None
    )

    plt.tight_layout()
    plt.savefig(savepath, format=savepath.split('.')[-1], bbox_inches="tight")
    plt.show()


def plot_yesno_percentage_barh(
    rounds: list[tuple[str, pd.DataFrame]],
    title: str,
    savepath: str,
):
    if len(rounds) == 1:
        _, df = rounds[0]
        return _plot_yesno_percentage_barh_single(df, title, savepath)

    n_rounds = len(rounds)
    categories = ['Yes', 'No']
    cat_colors = [style.COLORS['yes'], style.COLORS['no']]

    item_order = list(rounds[0][1].index)
    for _, df in rounds[1:]:
        item_order += [item for item in df.index if item not in item_order]
    n_tasks = len(item_order)

    fig_height = style.figure_height_for_rows(n_tasks)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    base_y = np.arange(n_tasks) * style.ROW_SPACING
    bar_height, offsets = _round_offsets(n_rounds, style.BAR_HEIGHT)

    round_handles = []
    for r_idx, (round_label, df) in enumerate(rounds):
        df = df.reindex(item_order).fillna(0.0)
        y_pos = base_y + offsets[r_idx]
        rstyle = _round_style(round_label, n_rounds)
        left_vals = np.zeros(n_tasks)
        for cat, color in zip(categories, cat_colors):
            vals = df[cat].to_numpy()
            ax.barh(y_pos, vals, left=left_vals, color=color,
                    height=bar_height, linewidth=0.8,
                    label=(cat if r_idx == 0 else None), **rstyle)
            left_vals += vals
        round_handles.append(Patch(facecolor='0.6', label=round_label, **rstyle))

    ax.set_yticks(base_y)
    ax.set_yticklabels(item_order)
    ax.set_xlabel('Percentage of Respondents')
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_title(title)

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)

    handles, legend_labels = ax.get_legend_handles_labels()
    handles += round_handles
    legend_labels += [h.get_label() for h in round_handles]
    ax.legend(
        handles, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        title=None
    )

    plt.tight_layout()
    plt.savefig(savepath, format=savepath.split('.')[-1], bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# "Number of Yes" horizontal bar (prevention / threats / training charts)
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

    labels_ = [label_func(col) for col in counts.index]

    n = len(labels_)
    fig_height = style.figure_height_for_rows(n)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    # shrink bar area, enlarge label area
    fig.subplots_adjust(left=0.5, right=0.82)
    y_pos = np.arange(n) * style.ROW_SPACING

    ax.barh(
        y=y_pos,
        width=counts.values,
        height=style.BAR_HEIGHT,
        color="white",
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_)
    ax.set_xlabel(f'Number of Respondents (n={n_responses})')
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
):
    if len(rounds) == 1:
        _, df = rounds[0]
        return _plot_yes_counts_barh_single(df, title, ylabel, label_func, savepath)

    n_rounds = len(rounds)
    per_round = []  # (round_label, {item_label: count}, n_responses)
    for round_label, df in rounds:
        raw_counts = df.apply(_count_column)
        n_responses = df.dropna(how="all").shape[0]
        by_label = {label_func(col): int(v) for col, v in raw_counts.items()}
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
        values = np.array([counts.get(item, 0) for item in item_order])
        y_pos = base_y + offsets[r_idx]
        rstyle = _round_style(round_label, n_rounds)
        ax.barh(y=y_pos, width=values, height=bar_height, color="white",
                linewidth=0.8, **rstyle)
        round_handles.append(Patch(facecolor="white", label=f"{round_label} (n={n_responses})", **rstyle))

    ax.set_yticks(base_y)
    ax.set_yticklabels(item_order)
    ax.set_xlabel('Number of Respondents')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=round_handles, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), ncol=n_rounds, frameon=False)

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)

    plt.savefig(savepath, format=savepath.split('.')[-1])
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

    # Per-item valid counts (exclude "I don't know"), unified label format
    relevant_cols = [c for c in df.columns if (scale1_key in c or scale2_key in c)]
    valid_counts = {}
    for c in relevant_cols:
        task = labels.extract_task_name(c)
        s = df[c]
        valid_counts[task] = s[~s.isin([exclude_value])].notna().sum()

    y_labels = [f"{task} (n={valid_counts.get(task, 0)})" for task in use_df.index]
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

    per_round = []  # (round_label, use_df, harm_df)
    for round_label, df in rounds:
        use_cols = [c for c in df.columns if scale1_key in c]
        harm_cols = [c for c in df.columns if scale2_key in c]
        use_data = {labels.extract_task_name(c): _percent_counts(df[c], USEFULNESS_LABELS, exclude_value) for c in use_cols}
        harm_data = {labels.extract_task_name(c): _percent_counts(df[c], HARMFULNESS_LABELS, exclude_value) for c in harm_cols}
        use_df = pd.DataFrame(use_data).T
        harm_df = pd.DataFrame(harm_data).T
        common = use_df.index.intersection(harm_df.index)
        per_round.append((round_label, use_df.loc[common], harm_df.loc[common]))

    # Task order: anchor round's usefulness share, then any later-round-only
    # tasks appended (sorted by their own usefulness share).
    _, anchor_use, _ = per_round[0]
    anchor_share = anchor_use[['Extremely useful', 'Very useful']].sum(axis=1)
    task_order = list(anchor_share.sort_values(ascending=False).index)
    for _, use_df, _ in per_round[1:]:
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
    for r_idx, (round_label, use_df, harm_df) in enumerate(per_round):
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

    ax.set_yticks(base_y)
    ax.set_yticklabels(task_order)

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
