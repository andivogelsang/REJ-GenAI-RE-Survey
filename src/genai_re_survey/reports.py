"""Assembles the demographics + RQ1-RQ4 figures for one round, or for a
round1-vs-round2 comparison, from a schema-normalized DataFrame (see
loading.py).

Column *blocks* (which columns belong to "prevention reasons", "elicitation
tasks", etc.) are selected by matching stable substrings of the question
text plus the `[bracket]` item label, not by position — this is what lets
the exact same selectors work against round 1 (202 cols) and round 2 (255
cols, differently ordered) without a second, hand-maintained index map. Every
selector here was validated against both real exports before being written.

generate_round_report() and generate_comparison_report() call the same
plotting.py functions — the former with a one-round list, the latter with a
two-round list — so a fix or new figure here always applies to both.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import labels, plotting, schema, style

# Canonical RE-activity bracket labels, in the order the paper's Figure 1
# uses them (RE in General is handled separately via the aliased
# `used_genai_for_re` column).
_ACTIVITY_BRACKETS = [
    "Requirements Elicitation",
    "Requirements Analysis &amp; Negotiation",
    "Requirements Specification / Requirements Modeling",
    "Requirements Validation / Quality Assurance",
    "Requirements Management",
]
_ACTIVITY_SHORT_NAMES = [
    "Elicitation", "Analysis and Negotiation", "Specification", "Validation", "Management",
]

# (block key, ALL-CAPS phase marker embedded in the question prefix, chart title)
PHASES = [
    ("elicitation", "REQUIREMENTS ELICITATION", "Elicitation"),
    ("analysis", "REQUIREMENTS ANALYSIS", "Analysis"),
    ("specification", "REQUIREMENTS SPECIFICATION", "Specification and Modeling"),
    ("validation", "REQUIREMENTS VALIDATION", "Validation and QA"),
    ("management", "REQUIREMENTS MANAGEMENT", "Requirements Management"),
]


def _is_comment_column(col: str) -> bool:
    brackets = schema.BRACKET_RE.findall(col)
    if not brackets:
        return False
    return brackets[-1].strip().lower().endswith("comment")


# ---------------------------------------------------------------------------
# Column-block selectors
# ---------------------------------------------------------------------------

def _select_application_domain_columns(df: pd.DataFrame) -> list[str]:
    anchor = "In which application domain(s) have you worked over the past 5 years?"
    return [c for c in df.columns if anchor in c]


def _usage_activity_column(df: pd.DataFrame, activity_bracket: str) -> str:
    for c in df.columns:
        if "did you use / apply GenAI" in c and c.endswith(f"[{activity_bracket}]"):
            return c
    raise KeyError(f"usage column for activity {activity_bracket!r} not found")


def _select_prevention_block(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if "reasons prevent" in c and not _is_comment_column(c)]
    return df[cols]


def _select_threats_block(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if "concerning AI in RE" in c and not _is_comment_column(c)]
    return df[cols]


def _select_training_interest_block(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in df.columns
        if "would you like to receive" in c and "training" in c and not _is_comment_column(c)
    ]
    return df[cols]


def _select_training_format_block(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if "training format would you prefer" in c and not _is_comment_column(c)]
    return df[cols]


def _select_phase_block(df: pd.DataFrame, phase_marker: str) -> pd.DataFrame:
    cols = [c for c in df.columns if phase_marker in c]
    return df[cols]


def _skills_value_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "skill set of requirements engineers" in c and not _is_comment_column(c):
            return c
    raise KeyError("skills value column not found")


def _skills_comment_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "skill set of requirements engineers" in c and _is_comment_column(c):
            return c
    raise KeyError("skills comment column not found")


# ---------------------------------------------------------------------------
# Raw-df -> plot-ready table
# ---------------------------------------------------------------------------

def _usage_percentage_df(df: pd.DataFrame) -> pd.DataFrame:
    # Index is the *clean* item name (no n baked in) — plotting.py aligns
    # round1/round2 rows by this index, and round-specific n's differ, so
    # embedding n here would make every item look round-unique and silently
    # break the round-to-round pairing. n travels in its own column instead.
    cols = ["used_genai_for_re"] + [_usage_activity_column(df, a) for a in _ACTIVITY_BRACKETS]
    names = ["RE in General"] + _ACTIVITY_SHORT_NAMES

    yes_counts, no_counts, n_counts = [], [], []
    for col in cols:
        vc = df[col].value_counts()
        yes_counts.append(vc.get('Yes', 0))
        no_counts.append(vc.get('No', 0))
        n_counts.append(df[col].notna().sum())

    plot_df = pd.DataFrame({'Yes': yes_counts, 'No': no_counts}, index=names)
    plot_df_percentage = plot_df.apply(lambda x: x / x.sum() * 100 if x.sum() else x, axis=1)
    plot_df_percentage['n'] = n_counts
    return plot_df_percentage


def _skills_percentage_df(df: pd.DataFrame) -> pd.DataFrame:
    col = _skills_value_column(df)
    vc = df[col].value_counts()
    yes, no = vc.get('Yes', 0), vc.get('No', 0)
    n = df[col].notna().sum()
    total = yes + no
    pct = {
        'Yes': [yes / total * 100 if total else 0],
        'No': [no / total * 100 if total else 0],
        'n': [n],
    }
    return pd.DataFrame(pct, index=["Skill set will change"])


def _application_domain_counts(df: pd.DataFrame) -> pd.Series:
    cols = _select_application_domain_columns(df)
    counts = df[cols].apply(lambda x: (x == 'Yes').sum())
    counts.index = [labels.label_from_brackets(c) for c in counts.index]
    return counts.sort_values(ascending=False)


# Ordinal single-choice questions about the respondent's own GenAI usage
# habits — collected and cross-round-aliased in schema.py, but general
# demographic/context questions rather than RE-task-specific RQ1-4 content.
FREQUENCY_LABELS = [
    'Never used', 'Tried once', 'At least once a month',
    'At least once a week', 'Daily', 'Other',
]
DURATION_LABELS = [
    'No Experience at all', 'Less than 1 year', '1-2 years',
    '3-4 years', 'More than 4 years',
]


def _categorical_percentage_df(df: pd.DataFrame, col: str, categories: list[str], row_label: str) -> pd.DataFrame:
    s = df[col].dropna()
    unexpected = set(s.unique()) - set(categories)
    if unexpected:
        raise ValueError(f"{col!r} has values outside `categories`: {sorted(unexpected)}")
    n = len(s)
    vc = s.value_counts()
    counts = pd.Series({cat: vc.get(cat, 0) for cat in categories}, dtype=float)
    pct = (counts / n * 100) if n else counts
    result = pd.DataFrame([pct.to_numpy()], columns=categories, index=[row_label])
    result['n'] = n
    return result


def _tool_usage_frequency_df(df: pd.DataFrame) -> pd.DataFrame:
    return _categorical_percentage_df(
        df, 'genai_tool_usage_frequency', FREQUENCY_LABELS, 'GenAI Tool Usage Frequency'
    )


def _experience_duration_df(df: pd.DataFrame) -> pd.DataFrame:
    return _categorical_percentage_df(
        df, 'genai_experience_duration', DURATION_LABELS, 'GenAI Experience Duration'
    )


# ---------------------------------------------------------------------------
# Qualitative (free-text) comments
# ---------------------------------------------------------------------------

def _select_usage_comment_block(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if "did you use / apply GenAI" in c and _is_comment_column(c)]
    return df[cols]


def _select_training_interest_comment_block(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in df.columns
        if "would you like to receive" in c and "training" in c and _is_comment_column(c)
    ]
    return df[cols]


def _print_nonempty_comments(df: pd.DataFrame, columns: list[str], heading: str) -> None:
    print(heading)
    for col in columns:
        non_empty = df[col].dropna()
        if not non_empty.empty:
            print(f"\n--- {col} ---")
            for idx, val in non_empty.items():
                print(f"{idx}: {val}")


def print_qualitative_comments(df: pd.DataFrame, round_name: str) -> None:
    """Print every non-empty free-text comment collected alongside the
    quantitative Yes/No blocks (usage, training interest, skills). These
    columns are deliberately excluded from the Yes/No charts (see
    `_is_comment_column`) since they're prose, not categorical data — this
    is where that prose actually gets read.
    """
    _print_nonempty_comments(
        df, list(_select_usage_comment_block(df).columns),
        f"=== {round_name}: GenAI usage — free-text comments ===",
    )
    print()
    _print_nonempty_comments(
        df, list(_select_training_interest_comment_block(df).columns),
        f"=== {round_name}: Training interest — free-text comments ===",
    )
    print()
    _print_nonempty_comments(
        df, [_skills_comment_column(df)],
        f"=== {round_name}: Skill set change — free-text comments ===",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_round_report(df: pd.DataFrame, round_name: str, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    style.apply_paper_style()

    # Demographics
    domain_counts = _application_domain_counts(df)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=domain_counts.index, y=domain_counts.values)
    plt.xticks(rotation=90)
    plt.title('Distribution of Application Domains Worked In')
    plt.xlabel('Application Domain')
    plt.ylabel('Number of Respondents')
    plt.tight_layout()
    plt.savefig(outdir / "application_domains.pdf", bbox_inches="tight")
    plt.show()

    # Demographics: GenAI usage frequency / experience duration
    plotting.plot_stacked_percentage_barh(
        [(round_name, _tool_usage_frequency_df(df))],
        categories=FREQUENCY_LABELS,
        colors=style.COLORS['frequency_purples'] + [style.COLORS['neutral_other']],
        title="GenAI Tool Usage Frequency",
        savepath=str(outdir / "genai_tool_usage_frequency.pdf"),
    )
    plotting.plot_stacked_percentage_barh(
        [(round_name, _experience_duration_df(df))],
        categories=DURATION_LABELS,
        colors=style.COLORS['frequency_purples'],
        title="GenAI Experience Duration",
        savepath=str(outdir / "genai_experience_duration.pdf"),
    )

    # RQ1: usage
    plotting.plot_stacked_percentage_barh(
        [(round_name, _usage_percentage_df(df))],
        categories=['Yes', 'No'],
        colors=[style.COLORS['yes'], style.COLORS['no']],
        title="GenAI Usage by RE Discipline",
        savepath=str(outdir / "usage.pdf"),
    )

    # RQ2: prevention + threats
    plotting.plot_yes_counts_barh(
        [(round_name, _select_prevention_block(df))],
        title="Reasons Preventing GenAI Usage in RE",
        ylabel="Reason",
        label_func=labels.label_strip_brackets_and_parens,
        savepath=str(outdir / "prevention.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _select_threats_block(df))],
        title="Perceived Threats",
        ylabel="Limitation / Threat",
        label_func=labels.label_strip_brackets_and_parens,
        savepath=str(outdir / "threats.pdf"),
    )

    # RQ3: usefulness/harmfulness per RE phase
    for phase_key, phase_marker, title in PHASES:
        phase_df = _select_phase_block(df, phase_marker)
        plotting.plot_diverging_usefulness_harmfulness(
            [(round_name, phase_df)],
            title_prefix=title,
            savepath=str(outdir / f"{phase_key}_assessment.pdf"),
            show_legend=False,
        )

    # RQ4: skills + training
    plotting.plot_stacked_percentage_barh(
        [(round_name, _skills_percentage_df(df))],
        categories=['Yes', 'No'],
        colors=[style.COLORS['yes'], style.COLORS['no']],
        title="Skill Set Change",
        savepath=str(outdir / "skills.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _select_training_interest_block(df))],
        title="Training Interest",
        ylabel="Training Interest",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "training_interest.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _select_training_format_block(df))],
        title="Preferred Training Formats",
        ylabel="Training Format",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "training_format.pdf"),
    )

    print_qualitative_comments(df, round_name)


def generate_comparison_report(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    outdir: str | Path,
    round1_label: str = "Round 1",
    round2_label: str = "Round 2",
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    style.apply_paper_style()

    rounds_of = lambda f: [(round1_label, f(df1)), (round2_label, f(df2))]

    plotting.plot_stacked_percentage_barh(
        rounds_of(_tool_usage_frequency_df),
        categories=FREQUENCY_LABELS,
        colors=style.COLORS['frequency_purples'] + [style.COLORS['neutral_other']],
        title="GenAI Tool Usage Frequency",
        savepath=str(outdir / "genai_tool_usage_frequency_comparison.pdf"),
    )
    plotting.plot_stacked_percentage_barh(
        rounds_of(_experience_duration_df),
        categories=DURATION_LABELS,
        colors=style.COLORS['frequency_purples'],
        title="GenAI Experience Duration",
        savepath=str(outdir / "genai_experience_duration_comparison.pdf"),
    )

    plotting.plot_stacked_percentage_barh(
        rounds_of(_usage_percentage_df),
        categories=['Yes', 'No'],
        colors=[style.COLORS['yes'], style.COLORS['no']],
        title="GenAI Usage by RE Discipline",
        savepath=str(outdir / "usage_comparison.pdf"),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_prevention_block),
        title="Reasons Preventing GenAI Usage in RE",
        ylabel="Reason",
        label_func=labels.label_strip_brackets_and_parens,
        savepath=str(outdir / "prevention_comparison.pdf"),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_threats_block),
        title="Perceived Threats",
        ylabel="Limitation / Threat",
        label_func=labels.label_strip_brackets_and_parens,
        savepath=str(outdir / "threats_comparison.pdf"),
    )
    for phase_key, phase_marker, title in PHASES:
        plotting.plot_diverging_usefulness_harmfulness(
            rounds_of(lambda df, m=phase_marker: _select_phase_block(df, m)),
            title_prefix=title,
            savepath=str(outdir / f"{phase_key}_assessment_comparison.pdf"),
            show_legend=False,
        )
    plotting.plot_stacked_percentage_barh(
        rounds_of(_skills_percentage_df),
        categories=['Yes', 'No'],
        colors=[style.COLORS['yes'], style.COLORS['no']],
        title="Skill Set Change",
        savepath=str(outdir / "skills_comparison.pdf"),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_training_interest_block),
        title="Training Interest",
        ylabel="Training Interest",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "training_interest_comparison.pdf"),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_training_format_block),
        title="Preferred Training Formats",
        ylabel="Training Format",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "training_format_comparison.pdf"),
    )

    return schema.diff_schemas(df1, df2)
