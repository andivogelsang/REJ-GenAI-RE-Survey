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

import pandas as pd

from . import labels, plotting, schema, stats, style

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


def _select_role_block(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if "current role or position" in c and not _is_comment_column(c)]
    return df[cols]


def _select_region_block(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if "regions do you typically work" in c and not _is_comment_column(c)]
    return df[cols]


def _select_org_type_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if schema.ORG_TYPE_ANCHOR in c and not c.endswith("[Other]"):
            return c
    raise KeyError("organization type column not found")


def _select_years_re_experience_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if schema.YEARS_RE_EXPERIENCE_ANCHOR in c:
            return c
    raise KeyError("years-of-RE-experience column not found")


def _one_hot_yesno_block(df: pd.DataFrame, col: str, categories: list[str]) -> pd.DataFrame:
    """Turn a single-select categorical column into one Yes/No column per
    category (column name = category label), so it can be charted with
    plot_yes_counts_barh exactly like the role/region multi-select blocks —
    one bar per category — instead of a single 100%-stacked bar.
    """
    s = df[col]
    data = {}
    for cat in categories:
        col_vals = pd.Series(index=s.index, dtype=object)
        col_vals[s.notna()] = 'No'
        col_vals[s == cat] = 'Yes'
        data[cat] = col_vals
    return pd.DataFrame(data)


def _years_of_re_experience_block(df: pd.DataFrame) -> pd.DataFrame:
    return _one_hot_yesno_block(df, _select_years_re_experience_column(df), YEARS_RE_EXPERIENCE_LABELS)


def _org_type_block(df: pd.DataFrame) -> pd.DataFrame:
    return _one_hot_yesno_block(df, _select_org_type_column(df), ORG_TYPE_LABELS)


def _participated_before_block(df: pd.DataFrame) -> pd.DataFrame:
    return _one_hot_yesno_block(
        df, "Did you participate in our previous survey as well?", PARTICIPATED_BEFORE_LABELS
    )


def _select_other_freetext_column(df: pd.DataFrame, anchor: str) -> str | None:
    for c in df.columns:
        if anchor in c and c.endswith("[Other]"):
            return c
    return None


def _re_discipline_experience_column(df: pd.DataFrame, activity_bracket: str) -> str:
    anchor = "Please assess your knowledge / experience in the various"
    for c in df.columns:
        if anchor in c and c.endswith(f"[{activity_bracket}]"):
            return c
    raise KeyError(f"RE-discipline experience column for {activity_bracket!r} not found")


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

def _multi_item_categorical_percentage_df(
    df: pd.DataFrame, cols: list[str], names: list[str], categories: list[str]
) -> pd.DataFrame:
    # Index is the *clean* item name (no n baked in) — plotting.py aligns
    # round1/round2 rows by this index, and round-specific n's differ, so
    # embedding n here would make every item look round-unique and silently
    # break the round-to-round pairing. n travels in its own column instead.
    rows, n_counts = [], []
    for col in cols:
        vc = df[col].value_counts()
        n = df[col].notna().sum()
        counts = pd.Series({cat: vc.get(cat, 0) for cat in categories}, dtype=float)
        rows.append((counts / n * 100) if n else counts)
        n_counts.append(n)

    result = pd.DataFrame(rows, index=names)[categories]
    result['n'] = n_counts
    return result


def _usage_percentage_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["used_genai_for_re"] + [_usage_activity_column(df, a) for a in _ACTIVITY_BRACKETS]
    names = ["RE in General"] + _ACTIVITY_SHORT_NAMES
    return _multi_item_categorical_percentage_df(df, cols, names, ['Yes', 'No'])


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


# Ordinal/nominal single- or multi-choice demographic questions — collected
# (and, where reworded between rounds, cross-round-aliased in schema.py) but
# general context about the respondent rather than RE-task-specific RQ1-4
# content. Mirrors loading._DEMOGRAPHIC_ANCHORS: every variable named there
# gets a chart/summary via this section.
FREQUENCY_LABELS = [
    'Never used', 'Tried once', 'At least once a month',
    'At least once a week', 'Daily', 'Other',
]
DURATION_LABELS = [
    'No Experience at all', 'Less than 1 year', '1-2 years',
    '3-4 years', 'More than 4 years',
]
YEARS_RE_EXPERIENCE_LABELS = [
    'none / new to RE', '< 2 years', '3-5 years', '6-10 years', '> 10 years',
]
ORG_TYPE_LABELS = [
    'Industry - Micro enterprise (up to 10 employees)',
    'Industry - Small enterprise (10-49 employees)',
    'Industry - Medium-sized enterprise (50-249 employees)',
    'Industry - Large enterprise (at least 250 employees)',
    'Research-technology transfer',
    'University / Research',
    'Other',
]
RE_DISCIPLINE_EXPERIENCE_LABELS = ['No Experience', 'Beginner', 'Intermediate', 'Advanced', 'Expert']
PARTICIPATED_BEFORE_LABELS = ['Yes', 'No', "I don't remember"]


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


def _re_discipline_experience_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [_re_discipline_experience_column(df, a) for a in _ACTIVITY_BRACKETS]
    return _multi_item_categorical_percentage_df(df, cols, _ACTIVITY_SHORT_NAMES, RE_DISCIPLINE_EXPERIENCE_LABELS)


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

    org_type_other = _select_other_freetext_column(df, "organization / business types")
    if org_type_other:
        print()
        _print_nonempty_comments(
            df, [org_type_other],
            f"=== {round_name}: Organization type — 'Other' free text ===",
        )

    role_other = _select_other_freetext_column(df, "current role or position")
    if role_other:
        print()
        _print_nonempty_comments(
            df, [role_other],
            f"=== {round_name}: Role / position — 'Other' free text ===",
        )

    genai_tools_col = "Which GenAI tools do you use primarily in the context of your work?"
    if genai_tools_col in df.columns:
        print()
        _print_nonempty_comments(
            df, [genai_tools_col],
            f"=== {round_name}: Primary GenAI tools used — free text (round 2 only) ===",
        )


# ---------------------------------------------------------------------------
# Descriptive statistics (respondent demographics)
# ---------------------------------------------------------------------------

def _describe_single_select(df: pd.DataFrame, col: str, categories: list[str]) -> tuple[list[tuple[str, int, float]], int]:
    s = df[col].dropna()
    n = len(s)
    vc = s.value_counts()
    rows = [(cat, int(vc.get(cat, 0)), (vc.get(cat, 0) / n * 100 if n else 0.0)) for cat in categories]
    return rows, n


def _describe_multi_select(df: pd.DataFrame, cols: list[str], label_func) -> tuple[list[tuple[str, int, float]], int]:
    # Same convention as plotting._count_column / plot_yes_counts_barh: a
    # shared per-block n (anyone who answered at least one item in the
    # block), and non-Yes/No columns (e.g. a free-text "[Other]" write-in)
    # count every non-null response rather than only "Yes".
    n = df[cols].dropna(how="all").shape[0]
    rows = []
    for col in cols:
        s = df[col].dropna()
        is_yes_no = set(s.unique()) <= {'Yes', 'No'}
        count = int((s == 'Yes').sum()) if is_yes_no else int(s.shape[0])
        rows.append((label_func(col), count, (count / n * 100 if n else 0.0)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows, n


def _print_single_select_summary(df: pd.DataFrame, col: str, categories: list[str], heading: str) -> None:
    rows, n = _describe_single_select(df, col, categories)
    print(f"{heading} (n={n})")
    for cat, count, pct in rows:
        print(f"  {cat}: {count} ({pct:.0f}%)")


def _print_multi_select_summary(df: pd.DataFrame, cols: list[str], label_func, heading: str) -> None:
    rows, n = _describe_multi_select(df, cols, label_func)
    print(f"{heading} (n={n}, % of respondents who selected — multi-select, need not sum to 100%)")
    for label, count, pct in rows:
        print(f"  {label}: {count} ({pct:.0f}%)")


def print_demographic_summary(df: pd.DataFrame, round_name: str) -> None:
    """Print n / % per category for every demographic variable named in
    `loading._DEMOGRAPHIC_ANCHORS` — the columns that decide who counts as a
    "real" respondent (see loading.py) get a numeric summary here, not just
    a chart, so exact figures don't have to be read off a bar's pixel width.
    """
    print(f"=== {round_name}: Respondent demographics ===\n")

    _print_single_select_summary(
        df, _select_years_re_experience_column(df), YEARS_RE_EXPERIENCE_LABELS, "Years of RE experience"
    )
    print()
    _print_single_select_summary(df, _select_org_type_column(df), ORG_TYPE_LABELS, "Organization type")
    print()
    _print_multi_select_summary(
        df, list(_select_role_block(df).columns), labels.label_from_brackets, "Role / position"
    )
    print()
    _print_multi_select_summary(
        df, list(_select_region_block(df).columns), labels.label_from_brackets, "Region"
    )
    print()
    _print_multi_select_summary(
        df, _select_application_domain_columns(df), labels.label_from_brackets, "Application domain"
    )
    print()
    for activity, short_name in zip(_ACTIVITY_BRACKETS, _ACTIVITY_SHORT_NAMES):
        _print_single_select_summary(
            df, _re_discipline_experience_column(df, activity), RE_DISCIPLINE_EXPERIENCE_LABELS,
            f"RE-discipline self-assessed experience — {short_name}",
        )
        print()
    _print_single_select_summary(df, 'genai_tool_usage_frequency', FREQUENCY_LABELS, "GenAI tool usage frequency")
    print()
    _print_single_select_summary(df, 'genai_experience_duration', DURATION_LABELS, "GenAI experience duration")

    participated_col = "Did you participate in our previous survey as well?"
    if participated_col in df.columns:
        print()
        _print_single_select_summary(df, participated_col, PARTICIPATED_BEFORE_LABELS, "Participated in previous survey")


def print_demographic_comparison(
    df1: pd.DataFrame, df2: pd.DataFrame, round1_label: str = "Round 1", round2_label: str = "Round 2"
) -> None:
    """Side-by-side round1 vs round2 n/% per category, for the same
    variables as `print_demographic_summary` — lets a reader see directly
    whether e.g. round 2 skews more senior or more Europe-heavy than round 1
    before trusting the RQ1-4 round-to-round comparisons.
    """
    print(f"=== {round1_label} vs {round2_label}: Respondent demographics ===\n")

    def _compare_single(col_getter, categories: list[str], heading: str) -> None:
        rows1, n1 = _describe_single_select(df1, col_getter(df1), categories)
        rows2, n2 = _describe_single_select(df2, col_getter(df2), categories)
        by_cat2 = {cat: (count, pct) for cat, count, pct in rows2}
        print(f"{heading} ({round1_label} n={n1}, {round2_label} n={n2})")
        for cat, count1, pct1 in rows1:
            count2, pct2 = by_cat2.get(cat, (0, 0.0))
            print(f"  {cat}: {round1_label} {count1} ({pct1:.0f}%) | {round2_label} {count2} ({pct2:.0f}%)")

    def _compare_multi(cols_getter, label_func, heading: str) -> None:
        rows1, n1 = _describe_multi_select(df1, cols_getter(df1), label_func)
        rows2, n2 = _describe_multi_select(df2, cols_getter(df2), label_func)
        by_label2 = {label: (count, pct) for label, count, pct in rows2}
        print(f"{heading} ({round1_label} n={n1}, {round2_label} n={n2}, % who selected)")
        for label, count1, pct1 in rows1:
            count2, pct2 = by_label2.get(label, (0, 0.0))
            print(f"  {label}: {round1_label} {count1} ({pct1:.0f}%) | {round2_label} {count2} ({pct2:.0f}%)")

    _compare_single(_select_years_re_experience_column, YEARS_RE_EXPERIENCE_LABELS, "Years of RE experience")
    print()
    _compare_single(_select_org_type_column, ORG_TYPE_LABELS, "Organization type")
    print()
    _compare_multi(lambda d: list(_select_role_block(d).columns), labels.label_from_brackets, "Role / position")
    print()
    _compare_multi(lambda d: list(_select_region_block(d).columns), labels.label_from_brackets, "Region")
    print()
    _compare_multi(_select_application_domain_columns, labels.label_from_brackets, "Application domain")
    print()
    for activity, short_name in zip(_ACTIVITY_BRACKETS, _ACTIVITY_SHORT_NAMES):
        _compare_single(
            lambda d, a=activity: _re_discipline_experience_column(d, a), RE_DISCIPLINE_EXPERIENCE_LABELS,
            f"RE-discipline self-assessed experience — {short_name}",
        )
        print()
    _compare_single(lambda d: 'genai_tool_usage_frequency', FREQUENCY_LABELS, "GenAI tool usage frequency")
    print()
    _compare_single(lambda d: 'genai_experience_duration', DURATION_LABELS, "GenAI experience duration")


# ---------------------------------------------------------------------------
# Significance testing (round1 vs round2)
# ---------------------------------------------------------------------------

def _matched_items(
    df1: pd.DataFrame, cols1: list[str], df2: pd.DataFrame, cols2: list[str], label_func
) -> list[tuple[str, pd.Series, pd.Series]]:
    """Pair up columns from df1/df2 by their clean label (same idea as
    plotting.py's round-alignment logic) — needed because a question's raw
    column text sometimes differs slightly between rounds even where the
    bracket/item label doesn't.
    """
    by_label1 = {label_func(c): c for c in cols1}
    by_label2 = {label_func(c): c for c in cols2}
    return [(label, df1[col], df2[by_label2[label]]) for label, col in by_label1.items() if label in by_label2]


def compare_round_significance(df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Round1-vs-round2 significance test for every family that already has
    a comparison chart: Fisher's exact for Yes/No items, Mann-Whitney U for
    ordinal items (5-point Likert usefulness/harmfulness, experience-level
    scales). Benjamini-Hochberg FDR correction is applied *within* each
    family — the same grouping each chart already uses — not once across
    every test, so one item's family doesn't inflate another's.
    """
    results: dict[str, pd.DataFrame] = {}

    usage_items = [("RE in General", df1['used_genai_for_re'], df2['used_genai_for_re'])] + [
        (short_name, df1[_usage_activity_column(df1, bracket)], df2[_usage_activity_column(df2, bracket)])
        for bracket, short_name in zip(_ACTIVITY_BRACKETS, _ACTIVITY_SHORT_NAMES)
    ]
    results['RQ1: Usage by RE discipline'] = stats.compare_categorical_family(usage_items)

    results['RQ2: Prevention reasons'] = stats.compare_categorical_family(_matched_items(
        df1, list(_select_prevention_block(df1).columns), df2, list(_select_prevention_block(df2).columns),
        labels.label_strip_brackets_and_parens,
    ))
    results['RQ2: Perceived threats'] = stats.compare_categorical_family(_matched_items(
        df1, list(_select_threats_block(df1).columns), df2, list(_select_threats_block(df2).columns),
        labels.label_strip_brackets_and_parens,
    ))

    # Training-interest and training-format each have their own unrelated
    # free-text "Other" write-in item; both reduce to the bare label "Other"
    # under labels.label_from_brackets, so without a prefix they'd collide
    # into a single, ambiguous "Other" row in this shared family. Prefixed
    # here; generate_comparison_report strips the prefix back off per-chart
    # via _significant_with_prefix_stripped (same pattern used for the
    # "Demographics: Experience (ordinal)" family's RE-discipline items).
    rq4_items = (
        [("Skill set will change", df1[_skills_value_column(df1)], df2[_skills_value_column(df2)])]
        + [
            (f"Training interest: {label}", s1, s2) for label, s1, s2 in _matched_items(
                df1, list(_select_training_interest_block(df1).columns),
                df2, list(_select_training_interest_block(df2).columns), labels.label_from_brackets,
            )
        ]
        + [
            (f"Training format: {label}", s1, s2) for label, s1, s2 in _matched_items(
                df1, list(_select_training_format_block(df1).columns),
                df2, list(_select_training_format_block(df2).columns), labels.label_from_brackets,
            )
        ]
    )
    results['RQ4: Skills + training'] = stats.compare_categorical_family(rq4_items)

    results['Demographics: Role / position'] = stats.compare_categorical_family(_matched_items(
        df1, list(_select_role_block(df1).columns), df2, list(_select_role_block(df2).columns),
        labels.label_from_brackets,
    ))
    results['Demographics: Region'] = stats.compare_categorical_family(_matched_items(
        df1, list(_select_region_block(df1).columns), df2, list(_select_region_block(df2).columns),
        labels.label_from_brackets,
    ))
    results['Demographics: Application domain'] = stats.compare_categorical_family(_matched_items(
        df1, _select_application_domain_columns(df1), df2, _select_application_domain_columns(df2),
        labels.label_from_brackets,
    ))

    # Organization type is single-select/nominal (not ordinal), so it's
    # tested one-vs-rest per category, reusing the same one-hot block the
    # bar chart already builds (_org_type_block) rather than duplicating
    # that logic here.
    org1, org2 = _org_type_block(df1), _org_type_block(df2)
    results['Demographics: Organization type'] = stats.compare_categorical_family(
        [(cat, org1[cat], org2[cat]) for cat in ORG_TYPE_LABELS]
    )

    for phase_key, phase_marker, title in PHASES:
        b1, b2 = _select_phase_block(df1, phase_marker), _select_phase_block(df2, phase_marker)
        use_items = _matched_items(
            df1, [c for c in b1.columns if 'Scale 1' in c], df2, [c for c in b2.columns if 'Scale 1' in c],
            labels.extract_task_name,
        )
        harm_items = _matched_items(
            df1, [c for c in b1.columns if 'Scale 2' in c], df2, [c for c in b2.columns if 'Scale 2' in c],
            labels.extract_task_name,
        )
        rows = (
            [{'item': f"{label} (Usefulness)", **stats.mannwhitney_test(s1, s2, plotting.USE_CENTER_OUT)}
             for label, s1, s2 in use_items]
            + [{'item': f"{label} (Harmfulness)", **stats.mannwhitney_test(s1, s2, plotting.HARM_CENTER_OUT)}
               for label, s1, s2 in harm_items]
        )
        results[f'RQ3: {title}'] = stats.finalize_family(rows)

    demo_ordinal_rows = [{
        'item': 'Years of RE experience',
        **stats.mannwhitney_test(
            df1[_select_years_re_experience_column(df1)], df2[_select_years_re_experience_column(df2)],
            YEARS_RE_EXPERIENCE_LABELS,
        ),
    }]
    for activity, short_name in zip(_ACTIVITY_BRACKETS, _ACTIVITY_SHORT_NAMES):
        demo_ordinal_rows.append({
            'item': f'RE-discipline experience — {short_name}',
            **stats.mannwhitney_test(
                df1[_re_discipline_experience_column(df1, activity)],
                df2[_re_discipline_experience_column(df2, activity)],
                RE_DISCIPLINE_EXPERIENCE_LABELS,
            ),
        })
    results['Demographics: Experience (ordinal)'] = stats.finalize_family(demo_ordinal_rows)

    return results


def print_significance_summary(results: dict[str, pd.DataFrame], alpha: float = 0.05) -> None:
    """Print only the items that remain significant after within-family
    Benjamini-Hochberg correction. Every item's full numbers — including the
    non-significant ones, which matter for judging whether "not significant"
    means "no difference" or just "underpowered" — are in the exported CSV
    (see export_significance_csv), not repeated here.
    """
    print(f"=== Round 1 vs Round 2: significance testing (Benjamini-Hochberg FDR, alpha={alpha}) ===\n")
    for family, df in results.items():
        sig = df[df['q'] < alpha]
        print(f"{family} ({len(df)} items tested)")
        if sig.empty:
            print("  no items significant after FDR correction")
        else:
            for item, row in sig.iterrows():
                print(
                    f"  {item}: {row['effect_label']}={row['effect_size']:.2f}, "
                    f"p={row['p']:.4f}, q={row['q']:.4f} (n1={int(row['n1'])}, n2={int(row['n2'])})"
                )
        print()


def export_significance_csv(results: dict[str, pd.DataFrame], outdir: str | Path, alpha: float = 0.05) -> None:
    """One combined CSV (every family, every item) — easier to sort/filter
    or pull into the paper than 14 small files.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(results, names=['family', 'item']).reset_index()
    combined['significant'] = combined['q'] < alpha
    combined.to_csv(outdir / "significance_tests.csv", index=False)


def _significant_set(result_df: pd.DataFrame, alpha: float = 0.05) -> set[str]:
    return set(result_df.index[result_df['q'] < alpha])


def _significant_subset(result_df: pd.DataFrame, allowed_items: set[str], alpha: float = 0.05) -> set[str]:
    """For a family combining several charts' items under unprefixed, unique
    labels (e.g. RQ4's "Skill set will change"), keep only the significant
    items that belong to one specific chart. For prefixed items sharing a
    family (e.g. RQ4's training-interest/training-format sub-blocks, which
    can each have their own "Other" — see _significant_with_prefix_stripped),
    match by prefix instead.
    """
    return _significant_set(result_df, alpha) & allowed_items


def _significant_with_prefix_stripped(result_df: pd.DataFrame, prefix: str, alpha: float = 0.05) -> set[str]:
    """For a family whose item labels are prefixed for readability in
    print_demographic_summary-style output (e.g. "RE-discipline experience
    — Elicitation") but whose chart uses the bare label ("Elicitation").
    """
    return {item[len(prefix):] for item in _significant_set(result_df, alpha) if item.startswith(prefix)}


def _phase_significant_sets(result_df: pd.DataFrame, alpha: float = 0.05) -> tuple[set[str], set[str]]:
    """Split one RQ3 phase family's significant items back into
    (significant_use, significant_harm) task-label sets for the diverging
    chart — each item is labelled "{task} (Usefulness)" / "{task}
    (Harmfulness)" (see compare_round_significance).
    """
    sig = _significant_set(result_df, alpha)
    use = {item[: -len(' (Usefulness)')] for item in sig if item.endswith(' (Usefulness)')}
    harm = {item[: -len(' (Harmfulness)')] for item in sig if item.endswith(' (Harmfulness)')}
    return use, harm


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_round_report(df: pd.DataFrame, round_name: str, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    style.apply_paper_style()

    # Demographics: application domain, region, role, organization type,
    # years of RE experience, self-assessed RE-discipline experience, GenAI
    # usage frequency / experience duration. Every variable named in
    # loading._DEMOGRAPHIC_ANCHORS gets a chart here.
    plotting.plot_yes_counts_barh(
        [(round_name, df[_select_application_domain_columns(df)])],
        ylabel="Application Domain",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "application_domains.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _select_role_block(df))],
        ylabel="Role",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "role.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _select_region_block(df))],
        ylabel="Region",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "region.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _org_type_block(df))],
        ylabel="Organization Type",
        label_func=lambda cat: cat,
        savepath=str(outdir / "organization_type.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _years_of_re_experience_block(df))],
        ylabel="Years of Experience",
        label_func=lambda cat: cat,
        savepath=str(outdir / "years_of_re_experience.pdf"),
    )
    plotting.plot_stacked_percentage_barh(
        [(round_name, _re_discipline_experience_df(df))],
        categories=RE_DISCIPLINE_EXPERIENCE_LABELS,
        colors=style.COLORS['experience_greens'],
        savepath=str(outdir / "re_discipline_experience.pdf"),
    )
    plotting.plot_stacked_percentage_barh(
        [(round_name, _tool_usage_frequency_df(df))],
        categories=FREQUENCY_LABELS,
        colors=style.COLORS['frequency_purples'] + [style.COLORS['neutral_other']],
        savepath=str(outdir / "genai_tool_usage_frequency.pdf"),
    )
    plotting.plot_stacked_percentage_barh(
        [(round_name, _experience_duration_df(df))],
        categories=DURATION_LABELS,
        colors=style.COLORS['frequency_purples'],
        savepath=str(outdir / "genai_experience_duration.pdf"),
    )
    participated_col = "Did you participate in our previous survey as well?"
    if participated_col in df.columns:
        plotting.plot_yes_counts_barh(
            [(round_name, _participated_before_block(df))],
            ylabel="Response",
            label_func=lambda cat: cat,
            savepath=str(outdir / "participated_before.pdf"),
        )

    print_demographic_summary(df, round_name)

    # RQ1: usage
    plotting.plot_stacked_percentage_barh(
        [(round_name, _usage_percentage_df(df))],
        categories=['Yes', 'No'],
        colors=[style.COLORS['yes'], style.COLORS['no']],
        savepath=str(outdir / "usage.pdf"),
    )

    # RQ2: prevention + threats
    plotting.plot_yes_counts_barh(
        [(round_name, _select_prevention_block(df))],
        ylabel="Reason",
        label_func=labels.label_strip_brackets_and_parens,
        savepath=str(outdir / "prevention.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _select_threats_block(df))],
        ylabel="Limitation / Threat",
        label_func=labels.label_strip_brackets_and_parens,
        savepath=str(outdir / "threats.pdf"),
    )

    # RQ3: usefulness/harmfulness per RE phase
    for phase_key, phase_marker, title in PHASES:
        phase_df = _select_phase_block(df, phase_marker)
        plotting.plot_diverging_usefulness_harmfulness(
            [(round_name, phase_df)],
            savepath=str(outdir / f"{phase_key}_assessment.pdf"),
            show_legend=False,
        )

    # RQ4: skills + training
    plotting.plot_stacked_percentage_barh(
        [(round_name, _skills_percentage_df(df))],
        categories=['Yes', 'No'],
        colors=[style.COLORS['yes'], style.COLORS['no']],
        savepath=str(outdir / "skills.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _select_training_interest_block(df))],
        ylabel="Training Interest",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "training_interest.pdf"),
    )
    plotting.plot_yes_counts_barh(
        [(round_name, _select_training_format_block(df))],
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

    # Computed once up front so significant items/tasks can be marked on the
    # charts below, not just reported afterward.
    significance = compare_round_significance(df1, df2)

    # Demographics — same variables/order as generate_round_report. No
    # comparison version for "participated in previous survey" (round2-only
    # question, no round1 counterpart to pair against).
    plotting.plot_yes_counts_barh(
        rounds_of(lambda df: df[_select_application_domain_columns(df)]),
        ylabel="Application Domain",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "application_domains_comparison.pdf"),
        significant_items=_significant_set(significance['Demographics: Application domain']),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_role_block),
        ylabel="Role",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "role_comparison.pdf"),
        significant_items=_significant_set(significance['Demographics: Role / position']),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_region_block),
        ylabel="Region",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "region_comparison.pdf"),
        significant_items=_significant_set(significance['Demographics: Region']),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_org_type_block),
        ylabel="Organization Type",
        label_func=lambda cat: cat,
        savepath=str(outdir / "organization_type_comparison.pdf"),
        significant_items=_significant_set(significance['Demographics: Organization type']),
    )
    # Years of RE experience is tested as one ordinal (Mann-Whitney) variable,
    # not per-category — so unlike the other demographic bar charts, no
    # single bar "is" the significant result. Marked on the ylabel instead of
    # on a bar (there's no title to carry it now that figures rely on the
    # paper's caption for that).
    years_re_ylabel = "Years of Experience"
    if 'Years of RE experience' in _significant_set(significance['Demographics: Experience (ordinal)']):
        years_re_ylabel += " (distribution differs significantly *)"
    plotting.plot_yes_counts_barh(
        rounds_of(_years_of_re_experience_block),
        ylabel=years_re_ylabel,
        label_func=lambda cat: cat,
        savepath=str(outdir / "years_of_re_experience_comparison.pdf"),
    )
    plotting.plot_stacked_percentage_barh(
        rounds_of(_re_discipline_experience_df),
        categories=RE_DISCIPLINE_EXPERIENCE_LABELS,
        colors=style.COLORS['experience_greens'],
        savepath=str(outdir / "re_discipline_experience_comparison.pdf"),
        significant_items=_significant_with_prefix_stripped(
            significance['Demographics: Experience (ordinal)'], 'RE-discipline experience — '
        ),
    )
    plotting.plot_stacked_percentage_barh(
        rounds_of(_tool_usage_frequency_df),
        categories=FREQUENCY_LABELS,
        colors=style.COLORS['frequency_purples'] + [style.COLORS['neutral_other']],
        savepath=str(outdir / "genai_tool_usage_frequency_comparison.pdf"),
    )
    plotting.plot_stacked_percentage_barh(
        rounds_of(_experience_duration_df),
        categories=DURATION_LABELS,
        colors=style.COLORS['frequency_purples'],
        savepath=str(outdir / "genai_experience_duration_comparison.pdf"),
    )

    print_demographic_comparison(df1, df2, round1_label, round2_label)

    plotting.plot_stacked_percentage_barh(
        rounds_of(_usage_percentage_df),
        categories=['Yes', 'No'],
        colors=[style.COLORS['yes'], style.COLORS['no']],
        savepath=str(outdir / "usage_comparison.pdf"),
        significant_items=_significant_set(significance['RQ1: Usage by RE discipline']),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_prevention_block),
        ylabel="Reason",
        label_func=labels.label_strip_brackets_and_parens,
        savepath=str(outdir / "prevention_comparison.pdf"),
        significant_items=_significant_set(significance['RQ2: Prevention reasons']),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_threats_block),
        ylabel="Limitation / Threat",
        label_func=labels.label_strip_brackets_and_parens,
        savepath=str(outdir / "threats_comparison.pdf"),
        significant_items=_significant_set(significance['RQ2: Perceived threats']),
    )
    for phase_key, phase_marker, title in PHASES:
        significant_use, significant_harm = _phase_significant_sets(significance[f'RQ3: {title}'])
        plotting.plot_diverging_usefulness_harmfulness(
            rounds_of(lambda df, m=phase_marker: _select_phase_block(df, m)),
            savepath=str(outdir / f"{phase_key}_assessment_comparison.pdf"),
            show_legend=False,
            significant_use=significant_use,
            significant_harm=significant_harm,
        )

    rq4_family = significance['RQ4: Skills + training']
    plotting.plot_stacked_percentage_barh(
        rounds_of(_skills_percentage_df),
        categories=['Yes', 'No'],
        colors=[style.COLORS['yes'], style.COLORS['no']],
        savepath=str(outdir / "skills_comparison.pdf"),
        significant_items=_significant_subset(rq4_family, {"Skill set will change"}),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_training_interest_block),
        ylabel="Training Interest",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "training_interest_comparison.pdf"),
        significant_items=_significant_with_prefix_stripped(rq4_family, "Training interest: "),
    )
    plotting.plot_yes_counts_barh(
        rounds_of(_select_training_format_block),
        ylabel="Training Format",
        label_func=labels.label_from_brackets,
        savepath=str(outdir / "training_format_comparison.pdf"),
        significant_items=_significant_with_prefix_stripped(rq4_family, "Training format: "),
    )

    print_significance_summary(significance)
    export_significance_csv(significance, outdir)

    return schema.diff_schemas(df1, df2)
