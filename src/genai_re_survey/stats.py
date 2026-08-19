"""Round1-vs-round2 significance testing, generic over item shape.

No dependency on reports.py/plotting.py — callers (reports.py) match up which
columns from df1/df2 represent "the same item" and pass in plain Series pairs;
this module only knows how to test one item and how to correct a family of
p-values. Kept generic so the test logic isn't tangled with this package's
column-selection conventions.

Two test kinds, chosen per item's data type:
- Fisher's exact (2x2) for Yes/No items — robust to the small/zero cell counts
  that show up often at these sample sizes (round 1 n~109, round 2 n~69, and
  smaller still per item), unlike chi-square's expected-cell-count assumption.
- Mann-Whitney U for ordinal items (5-point Likert, experience-level scales) —
  uses rank information a chi-square/Fisher test would discard.

Every item's own non-null count is used as its n (not a block-shared n some
charts show in their caption) — the correct denominator for a test that's
actually about that one item's responses.
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """FDR-adjusted p-values (q-values) via the BH step-up procedure.

    q_(i) = min over j>=i of (p_(j) * m / j), applied to p-values sorted
    ascending, then mapped back to the original order and clipped to [0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p-value down
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(q_sorted, 0, 1)
    return pd.Series(q, index=p_values.index if isinstance(p_values, pd.Series) else None)


def _yesno_counts(s: pd.Series) -> tuple[int, int]:
    """(positive_count, n). Yes/No column -> count of Yes; any other value set
    (e.g. a free-text "Other" write-in, or a one-hot block's own Yes/No column)
    -> every non-null counts as positive. Mirrors the exact heuristic each
    chart already uses (plotting._count_column) so the test matches what's
    plotted.
    """
    s = s.dropna()
    n = len(s)
    if n and set(s.unique()) <= {'Yes', 'No'}:
        return int((s == 'Yes').sum()), n
    return n, n


def fisher_test(s1: pd.Series, s2: pd.Series) -> dict:
    """One Yes/No item, round1 vs round2."""
    count1, n1 = _yesno_counts(s1)
    count2, n2 = _yesno_counts(s2)
    table = [[count1, n1 - count1], [count2, n2 - count2]]
    odds_ratio, p = fisher_exact(table)
    return {
        'n1': n1, 'n2': n2, 'count1': count1, 'count2': count2,
        'pct1': count1 / n1 * 100 if n1 else 0.0,
        'pct2': count2 / n2 * 100 if n2 else 0.0,
        'effect_size': odds_ratio, 'effect_label': 'odds_ratio',
        'p': p,
    }


def mannwhitney_test(
    s1: pd.Series, s2: pd.Series, category_order: list[str], exclude_value: str = "I don't know"
) -> dict:
    """One ordinal item, round1 vs round2. category_order must be ascending
    low->high; values outside it (besides exclude_value) are an input error.
    """
    rank = {cat: i for i, cat in enumerate(category_order)}
    x1 = s1[~s1.isin([exclude_value])].dropna().map(rank)
    x2 = s2[~s2.isin([exclude_value])].dropna().map(rank)
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        return {'n1': n1, 'n2': n2, 'effect_size': float('nan'), 'effect_label': 'rank_biserial_r', 'p': float('nan')}
    u_stat, p = mannwhitneyu(x1, x2, alternative='two-sided')
    effect_r = 1 - (2 * u_stat) / (n1 * n2)
    return {'n1': n1, 'n2': n2, 'effect_size': effect_r, 'effect_label': 'rank_biserial_r', 'p': p}


def finalize_family(rows: list[dict]) -> pd.DataFrame:
    """Turn a list of per-item result dicts (each already carrying an 'item'
    key plus whatever fisher_test/mannwhitney_test returned) into one family
    table with a BH-adjusted 'q' column, sorted by q ascending. Public so
    callers needing a mixed family (e.g. usefulness + harmfulness items,
    which use different category orders and so can't share one
    compare_ordinal_family call) can build `rows` by hand and still get the
    same correction/formatting as compare_categorical_family/
    compare_ordinal_family.
    """
    df = pd.DataFrame(rows).set_index('item')
    df['q'] = benjamini_hochberg(df['p'])
    return df.sort_values('q')


def compare_categorical_family(items: list[tuple[str, pd.Series, pd.Series]]) -> pd.DataFrame:
    """items: (label, round1_series, round2_series) triples for one family
    (e.g. all 13 prevention reasons). Returns one row per item, sorted by
    BH-adjusted q ascending.
    """
    rows = [{'item': label, **fisher_test(s1, s2)} for label, s1, s2 in items]
    return finalize_family(rows)


def compare_ordinal_family(items: list[tuple[str, pd.Series, pd.Series]], category_order: list[str]) -> pd.DataFrame:
    rows = [{'item': label, **mannwhitney_test(s1, s2, category_order)} for label, s1, s2 in items]
    return finalize_family(rows)
