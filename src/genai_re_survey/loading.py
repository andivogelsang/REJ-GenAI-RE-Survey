"""Round-specific loaders that normalize both exports to a shared schema.

Both loaders return a DataFrame where:
- bracket-label typos from round 1 are corrected (schema.LABEL_ALIASES)
- reworded top-level questions are renamed to a shared semantic key
  (schema.CORE_QUESTION_ALIASES)
- LimeSurvey timing/group-time telemetry columns are dropped
  (schema.METADATA_COLUMN_PATTERNS)
- rows with no substantive answers are dropped (_drop_empty_responses) —
  LimeSurvey creates a response row as soon as someone opens the survey
  link, so unstarted/abandoned visits show up in the export with every
  question column empty
- rows that only answered demographic/context questions are also dropped
  (_drop_demographic_only_responses) — these respondents reached the
  demographics page and then left before any RQ1-4 content, so they never
  actually engaged with the survey's research questions
- respondents affiliated with a university/research organization *and*
  early-career in RE (no experience, or less than 2 years) are also dropped
  (_drop_academic_early_career_responses) — see that function's docstring
- `df.attrs["round"]` is set, so downstream code (plotting, reports) knows
  which round a frame came from without it being threaded through every
  call explicitly.

Everything else — including columns unique to one round (e.g. round 2's
"Did you participate in our previous survey as well?") — passes through
unchanged; schema.diff_schemas() is how you find out what those are.
"""

from pathlib import Path

import pandas as pd

from . import schema

_DEFAULT_ROUND1_PATH = Path(__file__).resolve().parents[2] / "data" / "round1" / "raw.csv"
_DEFAULT_ROUND2_PATH = Path(__file__).resolve().parents[2] / "data" / "round2" / "raw.csv"


def _drop_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not schema.is_metadata_column(c)]
    return df[keep]


# LimeSurvey response-administration columns that exist for every response
# row regardless of whether the respondent answered a single question —
# not eligible to count as a "substantive" answer when checking for
# no-content (never-started/abandoned) responses.
_ADMIN_COLUMNS = {
    "Response ID", "Date submitted", "Last page", "Start language",
    "Seed", "Date started", "Date last action",
}


def _drop_empty_responses(df: pd.DataFrame) -> pd.DataFrame:
    content_cols = [c for c in df.columns if c not in _ADMIN_COLUMNS]
    return df[df[content_cols].notna().any(axis=1)].reset_index(drop=True)


def _rename_bracket_labels(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        m = schema.BRACKET_RE.search(col)
        if m and m.group(1) in aliases:
            new_label = aliases[m.group(1)]
            rename_map[col] = col[:m.start(1)] + new_label + col[m.end(1):]
    return df.rename(columns=rename_map)


def _rename_core_questions(df: pd.DataFrame, round_key: str) -> pd.DataFrame:
    rename_map = {
        spec[round_key]: semantic_key
        for semantic_key, spec in schema.CORE_QUESTION_ALIASES.items()
        if spec[round_key] in df.columns
    }
    return df.rename(columns=rename_map)


# Demographic/context questions about the respondent or their organization
# — not RQ1-4 survey content. Matched by substring (bracket questions +
# round-shared core questions) plus the semantic keys that
# _rename_core_questions() produces for the two per-round-worded GenAI
# habit questions, so this only needs to be applied *after* that renaming.
_DEMOGRAPHIC_ANCHORS = [
    "In which application domain(s) have you worked",
    "In which of the following regions do you typically work",
    "What is your current role or position in your organization",
    "Which of the following organization / business types",
    "Please assess your knowledge / experience in the various RE-related disciplines",
    "How many years of professional experience do you have in Requirements Engineering",
    "Did you participate in our previous survey as well?",
    "Which GenAI tools do you use primarily",
]
_DEMOGRAPHIC_CORE_KEYS = {"genai_experience_duration", "genai_tool_usage_frequency"}


def _drop_demographic_only_responses(df: pd.DataFrame) -> pd.DataFrame:
    demo_cols = [
        c for c in df.columns
        if c in _DEMOGRAPHIC_CORE_KEYS or any(a in c for a in _DEMOGRAPHIC_ANCHORS)
    ]
    content_cols = [c for c in df.columns if c not in _ADMIN_COLUMNS and c not in demo_cols]
    return df[df[content_cols].notna().any(axis=1)].reset_index(drop=True)


# Values used by _drop_academic_early_career_responses. Matches
# reports.YEARS_RE_EXPERIENCE_LABELS' two lowest categories and the
# organization-type category exactly as LimeSurvey exports it.
_ACADEMIC_ORG_VALUE = "University / Research"
_EARLY_CAREER_YEARS_VALUES = {"none / new to RE", "< 2 years"}


def _drop_academic_early_career_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Drop respondents who are both affiliated with a university/research
    organization and early-career in RE (no RE experience, or less than 2
    years).

    Motivated by a round-2 "purely academic respondent" overlap analysis
    across organization type, years of RE experience, and application
    domain: this specific combination consistently paired with an
    unspecific or absent application domain (no grounding in RE practice),
    unlike two combinations that were deliberately *not* excluded --
    university/research org with a research application domain and 3+
    years of experience (plausibly consultancy work), and early-career with
    a research domain but a non-academic organization (plausibly a junior
    researcher in an industry research lab).
    """
    org_col = next(c for c in df.columns if schema.ORG_TYPE_ANCHOR in c and not c.endswith("[Other]"))
    years_col = next(c for c in df.columns if schema.YEARS_RE_EXPERIENCE_ANCHOR in c)
    exclude = (df[org_col] == _ACADEMIC_ORG_VALUE) & df[years_col].isin(_EARLY_CAREER_YEARS_VALUES)
    return df[~exclude].reset_index(drop=True)


def load_round1(path: str | Path = _DEFAULT_ROUND1_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _drop_metadata_columns(df)
    df = _drop_empty_responses(df)
    df = _rename_bracket_labels(df, schema.LABEL_ALIASES)
    df = _rename_core_questions(df, "round1")
    df = _drop_demographic_only_responses(df)
    df = _drop_academic_early_career_responses(df)
    df.attrs["round"] = "round1"
    return df


def load_round2(path: str | Path = _DEFAULT_ROUND2_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _drop_metadata_columns(df)
    df = _drop_empty_responses(df)
    df = _rename_core_questions(df, "round2")
    df = _drop_demographic_only_responses(df)
    df = _drop_academic_early_career_responses(df)
    df.attrs["round"] = "round2"
    return df
