"""Round-specific loaders that normalize both exports to a shared schema.

Both loaders return a DataFrame where:
- bracket-label typos from round 1 are corrected (schema.LABEL_ALIASES)
- reworded top-level questions are renamed to a shared semantic key
  (schema.CORE_QUESTION_ALIASES)
- LimeSurvey timing/group-time telemetry columns are dropped
  (schema.METADATA_COLUMN_PATTERNS)
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


def load_round1(path: str | Path = _DEFAULT_ROUND1_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _drop_metadata_columns(df)
    df = _rename_bracket_labels(df, schema.LABEL_ALIASES)
    df = _rename_core_questions(df, "round1")
    df.attrs["round"] = "round1"
    return df


def load_round2(path: str | Path = _DEFAULT_ROUND2_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _drop_metadata_columns(df)
    df = _rename_core_questions(df, "round2")
    df.attrs["round"] = "round2"
    return df
