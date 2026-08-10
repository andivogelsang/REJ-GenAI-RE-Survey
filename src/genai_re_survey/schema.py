"""Crosswalk between round-1 and round-2 column naming.

The two LimeSurvey exports are ~97% aligned at the level of individual
`[bracket]` item labels (the actual RE tasks / threats / barriers), but
differ in three places worth tracking explicitly:

1. Three item labels were typo-corrected between rounds (LABEL_ALIASES).
2. A handful of top-level (non-bracketed) questions were reworded without
   changing what they measure (CORE_QUESTION_ALIASES).
3. Round 2's export includes ~55 columns of LimeSurvey timing telemetry
   that round 1's export doesn't have and that carry no analysis content
   (METADATA_COLUMN_PATTERNS).

`loading.py` applies all three before any RQ-specific code sees the data,
so the rest of the package can treat round 1 and round 2 as sharing one
column vocabulary. `diff_schemas()` re-derives the raw differences so a
future round 3 (or a correction to this table) is caught automatically
instead of silently breaking a chart.
"""

import re

import pandas as pd

# Round-1 bracket labels that were typo-corrected in round 2. Canonicalized
# to round 2's (corrected) spelling; applied to round-1 data on load.
LABEL_ALIASES = {
    "Cost and ressource constraints": "Cost and resource constraints",
    "Maintain requirements traceabilty": "Maintain requirements traceability",
    "Requirements refinement and clarfication": "Requirements refinement and clarification",
}

# Top-level questions that were reworded between rounds but measure the same
# construct. Keyed by a semantic name; both loaders rename the matching
# column (if present) to this key.
CORE_QUESTION_ALIASES = {
    "used_genai_for_re": {
        "round1": "Have you already used / applied GenAI for RE-related disciplines in your professional work?",
        "round2": "Have you already used / applied GenAI for RE-related activities in your professional work?",
    },
    "genai_tool_usage_frequency": {
        "round1": "How often do you use ChatGPT or similar AI chatbots?\xa0",
        "round2": "How often do you use GenAI (AI chatbots, AI agents, …)\xa0 in the context of your work",
    },
    "genai_experience_duration": {
        "round1": "\xa0How long have you been working in the field of GenAI?\xa0 ",
        "round2": "How long have you been working with GenAI in the context of your work?\xa0 ",
    },
}

# Columns present only because of the LimeSurvey export options used for
# round 2 (per-question / per-group timing telemetry) — not survey content.
METADATA_COLUMN_PATTERNS = [
    r"^Question time:",
    r"^Group time:",
    r"^Total time$",
]

# Public: reused by loading.py to locate the bracketed span of a column
# name when renaming it via LABEL_ALIASES.
BRACKET_RE = re.compile(r"\[(.*?)\]")


def _bracket_labels(columns) -> set[str]:
    labels: set[str] = set()
    for col in columns:
        labels.update(BRACKET_RE.findall(col))
    return labels


def is_metadata_column(col: str) -> bool:
    return any(re.match(pattern, col) for pattern in METADATA_COLUMN_PATTERNS)


def _core_questions(columns) -> set[str]:
    return {
        col for col in columns
        if "[" not in col and not is_metadata_column(col)
    }


def diff_schemas(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    """Report the raw round1-vs-round2 column differences.

    Compares columns *before* schema.py's aliasing is applied (pass the
    output of loading's internal raw readers, not the public load_round1 /
    load_round2, to see what the aliasing is correcting for). After
    aliasing, round1-only / round2-only should be empty except for genuinely
    round-specific questions (e.g. round 2's new "previous survey" /
    "which GenAI tools" questions).
    """
    b1, b2 = _bracket_labels(df1.columns), _bracket_labels(df2.columns)
    c1, c2 = _core_questions(df1.columns), _core_questions(df2.columns)

    report = {
        "bracket_labels_shared": sorted(b1 & b2),
        "bracket_labels_round1_only": sorted(b1 - b2),
        "bracket_labels_round2_only": sorted(b2 - b1),
        "core_questions_shared": sorted(c1 & c2),
        "core_questions_round1_only": sorted(c1 - c2),
        "core_questions_round2_only": sorted(c2 - c1),
    }

    print(f"Bracket labels: {len(b1 & b2)} shared, "
          f"{len(report['bracket_labels_round1_only'])} round1-only, "
          f"{len(report['bracket_labels_round2_only'])} round2-only")
    for label in report["bracket_labels_round1_only"]:
        print(f"  round1-only: {label!r}")
    for label in report["bracket_labels_round2_only"]:
        print(f"  round2-only: {label!r}")

    print(f"Core questions: {len(c1 & c2)} shared, "
          f"{len(report['core_questions_round1_only'])} round1-only, "
          f"{len(report['core_questions_round2_only'])} round2-only")
    for q in report["core_questions_round1_only"]:
        print(f"  round1-only: {q!r}")
    for q in report["core_questions_round2_only"]:
        print(f"  round2-only: {q!r}")

    return report
