# GenAI in RE: Survey Analysis

Replication package for the survey study on Generative AI usage in
Requirements Engineering, run in collaboration with the IREB Special
Interest Group on AI & RE (SIG#AIREB).

- **Round 1** — data collection Nov 2024–Mar 2025, reported in *"Opportunities
  and Limitations of GenAI in RE: Viewpoints from Practice"* (REFSQ 2026).
  Dataset archived at Zenodo: [10.5281/zenodo.18207273](https://doi.org/10.5281/zenodo.18207273)
  (CC-BY-4.0).
- **Round 2** — a second wave of responses collected for a journal extension
  of the same study, reporting round 1, round 2, and a comparison between
  them.

The survey was anonymous and collected no personal data (see the Construct
Validity discussion in the round-1 paper); the response data is published
as-is.

## Layout

```
data/
├── round1/          raw export + questionnaire + supplementary materials
└── round2/          raw export
codebook/
└── round1/          qualitative coding schema (14 themes / 52 codes)
src/genai_re_survey/ shared loading, schema-alignment, plotting, and report code
notebooks/           01_round1_analysis / 02_round2_analysis / 03_comparison
figures/             generated PDFs, one subfolder per notebook
```

`src/genai_re_survey/schema.py` documents exactly how round 1 and round 2
column names differ (a handful of reworded questions, three typo fixes, and
LimeSurvey's per-question timing metadata in round 2) and how `loading.py`
normalizes both to a shared vocabulary before any analysis code sees them.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.10 (developed and
verified against 3.14).

```bash
uv venv .venv
uv pip install --python .venv -e ".[notebooks]"
```

## Reproducing the figures

Run the three notebooks in `notebooks/`, in order — each is a thin wrapper
around `genai_re_survey.reports`:

- `01_round1_analysis.ipynb` reproduces the figures published in the REFSQ
  2026 paper (`figures/round1/`).
- `02_round2_analysis.ipynb` runs the same analysis against round 2
  (`figures/round2/`).
- `03_comparison.ipynb` renders round-1-vs-round-2 paired/grouped figures
  and prints a schema diff flagging anything unique to either round
  (`figures/comparison/`).

## Citation

See `CITATION.cff`. Code is MIT-licensed (`LICENSE`); survey data is
CC-BY-4.0 (`DATA_LICENSE`), matching the round-1 Zenodo deposit.
