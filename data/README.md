# Survey data

- `round1/raw.csv` — round-1 LimeSurvey export, 150 rows × 202 columns.
  Identical to the `results.csv` in the round-1 Zenodo deposit
  (10.5281/zenodo.18207273); `round1/questionnaire.pdf`,
  `round1/responses-freetext-experience.xlsx`, and
  `round1/ids-complete-participants.xlsx` are copied from the same deposit.
- `round2/raw.csv` — round-2 LimeSurvey export, 79 rows × 255 columns
  (converted from the original `.xlsx` export; the LimeSurvey per-question
  timing columns it adds are dropped by the loader, not here).

Load either with `genai_re_survey.loading.load_round1()` /
`.load_round2()` — see `src/genai_re_survey/schema.py` for exactly how the
two column sets are reconciled.
