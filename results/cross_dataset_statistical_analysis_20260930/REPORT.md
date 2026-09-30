# Frozen evaluation statistical analysis

This analysis uses only saved C0–C8 outputs. It makes no LLM or AGP calls.
Paired 95% confidence intervals use 10,000 bootstrap resamples with seed 90055.

## Paired retrieval-F1 comparisons

| Dataset | Comparison | Mean difference | 95% bootstrap CI |
| --- | --- | ---: | ---: |
| GitHub MUSAE | C1 − C2 | +0.0265 | [-0.0672, +0.1208] |
| GitHub MUSAE | C7 − C2 | +0.0000 | [+0.0000, +0.0000] |
| GitHub MUSAE | C8 − C2 | -0.0447 | [-0.0939, +0.0011] |
| Deezer Europe | C1 − C2 | +0.0316 | [-0.0561, +0.1199] |
| Deezer Europe | C7 − C2 | +0.0000 | [+0.0000, +0.0000] |
| Deezer Europe | C8 − C2 | -0.0239 | [-0.0521, +0.0012] |

Intervals crossing zero do not establish a clear paired difference at the
95% bootstrap level. These intervals are descriptive and are not corrected
for the multiple arm comparisons in the broader study.

## Figures

- `retrieval_f1_by_condition.svg`: overall retrieval comparison.
- `retrieval_f1_by_question_type.svg`: question-type heatmap.
- `answer_entity_f1_by_condition.svg`: conservative answer entity coverage.

## Interpretation

Direct neighbours lead overall because one quarter of the balanced questions
ask directly for neighbours. They score zero on similarity questions, where
propagation is necessary. C7 equals C2 because it always used its fallback.
C8 does not beat the strongest fixed arm on either dataset. The figures
therefore support question-type heterogeneity, but not successful adaptation
by the two frozen selectors.
