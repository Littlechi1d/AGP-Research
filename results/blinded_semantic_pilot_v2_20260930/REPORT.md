# Small blinded semantic-review pilot

This is an exploratory model-assisted review of saved frozen answers. It
does not rerun retrieval or answer generation. One question per type and
dataset was selected by a deterministic hash before answer files were opened.

## Mean blind-judge scores

| Condition | Correctness (1–5) | Completeness (1–5) |
| --- | ---: | ---: |
| C0 | 1.000 | 1.000 |
| C1 | 3.500 | 3.500 |
| C2 | 4.750 | 4.750 |
| C8 | 3.500 | 3.375 |

## Scores by dataset

| Dataset | Condition | Correctness | Completeness |
| --- | --- | ---: | ---: |
| GitHub MUSAE | C0 | 1.000 | 1.000 |
| GitHub MUSAE | C1 | 4.000 | 4.000 |
| GitHub MUSAE | C2 | 4.500 | 4.500 |
| GitHub MUSAE | C8 | 3.000 | 2.750 |
| Deezer Europe | C0 | 1.000 | 1.000 |
| Deezer Europe | C1 | 3.000 | 3.000 |
| Deezer Europe | C2 | 5.000 | 5.000 |
| Deezer Europe | C8 | 4.000 | 4.000 |

C2 has the highest mean score in this eight-question pilot. C0 receives
1/5 because it abstains despite nonempty verified gold-answer sets. C1 and C8
vary by question depending on whether the saved answer explicitly reaches the
requested entity list before truncation.

## Interpretation limits

The judge saw only randomized A–D labels, the question, the reference entity
set, and one answer at a time. The condition key was revealed only after all
32 ratings were complete. However, the judge is the same local model used for
answer generation, the sample contains only eight questions, and the reference
criteria remain topology-defined entity sets. These ratings are a pilot and
must not be presented as independent human evidence or definitive semantic
quality estimates.
