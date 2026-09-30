# Small blinded semantic-review pilot

This is an exploratory model-assisted review of saved frozen answers. It
does not rerun retrieval or answer generation. One question per type and
dataset was selected by a deterministic hash before answer files were opened.

## Mean blind-judge scores

| Condition | Correctness (1–5) | Completeness (1–5) |
| --- | ---: | ---: |
| C0 | 3.500 | 3.500 |
| C1 | 1.500 | 1.500 |
| C2 | 1.625 | 1.500 |
| C8 | 2.000 | 1.875 |

## Interpretation limits

The judge saw only randomized A–D labels, the question, the reference entity
set, and one answer at a time. The condition key was revealed only after all
32 ratings were complete. However, the judge is the same local model used for
answer generation, the sample contains only eight questions, and the reference
criteria remain topology-defined entity sets. These ratings are a pilot and
must not be presented as independent human evidence or definitive semantic
quality estimates.
