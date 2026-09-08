# LLM-parameters development evaluation

## Purpose

This pilot evaluates the `llm-parameters` ablation on all 20 Facebook development
questions. Local exact-title extraction supplies seed nodes; local
`qwen3:4b-instruct-2507-q4_K_M` selects depth, decay, and top-k. The Python exact
propagation backend then retrieves nodes. Answer generation was disabled.

No cloud API was called and the held-out test set was not read or evaluated. See
`manifest.json` for the model digest, code and input checksums, command, and run
environment. `results.jsonl` contains all question-level outputs.

## Overall result

| Development condition | Precision | Recall | Mean per-question F1 | Hit rate | MRR |
|---|---:|---:|---:|---:|---:|
| Selected fixed, k=5 | 0.4100 | 0.6244 | 0.4672 | 0.8500 | 0.3058 |
| Fixed, budget-matched k=10 | 0.3377 | 0.8800 | 0.4582 | 0.9000 | 0.3142 |
| Rules, k=10 | 0.4159 | 0.8800 | 0.5218 | 0.9000 | 0.3130 |
| LLM-parameters, k=10 | 0.4125 | 0.8300 | 0.5134 | 0.9000 | 0.3100 |

The LLM-parameters strategy is close to rules but does not beat it on this
development run. Relative to rules, its precision is lower by about 0.0033,
recall by 0.05, and mean F1 by about 0.0083. These are descriptive development
differences; no significance or held-out generalization is claimed.

Comparing the selected k=5 fixed result to k=10 systems changes the retrieval
budget. The budget-matched fixed row is therefore the fairer k=10 comparison.

## Result by question type

| Type (5 questions each) | Precision | Recall | Mean F1 | Hit rate |
|---|---:|---:|---:|---:|
| Direct | 0.8128 | 1.0000 | 0.8955 | 1.0000 |
| Common-neighbor comparison | 0.3467 | 1.0000 | 0.4976 | 1.0000 |
| Path intermediates | 0.1600 | 0.8000 | 0.2667 | 1.0000 |
| Same-category two-hop similarity | 0.3307 | 0.5200 | 0.3938 | 0.6000 |

All intended seeds were matched and all unmatched-keyword lists were empty. The
new strategy therefore removes LLM entity extraction as a failure source in this
run. Remaining differences come from parameter choices and graph ranking.

## Qwen parameter choices

| Depth | Decay | Top-k | Questions |
|---:|---:|---:|---:|
| 1 | 0.3 | 10 | 2 |
| 1 | 0.5 | 10 | 9 |
| 1 | 0.8 | 10 | 4 |
| 2 | 0.7 | 10 | 3 |
| 2 | 0.8 | 10 | 2 |

Qwen selected top-k 10 for every question. It selected depth 1 for all direct,
comparison, and path questions, and depth 2 for every similarity question. Thus
it adapted depth by template class, but never exercised depth 3 or top-k 5/20.

Depth 1 still retrieves many common neighbors and path intermediates because both
endpoint entities are seeds: a common neighbor is one edge from both seeds, and
each internal node on a three-edge path is one edge from one endpoint. However,
two path questions reached only 0.5 recall because relevant nodes fell outside the
top ten. Qwen's decay choices did not recover them.

## Timing

Mean `pipeline.run` time was approximately 1.108 seconds per question on the
local M3/16 GB machine with the model already installed. This includes local Qwen
planning and Python retrieval, but not answer generation. The freshly rerun rule
reference was about 0.534 seconds per question, so the observed local planning
overhead was roughly 0.57 seconds. Runtime varies with model warm state and other
machine activity; this is not a controlled latency benchmark.

## Interpretation and next step

This result supports keeping local extraction and evaluating LLM parameter choice
separately. It does not show an advantage over transparent rules. The current
prompt offers qualitative guidance but no graph statistics or labeled examples,
and Qwen ignores part of its advertised choice space.

Before changing the prompt, freeze this as the zero-shot development result. The
next clean experiment is a **few-shot LLM-parameters condition** with one example
of each question type, selected only from development-design knowledge. Give it a
new strategy name and prompt version; do not replace or relabel this zero-shot
result. Compare both conditions on the same 20 development questions before any
held-out test evaluation.
