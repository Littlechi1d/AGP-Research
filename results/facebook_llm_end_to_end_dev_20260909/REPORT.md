# End-to-end LLM development smoke run

## Purpose

This is the required development-only execution check for the frozen `llm`
condition. Local `qwen3:4b-instruct-2507-q4_K_M` jointly extracts keywords and
selects depth, decay, and top-k. The Python exact backend performs retrieval.
Answer generation is disabled. No prompt tuning followed this run, no cloud API
was called, and the held-out test set was not read or evaluated.

## Result

| Development condition | Precision | Recall | Mean per-question F1 | Hit rate | MRR |
|---|---:|---:|---:|---:|---:|
| Rules | 0.4159 | 0.8800 | 0.5218 | 0.9000 | 0.3130 |
| LLM parameters, local keywords | 0.4125 | 0.8300 | 0.5134 | 0.9000 | 0.3100 |
| LLM keywords, fixed parameters | 0.3500 | 0.5244 | 0.3922 | 0.7500 | 0.2558 |
| End-to-end LLM | 0.3059 | 0.7250 | 0.3977 | 0.7500 | 0.2558 |

The end-to-end condition passes the engineering smoke test but does not beat the
rules strategy on development mean F1. It gains recall over LLM-keywords/fixed
because it always retrieves ten nodes, but loses precision.

## Result by question type

| Type (5 questions each) | Precision | Recall | Mean F1 | Hit rate |
|---|---:|---:|---:|---:|
| Direct | 0.5128 | 0.6000 | 0.5527 | 0.6000 |
| Common-neighbor comparison | 0.3200 | 1.0000 | 0.4643 | 1.0000 |
| Path intermediates | 0.1800 | 0.9000 | 0.3000 | 1.0000 |
| Same-category two-hop similarity | 0.2107 | 0.4000 | 0.2738 | 0.4000 |

## Planner behavior

Qwen selected top-k 10 for all 20 questions and never selected depth 3:

| Depth | Decay | Top-k | Questions |
|---:|---:|---:|---:|
| 1 | 0.3 | 10 | 4 |
| 1 | 0.5 | 10 | 1 |
| 2 | 0.3 | 10 | 3 |
| 2 | 0.7 | 10 | 12 |

All intended seeds were matched on 15 questions. The two long direct-query titles
that failed in the keyword-only ablation failed again. One path title containing
a slash was split, leaving only one of two seeds. Two long similarity titles were
also split and produced no seed. Several otherwise successful questions included
extra unmatched phrases such as `liked by both`, `category`, and `two hops away`.

This shows an interaction hidden by aggregate retrieval metrics: joint planning
does not repair entity extraction, while its larger retrieval budget can partially
mask extraction problems by improving recall on successful questions.

## Requests and tokens

The run made 20 local network requests with zero cache hits: 2,422 prompt tokens,
1,039 completion tokens, and 3,461 total provider-reported tokens. Mean pipeline
latency was 2.131 seconds. See `manifest.json` for hashes and `results.jsonl` for
all question-level traces.

## Decision

The frozen end-to-end prompt is executable and returns allowed parameter values.
Do not modify it based on these development results. Commit this smoke-run bundle
before the one-time held-out experiment specified by `FROZEN_EXPERIMENT_PROTOCOL.md`.
