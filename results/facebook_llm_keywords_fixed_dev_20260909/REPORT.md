# LLM-keywords with fixed parameters: development evaluation

## Purpose

This ablation isolates entity extraction. Local
`qwen3:4b-instruct-2507-q4_K_M` returns only keyword strings; the Python exact
backend then uses the already selected fixed development parameters: depth 2,
decay 0.3, and top-k 5. The LLM does not select propagation parameters and answer
generation is disabled. No cloud API was called and held-out test data was not
read or evaluated.

## Result

| Condition | Precision | Recall | Mean per-question F1 | Hit rate | MRR |
|---|---:|---:|---:|---:|---:|
| Local keywords + selected fixed | 0.4100 | 0.6244 | 0.4672 | 0.8500 | 0.3058 |
| LLM keywords + selected fixed | 0.3500 | 0.5244 | 0.3922 | 0.7500 | 0.2558 |

The only experimental difference is keyword extraction. LLM keywords reduce mean
F1 by 0.0750 and recall by 0.10 on this development set.

## Result by question type

| Type (5 questions each) | Precision | Recall | Mean F1 | Hit rate |
|---|---:|---:|---:|---:|
| Direct | 0.4800 | 0.4143 | 0.4342 | 0.6000 |
| Common-neighbor comparison | 0.4800 | 0.8500 | 0.5853 | 1.0000 |
| Path intermediates | 0.2400 | 0.6000 | 0.3429 | 0.8000 |
| Same-category two-hop similarity | 0.2000 | 0.2333 | 0.2067 | 0.6000 |

## Extraction errors

Qwen exactly identified every intended graph title in 18 of 20 questions. It
split two long, valid page titles into smaller strings that do not exactly match
any graph node:

- `Headquarters Marine Corps, Henderson Hall, Headquarters & Service Battalion`
  became three separate keywords;
- `Chrisley Knows Best on USA` became `Chrisley Knows Best` and `USA`.

Both questions therefore had zero matched seeds. All other extracted keywords
matched their intended seed IDs. The result demonstrates why local graph-aware
title matching is a strong controlled baseline and why a production system would
need entity linking rather than raw exact matching after free-form LLM extraction.

## Requests and tokens

The run made 20 local network requests with zero cache hits: 2,222 prompt tokens,
346 completion tokens, and 2,568 total provider-reported tokens. Mean pipeline
latency was 1.071 seconds. The metadata log excludes prompts, responses, and API
keys. See `manifest.json` for hashes and `results.jsonl` for question-level traces.

## Decision

Preserve this result without prompt retuning on the same development questions.
The keyword ablation is complete. Freeze the listed experimental conditions and
do not inspect held-out results until the evaluation command and reporting code
are ready.
