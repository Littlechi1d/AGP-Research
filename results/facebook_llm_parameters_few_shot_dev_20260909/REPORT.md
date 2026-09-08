# Few-shot LLM-parameters development evaluation

## Purpose

This run compares a separately named `llm-parameters-few-shot` strategy with the
preserved zero-shot result. Local exact-title matching still supplies graph seeds.
Local `qwen3:4b-instruct-2507-q4_K_M` chooses only `depth`, `decay`, and `top_k`
after seeing four synthetic demonstrations: one for each benchmark question type.
Answer generation was disabled, no cloud API was called, and the held-out test set
was not read or evaluated.

## Development results

| Condition | Precision | Recall | Mean per-question F1 | Hit rate | MRR |
|---|---:|---:|---:|---:|---:|
| Zero-shot LLM parameters | 0.4125 | 0.8300 | 0.5134 | 0.9000 | 0.3100 |
| Few-shot LLM parameters | 0.3975 | 0.8800 | 0.4922 | 0.9000 | 0.3142 |
| Rules | 0.4159 | 0.8800 | 0.5218 | 0.9000 | 0.3130 |

Few-shot prompting recovered the same mean recall as rules and raised recall by
0.05 over zero-shot. However, the larger path-query retrieval budget reduced
precision, so mean F1 fell by 0.0212. It therefore did not beat either zero-shot
or rules on the declared development mean-F1 metric.

## Few-shot result by question type

| Type (5 questions each) | Precision | Recall | Mean F1 | Hit rate |
|---|---:|---:|---:|---:|
| Direct | 0.8128 | 1.0000 | 0.8955 | 1.0000 |
| Common-neighbor comparison | 0.3467 | 1.0000 | 0.4976 | 1.0000 |
| Path intermediates | 0.1000 | 1.0000 | 0.1818 | 1.0000 |
| Same-category two-hop similarity | 0.3307 | 0.5200 | 0.3938 | 0.6000 |

## Parameter behavior

Qwen followed every demonstration exactly for all five questions of its type:

| Question type | Depth | Decay | Top-k |
|---|---:|---:|---:|
| Direct | 1 | 0.4 | 10 |
| Comparison | 1 | 0.6 | 10 |
| Path | 2 | 0.75 | 20 |
| Similarity | 2 | 0.7 | 10 |

This confirms that few-shot examples can reliably control the small model, but
it also makes the strategy behave like an LLM-mediated lookup table on the
templated benchmark. The path example fixed zero-shot's missed recall by using
`top_k=20`; its precision cost explains the lower overall F1. The similarity
result was unchanged, so prompting alone did not solve those missed questions.

## Interpretation and next step

Keep both results as prompt ablations. Do not tune the demonstrations again on
these same 20 questions unless the additional tuning is explicitly reported.
The next clean engineering step is to add caching and token/request logging, then
freeze the zero-shot, few-shot, rules, and fixed conditions. After that, run the
held-out test set once. A separate semantic benchmark will still be needed because
these templated questions make the few-shot mapping unusually easy to imitate.

See `manifest.json` for hashes and run metadata and `results.jsonl` for all 20
question-level traces.
