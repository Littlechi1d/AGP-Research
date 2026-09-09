# Frozen experiment protocol

**Frozen on:** 9 September 2026, before held-out evaluation.

## Primary decision metric

The primary retrieval metric is mean per-question F1. Precision, recall, hit
rate, reciprocal rank, latency, and question-type summaries are secondary. The
held-out set must be run once after all listed code and prompts are committed.
No strategy may be changed in response to held-out results.

## Conditions

| Condition | Keywords | Parameters | Retrieval budget |
|---|---|---|---:|
| `seed-only` | Local exact title | depth 0, decay 0 | 10 |
| `fixed` | Local exact title | depth 2, decay 0.3 | 5 |
| `fixed` budget match | Local exact title | depth 2, decay 0.3 | 10 |
| `rules` | Local exact title | Frozen rules | 10 |
| `llm-keywords` | Zero-shot Qwen extraction | depth 2, decay 0.3 | 5 |
| `llm-parameters` | Local exact title | Zero-shot Qwen choice | Model chooses 5/10/20 |
| `llm-parameters-few-shot` | Local exact title | Four-example Qwen choice | Model chooses 5/10/20 |
| `llm` | Zero-shot Qwen extraction | Zero-shot Qwen choice | Model chooses 5/10/20 |

The model is `qwen3:4b-instruct-2507-q4_K_M`, digest
`0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`,
at temperature zero through the local OpenAI-compatible endpoint. Answer
generation is disabled for retrieval evaluation. The Python exact propagation
backend is the primary backend; the paper adapter is reported separately.

## Isolation and reproducibility rules

1. Commit code, prompts, tests, this protocol, and development artifacts first.
2. Record the commit hash, model digest, input hashes, command, cache status,
   request count, token totals, and environment in the held-out manifest.
3. Use the existing independent test questions without regenerating them.
4. Use a fresh output directory. Cached identical model responses are allowed
   and must be reported; cache identity includes model, endpoint, prompt, mode,
   temperature, and question.
5. Produce question-level traces plus overall and per-type summaries.
6. Report negative and null findings. Do not tune on held-out errors.

## Before running held-out evaluation

The end-to-end `llm` condition should first receive a development-only smoke run
to verify JSON validity and establish that its advertised parameter choices are
accepted. This is an execution check, not another prompt-tuning round. After the
smoke run and final commit, execute all frozen conditions on held-out data once.
