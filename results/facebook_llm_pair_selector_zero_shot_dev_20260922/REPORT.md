# Zero-shot LLM pair selector: development trial

This is an exploratory development-only result, not a new held-out evaluation. The V5 question wording was previously inspected; no V5 examples were used in this run.

The local Qwen model saw each question and the five predeclared AGP `(a,b)` options, but no graph, labels, or retrieval scores. It returned a JSON arm choice. The runner then looked up that arm's already-saved retrieval F1, so AGP was not rerun and the evaluation scores were never sent to the model.

| Selector | Mean retrieval F1 on 20 development questions |
| --- | ---: |
| LLM-selected pair | 0.4415 |
| Best fixed pair, C2 `(0,1)` | 0.4532 |

The model chose C4 `(0.5,0.5)` 19 times and C3 `(0.25,0.75)` once. It did not demonstrate useful question-specific adaptation in this trial. This result does not rule out better prompts or a trained selector, but any such changes would need development-only design and a genuinely untouched evaluation set for a confirmatory claim.

Reproduce with the local model configured in `.env`:

```sh
python3 scripts/try_llm_pair_selector.py \
  --questions data/facebook_large/facebook_questions_dev.json \
  --scores results/facebook_agp_pair_selector_dev_20260919/scores.jsonl \
  --output results/my_llm_pair_selector_trial
```

The output folder must not exist before running. `manifest.json` records the prompt, model, and input hashes; `decisions.jsonl` records each choice and request metadata; `summary.json` contains aggregate results. Do not use this script with a held-out questions file: it explicitly requires a `_dev.json` name and exact alignment with saved development scores.
