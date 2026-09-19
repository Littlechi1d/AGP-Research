# Eight-condition answer-generation smoke test

This is a **one-question development software check**, not an experimental
result. It used the already saved eight-condition contexts for
`facebook_direct_01` and local `qwen3:4b-instruct-2507-q4_K_M` at temperature
zero. Seven answer conditions were generated; C7 copied the answer of its
question-selected fixed arm C2. Six calls reached the local model and one was a
response-cache hit. No held-out question was used.

The output includes every answer and provider-reported token count in
`answers.jsonl`, randomized condition labels A–H in `blind_review.jsonl`, a
separate `answer_key.json`, and blank correctness/completeness and factual-
support rating forms. Only the review files and blank forms should be shared
with reviewers; keep the key and condition-labelled answers hidden until their
ratings are final.

## What the smoke test revealed

| Arm | Automatic entity F1 | Prompt tokens |
|---|---:|---:|
| C0 LLM-only | 0.000 | 79 |
| C1 direct neighbours | 1.000 | 392 |
| C2 `(0,1)` | 0.857 | 1,576 |
| C3 `(0.25,0.75)` | 1.000 | 1,470 |
| C4 `(0.5,0.5)` | 1.000 | 1,440 |
| C5 `(0.75,0.25)` | 1.000 | 1,279 |
| C6 `(1,0)` | 1.000 | 1,279 |
| C7 adaptive | 0.857 | Same answer as C2; no extra call |

These values are *not* comparative evidence from a meaningful sample. In this
one example, C2/C7's answer named an extra page that appeared in its context
but was not a direct neighbour of the seed. The automatic context-faithfulness
metric still scored 1.0 because it checks whether **named graph entities** occur
in the context, not whether every asserted relationship is true. Reviewers
must therefore check factual support against the exact saved context.

Input size also differed greatly across arms (79 to 1,576 prompt tokens), even
though graph arms had the same maximum 10-node budget. This is a possible
answer-quality confound; the final protocol needs a token-budget sensitivity
check or an explicit reason for retaining node-count-only budgets.

To reproduce in a new output directory:

```bash
python3 scripts/run_eight_condition_answers.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --questions data/facebook_large/facebook_questions_dev.json \
  --contexts results/facebook_eight_contexts_smoke_dev_20260919/contexts.jsonl \
  --output results/facebook_eight_answers_smoke_REPRODUCED \
  --limit 1
```
