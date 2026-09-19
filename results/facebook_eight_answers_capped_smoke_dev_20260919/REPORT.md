# Common output-limit development smoke check

This is a one-question development-only protocol check, not a comparative
answer-quality experiment. It reused the prior uncapped retrieval contexts for
`facebook_direct_01` so that only the answer request limit changed. The local
model was `qwen3:4b-instruct-2507-q4_K_M`, temperature zero, with
`max_tokens=256` on each of seven generated arms. C7 copied C2's answer.

The local OpenAI-compatible endpoint accepted the limit. Six requests reached
the model and one was a cache hit under the new limit-specific cache key. Every
generated arm reported `finish_reason=stop`; none reported `length`. Completion
token counts ranged from 71 to 84, well below the 256-token ceiling. The
summary records `max_answer_tokens` and `length_limited_answers=0`, and the raw
answer file records each arm's requested limit and finish reason.

This check does **not** equalize input prompt tokens or resolve the larger
context-budget confound. It also cannot establish that 256 tokens suffice for
all future questions. Before final evaluation, predeclare how length-truncated
answers will be counted and freeze the primary versus 3,000-character context
sensitivity analysis.

To reproduce in a fresh output directory:

```bash
python3 scripts/run_eight_condition_answers.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --questions data/facebook_large/facebook_questions_dev.json \
  --contexts results/facebook_eight_contexts_smoke_dev_20260919/contexts.jsonl \
  --output results/facebook_eight_answers_capped_smoke_REPRODUCED \
  --limit 1 \
  --max-answer-tokens 256
```
