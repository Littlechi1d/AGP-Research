# Eight-condition context runner: development smoke test

This is a retrieval-only software check on the **first two development
questions**. No answer model was called, and no held-out question was used.
The saved Qwen keyword outputs were mapped once per question, then shared by
C1–C7. The long title split into three keywords in the first question was
correctly recovered as seed `fb_8503`.

All eight condition records were produced for both questions. C0 has an empty
context; C1 returned respectively 3 and 8 direct neighbours; C2–C6 each
returned at most 10 native AGP-ranked nodes. C7 selected C2 for these direct
questions and copied its exact ranking and context. Seed-to-result edges appear
in C1's context even though the seed does not use a ranked-node slot.

## Important comparability finding

The shared budget currently limits **nodes**, not context text. For the first
question, C1's context had 1,463 characters, while the AGP contexts ranged
from 4,985 to 7,142 characters because they contained more nodes and induced
relationships. The second question ranged from 2,193 (C1) to about 2,883
characters (AGP). This is a potential answer-generation confound. Prompt-token
counts must be reported, and a predeclared token-capped sensitivity analysis
should be considered before final answer evaluation.

The first native query for each `(a,b)` handle includes graph initialization;
its `query_seconds` is therefore not steady-state query latency. C7's recorded
query time is copied from its chosen fixed arm and is not an eighth executed
AGP query. These facts are also recorded in `manifest.json`.

`contexts.jsonl` contains the exact text that a later answer runner would use.
The manifest records graph, question, selector, keyword, and native-library
hashes. This two-question smoke test is not a retrieval or answer-quality
conclusion.

To reproduce in a new directory:

```bash
python3 scripts/run_eight_condition_contexts.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --questions data/facebook_large/facebook_questions_dev.json \
  --selector results/facebook_agp_pair_selector_dev_20260919/selector.json \
  --keyword-results results/facebook_llm_keywords_fixed_dev_20260909/results.jsonl \
  --output results/facebook_eight_contexts_smoke_REPRODUCED \
  --limit 2
```
