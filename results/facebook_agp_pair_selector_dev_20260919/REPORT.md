# Development-only AGP `(a,b)` selector study

This run tested a question-type selector for C7 using the **20 development
questions only**. The earlier held-out retrieval set was not used. The selector
chooses among the five predeclared native AGP-Static++ pairs; depth 2, decay
0.30, top-k 10, query type `S`, relative error 0.10, graph, and saved Qwen
keywords were held fixed. Approximate entity matching used threshold 0.85 and
margin 0.10.

## Fixed-arm retrieval results

| Arm | `(a,b)` | Mean question-level F1 |
|---|---|---:|
| C2 | `(0.00,1.00)` | 0.4532 |
| C3 | `(0.25,0.75)` | 0.4415 |
| C4 | `(0.50,0.50)` | 0.4415 |
| C5 | `(0.75,0.25)` | 0.4273 |
| C6 | `(1.00,0.00)` | 0.3916 |

The development-trained selector chose C2 for direct, comparison, and path
questions, and C3 for similarity questions. Its **in-sample** mean F1 was
0.4582, only 0.0050 above C2. That difference came from one similarity
question; it must not be treated as a validated improvement.

Leave-one-out validation re-trained the selector on 19 questions and predicted
the remaining question each time. Its mean F1 was **0.4532**, exactly equal to
the best global-arm comparator trained in the same folds (**0.4532**). Thus this
small development study does **not** demonstrate that question-type adaptation
outperforms a fixed `(a,b)` setting.

## Interpretation and limits

- Most questions have the same F1 under several pairs. A small change in score
  ordering often does not change which relevant nodes enter the top 10.
- The selector recognizes the four benchmark wording templates. Unknown wording
  falls back to C2; this is not general semantic understanding.
- These five pairs cover only `a+b=1`, not the entire valid parameter region.
- AGP-Static++ is approximate. A three-query repeat check for two questions
  found identical top-10 rankings within each tested pair, but this is not a
  comprehensive randomness or stability study.
- Retrieval F1 is not answer quality. The eight-condition answer study remains
  necessary, using a new untouched evaluation set and blind ratings.

Do not replace the fixed C2 baseline with C7 merely because the in-sample mean
is larger. Preserve C7 as the predeclared adaptive candidate and report its
actual outcome against all five fixed arms.

## Artifacts

- `manifest.json`: exact settings, hashes, arm means, and selector summary;
- `scores.jsonl`: keywords, matched seeds, five rankings and F1 scores for each
  development question;
- `selector.json`: loadable question-type mapping and fallback;
- `leave_one_out.json`: every out-of-fold decision and its paired comparator.

To reproduce in a new output folder from the project root:

```bash
python3 scripts/tune_agp_pair_selector.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --questions data/facebook_large/facebook_questions_dev.json \
  --keyword-results results/facebook_llm_keywords_fixed_dev_20260909/results.jsonl \
  --output results/facebook_agp_pair_selector_dev_REPRODUCED
```
