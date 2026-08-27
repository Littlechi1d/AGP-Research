# Facebook Large Retrieval Benchmark

This benchmark evaluates whether query-adaptive graph propagation retrieves
topologically relevant Facebook pages better than fixed propagation.

## Files

- `data/facebook_large/facebook_questions_dev.json`: 20 questions for checking
  code and choosing parameters (5 per question type).
- `data/facebook_large/facebook_questions_test.json`: 40 held-out questions for
  one final evaluation (10 per question type).
- `data/facebook_large/facebook_questions_manifest.json`: generation settings and
  split statistics.
- `scripts/generate_facebook_questions.py`: deterministic generator and validator.

The two splits use different seed pages. Do not tune code or parameters after
looking at test-set scores.

## Ground-truth definitions

Each label is calculated from graph topology without inspecting AGP results:

1. **Direct**: all immediate neighbors of one seed page (3–10 answers).
2. **Comparison**: the common immediate neighbors of two nonadjacent seed pages
   (2–10 answers).
3. **Path**: the two internal pages on a unique shortest path of exactly three
   edges between two seed pages.
4. **Similarity**: pages in the seed's category whose shortest-path distance from
   it is exactly two (3–10 answers).

These are synthetic graph questions. They objectively test retrieval behavior,
but they do not claim that two Facebook pages are semantically similar merely
because they share a category and graph distance.

## Regenerate and validate

From `/Users/feiyuzhang/Desktop/COMP90055/agp_research`:

```bash
python3 scripts/generate_facebook_questions.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --targets ../datasets/facebook_large/raw/facebook_large/musae_facebook_target.csv \
  --output-dir data/facebook_large \
  --dev-per-type 5 \
  --test-per-type 10 \
  --random-seed 90055
```

The script checks the label rule for every example, verifies exact keyword
extraction behavior, requires nonempty labels, and rejects seed overlap between
development and test sets.

## Development experiment

```bash
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  experiment \
  --questions data/facebook_large/facebook_questions_dev.json \
  --output results/facebook_dev.jsonl \
  --strategies seed-only fixed rules
```

The initial reproducibility run produced these macro averages:

| Strategy | Precision@10 | Recall@10 | Hit rate | MRR |
|---|---:|---:|---:|---:|
| seed-only | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fixed | 0.3377 | 0.8800 | 0.9000 | 0.3142 |
| rules | 0.4159 | 0.8800 | 0.9000 | 0.3130 |

This is a development result, not the final research result. Rule adaptation
improves precision mainly because direct questions use depth 1 instead of the
fixed depth 2. Similarity questions are the current weakness: development recall
is 0.52 for both fixed and rule strategies at top-k 10.

## Final test (run only after choices are frozen)

Record the code version, fixed parameters, rule definitions, backend, and random
settings. Then run:

```bash
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  experiment \
  --questions data/facebook_large/facebook_questions_test.json \
  --output results/facebook_test.jsonl \
  --strategies seed-only fixed rules
```

Add `llm` only after configuring an API key and freezing the model and prompt.
The experiment command currently applies fixed parameters `depth=2`, `decay=0.6`,
and `top_k=10` inside `evaluation.py`. Changing CLI `ask` arguments does not alter
those experiment settings.

## Recommended next research step

Use only the development split to tune the fixed baseline and improve the
query-adaptive planner. In particular, test whether larger `top_k`, category-aware
filtering, or a changed propagation rule improves similarity recall. Once all
choices are frozen, run the held-out test once and report both overall and
per-question-type metrics. Finally, manually write natural questions and obtain
human relevance labels; the topology benchmark is controlled but cannot replace
a semantic evaluation.
