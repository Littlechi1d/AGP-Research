# Fixed-strategy development tuning

## Scope and protocol

This experiment tunes the three application-level fixed parameters on the
Facebook Large development benchmark. It does not tune the C++ paper backend,
modify propagation, call an LLM, or read/evaluate the held-out test questions.
The Python exact backend and the 20 existing development questions are used for
every configuration.

The search grid was declared before inspecting its results:

| Parameter | Candidates |
|---|---|
| Depth | 1, 2, 3 |
| Decay | 0.3, 0.5, 0.7, 0.9 |
| Top-k | 5, 10, 20 |

There are 36 configurations and 720 grid question-runs. An additional default
fixed reference and unchanged rule reference add 40 question-runs, for 760 total.
The grid is a practical search, not an exhaustive search over every allowed value
or a proof of a global optimum.

Selection maximizes the **mean of per-question F1**. For each question:

```text
F1 = 2 × precision × recall / (precision + recall)
```

F1 is zero when both precision and recall are zero. This is not the same as
computing F1 from the already-averaged precision and recall. Ties are resolved by
higher macro recall, then smaller top-k, smaller depth, and smaller decay.
Runtime is recorded but is not a selection criterion.

Precision uses the actual number of unique returned nodes as its denominator;
it is not padded to k when fewer nodes are returned. Seed nodes remain in the
ranked output and count as irrelevant when the labels ask only for related pages.
No changes were made to these evaluation conventions during tuning.

## Completed result (2026-09-03)

All 36 configurations and both references completed successfully: 760
question-runs, no LLM calls, and no held-out test evaluation.

**Selected fixed baseline: depth 2, decay 0.3, top-k 5.**

Seven configurations tied on maximum mean F1 and recall: depth 2 with any of the
four tested decay values, and depth 3 with decay 0.3, 0.5, or 0.7, all at k=5.
The predeclared tie-break selected depth 2 and decay 0.3. There is no evidence
that decay 0.3 is uniquely superior on this development set.

| Condition | Depth | Decay | k | Precision | Recall | Mean F1 | Hit rate | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original fixed reference | 2 | 0.6 | 10 | 0.3377 | 0.8800 | 0.4582 | 0.9000 | 0.3142 |
| Selected fixed | 2 | 0.3 | 5 | 0.4100 | 0.6244 | 0.4672 | 0.8500 | 0.3058 |
| Best fixed at k=10 | 2 | 0.3 | 10 | 0.3377 | 0.8800 | 0.4582 | 0.9000 | 0.3142 |
| Unchanged rules reference | Adaptive | Adaptive | 10 | 0.4159 | 0.8800 | 0.5218 | 0.9000 | 0.3130 |

The selected baseline improves mean F1 by about 0.0090 but loses about 0.2556
recall versus the original fixed baseline. This is a trade-off, not an improvement
on every metric. At k=10 the best fixed configuration ties the original baseline
on all reported retrieval metrics. The unchanged rules outperform both on mean
F1 in this development run; no statistical or held-out superiority is claimed.

### All grid results: mean per-question F1

| Depth | Decay | k=5 | k=10 | k=20 |
|---|---:|---:|---:|---:|
| 1 | 0.3 | 0.412010 | 0.414956 | 0.375965 |
| 1 | 0.5 | 0.412010 | 0.414956 | 0.375965 |
| 1 | 0.7 | 0.412010 | 0.414956 | 0.375965 |
| 1 | 0.9 | 0.412010 | 0.414956 | 0.375965 |
| 2 | 0.3 | 0.467248 | 0.458212 | 0.361294 |
| 2 | 0.5 | 0.467248 | 0.458212 | 0.361294 |
| 2 | 0.7 | 0.467248 | 0.458212 | 0.361294 |
| 2 | 0.9 | 0.467248 | 0.458212 | 0.361294 |
| 3 | 0.3 | 0.467248 | 0.441161 | 0.317637 |
| 3 | 0.5 | 0.467248 | 0.441161 | 0.317637 |
| 3 | 0.7 | 0.467248 | 0.441161 | 0.317637 |
| 3 | 0.9 | 0.452963 | 0.441161 | 0.317637 |

Best by F1 within each budget: k=5 uses (2, 0.3), k=10 uses (2, 0.3), and k=20
uses (1, 0.3). The k=20 F1 winner misses every similarity label, illustrating that
optimizing an overall average can hide a failed question category. The highest
recall in the grid is 0.97 at depth 3 and k=20, but that is not the declared F1
winner.

### Selected configuration by question type

| Type (5 questions each) | Precision | Recall | Mean F1 |
|---|---:|---:|---:|
| Direct | 0.7200 | 0.8143 | 0.7342 |
| Common neighbors | 0.4800 | 0.8500 | 0.5853 |
| Path intermediates | 0.2400 | 0.6000 | 0.3429 |
| Same-category two-hop | 0.2000 | 0.2333 | 0.2067 |

Similarity is still weak. Parameter tuning alone has not solved that retrieval
problem. The relevant nodes may be crowded out by seeds and nearer neighbors.

### Run the selected configuration

```bash
cd /Users/feiyuzhang/Desktop/COMP90055/agp_research
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  experiment \
  --questions data/facebook_large/facebook_questions_dev.json \
  --strategies fixed --depth 2 --decay 0.3 --top-k 5 \
  --output results/facebook_dev_selected_fixed.jsonl
```

Use `--top-k 10` for the budget-matched baseline. The application's defaults have
not been changed: always pass the selected settings explicitly. Individual
experiment output files are overwritten when reused.

## Reproduce the complete search

From the project directory:

```bash
python3 scripts/tune_fixed.py \
  --output-dir results/facebook_fixed_tuning_repeat
```

The default inputs are `data/facebook_large/nodes.csv`, `edges.csv`, and
`facebook_questions_dev.json`. The script refuses to reuse an existing output
directory. It does not load `.env` or construct an API client.

The completed run is saved in `results/facebook_fixed_tuning_20260903/`.
Each output directory contains:

- `manifest.json`: grid, objective, tie-breaks, Python version, code revision,
  source/input checksums, and timestamps;
- `grid_results.json`: overall and per-type metrics for all 36 configurations;
- `selection.json`: selected parameters, best at each k, and reference results;
- `fixed_d*_a*_k*.jsonl`: full question-level traces for each configuration;
- `reference_fixed.jsonl` and `reference_rules.jsonl`: fresh reference traces.

Each trace includes the effective parameters, seed IDs, ranked nodes and scores,
context, retrieval metrics, and latency. F1 is calculated by the tuning script;
the original experiment runner and its JSONL metric schema are unchanged.

## Interpretation limits

Only 20 templated development questions support this selection. Searching many
configurations on a small set risks overfitting; these are development results,
not evidence of generalization. The four question types are equally represented.
The same graph is used for all questions; this is not a graph-disjoint evaluation.

Changing top-k changes the retrieval budget. Compare a globally selected fixed
configuration against another system with that caveat, and use the separately
reported best fixed configuration at k=10 for a budget-matched comparison with
the current rule strategy.

Freeze the selected fixed configuration for subsequent comparisons. Continue
planner development and LLM ablations only on development data; leave the test set
untouched until all conditions, prompts, models, and metrics are frozen.
