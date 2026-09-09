# Query-Adaptive Graph Propagation Research Prototype

This standalone COMP90055 prototype tests whether graph-propagation parameters
selected for each natural-language question improve knowledge-graph retrieval over
fixed parameters. It does not modify or require Microsoft GraphRAG at runtime.

## Current status

Completed components include:

- a question → entity seeds → propagation → context → optional LLM answer pipeline;
- seed-only, fixed, rule-adaptive, and LLM-adaptive strategies;
- a dependency-free Python propagation backend;
- an optional C++ adapter for the paper authors' AGP-Dynamic code;
- the prepared UCI Facebook Large Page-Page graph;
- deterministic development and held-out test questions with topology labels;
- a completed three-strategy development experiment;
- 31 passing unit tests.

The held-out Facebook test set has intentionally not been evaluated. Freeze the
parameters and planner before using it.

Further documentation:

- [`PROJECT_REPORT.md`](PROJECT_REPORT.md): complete implementation report;
- [`FACEBOOK_LARGE_DATASET.md`](FACEBOOK_LARGE_DATASET.md): provenance,
  conversion, statistics, and checksums;
- [`FACEBOOK_BENCHMARK.md`](FACEBOOK_BENCHMARK.md): benchmark rules,
  regeneration, development results, and final-test protocol.

## Pipeline

1. Receive a natural-language question.
2. Extract entity keywords.
3. Map keywords to graph nodes using normalized exact-title matching.
4. Select `depth`, `decay`, and `top_k` using fixed values, rules, or an LLM.
5. Propagate relevance and rank nodes.
6. Convert selected nodes and edges into readable context.
7. Optionally ask an LLM to answer using only that context.

## Requirements

- Python 3.10 or newer;
- no third-party dependency for the core Python implementation;
- an OpenAI-compatible API key only for LLM planning or answering;
- a C++ compiler and external AGP-Dynamic checkout only for the paper backend.

## Important files

| Path | Purpose |
|---|---|
| `agp_research/config.py` | Reads `.env` and shell settings. |
| `agp_research/graph.py` | Loads CSV graphs and caches indexes for exact matching. |
| `agp_research/planner.py` | Extracts keywords and selects parameters. |
| `agp_research/propagation.py` | Runs local Python graph propagation. |
| `agp_research/paper_backend.py` | Calls the optional C++ backend. |
| `agp_research/pipeline.py` | Orchestrates the full workflow. |
| `agp_research/evaluation.py` | Calculates metrics and runs batches. |
| `agp_research/cli.py` | Implements `ask` and `experiment`. |
| `scripts/prepare_facebook_large.py` | Converts the UCI dataset. |
| `scripts/generate_facebook_questions.py` | Builds and validates the benchmark. |

## Run the small demonstration

```bash
cd /Users/feiyuzhang/Desktop/COMP90055/agp_research

python3 -m agp_research ask \
  "How are Donald Trump and Climate Change connected?" \
  --strategy rules --no-answer
```

Available strategies:

- `seed-only`: return exact seed nodes without propagation;
- `fixed`: use manually supplied parameters;
- `rules`: choose parameters from transparent question-wording rules;
- `llm`: ask an LLM for keywords and per-question parameters.
- `llm-keywords`: ask an LLM only for keywords while using supplied fixed
  parameters; this isolates entity extraction.
- `llm-parameters`: use local exact-title keywords and ask an LLM only for
  `depth`, `decay`, and `top_k`; this is the clean parameter-selection ablation.
- `llm-parameters-few-shot`: the same ablation with one synthetic parameter
  example for each of the four benchmark question types.

Fixed example:

```bash
python3 -m agp_research ask \
  "What did Donald Trump do regarding the Paris Agreement?" \
  --strategy fixed --depth 2 --decay 0.6 --top-k 5 --no-answer
```

## Configure optional LLM access

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENAI_API_KEY=your-key
AGP_MODEL=a-model-available-to-your-account
OPENAI_BASE_URL=https://api.openai.com/v1
AGP_LLM_TIMEOUT=60
AGP_LLM_CACHE_DIR=.agp_cache/llm
AGP_LLM_LOG_PATH=.agp_logs/llm_requests.jsonl
```

`.env` is ignored by Git. Shell environment variables take precedence. Never
commit, publish, or paste the real key into a report.

Identical calls are cached by default, so a rerun does not send the same model
request twice. Each call appends a metadata-only JSON record to
`.agp_logs/llm_requests.jsonl`, including latency, cache status, and provider
token counts when available. API keys, prompts, and responses are not logged.
Both directories are ignored by Git. Set either setting to an empty value to
disable that feature; delete `.agp_cache/llm` when you intentionally need fresh
model responses.

```bash
python3 -m agp_research ask \
  "Compare Donald Trump and Joe Biden regarding the Paris Agreement." \
  --strategy llm
```

To test LLM parameter selection while retaining reliable local title extraction:

```bash
python3 -m agp_research ask \
  "Which pages form a path between Donald Trump and Climate Change?" \
  --strategy llm-parameters --no-answer
```

Use `--strategy llm-parameters-few-shot` to run the separately named four-example
condition. It does not replace the zero-shot strategy.

This makes one LLM planning call. Without `--no-answer`, the same configured
model is called again to generate the final answer.

Record the provider, exact model, prompt, temperature, date, and API settings in
reproducible LLM experiments.

## Demonstration experiment

```bash
python3 -m agp_research experiment \
  --strategies seed-only fixed rules \
  --output results/experiment.jsonl
```

The runner reports macro-average precision, recall, hit rate, reciprocal rank,
and latency, and saves one JSON object per question/strategy pair.

## Facebook Large dataset

The converted graph in `data/facebook_large/` contains:

- 22,470 nodes;
- 170,823 usable undirected edges;
- one connected component;
- four categories: company, government, politician, and tvshow;
- 179 source self-loops removed;
- duplicate normalized titles deterministically disambiguated.

Regenerate it from the preserved UCI download:

```bash
python3 scripts/prepare_facebook_large.py \
  --source-dir ../datasets/facebook_large/raw/facebook_large \
  --output-dir data/facebook_large
```

Run one real-data query:

```bash
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  ask "Which pages are connected to NASA Student Launch through the network" \
  --strategy rules --no-answer
```

## Facebook benchmark and development result

The project contains 60 deterministic topology-grounded questions:

| Split | Questions | Per question type | Unique seed pages |
|---|---:|---:|---:|
| Development | 20 | 5 | 30 |
| Held-out test | 40 | 10 | 60 |

The splits have no seed overlap. The balanced types are direct neighbors, common
neighbors, unique shortest three-edge paths, and same-category pages exactly two
hops away. Labels are computed from graph topology before retrieval.

Regenerate and validate the benchmark:

```bash
python3 scripts/generate_facebook_questions.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --targets ../datasets/facebook_large/raw/facebook_large/musae_facebook_target.csv \
  --output-dir data/facebook_large \
  --dev-per-type 5 --test-per-type 10 --random-seed 90055
```

Run the development experiment:

```bash
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  experiment \
  --questions data/facebook_large/facebook_questions_dev.json \
  --output results/facebook_dev.jsonl \
  --strategies seed-only fixed rules
```

Verified macro averages:

| Strategy | Precision@10 | Recall@10 | Hit rate | MRR |
|---|---:|---:|---:|---:|
| Seed-only | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fixed AGP | 0.3377 | 0.8800 | 0.9000 | 0.3142 |
| Rule-adaptive AGP | 0.4159 | 0.8800 | 0.9000 | 0.3130 |

These are development results, not final findings. The rule strategy's precision
advantage mainly comes from depth 1 on direct questions. Similarity questions are
the current weakness.

The experiment command accepts `--depth`, `--decay`, and `--top-k` for the
`fixed` strategy only. Defaults remain 2, 0.6, and 10. For example, append
`--depth 1 --decay 0.3 --top-k 10` after `experiment` to evaluate another fixed
configuration. These options do not override the rules, seed-only, or LLM planner.
Use a distinct `--output` path for each configuration: existing logs are overwritten.
Do not run the test set until all choices are frozen.

## Data formats

`nodes.csv`:

```text
id,title,description
```

`edges.csv`:

```text
source,target,weight,description
```

Question JSON:

```json
[
  {
    "id": "q1",
    "question": "A natural-language question",
    "relevant_node_ids": ["n1", "n7"]
  }
]
```

All referenced IDs must exist. Research labels must be defined independently of
the evaluated rankings.

## Optional paper backend

Build the bridge against the external repository:

```bash
bash scripts/build_paper_backend.sh \
  /Users/feiyuzhang/Desktop/COMP90055/AGP-dynamic
```

Run it on Facebook Large:

```bash
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --backend paper --paper-a 0 --paper-b 1 --paper-query-type S \
  ask "Which pages are connected to NASA Student Launch through the network" \
  --strategy rules --no-answer
```

The build applies `graph_query_zero_based.patch` to a temporary copy of upstream
`Graph.cpp`; it does not alter the external checkout. The backend is undirected
and unweighted, approximate mode is randomized, and each CLI request currently
reloads the graph. Dynamic update lifecycle support is not implemented.

## Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Current verified result:

```text
Ran 44 tests
OK
```

## Next steps

1. Fixed tuning is complete: see [FIXED_PARAMETER_TUNING.md](FIXED_PARAMETER_TUNING.md).
   Use depth 2, decay 0.3, k=5 for the selected mean-F1 baseline; also report the
   budget-matched k=10 baseline. Defaults remain unchanged.
2. Improve and freeze the rule planner, especially for similarity questions.
3. Report metrics per question type (batch fixed parameters are now configurable).
4. Zero-shot and few-shot `llm-parameters` development runs are complete; see
   their reports under `results/facebook_llm_parameters*_dev_20260909/`. Few-shot
   improved recall but did not improve mean F1.
5. The `llm-keywords`/fixed-parameters ablation is complete; Qwen exactly matched
   18 of 20 development questions but split two long page titles.
6. Follow [FROZEN_EXPERIMENT_PROTOCOL.md](FROZEN_EXPERIMENT_PROTOCOL.md). Run an
   end-to-end `llm` development smoke test before the one-time held-out run. The
   smoke test is complete with mean F1 0.3977 and no prompt changes afterward.
7. Run the held-out test once and report paired uncertainty estimates.
8. Add manually authored questions and human relevance/answer labels.

## Limitations

- Exact matching does not handle aliases, misspellings, or implicit entities.
- The topology benchmark is controlled but cannot replace semantic evaluation.
- The graph is treated as undirected.
- The Python backend prioritizes clarity over large-scale optimization.
- Batch evaluation does not score answers or automatically search parameters.
- LLM responses are cached and request/token metadata is logged; bounded retries
  and provider-specific cost calculation are not yet implemented.
- The paper adapter currently supports static queries, not dynamic updates.
