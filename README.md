# Query-Adaptive Graph Propagation Research Prototype

This standalone COMP90055 prototype tests whether graph-propagation parameters
selected for each natural-language question improve knowledge-graph retrieval over
fixed parameters. It does not modify or require Microsoft GraphRAG at runtime.

## Current status

Completed components include:

- a question → entity seeds → propagation → context → optional LLM answer pipeline;
- seed-only, fixed, rule-adaptive, and LLM-adaptive strategies;
- a dependency-free Python propagation backend;
- a persistent in-process C++ API for the paper authors' AGP-Dynamic code;
- exact and thresholded approximate entity mapping, including split-title recovery;
- three prepared real social graphs: Facebook Large, GitHub MUSAE, and Deezer Europe;
- deterministic, seed-disjoint development and held-out questions with topology labels;
- completed development tuning and frozen held-out evaluations;
- a nine-condition comparison covering LLM-only, direct neighbours, five fixed
  AGP pairs, a rule selector, and a zero-shot LLM selector;
- a paired LLM-only versus AGP-grounded answer-quality study runner;
- 103 passing tests plus 20 parameterized subtests.

The original Facebook evaluation and the later GitHub/Deezer cross-dataset
evaluation are complete. The latter used a checksum-locked protocol and must not
be rerun as though it were unseen. Further runs are replications or exploratory.

Further documentation:

- [`PROJECT_REPORT.md`](PROJECT_REPORT.md): complete implementation report;
- [`FACEBOOK_LARGE_DATASET.md`](FACEBOOK_LARGE_DATASET.md): provenance,
  conversion, statistics, and checksums;
- [`FACEBOOK_BENCHMARK.md`](FACEBOOK_BENCHMARK.md): benchmark rules,
  regeneration, development results, and final-test protocol.
- [`ANSWER_QUALITY_PROTOCOL.md`](ANSWER_QUALITY_PROTOCOL.md): paired generation,
  automatic metrics, blind review, and final-study controls.
- [`EIGHT_CONDITION_EXPERIMENT.md`](EIGHT_CONDITION_EXPERIMENT.md): proposed
  eight-arm AGP experiment, shared budgets, controls, and remaining freeze items.
- [`ADDITIONAL_DATASETS.md`](ADDITIONAL_DATASETS.md): GitHub and Deezer source,
  conversion, question generation, and checksums.
- [`FINAL_EXPERIMENT_PROTOCOL.md`](FINAL_EXPERIMENT_PROTOCOL.md): frozen C0–C8
  settings and one-time evaluation rule.
- [`results/cross_dataset_frozen_evaluation_20260929/REPORT.md`](results/cross_dataset_frozen_evaluation_20260929/REPORT.md):
  final cross-dataset results.
- [`RESULTS_AND_DISCUSSION.md`](RESULTS_AND_DISCUSSION.md): dissertation-ready
  experimental setup, results by research question, discussion, and threats to
  validity.

## Pipeline

1. Receive a natural-language question.
2. Extract entity keywords.
3. Map keywords to graph nodes using normalized exact matching or conservative
   approximate title matching.
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
| `agp_research/graph.py` | Loads CSV graphs and caches exact/fuzzy title indexes. |
| `agp_research/agp_native_backend.py` | Calls a persistent in-process AGP C++ API. |
| `agp_research/eight_condition_retrieval.py` | Direct-neighbour baseline and five-pair native AGP handle pool for the proposed study. |
| `agp_research/agp_pair_selector.py` | Question-type selector that chooses one of the five fixed AGP pairs. |
| `scripts/tune_agp_pair_selector.py` | Scores five native pairs on development questions and validates C7 leave-one-out. |
| `scripts/run_eight_condition_contexts.py` | Saves C0–C7 base contexts before answer generation. |
| `scripts/run_eight_condition_answers.py` | Generates C0–C7 answers and blinded review artifacts. |
| `scripts/add_llm_c8.py` | Adds question-only LLM-selected C8 by reusing a fixed-arm context and answer. |
| `scripts/prepare_additional_social_graph.py` | Converts GitHub MUSAE and Deezer Europe archives. |
| `scripts/generate_additional_questions.py` | Generates balanced, seed-disjoint questions for the new graphs. |
| `scripts/validate_experiment_protocol.py` | Validates frozen inputs and implementation checksums. |
| `scripts/prepare_eight_condition_eval.py` | Builds an unscored, seed-disjoint candidate evaluation set and criteria-review form. |
| `scripts/rephrase_eight_eval_paths.py` | Makes a new candidate version with plain-language path questions while preserving answer IDs. |
| `agp_research/planner.py` | Extracts keywords and selects parameters. |
| `agp_research/propagation.py` | Runs local Python graph propagation. |
| `agp_research/paper_backend.py` | Calls the optional C++ backend. |
| `agp_research/pipeline.py` | Orchestrates the full workflow. |
| `agp_research/evaluation.py` | Calculates metrics and runs batches. |
| `agp_research/cli.py` | Implements `ask` and `experiment`. |
| `scripts/prepare_facebook_large.py` | Converts the UCI dataset. |
| `scripts/generate_facebook_questions.py` | Builds and validates the benchmark. |
| `scripts/tune_entity_mapping.py` | Tunes fuzzy matching on development data only. |

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

Use development-tuned approximate entity mapping with the persistent native AGP
backend:

```bash
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --backend native \
  ask "Which pages directly neighbor Chrisley Knows Best on USA" \
  --strategy llm-keywords --match-mode approximate --no-answer
```

Approximate mode tries exact matching first, then joins adjacent unresolved LLM
keywords when they may be pieces of one title, and finally applies normalized
edit similarity. The development-selected defaults are `--match-threshold 0.85`
and `--match-margin 0.10`. The margin requires the best candidate to beat the
runner-up, reducing ambiguous matches.

Reproduce the development-only mapping sweep:

```bash
python3 scripts/tune_entity_mapping.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --questions data/facebook_large/facebook_questions_dev.json \
  --keyword-results results/facebook_llm_keywords_fixed_dev_20260909/results.jsonl \
  --output results/facebook_entity_mapping_dev_REPRODUCED
```

The preserved result is in `results/facebook_entity_mapping_dev_20260915/`.

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
For any new benchmark, do not run its test set until all choices are frozen.

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

### Persistent native AGP API

The one-shot paper backend above is useful for compatibility checks, but reloads
the complete graph for every query. Build the persistent shared library instead:

```bash
bash scripts/build_agp_library.sh \
  /Users/feiyuzhang/Desktop/COMP90055/AGP-dynamic
```

Then select it through the existing pipeline:

```bash
python3 -m agp_research \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --backend native --paper-query-type S \
  ask "Which pages directly neighbor Census Australia" \
  --strategy rules --no-answer
```

`NativeAGPBackend` loads the graph and prepares AGP-Static++ once, then reuses the
same C++ graph handle for every query made through that backend instance. On the
22,470-node Facebook graph, the verified first call—including native graph setup—
took about 0.183 seconds; two subsequent calls using the same handle took about
0.0015 and 0.0014 seconds. These are implementation checks, not a controlled
performance study.

See [NATIVE_AGP_API.md](NATIVE_AGP_API.md) for the C/Python lifecycle, validation,
and current limitations.

### Eight-condition retrieval contexts (development smoke)

The proposed eight-arm answer study first saves every context before making any
answer-model calls. Reproduce its two-question development smoke run with:

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

Omit `--keyword-results` to call the configured LLM once per question for keyword
extraction. The runner refuses to overwrite an output directory. See the
[smoke report](results/facebook_eight_contexts_smoke_dev_20260919/REPORT.md)
for the important node-versus-token-budget limitation. Add
`--max-context-chars 3000` for the optional development-tested character-ceiling
sensitivity condition; it preserves complete evidence lines and records any
truncation, but is not a true token ceiling. The 20-question development output
is in `results/facebook_eight_contexts_char3000_dev_final_20260919/`.
The earlier `results/facebook_eight_contexts_char3000_dev_20260919/` is an
archived diagnostic with incomplete `context_truncated` flags; do not use it
for analysis. That Facebook candidate was not used for the later cross-dataset
final answer study. A one-question development answer smoke run using the earlier
uncapped contexts is documented in
[this report](results/facebook_eight_answers_smoke_dev_20260919/REPORT.md). The
eight-answer script creates condition-labelled raw answers, randomized A–H
review copies, separate answer keys, and blank rating forms. It requests a
shared 256-token output ceiling by default (`--max-answer-tokens` overrides it)
and records each call's finish reason. The [capped development smoke report](results/facebook_eight_answers_capped_smoke_dev_20260919/REPORT.md)
checks that the local model accepts the request.

### New eight-condition evaluation candidate

The [current candidate set](data/facebook_large/eight_condition_eval_candidate_v5_20260922/REPORT.md)
contains 40 new topology-labelled questions, ten of each type. Its seed pages
and exact question texts are disjoint from both earlier splits. Similarity
questions describe the relation in ordinary language: pages of the same dataset
category reached through one other page, without a direct connection. V3
clarified path-answer order, V4 replaced five questions with artificial
`[page ID]` suffixes in their expected titles, and V5 rephrased all ten path
questions in plain language. V5 remains unreviewed; V2–V4 are retained as its
revision trail. The still earlier V1 candidate is
archived separately and must not be used for the eight-arm evaluation. The
generator command below reproduces the original V2 candidate, not the later
revisions:

```bash
python3 scripts/prepare_eight_condition_eval.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --targets ../datasets/facebook_large/raw/facebook_large/musae_facebook_target.csv \
  --exclude-questions data/facebook_large/facebook_questions_dev.json \
    data/facebook_large/facebook_questions_test.json \
  --output data/facebook_large/eight_condition_eval_REPRODUCED \
  --per-type 10 --random-seed 20260919
```

An independent reviewer must check `criteria_review.csv` against the graph and
resolve ambiguous wording or incorrect labels before freezing this set. Do not
run the eight-arm answer comparison on it yet.
The [dated preflight record](EIGHT_CONDITION_PREFLIGHT_20260922.md) documents
the verified settings and remaining freeze gates. The original selector is
unchanged: on V5, C7 would choose C2 for all 40 questions. This cannot test an
adaptive-pair benefit and must be disclosed if V5 is used.

To reproduce V5 from the saved V4 candidate in a fresh directory:

```bash
python3 scripts/rephrase_eight_eval_paths.py \
  --previous data/facebook_large/eight_condition_eval_candidate_v4_20260919 \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --output data/facebook_large/eight_condition_eval_candidate_v5_REPRODUCED
```

### Completed GitHub and Deezer frozen evaluation

The later cross-dataset study used 40 untouched questions on GitHub MUSAE and
40 on Deezer Europe. Its machine-readable settings are in
`experiment_protocol_v1.json`; validate them with:

```bash
PYTHONPATH=. python3 scripts/validate_experiment_protocol.py
```

Direct neighbours achieved the highest overall retrieval F1 on GitHub (`0.4394`)
and Deezer (`0.4450`). C2 was the strongest fixed AGP arm (`0.4129`, `0.4134`).
C7 reproduced C2 because its frozen rules fell back for every new question. C8
scored `0.3682` and `0.3895`, so the zero-shot question-only LLM selector did not
beat the strongest fixed setting. See the
[complete report](results/cross_dataset_frozen_evaluation_20260929/REPORT.md).
Paired bootstrap intervals and dissertation-ready SVG figures are in
[`results/cross_dataset_statistical_analysis_20260930/`](results/cross_dataset_statistical_analysis_20260930/REPORT.md).
Representative direct, comparison, path, similarity, selector, and truncation
failures are documented in the accompanying
[`FAILURE_ANALYSIS.md`](results/cross_dataset_statistical_analysis_20260930/FAILURE_ANALYSIS.md).

## Tests

```bash
PYTHONPATH=. python3 -m pytest -q
```

Current verified result with `pytest`:

```text
103 passed, 20 subtests passed
```

## Next steps

1. Preserve the frozen Facebook, GitHub, and Deezer outputs; do not retune and
   present a rerun as unseen evaluation.
2. Add paired uncertainty estimates and compact figures for the final report.
3. Perform qualitative failure analysis by question type, especially the strong
   direct-neighbour baseline and the different behaviour on similarity questions.
4. Explain that C7 reduced to its C2 fallback and C8 did not beat the strongest
   fixed arm on either new graph.
5. Treat semantic human or model-assisted answer judging as optional follow-up:
   the exact-title automatic metric is conservative and many outputs reached the
   frozen 256-token ceiling.
6. Leave learned parameter classifiers and whole-graph LLM prompting as future
   work requiring a new training/evaluation design.

## Limitations

- Approximate title matching handles spelling variation and split titles, but not
  general aliases or implicit entities.
- The topology benchmark is controlled but cannot replace semantic evaluation.
- The graph is treated as undirected.
- The Python backend prioritizes clarity over large-scale optimization.
- Automatic answer metrics recognize explicit graph titles but not paraphrases;
  blind human evaluation remains necessary.
- GitHub and Deezer answer generation hit the 256-token ceiling for 140 and 166
  of 280 base calls respectively.
- C7 did not recognize the new question wording and always used C2; C8 selected
  C4 for every Deezer evaluation question.
- LLM responses are cached and request/token metadata is logged; bounded retries
  and provider-specific cost calculation are not yet implemented.
- The paper adapter currently supports static queries, not dynamic updates.
