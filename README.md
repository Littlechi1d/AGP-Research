# Query-Adaptive Graph Propagation Research Prototype

This is a small, standalone implementation of the research pipeline proposed for
COMP90055. It does not modify or require Microsoft GraphRAG.

For the prepared real-world Facebook Large Page-Page dataset, see
[`FACEBOOK_LARGE_DATASET.md`](FACEBOOK_LARGE_DATASET.md). It documents provenance,
conversion rules, checksums, exact run commands, and the remaining evaluation-label
work.

The topology-grounded development and test questions are documented in
[`FACEBOOK_BENCHMARK.md`](FACEBOOK_BENCHMARK.md).

## Research pipeline

1. Receive a natural-language question.
2. Extract graph entity keywords.
3. Map keywords to knowledge-graph nodes using normalized exact matching.
4. Select query-specific propagation depth, decay, and result count.
5. Propagate relevance through the graph.
6. Convert the highest-scoring nodes and relationships into textual context.
7. Ask an LLM to answer using only that context.

The principal research question is whether query-specific parameters improve
retrieval and answer quality compared with fixed parameters.

## Requirements

- Python 3.10 or newer
- No third-party package is required for the core prototype
- An OpenAI-compatible API key is optional

## Run without installation

From this directory:

```bash
python3 -m agp_research ask \
  "How are Donald Trump and Climate Change connected?" \
  --strategy rules \
  --no-answer
```

The command prints the extracted keywords, mapped seed nodes, selected parameters,
ranked nodes, propagation scores, and generated context.

Available strategies are:

- `seed-only`: return only exactly matched nodes; this is the no-propagation baseline.
- `fixed`: use the same depth, decay, and top-k for every question.
- `rules`: choose parameters from transparent question-type rules.
- `llm`: ask an LLM to extract keywords and predict parameters.

Example fixed configuration:

```bash
python3 -m agp_research ask \
  "What did Donald Trump do regarding the Paris Agreement?" \
  --strategy fixed --depth 2 --decay 0.6 --top-k 5 --no-answer
```

## Enable LLM parameter prediction and answer generation

Create a private settings file from the safe example:

```bash
cp .env.example .env
```

Open `.env` in a text editor and set your key and model:

```dotenv
OPENAI_API_KEY=your-key
AGP_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1
AGP_LLM_TIMEOUT=60
```

The real `.env` file is ignored by Git. Never commit or share it. Shell
environment variables still work and take precedence over values in `.env`.
For another OpenAI-compatible provider, change `OPENAI_BASE_URL`.

Then run:

```bash
python3 -m agp_research ask \
  "Compare Donald Trump and Joe Biden regarding the Paris Agreement." \
  --strategy llm
```

API behavior and model availability can change. Choose a model available to your
account and record its exact name, date, prompt, and settings in experiment logs.

## Run the sample experiment

```bash
python3 -m agp_research experiment \
  --strategies seed-only fixed rules \
  --output results/experiment.jsonl
```

This compares three retrieval conditions and reports macro averages for:

- precision@k;
- recall@k;
- hit rate;
- reciprocal rank;
- pipeline latency.

Each question-level trace is saved as JSON Lines for later analysis. Once an API
key is configured, add `llm` to `--strategies` to evaluate LLM-selected parameters.

## Use a real knowledge graph

Replace `data/nodes.csv` and `data/edges.csv`, or provide paths with `--nodes` and
`--edges`.

`nodes.csv` must contain:

```text
id,title,description
```

`edges.csv` must contain:

```text
source,target,weight,description
```

Every edge endpoint must reference an existing node ID. Edge weights must be
numeric. The current algorithm treats relationships as undirected because the
research task is contextual neighborhood retrieval.

## Prepare an experimental question set

Use the format in `data/questions.json`:

```json
[
  {
    "id": "q1",
    "question": "A natural-language question",
    "relevant_node_ids": ["n1", "n7"]
  }
]
```

Relevant nodes should be annotated before inspecting model results. Ideally, use
two annotators and document disagreement resolution.

## Recommended experiment

Compare these four systems on the same questions and graph:

1. Seed-only retrieval
2. Fixed AGP
3. Rule-adaptive AGP
4. LLM-adaptive AGP

Tune fixed parameters on a development set, freeze them, and report results once
on a separate test set. Keep LLM model, temperature, prompts, graph, and question
set constant. Run stochastic configurations more than once if temperature is not
zero.

After retrieval evaluation, add human or model-assisted answer evaluation for
correctness, relevance, and faithfulness. Store the retrieved context with every
answer so unsupported claims can be audited.

## Tests

With the standard library:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

If pytest is available:

```bash
python3 -m pytest
```

## Use the paper authors' AGP-Dynamic implementation

The project includes an optional adapter for the reference code accompanying
*Approximate Graph Propagation Revisited: Dynamic Parameterized Queries, Tighter
Bounds and Dynamic Updates*. The upstream executable is a benchmark and cannot
accept application seed nodes or return scores directly, so this project provides
a small C++ bridge that calls its `Graph::query` method.

The published `Graph::query` also uses 1-based vertex IDs as indices into 0-based
score arrays in five assignments. The build script applies the documented
`paper_backend/patches/graph_query_zero_based.patch` to a temporary build copy of
`Graph.cpp`. It does not modify the external checkout. Without this correction,
scores are shifted and the maximum vertex ID writes beyond the allocated array.

Clone the authors' repository beside this project (do not copy it into this
repository):

```bash
cd /Users/feiyuzhang/Desktop/COMP90055
git clone https://github.com/alvinzhaowei/AGP-dynamic.git
```

Build the bridge:

```bash
cd /Users/feiyuzhang/Desktop/COMP90055/agp_research
bash scripts/build_paper_backend.sh \
  /Users/feiyuzhang/Desktop/COMP90055/AGP-dynamic
```

Run a query with AGP-Static++ (`S`) from the authors' code:

```bash
python3 -m agp_research \
  --edges data/edges_unweighted.csv \
  --backend paper \
  --paper-a 0 --paper-b 1 \
  --paper-query-type S \
  ask "How are Donald Trump and Climate Change connected?" \
  --strategy rules --no-answer
```

The adapter maps `decay` to personalized PageRank weights
`w_i = (1 - decay) * decay**i`. It uses `delta = 1 / number_of_nodes`
unless `--paper-delta` is supplied, and computes the implementation's epsilon
from `--paper-relative-error` (default `0.1`).

Important differences from the local Python backend:

- the paper code supports undirected, **unweighted** graphs only;
- `a` and `b` must lie in `[0, 1]` and satisfy `a + b >= 1`;
- approximate mode `S` cannot safely initialize isolated nodes in the published
  implementation; remove them or use exact mode `N`;
- the returned values are randomized approximations for high-degree graphs;
- this adapter currently uses its static query algorithm; dynamic edge-update
  lifecycle support requires a persistent service rather than one process per query.

For experiments, add the global backend arguments before the subcommand:

```bash
python3 -m agp_research \
  --edges data/edges_unweighted.csv \
  --backend paper --paper-a 0 --paper-b 1 \
  experiment --strategies fixed rules \
  --output results/paper_backend.jsonl
```

## Next steps

1. Export a manageable real graph from GraphRAG into the two CSV files.
2. Create and manually annotate at least 50–100 questions.
3. Establish the seed-only and fixed-parameter results.
4. Evaluate rule-based parameter selection.
5. Evaluate zero-shot and few-shot LLM parameter selection.
6. Analyze failures: keyword extraction, node matching, parameter prediction,
   propagation, missing graph evidence, and answer generation.
7. Only consider fine-tuning a small model if prompting is measurably inadequate
   and enough labeled parameter examples exist.

## Important limitations

- Exact matching cannot handle aliases, spelling differences, or implicit entities.
- The sample graph and questions demonstrate software behavior, not research gains.
- Retrieval labels must be constructed for the actual evaluation dataset.
- LLM answers should not be evaluated without saving their supporting context.
- LLM fine-tuning is intentionally outside the first working version.
