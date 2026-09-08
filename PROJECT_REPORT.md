# Query-Adaptive Graph Propagation for Knowledge-Graph Retrieval

## Complete implementation and experimental report

**Project location:** `/Users/feiyuzhang/Desktop/COMP90055/agp_research`  
**Status:** Working prototype with a prepared real graph, development benchmark,
and held-out test set  
**Python requirement:** Python 3.10 or newer  
**External dependencies:** None for local retrieval and evaluation; an OpenAI-compatible API is optional

## 1. Executive summary

This project implements a small, understandable research prototype for query-adaptive graph propagation (AGP). It was created as a standalone project rather than as another modification to Microsoft GraphRAG. This separation makes the research algorithm easier to inspect, test, change, and evaluate.

The system accepts a natural-language question, extracts entity keywords, maps those keywords to knowledge-graph nodes, selects graph-propagation parameters, propagates relevance through the graph, and returns the highest-scoring graph evidence. If an LLM API is configured, the system can also ask the LLM to answer the original question using only the retrieved graph context.

The principal research question is:

> Does selecting graph-propagation parameters separately for each question improve retrieval and answer quality compared with using the same parameters for every question?

Five retrieval strategies are supported:

1. **Seed-only:** exact entity matches without graph propagation.
2. **Fixed AGP:** the same propagation parameters for every question.
3. **Rule-adaptive AGP:** transparent rules select parameters from the wording of each question.
4. **LLM-adaptive AGP:** an LLM extracts keywords and selects parameters for each question.
5. **LLM-parameter AGP:** local exact-title extraction is retained while an LLM
   selects only depth, decay, and top-k. This separates parameter selection from
   LLM entity-extraction errors.

The prototype includes a command-line interface, a demonstration graph, the UCI
Facebook Large Page-Page graph, deterministic topology-grounded development and
test questions, retrieval evaluation, JSON Lines logging, a C++ reference-code
adapter, `.env` configuration, and 31 passing unit tests. A three-strategy
development experiment is complete. The held-out test set remains intentionally
unused until model and parameter choices are frozen.

## 2. Scope and relationship to Microsoft GraphRAG

Microsoft GraphRAG is a large production-oriented system. It performs document ingestion, text chunking, entity and relationship extraction, graph construction, community detection, summarization, embedding, retrieval, and LLM answer generation. Understanding and changing that complete system is unnecessary for testing the central AGP hypothesis.

This prototype therefore does not require GraphRAG at runtime and does not modify GraphRAG. It uses a deliberately narrow interface:

- `nodes.csv` contains graph entities and their descriptions.
- `edges.csv` contains relationships between graph entities.
- `questions.json` contains evaluation questions and relevant-node labels.

GraphRAG can later be used as one possible source of graph data. Its entity and relationship tables can be exported into these simple files. The adaptive propagation experiment can then remain isolated from GraphRAG's internal indexing and query-control code.

## 3. End-to-end system behavior

For a question such as:

> How are Donald Trump and Climate Change connected?

the system performs the following sequence.

### 3.1 Keyword extraction

For `seed-only`, `fixed`, and `rules`, the local extractor finds complete graph-node titles occurring in the question. In the example, it finds:

```json
["Climate Change", "Donald Trump"]
```

For `llm`, an LLM is prompted to return entity keywords together with the propagation parameters in structured JSON.

If no graph title occurs in a question, a simple token heuristic is used only to expose potential unmatched keywords. It is not intended as a sophisticated entity recognizer.

### 3.2 Exact keyword-to-node mapping

Both keywords and graph titles are normalized by:

- converting text to case-insensitive form;
- removing leading and trailing whitespace;
- reducing repeated internal whitespace to one space.

The system then performs exact title matching. For example, `donald   trump` matches the node title `Donald Trump`. Aliases, abbreviations, spelling errors, and implicit entities are not handled in the current version.

Matched entities become seed nodes. Unmatched keywords are retained in the output so entity-mapping failures can be measured and inspected.

### 3.3 Parameter selection

The current AGP parameter space contains three variables:

| Parameter | Meaning | Allowed range in code |
|---|---|---|
| `depth` | Maximum number of propagation hops | 0–5 |
| `decay` | Discount applied to evidence at each additional hop | 0.0–1.0 |
| `top_k` | Maximum number of returned nodes | Positive integer |

The LLM planner further restricts `top_k` to 5, 10, or 20 to make its decisions easier to compare.

The rule planner currently uses these defaults:

| Detected question style | Depth | Decay |
|---|---:|---:|
| Direct factual | 1 | 0.40 |
| Comparison | 2 | 0.60 |
| Explanation or impact | 2 | 0.70 |
| Multi-hop connection or path | 3 | 0.75 |

These rules form an interpretable adaptive baseline. They are not claimed to be optimal.

### 3.4 Graph propagation

The implementation uses an undirected weighted graph. A relationship contributes in both directions because the immediate research objective is contextual neighborhood retrieval rather than directed logical inference.

Seed mass is distributed equally among all matched seed nodes. At each hop, the current mass at a node is distributed among its neighbors in proportion to non-negative edge weights. The relevance accumulated across all hops is:

\[
\pi = \sum_{i=0}^{L} \alpha^i M^i x
\]

where:

- \(x\) is the normalized seed vector;
- \(M\) is the row-normalized weighted adjacency operator;
- \(L\) is `depth`;
- \(\alpha\) is `decay`;
- \(\pi_v\) is the final relevance score for node \(v\).

The calculation is matrix-free: it uses adjacency dictionaries rather than constructing a dense matrix. This is easy to understand and adequate for a research prototype. It also avoids adding NumPy or graph-library dependencies.

Self-loops and zero-weight edges do not propagate mass. Missing edge endpoints are rejected while loading the graph. Results are ordered by decreasing score, with node ID as a deterministic tie-breaker, and truncated to `top_k`.

### 3.5 Context construction

The selected nodes are converted to readable context containing:

- node title;
- propagation score;
- node description;
- every graph relationship whose two endpoints are both selected.

This makes each retrieval result auditable. A researcher can see why an answer model received particular evidence instead of treating retrieval as an opaque step.

### 3.6 Answer generation

When an API client is configured, the answer model receives the original question and retrieved graph context. Its system instruction says to answer only from that context and to report when the evidence is insufficient.

Without an API key, all graph retrieval and evaluation features still work. Answer generation returns a clear skipped message rather than failing silently.

## 4. Implemented files

### Application package

| File | Responsibility |
|---|---|
| `agp_research/models.py` | Defines nodes, edges, parameters, ranked results, and pipeline output. |
| `agp_research/config.py` | Loads optional `.env` settings while giving shell variables precedence. |
| `agp_research/graph.py` | Loads CSV data, validates endpoints, and builds cached title indexes for exact matching. |
| `agp_research/propagation.py` | Implements matrix-free truncated graph propagation. |
| `agp_research/planner.py` | Implements local keyword extraction, rule-based parameters, and LLM planning. |
| `agp_research/llm.py` | Implements a minimal OpenAI-compatible chat client using the Python standard library. |
| `agp_research/pipeline.py` | Connects planning, matching, propagation, context construction, and answering. |
| `agp_research/evaluation.py` | Calculates retrieval metrics and executes batch experiments. |
| `agp_research/cli.py` | Provides the `ask` and `experiment` command-line interfaces. |
| `agp_research/__main__.py` | Allows execution with `python3 -m agp_research`. |
| `agp_research/paper_backend.py` | Maps application graphs and queries to the optional C++ reference backend. |

### Data and tests

| File | Responsibility |
|---|---|
| `data/nodes.csv` | Eight-node demonstration knowledge graph. |
| `data/edges.csv` | Nine weighted demonstration relationships. |
| `data/questions.json` | Three demonstration questions with relevant-node labels. |
| `tests/test_graph.py` | Tests loading and normalized exact matching. |
| `tests/test_propagation.py` | Tests propagation scores and empty-seed behavior. |
| `tests/test_evaluation.py` | Tests precision, recall, hit rate, and reciprocal rank. |
| `tests/test_config.py` | Tests `.env` parsing, precedence, and validation. |
| `tests/test_planner.py` | Tests rule-based query classification and parameters. |
| `tests/test_paper_backend.py` | Tests C++ adapter configuration and parsing behavior. |
| `results/experiment.jsonl` | Saved question-level outputs from the verified sample experiment. |
| `data/facebook_large/nodes.csv` | 22,470 converted Facebook page nodes. |
| `data/facebook_large/edges.csv` | 170,823 usable Facebook relationships. |
| `data/facebook_large/facebook_questions_dev.json` | 20 development questions. |
| `data/facebook_large/facebook_questions_test.json` | 40 held-out test questions. |
| `scripts/prepare_facebook_large.py` | Reproducibly converts and validates the UCI files. |
| `scripts/generate_facebook_questions.py` | Generates and validates topology-grounded benchmark labels. |
| `results/facebook_dev.jsonl` | 60 development question-strategy traces. |

`pyproject.toml` defines the package and an optional `agp` console command. `.gitignore` excludes generated caches, virtual environments, and result logs.

## 5. Installation and execution

### 5.1 Fastest method: run directly

Open a terminal and enter:

```bash
cd /Users/feiyuzhang/Desktop/COMP90055/agp_research
python3 --version
```

Python must be version 3.10 or newer. No package installation is needed for the core system.

Run a rule-adaptive question without calling an LLM:

```bash
python3 -m agp_research ask \
  "How are Donald Trump and Climate Change connected?" \
  --strategy rules \
  --no-answer
```

The JSON output includes:

- extracted keywords;
- matched seed-node IDs;
- unmatched keywords;
- selected propagation parameters;
- ranked nodes and their scores;
- textual graph context;
- elapsed time;
- whether an LLM was used.

### 5.2 Optional editable installation

For development, create an isolated environment and install the project:

```bash
cd /Users/feiyuzhang/Desktop/COMP90055/agp_research
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --editable .
```

The same program can then be invoked with `agp`:

```bash
agp ask "What did Donald Trump do regarding the Paris Agreement?" \
  --strategy fixed --depth 2 --decay 0.6 --top-k 5 --no-answer
```

### 5.3 Run the tests

The dependency-free test command is:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Current verified result:

```text
Ran 31 tests
OK
```

If `pytest` is already available, `python3 -m pytest` can also run the tests.

## 6. LLM configuration

LLM access is optional and credentials must not be committed. The easiest setup
is to copy the safe template and edit the private file:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your-api-key
AGP_MODEL=a-model-available-to-your-account
OPENAI_BASE_URL=https://api.openai.com/v1
AGP_LLM_TIMEOUT=60
```

The real `.env` is ignored by Git. Shell environment variables remain supported
and take precedence. For a compatible provider, change `OPENAI_BASE_URL`.

```bash
export OPENAI_BASE_URL="https://provider.example/v1"
```

Run LLM-adaptive retrieval and answer generation:

```bash
python3 -m agp_research ask \
  "Compare Donald Trump and Joe Biden regarding the Paris Agreement." \
  --strategy llm
```

The client uses a chat-completions-compatible endpoint and requests JSON for planning. Provider compatibility should be verified before a large experiment. Record the provider, exact model identifier, prompt, temperature, date, and any API version in the methodology.

For a fair first experiment, use temperature zero. If a stochastic temperature is later used, repeat each condition multiple times and report both mean and variation.

## 7. Data formats

### 7.1 Nodes

`data/nodes.csv` requires:

```csv
id,title,description
n1,Example Entity,Description used as retrieval evidence.
```

Node IDs must be unique. Titles should contain the canonical form used for exact mapping.

### 7.2 Edges

`data/edges.csv` requires:

```csv
source,target,weight,description
n1,n2,1.0,Explanation of the relationship.
```

Every endpoint must exist in `nodes.csv`. Weights must be numeric. Negative weights are accepted by loading but treated as zero during propagation; for clear experiments, input data should use non-negative weights explicitly.

### 7.3 Questions and relevance labels

`data/questions.json` requires:

```json
[
  {
    "id": "q1",
    "question": "A natural-language question",
    "relevant_node_ids": ["n1", "n7"]
  }
]
```

`relevant_node_ids` is the gold evidence set used to evaluate retrieval. These labels must be prepared independently of system results to avoid favoring one retrieval strategy.

## 8. Running experiments

### 8.1 Reproduce the demonstration

```bash
cd /Users/feiyuzhang/Desktop/COMP90055/agp_research

python3 -m agp_research experiment \
  --strategies seed-only fixed rules \
  --output results/experiment.jsonl
```

This executes all three sample questions under each of three strategies, producing nine JSON Lines records.

The verified demonstration produced approximately:

| Strategy | Precision | Recall | Hit rate | MRR |
|---|---:|---:|---:|---:|
| Seed-only | 1.000 | 0.444 | 1.000 | 1.000 |
| Fixed AGP | 0.633 | 1.000 | 1.000 | 1.000 |
| Rule-adaptive AGP | 0.656 | 0.889 | 1.000 | 1.000 |

Latency is also reported, but on this tiny graph it is measured in fractions of a millisecond and is not meaningful for system-level conclusions.

These numbers demonstrate expected trade-offs: seed-only retrieval is precise but misses connected evidence, while propagation increases recall and can introduce irrelevant nodes. The fixed configuration happens to obtain the highest recall on these three hand-built questions. This is not evidence that fixed AGP is generally superior because the graph, labels, and question count are only a software demonstration.

### 8.2 Add the LLM-adaptive condition

After configuring the API:

```bash
python3 -m agp_research experiment \
  --strategies seed-only fixed rules llm \
  --output results/experiment_with_llm.jsonl
```

Batch evaluation currently disables answer generation so retrieval strategies can be compared without paying for or conflating a second LLM call. Under the `llm` strategy, the planning call is still made to extract keywords and choose parameters.

### 8.3 Use another graph or question set

```bash
python3 -m agp_research \
  --nodes /absolute/path/to/nodes.csv \
  --edges /absolute/path/to/edges.csv \
  experiment \
  --questions /absolute/path/to/questions.json \
  --strategies seed-only fixed rules \
  --output results/my_experiment.jsonl
```

Global `--nodes` and `--edges` arguments must appear before the `experiment` or `ask` subcommand.

### 8.4 Facebook Large benchmark

The real-data stage now uses the UCI Facebook Large Page-Page Network. The source
archive is preserved outside this project, while deterministic conversion outputs
are stored in `data/facebook_large/`. The converted graph has 22,470 nodes,
170,823 non-self edges, one connected component, and four page categories. The
converter removes 179 self-loops, validates endpoints, disambiguates duplicate
normalized page names, and records checksums in `metadata.json`.

The benchmark generator creates labels from explicit graph rules without viewing
AGP rankings. Its four balanced question types are:

1. all direct neighbors of one seed;
2. common direct neighbors of two seeds;
3. two internal nodes on a unique shortest path of exactly three edges;
4. same-category nodes at shortest-path distance exactly two.

There are 20 development questions (5 per type, 30 unique seed pages) and 40
held-out test questions (10 per type, 60 unique seed pages). No seed page appears
in both splits. The generator also verifies that the project's local keyword
extractor maps every question to exactly the intended seed IDs.

The completed development run used `top_k=10`. The fixed experiment baseline in
`evaluation.py` is currently `depth=2`, `decay=0.6`, and `top_k=10`.

| Strategy | Precision@10 | Recall@10 | Hit rate | MRR |
|---|---:|---:|---:|---:|
| Seed-only | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fixed AGP | 0.3377 | 0.8800 | 0.9000 | 0.3142 |
| Rule-adaptive AGP | 0.4159 | 0.8800 | 0.9000 | 0.3130 |

These are development results, not final conclusions. Seed-only obtains zero for
this benchmark because labels deliberately describe related pages rather than the
seed itself. Rule adaptation improves overall precision mainly by using depth 1
for direct-neighbor questions. Both fixed and rule strategies achieve only 0.52
mean recall on the development similarity questions, making that the clearest
current failure mode. The held-out test set has not been run.

## 9. Experimental methodology for the real study

### 9.1 Research questions

A focused study can answer:

- **RQ1:** Does graph propagation improve evidence recall over exact seed retrieval?
- **RQ2:** Does query-adaptive propagation outperform one fixed parameter configuration?
- **RQ3:** Does an LLM select better parameters than transparent question-type rules?
- **RQ4:** Do retrieval improvements produce more correct and faithful answers?
- **RQ5:** What additional latency and API cost does adaptive planning introduce?

### 9.2 Experimental conditions

Run all questions under the same four conditions:

| Condition | Keyword extraction | Propagation parameters |
|---|---|---|
| B0: Seed-only | Local exact-title detection | Depth 0 |
| B1: Fixed AGP | Local exact-title detection | One tuned fixed configuration |
| A1: Rule AGP | Local exact-title detection | Question-type rules |
| A2: LLM AGP | LLM | Per-question LLM prediction |

This design means A2 changes both keyword extraction and parameter selection. To identify which component causes a difference, add two ablations if time permits:

- local keywords plus LLM parameters;
- LLM keywords plus fixed parameters.

That produces a cleaner causal analysis.

### 9.3 Development and test split

Do not select fixed parameters on the final test set. Instead:

1. divide questions into development and test sets;
2. use the development set to select fixed `depth`, `decay`, and `top_k`;
3. freeze the selected configuration;
4. evaluate every strategy once on the test set.

If the dataset is small, use repeated cross-validation while ensuring related or templated questions do not leak across folds.

### 9.4 Fixed-parameter tuning

A manageable grid is:

- `depth`: 1, 2, 3;
- `decay`: 0.3, 0.5, 0.7, 0.9;
- `top_k`: 5, 10, 20.

This creates 36 fixed configurations. Choose the best development configuration using the primary metric declared before testing, such as recall@10 or F1@10. Avoid choosing whichever metric happens to make the proposed method look strongest.

### 9.5 Retrieval metrics

The current implementation reports:

- **Precision@k:** fraction of retrieved nodes that are relevant;
- **Recall@k:** fraction of labeled relevant nodes that were retrieved;
- **Hit rate:** whether at least one relevant node was retrieved;
- **Mean reciprocal rank:** rewards placing the first relevant node near the top;
- **Latency:** elapsed pipeline time.

For the final study, also consider F1@k and nDCG@k if graded relevance labels are available. Report per-question results as well as macro averages. Confidence intervals or paired bootstrap intervals are preferable to relying only on mean values.

### 9.6 Answer-quality evaluation

Retrieval quality and answer quality must be evaluated separately. Extend the batch runner to generate an answer for each saved context, then score:

- correctness against a reference answer;
- faithfulness to retrieved evidence;
- completeness;
- relevance;
- abstention when evidence is insufficient;
- input/output tokens, API cost, and latency.

Human evaluation is strongest but expensive. A practical design is blind human assessment on a representative subset, supplemented by an LLM judge using a fixed rubric. Randomize system labels and order to reduce evaluator bias. Never ask a judge to score faithfulness without giving it the exact retrieved context.

### 9.7 Statistical comparison

All strategies answer the same questions, so comparisons are paired. Recommended reporting includes:

- mean and median per strategy;
- 95% bootstrap confidence intervals;
- paired bootstrap tests or a paired non-parametric test;
- effect sizes, not only p-values;
- correction for multiple comparisons when many conditions are tested.

Save raw outputs and analysis scripts so every table can be regenerated.

## 10. Real-data workflow status

### Phase 1: Establish reliable graph data — completed

The UCI Facebook Large graph has been downloaded, preserved, converted, and
validated. Stable prefixed node IDs, unique exact-match titles, categories, and
unit-weight relationships are available in `data/facebook_large/`. Detailed
provenance and checksums are recorded in `FACEBOOK_LARGE_DATASET.md`.

The conversion and validation process checks:

- duplicate or ambiguous titles;
- missing descriptions;
- disconnected components;
- invalid endpoints;
- extreme-degree hub nodes;
- whether directed relationships should remain directed;
- how original GraphRAG edge weights should be interpreted.

### Phase 2: Construct the controlled question dataset — completed

The deterministic generator produced 60 balanced topology questions across:

- direct factual questions;
- comparisons;
- unique three-hop paths;
- same-category two-hop similarity questions.

Every question records its intended seed nodes, relevant evidence nodes, question
type, and generation rule. This controlled benchmark does not include reference
answers or human semantic labels; those remain a later complementary evaluation.

### Phase 3: Establish fixed baselines — development tuning completed

Seed-only, fixed AGP, and rule-adaptive AGP have been run on development data.
A 36-configuration search selected depth 2, decay 0.3, top-k 5 by mean F1,
with a separately recorded k=10 baseline for matched-budget comparisons.
Per-question-type summaries and all traces are saved. See
[FIXED_PARAMETER_TUNING.md](FIXED_PARAMETER_TUNING.md).

### Phase 4: Evaluate adaptation — partially completed

Rule-adaptive and zero-shot `llm-parameters` development results are recorded.
The latter uses local keywords so parameter selection is isolated. It achieved
precision 0.4125, recall 0.8300, and mean per-question F1 0.5134. Few-shot
planning, end-to-end `llm`, further ablations, and prompt freezing remain.

### Phase 5: Evaluate answers — not started

Generate answers from identical answer prompts and model settings, varying only retrieved context. Evaluate correctness and faithfulness, and compare retrieval improvements with answer improvements.

### Phase 6: Analyze failures

Classify each failure into:

1. keyword extraction failure;
2. exact node-mapping failure;
3. inappropriate parameter prediction;
4. graph propagation noise;
5. missing or incorrect graph evidence;
6. context-construction failure;
7. unsupported or incomplete LLM answer.

This taxonomy will make the discussion more informative than reporting aggregate scores alone.

## 11. Current validation

The delivered project has been checked in the target directory:

- 31 unit tests pass using the Python standard library;
- the demonstration graph and 22,470-node Facebook graph load successfully;
- the Facebook conversion checksum and row-count checks pass;
- both Python and paper backends complete a real-data NASA query;
- the generator reproducibly creates 20 development and 40 test questions;
- topology definitions, nonempty labels, ID validity, keyword extraction, type
  balance, and split seed independence are validated;
- 60 development question-strategy traces are saved in
  `results/facebook_dev.jsonl`;
- regenerating the benchmark with seed 90055 produces identical question files.

The OpenAI-compatible path is implemented and configurable through `.env`. A
local Qwen zero-shot `llm-parameters` development run is recorded with model and
input checksums. It does not beat the rules condition on development mean F1.
The end-to-end `llm` condition is not yet a frozen research run.

## 12. Limitations

The current version intentionally prioritizes clarity over production complexity.

- Exact matching cannot resolve aliases, abbreviations, spelling errors, or implicit entities.
- Local keyword extraction knows graph titles and therefore is not a general named-entity recognizer.
- LLM planning combines keyword extraction and parameter selection, which requires ablations for causal interpretation.
- The graph is treated as undirected.
- Propagation is implemented with Python dictionaries and is not optimized for very large graphs.
- Edge weights are assumed to represent positive retrieval strength.
- The rule planner is based on surface phrases and may misclassify questions.
- The small demonstration contains only eight nodes and three questions; it is
  retained for teaching and software checks, not used as the real benchmark.
- The Facebook questions have objective topology labels but are templated and do
  not establish semantic relevance or answer correctness.
- Current batch evaluation measures retrieval but does not generate or score answers.
- The LLM client depends on a chat-completions-compatible API and structured JSON support.
- There is no caching, retry, or token/cost logging for API experiments yet.
- The batch runner accepts fixed parameters through `experiment --depth`,
  `--decay`, and `--top-k`, but automated grid search is not implemented.
- Fine-tuning a small model has not been implemented and is not justified until prompted baselines are evaluated.

These limitations are appropriate for the first prototype and provide concrete directions for extensions and ablation studies.

## 13. Prioritized next steps

### Immediate

1. Use the now-configurable batch `--depth`, `--decay`, and `--top-k` options to
   compare fixed configurations; other strategies retain their own parameters.
2. Add per-question-type summaries and preferably F1@k.
3. Fixed tuning is complete on the 20 development questions. The declared
   36-configuration search selected depth 2, decay 0.3, top-k 5 by mean F1.
   See [FIXED_PARAMETER_TUNING.md](FIXED_PARAMETER_TUNING.md) for all scores,
   ties, recall trade-offs, and the budget-matched k=10 baseline.
4. Analyze the two missed development similarity questions and decide whether
   planner or propagation changes are justified.
5. Freeze the chosen fixed baseline and rule definitions.

### Before the main experiment

6. Preserve the completed zero-shot `llm-parameters` result and add a separately
   named few-shot parameter-planning condition.
7. Add API retries, caching, latency, token, and cost logging.
8. Implement local-keywords/LLM-parameters and LLM-keywords/fixed-parameters
   ablations.
9. Freeze all code, prompts, model versions, metrics, and random settings.

### Main evaluation

10. Run the held-out Facebook test set once for every frozen condition.
11. Calculate paired uncertainty estimates and analyze results by question type.
12. Add manually authored semantic questions with independently prepared human
    relevance labels.
13. Generate answers from saved contexts and conduct blind correctness and
    faithfulness evaluation.
14. Classify failures using the taxonomy in Section 10.

### Optional extensions

15. Compare exact matching with alias or embedding-based matching.
16. Implement a persistent C++ service if dynamic edge updates become part of the
    research question.
17. Test directed propagation or relation-type-specific weights on a dataset that
    contains those semantics.
18. Consider fine-tuning only after prompted baselines show a measurable weakness
    and sufficient labeled parameter examples exist.

## 14. Suggested final dissertation/report structure

The eventual academic report can use this structure:

1. **Introduction:** motivation, problem statement, research questions, and contributions.
2. **Background:** GraphRAG, knowledge-graph retrieval, graph propagation, and adaptive retrieval.
3. **Method:** graph representation, entity mapping, propagation equation, parameter strategies, and answer generation.
4. **Experimental setup:** corpus, graph statistics, questions, annotations, baselines, models, prompts, metrics, and statistical tests.
5. **Results:** retrieval, answer quality, efficiency, ablations, and question-type breakdowns.
6. **Discussion:** why adaptation succeeds or fails, qualitative examples, limitations, and threats to validity.
7. **Conclusion:** direct answers to the research questions and future work.

The current document can support the implementation and methodology chapters, but final results and conclusions must be written only after the real experiment is completed.

## 15. Conclusion

The project now has a complete, executable foundation for studying query-adaptive graph propagation without requiring detailed knowledge of the Microsoft GraphRAG codebase. It supports reproducible local retrieval, multiple baselines, LLM-based adaptation, transparent context construction, batch logging, and standard retrieval metrics.

The project has progressed beyond architecture and data preparation: a real graph,
a controlled 60-question benchmark, split isolation, and initial development
baselines and a completed fixed-parameter search now exist. The immediate work is to freeze the rule
conditions on development data, evaluate the LLM condition and ablations, and only
then run the held-out test set. A later human-labeled semantic evaluation is still
needed because topology-defined questions alone cannot demonstrate answer quality.
Fine-tuning should remain optional until simpler prompted baselines establish a
clear need.

## Appendix A. Interface to the paper's AGP-Dynamic source code

An optional backend now connects this project to the authors' C++ reference
implementation. The published `DAGP` program is an experimental benchmark: it
hard-codes the seed vector and several query values, prints query time, and does
not serialize the returned score vector. It therefore cannot be used through its
original command line as a retrieval component.

The integration consists of:

- `paper_backend/agp_query_bridge.cpp`, which loads the authors' binary graph
  format, reads a query file, calls `Graph::query`, and emits every node score;
- `agp_research/paper_backend.py`, which maps application node IDs to contiguous
  1-based integers, exports the graph and query, invokes the bridge, parses scores,
  and maps results back to application nodes;
- `scripts/build_paper_backend.sh`, which compiles the bridge against an external
  checkout of the authors' source without copying that source into this project.

During integration testing, five assignments in the published `Graph::query`
were found to use 1-based `neighborID` values directly as indices into 0-based
arrays. This shifts scores and writes out of bounds for the maximum vertex ID.
The build script therefore applies
`paper_backend/patches/graph_query_zero_based.patch` to a build copy of
`Graph.cpp`; the external checkout remains unchanged. This compatibility change
must be disclosed when reporting experiments that use the reference code.

The interface exposes the paper parameters `a`, `b`, `delta`, relative error, and
exact (`N`) versus AGP-Static++ (`S`) query mode. The existing `depth` becomes
the truncation level `L`; existing `decay` is translated to personalized PageRank
weights `w_i = (1-decay) decay^i`; and exact-matched seeds form a normalized vector
`x`. See the README for build and execution commands.

This integration currently exercises the paper's query algorithm on a static graph.
Using the dynamic update advantage requires keeping one C++ graph instance alive
while processing insertions, deletions, and queries. A persistent JSON Lines or
socket service is the appropriate next extension if graph updates are part of the
experimental research question.
