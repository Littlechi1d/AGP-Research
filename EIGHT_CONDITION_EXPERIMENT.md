# Eight-condition AGP answer experiment: design specification

**Status:** design for implementation and development checks, **not yet frozen for
final evaluation**. The adaptive selector, a new untouched question set, and the
human-rating procedure still require validation and a dated freeze. Do not treat
the earlier 40-question held-out retrieval set as unseen again.

## Research question

Does choosing AGP degree-normalization parameters `(a, b)` for each question
improve retrieval and answer quality over (i) no graph evidence, (ii) immediate
neighbour evidence, and (iii) five fixed AGP configurations?

The principal adaptive-versus-fixed comparison changes **only `(a, b)`**. It
does not also adapt depth, decay, `top_k`, entity mapping, or answer prompts.
Other adaptive strategies in the prototype are separate prior experiments.

## The eight conditions

| ID | Name | Evidence supplied to the answer model | `(a, b)` |
|---|---|---|---|
| C0 | LLM-only | No graph context | N/A |
| C1 | Direct neighbours | Immediate neighbours of the mapped seeds | N/A |
| C2 | Fixed AGP 0/1 | Native AGP-Static++ ranking | `(0.00, 1.00)` |
| C3 | Fixed AGP 1/4–3/4 | Native AGP-Static++ ranking | `(0.25, 0.75)` |
| C4 | Fixed AGP 1/2–1/2 | Native AGP-Static++ ranking | `(0.50, 0.50)` |
| C5 | Fixed AGP 3/4–1/4 | Native AGP-Static++ ranking | `(0.75, 0.25)` |
| C6 | Fixed AGP 1/0 | Native AGP-Static++ ranking | `(1.00, 0.00)` |
| C7 | Adaptive AGP | Native AGP-Static++ ranking from a question-selected pair | One of C2–C6 |

The five pairs obey the current API constraints `0 <= a,b <= 1` and `a+b >= 1`.
They sample only the `a+b=1` boundary. The study must not claim to have searched
the entire admissible `(a,b)` region. These five settings are predeclared
experimental arms, **not** five values selected after looking at test scores.

## Shared retrieval rules

- Graph: the same prepared, undirected, unweighted Facebook graph in all graph
  conditions. Record input hashes and graph/node counts in each run manifest.
- Seeds: run the existing `llm_keywords` extractor **once per question** with a
  frozen local model and prompt, then apply the development-selected approximate
  title mapper (threshold `0.85`, ambiguity margin `0.10`). Pass the *same*
  resulting seed IDs to C1–C7. Record the extracted keywords, model metadata,
  match evidence, and unmatched keywords. Never provide gold seed IDs to a
  retrieval condition. The LLM-only answer arm does not receive the extracted
  keywords or graph metadata.
- Budget: at most **10 ranked graph nodes** in the context for each of C1–C7.
  Count seeds if a method returns them; do not inject extra seed nodes beyond the
  cap. C0 receives no graph nodes or graph context.
- Direct neighbours: take the union of all immediate neighbours of the mapped
  seeds, excluding the seed nodes themselves. Rank by descending number of seed
  nodes to which each candidate is adjacent, breaking ties by ascending stable
  node ID. Take the first 10. This deterministic rule is chosen before final
  evaluation and does not use relevance labels.
- AGP: use the persistent native backend with query type `S`, depth `L=2`, decay
  `0.30`, and `top_k=10` for **every** C2–C7 query. Seed mass is uniform. The
  adapter uses `w_i=(1-decay)decay^i` for `i=0..L`, default `delta=1/n`, and
  relative error `0.10`. These values remain identical across the six AGP arms.
  Because `S` is approximate and may be stochastic, preserve query-order and
  repeatability metadata; run a separate fixed-seed or repeated-query stability
  check before interpreting small differences. Do not silently choose the best
  random run for any condition.
- Context: use one formatter for all graph arms. Include the selected titles,
  descriptions, relationships among selected nodes, and seed-to-selected-node
  relationships. Seed titles may therefore occur in relationship lines even
  when seeds are not ranked evidence nodes. Keep node order as ranked. Do not
  give the answer model gold labels, score values, or condition names.
- Empty mapping or evidence: save an explicit empty-context result; never fall
  back to gold seeds or a different retriever.

This is a **node-count** budget, not a strict token budget: descriptions and
relationship counts may vary. Record prompt-token counts and context lengths,
and report them as a potential confound. A later token-capped sensitivity check
may be added if these lengths differ substantially.

## Adaptive `(a, b)` selector

C7 must choose exactly one of the five predeclared pairs from the question text
before retrieval. It may use a development-trained rule or an LLM, but must not
inspect test relevance labels, generated answers, or the five test rankings to
choose the best one. Record the selected pair, selector output, any LLM call,
latency, and errors for each question. Use one persistent native graph handle per
pair and reuse handles across questions. Selector design, prompt/model if used,
and fallback behaviour must be frozen on development data before final testing.

An oracle that chooses the best pair *after seeing labels* may be reported as an
upper bound, but it is not C7 and must never be included in the eight-condition
quality comparison.

## Answer generation and evaluation

Save all seven graph contexts **before** generating answers. For each question,
generate eight answers with the same model, model file/quantization, temperature
`0`, output limit, and substantive answer instruction. The only intended input
difference is whether and which graph context is supplied. Record input/output
tokens, answer latency, and cache status separately from retrieval latency.

Primary answer measures: blinded human correctness and completeness ratings
against independently prepared answer criteria. Reviewers also check factual
support for each graph-grounded answer against its *exact saved context* and
mark appropriate abstentions. Secondary measures: explicit-title entity
precision/recall/F1 and retrieval precision/recall/F1. The automatic
`context_faithfulness` measure checks named graph entities only; it does not
verify all factual claims and must not replace human support assessment.

Randomize all eight condition labels separately for each question. Keep the
answer key hidden until ratings are complete. At least two independent raters
should score a predeclared subset (ideally all questions); report agreement and
the handling of disagreements. Aggregate paired question-level differences,
including C7 versus each fixed AGP arm and C7 versus the best fixed arm chosen
on development data. Report 95% paired bootstrap intervals, per-question-type
breakdowns, and effect sizes. Treat multiple comparisons explicitly rather than
declaring a winner from raw mean scores alone.

## Data split and execution order

1. Develop and smoke-test retrieval, the selector, and the eight-answer runner
   using the existing **development** questions only.
2. Assemble a **new untouched evaluation set** with independent answer criteria.
   The previous 40-question held-out retrieval set has already been evaluated.
3. Freeze code revision, graph hashes, five pairs, selector, prompts/model,
   question set, review rubric, random seed, and analysis script in a dated
   manifest before generating final outputs.
4. Run each question through all eight conditions once at temperature zero.
   Preserve raw contexts, answers, model metadata, ratings, and analysis inputs.
5. Keep development findings, final findings, and any post-hoc analyses clearly
   separated in the report.

## Implementation checklist

- [x] Implement C1 deterministic direct-neighbour retriever and tests.
- [x] Build/cache the five native `(a,b)` handles behind one query pool; test
      shared AGP settings and handle reuse. C7's question selector is still open.
- [x] Implement and development-check a question-type C7 selector using the 20
      development questions. Leave-one-out F1 currently ties the best fixed arm;
      no adaptive benefit has been demonstrated. See the preserved selector
      artifact and report in `results/facebook_agp_pair_selector_dev_20260919/`.
- [x] Write one eight-condition retrieval-only context runner with an output
      manifest, input hashes, and refusal to overwrite prior results.
- [ ] Generalize the two-condition answer runner and blind-review form to eight.
- [ ] Prepare independently checked new evaluation questions and answer criteria.
- [ ] Conduct a development smoke run and resolve failures before freezing.
- [ ] Freeze, execute, rate, and analyze the new final evaluation once.
