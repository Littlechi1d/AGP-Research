# Frozen cross-dataset experiment protocol

Frozen on 29 September 2026, before querying or scoring the GitHub MUSAE and
Deezer Europe evaluation splits.

The machine-readable source of truth is `experiment_protocol_v1.json`. Validate
it immediately before an evaluation run:

```bash
PYTHONPATH=. python3 scripts/validate_experiment_protocol.py
```

The validator checks the two graphs, two 40-question evaluation sets, native
AGP library, rule selector, keyword planner, LLM selector, and answer generator
against their frozen SHA-256 checksums. A mismatch stops the experiment.

## Conditions

- C0: LLM without graph context.
- C1: direct neighbours.
- C2–C6: native AGP with the five predeclared `(a,b)` pairs.
- C7: the unchanged rule selector developed on Facebook questions.
- C8: the unchanged zero-shot local-LLM selector developed before evaluation.

C7 and C8 select one fixed arm and reuse that arm's context and answer. This
avoids treating duplicate answer-generation calls as independent evidence.

## Frozen settings

- Native AGP: depth 2, decay 0.3, top-k 10, query type `S`, relative error 0.1.
- Mapping: similarity threshold 0.85 and ambiguity margin 0.10.
- Contexts: complete top-10 evidence with no character ceiling.
- Local model: `qwen3:4b-instruct-2507-q4_K_M`, temperature zero.
- Answer ceiling: 256 generated tokens.
- Random seed: 90055.

The model endpoint and API-key placeholder come from `.env`; the model name in
that environment must match the protocol. Caching may avoid repeated network
requests but must not change prompts or model outputs.

## One-time rule

Each evaluation split is run once. Results may be analysed and reported, but
no prompt, mapping rule, selector, parameter, question, label, or generation
setting may be changed in response and then presented as unseen evaluation.
Any later run must be labelled exploratory or a replication.
