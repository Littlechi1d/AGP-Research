# Answer-quality development smoke test

This run validates the paired answer-generation and scoring machinery on the
first two development questions. It is not a final experiment.

## Frozen inputs for this smoke check

- Model: `qwen3:4b-instruct-2507-q4_K_M` through local Ollama
- Retrieval condition: saved `rules` contexts
- Questions: first 2 of the 20-question development split
- Blind-order seed: `90055`
- Temperature: `0`

## Automatic smoke result

| Condition | Entity precision | Entity recall | Entity F1 | Context faithfulness |
|---|---:|---:|---:|---:|
| LLM-only | 0.000 | 0.000 | 0.000 | N/A |
| AGP-grounded | 1.000 | 1.000 | 1.000 | 1.000 |

Both grounded answers mentioned all reference pages and only reference pages,
and all their graph-node mentions were supported by the saved contexts. The
LLM-only answers appropriately abstained because these obscure graph-specific
relationships were unavailable from the question alone.

The sample is intentionally tiny and selected from development data. These
numbers demonstrate correct operation, not a general quality advantage. Use the
blind review artifacts and `ANSWER_QUALITY_PROTOCOL.md` before drawing research
conclusions.
