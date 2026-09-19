# Development-only character-ceiling check

This run used all 20 development questions, saved LLM keywords, the existing
question-type selector, and the native five-pair AGP backend. It is retrieval
only: no answer-model calls, human ratings, or held-out questions were used.

The optional ceiling was 3,000 **characters per graph context**, retaining
whole ranked-entity lines first and then whole eligible relationship lines.
The C0 LLM-only arm still has an empty context. The run generated 20 rows and
all eight arms; no saved context exceeded 3,000 characters.

| Arm | Mean context characters | Max | Contexts shortened |
|---|---:|---:|---:|
| C1 direct neighbours | 2,243.8 | 3,000 | 11/20 |
| C2 fixed (0,1) | 2,910.4 | 3,000 | 18/20 |
| C3 fixed (.25,.75) | 2,904.2 | 3,000 | 16/20 |
| C4 fixed (.5,.5) | 2,873.7 | 3,000 | 15/20 |
| C5 fixed (.75,.25) | 2,861.1 | 3,000 | 15/20 |
| C6 fixed (1,0) | 2,879.8 | 3,000 | 16/20 |
| C7 adaptive | 2,907.1 | 3,000 | 18/20 |

The shortened flag includes omitted relationships as well as entities. All
arms retained their ranked entities in nearly every question; C2–C7 displayed
9.8 entities on average, and C1 displayed 7.2. Because graph relationships
are often omitted, this cap may remove evidence needed to verify answers.
The context ceilings are unequal in actual model tokens, and C1 is still much
shorter on average than AGP arms. Therefore this is a sensitivity condition,
not proof of a fair equal-token comparison or answer-quality improvement.

To reproduce in a fresh output directory:

```bash
python3 scripts/run_eight_condition_contexts.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --questions data/facebook_large/facebook_questions_dev.json \
  --selector results/facebook_agp_pair_selector_dev_20260919/selector.json \
  --keyword-results results/facebook_llm_keywords_fixed_dev_20260909/results.jsonl \
  --output results/facebook_eight_contexts_char3000_REPRODUCED \
  --max-context-chars 3000
```

An initial diagnostic output in the sibling directory without `_final_` used
an incomplete truncation flag that counted only omitted entities. Use this
directory for any analysis of whether relationship text was shortened.
