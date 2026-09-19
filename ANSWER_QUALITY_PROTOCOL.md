# Paired answer-quality evaluation protocol

This document describes the implemented **two-condition** smoke-test runner.
The proposed final eight-condition study and its additional controls are specified
in [EIGHT_CONDITION_EXPERIMENT.md](EIGHT_CONDITION_EXPERIMENT.md).

## Research comparison

For every question, use the same model, temperature, and question wording to
generate two answers:

- **LLM-only:** the model receives the question but no graph evidence;
- **AGP-grounded:** the model receives the question and a previously saved AGP
  retrieval context.

Using saved contexts is important: it freezes retrieval and isolates the effect
of adding graph evidence. Each question produces a pair, so analysis must use
paired rather than independent statistical tests.

## Measurements

The runner calculates conservative automatic metrics from explicit graph-title
mentions:

- entity precision, recall, and F1 against the question's relevant node IDs;
- grounded-answer context faithfulness: the proportion of mentioned graph nodes
  that also occur in the supplied context.

Seed entities repeated from the question are excluded, and overlapping titles
use the longest match. These automatic metrics do not recognize paraphrases and
therefore must not be the only evidence of answer quality.

The runner also creates a randomized blind review file and an empty ratings CSV.
Reviewers see Answer A and Answer B without knowing the condition. They score
correctness and relevance from 1 (worst) to 5 (best), choose a preferred answer,
and may add notes. At least two reviewers should independently rate a useful
subset. Report agreement and resolve neither disagreements nor labels by looking
at the answer key before ratings are finalized.

Faithfulness needs a separate evidence-aware check of the AGP-grounded answer
against its exact saved context. Completeness and appropriate abstention should
also be manually audited. An LLM judge may supplement, but not replace, blind
human ratings; its model, prompt, order randomization, and context must be frozen.

## Development smoke command

```bash
cd /Users/feiyuzhang/Desktop/COMP90055/agp_research

python3 scripts/run_answer_quality_study.py \
  --nodes data/facebook_large/nodes.csv \
  --edges data/facebook_large/edges.csv \
  --questions data/facebook_large/facebook_questions_dev.json \
  --retrieval-results results/facebook_fixed_tuning_20260903/reference_rules.jsonl \
  --retrieval-strategy rules \
  --output results/answer_quality_dev_REPRODUCED \
  --limit 2
```

Remove `--limit 2` only after freezing the protocol. The command refuses to
overwrite an existing output directory.

## Outputs

- `answers.jsonl`: questions, references, contexts, both answers, metrics, and
  per-call token/latency metadata;
- `summary.json`: macro automatic metrics, model identity, settings, and hashes;
- `blind_review.jsonl`: randomized A/B answers and reference titles;
- `human_ratings.csv`: blank form for reviewer scores;
- `answer_key.json`: mapping from A/B to experimental condition; keep this hidden
  from reviewers until ratings are complete.

## Final-study controls

Before a final run, freeze the question set, saved retrieval condition, model
identifier, model file/quantization, prompts, random seed, scoring rules, human
rubric, number of reviewers, and statistical tests. Compare paired per-question
scores with bootstrap confidence intervals and a paired permutation or Wilcoxon
test. Report latency and token counts alongside quality.

The two-question result in `results/answer_quality_smoke_dev_20260915/` validates
the machinery only. It is too small and uses development data, so it is not a
research conclusion.
