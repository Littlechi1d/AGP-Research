# V5 eight-arm exploratory run — 22 September 2026

**Status: exploratory, not a frozen held-out or human-rated answer-quality result.** The V5 wording was inspected before this run, and the independent human criteria review described in the preflight record remains incomplete. No code, question, parameter, or prompt was changed during this run. Git revision: `3236da9`.

## What ran

- 40 V5 questions, 10 per question type; eight conditions C0–C7.
- C0: local Qwen without graph context; C1: direct neighbours; C2–C6: five fixed native AGP `(a,b)` pairs; C7: the unchanged development-trained question-type rule selector.
- One live keyword-extraction call per question, approximate title matching, and native AGP query type `S` with depth 2, decay 0.30, top-k 10, and relative error 0.10. Contexts were saved before answer generation.
- Local `qwen3:4b-instruct-2507-q4_K_M`, temperature 0, maximum 256 generated tokens, randomized generation order, blind answer labels using seed 90055. C7 copied its selected fixed-arm answer rather than making another call.
- 280 logical answer calls (seven per question): 242 network requests and 38 cache hits.

## Automatic screening measures

| Arm | Retrieval F1 | Explicit-title answer F1 | Mean context characters | Answers cut off at 256 tokens |
| --- | ---: | ---: | ---: | ---: |
| C0 LLM-only | — | 0.000 | 0 | 0 |
| C1 neighbours | 0.419 | 0.443 | 2,919 | 14 |
| C2 AGP `(0,1)` | 0.453 | 0.603 | 3,710 | 18 |
| C3 AGP `(.25,.75)` | 0.418 | 0.549 | 3,380 | 20 |
| C4 AGP `(.5,.5)` | 0.382 | 0.524 | 3,252 | 17 |
| C5 AGP `(.75,.25)` | 0.365 | 0.487 | 3,215 | 17 |
| C6 AGP `(1,0)` | 0.354 | 0.454 | 3,168 | 17 |
| C7 unchanged rule | 0.453 | 0.603 | 3,710 | copied from C2 |

These are descriptive means, not significance-tested effects. C2's answer F1 exceeded C1's on 19 questions, tied on 13, and was lower on 8. Retrieval F1 here is the harmonic mean of precision and recall of ranked node IDs against the candidate graph labels. Explicit-title answer F1 counts named reference entities; it is **not** a general correctness or factual-support score. In particular, C0 produced nonempty answers to all 40 questions, but its zero means the model did not name the exact graph-page titles required by this metric; it does not show that every LLM-only answer was wrong. The model reached the output ceiling in **103 of 280 generated answers**, making completeness comparisons especially fragile. Graph-context lengths also differ substantially across arms, and there was no equal-token context budget.

The unchanged selector chose C2 on all 40 V5 questions. C7's saved context and answer exactly equal C2's for every question, so **this run provides no adaptive-versus-fixed evidence**. Five questions had at least one unmatched extracted keyword, but none had an empty mapped-seed set. The automatic named-entity `context_faithfulness` measure does not verify all factual claims.

## Files and interpretation

- Retrieval inputs and per-question contexts: `../facebook_eight_contexts_v5_exploratory_20260922/contexts.jsonl` and its `manifest.json`.
- Generated answers, per-question automatic metrics, token and latency logs: `answers.jsonl`; aggregate results and input hashes: `summary.json`.
- Blinded reviewer materials were generated but will not be scored in this study: `blind_review.jsonl`, `faithfulness_review.jsonl`, blank `human_ratings.csv`, and blank `faithfulness_ratings.csv`. `answer_key.json` is retained for reproducibility; there is no active blinded rating exercise.

The project owner chose to skip human ratings and report automatic measures only. Accordingly, this run supports the narrow observation that C2 has the highest explicit-title F1 among these arms on V5. It does **not** establish better general answer correctness, completeness, or factual support. A genuinely confirmatory adaptive-pair claim would also require a selector that varies on the evaluation wording and an untouched evaluation design; the current V5 run cannot supply that claim.
