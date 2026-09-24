# V5 nine-condition exploratory run with LLM selector — 24 September 2026

**Status: exploratory automatic-metric result.** This extends the committed C0–C7 run with C8, a zero-shot question-only LLM selector. V5 had already been inspected, and no human ratings are used. The C8 selector was not modified after viewing V5 outcomes.

## C8 method

For each of the 40 questions, local `qwen3:4b-instruct-2507-q4_K_M` received only the question and the five allowed choices C2–C6. It did not receive graph data, relevance labels, fixed-arm rankings, saved answers, or automatic scores. All 40 choices were completed and recorded before the extension opened the answer file.

C8 then copied the chosen fixed arm's saved context and answer. This is exact reuse, not a new AGP query or answer-generation call, and ensures identical evidence has identical output. The original C7 rule selector remains unchanged.

Qwen selected:

- C4 `(0.5,0.5)` for 34 questions: all direct, comparison, and similarity questions, plus four path questions.
- C3 `(0.25,0.75)` for six path questions.

## Automatic results

| Arm | Method | Retrieval F1 | Explicit-title answer F1 |
| --- | --- | ---: | ---: |
| C0 | LLM only | — | 0.000 |
| C1 | Direct neighbours | 0.419 | 0.443 |
| C2 | Fixed AGP `(0,1)` | **0.453** | **0.603** |
| C3 | Fixed AGP `(.25,.75)` | 0.418 | 0.549 |
| C4 | Fixed AGP `(.5,.5)` | 0.382 | 0.524 |
| C5 | Fixed AGP `(.75,.25)` | 0.365 | 0.487 |
| C6 | Fixed AGP `(1,0)` | 0.354 | 0.454 |
| C7 | Rule selector; C2 for all V5 questions | 0.453 | 0.603 |
| C8 | LLM selector; 34 C4 and 6 C3 | 0.382 | 0.520 |

Against C2 question by question, C8's answer F1 was higher on 9 questions, tied on 13, and lower on 18; its mean difference was −0.0835. C8's answer F1 by question type was 0.923 direct, 0.364 comparison, 0.274 path, and 0.517 similarity. Nineteen C8 answers inherited a 256-token truncation from their selected fixed arms.

## Interpretation

C8 did make question-dependent choices, unlike C7 on V5, but it did not improve the automatic measures. Fixed C2 remained best for both retrieval and explicit-title answer F1. The LLM mostly selected C4, reproducing C4's behavior; its six C3 choices did not improve the overall average.

This supports only a narrow result: **this zero-shot LLM parameter selector did not outperform the best fixed pair on this exploratory automatic evaluation.** Explicit-title F1 is not a complete correctness or factual-support measure, C0's zero must not be interpreted as every LLM-only answer being wrong, and extensive truncation limits answer comparisons.

The corresponding nine-condition contexts, selector decisions, request metadata, and checksums are in `../facebook_nine_contexts_v5_exploratory_20260924/`. `summary.json` and `answers.jsonl` in this directory preserve C8's aggregate and per-question results.
