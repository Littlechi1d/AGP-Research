# Question-only LLM selector development check — 29 September 2026

This is a development-only analysis of C8 on the GitHub MUSAE and Deezer
Europe graphs. The 40-question held-out evaluation split for each dataset was
not opened, queried, or scored.

## Method

C8 used the local `qwen3:4b-instruct-2507-q4_K_M` model to choose one of the
five fixed AGP arms from the natural-language question alone:

- C2: `(a,b) = (0,1)`
- C3: `(a,b) = (0.25,0.75)`
- C4: `(a,b) = (0.5,0.5)`
- C5: `(a,b) = (0.75,0.25)`
- C6: `(a,b) = (1,0)`

All 20 choices for a dataset were completed before the evaluator opened the
saved contexts or relevance labels. This prevents the selector from seeing
the retrieval result that it is meant to predict.

## Results

| Dataset | Best fixed arm | Best fixed F1 | C8 F1 | Oracle F1 | C8 selections | C8 vs best fixed W/T/L |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| GitHub MUSAE | C2 | **0.4611** | 0.4520 | 0.4847 | C3: 11, C4: 9 | 1 / 15 / 4 |
| Deezer Europe | C3 | **0.4295** | 0.4178 | 0.4612 | C4: 20 | 2 / 15 / 3 |

The oracle is an analysis-only upper bound that chooses the highest-scoring
fixed arm after seeing the relevance labels. It is not a valid deployable
selector.

## Interpretation

C8 did not beat the strongest fixed setting on either development set. On
GitHub it made two different choices but still lost to C2 on average. On
Deezer it selected C4 for every question, so it did not behave as an adaptive
selector. The gaps between C8 and the best fixed arm were approximately
0.0092 F1 on GitHub and 0.0117 F1 on Deezer.

This negative result is still useful: question wording alone may contain too
little information for a zero-shot LLM to infer how node degree should be
weighted on an unseen graph. The oracle gaps show that per-question choices
could help in principle, but they do not show that the current C8 can identify
those choices without labels.

The selector prompt and code should not now be tuned using the held-out
questions. For the final comparison, retain C8 unchanged as the predeclared
adaptive baseline and report this failure to outperform fixed AGP honestly.

Detailed decisions and reproducibility metadata are saved in:

- `results/github_llm_pair_selector_dev_20260929/`
- `results/deezer_llm_pair_selector_dev_20260929/`

The next step is to freeze the full evaluation configuration—including the
answer-generation token limit—and then run each untouched evaluation split
once across C0–C8.
