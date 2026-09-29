# Frozen cross-dataset evaluation — 29 September 2026

This is the one-time held-out evaluation of the frozen C0–C8 protocol on the
GitHub MUSAE and Deezer Europe graphs. Each dataset contains 40 evaluation
questions: 10 each of direct-neighbour, comparison, connection-chain, and
similarity questions. These results must not be used for further tuning and
then presented as unseen evaluation.

## Reproducibility checks

- `experiment_protocol_v1.json` passed all 11 checksum checks immediately
  before the run.
- Both evaluation question files contained 40 questions and matched their
  frozen SHA-256 values.
- The local model was `qwen3:4b-instruct-2507-q4_K_M`, temperature zero.
- Native AGP used depth 2, decay 0.3, top-k 10, query type `S`, and relative
  error 0.1.
- Approximate title mapping used threshold 0.85 and ambiguity margin 0.10.
- Intended seed mapping was exact for 40/40 GitHub questions and 40/40 Deezer
  questions.
- Answers used a 256-token maximum and random seed 90055.

## Retrieval F1

| Condition | GitHub | Deezer |
| --- | ---: | ---: |
| C1 direct neighbours | **0.4394** | **0.4450** |
| C2 AGP `(0,1)` | 0.4129 | 0.4134 |
| C3 AGP `(.25,.75)` | 0.3664 | 0.3927 |
| C4 AGP `(.5,.5)` | 0.3530 | 0.3895 |
| C5 AGP `(.75,.25)` | 0.3348 | 0.3728 |
| C6 AGP `(1,0)` | 0.3132 | 0.3308 |
| C7 frozen rule selector | 0.4129 | 0.4134 |
| C8 frozen LLM selector | 0.3682 | 0.3895 |

C0 has no retrieved graph context, so retrieval F1 is not applicable. C1 had
the highest overall retrieval F1 on both datasets. Among fixed AGP settings,
C2 was strongest on both. C7 fell back to C2 for every new-dataset question,
so its result is identical to C2 rather than evidence of successful adaptation.

C8 selected C3 19 times and C4 21 times on GitHub. It selected C4 for all 40
Deezer questions. C8 did not outperform the best fixed AGP arm or the direct-
neighbour baseline on either dataset.

The aggregate result hides an important question-type trade-off. C1 scored
1.0 on direct-neighbour questions but 0 on similarity questions. On GitHub
similarity questions, F1 increased from C2's 0.3623 to C6's 0.4492. This shows
that propagation parameters affect question types differently, even though the
current adaptive selectors did not exploit the difference reliably.

## Conservative automatic answer metrics

The following metric searches generated answers for the topology-defined
reference entity titles. It is useful for exact entity coverage but is not a
semantic correctness judgement.

| Condition | GitHub entity F1 | Deezer entity F1 |
| --- | ---: | ---: |
| C0 LLM only | 0.0000 | 0.0000 |
| C1 direct neighbours | 0.4850 | 0.3824 |
| C2 AGP `(0,1)` | **0.5813** | 0.3615 |
| C3 AGP `(.25,.75)` | 0.5427 | **0.4080** |
| C4 AGP `(.5,.5)` | 0.4919 | 0.4055 |
| C5 AGP `(.75,.25)` | 0.4873 | 0.3859 |
| C6 AGP `(1,0)` | 0.4603 | 0.3410 |
| C7 frozen rule selector | **0.5813** | 0.3615 |
| C8 frozen LLM selector | 0.5224 | 0.4055 |

C0's zero score must not be interpreted as proof that its prose answers are
semantically worthless. The entities are graph usernames or anonymized user
IDs, so a model without graph context cannot know the benchmark-specific target
titles. This is exactly why the automatic metric is described as conservative.

Graph-backed automatic context-faithfulness values ranged from 0.9268 to 0.9506
on GitHub and from 0.9458 to 1.0000 on Deezer. These string-based values also do
not replace human or model-assisted semantic assessment.

## Generation diagnostics

| Dataset | Base answer calls | Network requests | Cache hits | Length-limited answers |
| --- | ---: | ---: | ---: | ---: |
| GitHub | 280 | 257 | 23 | 140 |
| Deezer | 280 | 240 | 40 | 166 |

C7 and C8 reused their selected fixed-arm answers, so neither required an
additional answer call. The high number of length-limited answers is a study
limitation: the 256-token ceiling was applied fairly across arms, but many
answers reached it.

## Conclusion

On these two held-out graphs, direct neighbours were the strongest retrieval
baseline overall, and C2 was the strongest fixed AGP setting. The frozen rule
selector did not generalize beyond its C2 fallback. The zero-shot LLM selector
also did not improve retrieval over the strongest fixed arm. Therefore, the
current results do not support the hypothesis that these question-only
selectors reliably choose better `(a,b)` values on unseen datasets.

The results do support a narrower observation: the best propagation behaviour
depends on question type, particularly for direct versus similarity questions.
A learned selector using training questions and graph features is reasonable
future work, but it must be evaluated on a new held-out split or dataset.

## Saved evidence

- `results/github_frozen_eval_contexts_20260929/`: original C0–C7 contexts.
- `results/github_frozen_eval_answers_c0_c7_20260929/`: original answers,
  blind-review files, request metadata, metrics, and checksums.
- `results/github_frozen_eval_contexts_c0_c8_20260929/` and
  `results/github_frozen_eval_answers_c0_c8_20260929/`: C8 extension.
- Corresponding `deezer_frozen_eval_*_20260929/` directories contain the
  Deezer evidence.
