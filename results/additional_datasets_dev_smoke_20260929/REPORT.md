# GitHub and Deezer development smoke test — 29 September 2026

This is a development-only compatibility check. No evaluation questions were
opened or scored.

## Configuration

- 20 development questions per dataset: five each of direct, comparison,
  connection-chain, and similarity questions.
- Local keyword model: `qwen3:4b-instruct-2507-q4_K_M` at temperature zero.
- Approximate title mapping: threshold 0.85 and ambiguity margin 0.10.
- Native AGP-Static++: query type `S`, depth 2, decay 0.30, top-k 10,
  relative error 0.10, and the five fixed C2–C6 `(a,b)` pairs.
- C7 used the existing Facebook-developed rule selector unchanged. The new
  wording is not recognized by that selector, so it fell back to C2 for all 40
  development questions. C7 is therefore not an adaptive result here.

## Keyword-mapping check

| Dataset | Exact seed sets | Partial | Empty |
| --- | ---: | ---: | ---: |
| GitHub MUSAE | 20/20 | 0 | 0 |
| Deezer Europe | 20/20 | 0 | 0 |

The initial prompt split some anonymized Deezer titles or copied their digits
incorrectly. Development-only prompt clarification improved mapping, but one
digit-copy error remained. The final implementation therefore preserves every
explicit `Deezer user <number>` mention deterministically after the LLM call.
This rule was fixed using development questions before any evaluation run.

## Development retrieval F1

| Arm | GitHub | Deezer |
| --- | ---: | ---: |
| C1 direct neighbours | 0.4389 | **0.4350** |
| C2 AGP `(0,1)` | **0.4611** | 0.4242 |
| C3 AGP `(.25,.75)` | 0.4538 | 0.4295 |
| C4 AGP `(.5,.5)` | 0.4300 | 0.4178 |
| C5 AGP `(.75,.25)` | 0.3952 | 0.3741 |
| C6 AGP `(1,0)` | 0.3259 | 0.3581 |
| C7 rule selector | 0.4611 | 0.4242 |

These descriptive development numbers confirm end-to-end operation; they must
not be presented as evaluation results. They do not justify dataset-specific
retuning if the research question is whether the previously frozen settings
generalize across graphs.

Final saved contexts and request metadata are in:

- `results/github_eight_contexts_dev_smoke_v3_20260929/`
- `results/deezer_eight_contexts_dev_smoke_v3_20260929/`

The next step is to freeze the cross-dataset procedure, add the C8 LLM selector
to both saved development runs, and decide whether answer generation will use
the existing 256-token limit or a larger predeclared limit before evaluating
the untouched 40-question splits.
