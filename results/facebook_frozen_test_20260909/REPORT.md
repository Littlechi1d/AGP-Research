# Frozen held-out evaluation

## Protocol

This is the one-time held-out retrieval evaluation declared in
`FROZEN_EXPERIMENT_PROTOCOL.md`. It ran from clean, pushed commit `4dbe755` after
44 tests and a full development rehearsal passed. The installed local Qwen model
digest matched the frozen digest immediately before execution. No strategy,
prompt, parameter, label, or metric was changed after test results became visible.

The run evaluated 40 held-out questions across eight conditions, producing 320
question-condition records. Answer generation was disabled. The primary metric
is mean per-question F1; other metrics and type summaries are secondary.

## Overall held-out results

| Rank | Condition | Precision | Recall | Mean F1 | Hit rate | MRR |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Rules | 0.4534 | 0.9211 | **0.5625** | 1.0000 | 0.3224 |
| 2 | Zero-shot LLM parameters | 0.4507 | 0.8836 | 0.5557 | 1.0000 | 0.3197 |
| 3 | Few-shot LLM parameters | 0.4388 | **0.9586** | 0.5404 | 1.0000 | **0.3340** |
| 4 | Fixed, k=10 | 0.3906 | 0.9211 | 0.5147 | 1.0000 | **0.3340** |
| 5 | End-to-end LLM | 0.4021 | 0.8336 | 0.5020 | 0.9250 | 0.3096 |
| 6 | LLM keywords + fixed k=5 | 0.4350 | 0.5999 | 0.4726 | 0.9000 | 0.3275 |
| 7 | Selected fixed, k=5 | 0.4300 | 0.5974 | 0.4692 | 0.9000 | 0.3192 |
| 8 | Seed-only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Rules has the highest held-out mean F1. Zero-shot LLM parameter selection is
close but does not exceed it. Few-shot parameter selection has the highest recall
but its larger path-query budget lowers precision and mean F1.

## Paired uncertainty versus rules

Intervals are deterministic percentile intervals from 10,000 paired bootstrap
resamples of the 40 per-question F1 differences, using seed 90055.

| Condition minus rules | Mean F1 difference | 95% bootstrap interval |
|---|---:|---:|
| Zero-shot LLM parameters | -0.0068 | [-0.0267, 0.0116] |
| Few-shot LLM parameters | -0.0221 | [-0.0418, -0.0041] |
| Fixed, k=10 | -0.0478 | [-0.0823, -0.0178] |
| End-to-end LLM | -0.0605 | [-0.1344, -0.0034] |
| LLM keywords + fixed k=5 | -0.0899 | [-0.1705, -0.0095] |
| Selected fixed, k=5 | -0.0933 | [-0.1731, -0.0113] |
| Seed-only | -0.5625 | [-0.6399, -0.4861] |

Only the zero-shot LLM-parameters interval includes zero. On this benchmark,
there is therefore no clear paired mean-F1 difference between that condition and
rules. This is not evidence that the methods are equivalent, nor that either will
generalize beyond the templated Facebook benchmark.

## Results by question type

| Condition | Direct F1 | Comparison F1 | Path F1 | Similarity F1 |
|---|---:|---:|---:|---:|
| Rules | **0.9037** | 0.4589 | 0.2833 | **0.6041** |
| Zero-shot LLM parameters | **0.9037** | 0.4720 | 0.2430 | **0.6041** |
| Few-shot LLM parameters | **0.9037** | 0.4720 | 0.1818 | **0.6041** |
| Fixed, k=10 | 0.7123 | 0.4589 | 0.2833 | **0.6041** |
| End-to-end LLM | 0.7214 | 0.4589 | 0.2697 | 0.5579 |
| LLM keywords + fixed k=5 | 0.7271 | **0.6486** | **0.2857** | 0.2289 |
| Selected fixed, k=5 | 0.7271 | 0.6352 | **0.2857** | 0.2289 |

Rules and both parameter-only LLM conditions perform strongly on direct and
similarity questions. All methods remain weak on three-edge path intermediates.
The few-shot strategy retrieves every relevant path node (recall 1.0), but its
fixed top-k 20 produces path F1 0.1818. This illustrates why recall alone cannot
be the selection criterion.

## LLM behavior

Zero-shot parameter selection again chose top-k 10 for every question and never
used depth 3. The few-shot strategy copied its four demonstrations exactly for
all ten questions of each type, including top-k 20 for every path question.

The keyword-only condition matched every intended seed on 38 of 40 questions and
never produced a completely seedless query. Its two failures split compound page
titles. The end-to-end condition matched all intended seeds on 36 questions and
produced three seedless queries. It also emitted unmatched non-entity phrases on
25 questions. Joint planning therefore did not repair keyword extraction.

## Runtime and model use

The four LLM conditions made 160 local network requests with zero cache hits.
The provider reported 24,508 prompt tokens, 4,577 completion tokens, and 29,085
total tokens. No cloud API was called. Mean pipeline latency ranged from about
0.85 seconds for keyword-only extraction to 2.06 seconds for joint planning.

## Conclusions

1. Transparent rules are the strongest condition by the predeclared held-out
   primary metric.
2. The predeclared paired bootstrap analysis detects no clear mean-F1 difference
   between zero-shot LLM parameter selection and rules, while the LLM adds model
   cost and latency without a measured F1 benefit.
3. Few-shot prompting reliably controls Qwen and maximizes recall, but behaves
   like a template lookup and sacrifices precision.
4. LLM keyword extraction is the main failure source in the end-to-end pipeline;
   exact graph-aware title matching is substantially more reliable here.
5. Path retrieval remains the clearest technical weakness across strategies.

These conclusions apply to the controlled topology benchmark. The next research
phase should use independently authored semantic questions and human relevance
or answer-quality labels. Any subsequent system improvement must be reported as
post-test work and must not replace this frozen result.
