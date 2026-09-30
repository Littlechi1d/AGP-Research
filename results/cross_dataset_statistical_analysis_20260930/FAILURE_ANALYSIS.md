# Qualitative failure analysis

This analysis uses only the frozen GitHub MUSAE and Deezer Europe evaluation
outputs. No question was rerun and no parameter, prompt, label, or result was
changed after inspection.

## Failure taxonomy

| Stage | Observed result | Interpretation |
| --- | --- | --- |
| Keyword extraction and mapping | 80/80 questions mapped to exactly the intended seed IDs | Mapping was not the limiting factor in this evaluation. |
| C7 parameter selection | C7 selected C2 for all 80 questions | The Facebook-developed wording rules did not generalize; C7 was a fixed fallback in practice. |
| C8 parameter selection | GitHub: C3 19 times and C4 21 times; Deezer: C4 40 times | The zero-shot LLM used little of the available parameter range and could not infer graph-specific behaviour from wording alone. |
| Retrieval budget | Every graph arm was limited to 10 ranked nodes | Seed nodes and irrelevant high-score nodes sometimes displaced relevant evidence, especially for multi-seed and large-answer questions. |
| Context construction | Relationships were included only among displayed ranked nodes and seeds | Missing ranked nodes also removed relationships needed to prove intersections or paths. |
| Answer generation | 140/280 GitHub and 166/280 Deezer base answers ended at the 256-token limit | Many answers spent the budget restating the task or showing steps and were cut off before a concise final list. |
| Automatic answer scoring | Exact graph-title matching only | It measures explicit entity coverage, not semantic correctness, reasoning quality, or paraphrases. |

## Representative cases

### 1. Direct neighbours: propagation adds noise without adding evidence

**Question:** `github_eval_direct_04` — “Which developers have a direct
mutual-follow connection with alliejones”

The gold answer contains `thomasboyt`, `nfultz`, and `leonstafford`. C1 retrieved
exactly those three nodes and scored 1.000 retrieval F1. C2 retrieved all three,
but also the seed and six unrelated high-ranked nodes, reducing F1 to 0.462.
C8 chose C4, which also scored 0.462.

All three contexts still allowed the answer model to name the correct developers.
This separates retrieval ranking quality from final answer quality: extra nodes
hurt retrieval precision even when the LLM successfully filters them. It also
explains much of C1's overall advantage on the balanced benchmark, where direct
questions comprise one quarter of the test set.

### 2. Comparison: a wrong C8 choice removes the intersection

**Question:** `github_eval_comparison_06` — common neighbours of `ankitshah009`
and `horse-latitudes`.

The four gold developers are `JonnyBanana`, `nfultz`, `dalinhuang99`, and
`rfthusn`. C1 and C2 retrieved all four among ten nodes and each scored 0.571.
C8 chose C4; its ranking contained none of the four gold developers and scored
0.000. Its answer therefore concluded that no common developer was present.

The wording alone did not tell C8 that the low-`a`, high-`b` C2 setting would
preserve the useful common-neighbour evidence on this graph. This is a concrete
parameter-selection failure rather than a mapping failure.

### 3. Similarity: propagation is necessary and a different pair can help

**Question:** `github_eval_similarity_05` — same-class developers reachable
from `Mhaiyang` through one intermediate developer, excluding direct neighbours.

C1 returned only the two direct neighbours and scored 0.000. C2 scored 0.526,
while C4–C6 each scored 0.737. C8 chose C4 and therefore matched the strongest
fixed result for this example.

This case demonstrates the reason to retain AGP: direct neighbours cannot answer
a two-step similarity query. It also supports the narrower claim that useful
parameters differ by question type. However, isolated correct C8 choices did not
produce an aggregate advantage.

### 4. Path: top-10 evidence and the answer limit fail together

**Question:** `deezer_eval_path_02` — shortest connection from Deezer user 27469
to Deezer user 165.

The two internal users are `Deezer user 12452` and `Deezer user 18280`. C2
retrieved both but placed them among ten nodes, producing F1 0.333. C8 chose C4,
whose ranking contained neither target and scored 0.000. C1 happened to retrieve
both internal users but did not provide a sufficiently focused ordered path.

Every inspected base answer for this example ended because of the 256-token
limit. The model repeatedly enumerated adjacency lists and was cut off before a
short final answer. This is both a retrieval/context problem and a generation-
prompt problem; increasing only one budget would not guarantee success.

### 5. Large similarity answer: a fixed node budget limits recall

**Question:** `deezer_eval_similarity_10` — same-class two-step users for Deezer
user 13333.

There are ten gold users. C2 scored 0.500, C8 chose C4 and scored 0.300, and C1
scored 0.000. Because each arm may display at most ten ranked nodes and AGP also
ranks the seed and direct neighbours, no arm can freely devote all ten positions
to the ten target users. The answer model then used much of its token budget to
explain filtering and was truncated.

This exposes a benchmark-design interaction: a uniform top-10 budget is fair
across arms, but it creates a hard ceiling for questions with ten relevant nodes.

## Cross-cutting conclusions

1. **The mapping improvement worked.** There is no evidence that approximate
   matching caused the final retrieval differences.
2. **C1's overall lead is task-dependent.** It dominates direct questions and
   fails similarity questions completely.
3. **AGP remains useful for multi-hop retrieval.** Its weakness is selecting and
   budgeting the right evidence, not an inability to reach multi-hop nodes.
4. **The tested selectors did not adapt reliably.** C7 became C2; C8 used only
   C3/C4 and sometimes chose an arm that removed all gold evidence.
5. **Retrieval F1 and answer quality are related but not identical.** The model
   can filter noisy contexts, while a high-recall context can still yield an
   incomplete answer when generation is truncated.
6. **The negative adaptive result should be reported directly.** A learned
   selector using graph features and training questions is reasonable future
   work, but testing it fairly requires a new untouched evaluation split.

## Recommended dissertation use

Use the direct-neighbour and similarity examples as a contrasting pair in the
Results section. Use the comparison and path examples in the Discussion section
to distinguish parameter-selection, context-budget, and generation failures.
Report the Deezer large-answer example as a threat to validity of the uniform
top-10 design rather than silently treating it as a model error.
