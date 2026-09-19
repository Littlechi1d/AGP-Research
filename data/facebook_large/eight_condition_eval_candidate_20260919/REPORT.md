# New eight-condition evaluation candidate: review required

**Archived V1 — superseded. Do not use for the eight-arm evaluation.** The
current review-pending set is
`data/facebook_large/eight_condition_eval_candidate_v4_20260919/`. This first
candidate is preserved only to show the question-revision history. In
particular, later versions improved similarity/path wording, ordered path
answers, and replaced five questions with artificial `[page ID]` suffixes in
their expected titles.

This is an **unscored candidate set**, not a frozen or evaluated test set. It
contains 40 questions (10 direct-neighbour, 10 comparison, 10 three-hop path,
and 10 similarity). Labels were generated from the original graph topology and
page categories before any AGP retrieval or LLM answer generation.

The generator excluded the 20 prior development and 40 previously used test
questions. It verified zero seed-page overlap against their combined 90 unique
seeds and zero normalized exact-question overlap. The candidate uses 60 unique
seed pages. The manifest hashes graph inputs, source categories, both excluded
question files, the new questions, and the review form.

`questions.json` contains machine-readable questions and provisional reference
node IDs. `criteria_review.csv` lists the expected answer titles and a blank
independent-review section. A reviewer should verify for **each** question that:

1. Its wording unambiguously asks for the generated graph relation.
2. The seed titles identify the intended pages and not title duplicates.
3. The expected titles are complete and correct against the source graph.
4. Any category-based similarity criterion is interpretable to an answer
   reviewer, given the graph's page-category descriptions.

The reviewer should enter their ID, yes/no decisions, and notes in a **copy**
of the review form. Do not overwrite the original blank form or questions;
preserve any corrections as a new version with an audit trail. This automated
generation is not a substitute for independent human adjudication. It also
uses the same four templates as the earlier benchmark, so seed disjointness
does not imply independence of question style.

No AGP retrieval, keyword extraction, answer generation, or quality scoring
has been run on these 40 questions. The next step is review and a dated freeze
of the accepted questions, criteria, primary/sensitivity context budgets,
model digest, prompts, code revision, and analysis plan. Only then should the
one-time eight-arm evaluation begin.
