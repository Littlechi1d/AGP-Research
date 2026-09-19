# Revised eight-condition evaluation candidate: review required

This is the current **unscored candidate set**, not a frozen or evaluated test
set. It contains 40 questions: 10 direct-neighbour, 10 comparison, 10 path, and
10 similarity. Labels come from graph topology and source page categories, not
AGP results or LLM answers.

The only change from the first candidate set is the wording of the 10
similarity questions. Instead of “two hops away,” each asks for pages of the
same type that can be reached through one other page but are not directly
connected. This keeps the exact graph criterion—shortest-path distance two and
same category—while avoiding technical graph terminology. IDs, seed pages,
provisional label IDs, and generation rules are identical across the two
versions. The old candidate directory is superseded and must not be used for
evaluation.

The generator excluded the prior 20 development and 40 used test questions.
Against their combined 90 unique seeds, this candidate has zero seed overlap;
its 60 seed pages and all normalized question texts are new. `manifest.json`
records source hashes and these structural checks. `questions.json` contains
provisional reference IDs. `criteria_review.csv` provides titles and blank
review fields; it has not been filled or signed off.

An independent reviewer should check each question's wording, exact seed
identity, complete answer titles, and whether “same type” accurately expresses
the dataset category. Fill a **copy** of `criteria_review.csv` and preserve
corrections as a new version with an audit trail. Human review is still required
before freezing this set or running the one-time eight-arm evaluation.
