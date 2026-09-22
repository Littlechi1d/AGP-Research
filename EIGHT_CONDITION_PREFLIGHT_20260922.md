# Eight-condition study: preflight record (22 September 2026)

**Status: not frozen; do not run the final eight-arm evaluation yet.** This is
a preflight record, not a claim of independent human review or a result from
the new 40-question candidate set.

## Verified inputs and settings

- Current candidate: `data/facebook_large/eight_condition_eval_candidate_v5_20260922/questions.json`;
  40 questions, 10 of each type; SHA-256
  `c4fbf01fc2feb5fbd46a6cff9f84c146fa6d8517bc9d131d6fedce4a4d1cdb35`.
  Its labels passed an independent *computational* graph/category audit (40/40),
  including path order. It has no seed overlap with the earlier development or
  used test sets. The user's approval covers question wording; the blank
  `criteria_review.csv` does not document independent human label approval.
- Graph: `data/facebook_large/nodes.csv` SHA-256
  `a5a89bb0e0bf77eaaff9059a784da0cd5b930ba5e3d9018f80f6156c9af28daa`;
  `edges.csv` SHA-256
  `f3c3aeec27afa4389cd5042fabd21585804cf41506dce79df7ee0c74a81f2fb0`.
- Native AGP library: `build/paper_backend/libagp_api.dylib` SHA-256
  `8b881f6089cfc88632ee79c11c7380f51e66b7f3892ddbb082f9c67a832f3974`.
  The six AGP conditions use the five predeclared `(a,b)` pairs, depth 2,
  decay 0.30, top-k 10, query type `S`, and relative error 0.10.
- C7 selector: `results/facebook_agp_pair_selector_dev_20260919/selector.json`
  SHA-256 `eb86f704344bf4e6fd85fb5e1c900ac838900c16f4b2693c59f4b7e4a19041eb`.
  It selects C2 for direct/comparison/path and C3 for similarity questions.
  Development leave-one-out retrieval F1 tied C2; adaptation has not yet shown
  an advantage.
- Answer model currently installed: local
  `qwen3:4b-instruct-2507-q4_K_M`, Ollama digest
  `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`
  (read from the local model API on this date). Temperature 0; maximum 256
  generated tokens per answer. The answer-system prompt SHA-256 is
  `beaf35db2a34d1a4b93407d5cb69020e285e57dbc7f418234a2de80c6c22fb34`.
  Verify the model digest again immediately before the final run.
- Proposed primary context policy: up to 10 ranked entities, with no character
  ceiling. Report the unequal prompt-token counts. Predeclare the 3,000-character
  ceiling as a **sensitivity analysis**, not a token-equalized primary condition.
  C7 reuses its selected fixed arm's context and generated answer.
- Randomized blind labels use seed 90055. The answer key remains hidden from
  raters until their ratings are complete.

## Preflight issue found; selector deliberately unchanged

The original question-type recognizer matched the old technical phrases only.
On V5 it falls back to C2 for all 10 path and 10 similarity questions. The
other 20 questions also select C2 through their recognized types. Therefore
**C7 would be identical to fixed C2 for all 40 V5 questions**; it could not
test whether question-adaptive `(a,b)` selection improves results. The user
correctly objected to changing the selector after inspecting proposed test
question wording. The attempted uncommitted recognizer change was reverted.
The saved selector and its wording rules remain as developed before V5.

This is a design limitation, not an observed retrieval or answer outcome. No
V5 relevance scores or generated answers were used. If V5 is used unchanged,
the final report must disclose C7=C2 by construction and must **not** claim a
meaningful adaptive-versus-fixed comparison. No new evaluation set is being
generated at the user's request.

## Gates before a dated final freeze

1. Commit the V5 candidate, rephrasing code, tests, and this corrected
   record. Record the resulting clean Git revision; the current HEAD predates
   these changes and must not be cited as the frozen implementation.
2. Have an independent reviewer verify the expected titles and graph criteria
   in a **copy** of V5 `criteria_review.csv`. The original blank form remains
   unchanged. Resolve any corrections as a new candidate version; do not edit
   the hashed V5 artifact in place.
3. Decide whether a study in which C7 duplicates C2 on every V5 question is
   acceptable. If not, the current V5 design cannot answer the adaptive-pair
   research question without changing the evaluation design. Do not silently
   relabel C7 as adaptive evidence.
4. Rehearse keyword extraction and answer generation on development questions
   only. Do not use V5 to choose prompts, pairs, thresholds, or context budget.
5. Finalize the blind-rating rubric, number of raters, the primary paired
   comparison, confidence-interval method, handling of truncated answers, and
   the analysis code *before* opening final answer outputs.
6. Confirm the primary-versus-sensitivity context policy and recheck hashes,
   local model digest, code cleanliness, and native AGP availability. Then
   write a separate dated **frozen** manifest. Only after that manifest is
   committed should the one-time V5 retrieval and answer runs begin.

The previous 40-question retrieval test set has already been evaluated; it is
not an untouched substitute for V5. No V5 AGP retrieval, keyword extraction,
or answer-quality scoring was performed in this preflight.
