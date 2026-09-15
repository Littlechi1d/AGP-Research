# Approximate entity-mapping development result

This tuning used only the labelled development questions. It did not inspect or
overwrite the frozen held-out results.

## Selection rule

Maximize micro precision first because a false seed sends graph propagation into
the wrong region; break ties by F1, recall, threshold, and ambiguity margin.

## Result

- Exact baseline: precision 1.000, recall 0.933, F1 0.966, exact sets 18/20.
- Selected approximate configuration: threshold `0.85`, margin `0.1`.
- Approximate mapping: precision 1.000, recall 1.000, F1 1.000, exact sets 20/20.

The compound recovery pass joins adjacent unresolved LLM keywords before
individual fuzzy matching. This recovers long titles that the LLM split into
several pieces.
