# Invalid diagnostic pilot

Do not use the aggregate scores in this directory as semantic-review results.

The first blind run revealed an ambiguity in the judge rubric. The judge treated
the supplied reference entity set as incomplete context rather than as the
verified gold answer. It consequently rewarded C0 abstentions with scores of 5
and rejected correct graph-grounded answers for lacking additional relationship
evidence. This contradicts the intended rubric and makes the aggregate condition
comparison invalid.

The original files are retained as an audit trail. The rubric was clarified and
a separately named V2 pilot was run; the first output was not overwritten.
