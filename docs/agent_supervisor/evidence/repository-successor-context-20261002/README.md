# Successor context and exact native parent replay

The `historical-task-only-replay` directory preserves the original ten-test qualification, exact producer hashes, request and signed result. The historical producer and test source snapshots in this directory are byte-checked against those original hashes. Those runs predate the independent audit that identified missing native parent/head checks and are not relabeled as tests of the revised implementation.

The revised implementation also verifies native objective, goal and plan content/revisions; the unique active plan head; and the actual stored signed planning receipt. Partial existing parents cannot be overwritten. A bound-transaction helper lets another native owner join an exact materialization/replay with its own durable decision. New corruption controls alter bodies without revision changes, delete parents or the receipt, and insert a competing head.

Eleven distinct tests pass under the revised final bytes: nine native parent/missing/body/head/receipt controls (17.92 seconds), followed by two actual source-successor and separate-process replay cases (88.67 seconds, including fresh native fixture setup). `qualification.json` and `producer-sources.json` record this scope separately. The final request and both signed results are retained; serialized receipts remain historical evidence, never live proof capabilities. Historical admission/resource refusals remain visible; no scheduler threshold was weakened. This API preserves the entire task population and grants no proof, execution, completion or omission authority.

See [the API guide](../../repository_successor_context.md).
