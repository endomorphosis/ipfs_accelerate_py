# Source request attribution qualification

The live compact-worker comparison retained a refused request without its
operation. This diagnostic change adds closed operation, source-state and view
labels to owner audit `supervisor-worker-source-session@5`, without adding raw
queries or source bodies. Parsed requests are classified before peer/context
admission; malformed requests remain unknown. A requested operation does not
establish authorization, a successful action or a proof.

The collector validates labels against retained request hashes and successful
page kinds while retaining strict response accounting. Historical version 1–4
records preserve their previous output shape. Mixed histories expose attribution
coverage instead of assigning invented labels to older requests.

The [qualification](qualification.json) records **627 distinct passing tests**
across 14 files, including actual sockets, refused and successful mapped edits,
malformed requests, historical and mixed collectors, archive checks and the
router-to-worker integration. The [ledger](tests-ledger.json) selects the latest
passing result for each file: 511 tests from `focused-02` and 116 from
`capability-03`. Counts from superseded runs are not added. Earlier fixture and
archive-allowlist expectation failures are [retained](retained-failures.json),
with their JUnit records, rather than omitted.

The [independent review](independent-review.json) found no blocker. Ten
[interface source files](unchanged-interfaces.json) remain byte-identical,
including client protocols, guidance, mapping and router prompt construction.
This does not constitute a new cross-UID Docker or full-supervisor qualification.
No model calls were made by these tests.

Tests ran in an isolated worktree at implementation commit
`69bc0b87bba2b27e89d246fdff3caa7ed3330c83`, based on
`3f08cc43cbd96c9f8a5114091f65566a5962dbbc`. The change was cherry-picked as
`e60dc768cf` only after all three live benchmark readouts and their source-pin
checks had finished. Every qualified source and test hash was then checked
against the destination worktree. The
[live evidence](../compact-worker-live-20261014/README.md) predates this change;
its refused requests cannot be classified retrospectively.

The manifest binds this README, qualification, reviews, test ledgers and the
retained command, source-hash, exit and JUnit records. Private databases and the
test environment are excluded. Re-running requires the matching repository and
its test dependencies; commands and selected file hashes are retained for that
purpose.
