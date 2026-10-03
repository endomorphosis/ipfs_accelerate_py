# Source384 task context and publication

The optional Source384 preparation receipt travels with the exact task in
`supervisor-task-context-nominations@2`. A v2 bundle requires a selected receipt
for every task; an absent or malformed selection cannot silently become a v1
bundle. Unselected runs retain the v1 format. The launch configuration binds the
bundle path and digest, and the loader checks both the task CID and task alias.

`load_task_context_nomination` and `load_task_context_selection` call the
canonical Source384 consumer before returning a selected task's context. These
live checks apply to native START, child bootstrap, worker prompt construction,
and retries. The worker reads the admitted base repository, while implementation
edits occur in its separate worktree. Learned candidates remain advisory and
confer no proof, execution, or completion authority.

Symbolic planning reloads selected initial context after operation selection and
before admission, within the declared planning budget. Warm context rebinding
validates the selected bundle at entry and again after the new native world
capture. It carries the same receipt into the new v2 bundle and does not obtain
that receipt from an unbound field in the preparation result.

An accepted publication changes the source that the original receipt described.
The live checks continue to reject that original receipt for another dispatch.
STOP uses the original signed lifecycle grant and exact process identities; it
does not require stale context to become current again.

For observation only, an independently verified completed native publication may
retain the exact old bundle through `read_task_context_historical_selection`.
This reader establishes bounded envelope identity, not current source, model,
or inferred-meaning validity. The native publication verifier must still accept
the current source transition and completed task. Arbitrary source edits,
unvalidated publication, incomplete tasks, changed bundle bytes, or changed
task identity do not qualify this historical observation.

Automatic Source384 successor inference is not implemented in this slice.
Post-publication refresh reports `status="successor_unavailable"`,
`source384_current=false`, and `needs_successor=true`, retaining the predecessor
bundle and receipt. It writes no replacement v1 bundle and does not alter native
completion. Preparing and validating a new Source384 selection is required for
future planning or dispatch against the changed source. This limitation is not
a successful cold benchmark or a measured efficiency advantage.

The focused controls cover a real symbolic builder with native indexes, warm
native-owner recapture, exact bundle transport, native publication and completion,
and actual START/STOP. Tests that isolate bundle or lifecycle behavior use an
explicitly authored Source384 receipt/validator; they do not establish numerical
inference quality. Checkpoint inference and source-unit replay require their
separate consumer qualification.
