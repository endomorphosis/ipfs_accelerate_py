# Native source-query replay, 2026-10-14

The native typed gateway now captures original SERVER, connection, reader,
store/CAS, owner/source binding, active issued grant and continuation registry
witnesses before source-specific callbacks. A short-held private guard marks one
read busy without holding its mutex across callbacks. Nested and concurrent
source reads refuse before waiting for SERVER.

The gateway samples source-specific clocks outside SERVER and the grants lock.
It releases SERVER for the original reader's CAS load, then stages the exact
response privately on the original native connection. A final detached clock
sample precedes actual grant, owner/task revision/body, complete source and
selected working-byte validation. No supported CAS or clock callback follows
that closing section. Expiry uses the supplied real clock sample; checks remain
sequential and provide no writer lease or atomic delivery-time expiry guarantee.

A grant renewal completed before capture remains usable. Replacing that captured
native grant during the read refuses. Missing, non-boolean or active original
connection transaction state also refuses; the read performs no BEGIN, writes,
COMMIT or ROLLBACK. Refusal does not undo a transaction owned by another caller.
The common grant helper retains its default behavior for unrelated operations;
this phase does not qualify every gateway clock or authentication-open callback.

The private read state pins the original connection/grant, copied request and
binding bytes, retained replay result and staged response. Only after closing
checks does publication prune expired continuations, consume the predecessor
from the retained request and issue its bounded successor. A refused read leaves
the predecessor intact. The original continuation registry is retained, and no
caller-supplied restored inventory supplies publication authority. Exact payload
and response schemas, UTF-8/base64 content, conservative envelope halving and
validated response/wire-byte accounting remain the existing contract.

The [reader](../../../../ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup.py)
and [typed gateway](../../../../ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py)
implement the split. The [channel guide](../../source_lookup_worker.md#detach-native-source-query-replay)
and [improvement plan](../../reversible_codebase_views_plan.md#native-source-query-replay-2026-10-14)
keep initial binding, deployed worker handoff and learned reconstruction separate.
The reader's standalone compatibility entry point also closes native/source
state after its final clock; its caller supplies connection serialization.

## Qualification and provenance

The [unpatched control](baseline-01/commands.json) executed the actual retained
CAS callback against accelerator base `b37a248a74d2cbad55dec7b1289aa7cfb367d7e4`.
It failed with `native query CAS replay must release SERVER` before genuine
native reentry, with one failure, no errors/skips and 25.289 seconds. The initial
one-case test snapshot remains historical as further cases are added.

The [first full attempt](qualification-01/commands.json) collected 250 cases
and stopped at a connection-substitution fixture that tried to reopen the
database held by the real native owner. Its seven passes, one failure, zero
errors/skips and 164.568 seconds remain separate from final passing counts.
The [fixture correction](fixture-correction.json) moves the two replacement
handles to separate private databases; production sources are unchanged.

The [historical full record](qualification-02/commands.json) passed all 250
distinct cases in 1343.765 seconds, with zero failures, errors or skips. It used
the exact [11-suite roster](targets.json): the new query suite, complete existing
current-source lookup suite, and nine preserved constructor, delivery, channel,
launch, nomination and allocation suites. One pytest invocation retains raw
collection, distinct JUnit identities, outcomes and all four changed-source
snapshots. Failed attempts remain separate; previous counts are not aggregated.

That run used datasets `4b50ae`. Its four documentation blocks passed with 113
relative links and 17 anchors; the original documents and checker are retained
in [historical snapshots](historical-documentation/documents.json). During
publication review, incoming main changed the listed semantic-state API. The
[fresh unpatched control](baseline-02/commands.json) reproduced the same lock
error at accelerator `f15c146467c29c95f39ae1f041c8eb84e76d8bc8` with
datasets `f4a823231ad52c9252504810a1b00ed4738346b6`: one expected failure,
zero errors/skips, 23.141 seconds. The two runtime Git objects were unchanged
from original base; exact previously qualified patch bytes were restored after
this control.

The [selected fresh full record](qualification-03/commands.json) passed all 250
distinct cases in 1178.039 seconds, with zero failures, errors or skips, using
the same 11-suite roster on fixed datasets `f4a823`. The separate
upstream semantic-state API qualification is `dependency-01/commands.json`; it
uses its own source snapshots and raw JUnit, and its counts are never added to
the supervisor's 250 cases. All 45 distinct API cases passed in 1.011 seconds,
with zero failures/errors/skips and 12 separately listed input bindings.
Historical results are not relabelled for new code.

The [listed inventory](source-inventory.json) covers 277 dependencies: 226
accelerator and 51 datasets files. It is not a complete import/toolchain closure.
The selected fresh datasets and kit execution checkouts are
`f4a823231ad52c9252504810a1b00ed4738346b6` and
`40c98c71b95be750509ffdb21e7e5494d91b4e79`. Tests use offline settings,
one numerical thread, explicit inherited AST sealing, unique short private
runtime roots and scrubbed private-owner environment names. Fixed HEAD/index
and source bytes are guarded before and after; shared references are separately
observed without attributing an actor. Runtime databases and credentials are not
published.

The [parallel survey](parallel-survey.json) retains the scoped incoming review.
Base main advances only its embedded datasets pin to
`667033fc7040fedeb67c1d359f10ed44fa2e27b5`; all 51 listed datasets blobs
matched the historical fixed `4b50ae` checkout. This preserves parallel LegalIR engine and
publication fixes without claiming this query qualification tested those fixes.
Incoming accelerator `f15c146` advanced the embedded pin to `cd420d7301`;
incoming parent `683fb27c` preserved the new F17 audit and advanced datasets to
`f4a823`. Fifty of 51 listed datasets files are identical to the historical
checkout. The changed API reads the live backend before parsed-index cache use,
and requires fresh qualification. Both incoming contributions remain preserved.
The selected local documentation receipt will be `documentation-checks-02.json`,
with exclusive raw log `documentation-workflow-02.log`; the release checker recounts
raw evidence, executes four exact pinned workflow blocks and checks three
maintained documents. Final `manifest.json` binds the phase without itself.

## Scope and next boundary

The [new suite](../../../../test/api/test_source_query_replay.py) uses real
Git/CAS, typed owner/client transport and native DuckDB, with an actual native
task/allocation fixture for callback fencing. It launches no child or worker
peer. Other selected suites launch and reap inert same-owner Python children.
This is no separate worker-UID, deployed coding-child, model/provider, solver,
Docker or official-task qualification. Proof, edit, completion, execution and
whole-session token-saving authority remain false.

Its 27 cases cover eight replay mutations, four final-authorization mutations,
native reentry during CAS and source clocks, four clock mutations, borrowed
transaction refusal, nested/concurrent busy reads, expired-cursor retry,
pre-request and in-flight renewal, admission registry substitution, redundant
cursor-field injection and staged-connection substitution. Refusal checks retain
the exact original registry and private-state cleanup. Stable reads retain exact
source bytes and encoded-envelope/wire accounting. Candidate replacement handles
are genuine separate private DuckDB databases; the staged-connection case
requires zero SQL on the candidate.

Initial native source binding still constructs its reader under SERVER. Qualify
that independent CAS seam next, preserving consumed task/allocation bind-once
markers and refusing replacement owner/store witnesses. The [initial-binding proposal](initial-binding-followup.json)
records three nested SERVER layers and exact publication-candidate custody; it
contains no implementation or executed tests. Preserve the separate
native permission delegation and daemon/wrapper/child provenance plan, and join
signed profiles and compiled instructions independently before deployed trials.
Existing CodebaseIR, IntentIR, SecurityIR, LegalIR and UI/UXIR assets remain
separate by family, schema/version, decoder task and 8D/384D/768D size. Reuse
existing checkpoints and embeddings; actual encoder conditioning, fresh
semantic/text reconstruction and per-family ModelManager custody still gate
teacher promotion and distillation. This query change supplies no model-quality
or whole-session token-saving result.
