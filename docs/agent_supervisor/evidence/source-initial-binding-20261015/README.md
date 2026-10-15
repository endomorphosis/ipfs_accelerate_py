# Initial native source binding, 2026-10-15

The native source binder now configures its literal reader candidate under the
original SERVER lock, then releases SERVER for retained-CAS inventory replay.
The gateway retains its original lock, connection, identity, source store/CAS,
request and limits, scope bytes and empty continuation registry. A short-held
private busy marker prevents overlapping initial binds and source reads before
they wait for SERVER. An early original-connection transaction gate precedes ready/native SQL and
coordinator fences on the respective server/task/allocation paths. An active or
unsupported transaction refuses; this operation does not repair or roll back a
borrowed transaction.

When an existing task admission or managed allocation invokes the binder, its
actual retained guards supply publication checks. Genuine coordinator task and
allocation fences run without SERVER. Closing checks then revalidate the
original server/owner/candidate, native task row and complete current source
population on the original connection. The exact publication witness is
installed with that literal candidate before assertions which may raise. Task
and allocation finally paths retain only their own published candidate; they
cannot discover and adopt a substituted gateway reader. Once published, that
binding stays consumed even if a later check refuses. Pre-publication refusal
publishes no reader for this attempt and leaves its native markers empty. A
competing genuine reader is preserved rather than adopted or cleared. Repaired
original inputs can be retried only while the gateway remains unbound.

The public source binding and response schemas remain unchanged. A direct
server or gateway bind retains its existing contract; it does not acquire the
native task-admission or allocation authority of a guard the caller did not
supply. The private publication witness is a live object join, not a serialized
receipt or a new public capability. No source proxy, worker endpoint or coding
child is created by this binder.

The [reader](../../../../ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup.py),
[typed gateway](../../../../ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py),
[server](../../../../ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py),
[task admission](../../../../ipfs_accelerate_py/agent_supervisor/runtime/current_source_task_admission.py)
and [allocation](../../../../ipfs_accelerate_py/agent_supervisor/runtime/current_source_worktree_allocation.py)
implement this join. The [channel guide](../../source_lookup_worker.md#publish-the-initial-native-source-reader)
and [improvement plan](../../reversible_codebase_views_plan.md#initial-native-source-binding-2026-10-15)
keep this owner-side publication boundary separate from deployed worker adoption.

After all detached native fences, this source-binding phase samples its final
clock outside SERVER and passes that finite sample to closing task/grant and
allocation checks. Those closing checks do not sample a fresh source-binding
clock. Existing native fences and their default clock callbacks retain their
own contracts. Clock and lease observations remain sequential: this does not
qualify every arbitrary clock or coordinator callback, eliminate filesystem
writers, or establish an atomic delivery-time expiry guarantee. Native
permission delegation, separate worker process provenance and signed compiled
context adoption still require their own independently checked joins.

## Qualification and custody

The [unpatched control](baseline-01/commands.json) exercised the allocated-root
reader's actual retained-CAS callback at accelerator base
`55c42dd740d75c3ba04d6991c90cf0aa0dccd8a9`. It failed with
`initial source binding CAS replay must release SERVER`: one expected failure,
zero errors/skips, 23.079 seconds. All five production files were unchanged
from their BASE Git objects; before/after source, HEAD and index guards passed.
The initial four-route test snapshot remains historical and is not relabelled
as a passing qualification.

The [target migration](target-roster-migration.json) preserves the original
12-module target bytes under the baseline source snapshot and appends the
complete native task-admission test module. That module already belonged to the
listed source inventory. The [current roster](targets.json) therefore contains
13 complete modules, while the driver and [279-source inventory](source-inventory.json)
remain unchanged. The inventory contains 228 accelerator and 51 datasets files;
it is not a complete import, runtime or toolchain closure. Seven frozen changed
snapshots cover five production files, the new regression module and its driver.

The selected [full qualification](qualification-01/commands.json) passed **299
unique cases across 13 complete modules**, including **32 new initial-binding
cases**, in 1691.656 seconds. Raw collection and
[JUnit](qualification-01/junit.xml) independently contain 299 distinct cases,
with zero failures, errors or skips. Before/after listed source bytes, fixed
HEADs and indexes passed. This result includes the full native task-admission
module and the retained query, constructor, delivery, channel, spawn, lock-order,
nomination-custody and allocation suites. The new suite covers all four entry
routes, callback custody loss, borrowed transactions, literal post-publication
retention, genuine replacement readers, nested/concurrent host binds and the
finite closing-clock seam. Representative repaired CAS failures retry cleanly.

The [architecture review](architecture-review.json) binds the five final runtime
sources and records the two resolved ordering findings. The original unpatched
control remains a separate expected regression, not part of the passing count.

Execution uses the existing fixed datasets checkout
`f4a823231ad52c9252504810a1b00ed4738346b6` and kit checkout
`40c98c71b95be750509ffdb21e7e5494d91b4e79`, offline model settings,
one numerical thread, inherited AST sealing and short private runtime roots.
Runtime databases, grants and owner credentials are not published. Fixed
execution HEAD/index and listed source bytes must remain unchanged before and
after qualification; shared references are observations without actor attribution.

The [scoped parallel survey](parallel-survey.json) found no branch-only initial
binding change on the five reviewed paths in the six named cached worktrees.
The prior query, constructor, delivery and signed-closing stacks are already
published ancestors. Candidate dispatch/reporting uses a separate worker broker;
its continuation implementation is preserved rather than treated as an
alternative native task/allocation admission. This cached survey makes no
universal GitHub absence claim.

The release checker binds prior query/construction/delivery manifests as
immutable BASE Git evidence and preserves old evidence modes/blob identities.
It does not rerun or add the prior phase's 45 standalone datasets API cases.
It recounts this phase's raw control and selected full qualification, checks
exact current source/HEAD/index custody, executes the four pinned documentation
workflow blocks and checks links/anchors in this README and the two maintained
guides. Its exclusive documentation receipt and raw workflow log record the
observed gate outcomes; publication requires all four blocks and selected links
to pass.

## Scope and preserved research

The [new suite](../../../../test/api/test_source_initial_binding.py) uses real
Git/CAS, native DuckDB, task/claim/grants, Portal allocation and lifecycle state.
Listener and deployment UID metadata are authored seams. These new owner-side
cases create no broker, worker peer or child and execute no source program,
provider, model, solver or training. Retained launch/channel suites exercise
actual inert children within their original scope. This is no new separate-UID,
Docker, deployed coding-child or official-task qualification. Proof, execution,
completion, omission and whole-session token-saving claims do not follow.

Preserve the independent CodebaseIR, IntentIR, SecurityIR, LegalIR and UI/UXIR
families, schema/version and decoder-task inventories across 8D, 384D and 768D.
Reuse the existing checkpoints, datasets and embeddings. Published model and
Hugging Face records from other sessions retain their own lineage; this phase
performs no model upload, registration, decoder training or numerical reload.
Recent LegalIR failures on fresh semantic reconstruction panels and disabled
latent conditioning do not qualify a 384D teacher. Actual encoder conditioning,
fresh semantic and legal-text reconstruction, complete decoder/checkpoint
identity and per-family ModelManager custody still gate teacher promotion and
distillation. Current source-byte preservation is a separate measurement.

Next qualify explicit native worker permission delegation and daemon/wrapper/
child provenance with signed profile and compiled-context joins. Then compare
matched complete sessions, including planning, coding, retrieval, expansion,
edits, source refresh, verification, refusals and repair, at matched official
success. The source-custody improvements supply no measured total-token saving.

The [direct-host reader-identity followup](direct-host-reader-identity-followup.json)
records a separate residual: between operations, an internal host substitution
of a genuine equal-binding reader can satisfy the direct gateway's current-slot
check. Native task/allocation/broker joins and in-flight reads independently pin
the original reader; this is not a new native-claim bypass. The proposed durable
publication-identity check is unimplemented and needs its own baseline and
qualification. The [worker-permission followup](worker-permission-followup.json)
separately proposes pre-capture read-only permissions, authenticated wrapper/child
provenance and an independent compiled-context join. Both artifacts are proposals
and contribute no current production changes, tests or additional qualification.
