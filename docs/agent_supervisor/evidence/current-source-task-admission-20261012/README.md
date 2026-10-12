# Native task admission qualification, 2026-10-12

The owner-private admission guard joins an existing process-serialized native
claim to the original Quack owner's current task row and active owner-attested
daemon grant. It derives the actual alias, revision and complete canonical body
CID and can bind the existing source reader with those task fields. A marked
reader requires the same private handle in the optional worker-broker guard.
No new database owner, coordinator, CAS, worktree or model is created by the API.

The final host run passed **82 distinct tests**, with zero failures, errors or
skips. It contains 17 new admission cases, 21 existing worker channel/router
cases, 40 current-source reader/CLI/transport cases and four typed-owner controls.
The runner retained normal pytest plugins and used distinct fresh orchestration
directories and AST-seal databases for collection and execution. Collection took
59.63 seconds; execution took 179.94 seconds. All 29 recorded direct source/test
bindings were identical before and after the final run and at publication
freeze. This records a bounded source closure, not every transitive dependency.

- [Exact final commands, environment updates and source hashes](final/commands.json)
- [Collected distinct node IDs](final/nodeids.json)
- [Raw final collection log](final/collection.log)
- [Raw final test log](final/tests.log) and [JUnit](final/junit.xml)
- [Qualification runner](qualification_runner.py)
- [Independent component review](independent-review.json)
- [Integration review](integration-review.json)
- [Documentation workflow record](documentation-checks.json)
- [Final documentation checks](documentation-final-checks.json)
- [Newer parent datasets equivalence](parent-datasets-equivalence.json)

The new cases materialize a real native task, seal its route, issue actual typed
grants, call `DatabaseImplementationDaemon.claim_next()` and retain its actual
process-serialized coordinator. A real two-lane strict-sharding configuration
selects that adapter. They exercise exact full-body/claim binding, renewal,
ready-task substitution, forged attempt/fence identity, wrong live peer/client,
released/expired claims, revoked daemon grants, task revision/body/phase drift,
source/CAS substitution, different allocation, omitted/serialized guards and
late drift after a real backend response. The closing-clock control performs a
source edit during the final coordinator fence; closing source checks refuse
delivery while preserving the native success-byte ledger.

These are owner-side API tests. Quack listener capability/lifecycle and worker
deployment metadata are authored seams. DuckDB, typed Unix requests/grants,
native task receipts, claim fencing, retained CAS and Git worktrees are real.
No provider, decoder, prover, source-program execution, separate coding-worker
UID or official Terminal Bench reward runs in this phase. Earlier separate-UID
source-worker evidence remains historical; it does not qualify this new guard
through the normal remote daemon launch.

The [first failed execution](first/tests.log) stopped at one setup error after
collecting the same 82 cases; [its command/source bindings](first/commands.json)
and [JUnit](first/junit.xml) are retained. The [exact failed-run fixture](first/test-fixture.py)
matches the first run's recorded SHA256. Host umask created authority database
files with mode `0664`, which the new profile correctly refused. The fixture
then explicitly protected the existing real control/coordinator/execution files
with `0600` and the daemon lane directory with `0700`; production code and the
actual materialization/claim flow were unchanged. Quack startup does not
automatically provision this narrower file-permission prerequisite. The final
fixture is qualified only after that explicit host provisioning.

The passing run starts at accelerator `8daaf4356a78c12c36cfcbc692e5feaaf896cad4`,
preserving the parallel working-source and candidate capsule/formula refreshes.
It uses the retained datasets checkout
`f47161b757190380e9898aff155067c1ad609217`. The parent preserves the newer
LegalIR curriculum pin `e39fe996bd9e786a84f739fda9dde9a6c70546da`: all six direct
datasets dependencies and the entire `software_contracts` Git subtree are
identical. This comparison does not qualify the whole parent or the independent
LegalIR experiment. Existing assets, embeddings and family/schema/task/dimension
checkpoint separation remain intact.

The existing native coordinator can repair bookkeeping metadata while opening
and performs its normal fence/event/expiry writes. This guard is not a strict
read-before-repair schema audit or a zero-write observer. Checks remain
sequential; source publication is bind-once and a late failure can retain an
already published reader. The caller retains owner cleanup responsibility.
Direct privileged SDK reads retain generic current-source checks; coordinator
refencing is specific to the optional guarded broker path.

Normal Terminal activation still needs original-runtime bootstrap custody,
authenticated allocated-worktree/context custody and the actual coding-child
birth handoff to the original Quack owner. A daemon or run root is not a coding
child. Native committed-source admission also remains distinct from the
parallel semantic channel's unverified working-candidate refresh. Accepted
proof/index publication, mapped compact edits, decoder reconstruction and
complete provider-session token savings require their own qualification.

Public artifacts contain source code, commands, test logs, bounded reviews and
hashes. Private database/CAS contents, credentials, model weights and pytest
state databases are not published. Backend success byte accounting is preserved
on a later refusal; unknown handshake/error/partial costs remain unknown.
