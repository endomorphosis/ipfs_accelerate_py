# Resumable inventory, signed admission and isolated worker integration

The additive datasets scanner seals a complete inventory root, writes bounded
immutable pages and resumes from a root-bound cursor. A final completion record
requires the entire ordered population. The supervisor binds that advisory
context into an explicitly selected signed manifest and planning receipt, keeps
every original task and check, and reserves a separate native execution scope
before launching a worker.

The [scanner](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_resume.py)
accepts at most 1,024 inventory members and 64 members per page. Callers can
narrow these limits. The root binds source head, full membership, frozen model,
feature basis, budgets and implementation generation. Pages bind exact offsets,
predecessors, source dispositions and numerical results. Completion checks the
whole chain; a prefix remains incomplete and cannot establish absence. Changed
source requires a new root. The legacy 256-member scanner remains separately
versioned.

Fresh entry observations replay the native source/AST projection, source CAS
objects, current checkout, model ancestry, registry and artifacts. Closing
observations repeat the native source, SQL, history, model and artifact fences.
Receiving validation performs no numerical forward pass or fitting. A raw CID
protects exact retained bytes; it does not authenticate arbitrary rehashed
numerical output or prove a source property. Fourteen authority flags stay
false. Coverage counts source membership, inference and unsupported/deferred
members separately.

The default optimized path batches exact relational projection replay in the
[native helper](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_projection_replay.py).
It preflights conservative result bytes before fetching, compares all thirteen
native tables and preserves intrinsic invalidation checks. It also charges
canonical request bytes incrementally, avoids retaining target bodies during
receiving, and replays each native target once in the numerical worker. An
operation-local snapshot CID removes two repeated whole-inventory hashes from
ordinary AST provenance checks; the operation closes with recomputed manifest
and snapshot identities.

The default also reuses native candidate reads and complete validated target
envelopes while reconstructing model ancestry. Each distinct canonical target
still passes native replay before its binding is reused. Candidate and target
byte invariants are checked, and private cache indexes are discarded before
the subsequent history and fresh source/model fences. `optimized=False`
selects the original lookup, serialization, target and ancestry replay paths.
No caller-provided cache or long-lived freshness token is trusted.

The private [paired receiver](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_receiving.py)
performs full completion and target reconstruction at entry, then retains
bounded hashes and structural metadata. After owner callbacks, its one-use
closing gate rereads every typed CAS page and the complete coverage ledger,
checks the frozen target bindings and repeats fresh native closing observations.
The supervisor groups construction and START separately; each receives a fresh
full entry observation. START performs that observation before control dispatch,
then closes under the original task-owner lock immediately before native Popen.
Canonical START and STOP remain bounded to 30 seconds. The default receiving operation
shares one 120-second deadline across entry, callbacks and closure. The native
qualification uses this default; explicit bounded deadlines may be selected up
to 600 seconds. Cleanup does
not replay source after an authorized worker can edit it. `optimized=False`
keeps the original repeated full receiving paths.

Source-path accounting for the signed workflow changes from 40 full completion
and target walks to 12: five standalone admission/preparation walks, five
reservation walks, one construction walk and one START walk. Each still has its
own fresh entry and closing observations. These counts describe the inspected
implementation; they are not native instrumentation or a throughput measurement.

The [signed admission boundary](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_inventory_evidence_admission.py)
uses manifest version 5 and planning receipt version 3 only when explicitly
requested. A compact signed declaration binds the full context CID, complete
scan summary and optional exact evidence references. Full context remains in
immutable artifacts and the public worker artifact. Compaction reduces repeated
ledger storage in native task bodies; their unchanged 262,144-byte bound remains
enforced, and oversized declarations are refused. Old manifest and public
artifact versions retain their own bounds. Native task installation surrounds
all writes with a transaction and current-source checks; closing failure leaves
no newly schedulable task rows.

Launch binding projects only the native `EvidenceRequirementKind` and
`AssuranceLevel` enum fields into their signed JSON values. A separate bounded
ledger pins the enum paths, classes, members and values alongside the original
caller object. Construction and START reject callback substitutions, including
equivalent strings, dictionaries or another caller object. The global canonical
identity encoder remains unchanged.

The [worker context](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_inventory_evidence_worker_context.py)
retains the signed tasks, dependencies, checks, pending requirements and complete
advisory inventory. The public reader verifies signatures and exact context
projection, then checks the allocated worktree against the signed source
baseline. It needs no private model or evidence database. Its receipt explicitly
does not claim native inventory verification or persistence authority.

The explicit inventory public route uses the same full source, tracked-membership
and declared-output checks as owner observation. After verifying the signature,
baseline and distinct allocated worktree/common Git directory, it permits only
three fixed Git reads with per-command trust for the exact canonical root,
sanitized configuration, disabled hooks/fsmonitor and replacement objects, and
a ten-second command bound. It writes no Git trust configuration and leaves
ordinary owner observation unchanged.

The [execution scope](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_inventory_execution.py)
holds a real resource reservation through launch, claim, validation, publication
and STOP. It checks the sealed owner, complete signed task population, native
prerequisite evidence, public artifact bytes and root-controlled launcher.
Manifest version 5 refuses execution without this explicit scope. Final launch
checks run after callbacks and include detached native SQL and guarded file
identities. Typed Quack and signing callbacks finish before the final receiving and
Popen section under the original owner lock. Construction closes under that lock; START prepares a private one-use
fence outside it, then closes receiving and checks task relations, completion,
plans, goals, validation evidence, owner generation, metadata and exact retained
planning receipt bytes inside it immediately before Popen. The preparation pins
the process, thread, runtime, owner, lease, admission and candidate; callback
substitution or reuse is refused. This does not qualify general crash recovery or automatic worktree
pool cleanup.

The scan's default serialized retention estimate is 224 MiB; the paired receiver
adds three bounded metadata ledgers for a 236-MiB estimate within a 1,024-MiB
reservation. These estimate retained bytes, not an RSS ceiling. Numerical
pages carry at most 64 targets and reuse one frozen model per page. Receiving
walks retain one page at a time; repeated prefix validation still contributes
work as the chain grows. Source and database observations are sequential, not
one atomic transaction spanning the checkout and all owners.

## Qualification status

The bounded composed integration passed on its exact frozen staged sources.
The [fourteenth native run](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-14/native/result.json)
and [independent closed-archive audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-14-readonly-audit-20261003-01.json)
join the ninth run's complete 300-member, ten-page scan and four historical
process resumptions to fresh signed admission and native worker execution.
Both original tasks completed. The actual UID-1001 worker read its signed public
context, patched only `calc.py`, and produced the required two-parent merge
with passing public checks. START, STOP, UID cleanup, explicit owner-authorized
fixture worktree cleanup and rejection of the old completion after the source
changed all passed. No processes, linked-worktree admin directory or reservations
remained. The independent reader preserved every archived entry across its
before/after observations.

The fresh run created no scan pages or resume subprocesses, performed no
subsequent registry reopenings and recorded zero new fits with two inherited
setup epochs. It freshly received the entire completed chain against its current
source/model owners; the model and optimizer state stayed unchanged. Native
execution took 1,312.62 seconds and the container wrapper took 1,336.32 seconds.
These qualify the authored 8D CPU fixture, not production latency or general
source semantics. The [machine review](../architecture/repository_proof_index_and_codebase_ir.inventory_resume_worker_review.json)
records exact qualified staged hashes, the current working hashes separately,
all fourteen attempts and their costs.

The
[scanner, lineage and receiving suite](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-scanner-receiving-tests-20261003-01.xml)
has 317 passing cases, including 64 actual native DuckDB/CAS, target-adapter or
lineage controls, three genuine nested resource accounting, cancellation and
deadline controls, and 250 protocol or inert controls. The supervisor has
368 distinct passing controls: 334 earlier cases across its
[expanded regression](../../../../artifacts/codebase_ir_terminal_bench/inventory-paired-supervisor-tests-20261003-03.xml)
and [affected rerun](../../../../artifacts/codebase_ir_terminal_bench/inventory-paired-supervisor-tests-20261003-07.xml), plus
[28 new native enum controls](../../../../artifacts/codebase_ir_terminal_bench/inventory-enum-guard-tests-20261003-02.xml),
and [six additional launch controls](../../../../artifacts/codebase_ir_terminal_bench/inventory-quack-lock-tests-20261003-02.xml).
The launch controls include actual threaded Quack in both optimized and reference modes, with eight
post-preparation authority mutations refused in each mode. Scan and signature
adapters in these tests remain controlled; the held-lock launch probe is not a
worker qualification.
These exercise signatures, Git, native pending-task SQL, Quack, leases and
explicit launch/replay doubles. Another 54 legacy public-reader cases pass.
A further
[34 pure setup-transport guards](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-setup-guard-tests-20261003-01.xml)
pass without opening native owners or training, as do
[29 completed-scan transport guards](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-scan-guard-tests-20261003-01.xml).
The [public read regression](../../../../artifacts/codebase_ir_terminal_bench/inventory-public-git-tests-20261003-01.xml)
adds 25 cases; the [affected owner and public rerun](../../../../artifacts/codebase_ir_terminal_bench/inventory-public-git-legacy-tests-20261003-01.xml)
reports eight declared-create cases beyond the previous selected reports and
36 repeated identities. The [isolated public repair](../../../../artifacts/codebase_ir_terminal_bench/inventory-isolated-public-git-tests-20261003-01.xml)
passed all 69 affected controls. These exercise real Git ownership refusal via
its test switch; they do not claim an actual UID boundary.
The retained successful JUnit reports have zero skips and 835 distinct test
identities after removing repeated runs. The
[isolated repaired snapshot](../../../../artifacts/codebase_ir_terminal_bench/inventory-isolated-quack-lock-tests-20261003-01.xml)
also passed all 175 affected supervisor controls. Separately,
[fifteen independent-reader cases](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-independent-reader-guard-tests-20261003-05.json)
cover 89 named tamper controls, including both exact authored-worker log lookups
against retained worker bytes and refusals for the wrong directory spelling.
These controls remain separate from the trained 300-file Docker qualification.

The [public-reader Docker diagnostic](../../../../artifacts/codebase_ir_terminal_bench/inventory-public-worker-read-20261003-01/result.json)
uses the twelfth attempt's actual signed public artifact and copied repository.
Six actual UID-1001/group-1000 children perform two initial and two restored
public replays, plus four refusals for allocated-source drift, canonical-source
drift, extra indexed source and a foreign common Git directory. Receipts match,
private and artifact writes are denied, and unscoped Git still rejects ownership.
It completed in 10.8 seconds with no native owner/fit attempts or private database
or signing-key inputs. Its one-CPU, two-GiB, 128-PID, network-disabled container
was deleted before releasing the host reservation; consumed source bytes were
preserved. This qualifies public reading only, with no claim, patch, publication
or native worker completion.

An isolated [diagnostic on a copy of a closed fixture](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-operation-profile-20261003-08/diagnostic.json)
reconstructed the identical 35,222,429-byte ancestry chain in both paths. Native
target validations fell from 33 to 3. Reference replay took about 22 seconds;
optimized replay took about 4.5–4.6 seconds, with no fitting and unchanged model
and source owners. These are diagnostic replay measurements, not a production
throughput result or completed native worker qualification.

The [Docker harness](../../benchmarks/agent_supervisor/container_coding/run_codebase_inventory_resume_docker.py)
uses a 300-file authored repository and a private 8D CPU model. It can train the
two-epoch root/child setup on its explicit training selections or reuse a
verified closed setup. Complete inventory scanning does not train all 300
members. Setup reuse
copies only the captured source/AST artifacts, original repository, native
source/model databases and exact checkpoints; it copies no previous scan pages.
Fresh native owners and source/model lineage still validate before scanning.
Two inherited epochs are recorded separately from zero new setup fits.

The closed ninth attempt completed the actual ten-page scan, four process
resumptions, default/reference parity and seven pre-worker refusal controls.
Its independent
[completed-scan observation](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-09-readonly-audit-20261003-02.json)
verifies all 300 dispositions: 205 inferred, 89 deferred by budget and six
unsupported, opaque, unindexed or parse-failed members. That attempt failed
before constructing the worker because an owner identity guard sent its valid
120.0-second timeout to an integer-only encoder. The repaired guard explicitly
binds the timeout's numeric type and exact value; the failed attempt remains
unqualified.

The tenth attempt freshly received that completed scan, signed and admitted the
full graph, installed both tasks, completed the native prerequisite check and
sealed the execution scope. Construction then rejected a native
`EvidenceRequirementKind` enum in its private identity projection. It stopped
before START with zero new fitting and drained reservations. Its
[independent archive audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-10-readonly-audit-20261003-01.json)
retains the actual failure and verifies fresh completed-chain receiving; it
does not qualify worker execution.
The separate
[pre-runtime observation](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-10-readonly-audit-20261003-03.json)
also verifies the signatures, all 300 baseline sources, both native task
contracts and the prerequisite claim, public check and committed completion.
The intermediate reader refusal is retained alongside the corrected task-CID
regression; neither observation records a worker launch.

The eleventh attempt passed typed admission binding, then stopped before START
on a Quack transport timeout. Construction held the task-owner lock while its
callback waited for a handler requiring that same lock; the Popen callback had
the same pattern. Its
[closed archive audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-11-readonly-audit-20261003-01.json)
preserves the failure, verifies fresh scan receiving and records zero new fits
and drained reservations. The repair moves RPC callbacks outside the lock while
retaining fresh source/model closure and detached native SQL immediately before
commit or Popen.

The twelfth attempt passed construction, START and STOP with no bootstrap errors
and no remaining processes. Its native residual claim reached `in_progress`,
but the UID-1001 worker's public reader rejected Git ownership during the
inventory-only source observation. The public reader's existing exact-root Git
policy had already passed its other checks; the added observation used the
owner-only helper. The [closed audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-12-readonly-audit-20261003-01.json)
preserves that failure and fresh completed-scan receiving. A separate
[START/STOP observation](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-12-readonly-audit-20261003-02.json)
verifies retained lifecycle receipts and cleanup without qualifying task
completion or publication. Zero new fits and drained reservations are retained.
The repair shares the full inventory checks through an explicit public read
route with per-command exact-root Git trust; the ordinary owner route remains
unchanged.

The thirteenth attempt completed the residual task, produced the required
two-parent `calc.py`-only merge and passed START/STOP with no bootstrap errors or
remaining processes. Its receiving gate searched `implementation_logs` while
the actual daemon wrote `implementation-logs`, so the run stopped before its
explicit worktree cleanup and old-scan invalidation checks. The
[closed archive audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-13-readonly-audit-20261003-01.json)
preserves the failed native result. The corrected two exact directory predicates
allow a separate [authored-worker receiving observation](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-13-authored-worker-receiving-20261003-01.json)
to verify the actual signatures, full context, boundary, receipt and patch bytes.
It leaves the original attempt and unexecuted tail unqualified. No numerical or
worker behavior changes are needed for this repair.
A later [scoped observation](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-13-readonly-audit-20261003-03.json)
verifies the retained task completion and raw two-parent merge. It discloses a
Git-index change after the first audit; a downstream `git status` diagnostic may
have refreshed index metadata, but that cause is not independently attested.
All other archived entries match that audit. Its preservation claim applies only to its own
before/after observation, and the thirteenth attempt remains unqualified.

Completed-scan reuse copies only 14 immutable CAS objects into a fresh setup;
27 additional files retain provenance for the transfer. The transport opens no
native owners, runs no fitting and creates no pages. The receiving owner must
still validate the full completion against its current source/model generation.
The composed worker run records the four earlier scan processes as historical,
with zero new scan processes, pages or registry reopenings. It creates fresh
signatures, tasks, admission, reservations and worker evidence. An independent
reader joins the exact earlier scan audit and current lifecycle without treating
the failed earlier worker attempt as qualified.

In a new scan the harness creates two pages, then reopens owners in four separate subprocesses
that each return at most two more pages. Only the final full ten-page completion
can enter signed admission. Each chunk retains its request cursor, exact returned
page IDs, next cursor, process receipt, immutable checkpoint state and drained
resource counts. Page deadlines remain 120 seconds and each fresh process has a
420-second budget. The successful composed run declares a 3,000-second overall
budget while the CLI default remains 900 seconds, with the same
12-CPU, 8-GiB, 512-PID kernel limits. This is a qualification budget, not a
production latency result.

Its isolated authored worker uses
the actual native claim/check/publication path. This fixed offline fixture is
not a learned patch generator. The independent
[audit](../../benchmarks/agent_supervisor/container_coding/audit_codebase_inventory_resume_worker.py)
reads retained files and verifies signatures without importing product owners.
Failed native results and historical audits remain retained; later archive
endpoint changes are disclosed separately. Setup fitting and execution costs
are accounted separately.

Qualification applies to the exact frozen staged implementation files and
their recorded hashes. The mutable working runtime and its advanced endpoint
require their own qualification; source similarity does not transfer the result.

This is bounded integration work toward RPI-013/022/023/029/032. It does not
activate production defaults, qualify 384D or CUDA, prove general source
semantics, or close any of the 32 production acceptance tasks.
