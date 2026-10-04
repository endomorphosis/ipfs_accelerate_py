# Source successors and current runtime integration

The additive [source delta API](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_successor.py)
records the exact transition between two complete native source publications.
It identifies retained, changed, added and removed inventory entries without
opening a model registry, fitting, performing inference or reusing numerical
output. It complements the [qualified resumable scan and signed worker](codebase_inventory_resume_worker_qualification.md).
The source and current runtime increments have passed native and independent
closed-archive qualification. Their results are recorded separately below.

`build_current_codebase_source_delta` receives an already published predecessor
and current head. The current native publication receipt must name that exact
predecessor, with the same repository, the next generation, and identical
capture bounds and declared exclusions. Publication remains an explicit call to
`RepositoryCodebaseIndex.prepare_current`; observation never moves a source or
model head. `validate_current_codebase_source_delta` freshly reconstructs the
record against native owners and the current checkout. CAS loading alone is a
historical integrity check and cannot establish current eligibility.

The versioned `codebase-inventory-source-delta@1` artifact contains a sorted union
of raw source keys, each complete old/new snapshot entry and structural member,
exact publication receipts, capture policy, membership identities, coverage and
implementation pins. It admits at most 1,024 members per capture, 2,048 union
members, 64 KiB per admitted source file, 4 MiB per manifest and 8 MiB per delta.
The 128-MiB retention estimate is serialized admission accounting, not an RSS
limit; the default resource reservation remains 1,024 MiB.

Retained means equal `SnapshotEntry.entry_cid`, including acquisition and Git
disposition metadata. Equal bytes alone do not imply retained entry identity.
The separate byte comparison remains unavailable for opaque or missing sides.
AST comparisons remain unavailable for unindexed/opaque members and bind exact
captured provenance. A change elsewhere can rebind the AST of a file whose bytes
remain equal.

Removed means absent from the current complete capture. The artifact explicitly
sets `physical_absence_verified=False`: Git ignore selection can omit an
untracked file that still exists. Neither a scan prefix nor an unchanged Git
commit establishes absence. Capture-policy changes require a separate profile;
the delta refuses to reinterpret those changes as removals.

The default optimized route uses the existing batched native projection replay;
`optimized=False` preserves the original current-member lookups and repeated
historical snapshot identities. Historical source/AST CAS and publication
receipts are checked independently; invalidated old SQL ASTs are not treated as
current facts. Current receiving replays all thirteen native relational families
and the live checkout. Its final fresh observation follows durable-record and
producer callbacks. All observations remain sequential, without a filesystem
lock or execution attestation. Each operation has the default shared 120-second
deadline, with explicit finite API deadlines limited to 600 seconds.

All fourteen inventory authority flags remain false. `numerical_reuse` and
`model_advanced` are exact false values. The original resumable scan still
requires its model provenance to match its source head; this increment does not
weaken that rule. Advancing a learned representation requires a separately
selected and validated source-bound model generation.

## Native source qualification

The [300-member native run](../../../../artifacts/codebase_ir_terminal_bench/source-delta-qualification-20261003-01/result.json)
passed in 104.79 seconds. It explicitly captures an authored repository, commits
a literal edit, addition, removal and rename, and publishes a second 300-member
source generation. The 302-key union contains 297 retained entries, one changed
entry, two additions and two removals. Renames remain removal/addition pairs;
the protocol does not infer semantic rename correspondence.

The two paths produce identical artifacts after removing only the `optimized`
field. Default delta construction took 12.72 seconds and reference construction
took 41.89 seconds. These are timings on this authored CPU fixture, not a
production throughput qualification. Fresh receiving took 8.34 seconds and
receiving through a newly opened source connection took 9.17 seconds. The latter
reopens the owner in the same process; it does not claim a process restart.

The fixture's 295 comparable source pairs have equal bytes, while all 295
comparable AST identities differ after the snapshot changes. Two retained opaque
members retain unknown byte comparisons. These distinctions prevent byte
retention from being interpreted as numerical or semantic reuse.

Receiving refuses same-head source drift and altered native SQL. A separate
pre-cancelled operation verifies resource cancellation; it is not an integrity
control. All thirteen relational table digests and the current head stay
unchanged through successful receiving and owner reopening. Original source/AST
artifacts remain preserved. Named constructor, fitting and inference guards
record zero attempts, and all leases and waiters drain. Host leases provide
admission accounting; this run does not claim kernel memory enforcement.

The [59 focused tests](../../../../artifacts/codebase_ir_terminal_bench/inventory-source-successor-tests-20261003-01.xml)
pass with zero skips: sixteen native source/DuckDB/CAS controls, three actual
resource controls, and forty pure protocol or type controls. They include
late source mutation, corrupt artifacts, policy changes, partial or forged
ledgers, opaque identities and ignored-file removal without physical absence.
These cases are separate from the earlier 835-case resumable-worker regression.

The [independent closed-archive reader](../../../../artifacts/codebase_ir_terminal_bench/source-delta-qualification-20261003-01-readonly-audit-20261003-01.json)
passes for both this run and the [separate ignored-file control](../../../../artifacts/codebase_ir_terminal_bench/source-delta-ignore-qualification-20261003-02/result.json).
It rebuilds the complete union and CID/source/AST/publication closure, joins the
native observations, and compares two full archive passes without opening native
owners, running Git or SQL, or writing to either primary archive. Its retained
reader has SHA-256 `df684efd4027534c8b435d982ced2556d83d70bd6dedf39bf5269992596d3e44`.
The [portable reader controls](../../../../artifacts/codebase_ir_terminal_bench/source-delta-reader-guard-tests-20261003-01.json)
pass all 54 stdlib cases with zero failures, errors or skips; these are separate
from pytest and native execution.

The ignored-file run passes in 1.19 seconds: one entry leaves the complete
capture while the actual ignored file remains present with unchanged bytes.
The earlier [failed fixture attempt](../../../../artifacts/codebase_ir_terminal_bench/source-delta-ignore-qualification-20261003-01/result.json)
is retained. Changing Git disposition also changed the supposedly retained
normal entry's identity; the fixture now keeps that disposition stable between
captures. No model, inference or fitting calls occur in either attempt.

## Current runtime cleanup

The advanced working runtime moved completion-service binding after allocation
of its bootstrap listener/thread and fenced run lease. Inventory construction
previously lacked the finite profile's exception cleanup at that point. The
increment captures the original native resources before callbacks, performs
native STOP and exact disposal after failure, and keeps the execution lease and
heartbeat when cleanup remains unproved. Clearing a diagnostic flag cannot
authorize resource release.

Completion-service construction also needs rollback after installing its
handler. The bridge removes only its exact newly installed handler and binding
token. A failed rollback retains private custody for a native STOP-bound retry;
it cannot replace or retire a foreign handler. Normal finite cleanup and the
inventory's fresh receiving/Popen ordering retain their existing contracts.

The [affected regression suite](../../../../artifacts/codebase_ir_terminal_bench/inventory-constructor-affected-tests-20261003-02.xml)
passes 191 cases with zero failures, errors or skips, including five new
constructor controls. Those controls use actual native Quack, transport,
bootstrap-thread, fenced-lease and lifecycle resources, with controlled receiving
and launcher adapters and zero Popen/model calls. Two existing finite-worker
constructor callback controls also [pass](../../../../artifacts/codebase_ir_terminal_bench/inventory-constructor-cleanup-tests-20261003-03.xml).
The first affected regression attempt is retained with eight fixture failures:
seven manually constructed fixtures lacked the optional inventory-scope
attribute, and one STOP double lacked an explicit successful result. Minimal
compatibility fixes preserve the strict native cleanup guard.

Across the retained successful new pytest receipts there are 252 distinct case
identities, including 175 already present in the earlier 835. The increment adds
77 identities, bringing the combined total to 912. Parameters remain part of
identity; only the isolated `.test.api.` classname prefix is normalized. The 54
stdlib reader controls remain separate. Failed constructor/regression receipts
retain 105.036 seconds of JUnit time; they add no successful case identities.

The [selected current source receipt](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-current-runtime-source-20261003-01/selected-source-receipt.json)
pins 19,381 files across the three repositories, including the complete new
finite-proof-query import dependencies. Host execution generated fourteen
unselected bytecode files in that candidate; they remain preserved outside the
snapshot policy. Qualification concerns the selected file bytes and the
container's separately copied sources, with no claim that the candidate's entire
directory remained unchanged.

The [fresh current-runtime worker run](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-15/native/result.json)
passes in 1,337.07 native seconds, with 1,353.65 seconds for the outer run. It
independently receives the previously completed 300-member scan, signs and
installs the original complete task graph, checks the prerequisite, binds the
native runtime and launches the authored worker as UID 1001. Both original
tasks complete. Publication creates a two-parent merge affecting only `calc.py`,
and both public checks pass. Native START and STOP succeed; bootstrap errors
and remaining processes are zero. Explicit post-STOP owner cleanup removes the
worktree and its Git administration, and the published source refuses the old
scan completion.

This run creates zero new scan pages or fitting epochs and inherits exactly two
setup epochs from the closed seed. The five reported numerical/Adam summary
fields remain identical, and the independent reader also reconstructs the full
retained checkpoint state. Native leases and waiters drain; the host lease is
released and the container is removed. The [closed independent worker audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-resume-worker-qualification-20261003-15-readonly-audit-20261003-01.json)
passes with two preserved full archive observations in 16.05 seconds. It uses the
unchanged reader SHA-256 `db0d93f8321ea0a09a8dd22bc7b42b6195c3ce31e7eb638b5c15d9b7cc111076`.
Its 19,381 selected-file join and actual worker stdout receipt bind this result
to the exact staged generation; execution qualification does not transfer to a
mutable checkout or arbitrary repository.

The [machine review](../architecture/repository_proof_index_and_codebase_ir.source_delta_runtime_review.json)
keeps the source, ignored-file, regression and current-worker receipts separate,
with historical failures and initialization costs. It preserves the earlier
qualification and its native13 Git-index endpoint disclosure. The retained
[review builder](../../../../artifacts/codebase_ir_terminal_bench/source-delta-runtime-review-builder-20261003-01.py)
rebuilds the record without native owners, Git, SQL, models or jobs; `--final`
requires the passing actual worker result and independent closed audit.

All 32 production acceptance tasks remain open, including general acquisition,
incremental learned inference, cross-shard proof dependencies, CUDA/384D and
production activation. The next source-successor integration must select a new
source-bound model generation, recheck affected proof dependencies and sign a
new planning context. This source-only delta grants no reuse of prior learned
output or evidence.
