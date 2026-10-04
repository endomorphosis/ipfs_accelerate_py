# Bounded whole-inventory CodebaseIR inference

The additive `source_bound_inventory_scan_v1` API scans the complete admitted
repository inventory with an explicitly selected, already trained model bound
to that exact source head. Optimization defaults to enabled. The existing
cohort API, checkpoint encoding, training split limits and registered runtime
remain unchanged.

```python
from ipfs_datasets_py.logic.software_contracts.codebase_inventory_scan import (
    scan_current_codebase_features,
)

record = scan_current_codebase_features(
    index, repository,
    expected_head=head,
    registry=registry,
    version_id=version_id,
    scheduler=scheduler,
)
reference = scan_current_codebase_features(
    index, repository,
    expected_head=head,
    registry=registry,
    version_id=version_id,
    scheduler=scheduler,
    optimized=False,
)
```

Supply an existing datasets `parent_lease` instead of `scheduler` when the
supervisor already holds admission. Both forms retain the native resource
owner. The operation reserves at least 1,024 MiB, defaults to a 120-second
overall deadline and uses a bounded numerical child with one CPU thread.
Cancellation combines the caller's signal with the native lease's signal.
Child accounting remains held until the bounded runner drains the process and
cleans its workspace. Memory enforcement samples process-tree RSS and may
overshoot; native Git capture has bounded individual commands and cooperative
deadline checks. This profile does not claim aggregate OS containment.

The default bounds are 256 entries, 64 KiB per source, sixteen shards of at most
sixteen rows, 4 MiB per restored native target, 8 MiB per serialized compact
shard, 32 MiB numerical input and 16 MiB output. The closed model-owner namespace
is limited to 4,096 rows and 8 MiB. Aggregate retained source and AST payloads
are each bounded to 16 MiB. Explicit smaller row, shard or input budgets retain
remaining entries as `deferred_budget`. Capture, evidence, registry and output
integrity failures refuse the operation. A failed operation returns no scan
record. Limits do not increase the capture profile or silently omit entries.

Every admitted entry has one disposition: `inferred`, `opaque`, `unindexed`,
`parse_failed`, `parse_partial`, `unsupported_target`, `feature_incompatible`
or `deferred_budget`. The ordered membership root binds native entry CIDs and
raw source keys. Each row retains exact path bytes, source and AST identities,
parse status, coverage, target digest and explicit frontiers. Registered cohort
members retain their original role and authored contracts. Additional files
are labelled `not_in_registered_cohort` with `none_declared` authored contracts;
they gain no training or held-out designation. Opaque entries can retain a
source CID whose bytes were never admitted into CAS by native capture; their
availability and missing SHA are explicit. Missing required nonopaque source
or AST CAS always refuses.

The optimized path prepares the fitted vocabulary and four private CPU
float64 weight tensors once, then reuses them across a finite shard sequence
in one isolated child. It also reuses verified ancestry material and shared
native manifest/publication membership within that operation. Opt-out calls
unchanged native lineage and target validation plus the v1 inference routine
for each shard in the same transport and resource envelope. Both independently
replay every restored native target. One shared manifest/publication envelope
replaces repeated copies during transport; reconstructed native targets retain
their original canonical bytes and digests. These are operation-scoped caches;
there is no cross-operation resident service or process.

The coordinator replays bounded ancestry once and reuses its selected payload.
It verifies complete historical source/AST inventories and native publication
receipts, with current-head evidence covered by the entry and final live
observations. Final checkpoints, ancestry rows, the closed registry namespace
and listed implementation files must agree with entry identities. The final
live observation follows the historical/model checks, and closing registry and
implementation checks precede the returned record. An additional ancestry row
and artifact check follows the final live observation. Coverage counts require
exact canonical JSON equality, including integer types. These checks are
point-in-time observations and do not lock a checkout against subsequent edits.
No fitting, registration, promotion or source execution occurs during scanning.

`record.artifact_cid` is CIDv1 over raw canonical finite native JSON. Numerical
floats remain native floats; the project's float-free DAG-JSON codec is used
for structured membership identity. `record.to_dict()` returns a detached copy.
The scanner computes immutable result bytes but publishes neither a CAS object
nor a mutable proof-index row. A caller must persist the exact bytes under that
raw CID and use a separately qualified publication/query boundary before making
them available to the supervisor.

The [native fixture](../../benchmarks/agent_supervisor/container_coding/qualify_codebase_inventory_scan.py)
uses a genuine `for_proof_host` sampler, fixed private root/child training,
more than sixteen supported files and explicit non-inferred entries. Its
current measurements and execution histories are retained in the
[inventory review](../architecture/repository_proof_index_and_codebase_ir.inventory_scan_review.json).
The [protocol/matrix controls](../../../ipfs_datasets/tests/unit/logic/software_contracts/test_codebase_inventory_scan.py)
are separate from actual native numerical and cancellation evidence.

The [earlier native run](../../../../artifacts/codebase_ir_terminal_bench/inventory-scan-qualification-20261002-04/result.json)
completed in 59.347 seconds, including three fixed setup training epochs,
five positive scans and thirteen refusal controls. All 28 admitted entries were
accounted for; 22 were inferred in shards of sixteen and six, with zero numerical
difference between optimized and opt-out float64 outputs. A four-row budget
retained eighteen deferred rows; a one-byte numerical input budget retained all
22 compatible rows as deferred and launched no child. A 973,786-byte input cap
inferred twenty rows and retained two deferred rows using a 936,843-byte request.
Scans introduced no fitting or owner mutation; after control restoration the
complete owner/artifact and checkpoint state matched baseline. All owned leases
and waiters drained. Eighty-four distinct protocol/matrix controls passed with
automatic pytest plugin loading disabled.

The default run built four weight tensors and one vocabulary, while the
two-shard reference built eight tensors and two vocabularies. Shared plus compact
target transport occupied 941,916 bytes, compared with 3,471,834 bytes for the
same native targets with repeated manifests. This is approximately 73% less
target transport. Both public calls used one isolated process, two live source
observations and one ancestry replay. The default scan took 6.153 seconds and
the opt-out scan 5.801 seconds through final fences, excluding record
serialization. This single small fixture demonstrates reduced repeated setup
and transport; it establishes no throughput improvement. Native target replay,
ancestry verification and numerical-library import remain substantial costs.

The refusal controls cover missing/tampered source and AST CAS, a missing
selected checkpoint, tampered selected/ancestor checkpoints, changed registry
metadata, source/control changes after a genuine worker return, pre-cancellation,
observed running-child cancellation and an inventory-cap refusal. The observed
child was cancelled before its numerical import completed: return code -15,
23 ms, cleaned workspace, no surviving observed PID and zero owned leases or
waiters. This qualifies running-process cleanup; cancellation during matrix
execution remains untested. Native setup uses a same-head root and child.
Different historical-head CAS mutation, the additional native parser-failure
envelope, cross-operation reuse, capacity-pressure refusal and crash recovery
retain separate qualification gates.

Selected original and ordinary detached source copies of 21 producer/harness/
test files agreed at the end. This sequential source closure is diagnostic,
without execution attestation. Mutation controls use the new isolated fixture
and restore its exact bytes/control state; they do not mutate prior receipts or
claim detached fixture copies. The earlier successful generation remains
historical; a second run correctly refused changed model metadata but failed a
harness error-message expectation. A regression control also reproduced numeric
`0` being accepted as an authority flag in manually constructed records. The
final constructor requires exact boolean `False`; its 84 controls and that
native run pass. Earlier native outputs already emitted exact booleans. Across
all four distinct runs, twelve setup epochs and 192.456 recorded seconds are
retained, with no unknown fitting.

The separate [read-only archive audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-scan-readonly-audit-20261002-04.json)
passed 996 substantive assertions and 8,345 structured-key checks, using only
bounded regular-file reads. It recomputed numerical raw CIDs and checkpoint,
source/AST, target/publication and owner identities, and confirmed all 21 selected
files matched retained copies at that review. Subsequent inference reuse changes
have their own source generation and qualification. The
[retained stdlib reader](../../../../artifacts/codebase_ir_terminal_bench/inventory-scan-readonly-audit-20261002-04.py)
opens no native owner and runs no scanner, inference, fitting or checker.
Archive JSON files include a final newline; numerical record CIDs bind the
canonical payload before that archive newline. File SHA pins and raw record
CIDs describe those distinct byte sequences.

This contributes partial RPI-029/032 coverage. It does not close any of the 32
production acceptance tasks. The scanner reconstructs native structural
features; formal logic extraction, source correspondence and independent proof
checking retain their existing owners. No learned semantic decoder, CUDA
384D implementation, unrestricted language support, remote federation,
gradient collective, proof reuse, planner authority or production default
activation is qualified by this increment.

Next join these records to the existing normalized conditional-evidence
`CodebaseVerificationCatalog.query_current` API, preserving canonical-key membership, reverse dependencies,
source/evidence-bound cursors and unknown runtime requirements. Qualify complete
inventory-to-proof query consumption, then persist or deterministically rebuild
the derived scan projection. Larger captures need resumable globally sealed
shards, cross-shard dependency resolution and crash/restart qualification.
Operation-scoped ancestry and native replay reuse are described in the
[paired inference qualification](codebase_inventory_inference_reuse_qualification.md).
Bounded Git batching and larger populations need separate measurements before
a persistent inference service is introduced.
