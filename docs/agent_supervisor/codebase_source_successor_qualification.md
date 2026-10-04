# Source successor models and planning

The new [successor scan coordinator](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_successor_model.py)
and [planning adapter](../../ipfs_accelerate_py/agent_supervisor/planning/codebase_source_delta_context.py)
connect a complete source transition to explicit private model selection and
current model-off planning. Training remains a separate caller action. The
coordinator creates a fresh ordinary resumable scan root; the planner consumes
the complete source ledger without creating behavioral facts or dropping tasks.

This increment extends the [source delta and signed runtime qualification](codebase_source_delta_and_runtime_qualification.md).
Its source, scanner and six checkpoint producers retain their earlier byte
identities. The new modules have separate implementation pins and immutable
protocols. All 32 production acceptance tasks remain open.

## Explicit source and model progression

The caller publishes the changed repository with `prepare_current`, then builds
the already qualified complete source delta. If the existing training contract
remains suitable, the caller explicitly invokes `train_current_codebase_features`
with the new head and the previous model's `parent_version_id`. The training
implementation keeps its existing fixed evaluation cohort, feature basis,
contract, ancestry replay and saved Adam state rules.

`start_current_codebase_successor_scan` accepts that already trained direct
child, the exact previous model and the source delta. It requires the parent's
full provenance head to equal the previous publication, the child's to equal
the current publication, and the registered parent pointer to match. It never
trains a fallback lineage, performs forward inference or promotes a model head.
The selected paths, contracts and train/tune/canary roles stay fixed. Tune and
canary raw content and normalized projections remain unchanged, along with the
saved learning rate, vocabulary, numerical basis and original root replay.
Changed training source may introduce unknown atoms, which remain explicit
advisory coverage; newly added unselected paths are not automatically trained.
Selected-path deletion or rename, evaluation/cohort change, a new numerical
basis or ancestry-cap exhaustion requires the caller to declare a fresh root
lineage and use the ordinary scan starter. This direct-child coordinator
refuses that case. An old model cannot be selected for a new head.

The canonical `codebase-inventory-successor-scan@1` record binds both heads,
both membership identities, both published training records, both complete
model descriptors and a fresh `root_cid`. The root uses the unchanged ordinary
resume protocol. Previous pages cannot become pages of the new root. Selection
does not confer numerical reuse or proof authority; all fourteen authority
flags and all training/inference/promotion flags remain false.

Every selection and receiving call verifies native lineage and checkpoint
bytes, registry namespace and ownership, publication history, complete source
membership, immutable CAS and current checkout. Closing checks put source CAS
callbacks before the final model fence, then repeat native SQL and live source
observations without further CAS callbacks. Pure immutable-input guards follow
those observations. These are sequential checks, not an atomic filesystem and
model snapshot or execution attestation.

The coordinator defaults to `optimized=True`; `optimized=False` retains the
reference replay. Both keep the existing shared resource lease, cancellation,
120-second default deadline and explicit maximum of 600 seconds. Its 256-KiB
selection limit and serialized retention estimates do not bound process RSS.

## Complete source metadata in planning

`preview_current_source_delta_plan` accepts an independently authored native
request, typed intent, producers, complete task population and frozen goal.
It adds a reserved `codebase_source_delta_advisory` field with every raw source
key, classification, old/current structural member and comparison. The full
heads, membership, capture policy, coverage, delta CID and authority fields
enter the native semantic material identity.

The adapter calls the existing `preview_repository_plan` and actual
`PlanCreateService`. Fresh native source and complete policy roots are checked
at service boundaries and again before return. Closing plain-state guards
reject material changed during the final source receiver, even if the source
itself remains valid. The result preserves every declared requirement and task
as a residual runtime obligation. It supplies zero current facts, performs no
training or inference and authorizes no worker launch or signed admission.

Source-property solver calls remain zero. The native symbolic planner may use
its own solver; this is not a claim that the whole preview is solver-free.
References and results are body-free and bounded to 4 MiB. A delta too large
for that envelope is refused rather than truncated. The adapter has a shared
120-second default deadline, explicitly bounded to 600 seconds; the existing
repository preview retains its separate 90-second maximum within that budget.

## Validation and recorded cost

The [53 successor-model tests](../../../../artifacts/codebase_ir_terminal_bench/inventory-model-successor-tests-20261003-01.receipt.json)
pass with no failures, errors or skips in 35.705 JUnit seconds. They comprise
24 tiny native-owner controls and 29 pure protocol controls. The fixture
explicitly trains one root epoch and one child epoch. Coordinator and receiver
calls prohibit fitting and forward inference; two ordinary pages separately
exercise actual CPU 8D inference. Closing CAS/model/registry/source mutation,
stale parent, rehashed record mutation, owner replacement and cancellation are
covered.

The [35 native planning tests](../../../../artifacts/codebase_ir_terminal_bench/source-successor-planning-tests-20261003-03/receipt.json)
pass with no failures, errors or skips in 55.916 JUnit seconds. They exercise
actual private source DuckDB/CAS and native planning stages. Two final-CAS
callbacks alter previously consumed authored or bound material while preserving
the source and head; both are refused after actual planning completes.
Controlled cancellation and clock advancement test resource behavior separately
from integrity. Every fixture releases its native leases and waiters.

The initial planning test attempt is preserved: 23 cases passed and ten failed
because the test looked for scheduler counters at the wrong telemetry level.
It cost 50.853 JUnit seconds. The corrected twelve-case targeted run cost
25.932 seconds; the final complete run above is the qualification receipt.
The [88 final cases](../../../../artifacts/codebase_ir_terminal_bench/source-successor-test-dedup-20261003-01.json)
have no overlap with the prior 912 distinct recorded cases.
These are separate source-generation receipts, not a rerun of all 1,000 cases
against this new implementation.

The [first fresh 300-entry attempt](../../../../artifacts/codebase_ir_terminal_bench/source-successor-qualification-20261003-01/result.json)
is an overall failure preserved with its producer copies. It cost 394.949
seconds and completed two explicit setup epochs. Default successor selection
passed in 80.190 seconds; the reference route exceeded its 120-second deadline
and refused after 125.638 seconds. It created no inference pages and ended
with zero active leases and waiters. This resource refusal is not an integrity
failure or successful reference comparison.

The [second fresh native qualification](../../../../artifacts/codebase_ir_terminal_bench/source-successor-qualification-20261003-02/result.json)
passes in 809.916 seconds. It authors another complete 300-entry repository,
publishes both source heads and explicitly trains a one-epoch root and a
one-epoch direct child against their respective heads. Both model/Adam states
and the full parent checkpoint remain unchanged after setup. The source union
contains 302 keys: 297 retained, one changed, two added and two removed.

| Operation | Default seconds | Reference seconds |
| --- | ---: | ---: |
| Fresh successor selection | 75.909 | 146.545 |
| Fresh 32-entry inference prefix | 53.177 | 102.745 |

The reference selection uses an explicit 600-second qualification allowance;
the optimized selection retains the unchanged 120-second default. Each page
contains exactly 22 inferred rows, nine budget-deferred entries and one
unindexed entry. Entries, coverage and numerical output match exactly. These
are timings on this authored CPU 8D fixture, not a general throughput claim.
Both complete roots describe all 300 members; neither prefix completes the
300-member scan. No historical pages or setup epochs are reused.

The complete ledger passes through actual native planning in 82.766 seconds.
Both authored requirements and both tasks remain residual, with zero current
facts and zero task omissions. Full typed semantic field preimages are
explicitly reconstructed and checked against the consumed native material
binding. This export supports independent meaning/digest verification without
instrumenting or changing the production planner.

Closing and reopening the source and model owners once in the same process
preserves every logical owner field except the expected registry generation
increment. Cold successor receiving takes 34.930 seconds, page receiving
31.506 seconds and native planning 84.762 seconds. Cold planning returns the
identical full result and consumed input identity. Its separate preimage export
reuses the warm reconstructed preimages against that identical cold binding;
it does not claim an independent cold projection capture or process restart.
Selection and cold receiving prohibit inference, and every post-setup operation
prohibits fitting. The native old-model/new-head refusal and pre-cancelled
receiver pass; all leases and waiters drain.

Across both native attempts, four setup epochs were explicitly completed. The
tiny model test fixture's two epochs are recorded separately. The failed first
attempt, its two epochs and its 394.949-second cost remain historical rather
than becoming inherited setup for the successful run.

## Independent closed review

The [final closed audit](../../../../artifacts/codebase_ir_terminal_bench/source-successor-qualification-20261003-02-readonly-audit-20261003-01.json)
passes with no errors and preserves both full archive inventories. It reads
90,011,079 guarded artifact bytes in 4.723 seconds; the archive contains 1,593
regular files totaling 75,434,026 bytes. It opens no native owners, SQL, Git,
models or solvers and performs no writes to the primary archive.

The independent reader reconstructs source membership and both publication
receipts, complete source/AST CAS byte joins, both selected model fingerprints,
parent-state continuation, fixed evaluation/replay cohorts, training-source
bindings, request hashes, native version IDs and completed-run lease/command/
event records. It joins both fresh page schemas and worker limits, exact
default/reference output and coverage, every typed planning field preimage,
predicate meaning, producer effect, task obligation, frozen policy, native
snapshot/receipt/stage identity, warm/cold owner snapshots and resource/cost
records. All 34 selected local producer files match retained copies.

The [93 standalone reader controls](../../../../artifacts/codebase_ir_terminal_bench/source-successor-reader-guards-20261003-01/receipt.json)
pass in 32.839 seconds with zero failures, errors or skips. They include
coherently rehashed model/cohort and planning-meaning mutations, native registry
joins, page schemas and filesystem read guards. They remain separate from the
1,000 historical product test cases. The final reader is exactly the tested
118,630-byte source with SHA-256
`ada264cb162b4d20751e36c80d5504348a5439b05d04ad2469ee41450ea0bf14`.

This checks stored intrinsic joins and retained native observations. It does
not replay the optimizer, execute numerical inference, independently lower
source programs or attest process origin/current deployment. The immutable
archive cannot replace a fresh native receiver for later current eligibility.
One failed draft reader assumption about checkpoint provenance was corrected;
its report/source copy, the subsequent expected failed-native audit and the
earlier narrower successful draft audit are all retained. The final audit uses
the stronger, separately tested reader generation.

The [machine review](../architecture/repository_proof_index_and_codebase_ir.source_successor_review.json)
binds exact audit/result/test/reader generations, selected source bytes,
historical failures, measured costs and remaining gates. Its builder and the
append-only closed state support resuming the next increment without modifying
these archives.

## Remaining integration and speed work

Successor creation currently replays a full lineage three times. A separately
qualified optimization can retain one completely validated chain within that
single operation, construct the exact unchanged root protocol and retain final
checkpoint, registry, source and CAS fences. The public receiver must perform
fresh replay on every later use; no currentness token or cross-call numerical
cache can replace those checks.

Clean Git acquisition still issues separate size and body processes for each
captured blob. A bounded batch profile should preserve size-before-body checks,
full entry/CID equivalence, cancellation, aggregate deadlines and final
HEAD/index/worktree fences. Captures beyond the current 1,024-member source
profile require a complete global root and staged publication; inference paging
alone does not solve acquisition or memory scaling.

The new source-advisory preview is not yet a signed successor execution
profile. Fresh signed baseline and evidence admission, full successor scan
completion, worker installation and publication remain separate acceptance
work. Proofs remain keyed to complete heads and their original dependency
closures; an empty invalidation plan cannot authorize inherited proof reuse.
CUDA and 384D CodebaseIR, learned intent alignment, DuckLake sharding and
on-the-fly training policy retain their existing qualification gates.
