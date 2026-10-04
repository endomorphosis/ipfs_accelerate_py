# Operation-scoped CodebaseIR inference reuse

`scan_current_codebase_features` enables the reuse profile by default. Set
`optimized=False` to retain native reference ancestry/target replay and per-shard
v1 numerical inference. Capture, complete inventory dispositions, model identity,
source and registry fences, CPU float64 arithmetic and the resource owner are
shared by both paths. The [scanner contract](codebase_inventory_scan_qualification.md)
describes the finite population and transport limits.

The coordinator reads each native candidate once per operation and reuses that
verified immutable snapshot for parent-state and lineage checks. Each distinct
complete canonical target still passes native validation; exact target bytes,
rather than a digest alone, identify a reusable result. Binding extraction and
registered current-cohort preparation then reuse that result. Unchanged native
diagnostics, runtime construction, completed-run metadata and all inherited
lineage conditions remain required. Reviewed reference-file hashes reject an
unsupported producer generation. Checkpoint-pinned training and runtime files
remain unchanged.

The receiving child independently validates the shared native manifest and
publication receipt once and prepares immutable exact membership lookups. Every
restored target still checks captured source bytes and CID, AST identity and
provenance, authored contracts, native lowering, correspondence/frontiers and
full rebuilt canonical equality. A valid transport checksum does not substitute
for this replay. Reuse does not cross an operation or grant proof authority.

The ancestry cache has a 40 MiB serialized-byte cap under default lineage
limits. At most one separately bounded historical seal is retained, then released
before worker streams are allocated. The larger retained serialized phase is
236 MiB under default scanner limits, within one quarter of the 1,024 MiB
reservation. These are serialized-byte accounting bounds, not Python RSS or
kernel containment. A lineage exceeding the cache cap is refused explicitly;
the reference path remains available. Final live source observations and closing
ancestry row, artifact, registry and implementation checks remain mandatory.
These are sequential point-in-time fences rather than an atomic checkout lock.

Review also tightened two receiving/final boundaries. Coverage equality now
distinguishes integer counts from boolean or float aliases. Artifact verification
runs again after the final live source observation, so a checkpoint changed
during that observation refuses the operation.

The [paired native harness](../../benchmarks/agent_supervisor/container_coding/qualify_codebase_inventory_inference_reuse.py)
uses one fixed root/child model with three setup epochs. Four unprofiled public
calls follow optimized/reference/reference/optimized order. Two additional
profile probes count actual parent-side native calls and are excluded from the
timing comparison. All calls preserve owner, checkpoint and numerical state;
the fixed inventory contains 28 entries and 22 inferred rows in two shards.
The separate unit suites cover replay corruption, inherited algorithm parity,
cache bounds/mutation and strict response types. Pure cache stubs do not qualify
native execution.

The [native result](../../../../artifacts/codebase_ir_terminal_bench/inventory-inference-reuse-qualification-20261002-01/result.json)
qualified in 88.353 recorded seconds, including the three setup epochs, four
measurements, two profile probes and fifteen refusal controls. The fitted model
has eight latent dimensions and 53 feature columns. All 27 selected
producer, harness and test files matched their retained copies. Complete model
SQL/source-head exports and source/model artifact populations matched baseline,
and model state, Adam steps and completed epochs remained unchanged. Owned
leases and waiters drained. The four focused suites passed 205 distinct tests.

| Public call, including serialization and final fences | Optimized | Reference |
| --- | ---: | ---: |
| First unprofiled sample | 3.685 s | 5.965 s |
| Second unprofiled sample | 3.841 s | 6.135 s |
| Median of two samples | 3.763 s | 6.050 s |
| Coordinator ancestry replay, median | 0.347 s | 1.859 s |
| Child native target replay, median | 0.454 s | 0.842 s |

The reference median was 1.608 times the optimized median on this fixed CPU
fixture, corresponding to 37.8% less public-call latency. All 28 dispositions and 22 inferred rows matched, including
exact float64 numerical output. Two samples per mode establish neither
statistical significance nor a generalized production speedup. The profiled
calls are excluded from these medians; their tracer overhead was substantial.
Shared replay preparation added approximately 0.016 seconds per optimized child.
Numerical-library import still cost approximately 0.864 seconds, so batching
across more work may help without a persistent cross-operation cache.

| Actual parent calls in separate profile probes | Optimized | Reference |
| --- | ---: | ---: |
| Native full target validator | 3 | 41 |
| Native target lowering | 26 | 70 |
| Native source-binding replay | 0 | 33 |
| Runtime candidate reader | 2 | 5 |
| Manifest reads | 2 | 8 |
| Publication receipt reads | 6 | 12 |

Parent counters include the complete public scan, not child calls. The optimized
ancestry retained 2,858,289 serialized cache bytes, recorded 41 validation cache
hits on three distinct targets and six preparation cache hits, and constructed
both native runtimes. The receiving child
still performed 22 source checks, AST checks, contract replays, native lowerings
and full canonical comparisons. It avoided 22 repeated manifest parses and 22
receipt parses. Weight/vocabulary construction remained four/one versus
eight/two for the two-shard reference.

Four controls rehashed corrupted shared context, projections, authority and
source bytes, then launched the actual receiving child. Native validation
refused all four; process workspaces were cleaned. Source/AST CAS and selected/
ancestor checkpoints also refused tampering. Late source, registry, artifact
and catalog faults after genuine worker return were rejected. A checkpoint
changed after the second successful source observation was rejected by the new
closing artifact check. Pre-cancellation and cancellation after observing an
actual child PID both refused; the latter returned -15 in 23 ms, cleaned its
workspace, left no observed child PID and drained resource accounting. This
does not qualify cancellation during matrix execution.

The earlier four inventory qualification generations remain unchanged and
historical. Their 192.456 recorded seconds and twelve setup epochs, plus this
88.353-second run and three epochs, total 280.808 recorded seconds and fifteen
setup epochs across these five distinct attempts. There was no failed native
attempt or unknown fitting in this reuse increment; the earlier failed harness
generation remains failed in its retained history.

The independent [read-only archive audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-inference-reuse-readonly-audit-20261002-01.json)
passed 4,713 substantive assertions and 8,345 structured-key checks. Its
[stdlib reader](../../../../artifacts/codebase_ir_terminal_bench/inventory-inference-reuse-readonly-audit-20261002-01.py)
used 645 bounded reads over 209 unique paths, recomputed raw scan CIDs, source/
AST and checkpoint identities, all six record comparisons and timing summaries,
and confirmed all 27 current selected files matched retained copies. The primary
archive's bytes, modes, timestamps, namespace and inert source-alias symlink were
preserved. It opened no native owner and invoked no scanner, checker, fitting or
test. Export equality is scoped to recorded model SQL rows, source head and
artifact populations; this reader does not replay a DuckDB owner or attest native
execution. The [machine review](../architecture/repository_proof_index_and_codebase_ir.inference_reuse_review.json)
retains source/result pins and the distinct execution history.

Reproduce from the accelerator checkout with sibling datasets/kit imports and
the qualified native DuckDB/Quack environment, using a fresh output directory:

```sh
python -m benchmarks.agent_supervisor.container_coding.qualify_codebase_inventory_inference_reuse /tmp/new-inventory-inference-reuse
```

This profile remains bounded structural inference, with no CUDA, semantic
decoding, proof reuse, planning authority or production activation claim. The
setup uses a same-head root and child; different historical-head transitions,
capacity pressure, crash recovery, larger populations and cross-operation reuse
retain separate gates. It adds partial RPI-029/032 evidence and closes no
production acceptance task.

Next join immutable scan rows to the existing normalized
[`CodebaseVerificationCatalog.query_current`](../../../ipfs_datasets/ipfs_datasets_py/duckdb_control/codebase_verification_catalog.py)
and supervisor consumer, using exact source/head and canonical evidence keys.
That independently qualified [query route](../../../ipfs_datasets/workspace/codebase-evidence-query-qualification-20261002/REPORT.md)
already provides reverse dependencies, sealed inventories, monotonic evidence
epochs and root-bound pagination. A structural feature row can nominate a unit;
it cannot supply a checked property or satisfy a runtime requirement. Qualify
the complete inventory-to-query-to-planning material join before activating it,
then measure bounded Git batching and resumable larger captures.
