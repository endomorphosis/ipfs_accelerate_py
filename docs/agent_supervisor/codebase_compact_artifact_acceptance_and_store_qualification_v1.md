# Compact artifact acceptance and local store qualification

**Selected checks passed; live integration remains pending.** Separate owned jobs completed 39 acceptance/replay cases and 37 store cases without failures. Candidates remain unapplied; general `qualified`, candidate-qualification and production flags remain False. This extends the [paired full-300 qualification](codebase_full300_paired_shared_successor_qualification_v1.md) without borrowing its receiving credit. The [final finite integrity closure](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-final-close-20261007-01.json) records later bounded byte checks of selected evidence and report links; it adds no runtime replay or execution authority.

| Completed job | Exact population | Owned elapsed / bounded child |
| --- | --- | --- |
| [Acceptance/replay04](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-parity-owned-checks-20261007-04/receipt.json) | 31 serialized-record cases plus eight historical parent cases | 7.958 s / 7.822 s |
| [Store01](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-store-owned-checks-20261007-01/receipt.json) | 35 synthetic controls plus two retained-artifact probes | 4.966 s / 4.860 s |

Each job used existing MAIN: 2 CPU slots, 2048 MiB, one child, a 180-second deadline, 120-second lease TTL and 30-second heartbeat. Both closed unchanged source/output pins under the reservation, reaped the child, removed its workspace, released the lease and left MAIN idle. Observations are sequential; continuous currentness and kernel containment are not attested.

## Acceptance agreement and historical signatures

The [sealed acceptance candidate03](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-acceptance-parity-candidate-20261007-03/plan.json) calls real frozen native constructors and the full chain binder, without AST-slicing. Default/reference baselines retain 300 members, 294 rows and 588 projections. Four positives, 23 shared refusals and four declared restrictions matched expectations:

| Declared restriction | Observed boundary |
| --- | --- |
| Finite integer `2**129` | Native numerical records accept it; candidate's 128-bit JSON limit refuses it |
| Lone-surrogate RAW text | Native ASCII-escaped worker-profile record accepts it; candidate's strict UTF-8 string framing refuses it |
| Root inventory limit 513 | Native root@1 accepts the standalone root; shared full binding refuses its shared512 limit, while candidate refuses earlier |
| 33 one-entry opaque pages | Complete native closure accepts 35 records; candidate's 32-record ceiling refuses it |

Aliases deduplicate. A coherently rehashed scalar change also passes serialized validation: this establishes neither inference correctness nor source behavior. `native_record_validation_equivalence_qualified` remains False.

Historical replay preserved the V6 `{manifest, graph, receipt}` parent and V4 receipt, verified both original Ed25519 signatures and recomputed complete planning for **300 sources, two tasks and one goal**. Seven negatives refused. Independently pinned public owner identifiers required no private profile/key. Native ASCII signing and UTF-8 DAG-JSON artifact preimages remain distinct. Historical verification grants no current-owner capability.

The job imported and fenced **138 chosen implementation source files**. Native validator/signature/planning calls were real; repository-target imports, model-artifact/database reads, network/subprocess calls, inference and training were zero. No current source/model receiving occurred. Third-party libraries remain existing environment provenance, outside the frozen project-source closure.

## Actual local publication and recovery

The [sealed owner-store candidate](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-owner-store-candidate-20261007-01/README.md) published **25 unique retained artifacts, 2,362,315 bytes**, preserving roles, bytes, CIDv1 and SHA256. Numerical chunks/pages/completion remain RAW ASCII JSON; roots/parent remain float-free UTF-8 DAG-JSON.

| Complete retained closure | Members | Charged bytes |
| --- | ---: | ---: |
| Historical parent plus default scan | 13 | 1,494,217 |
| Historical parent plus reference scan | 13 | 1,494,240 |

Both expected pins reclosed after reopening; parent bytes deduplicated and the temporary store was removed. Controls covered held `NOFOLLOW` directories, UID/mode/single-link and replacement checks, input snapshots, full-closure precharge, fsync/lock failures and conservative retention. Recovery checked recorded consistency, without signatures or new receiving.

Limits remain provisional: 8 MiB/parent, 4 MiB/root, 16 MiB/RAW artifact, 32 members/32 MiB/closure, 64 pins and 64 MiB repeated recovery charge. Unreferenced-blob quota, automatic GC and crash repair are absent. A link/unlink crash can leave two links requiring external repair. Expected pin IDs need independent durability; undeclared deleted pins cannot be detected. Fsync success is not hardware crash attestation.

## Retained failures and next gate

Failed runs [01](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-parity-owned-checks-20261007-01/receipt.json)/[02](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-parity-owned-checks-20261007-02/receipt.json) lacked `multiformats`: the bounded interpreter resolves symlinks independently of virtualenv activation. The successful plan supplies the local library site path. Run [03](../../../../artifacts/codebase_ir_terminal_bench/compact-artifact-parity-owned-checks-20261007-03/receipt.json) exposed stale mutation-fixture descriptor bytes/SHA. Only new candidate03 repaired transitive accounting; executed candidates and core code stayed unchanged. Detailed diagnostics now persist.

Next: pure V8 compact context/pending replay preserving separate immutable legacy parent/completion and current shared selection/delta bindings; then actual admission, offline owner-store resolution, worker mirror/Intent transactions, reservation, construction, START, authored checks, completion and STOP. V8 signatures, current receiving, worker/Intent activation, proof, CUDA, semantic 8D/384D/768D/4096D qualification and production activation remain pending.
