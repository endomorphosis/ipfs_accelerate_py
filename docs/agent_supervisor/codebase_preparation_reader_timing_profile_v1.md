# Bounded reader subphase timing profile

The authored 300-entry / 299-Python initial scan and complete one-edit successor passed in four isolated processes. Equal-default baseline and instrumented receivers match the independent normal reference in complete manifests, semantic state, all 299 projections, 13 native table families, histories, heads and operation receipts.

Production sources and the existing focused CI source remain unchanged. All 73 preceding controls remain historical evidence; this profiling milestone does not replay those controls or the ordinary package-wide CI entry point.

Actual owner receipt: [receipt.json](../../../../artifacts/codebase_ir_terminal_bench/preparation-reader-profile-checks-20261013-01/receipt.json), 415,447 bytes; SHA256 `d4244ad650107f5f38f1caf62c280dce2ac1c87e442784510c152736adeadd81`. Campaign 522.543074 seconds.

The baseline retains the existing scalar work and semantic frontend observers. The profiled route adds the selected timing wrappers. Every one of the 16 independent reader, JSON, cache, projection and rebuild counters matches between receivers in each phase.

| Phase | Baseline seconds | Profiled seconds | Profiled minus baseline seconds | Difference percent |
| --- | ---: | ---: | ---: | ---: |
| initial | 100.307907 | 104.227422 | +3.919515 | +3.9075 |
| successor | 119.405450 | 126.694451 | +7.289001 | +6.1044 |

Combined phases: baseline 219.713358s; profiled 230.921873s; signed difference +11.208516s. This is one sequential pair on this host. The difference includes host variation and is an observation of the incremental instrumented run, not a causal estimate of observer overhead or a general speedup. Total overhead against an observer-free workload was not measured.

| Phase | Selected scope category | Calls | Inclusive seconds | Exclusive seconds |
| --- | --- | ---: | ---: | ---: |
| initial | Prior projection serialization components | 4,576 | 1.815183 | 0.597276 |
| initial | Reader internal subphases | 51,524 | 40.325467 | 39.865348 |
| initial | AST and extraction context verification | 66,418 | 124.050299 | 23.039821 |
| initial | Implementation source hashing | 58,322 | 1.737290 | 1.704058 |
| initial | Source, head, custody and CAS fences | 39,122 | 60.239926 | 12.350173 |
| initial | Semantic assembly | 901 | 11.627394 | 7.612279 |
| initial | Catalog and SQL publication | 2,480 | 15.894875 | 6.622596 |
| initial | Remaining preparation orchestration | 16 | 97.525025 | 9.378391 |
| successor | Prior projection serialization components | 5,472 | 2.160453 | 0.736564 |
| successor | Reader internal subphases | 58,607 | 52.740520 | 52.245300 |
| successor | AST and extraction context verification | 73,363 | 144.502543 | 26.294768 |
| successor | Implementation source hashing | 79,160 | 2.400212 | 2.354788 |
| successor | Source, head, custody and CAS fences | 56,677 | 80.201573 | 14.516943 |
| successor | Semantic assembly | 901 | 10.890893 | 7.262418 |
| successor | Catalog and SQL publication | 2,526 | 17.172038 | 7.423334 |
| successor | Remaining preparation orchestration | 22 | 125.535976 | 15.194979 |

Category scopes are disjoint and cover the 58 selected Python call scopes. Inclusive totals contain nested calls and overlap across categories. Exclusive time subtracts selected descendants only; uninstrumented work and some observer bookkeeping remain in that attribution. The sum of selected exclusive times is bounded by the actual phase wall interval. Call totals count selected method invocations, not transactions or native SQL statements.

| Phase | Prior encode calls | Prior wire bytes | Implementation bytes hashed | CAS physical bytes | CAS read calls | Semantic scan calls | Catalog publish calls | Store batch calls | Persistence AST rows offered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 2,288 | 44,569,515 | 2,210,919,260 | 161,484,575 | 13,136 | 1 | 1 | 1 | 299 |
| successor | 2,736 | 53,301,224 | 2,997,972,530 | 164,075,203 | 15,448 | 1 | 1 | 1 | 299 |

| Phase | Reader component | Calls | Inclusive seconds | Exclusive seconds |
| --- | --- | ---: | ---: | ---: |
| initial | Reader total; exclusive column is residual | 13,136 | 41.352846 | 1.487499 |
| initial | Reader checkpoints | 26,272 | 36.898031 | 36.898031 |
| initial | Artifact canonical wire | 4,252 | 0.540416 | 0.540416 |
| initial | Structured CID including its canonical wire | 4,252 | 0.747697 | 0.287577 |
| initial | Canonical wire inside structured CID | 4,252 | 0.460119 | 0.460119 |
| initial | Raw source CID | 5,272 | 0.233740 | 0.233740 |
| initial | Cache freeze admission | 3,612 | 1.114563 | 1.114563 |
| initial | Outermost cache thaw | 3,612 | 0.330900 | 0.330900 |
| successor | Reader total; exclusive column is residual | 15,448 | 54.059274 | 1.813973 |
| successor | Reader checkpoints | 30,896 | 48.874663 | 48.874663 |
| successor | Artifact canonical wire | 4,315 | 0.645464 | 0.645464 |
| successor | Structured CID including its canonical wire | 4,315 | 0.775089 | 0.279870 |
| successor | Canonical wire inside structured CID | 4,315 | 0.495219 | 0.495219 |
| successor | Raw source CID | 7,500 | 0.349248 | 0.349248 |
| successor | Cache freeze admission | 3,633 | 1.257629 | 1.257629 |
| successor | Outermost cache thaw | 3,633 | 0.343208 | 0.343208 |

Reader total inclusive time equals the seven reader subphase exclusive times plus the parent exclusive residual. Structured CID inclusive time contains its nested canonical-wire observation; inclusive columns must not be added. The residual contains physical reads, file custody/path/CID prechecks, JSON parsing, other uninstrumented reader work and observer bookkeeping. It is not a direct disk-I/O measurement.

| Phase | Source bytes hashed | Fresh artifact wire bytes | Structured CID canonical bytes | Cache freezes returned | Cache freezes raised and fell back | Outermost cache thaws |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 239,814 | 85,289,947 | 85,289,947 | 3,612 | 0 | 3,612 |
| successor | 340,825 | 87,401,260 | 87,401,260 | 3,633 | 0 | 3,633 |

Reader subphase wrappers admit only the exact captured direct callers under an active Reader.read. The two checkpoint calls per successful read remain intact. Fresh artifact wire/CID calls match the independent structured JSON decode count; source CID calls match source reads, and outermost cache thaws match cache hits. Source hashing bytes plus fresh canonical artifact bytes are bounded by actual physical read bytes; cache hits still reread complete leaves. A bounded cache-freeze refusal may fall back to the unchanged ordinary path. No extra reads, serialization or traversal is added by these observations.


Prior-projection observations measure the existing encode and canonical-wire components inside the captured prepare_page sum generator. They exclude generator iteration and the complete sum operation. Implementation hash bytes and CAS read bytes come from actual existing arguments/results/counters; wrappers add no source or artifact reads and no serialization calls. The full timing documents declare byte applicability separately for each scope. Persistence row counters describe method input rows and do not attest durable commit counts; physical database I/O and native execute-call counts were not observed.

Both receivers independently admit the same 128-source parser-cache handoff, parse the remaining 171 Python sources and the single edited successor source, and retain semantic syntax parsing. Every runtime optimization remains at its qualified default. Source/context/head/CID/custody checks, physical byte charging, aggregate bounds, cancellation and checkpoints remain active. No private byte authority, memo or new sizing optimization is introduced. This milestone instruments existing work; it implements no inference or scan optimization.

All child cleanup completed, own reservations released and MAIN returned idle. Numerical 8D/384D/768D/4096D Leanstral heads, CUDA, inference, training, federation, signed-worker dispatch and formal proof are outside this structural profiling qualification. No database binary is retained or independently reopened.

The accompanying source observation joins the same two production files and existing focused CI source to their historical frozen bytes and current held identities. The separate finite integrity closer has not run when this report is written; its [receipt](../../../../artifacts/codebase_ir_terminal_bench/preparation-reader-profile-final-close-20261013-01.json) is excluded from its own input closure.
