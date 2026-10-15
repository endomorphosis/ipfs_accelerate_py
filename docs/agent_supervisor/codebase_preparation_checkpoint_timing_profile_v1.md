# Bounded checkpoint subphase timing profile

The authored 300-entry / 299-Python initial scan and complete one-edit successor passed in four isolated processes. Equal-default baseline and instrumented receivers match the independent normal reference in complete manifests, semantic state, all 299 projections, 13 native table families, histories, heads and operation receipts.

Production sources and the existing focused CI source remain unchanged. All 73 preceding controls remain historical evidence; this profiling milestone does not replay those controls or the ordinary package-wide CI entry point.

The first checkpoint profiling attempt stopped on timing-observer count validation after three completed children. Its reference route did not run; the baseline/profiled initial and successor publications remain zero-credit diagnostic evidence. All three actual tools and child observations confirm cleanup. The failed attempt diagnostic and all 13 actual outputs, including its receipt, are retained by the separate finite close. The measurements below describe the completed four-role retry only.

Actual owner receipt: [receipt.json](../../../../artifacts/codebase_ir_terminal_bench/preparation-checkpoint-profile-checks-20261013-02/receipt.json), 431,480 bytes; SHA256 `33502113af416031dfcdf407e3c9fa2a82c948e8b0f9ddb611b26e166cebd254`. Campaign 551.318191 seconds.

The baseline retains the existing scalar work and semantic frontend observers. The profiled route adds the selected timing wrappers. Every one of the 16 independent reader, JSON, cache, projection and rebuild counters matches between receivers in each phase.

| Phase | Baseline seconds | Profiled seconds | Profiled minus baseline seconds | Difference percent |
| --- | ---: | ---: | ---: | ---: |
| initial | 106.658392 | 109.267915 | +2.609523 | +2.4466 |
| successor | 122.914422 | 135.667118 | +12.752695 | +10.3753 |

Combined phases: baseline 229.572814s; profiled 244.935033s; signed difference +15.362219s. This is one sequential pair on this host. The difference includes host variation and is an observation of the incremental instrumented run, not a causal estimate of observer overhead or a general speedup. Total overhead against an observer-free workload was not measured.

| Phase | Selected scope category | Calls | Inclusive seconds | Exclusive seconds |
| --- | --- | ---: | ---: | ---: |
| initial | Prior projection serialization components | 4,576 | 1.807192 | 0.602690 |
| initial | Reader internal subphases | 51,524 | 43.066621 | 3.030994 |
| initial | Scheduler checkpoint subphases | 210,160 | 79.113745 | 39.545073 |
| initial | AST and extraction context verification | 66,418 | 131.814486 | 24.084058 |
| initial | Implementation source hashing | 58,322 | 1.763990 | 1.730849 |
| initial | Source, head, custody and CAS fences | 39,122 | 63.375425 | 12.688329 |
| initial | Semantic assembly | 901 | 12.341964 | 8.033616 |
| initial | Catalog and SQL publication | 2,480 | 16.647043 | 6.963801 |
| initial | Remaining preparation orchestration | 16 | 102.101611 | 9.419233 |
| successor | Prior projection serialization components | 5,472 | 2.205365 | 0.776598 |
| successor | Reader internal subphases | 58,607 | 58.240405 | 3.218445 |
| successor | Scheduler checkpoint subphases | 247,168 | 109.166537 | 54.544388 |
| successor | AST and extraction context verification | 73,363 | 153.425738 | 26.855773 |
| successor | Implementation source hashing | 79,160 | 2.374207 | 2.328973 |
| successor | Source, head, custody and CAS fences | 56,677 | 88.521828 | 15.862687 |
| successor | Semantic assembly | 901 | 12.739779 | 8.297459 |
| successor | Catalog and SQL publication | 2,526 | 17.289330 | 7.498834 |
| successor | Remaining preparation orchestration | 22 | 134.488686 | 15.601838 |

Category scopes are disjoint and cover the 64 selected Python call scopes. Inclusive totals contain nested calls and overlap across categories. Exclusive time subtracts selected descendants only; uninstrumented work and some observer bookkeeping remain in that attribution. The sum of selected exclusive times is bounded by the actual phase wall interval. Call totals count selected method invocations, not transactions or native SQL statements.

| Phase | Prior encode calls | Prior wire bytes | Implementation bytes hashed | CAS physical bytes | CAS read calls | Semantic scan calls | Catalog publish calls | Store batch calls | Persistence AST rows offered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 2,288 | 44,569,515 | 2,210,919,260 | 161,484,575 | 13,136 | 1 | 1 | 1 | 299 |
| successor | 2,736 | 53,301,224 | 2,997,972,530 | 164,075,203 | 15,448 | 1 | 1 | 1 | 299 |

| Phase | Reader component | Calls | Inclusive seconds | Exclusive seconds |
| --- | --- | ---: | ---: | ---: |
| initial | Reader total; exclusive column is residual | 13,136 | 44.041863 | 1.465796 |
| initial | Reader checkpoints | 26,272 | 39.664749 | 0.119676 |
| initial | Artifact canonical wire | 4,252 | 0.538697 | 0.538697 |
| initial | Structured CID including its canonical wire | 4,252 | 0.786932 | 0.296378 |
| initial | Canonical wire inside structured CID | 4,252 | 0.490554 | 0.490554 |
| initial | Raw source CID | 5,272 | 0.233365 | 0.233365 |
| initial | Cache freeze admission | 3,612 | 1.017006 | 1.017006 |
| initial | Outermost cache thaw | 3,612 | 0.335319 | 0.335319 |
| successor | Reader total; exclusive column is residual | 15,448 | 59.481317 | 1.718483 |
| successor | Reader checkpoints | 30,896 | 54.695955 | 0.151567 |
| successor | Artifact canonical wire | 4,315 | 0.556160 | 0.556160 |
| successor | Structured CID including its canonical wire | 4,315 | 0.820866 | 0.343294 |
| successor | Canonical wire inside structured CID | 4,315 | 0.477572 | 0.477572 |
| successor | Raw source CID | 7,500 | 0.369327 | 0.369327 |
| successor | Cache freeze admission | 3,633 | 0.986199 | 0.986199 |
| successor | Outermost cache thaw | 3,633 | 0.334326 | 0.334326 |

Reader total inclusive time equals the seven Reader subphase exclusive times, the six nested scheduler subphase exclusive times and the parent exclusive residual. Reader checkpoint inclusive time contains the new scheduler observations, and structured CID inclusive time contains its nested canonical-wire observation; inclusive columns must not be added. The Reader residual contains physical reads, file custody/path/CID prechecks, JSON parsing, other uninstrumented reader work and observer bookkeeping. It is not a direct disk-I/O measurement.

| Phase | Source bytes hashed | Fresh artifact wire bytes | Structured CID canonical bytes | Cache freezes returned | Cache freezes raised and fell back | Outermost cache thaws |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 239,814 | 85,289,947 | 85,289,947 | 3,612 | 0 | 3,612 |
| successor | 340,825 | 87,401,260 | 87,401,260 | 3,633 | 0 | 3,633 |

| Phase | Checkpoint component | Calls | Inclusive seconds | Exclusive seconds |
| --- | --- | ---: | ---: | ---: |
| initial | Reader checkpoint total; exclusive column is residual | 26,272 | 39.664749 | 0.119676 |
| initial | Scheduler cancellation poll | 26,270 | 39.545073 | 1.701052 |
| initial | Locked-state entry before yield | 26,270 | 34.886445 | 34.721048 |
| initial | Stored configuration validation | 26,270 | 0.165398 | 0.165398 |
| initial | Current kernel boot identity | 26,270 | 1.133931 | 1.133931 |
| initial | Owner liveness probe including birth marker | 52,540 | 1.823644 | 0.264392 |
| initial | Owner process birth marker | 52,540 | 1.559253 | 1.559253 |
| successor | Reader checkpoint total; exclusive column is residual | 30,896 | 54.695955 | 0.151567 |
| successor | Scheduler cancellation poll | 30,896 | 54.544388 | 2.096887 |
| successor | Locked-state entry before yield | 30,896 | 48.724428 | 48.509912 |
| successor | Stored configuration validation | 30,896 | 0.214516 | 0.214516 |
| successor | Current kernel boot identity | 30,896 | 1.422348 | 1.422348 |
| successor | Owner liveness probe including birth marker | 61,792 | 2.300725 | 0.340593 |
| successor | Owner process birth marker | 61,792 | 1.960133 | 1.960133 |

Reader checkpoint inclusive time equals the six selected scheduler subphase exclusive times plus the checkpoint residual. Cancellation-poll inclusive time includes locked-state entry, configuration, boot and owner probes; locked-state entry includes configuration validation, and owner liveness includes its birth marker probe. These inclusive observations overlap. The locked-state entry timer stops at the original first yield: it includes acquiring the existing thread/file locks and loading/parsing/validating the existing state, and excludes the yielded operation body and context exit. The unchanged original context manager performs its usual exit and unlocking. The cancellation-poll exclusive residual includes unselected poll body/ancestor work and context exit; neither residual nor lock-entry time isolates disk latency, lock wait or JSON parsing. Successful entry, base-configuration and boot counts match scheduler polls. The initial receiver restores its root with one ordinary load_preparation_root read whose callback is None: its two Reader.check calls do not poll MAIN. All other initial checks and every successor Reader.check poll MAIN. The base-configuration timer covers the captured scheduler validator; the separate configuration wire comparison in the child facade remains in locked-state entry residual. Birth probes are positive and no more numerous than liveness probes; no fixed ancestor population is assumed. All wrappers retain existing checkpoint cadence, cancellation, TTL and freshness semantics. No scheduler optimization is implemented.


Reader subphase wrappers admit only the exact captured direct callers under an active Reader.read. The two checkpoint calls per successful read remain intact; scheduler timing is restricted to these selected checkpoints. Fresh artifact wire/CID calls match the independent structured JSON decode count; source CID calls match source reads, and outermost cache thaws match cache hits. Source hashing bytes plus fresh canonical artifact bytes are bounded by actual physical read bytes; cache hits still reread complete leaves. A bounded cache-freeze refusal may fall back to the unchanged ordinary path. No extra reads, serialization or traversal is added by these observations.


Prior-projection observations measure the existing encode and canonical-wire components inside the captured prepare_page sum generator. They exclude generator iteration and the complete sum operation. Implementation hash bytes and CAS read bytes come from actual existing arguments/results/counters; wrappers add no source or artifact reads and no serialization calls. The full timing documents declare byte applicability separately for each scope. Persistence row counters describe method input rows and do not attest durable commit counts; physical database I/O and native execute-call counts were not observed.

Both receivers independently admit the same 128-source parser-cache handoff, parse the remaining 171 Python sources and the single edited successor source, and retain semantic syntax parsing. Every runtime optimization remains at its qualified default. Source/context/head/CID/custody checks, physical byte charging, aggregate bounds, cancellation and checkpoints remain active. No private byte authority, memo or new sizing optimization is introduced. This milestone instruments existing work; it implements no inference or scan optimization.

All child cleanup completed, own reservations released and MAIN returned idle. Numerical 8D/384D/768D/4096D Leanstral heads, CUDA, inference, training, federation, signed-worker dispatch and formal proof are outside this structural profiling qualification. No database binary is retained or independently reopened.

The accompanying source observation joins the same two production files and existing focused CI source to their historical frozen bytes and current held identities. The separate finite integrity closer has not run when this report is written; its [receipt](../../../../artifacts/codebase_ir_terminal_bench/preparation-checkpoint-profile-final-close-20261013-03.json) is excluded from its own input closure.
