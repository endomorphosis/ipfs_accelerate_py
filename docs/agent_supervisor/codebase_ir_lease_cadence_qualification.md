# Codebase IR lease cadence and head compatibility review

The new 8D and 384D formula candidate retains four fresh authenticated resource checks per inference call and renews its own heartbeat only when due. Two fresh-process CUDA cohorts compare the complete calls against the original and held bitwise-v2 implementations. The 768D and 4096D cached-input candidates also passed complete-session compatibility probes. Existing selected profiles and all 32 production acceptance rows remain unchanged.

## Complete formula inference observations

| Cohort | Dimension | Rows | Original ms | Held v2 ms | Lease candidate ms | v2 over candidate | Paired ratio range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 8 | 1 | 5.592 | 41.112 | 16.918 | 2.430 | 2.163 to 2.589 |
| A | 8 | 16 | 33.704 | 68.919 | 44.842 | 1.537 | 1.479 to 1.649 |
| A | 8 | 32 | 7.530 | 41.381 | 18.311 | 2.260 | 2.138 to 2.376 |
| A | 384 | 1 | 7.384 | 45.960 | 20.888 | 2.200 | 2.080 to 2.300 |
| A | 384 | 16 | 39.498 | 80.290 | 56.086 | 1.432 | 1.374 to 1.621 |
| A | 384 | 32 | 13.408 | 57.390 | 34.452 | 1.666 | 1.642 to 1.745 |
| B | 8 | 1 | 6.317 | 44.330 | 19.565 | 2.266 | 1.249 to 2.713 |
| B | 8 | 16 | 36.957 | 77.508 | 50.372 | 1.539 | 1.453 to 1.600 |
| B | 8 | 32 | 8.546 | 57.429 | 21.401 | 2.683 | 2.006 to 3.055 |
| B | 384 | 1 | 7.831 | 46.432 | 22.952 | 2.023 | 1.673 to 2.520 |
| B | 384 | 16 | 39.542 | 81.546 | 56.801 | 1.436 | 1.355 to 1.601 |
| B | 384 | 32 | 12.899 | 57.272 | 32.033 | 1.788 | 1.510 to 1.926 |

Each cohort has 240 complete public returns and independent projection snapshots, including 216 timed CUDA calls, with 12 Adam state restorations and no training fits or optimizer steps. It also exercises 28 inherited refusal controls and 10 new lease custody and cancellation controls. Timing excludes observers; untimed main-thread observations count fresh helper checks and fsync/replace calls inside those checks. They do not observe whole-interval, timed or background I/O; writer-context segments describe persistence intent. Complete decisions, numerical projections, checkpoint/model state and RNG are checked. Ratios describe these fixed fixtures and runs; they grant no general performance or production admission.

## Fresh resource authority and renewal

Every boundary reads current shared state under the unchanged scheduler lock, authenticates the exact child lease key, and checks live ancestry, cancellation and expiry. A successful nondue read is never cached as authority. With the held 120 second TTL, renewal becomes due at 40 seconds; the writer rechecks all authority and cadence before changing only the child heartbeat and expiry. It keeps the existing file fsync, replace and directory fsync sequence. The helper performs no parent renewal, stale recovery or reservation pruning. Isolated CPU controls exercise due renewal and read-to-write races; the short native runs do not qualify actual due renewal or TTL expiry.

## Complete span head compatibility

Both 768D and 4096D probes compare baseline and candidate CPU opt-out routes and CUDA routes at 1, 16 and 32 rows:12complete public reports and 12 separate four-logit panels per head. Canonical decisions match; all four output tensors stay within 5e-5 of the retained CPU reference. The observed maximum difference is at most 1.2e-7. No training runs. The 768D checkpoint has one historical optimizer step and historical embedding receipts; the 4096D fixture has zero steps and synthetic unreceipted inputs. No fresh encoder execution, native Leanstral output provenance or trained 4096D qualification is established. Single CUDA observations with no balanced repetition cannot establish speedup.

The first 768D probe failed memory closure after decisions and numeric comparisons passed. A new probe successor collects unreachable session cycles before clearing process-owned framework workspaces. Successful successor probes close all four children and the root, restore ambient settings, return allocated GPU bytes to zero and leave shared reservations at zero. The failed predecessor remains retained.

## Validation and next milestones

1018 current positive test cases passed with zero failures, errors or skips. Failed or predecessor stages are retained separately and excluded from this count. Ordinary audits recheck bounded retained bytes, current sources, input bindings, decision coverage and numeric comparisons. They execute no retained Python or model and grant no proof authority. Local CIDv1 review manifests describe retained bytes; they are not IPFS publications.

Next work is repeated 768D/4096D performance qualification, cadence integration for those heads, remaining checkpoint costs, larger signed successor scans and per-lineage federation. Repository formalizations and intent matches must stay tied to the exact code snapshot and explicit checker outcomes before supervisor planning treats them as verified evidence.

Retained evidence: [Formula cohort A](../../../../artifacts/codebase_ir_terminal_bench/formula-lease-cadence-qualification-20261004-02/result.json), [Formula cohort B](../../../../artifacts/codebase_ir_terminal_bench/formula-lease-cadence-qualification-20261004-03/result.json), [Ordinary formula audit A](../../../../artifacts/codebase_ir_terminal_bench/formula-lease-cadence-audit-20261004-01/audit.json), [Ordinary formula audit B](../../../../artifacts/codebase_ir_terminal_bench/formula-lease-cadence-audit-20261004-02/audit.json), [768D compatibility probe](../../../../artifacts/codebase_ir_terminal_bench/cached-span-native-compatibility-20261004-02/result.json), [4096D compatibility probe](../../../../artifacts/codebase_ir_terminal_bench/cached-span-native-compatibility-20261004-03/result.json).
