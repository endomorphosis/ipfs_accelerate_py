# Whole-call input guard: partial qualification

The first 768D CUDA cohort and its independent ordinary archive audit passed.
The whole-call input guard candidate did not improve median latency in this
cohort. The existing baseline remains selected. This review is incomplete:
one of four required CUDA cohorts has completed; 768D B and 4096D A/B remain open.

| Rows | Baseline median (ms) | Candidate median (ms) | Candidate change |
| --- | ---: | ---: | ---: |
| 1 | 23.813 | 37.994 | +59.55% |
| 16 | 88.657 | 95.070 | +7.23% |
| 32 | 162.133 | 163.187 | +0.65% |

Each row uses 12 paired samples per route with both execution orders repeated
six times. These are observations from one cohort, with the original timing
distribution retained. They do not establish repeatability or isolated-device
performance. The separately captured start/end/presence/modality tensors matched
within the inherited tolerance; maximum absolute error was 1.1920928955078125e-7.
All 84 public returns, 84 four-logit snapshots and 72 timed CUDA calls are retained.
The audit independently checked 208 archive members and 36 current-source pins.

The candidate changes only the outer input snapshot/comparison. It retains
checkpoint and receipt integrity checks, independently anchored nested snapshot
fields, row and result guards, source/current-binding checks, lease polls,
numerical kernels and CPU opt-out behavior. Three portable suites passed:
212 adapter controls, 105 archive-reader controls and 79 full-review assembly
controls, totaling 396 with no errors, failures or skips. The full-review
assembler still requires all four positive cohorts.

Both attempts at 768D B failed on the initial CUDA scalar allocation, before
restoring any model or producing a public report. Their negative receipts are
retained with their original incomplete cleanup status. A later independent
observation confirms zero shared scheduler reservations. The positive A run
also established release of its own CUDA allocations and leases. This does not
assert that the entire device is unused.

Read-only diagnostics show a separate llama-server reporting 63,527 MiB of GPU
residency. It is a possible contributor to the allocation failures; causation
has not been established. No foreign process, service or CUDA context was
stopped or reset. Further CUDA qualification waits for available device memory.

The next candidate deduplicates identical binding registrations by exact owner
identity and attribute name. Distinct aliases and receivers, wrapped methods,
closures, defaults, private namespaces and every verification boundary remain
required. Portable mutation controls and fresh CUDA comparisons must pass
before changing profile selection. No speedup is claimed for that candidate.

Earlier 8D/384D and 768D/4096D results remain pinned by predecessor closures.
The 4096D fixtures remain synthetic and untrained; native Leanstral output and
fresh encoder qualification remain open. This ordinary review supplies no proof
authority, execution attestation, training, federation admission or production
qualification. All 32 production milestone rows remain unchanged and open.

Evidence is listed in the adjacent partial-review JSON. Its separate verifier
rereads the selected files, the positive archive and current sources and computes
a local CIDv1 DAG-JSON identifier for the canonical review. The CID identifies
these bytes; it does not certify model execution or proof validity.
