# Fresh identical-state scheduler parsing reuse

Scheduler poll parsing reuse is enabled by default. Set `cached_poll_state=False` on `GlobalResourceScheduler` to opt out; the constructor accepts only an exact boolean. The qualified change replaces one scheduler implementation file and adds persistent focused CI coverage.

Each existing poll still takes the same thread/file locks and reads the complete current state file. Only an exactly identical, freshly read full text can reuse a private immutable prevalidation snapshot. Each cache hit creates a fresh mutable tree with marshal before invoking the existing virtual configuration validator. Cancellation, parent/ancestor expiry, current boot identity, PID birth and liveness checks still run on the ordinary path at every existing poll. The cache grants no cancellation or lease authority.

The single-entry cache admits only finite ordinary JSON trees and is limited to 262,144 text characters, 32,768 nodes, depth 64 and 4 MiB for the active entry's retained private text/snapshot storage. On the supplied nonreentrant polling route, the changed-text path drops both the old scheduler slot and the local old-tuple reference before decoding or admitting a replacement. Custom, preimported or subsequently changed decoder hooks, unusual or oversized states and clone refusals use fresh ordinary JSON decoding. Writes, reconfiguration, missing/blank state, read/parse/validation errors and changed text invalidate the memo. Marshal inputs are private bytes produced by this facade from a strict plain JSON tree; no externally supplied marshal data is accepted. This is an active-entry bound, with eviction lifetime tested for ordinary nonreentrant polling. Recursive custom validators or held contexts can retain additional pending snapshots or cloned trees. Existing full mutable virtual-validator compatibility remains intact; no overall or peak memory bound across those contexts, total process memory, decoded trees or temporary allocations is claimed.

The corrected source passed the authored 300-entry / 299-Python initial scan and full one-edit successor in four isolated producer, opt-out baseline, default receiver and normal-reference processes. Both receivers match the reference in complete source snapshots, manifests, semantic state, all 299 projections, all 13 native table families, histories, heads and operation receipts. All 16 independent preparation counters match between receivers in each phase. Other qualified preparation defaults remain enabled.

Actual corrected owner receipt: [receipt.json](../../../../artifacts/codebase_ir_terminal_bench/checkpoint-parse-reuse-checks-20261013-02/receipt.json), 259,396 bytes; SHA256 `d2c29355a1bf2210642b764109cfb74455b90718c703e1a036d5fd3c8b75b54e`. Campaign 351.745909 seconds.

| Phase | Opt-out baseline seconds | Default receiver seconds | Default minus baseline seconds | Difference percent |
| --- | ---: | ---: | ---: | ---: |
| initial | 78.377209 | 59.642287 | -18.734921 | -23.9035 |
| successor | 90.538336 | 67.827797 | -22.710539 | -25.0839 |

Combined phases: opt-out 168.915545s; default 127.470084s; signed difference -41.445461s (-24.5362%). These wall times are one sequential pair on this host and fixture. They include host variation and retained scalar/semantic observer cost. They do not qualify general throughput, causal observer overhead or model-inference speed. No observer-free overhead estimate was measured.

| Phase | Opt-out JSON parses | Opt-out clones | Default JSON parses | Default clones | Selected total per receiver |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial | 38,768 | 0 | 51 | 38,717 | 38,768 |
| successor | 45,100 | 0 | 71 | 45,029 | 45,100 |
| combined | 83,868 | 0 | 122 | 83,746 | 83,868 |

The two observers count selected owner-thread calls to the existing JSON parse and private snapshot clone helpers. Both receivers use the same count-only observer. Calls return successfully; no helper raises. These are helper invocation counts, not direct physical state-read, checkpoint, native-SQL or durable-commit counts. The observer adds no state/source reads or serialization. Identical-text reuse replaces parsing with a detached clone; it does not eliminate the fresh locked file read or any authority check.

| Phase | CAS physical bytes per receiver | CAS read calls | Structured JSON decodes | Artifact cache hits | Projection private decodes | Canonical rebuild calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 161,484,575 | 13,136 | 4,252 | 3,612 | 3,441 | 6,626 |
| successor | 164,075,203 | 15,448 | 4,315 | 3,633 | 3,334 | 6,967 |

The table lists six of the 16 exactly equal independent preparation counters. Cache highwater counters are cumulative maxima through each phase end. These CAS counters describe codebase preparation leaves, separately from scheduler state parsing. Source/context/head/CID/custody fences, physical byte charging, aggregate bounds, cancellation and checkpoint cadence remain active.

| Phase | Active AST projections | Native source-file rows | Revision rows | Invalidation rows | Catalog generation | Parity SHA256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| initial | 299 | 299 | 1 | 0 | 1 | `b3ee5843a231a79a1eb3a1484069bb3775adca7a204bc189a655767bc4d1a51d` |
| successor | 299 | 598 | 2 | 1 | 2 | `5f3586d9e16a3c3c56858391a39314723d0ea88708f330a517d8c406e2b9580c` |

The independent producer contributes the same 128-source prefix and parser-cache handoff. Receivers parse the remaining 171 initial Python sources and the single edited successor source; the reference parses all 299 initially. Required semantic syntax parsing remains active. Complete parity bodies and bytes are compared in both phases, including all 13 table families and publication history, not only row counts or the fingerprints displayed here.

Focused validation passed 123 distinct cases in 150 pass events: 96 existing scheduler cases, 27 authored cases and a persistent-CI rerun of those same 27 authored cases. The suite covers default-on/exact-bool opt-out, fresh reads and in-place changes, cancellation/release, corruption and read failures, current configuration and virtual validator mutations, expiry/boot/PID-birth and liveness freshness, detached mutable clones, bounded admission and refusal, changed/preimport decoder hooks and old-memo release before replacement admission. Existing spawn tests use the same captured scheduler candidate in each fresh process. The standalone fresh-interpreter case remains excluded because it would import the old production source during candidate qualification. The package-wide CI entry point was not replayed. Existing collection/spawn failures remain zero-credit diagnostics with cleanup recorded.

The earlier scheduler candidate completed its own four-role functional/parity and local-performance campaign, but a later source audit found that a changed-text miss retained the evicted tuple in a generator local while admitting a replacement. All 16 earlier native outputs and the earlier focused receipts and sources remain retained. That campaign receives no aggregate 4-MiB residency qualification and was never applied to production. This report uses only the corrected-source campaign numbers above.

All 73 preceding preparation controls remain historical evidence and were not rerun. All 125 other production/dependency/fixture/control source inputs retain their exact historical bytes; the selected scheduler and child are the two new native inputs. The original prospective declaration, corrected spawn addendum and new resident-bound admission retain the earlier evidence and declare the enlarged finite integrity envelope before the corrected run.

All four children cleaned up, own reservations released and MAIN returned idle. The accompanying application records the exact tested scheduler replacement and exclusively created persistent CI test source, plus the unchanged preceding codec, runtime and projection-CI source pairs. It retains separate focused and native qualification receipts. Numerical 8D/384D/768D/4096D Leanstral heads, CUDA, model inference, training, federation, signed-worker dispatch and formal proof remain outside this scan/scheduler qualification. No database binary is retained or independently reopened.

The separate finite integrity closer has not run when this report is written; its [receipt](../../../../artifacts/codebase_ir_terminal_bench/checkpoint-parse-reuse-final-close-20261013-02.json) is excluded from its own input closure.
