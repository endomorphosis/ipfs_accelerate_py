# Balanced full scheduler polling measurements

Repeated paired measurements confirm a large benefit for unchanged scheduler text with thousands of wait-history entries and a substantial regression when that text changes on every poll. The next candidate is deferred snapshot admission after repeated identical fresh reads. This milestone measures the previously applied default-on cache; it changes no production source.

The primary metric is the sum of full public `is_cancelled` call durations, excluding fixture replacement and instrumentation. Each of 15 state/update cells has six adjacent pairs with orders AB, BA, AB, BA, AB, BA and 120 steady polls per block. A is `cached_poll_state=False`; B omits the option and therefore uses the qualified default `True`. One separately timed cold poll primes each new facade. Neither timed mode uses observer wrappers.

Each fixture has three genuine nested leases owned by the current process, valid birth/boot markers, and a private TTL of 3600 seconds. Every call checks the ancestor chain; process observations deduplicate only within that call. The shared MAIN reservation retains its existing configuration, TTL 120.0 seconds and auto-renew policy.

## Results

Times below are medians of six block means in milliseconds per poll. The ratio is the median of six matched default/baseline ratios, which need not equal the ratio of the separately reported time medians. A ratio below 1 means less elapsed poll time. Every measured block and pair is retained, including unfavorable results.

| Fixture | Changes | Characters | Baseline ms | Default ms | Paired ratio | Pair range | AB median | BA median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | unchanged | 3008 | 0.0511 | 0.0478 | 0.943 | 0.926–0.951 | 0.944 | 0.938 |
| small | every10polls | 3008 | 0.1159 | 0.1775 | 1.561 | 1.047–2.274 | 1.732 | 1.391 |
| small | everypoll | 3008 | 0.2530 | 0.3107 | 1.399 | 0.658–1.475 | 1.390 | 1.464 |
| observed_like | unchanged | 62568 | 0.4208 | 0.1044 | 0.248 | 0.156–0.251 | 0.248 | 0.249 |
| observed_like | every10polls | 62568 | 0.6317 | 0.4017 | 0.618 | 0.524–0.828 | 0.639 | 0.596 |
| observed_like | everypoll | 62568 | 0.8768 | 1.9853 | 2.325 | 1.440–3.028 | 2.192 | 2.444 |
| max_history | unchanged | 82017 | 0.5431 | 0.1229 | 0.227 | 0.169–0.232 | 0.227 | 0.227 |
| max_history | every10polls | 82017 | 1.1717 | 0.5026 | 0.490 | 0.341–0.556 | 0.503 | 0.438 |
| max_history | everypoll | 82017 | 1.1129 | 2.1306 | 2.007 | 1.793–2.987 | 1.883 | 2.131 |
| near_cap | unchanged | 262144 | 0.7936 | 0.1627 | 0.205 | 0.203–0.208 | 0.204 | 0.206 |
| near_cap | every10polls | 262144 | 0.8221 | 0.3293 | 0.398 | 0.268–0.409 | 0.400 | 0.395 |
| near_cap | everypoll | 262144 | 1.4374 | 2.9833 | 2.080 | 1.207–2.988 | 1.971 | 2.188 |
| above_cap | unchanged | 262145 | 0.8128 | 0.8164 | 1.007 | 0.978–1.013 | 1.011 | 1.004 |
| above_cap | every10polls | 262145 | 0.7896 | 0.7936 | 1.001 | 0.982–1.020 | 0.998 | 1.004 |
| above_cap | everypoll | 262145 | 1.5716 | 1.6826 | 1.096 | 0.820–1.357 | 1.153 | 1.039 |

The three larger eligible fixtures show 75.2%, 77.3% and 79.5% less elapsed poll time with unchanged text. At one change per ten polls, their median reductions are 38.2%, 51.0% and 60.2%. With a change before every poll, they instead require 2.325, 2.007 and 2.080 times the baseline elapsed time.

Small unchanged text improves by 5.7%, while its one-in-ten change cell uses 56.1% more time. The small every-change cell has a median ratio of 1.399 with a wide 0.658–1.475 pair range. Above-cap fallback has median ratios of 1.007, 1.001 and 1.096; the every-change range is 0.820–1.358. This variation and order sensitivity remain part of the result. Six adjacent pairs in one process are not independent host-level replications, and these results do not establish a general workload speedup.

The observed-like fixture matches the previously held MAIN history count of 3087 entries; its actual compact text is 62568 characters. It does not copy live lease authority or exactly reproduce the prior 71299-byte state. The maximum-history fixture uses 4096 entries. Near-cap and above-cap fixtures add a labeled scalar string to the 4096-entry shape, reaching exactly 262144 and 262145 ASCII characters. Boundary behavior is specific to these shapes.

## Cold polls and excluded mutations

The table records median first-poll latency and total replacement time per 120-poll block. Replacement includes the private sibling lock, complete write, file fsync, atomic replace and directory fsync. That time is outside the primary poll intervals; it remains visible here and in every raw block. Zero replacements occur for unchanged text, 12 for one-in-ten changes, and 120 for every-change blocks. New facade construction occurs before variant A is written so constructor formatting cannot alter the declared fixture.

| Fixture | Changes | Baseline cold ms | Default cold ms | Baseline writes ms | Default writes ms | Baseline block wall ms | Default block wall ms |
|---|---|---:|---:|---:|---:|---:|---:|
| small | unchanged | 0.0651 | 0.1271 | 0.00 | 0.00 | 12.95 | 12.84 |
| small | every10polls | 0.2306 | 0.4091 | 85.46 | 88.31 | 108.75 | 118.16 |
| small | everypoll | 0.2441 | 0.2915 | 884.49 | 882.23 | 922.42 | 919.69 |
| observed_like | unchanged | 0.6552 | 1.4124 | 0.00 | 0.00 | 59.58 | 22.40 |
| observed_like | every10polls | 0.8282 | 2.6254 | 86.17 | 86.42 | 169.74 | 148.64 |
| observed_like | everypoll | 1.1160 | 2.7249 | 910.58 | 875.47 | 1019.16 | 1119.63 |
| max_history | unchanged | 0.7555 | 1.7610 | 0.00 | 0.00 | 74.26 | 25.57 |
| max_history | every10polls | 1.0524 | 3.4322 | 84.50 | 89.32 | 231.41 | 174.26 |
| max_history | everypoll | 1.2660 | 2.4235 | 845.47 | 885.80 | 994.72 | 1132.01 |
| near_cap | unchanged | 0.9409 | 1.9411 | 0.00 | 0.00 | 105.96 | 30.64 |
| near_cap | every10polls | 0.9852 | 1.9785 | 84.98 | 85.93 | 193.58 | 136.50 |
| near_cap | everypoll | 1.7190 | 2.8006 | 831.04 | 833.52 | 994.11 | 1215.32 |
| above_cap | unchanged | 0.9735 | 0.9857 | 0.00 | 0.00 | 107.79 | 107.77 |
| above_cap | every10polls | 0.9624 | 0.8806 | 86.77 | 88.17 | 192.95 | 194.58 |
| above_cap | everypoll | 1.7021 | 1.6549 | 913.48 | 924.35 | 1123.97 | 1141.75 |

Complete campaign wall time was 90.36 seconds, including setup, cold polls, writers, timed calls, observers and cleanup. Initial fixture setup was 38.96 ms. Opening and closing selected-input checks were 1.03 and 1.17 ms. Every observer records its separate wall time. Clock/loop overhead is not subtracted from measured intervals.

## Observed behavior and custody

Thirty separate instrumented blocks verified actual helper counts. Each of their 120 steady calls performed one full private state read, configuration validation, boot observation, owner alive check, birth-marker check, exclusive flock acquisition and unlock. Baseline always decoded 120 times. Eligible default cells decoded/cloned 0/120 for unchanged text, 12/108 for one-in-ten changes, and 120/0 for every-change text. Above-cap default cells decoded 120 times and cloned zero times. Cold calls decoded once; eligible default cold calls admitted a snapshot, while above-cap attempts were rejected. All snapshot-attempt/admission/rejection joins passed. Wrappers were restored before any subsequent timed block.

All 25410 fixture public polls returned exactly False and left private JSON unchanged between declared replacements. This includes 21600 steady timed polls, 180 cold timed polls, and 3630 instrumented polls including their primes. Fifteen separate MAIN authority polls bring the public-call total to 25425, below the declared 30000 limit. Both private and shared reservations were released, the heartbeat joined, private files removed, and MAIN ended with zero leases and zero waiters. The eleven selected runner/declaration/captured-source/origin/config pins match their closing pins.

Known ASCII JSON payload read by those fixture public calls totals 3414504324 bytes. This is derived payload volume, not physical disk traffic or all workload IO. The independent 2.5 GiB cumulative integrity-read cap applies to the evidence closer's preflight and two custody passes; workload polls have their own 30000-call, 1 MiB fixture and 900-second limits. Constructors, fixture writes and owner-proc reads are additional workload IO.

The finite standard-library closer retains all 1914 complete predecessor pin dictionaries, their entire non-pin integrity record, the immediate predecessor close, and the new selected declarations, captured source, measurement receipt, summary and report. It hashes each selected leaf twice under the existing 1984-member and 2560-MiB integrity-read ceilings. The closer replays no scheduler or native workload. The subsequent mutable work-note sync records its exact resulting union and CIDv1 close; no current work note is claimed as an immutable input.

## Next optimization and limits

The every-change observer confirms that default mode decodes and attempts a full snapshot for every changed eligible state. That extra traversal and marshal construction is a justified optimization target. A candidate should retain only bounded pending text on the first observation, attempt snapshot admission after a second consecutive identical complete read, and clone on later hits. Pending and admitted entries must be mutually exclusive; all existing error, decoder, validator and persistence invalidation fences must clear both. Capture pristine bytes before virtual validation and publish only after successful validation and the decoder fence. This design is not implemented or qualified by this measurement.

Keep default-on behavior with exact-boolean opt-out. A source change requires focused cancellation/expiry/liveness/mutation/decoder/bounds controls and complete initial/successor/reference parity, followed by this balanced benchmark. The prior 24.5% preparation elapsed reduction remains historical evidence from its own sequential full-scan comparison; this benchmark does not rerun that scan.

These private, single-process measurements use warm local files and no competing fixture writer. They do not measure worker lock contention, network filesystems, cold disks or writer-plus-reader throughput. No model inference, CUDA, training, federation, signed dispatch or formal-proof qualification ran here. Separate work still includes source-grounded codebase_IR/intent_IR, 8D/384D/768D/4096D Leanstral qualification, gte-small CUDA auto-selection, one federated model per compatible lineage, and later gradient synchronization through ipfs_accelerate_py and MCP++.

## Artifacts

- Receipt: `/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/poll-parse-reuse-balanced-candidate-20261013-01/receipt.json`; SHA256 `013de7b73731c650ff7ab741d97da2a2e8361725bd62224f488a2fa8deb552ba`; raw CIDv1 `bafkreiabhxt3onzryzip66vxihmx3ivc5a3bojn5mire6sekf6un5nksxi`.
- Raw block aggregates and counters: receipt `measurement.cells`.
- Recomputed table data: `/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/poll-parse-reuse-balanced-candidate-20261013-01/measurement-summary.json`.
- Frozen protocol, prospective custody profile, runner and source pins: `/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/poll-parse-reuse-balanced-candidate-20261013-01`.
- Integrity close destination: `/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/poll-parse-reuse-balanced-final-close-20261013-01.json`.
