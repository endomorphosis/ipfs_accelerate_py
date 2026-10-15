# Deferred scheduler poll-cache admission qualification

The candidate waits for two consecutive identical fresh state reads before constructing a reusable snapshot. This removes snapshot traversal from constantly changing scheduler state while preserving full file reads, configuration validation and current lease-owner checks on every poll. The candidate remains default-on with the existing exact-boolean `cached_poll_state=False` opt-out.

Application status: Applied the qualified scheduler and persistent cache tests after focused, balanced and full native checks passed. The constructor still defaults to True; opt out with the exact boolean `cached_poll_state=False`.

The bounded balanced benchmark completed all 15 cases in 129.379 seconds. For the three larger eligible fixtures changing on every poll, candidate polling used 52.5-68.4% less measured time than the previous immediate-admission cache. Unchanged-state polling retained the main parsing-reuse benefit, with 1.3-1.9% more time than immediate admission. State changing every tenth poll had a measured tradeoff: candidate median time was 6.4-16.2% higher than immediate admission, while remaining lower than baseline in its median for each larger fixture.

These percentages are medians of six matched within-process ratios per case. They measure full public scheduler polling on a private local fixture. They do not establish model inference or CUDA throughput, and they must not be pooled into a production-workload speedup.

## Measured design

- B: original scheduler source with `cached_poll_state=False`; each poll parses JSON.
- I: original scheduler source with the default immediate snapshot admission.
- D: candidate scheduler source with default deferred admission.
- Each state/churn case ran all six mode permutations: BID, BDI, IBD, IDB, DBI and DIB. Each mode appeared twice at each position.
- Each block began with a fresh facade, then wrote fixture A and made two separately recorded same-text prime polls. Exactly 120 steady public `is_cancelled` calls followed. Prime costs are excluded from the steady metric.
- Variant writes, fsyncs, atomic replacement and setup were outside the individual poll timing interval and recorded separately. The public call itself included locks, full fresh state read, parsing or cloning, validation, current clock, boot identity and owner liveness/birth checks.
- Every mode used the same private state path, configuration and three genuine current-process ancestor leases. The fixture had no competing writer. Two equal-length ASCII variants differed only in a synthetic diagnostic nonce.
- Exact boundaries were 262144 characters for eligible near-cap input and 262145 for above-cap fallback. History fixtures contained 3087 or 4096 deterministic samples; observed-like means history-count matched, rather than a copied production authority state.

## All measured cases

Each entry is median ratio (minimum-maximum) across all six matched groups. A ratio below 1 means the numerator used less polling time. No outliers were discarded.

| Fixture | Bytes/chars | Changes | I/B | D/B | D/I |
| --- | ---: | --- | ---: | ---: | ---: |
| small | 3006 | unchanged | 0.940 (0.927-0.966) | 0.968 (0.953-2.647) | 1.034 (0.986-2.815) |
| small | 3006 | every10polls | 1.121 (1.082-1.361) | 1.160 (1.148-2.242) | 1.037 (0.854-2.067) |
| small | 3006 | everypoll | 1.620 (0.893-1.877) | 1.076 (0.620-1.299) | 0.662 (0.592-1.232) |
| observed_like | 62566 | unchanged | 0.250 (0.249-0.250) | 0.254 (0.254-0.255) | 1.018 (1.015-1.020) |
| observed_like | 62566 | every10polls | 0.511 (0.500-0.585) | 0.573 (0.565-0.626) | 1.119 (0.995-1.209) |
| observed_like | 62566 | everypoll | 2.814 (1.408-4.947) | 1.198 (0.656-1.682) | 0.375 (0.315-0.769) |
| max_history | 82015 | unchanged | 0.229 (0.167-0.229) | 0.232 (0.168-0.233) | 1.013 (1.006-1.017) |
| max_history | 82015 | every10polls | 0.577 (0.229-0.920) | 0.710 (0.388-1.141) | 1.162 (0.624-2.477) |
| max_history | 82015 | everypoll | 2.020 (1.699-2.504) | 0.896 (0.683-1.752) | 0.475 (0.353-0.700) |
| near_cap | 262144 | unchanged | 0.251 (0.250-0.260) | 0.256 (0.250-0.259) | 1.019 (0.981-1.030) |
| near_cap | 262144 | every10polls | 0.545 (0.529-0.728) | 0.575 (0.555-0.597) | 1.064 (0.761-1.111) |
| near_cap | 262144 | everypoll | 2.701 (2.304-3.489) | 0.883 (0.606-1.151) | 0.316 (0.238-0.436) |
| above_cap | 262145 | unchanged | 1.005 (1.002-1.014) | 1.007 (1.003-1.010) | 1.001 (0.993-1.004) |
| above_cap | 262145 | every10polls | 1.009 (0.991-1.042) | 1.008 (0.984-1.015) | 0.992 (0.970-1.018) |
| above_cap | 262145 | everypoll | 0.932 (0.756-1.565) | 1.009 (0.733-1.317) | 1.123 (0.695-1.444) |

The every-poll D/I ranges for observed-like, max-history and near-cap fixtures were respectively 0.315-0.769, 0.353-0.700 and 0.238-0.436: every matched group improved relative to immediate admission. Their D/B ratios had broader ranges crossing 1, so this run does not support a consistent improvement relative to ordinary parsing under every-poll churn.

The small unchanged case retained a D/I outlier of 2.815; the small every-tenth-poll case retained a D/I maximum of 2.067. Above-cap every-poll fallback had D/I median 1.123 with range 0.695-1.444. These unstable and unfavorable observations remain part of the result. The data establish elimination of snapshot work more clearly than a universal latency improvement.

## Absolute steady polling times

Values are medians of six block averages in microseconds per public poll. A ratio of these separate medians can differ from the median matched ratios reported above.

| Fixture | Changes | B us/poll | I us/poll | D us/poll |
| --- | --- | ---: | ---: | ---: |
| small | unchanged | 51.4 | 48.3 | 49.8 |
| small | every10polls | 54.5 | 64.6 | 63.0 |
| small | everypoll | 217.8 | 299.8 | 217.7 |
| observed_like | unchanged | 414.6 | 103.7 | 105.4 |
| observed_like | every10polls | 424.9 | 217.5 | 243.5 |
| observed_like | everypoll | 571.2 | 1471.9 | 694.6 |
| max_history | unchanged | 532.5 | 121.9 | 123.5 |
| max_history | every10polls | 691.8 | 459.3 | 567.5 |
| max_history | everypoll | 1015.8 | 2044.3 | 891.0 |
| near_cap | unchanged | 645.9 | 162.2 | 164.9 |
| near_cap | every10polls | 1386.7 | 739.4 | 802.2 |
| near_cap | everypoll | 1202.1 | 3407.5 | 908.3 |
| above_cap | unchanged | 650.0 | 653.7 | 655.4 |
| above_cap | every10polls | 677.3 | 674.2 | 673.3 |
| above_cap | everypoll | 1083.0 | 1081.3 | 1117.3 |

## Cache-work and authority observations

The separate instrumented observer ran outside every timed block and restored all wrappers afterward. All three modes read, validate, read current boot identity, check owner liveness/birth and lock/unlock exactly once per public poll. Shared ancestor ownership deduplicates owner checks within one call and does not reuse them across calls.

For an eligible fixture after two identical primes:

| Changes | B decode/clone/snapshot | I decode/clone/snapshot | D decode/clone/snapshot | D final state |
| --- | --- | --- | --- | --- |
| unchanged | 120/0/0 | 0/120/0 | 0/120/0 | admitted |
| every tenth poll | 120/0/0 | 12/108/12 | 23/97/11 | pending text |
| every poll | 120/0/0 | 120/0/120 | 120/0/0 | pending text |

Above-cap input parsed on all 120 calls in all modes. Immediate admission attempted 120 rejected snapshots; deferred admission retained neither pending text nor a snapshot and made zero snapshot attempts.

The candidate first prime parsed once and retained only bounded pending text. Its second identical prime parsed once, constructed a pristine snapshot before virtual validation, and admitted it after validation and the decoder identity fence. The third unchanged call cloned private immutable bytes into fresh containers. Changed text releases the previous slot and local references before decoding. Persisting, reconfiguration, blank/missing input, read/parse/validation failures, decoder changes and clone fallback invalidate both private slots.

The active retained cache bound covers either pending text or admitted text plus snapshot. It is not a bound on all Python working memory, recursive custom-validator contexts or externally held private contexts.

## Prime and write costs

Each state row combines 18 blocks per mode across the three declared churn cases. Values are median microseconds for the first and second separately timed primes. They show the extra admission parse paid by D before its steady hits.

| Fixture | B first/second us | I first/second us | D first/second us |
| --- | ---: | ---: | ---: |
| small | 102.2/61.1 | 246.6/70.8 | 229.0/143.5 |
| observed_like | 489.0/432.8 | 1278.2/120.9 | 535.2/1182.5 |
| max_history | 658.0/564.4 | 1681.6/149.6 | 750.5/1530.6 |
| near_cap | 799.3/679.0 | 1966.3/264.7 | 868.0/1673.0 |
| above_cap | 837.9/688.7 | 835.6/700.8 | 868.8/697.2 |

Across the 270 timed blocks, 11880 steady variant replacements took 87.104 seconds, and their 270 setup writes took 2.040 seconds. These writer costs are excluded from the primary poll sum. They are not a separate fsync-throughput benchmark.

## Order sensitivity

Within each bracket, the values correspond to BID, BDI, IBD, IDB, DBI and DIB in that fixed order. These are all raw matched ratios rounded to four decimal places; the receipt retains full precision.

| Fixture | Changes | Ratio | Six ordered matched ratios |
| --- | --- | --- | --- |
| small | unchanged | I/B | 0.9664, 0.9404, 0.9404, 0.9269, 0.9417, 0.9405 |
| small | unchanged | D/B | 0.9530, 2.6473, 0.9742, 0.9650, 0.9649, 0.9714 |
| small | unchanged | D/I | 0.9861, 2.8150, 1.0360, 1.0412, 1.0247, 1.0328 |
| small | every10polls | I/B | 1.0846, 1.1364, 1.3610, 1.1572, 1.1049, 1.0822 |
| small | every10polls | D/B | 2.2417, 1.1478, 1.1621, 1.1582, 1.1811, 1.1522 |
| small | every10polls | D/I | 2.0668, 1.0100, 0.8538, 1.0008, 1.0690, 1.0647 |
| small | everypoll | I/B | 1.8768, 1.0214, 0.8935, 1.6133, 1.6269, 1.8670 |
| small | everypoll | D/B | 1.2993, 0.6205, 1.1009, 0.9544, 1.0501, 1.2659 |
| small | everypoll | D/I | 0.6923, 0.6075, 1.2322, 0.5916, 0.6455, 0.6780 |
| observed_like | unchanged | I/B | 0.2494, 0.2503, 0.2500, 0.2500, 0.2493, 0.2495 |
| observed_like | unchanged | D/B | 0.2543, 0.2544, 0.2537, 0.2545, 0.2540, 0.2546 |
| observed_like | unchanged | D/I | 1.0199, 1.0164, 1.0149, 1.0178, 1.0188, 1.0204 |
| observed_like | every10polls | I/B | 0.5002, 0.5062, 0.5179, 0.5146, 0.5855, 0.5073 |
| observed_like | every10polls | D/B | 0.5774, 0.5663, 0.6260, 0.5653, 0.5826, 0.5677 |
| observed_like | every10polls | D/I | 1.1543, 1.1188, 1.2086, 1.0986, 0.9951, 1.1190 |
| observed_like | everypoll | I/B | 3.6522, 2.3061, 1.4075, 4.9466, 1.6766, 3.3216 |
| observed_like | everypoll | D/B | 1.3123, 0.7667, 1.0830, 1.5604, 0.6556, 1.6819 |
| observed_like | everypoll | D/I | 0.3593, 0.3325, 0.7695, 0.3154, 0.3911, 0.5063 |
| max_history | unchanged | I/B | 0.1671, 0.2285, 0.2281, 0.2286, 0.2294, 0.2293 |
| max_history | unchanged | D/B | 0.1684, 0.2316, 0.2317, 0.2325, 0.2321, 0.2306 |
| max_history | unchanged | D/I | 1.0078, 1.0133, 1.0155, 1.0173, 1.0120, 1.0058 |
| max_history | every10polls | I/B | 0.9204, 0.8018, 0.6220, 0.2286, 0.5317, 0.4962 |
| max_history | every10polls | D/B | 1.1415, 0.8684, 0.3879, 0.5663, 0.5690, 0.8506 |
| max_history | every10polls | D/I | 1.2401, 1.0831, 0.6236, 2.4771, 1.0702, 1.7141 |
| max_history | everypoll | I/B | 2.0391, 1.6990, 2.0014, 1.7512, 2.1202, 2.5035 |
| max_history | everypoll | D/B | 0.7189, 0.6834, 1.0342, 0.7572, 1.1035, 1.7515 |
| max_history | everypoll | D/I | 0.3526, 0.4022, 0.5167, 0.4324, 0.5205, 0.6996 |
| near_cap | unchanged | I/B | 0.2497, 0.2526, 0.2605, 0.2508, 0.2501, 0.2515 |
| near_cap | unchanged | D/B | 0.2567, 0.2498, 0.2556, 0.2568, 0.2537, 0.2590 |
| near_cap | unchanged | D/I | 1.0283, 0.9889, 0.9811, 1.0238, 1.0144, 1.0300 |
| near_cap | every10polls | I/B | 0.5836, 0.7284, 0.5372, 0.5528, 0.5316, 0.5293 |
| near_cap | every10polls | D/B | 0.5603, 0.5545, 0.5965, 0.5813, 0.5801, 0.5702 |
| near_cap | every10polls | D/I | 0.9601, 0.7612, 1.1106, 1.0516, 1.0912, 1.0773 |
| near_cap | everypoll | I/B | 2.6408, 2.5836, 3.4892, 2.3036, 2.7619, 2.7834 |
| near_cap | everypoll | D/B | 1.1514, 0.6153, 1.1127, 0.6059, 0.8927, 0.8737 |
| near_cap | everypoll | D/I | 0.4360, 0.2382, 0.3189, 0.2630, 0.3232, 0.3139 |
| above_cap | unchanged | I/B | 1.0050, 1.0022, 1.0052, 1.0093, 1.0139, 1.0049 |
| above_cap | unchanged | D/B | 1.0067, 1.0033, 1.0088, 1.0100, 1.0066, 1.0077 |
| above_cap | unchanged | D/I | 1.0017, 1.0011, 1.0036, 1.0006, 0.9928, 1.0027 |
| above_cap | every10polls | I/B | 1.0186, 1.0235, 1.0422, 0.9999, 0.9925, 0.9911 |
| above_cap | every10polls | D/B | 1.0053, 1.0149, 1.0106, 0.9999, 1.0102, 0.9845 |
| above_cap | every10polls | D/I | 0.9870, 0.9916, 0.9697, 1.0000, 1.0178, 0.9933 |
| above_cap | everypoll | I/B | 1.1649, 0.7556, 1.0550, 0.8086, 1.5652, 0.7817 |
| above_cap | everypoll | D/B | 1.3154, 1.0909, 0.7335, 0.9273, 1.3169, 0.8727 |
| above_cap | everypoll | D/I | 1.1292, 1.4438, 0.6952, 1.1467, 0.8414, 1.1165 |

All six groups were adjacent repetitions in one interpreter, rather than independent machines or production workload samples. Different mode positions expose variation in host scheduling and warm-system behavior; the data do not identify the cause of individual timing outliers.

## Execution, integrity and remaining qualification

Balanced receipt: `artifacts/codebase_ir_terminal_bench/deferred-poll-admission-candidate-20261013-01/receipt.json`, 386813 bytes, SHA256 `3e115e70dd376b904415b8e5ed3ea96e8f20ac3d14b7476a065fe951d1f185f7`.

Execution completed 32400 timed public polls, 630 explicit primes, 5400 separate observer steady polls and 15 MAIN reservation checks: 38445 total public calls. All returned the expected live-authority result. The 12 complete opening/closing source/configuration fences matched; private leases and workspace were cleaned up; the owned MAIN lease and heartbeat were released/joined; MAIN was idle afterward with unchanged configuration.

Candidate scheduler: 107765 bytes, SHA256 `621904713c3faef18232afb81f504830922eae08e3d2d024f25d09792ea93af4`. Candidate tests: 45554 bytes, SHA256 `ec1485297769852b8157a795c35e2f219e40fde0382dc4f8ca7bc2b18f53c6b7`.

Focused validation: 96 existing tests and 36 cache tests passed before application (132 distinct cases). The 36 persistent tests passed again after application (168 pass events total). The existing standalone fresh-interpreter reboot test remained explicitly deselected because candidate qualification loads captured leaf sources. Package-wide CI was not replayed.

Full native repository and one-edit successor validation: Four isolated producer, explicit-optout receiver, default receiver and independent reference processes completed with confirmed cleanup. All 299 projections, 13 table families, history, heads, receipts and canonical bytes matched for the full 300-file initial scan and complete one-edit successor. All 16 preparation-work counters matched. Initial preparation: 80.497s optout versus 61.851s default; successor: 94.480s versus 69.583s. Combined 174.976s versus 131.434s (24.9% less elapsed in one sequential pair). These timings compare the candidate default with its optout, and do not establish an end-to-end improvement over the previous immediate-admission native campaign. Default helper counts were 168 JSON parses plus 83700 clones versus 83868 optout parses, with equal total operation counts. Native receipt: `/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/deferred-poll-admission-checks-20261013-01/receipt.json`, SHA256 `1968985e3903f8ad36f1e31ad2909109354a513324672853b548782fb2a9b293`.

Production application and post-application CI: Both repository files exactly match the frozen qualified candidates. Application receipt: `/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/deferred-poll-admission-candidate-20261013-01/application.json`, SHA256 `960414736503ad86692c18bbded0bf07c10040ab42adbb0176bc4120f05ba2ba`. Persistent CI receipt: `/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/deferred-poll-admission-candidate-20261013-01/post-application-ci-checks.json`, SHA256 `849809995787c77e23c6cfbc97376717c77d31312962a3402613780680e06768`. MAIN was idle after each qualification and post-application test; no worker or model service was activated.

Finite integrity close and resumable cohort notes: Close destination: `/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/deferred-poll-admission-final-close-20261013-01.json`. The bounded closer retains all 1933 full predecessor dictionaries and the complete prior non-pin history. The superseded live scheduler dictionary remains historical: its old bytes are checked through the already-retained immutable archive, with no claim that the old original inode or metadata is current. The applied live scheduler is separately read twice and counted in member and physical-byte budgets. All other historical paths retain ordinary custody checks. The hard ceilings remain 1984 members and 2560 MiB cumulative integrity reads; no path is retired. Detailed qualification is governed by the completed close record. The subsequent 24-note sync records its exact result and carries source-grounded codebase_IR/intent_IR, signed workers, all 8D/384D/768D/4096D Leanstral heads, gte-small CUDA auto-selection and one federated model per compatible lineage forward; gradient synchronization through ipfs_accelerate_py/MCP++ remains future work.

The benchmark measured private warm-local-filesystem scheduler polling. Contended workers, network filesystems and cold storage need separate qualification. Full repository scan results belong to their separate native campaign. Model inference, CUDA, gte-small, 8D/384D/768D/4096D Leanstral heads, federation, gradient transport, signed dispatch and formal proof soundness retain their existing qualification boundaries.

JSON remains the shared scheduler state format. The private ephemeral marshal snapshot is derived only from this facade's own parsed plain JSON and is never received from peers or loaded as an external checkpoint.
