# Source384 Docker stage diagnostics

These are two independent, instrumented diagnostic runs, not production
qualification, official benchmark scores, token comparisons, or proof results.
Both reused runtime archive
`43ba7c9bc830b9c6c684589db15ddad25a37339014a5287d41a962c10d98f9c3`,
the complete original 218 tracked inputs, and the common 5-CPU/12288-MiB
profile. Native context remained limited to 90 seconds, the container probe
to 270 seconds, and Harbor exec to 300 seconds. No provider, official verifier,
training, or model download ran. Ordinary runtime dependency installation was
separate from native context preparation.

The wrappers changed in-memory method callables only. Runtime owner files and
timeout arguments stayed unchanged. Source-unit wrappers were installed after
the production import inside its native budget. The manifest producer guard
remained recognized and unchanged; CID cache snapshots did not initialize the
lazy registry cache. Events contain stage names, nesting, durations, numeric
timeout/memory arguments and exception types, without argument contents,
source paths, formulas, query text or model values.

| Observation | Worker-budget run | Cold-preparation run |
|---|---:|---:|
| Wrapper controls | 7 passed | 9 passed |
| Deployment including native START/STOP | 80.807 s | 102.660 s |
| Cold preparation | 79.331 s | 79.481 s |
| Initial context | 97.161 s | 98.219 s |
| Recorded events / dropped | 258 / 0 | 2916 / 0 |
| Manifest memo hits / misses / bypasses | 3 / 1 / 0 | 2 / 1 / 0 |
| Outcome | Worker timeout | Observation deadline refusal |

The first run reached the bounded worker with only **0.917 seconds** remaining
before child admission and subprocess allocation. The worker returned a timeout
after 0.955 seconds. Child import output establishes worker invocation, not a
successful model load or completed inference.

The deeper run attributed **39.979 seconds across 31 calls** to
`DuckDBASTStore._persist_projection`, 50.3% of the 79.481-second cold preparation.
This owner includes row materialization and SQL execution; individual queries
were not timed. Its largest call took 17.315 seconds, and the median took
0.419 seconds. Source paths were deliberately omitted. Scanning took 14.140
seconds. Catalog candidate validation took 10.832 seconds inclusive, or 8.876
seconds after subtracting its timed children. Parsing took 0.705 seconds.
Manifest sealing took 3.111 seconds inclusive, including its artifact writes.
The first observation received 0.588 seconds and refused after 1.112 seconds;
this run did not reach the worker.

Inclusive parent/child totals overlap. The analysis scripts subtract only
immediate timed children for exclusive totals; remaining time is not a finer
causal attribution. Instrumentation and run-to-run conditions prevent treating
these values as production latency or measured speed improvements. Resource
samples alone do not identify a pressure cause. Both containers were deleted.

`worker-budget/` and `cold-preparation/` retain each generation's independent
scripts, exact commands, raw wrapper returns/failures, reviews, control logs,
producer checks, phase events, source hashes, resource samples and cleanup.
The initial cold-wrapper controls caught a classmethod binding error before
Docker; the corrected generation passed all nine controls. Older control logs
are retained without treating them as the executed script generation.
`phase-analysis.json` in each directory is reproducible with its adjacent
`analyze_phases.py`. The cold directory also contains the per-call persistence
distribution.

`common/archive-manifest.json` binds the reused archive. `common/owners/` holds
only explicitly selected framework source files extracted from that archive
and verified against it. The package excludes the runtime archive, checkpoint
weights, mutable databases, authority state, hidden task verifiers and oracle
solutions. Reproducing Docker deployment requires the separately retained
archive with the stated digest. The closed manifest lists every included file,
size and SHA256. Earlier production-failure packages remain unchanged.
