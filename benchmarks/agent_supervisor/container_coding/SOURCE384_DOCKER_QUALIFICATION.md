# Original-container Source384 qualification

This qualification exercises runtime deployment and pinned-parent preparation
on the unmodified public `fix-code-vulnerability` checkout. It does not run a
coding provider, the official verifier, training, or Lean, and produces no
benchmark score or token comparison. It preserves the complete original source
population; authored model fixtures are never inserted into the task.

Build a fresh portable bundle with the Source384 config described in
[the asset contract](SOURCE384_HARBOR_ASSETS.md), then run:

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_deployment bundle \
  --source /absolute/accelerate-checkout --datasets /absolute/datasets-checkout \
  --kit /absolute/kit-checkout --extension-dir /absolute/duckdb-native-extensions \
  --source384-config /absolute/pinned-source384-config.json \
  --output /absolute/fresh-runtime-bundle
python -m benchmarks.agent_supervisor.container_coding.terminal_deployment qualify \
  --task-dir /absolute/terminal-bench-2/fix-code-vulnerability \
  --archive-dir /absolute/fresh-runtime-bundle --output /absolute/fresh-qualification \
  --resource-profile source384-5cpu-12gib@1 --source384-context --no-codex
```

The named profile enforces 5 CPUs and 12288 MiB through Harbor. Use the same
`--resource-profile` for native Codex, no-index supervisor, and full supervisor
when preparing a subsequent matched benchmark. Defaults still use the task's
original resource settings. The qualifier checks actual cgroup limits and live
resource observations, exact relocated checkpoint/config and producer hashes,
complete original source hashes, one actual model load, and a warm observation
without neural replay. It exports the immutable native inference JSON and
checks its SHA256 before container deletion. Cleanup runs even if writing the
qualification receipt fails.

The container context probe has a 270-second overall bound, inside Harbor's
300-second exec bound. Native source, function, selection, memory and operation
deadlines remain active. Ordinary pinned runtime dependencies may be downloaded
during installation; the selected model runtime stays offline. Optional Lean
assets are omitted because this qualification grants no projection or proof
authority.

The first, retained 8 GiB generation deployed successfully in 96.26 seconds,
preserved all 218 tracked task files, and observed `cpu.max=500000 100000` and
`memory.max=8589934592`. Preparation then refused admission with
`LeaseTimeoutError` after 49.56 seconds. Its initial live availability was
6506 MiB; the default scheduler derives 1639 MiB headroom and requires at least
7783 MiB available for the 6144 MiB parent. That comparison explains a likely
memory refusal; the initial sample is not a native per-request reason receipt.
The new, explicit 12 GiB profile preserves scheduler rules and this failed
generation as evidence.

The first 12 GiB generation stopped earlier in the existing empty-supervisor
lifecycle check: START returned a receipt, but the two observations taken
0.75 seconds apart did not establish a stable healthy tree with an advancing
heartbeat. STOP fenced the tree and confirmed absence. The run ended after
116.93 seconds, before cgroup observation or Source384 preparation; its declared
12 GiB resources are not an observed successful admission. The native check now
waits within a bounded deadline for fresh heartbeat evidence while keeping
its health, birth-identity and shutdown requirements.

The final 12 GiB retry passed native START, stable heartbeat observation and
STOP, deployed in 83.69 seconds, and preserved all original sources. Actual
limits were `cpu.max=500000 100000`, `memory.max=12884901888`; live availability
was 10382 MiB against a derived default minimum of 8602 MiB. Native index
publication then refused at `apply_batch`'s pre-commit deadline check, before
inference: the initial-context phase took 137.49 seconds and the probe took
149.00 seconds. The 90-second native budget is cooperative, so this observed
overrun remains a performance gap. The final failure sample reported 10070 MiB
available and CPU stall 26.89%; those observations do not prove a particular
performance cause.

That executed generation also exposed merged stdout/stderr: diagnostics
preceded the final JSON, causing the host collector to raise `JSONDecodeError`.
The original merged log and exact final JSON diagnostic are retained without
rewriting the failed outcome. The current collector downloads a separate,
bounded JSON report file, preserving raw logs and nonzero exit status. This
collection correction is covered by 158 controls, including malformed and
oversized reports, after the 156-control generation used for the Docker retry.
The next generation below exercises this collection correction in Docker.

The controls use explicit Docker/model doubles.
Thirteen separate real-checkpoint consumer tests cover the unchanged native
consumer, including the 256-file signed population boundary. The static input
audit establishes 218 files, 30 Python files and 944 functions within the
declared input bounds; it does not establish graph admission, inference or
benchmark success. Actual Docker results are recorded separately from these
component and static checks.

The lifecycle owner has 28 unique passing controls across two retained runs:
the first had 26 passes and two native START failures before the new polling
check; the unchanged-source retry passed those two native cases at the same
20-second limits. This supports the bounded observation fix, not general
qualification under arbitrary host pressure. All three Docker attempts remain
failures for complete Source384 initial-context qualification. There is no new
benchmark score, token saving, successful in-container neural inference, or
source-qualified proof claim in this evidence package.

## Cold-publication optimization generation

A fresh run with the bounded canonical-content cache and batched AST SQL
publication passed deployment and the empty-supervisor lifecycle in 195.55
seconds. All 218 original tracked files remained hash-identical. The observed
cgroup limits were again 5 CPUs and 12288 MiB, with 10011 MiB available before
preparation. The portable archive SHA256 was
`80685f4df3811cbd163d140b4f05968f7e6d7ee87591a7279b70f6b3f74c2eb9`.

Cold index publication returned, but the first `observe_current` inside
`infer_shared_parent_units` exceeded the remaining combined 90-second context
budget before the neural worker started. The preparation wrapper took 10.40
seconds, initial context 100.86 seconds, and the complete probe 112.17 seconds;
the whole deployment and qualification attempt took 328.79 seconds. These
cooperative-deadline overruns remain failures. The final resource sample showed
9630 MiB available and 26.75% CPU stall; it does not isolate the cost of each
native stage or establish a resource-pressure cause.

The corrected collector retained the separate structured
`LeaseTimeoutError: codebase observation deadline exceeded` receipt, probe exit
status 1 and the complete merged traceback. The container was deleted. No
native inference artifact exists for this attempt because it stopped before
worker invocation; failed later-stage attempts also do not export partial
inference through this collector. Thirteen current-producer real-checkpoint
consumer tests passed separately in 45.96 seconds. Those component results do
not turn the refused Docker context into successful in-container inference.

This fourth Docker generation is frozen separately in
`docs/agent_supervisor/evidence/source384-docker-publication-20261003`, including
the exact source overlays, archive manifest, public input hashes, resource
samples, raw failure and current-producer component tests. The three earlier
generations remain unchanged in the preceding qualification evidence package.
There is still no new official benchmark score, token comparison, or
source-qualified proof claim.

## Canonical-reconstruction cache generation

A fifth fresh run added the bounded immutable manifest reconstruction cache and
closed catalog serialization checks, keeping the original 218 inputs, 5 CPUs,
12288 MiB and 90-second Source384 preparation budget. Archive SHA256 was
`43ba7c9bc830b9c6c684589db15ddad25a37339014a5287d41a962c10d98f9c3`.
Deployment passed in 1109.51 seconds, including a slow dependency download;
read-only samples showed the pinned PyTorch wheel growing by 17.56 MB over
66.49 seconds. Setup finished within its existing bounds and remains separate
from native context timing. Native START/STOP passed and every original tracked
file remained hash-identical. Actual cgroup limits matched the named profile,
with 10483 MiB available before context preparation.

This run passed the first repository observation and reached the bounded
source-unit worker, which then reported a timeout. The retained child output
contains import diagnostics, but does not establish a successful model load
or completed inference. Preparation took 10.52 seconds, initial context 98.30
seconds, the probe 109.86 seconds and the whole attempt 1242.32 seconds. The
structured failure and probe exit status 1 were collected, and the container
was deleted. No native inference artifact was exported. The final resource
sample showed 9997 MiB available, 27.40% CPU stall and 3.02% I/O stall; the
current receipt does not break down publication, observation and worker time,
so it cannot identify the remaining performance cause.

Thirteen real-checkpoint component tests passed separately in 43.83 seconds
against these exact producers. The new closed package is
`docs/agent_supervisor/evidence/source384-docker-reconstruction-20261003`;
earlier packages remain unchanged. Docker qualification is still refused,
with no new benchmark score, token comparison or source-qualified proof.

## Preparation stage diagnostics

Two instrumented runs reuse the fifth generation's exact archive, original
218 files, five CPUs/12288 MiB and native 90-second context budget. Seven and
nine wrapper controls pass; producer guards remain unchanged and all timing
events are retained. Cold preparation takes 79.331 and 79.481 seconds. The
first run reaches the worker with only 0.917 seconds remaining before child
admission; its timeout does not measure inference throughput.

The deeper run spends 39.979 seconds across 31 projection-persistence calls,
including row materialization and SQL execution. Scanning takes 14.140 seconds
and parsing 0.705 seconds. Its first observation receives 0.588 seconds and
refuses before worker invocation. Inclusive nested timings overlap; they do
not establish a production latency or speed improvement. A database-only
replay is needed to separate SQL calls from Python batching.

The independently retained runs, controls, analyses and cleanup are in
`docs/agent_supervisor/evidence/source384-docker-stage-diagnostics-20261003`.
These diagnostics identify preparation budget starvation without qualifying
Docker inference, a benchmark score, token savings or any source property.

## SQL replay and runtime dependency diagnosis

An exact replay of 31 retained projections preserves all 13 catalog tables and
cold native reconstruction. The native batched VALUES path takes 3.399 seconds
of execute time on the host and 40.269 seconds in Docker. An external column-list
UNNEST candidate reduces host execute time to 0.946 seconds, but Docker still
takes 38.692 seconds. Both paths bind the same 870,877 scalar values through the
same DuckDB 1.5.5 binary. The candidate SQL patch remains unapplied.

DuckDB's scalar conversion probes pandas types before ordinary Python scalar
types. With pandas absent, its optional import cache retries failed imports.
Four fresh subprocesses with standard import finders isolate real pandas
availability while retaining the same interpreter, DuckDB binary, NumPy and
36,864 mixed scalar values. Available-pandas execute times are 0.0215/0.0217
seconds; missing-pandas times are 0.9992/0.9958 seconds. Every returned value
matches. The separate import-count control records two failed import attempts
per non-null scalar. These small native controls establish dependency-related
conversion overhead; they do not measure full container qualification.

The deployment now pins pandas 3.0.2 and NumPy 1.26.4 in its base requirements.
Its native import probe checks and reports both before START. All 91 focused
deployment, transport and qualification controls pass, including missing-pandas
refusal. The source catalog, decoder weights and inference contracts are
unchanged. The closed diagnostic package is in datasets at
`docs/software_contracts/evidence/source-sql-column-performance-20261003`.

The first fresh archive with that correction has SHA256
`aeba7f2233a2b7dc042d6c6eb5d74debad64e0be36ae61196a0ebf9c425ea91e`.
It stops during the existing 600-second CPU PyTorch installation limit, before
pandas installation, native START or Source384 preparation. The whole attempt
takes 812.484 seconds. Its raw failure, tested owners, archive inventory and
container cleanup are retained in
`docs/agent_supervisor/evidence/source384-docker-pandas-20261003`. This setup
failure does not test the correction's full-container inference outcome.

## Packaged Torch wheel and inference publication

The next uninstrumented run used an explicitly selected Torch 2.13.0 CPU
wheel, verified against its SHA256 before transport and again after extraction.
Installation, the pandas/NumPy import probe, native START/STOP, and preservation
of all 218 original input files passed. Deployment took 65.266 seconds. Actual
cgroup limits matched five CPUs and 12288 MiB; the native 90-second Source384
budget and all other execution bounds stayed unchanged.

Source384 qualification remained false. Initial context failed after 78.045
seconds, and the full probe took 89.425 seconds. The traceback reached the
publication helper import and raised `SchemaLakeError: canonical workspace
checkout is unavailable`. Frozen code and the traceback indicate that the
bounded worker returned and output/source checks passed before that import.
No model-load count, decoded coverage, final Source384 receipt, or committed
inference artifact was exported or verified. The container was removed. The
closed failure package is
`docs/agent_supervisor/evidence/source384-docker-wheel-20261003`, manifest
`bc0bea276d89b2f70367488d0cd0641c8190155ae7dbfeacfe17d921029d6272`.

The cause was inference publication importing a training module solely to
stage canonical JSON. That import also loaded training/proof workspace
initialization. The datasets fix moves the identical staging operation into
the existing shared Source384 owner; inference calls it directly and the
training helper delegates to it. Canonical serialization, registry staging,
source/model checks and deadlines remain unchanged. The canonical-workspace
guard itself is unchanged.

Validation remains separated by scope: 136 accelerate deployment, wheel,
transport and qualification controls; 43 datasets publication, inference and
training integration tests; and 13 accelerate tests using the actual pinned
checkpoint. These are distinct suites, not an official benchmark score. The
datasets regressions cover publication without importing training/proof
owners and replay after reopening the registry. The corrected Docker result follows separately.

## Successful original-source checkpoint inference

The fresh corrected archive has SHA256
`50ed24823090ca9034819bc27e350bbbd098de2345c120c2db4a3c720aceed6c`.
All 11,719 members passed independent byte/hash/mode verification. Relative
to the preceding wheel archive, only the three datasets staging-helper owners
changed. The checkpoint, GTE snapshot, supervisor runtime, original 218 files,
five-CPU/12288-MiB limits and all deadlines were retained.

Deployment passed in 458.743 seconds, including dependency installation and
native START/STOP. Preparation took 10.364 seconds. Actual Source384 capture,
inference and publication completed in 82.872 seconds against its cooperative
90-second budget. The complete initial context took 147.267 seconds, including
other indexing, hydration and evidence checks. The subsequent warm observation
took 13.403 seconds without rerunning neural inference. The whole native probe
took 171.962 seconds within its 270-second limit; the complete deployment and
qualification attempt took 656.825 seconds. One observed success does not
establish a latency distribution or a network-installation speedup.

The saved artifact verifies actual checkpoint consumption, one GTE model load,
and a successful CPU worker with a 12.736-second receipt. It accounts for all
220 permitted inputs (218 original plus two framework inputs), 31 Python files
and 944 functions. Of 128 selected units, 127 decode as unverified candidates
that the source-contract guard rejects as unsupported, and one exceeds the GTE
token limit. Another 737 units are deferred by selection limits and 79 have
unsupported normalization. No property becomes proved or source-qualified.

The container was removed. The closed evidence is
`docs/agent_supervisor/evidence/source384-docker-inference-publication-20261003`.
The full 4,505,485-byte native artifact remains local under SHA256
`2fc72e0de92a97547661885cc5dcb9c00f2257356a49efe8b6537adecb95ffb7`;
it contains benchmark source text, so the public package retains reviewed
metadata and verification receipts instead. No checkpoint weights or runtime
archive are included in that package.

This qualifies original-container deployment and the full Source384 initial
context path. It performs zero provider calls, training steps or official
verifier runs. A full task trial, automatic successor inference after accepted
publication, a fresh reward/token comparison and the header grammar extension
remain separate work. The 32-item backlog's closure counts remain unchanged.

For the successful closed package, `qualification.json` retains several
original host-relative receipt names. Resolve
`distinct_component_suites.inherited_accelerate_transport.artifact`
(`controls.xml`) as `host/controls.xml`. In
`command_provenance.actual_execution_receipts`, the four bare JSON filenames
resolve under `host/`; `docker-01/qualification.json` is already package-relative.
The manifest contains each resolved file and its digest. This clarification
preserves the sealed evidence bytes and does not change the test or run result.

## First full trial with native proof backends

A fresh archive adds the complete pinned Lean toolchain to the qualified
Source384 runtime. All 11,719 preceding members remain identical; 15,186 Lean
assets add 3,305,142,714 uncompressed bytes. Archive SHA256 is
`a83b157f478dbcb869f134e71bfd02b17bdb7f82f3de8d281cc6202f66464d0b`.
Existing deployment installs Z3, the pinned Codex provider and the worker
boundary. The single full-arm `fix-code-vulnerability` trial retains the
five-CPU/12-GiB profile, 300-second agent limit, 285-second driver limit and
245-second work cutoff.

The official verifier returns reward **0.0**. Preparation takes 10.369 seconds;
initial context refuses a resource lease after 37.438 seconds, before planning,
Doctor dispatch or coding. The driver takes 48.441 seconds, and Harbor records
51.914 seconds of agent execution. There are zero observed provider calls;
token and cost counters remain null. Intent preprocessing fails open because
no Intent checkpoint is selected. The deployment preserves all 218 originals;
the narrower post-agent public capture confirms unchanged Bottle and an absent
report. The container is removed. This trial did not retain a traceback or
actual cgroup sample, so its generic error alone cannot establish a cause.

A separate diagnostic repeats the exact archive, Codex setup and worker
boundary without provider calls or an official verifier. It observes the
actual five-CPU/12-GiB limits and reproduces refusal at Source384's root lease,
before indexing or inference. Available memory is 5,484 MiB at failure versus
8,602 MiB required by the existing 6,144-MiB request and 2,458-MiB headroom.
The earlier sample records 6,487,437,312 bytes of file cache; CPU, memory and
I/O pressure samples remain below their refusal thresholds. No admission
policy, limit or cache state is changed by this diagnostic. These observations
establish the reproduced memory shortfall, not a retroactive resource sample
for the original trial.

The driver now preserves a bounded traceback of file/function/line metadata,
the failing phase and a canonical resource sample taken while handling the
error. Sampling failures preserve the primary error and cleanup. Thirty
controls pass; no source text or local variable values are exported. The
closed evidence package is
`docs/agent_supervisor/evidence/source384-full-admission-diagnostics-20261003`.
This improves diagnosis and leaves the full task acceptance gate open.

## Managed Python and Lean layout collision

Two bounded, manifest-only cache experiments refuse before issuing any cache
advice: the deployed `toolchains` ancestor is a symlink. A complete tar audit
finds 26,905 regular members, with no links, duplicate names or prefix collisions.
A separate setup-step observer then records the actual transition during
`python-runtime-install`: the normal Lean directory becomes a link to
`/opt/ipfs-supervisor/python`. The observer stops at that operation and removes
the container; it runs no inference, provider or verifier.

The pinned [UV 0.9.24 implementation](https://github.com/astral-sh/uv/blob/0.9.24/crates/uv-python/src/managed.rs#L164-L177)
migrates an existing sibling named `toolchains` when the selected Python
installation directory is absent. Deployment now selects
`/opt/ipfs-supervisor/python-runtime/python` for both Python installation and
virtual-environment creation. The interpreter entrypoint remains
`/opt/ipfs-supervisor/venv/bin/python`; Lean's path and all resource and time
limits are unchanged. No symlink check is relaxed.

Fifty-six focused controls pass with no skips. A real, pinned UV offline
control reproduces the original migration and verifies that the nested layout
preserves the Lean directory's inode and contents. Its missing-Python refusal
is expected; this control does not claim a successful Python installation.
The [closed layout evidence](../../../docs/agent_supervisor/evidence/source384-uv-layout-fix-20261003/README.md)
separates these results from the subsequent cache and inference qualification.
The layout fix alone does not establish sufficient memory or a task score.

## Cache diagnostics after the layout fix

Three subsequent instrumented runs retain the five-CPU/12-GiB profile and
all admission thresholds and deadlines. They use the corrected host deployer
with the preceding immutable `a83b157f` archive; they do not qualify a new
production archive or cache policy.

Archive-only advice succeeds for 26,905 files, then the Source384 root lease
is refused. Available memory at failure is 8,195 MiB against the existing
8,602-MiB requirement. Advice to four additional, independently pinned public
Codex executable copies first refuses an unprotected transport receipt. The
retry fixes only that receipt's ownership and mode, then passes all four
executable checks and advice operations. The corresponding helper suites pass
12, 14 and 18 controls per generation; these overlapping counts are separate.

In the retry, available memory rises from 5,256 to 8,194 MiB across archive
advice and then to 8,812 MiB across executable advice. The root lease is
admitted, but the first index child lease times out before source capture,
publication or numerical inference. Owner-store initialization may already
have occurred. The later failure-handling sample is 8,592 MiB; it is not the
exact child-admission sample. The scheduler adds zero new global memory for
children while conservatively checking outstanding root reservations plus
headroom. That behavior is consistent with the observed shortfall and does
not establish a child double-counting defect.

The [closed diagnostic evidence](../../../docs/agent_supervisor/evidence/source384-setup-cache-diagnostics-20261003/README.md)
preserves all three failures separately. Successful advice counts and logical
byte lengths are not measurements of reclaimed memory. No inference artifact,
model coverage, provider use, official task score or token improvement is
qualified by these runs. All containers were removed. Further library-cache
experiments remain separate until their outcomes and production integration
are qualified.

The next [seven-library diagnostic](../../../docs/agent_supervisor/evidence/source384-native-library-cache-diagnostic-20261003/README.md)
adds three installed DuckDB extension copies and four Torch libraries. Their
414,442,268 bytes match independent hashes from the pinned archive and CPU
wheel. All three advice stages pass, and available memory before context is
9,216 MiB. Twenty-five helper controls pass with no skips.

This run returns from index preparation and the numerical worker, and validates
the worker's output shape. The following source-observation child lease then
times out, before publishing an inference artifact. The retained scheduler
snapshot explicitly records `proof_memory_headroom` after unwind; the later
8,637-MiB memory sample is not the original decision sample. Initial context
lasts 97.014 seconds and the probe 108.467 seconds. All leases and waiters are
released and the container is removed. Control-flow evidence of a returned
worker does not establish published inference, model-load counts or coverage;
those counters remain unavailable. This run has zero provider calls and no
official verifier. The production cache integration remains unqualified.
