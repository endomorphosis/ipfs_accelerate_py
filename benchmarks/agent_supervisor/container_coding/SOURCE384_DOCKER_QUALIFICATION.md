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

The subsequent [top128 wheel payload diagnostic](../../../docs/agent_supervisor/evidence/source384-wheel-payload-cache-diagnostic-20261003/README.md)
passes 30 helper and instrumentation controls. All advice stages pass, including
131 installed payloads totaling 565,702,269 logical bytes. Initial Source384
preparation returns, but final initial-context validation fails at the second
source observation around cold replay. No inference artifact was exported;
model-load and decoded-coverage counters remain unavailable.

Eighteen source-free stage events localize 73.285 MiB of anonymous growth to
the preceding observation, with file cache nearly unchanged. The next observation
enters and fails with 8,561 MiB available against the 8,602-MiB requirement.
These boundary samples do not capture each internal admission decision.
Owner teardown releases memory afterward. The trace does not yet distinguish
DuckDB buffers from Python allocations or allocator retention. Initial context
lasts 177.945 seconds and the native probe 189.732 seconds. The same resource
limits and deadlines remain in force, the container is removed, and no provider
or verifier runs. Production integration and the full task remain unqualified.

## Coherent worker context construction

The [worker context change](../../../docs/agent_supervisor/evidence/worker-context-bundle-snapshot-20261003/README.md)
loads one fresh task-bound nomination per construction, replacing eight
separate metadata loads initially and eleven on a complete-context retry.
Semantic, retrieval and world fields come from that immutable snapshot. Each
artifact retains its source/digest checks; original read and evidence order
is preserved. Direct helper calls and subsequent dispatches validate anew.

Thirty-six actual-source tests pass with no failures or skips. The eight new
native-class controls use controlled Source384 and artifact-reader seams;
the remaining 28 are compatible existing context tests. A separate broad
legacy test module still fails collection on a removed Copilot timeout import.
This component result establishes neither a benchmark speedup nor a fix for
the earlier cold-replay memory refusal.

## Explicit setup cache integration

The [production integration controls](../../../docs/agent_supervisor/evidence/source384-setup-cache-production-20261003/README.md)
cover the opt-in `source384-native-aarch64-dontneed@1` setup policy in both
Harbor and the ordinary native qualifier. The policy binds the archive,
four public executable copies and 131 installed extension/wheel payloads to
their exact source hashes. Advice runs after worker setup and before context
construction. Omitting the policy preserves the existing setup behavior.

All 204 actual-source tests pass with no skips. Six regression controls expose
the earlier timeout bug: `TimeoutError` is an `OSError`, so the best-effort
filesystem handlers swallowed the deadline exception. Both advice loops now
propagate it. This component result does not qualify the full Docker profile;
the memory admission failure still requires a fresh production-archive run.

## Manifest replay and report lifetimes

The current local candidate reuses a validated private manifest only after a
fresh bounded CAS read, exact byte-key match and fresh CID check. Native reader,
JSON dependency, producer, registry and instance guards remain mandatory;
custom readers retain their ordinary validation path. Compaction shares exact
immutable metadata inside the private graph while keeping returned records
detached. The consumer also releases its parsed input bytes before native
replay, and the datasets validator releases the duplicate replay graph before
its second source observation. Final source/model/digest checks are unchanged.

Datasets qualification records 273 distinct passing controls: 266 in the broad
actual-source run and seven native Source384 cases in a configured-client retry.
The supervisor passes eight actual lifetime controls and all 13 existing
real-checkpoint context controls. The native retry and supervisor context run
join the exact existing four-CPU/9,830-MiB scheduler through its supported API,
using the same ledger, sampler, headroom and deadlines. Earlier host-default
configuration refusals remain retained. The foreign lease had naturally ended
before these retries; this does not qualify host-default startup or concurrent
operation alongside that lease.

A single instrumented, source-only three-arm comparison measures baseline and
combined cold observations at 14.550 and 14.416 seconds, and warm observations
at 5.339 and 4.587 seconds. Accounted private memo retention decreases from
89,210,779 to 29,251,864 bytes, and the sampled 40.516-MiB warm-load anonymous
allocation increase is absent. Anonymous RSS at the second observation return
is nevertheless higher: 285.156 MiB combined versus 258.812 MiB baseline.
These measurements establish neither an admission-memory fix nor a general
throughput improvement. They exclude the source-report lifetime changes and
numerical inference.

The [closed lifetime evidence](../../../docs/agent_supervisor/evidence/source384-replay-lifetime-20261003/README.md)
retains the two expected pre-fix lifetime failures, controlled object-release
checks and actual-source results. The datasets candidate and three-arm evidence
are in local datasets commit `0d6ed4b7d3dfd1935c60a3db414dafb0ad15ffef`, under
`docs/software_contracts/evidence/source384-manifest-replay-candidate-20261003/`
and `docs/software_contracts/evidence/source384-manifest-combined-comparison-20261003/`.
Their manifests are respectively `60da8174b275aac1ddf967e50ac5bc5fe2d73bdc537725ce75f7b3c98bb6b0db`
and `a85f841e16040803863c5845d23e6302baecfe9563b67abfdc80c0d7ecc52a48`.

The [fresh ordinary Docker qualification](../../../docs/agent_supervisor/evidence/source384-production-cache-qualification-20261003/README.txt)
passes setup, native START and all three cache-advice populations, then fails
at the second source-observation child lease during cold replay. Its archive
contains 26,918 verified files. The probe takes 170.816 seconds, including
159.394 seconds in initial context; the controller takes 690.414 seconds.
Cleanup and unchanged runtime/task hashes are verified.

Preflight reports 9,350 MiB available; the post-unwind sample reports 8,667 MiB.
The latter is not a decision-time sample and cannot establish the refusal's
specific cause. No inference artifact was exported, so model-load and coverage
counts remain unavailable for this run. There are no provider or verifier calls.
The five-CPU/12-GiB profile, admission rules and deadlines are unchanged. The
full task remains gated on qualification; RPI-019 remains open and the backlog
remains 18 closed criteria out of 32. No new reward or token score is available.


## Ordinary qualification with verified GTE advice

The [subsequent ordinary qualification](../../../docs/agent_supervisor/evidence/source384-lifetime-advice-qualification-20261003/README.md)
passes setup, native START, checkpoint inference, context replay and cleanup in
the unchanged five-CPU/12-GiB profile. Source384 completes in 76.884 seconds
against its 90-second deadline; initial context takes 137.809 seconds, the native
probe 159.484 seconds, and the controller 423.840 seconds including setup. A
separate warm observation takes 10.264 seconds. The 33 frozen runtime pins and
public task inputs remain unchanged.

The 220 signed inputs yield 31 Python files and 944 functions. Of 128 selected
units, 127 decode into unverified candidates and one exceeds the token limit.
All 127 candidates remain unsupported by the source-contract guard; the planner
summary exposes two samples and explicitly omits 125. Actual inference uses one
model load, with zero provider calls, training steps or proof authority.

The shared GTE verifier now optionally advises the same nine fully verified
file descriptors after all byte counts, hashes and identities pass; Source384
binds this requested policy in its inference key. Other callers keep the
unchanged default. The caller also drops completed construction objects before
fresh validation. Forty datasets unit controls and seven native checkpoint
controls pass; 18 caller controls pass. The coordinated host consumer suite has
11 passes and two initial admission refusals under a foreign 4096MiB reservation
in its later 9830MiB ledger. Those refusals remain distinct from the current
container success. Failure diagnostics pass 39 controls without changing policy.

This is one retained successful run with multiple changes and uncontrolled host
conditions, not measured cache reclamation or a general speedup. It permits a
fresh full task trial; it supplies no benchmark reward or token score. RPI-019
remains open, automatic post-publication Source384 successor inference remains
unavailable, and the backlog stays 18 closed criteria out of 32.


## Full trial reached planning, then failed receipt transport

The [subsequent full Harbor trial](../../../docs/agent_supervisor/evidence/source384-full-trial-receipt-failure-20261003/README.md)
received official reward **0**. Initial context returned in 139.991s, including
78.254s for Source384 within its unchanged 90s deadline. Planning returned in
68.953s with two goals and one task. Task-context publication then rejected the
78,563-byte selected receipt against the archived 32,768-byte inline transport
limit. No coding worker was dispatched or production supervisor activated. The
driver returned 1; Harbor and the host controller returned 0 after collecting
the failed task result. The later receipt-reference fix is separate evidence.

One llm_router → Codex CLI planning call recorded 21,497 input tokens and 1,090
output tokens: **22,587 total**. The 11,264 cached-input and 151 reasoning-output
tokens are subsets, not additional tokens. Cost is unavailable, billing totals
are unverified, and the requested output cap was not enforced. This is usage
from planning in an unsuccessful trial, not a completed-work token score or a
matched-arm efficiency advantage. All 127 decoded Source384 candidates remained
unsupported by the source-contract guard; IntentIR preprocessing remained
fail-open with no selected checkpoint.

Actual cgroups observed five CPUs and 12,288MiB. The driver took 244.323s, Harbor
agent setup 536.169s, agent execution 248.231s, and the outer controller 816.954s;
these nested intervals are recorded separately. The exact container was removed
and worker cleanup returned 0, while the native remaining-processes receipt is
null. Post-task observation covered unchanged `bottle.py` and missing
`report.jsonl`, not every original input. RPI-019 and RPI-020 remain open; the
backlog stays **18/32 closed**.


## Runtime-captured applicability and symbolic planning

The [actual-checkout integration evidence](../../../docs/agent_supervisor/evidence/header-intent-runtime-nomination-20261003/README.md)
records 354 distinct passing tests. A reviewed static IntentIR operation selector
is signed before initialization. Real SecurityIR Source384 checkpoint inference
and native source capture produce a separate runtime nomination; bounded Z3 checks
then establish a narrow operation precondition for the symbolic planner. The
authored normal-flow case admits two goals and one task with zero planner provider
calls, preserves the original signed manifest and revalidates admitted context
without another model load. It does not repair code or mark the security goal done.

The explicit `terminal-source384-config@2` profile includes the reviewed selector
CID, the leased checker profile and the hash of the runtime-installed Z3 binary.
The hash must be established for the selected container environment; copying a
host hash is not a valid substitute. Deployment checks it before native startup.
The `qualify` command accepts `--intent-requirement-contract` alongside
`--source384-context`, and passes the exact reviewed mapping into preparation.
Harbor setup/run and the qualifier reject missing or mismatched selections.

The native tests use isolated declared scheduler resources. Docker APIs are mocked
only in transport tests. Subsequent ordinary Docker qualification and the fresh
full task trial are recorded below; component results do not imply a task score.

## Captured-header Docker qualification and full trial

The [captured-header archive qualified](../../../docs/agent_supervisor/evidence/captured-header-docker-qualification-20261003/README.md)
through native START/STOP, checkpoint inference, source capture and warm replay.
Source384 took 81.664s, initial context 143.024s and the native probe 165.779s.
The selected checkpoint, GTE assets, five-CPU/12-GiB envelope and deadlines were
unchanged. The reviewed intent mapping is administrative coverage; its review
token cost is unavailable and excluded from runtime, so these timings cannot
support a complete efficiency comparison.

The [fresh full Harbor trial](../../../docs/agent_supervisor/evidence/captured-header-full-trial-20261003/README.md)
received official reward **0**. A source-currentness observation immediately
after worker return/output validation refused a resource lease. Preparation
took 10.851s, initial context 98.124s and the driver 109.629s. Planning, Doctor
and coding were not reached. Zero provider invocations describe a failed prefix,
not completed-work token performance. Post-unwind headroom/backoff observations
do not establish the admission-time cause. The container was removed.

## Execution import lifetime and remaining timing gap

The [driver import fix](../../../docs/agent_supervisor/evidence/supervisor-import-lifetime-20261003/README.md)
defers Doctor/native execution dependencies until their phases, under the same
work deadline. All 56 focused controls execute and pass in a fresh store.
Fresh-process host RSS falls from 189280 to 44916 KiB at import and from 297432
to 246880 KiB after shared imports and a tiny vector index. These single-process
samples establish less retained memory, not a container failure diagnosis.

The [new ordinary Docker attempt](../../../docs/agent_supervisor/evidence/source384-import-lifetime-docker-20261003/README.md)
passes deployment but times out in the numerical worker. Its probe takes
109.868s. The qualifier does not import the full driver, so this attempt does
not evaluate the deferred imports' effect in a full task. The full task retry
is withheld after the failed prerequisite.

A [host worker diagnostic](../../../docs/agent_supervisor/evidence/source384-host-worker-diagnostic-20261003/README.md)
completes 128 rows with one model load under the existing shared scheduler.
The separate [container stage diagnostic](../../../docs/agent_supervisor/evidence/source384-container-stage-diagnostic-20261003/README.md)
records where the unchanged preparation deadline is spent. Instrumented runs
are not production qualifications or official benchmark outcomes. All earlier
failures are retained. RPI-019 and RPI-020 remain open; the backlog is still
**18/32 closed**, with no new completed token score or matched-arm advantage.

The [ordinary retry](../../../docs/agent_supervisor/evidence/source384-warm-replay-refusal-20261003/README.md)
returned initial context in 156.106s but refused warm replay. A separate
[reclamation diagnostic](../../../docs/agent_supervisor/evidence/source384-reclamation-diagnostic-20261003/README.md)
released only 8756 KiB through malloc_trim with unchanged live context; it began
with much more headroom and did not reproduce the refusal. Allocator trimming
was not added to production.

The [qualifier bulk-copy fix](../../../docs/agent_supervisor/evidence/source384-probe-bulk-lifetime-20261003/README.md)
releases raw inference JSON and parsed rows after cold checks, retaining only
export-binding metadata before independent replay. All 37 actual controls pass.
The [fresh final archive](../../../docs/agent_supervisor/evidence/source384-bulk-lifetime-io-refusal-20261003/README.md)
stops earlier in initial-context replay, with proof_io_stall retained after
unwind (43.9 percent sampled I/O stall, 11491 MiB available). Its probe is
170.903s; the full task is not launched. The bulk-copy boundary is not reached
in that failed run. Read-only external cgroup and public-asset residency samples
are disclosed separately; they do not identify the complete charged cache
population, prove the admission-time cause or support a timing advantage.


## Frozen retry and admission-time diagnostics

The [unchanged archive retry](../../../docs/agent_supervisor/evidence/source384-frozen-retry-20261004/README.md)
passes deployment in 258.012s, then refuses the final source-currentness lease
after inference has been saved and reloaded. Initial context takes 99.094s,
the native probe 111.573s and the controller 449.676s. The later scheduler
snapshot retains proof_memory_stall; the post-unwind sample reports 14.59
percent memory stall and 11320 MiB available. These are not admission-time
measurements. Source/task pins and container cleanup pass. No full task is
launched, and no new official reward or completed token score is available.

The subsequent [request-local diagnostic change](../../../docs/agent_supervisor/evidence/source384-admission-observation-20261004/README.md)
retains the existing primary proof-gate sample on timeout/cancellation errors.
The qualifier and full supervisor expose this bounded record separately from
post-unwind samples. It distinguishes a fresh refusal from an existing cooldown;
a cooldown created by another request has no invented sample. Capacity,
fairness and secondary pressure decisions remain outside the recorded scope.
The record grants no authority and does not identify the cause of host pressure.

All 225 focused controls execute and pass: 121 shared scheduler tests and 104
supervisor tests. Resource formulas, thresholds, sampler calls and deadlines
are unchanged. This diagnostic was not in the frozen retry; its new combined
runtime requires a fresh archive and ordinary Docker qualification. RPI-019
and RPI-020 remain open, and the backlog stays **18/32 closed**.


## Native diagnosis, qualified lifetime fix and checked Doctor candidate

The [native request-local observation](../../../docs/agent_supervisor/evidence/source384-native-admission-20261004/README.md)
captures a post-worker source-fence refusal at 8601 MiB available, one MiB below
6144 MiB of root reservations plus 2458 MiB headroom. All sampled stall metrics
are below their thresholds. The final gate is the resulting cooldown; this
record identifies that refusal without identifying the source of memory charge.

The [construction lifetime fix](../../../docs/agent_supervisor/evidence/source384-construction-lifetime-20261004/README.md)
releases completed checkpoint/input objects before the next source fence and
construction graphs before independent replay. Three pre-fix failures become
17 passing lifetime controls; seven actual checkpoint tests pass after correcting
a host runner package-path mismatch, whose refusal is retained separately.

The [fresh ordinary qualification](../../../docs/agent_supervisor/evidence/source384-construction-docker-20261004/README.md)
passes. Source384 takes 88.944s within its 90s deadline, initial context 150.983s,
warm observation 10.329s and the native probe 173.323s. The margin is narrow,
and one run does not establish a causal or general performance improvement.

The [full task](../../../docs/agent_supervisor/evidence/source384-doctor-candidate-full-trial-20261004/README.md)
reaches two goals, one task and a checked Doctor candidate with zero provider
calls. Z3/Lean validate the reviewed local header-control contract; whole-program
security and natural-language semantic alignment remain unproved. The initial
index is reused, with 426 fact rows and 531 full semantic capsules reduced to
one worker capsule of 27276 bytes. No candidate is dispatched or published.

| Full-task phase | Seconds |
| --- | ---: |
| Preparation | 11.006 |
| Initial context | 138.971 |
| Symbolic planning | 33.712 |
| Context preparation | 20.915 |
| Doctor candidate/proof | 35.585 |

Official reward is **0**. At implementation setup, remaining(25) rejects the
remaining work budget while calculating a model timeout that the candidate
route does not use. The [route-aware correction](../../../docs/agent_supervisor/evidence/doctor-candidate-budget-20261004/README.md)
passes 110 supervisor controls, preserving the 245-second work cutoff and
40-second cleanup reserve. That later driver change is not part of the native
runs above and still requires a fresh Docker generation.

Preparation time must fall enough to leave useful native execution time.
The next measured targets are semantic reconstruction (39.527s within initial
context), planning/context replay and Doctor proof construction. Any immutable
reuse or parallel construction must retain exact input/producer bindings,
bounded shared leases and fresh source/proof gates. Native dispatch, accepted
publication and successor validation remain unqualified. No completed token
score or matched-arm advantage is claimed; the backlog remains **18/32 closed**.

## Immutable preparation reuse and bounded AST observation

The [preparation reuse controls](../../../docs/agent_supervisor/evidence/semantic-preparation-reuse-20261004/README.md)
cover call-local immutable graph identities and bounded canonical-pointer syntax
reuse. All 494 selected controls pass. On the public source input, all 6261
semantic blocks, worker payload and Doctor diagnostic bytes remain identical.
The graph timing pair lacks contemporaneous producer pins and is only a host
diagnostic; the separately pinned pointer profile is also a component measurement.
Neither establishes a Docker or completed-task speedup.

The [first fresh Docker archive](../../../docs/agent_supervisor/evidence/semantic-preparation-source-timeout-20261004/README.md)
passes deployment but exceeds the unchanged Source384 deadline during its
post-worker source-currentness observation. Initial context fails after 98.027s;
the native probe takes 110.208s. Semantic context construction is not reached,
so the run does not measure graph reuse. No admission-time refusal is attached,
and later resource samples do not establish the cause. Source/task pins and
container cleanup pass; no full task is launched.

The [subsequent observation controls](../../../docs/agent_supervisor/evidence/source-observation-batching-20261004/README.md)
pass 207 combined integration tests and seven actual checkpoint/GTE inference
and replay tests. Including the separately retained span and batch controls,
the package contains 285 unique passing test identities, with repeated
executions counted separately. Active AST reconstruction uses fresh bounded
SQL snapshots: the 31-AST control falls from 403 SELECTs to 13. Oversized
multi-AST reads raise an explicit limit error and are split before the next
part is loaded. Single-AST limits, canonical relational reconstruction, CAS
verification, cancellation, source recapture and final head checks remain.
A fresh blob/file/revision fence catches invalidation during artifact reads;
it is not a second relation-table audit or a transaction across source files
and SQL. Encoded-input and row limits are not hard RSS limits.

Function extraction now builds one source-local line index while retaining
per-function byte-map and isolated AST checks. Twenty alternating public Bottle
extractions preserve all outputs; median host CPU falls from 0.400s to 0.201s.
These component results do not establish completed-work efficiency.

The separate [host observation diagnostic](../../../docs/agent_supervisor/evidence/source-observation-host-diagnostic-20261004/confounded-host-diagnostic-summary.json)
was interrupted under observed CPU starvation and its outer controller later
timed out. Normal lease renewal was verified, but no AST-query timing pair
completed. The retained database, source and CAS hashes are unchanged; the
owned process is absent and its isolated ledger has no leases or waiters.
This diagnostic supplies no performance comparison.

The [fresh Docker generation](../../../docs/agent_supervisor/evidence/source-observation-docker-20261004/README.md)
passes with unchanged checkpoint, GTE, solver, reviewed intent and resource
limits. Source384 takes **83.774s / 90s**, initial context **143.801s**, warm
observation **10.988s** and the probe **167.404s**. Source/task pins and cleanup
pass. Actual inference still returns 127 unsupported, unverified candidates
and one token deferral from 128 selected units; no learned proof is granted.
This single run has narrow headroom and does not establish a general speedup.

The [fresh full task](../../../docs/agent_supervisor/evidence/source-observation-context-refusal-20261004/README.md)
receives official reward **0**. Source384 passes in 77.595s; driver initial
context takes 136.021s and symbolic planning 32.598s, producing two goals and
one task with zero provider calls. Context then refuses a source-currentness
child lease before Doctor or coding. The request-local gate reports **8593 MiB
available against 6144 + 2458 = 8602 MiB required**, while stall thresholds pass.
This identifies the gate refusal, not the source of the memory charge.

The driver fails after 223.031s; source/task pins and exact-container cleanup
pass. The outer controller returns zero after 473.885s, which is not a task
success. Zero provider calls describe a failed prefix. No completed token score
or matched-arm advantage is claimed; the backlog remains **18/32 closed**.

The later [admitted-context lifetime fix](../../../docs/agent_supervisor/evidence/admitted-context-lifetime-20261004/README.txt)
retains six scalar identities and releases completed construction aliases before
the existing fresh gate. All **188 supervisor controls execute and pass** in
new per-run stores. Test-only collection establishes collectibility, with no
production GC or measured RSS reduction. Original-owner regressions, earlier
witness failures and a seal-skipped run remain separate evidence. This fix was
not present in the preceding trial; recovery of its admission boundary still
requires a fresh Docker generation under the same limits.


The [fresh lifetime-fix archive](../../../docs/agent_supervisor/evidence/admitted-context-lifetime-pressure-refusal-20261004/README.md)
passes deployment in 215.209s but refuses an initial-context child lease before
numerical inference. The primary gate records **14.71 percent memory stall
against a 2 percent limit**, with sufficient headroom (8746 MiB available versus
8602 MiB required). Initial context fails after 82.569s; the probe takes 94.685s
and the controller 378.763s. Exact source/task pins and cleanup pass.

The changed admitted-context release boundary is not reached. Its memory effect
and recovery of the preceding full trial remain unmeasured. This failed archive
is not qualified by the earlier archive's success: no new full task is launched,
no official verifier runs, and no new reward or completed token score is available.
The sampler conservatively uses the maximum host and visible cgroup/ancestor
memory `full avg10` percentage. Its aggregate receipt identifies the refusal but
cannot attribute it to either scope. A source review found no units or stale-sample
defect. Future diagnosis needs separate pressure scope and memory-composition
observations under the same gates; the failure alone does not justify reducing
reservations or raising pressure thresholds. The backlog stays **18/32 closed**.


## Pressure attribution and fresh AST validation

The [current component generation](../../../docs/agent_supervisor/evidence/pressure-replay-components-20261004/README.md)
passes **212 supervisor controls and 308 unique datasets controls**, including
seven source-unit cases: one bounded-preparation case, five source-map tamper
cases, and one actual pinned-checkpoint/GTE inference case with warm and
reopened-registry replay in the same host process. The numerical worker uses
a fresh subprocess; these are not seven independent inference runs.
Every selected run has zero
failures/errors/skips and unchanged producer pins. Earlier failed or source-drift
runs remain separate, nonqualifying evidence.

Primary-gate errors now retain anonymous host/cgroup PSI readings from the
existing samples. Missing and malformed readings differ from an observed zero;
all ancestors still contribute to the unchanged scalar maxima even when the
bounded attribution list omits them. Thresholds, reservations, deadlines and the
scheduler ledger remain unchanged. The supervisor validates optional attribution
within its existing 4096-byte bound and preserves the separate six-field
post-unwind resource report.

Native current AST observation joins fully reconstructed fresh SQL projections
to freshly read, canonical/CID-checked, byte-identical CAS payloads. It avoids
reconstructing a second ASTRecord that was discarded. Historical/custom readers,
source/provenance checks, active identity fences and final source/head checks
remain. Reader replacement or unsupported ownership retains the historical path.
The public Bottle diagnostic preserves exact output/CIDs and reduces AST
reconstruction from two to one. Two alternating untraced samples per mode show
17.7 percent lower local component CPU; the small host diagnostic establishes
neither RSS reduction nor admission recovery or a general speedup.

The [fresh ordinary Docker attempt](../../../docs/agent_supervisor/evidence/pressure-replay-source-timeout-20261004/README.md)
passes deployment in 285.209s but fails the final Source384 currentness checkpoint
after inference/save/reload. Initial context takes 98.113s under the unchanged
cooperative 90-second Source384 deadline, the probe 110.534s and the controller
466.765s. No request-local admission observation is attached. Later post-unwind
resource values do not establish the timeout cause. Source/task pins and cleanup
pass; no full task or official verifier follows the failed prerequisite.

The earlier stage diagnostic measured cold index publication at 43.570s; it
motivates further profiling but does not isolate this run's cause. The component
evidence manifest also corrects three export labels for bounded logs already
present in nested packages, with no runtime/test change. Checkpoint weights,
proof authority and resource limits remain unchanged. There is no new reward,
completed token score or matched-arm advantage; the backlog stays **18/32 closed**.


## Reuse within native semantic state serialization

The [joined consumer controls](../../../docs/agent_supervisor/evidence/semantic-serialization-consumer-controls-20261004/README.md)
pass **558 distinct tests** with zero failures, errors or skips and stable source
pins. Native state serialization now hashes one freshly built identity payload
before adapting it to the existing detached public record. Custom serializers
retain historical dispatch; schema, exact bytes/CIDs, validation and source/proof
fences remain unchanged.

The datasets package `semantic-state-serialization-20261004` retains two fresh
public Bottle samples per mode: median local CPU 15.621 to 14.653 seconds and
wall 15.721 to 14.738 seconds, with identical manifest/AST bodies. This small
component observation does not establish a general speedup, RSS reduction,
admission recovery or completed task. The corrected joined-component package
clarifies its seven source-unit cases as one actual checkpoint/GTE inference
case, one bounded-preparation case and five tamper cases. Exact commands, exits,
test identities and producer pins are unchanged; earlier command/admission
wording remains historical evidence. The corresponding pressure-component
package has the same scope clarification.

The [ordinary Docker qualification](../../../docs/agent_supervisor/evidence/semantic-serialization-docker-20261004/README.md)
passes on archive `f8cd6fe9a3283a8df668394370b122743d77bc5ffa4ffcbd5b3ddf6f421d166c`,
with accelerate `e80fe6a59cacf4bb61386f0823cbd8545274173b` and datasets
`28b4a43c04e89c889b8fce7910e54fb8f67b05ff`. Source384 takes **69.851s / 90s**,
initial context **125.658s**, warm observation **9.320s** and the probe
**146.743s**. The checkpoint, GTE assets, solver, source population and
five-CPU/12-GiB profile are unchanged. This single run qualifies preparation
and replay for that archive; it does not isolate a causal speedup or establish
reliable admission recovery. All 127 decoded candidates remain unsupported
and unverified, with one token deferral among 128 selected function units.
No learned proof authority is granted.

The [first full trial](../../../docs/agent_supervisor/evidence/semantic-serialization-full-trial-host-pressure-20261004/README.md)
of the same archive, `fix-code-vulnerability__iRSHBCs`, receives official
reward **0**. Source384 passes in **83.119s** and symbolic planning produces
two goals and one task without provider calls. Context then refuses a
source-currentness lease before Doctor or implementation: the request-local
sample attributes **12.11% memory full avg10 to host PSI**, against the
unchanged **2%** threshold; the visible cgroup sample is 0%. This identifies
the rejecting scope, not which process caused host pressure. The separate
post-unwind sample is not substituted for that decision. Source/task pins
and exact-container cleanup pass. Zero provider calls describe a failed
prefix, not completed-task token savings.

The separate [frozen retry](../../../docs/agent_supervisor/evidence/semantic-serialization-full-trial-native-timeout-20261004/README.md),
`fix-code-vulnerability__rY4WVqt`, also receives official reward **0**.
Source384 takes **72.675s**, initial context **128.474s**, symbolic planning
**31.133s**, context **19.788s** and Doctor **34.219s**. The deterministic
Doctor proves the scoped local header contract and returns a
`candidate_ready` / `doctor_contract_candidate` dispatch with zero provider
calls. The receipt grants neither whole-program correctness nor discharge
of SecurityIR obligations or learned proof authority. The driver fails in
`native_execution` at **245.743s**, at the unchanged **245s** work cutoff;
the task does not complete. That phase label alone does not establish
successful START, candidate materialization or accepted publication. No
START receipt is present. The bounded traceback reaches source-currentness
validation during task-context nomination, ending in DuckDB AST record
reconstruction and its text validator when the work alarm fires. This
locates the interrupted call but does not profile its cumulative cost. No
request-local admission refusal is attached, and the post-unwind resource
sample does not establish a timeout cause. The Doctor dispatch evidence
must be read separately from the optional worker-materialization receipt
list; an absent list does not mean that Doctor did not run.

Accepted publication and current successor preparation remain unqualified.
The Source384 profile explicitly reports its original learned selection as
`successor_unavailable` after publication until a fresh successor is prepared;
post-STOP refresh budget alone cannot make that historical selection current.
No completed token score or matched-arm advantage is claimed; the backlog
remains **18/32 closed**.
