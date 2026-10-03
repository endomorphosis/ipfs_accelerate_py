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
12288 MiB and 90-second context budget. Archive SHA256 was
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
