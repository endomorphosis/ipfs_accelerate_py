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
It has not been claimed as another successful native container run.

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
