# Benchmark timeout configuration

Fresh full, no-index and native Codex preparations default to **600 seconds
(10 minutes)** for one agent execution phase. Both host launchers accept
`--agent-timeout-seconds` during `prepare`; the original 300-second profile
and explicit integer limits from 120 through 600 remain available. Fresh
preparations also set **1800 seconds (30 minutes)** for agent installation on
all three arms, separately from agent execution. Environment build and the
original verifier keep their existing bounds. There is still one attempt,
one concurrent trial and no automatic retries.

Run the preparation modules with the existing Harbor Python from the
`external/ipfs_accelerate` checkout. For example:

```bash
../../.venvs/terminal-bench-harbor/bin/python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark prepare --dataset /absolute/dataset --archive /absolute/fresh-runtime-archive --output /absolute/fresh-full-trial --arm full --agent-timeout-seconds 600
../../.venvs/terminal-bench-harbor/bin/python -m benchmarks.agent_supervisor.container_coding.native_codex_baseline prepare --dataset /absolute/dataset --output /absolute/fresh-native-trial --agent-timeout-seconds 600
```

Use `--arm no-index` with a separate fresh output directory for that comparison
arm. Use the same timeout on every arm being compared. Preparation validates
configuration and performs the existing preflight; execution remains a separate
one-shot operation. `execute` and `collect` use the retained preparation budget,
so passing a fresh-timeout option to those operations does not rewrite a trial.

## One shared deadline and bounded finalization

The supervisor adapter receives the same budget as Harbor. Instruction upload,
public-input capture, driver work and evidence export share one monotonic
deadline. Immediately before dispatch, it allocates the integer remaining time
minus 30 seconds to the driver: 10 seconds for execution grace, 15 for three
bounded exports, and 5 for final receipt work. With no elapsed transport a
600-second run allocates 570 seconds; real elapsed time lowers that allocation.

The driver's direct-entry default is 570 seconds, with explicit bounds of
90–600. Its existing 40-second cleanup reserve remains inside that allocation.
Planning still has a maximum of 90 seconds. Cold context, preparation and
planning consume time before coding, so neither 600 nor 570 promises that much
coding time. Preparation and ordinary/symbolic planner receipts record the
actual driver budget. The generated coding worker accepts up to 600 seconds;
the existing router and signed runtime caps already support that value.

If transport leaves less than 90 seconds after adapter reserves, the adapter
records budget exhaustion and refuses dispatch. The accepted 120-second lower
bound leaves no transport allowance and will usually exhaust before launch;
300 and 600 are the comparison profiles covered here. Limits are never rounded
up to bypass cleanup or the outer deadline.

## Fresh runtime archives and historical evidence

Rebuild the runtime archive with the existing `terminal_deployment bundle`
workflow, preserving the selected checkpoints and native extensions. Fresh
supervisor preparation verifies the actual archived timeout policy, driver,
worker deployment and indexed preparation against current source, as well as
the archive digest and manifest entries. An older archive must be rebuilt
before a fresh preparation can proceed. New immutable preparation directories
are required; old configuration files and reports are not upgraded in place.

Native preparation also pins the shared timeout-policy source. Execution
rejects changed pinned policy or a preparation/configuration budget mismatch
before launching Harbor. Fresh preparations record and validate the shared
installation allowance through Harbor's `override_setup_timeout_sec` field.
Historical records without an execution-budget field retain the legacy
300-second interpretation. For installation, a legacy explicit override is
retained; a missing override remains an unknown framework default. Historical
native records without a policy pin report its currentness as unknown. Receipt
collection preserves the original preparation bytes and records budget/profile
mismatches explicitly.

The targeted qualification evidence is retained under
`artifacts/supervisor-extended-timeout-20261012-01` in the workspace. Its
[validation report](../../../../artifacts/supervisor-extended-timeout-20261012-01/validation.json)
records actual pytest reports, mandatory AST seals, source hashes and lint
results. Checks use the existing Harbor 0.23.0 packages through a read-only
overlay and isolated fixture catalogs. That 142-check qualification batch
included no new provider, Docker or original-task acceptance run.

## Subsequent live pilot

The subsequent [pilot validation](../../../../artifacts/supervisor-timeout-runtime-preparation-20261012-01/validation.json)
and [retained comparison](../../../../artifacts/supervisor-timeout-runtime-preparation-20261012-01/comparison/comparison.md)
record actual Docker and model executions on one original task. Full indexed,
no-index and native Codex each passed the original verifier with the same
600-second work allowance and 1800-second installation allowance. Their
observed input-plus-output token totals were 22,073, 294,688 and 203,219,
respectively. Cached input is already included in those totals.

Full indexed supervision used a native Doctor contract candidate after one
planning-model session, with no coding-model invocation. Its observed total
was 89.1% lower than native Codex on this task. Semantic context projection
remained disabled and the slot policy remained `legacy@1`; the result does
not measure whole-codebase minification or establish a general benchmark
advantage. Cold indexing and refresh were included in supervisor agent time.

The first native attempt stopped during installation at Harbor's implicit
360-second limit, before agent execution or verification. That failed attempt
is retained separately with unknown token usage. Matching its installation
allowance to the supervisor's existing 1800 seconds required four host/test
file changes and passed 19 additional checks with mandatory AST seals. A
fresh native attempt then passed. The runtime deadline sources and both
archives remained unchanged by that setup correction.
