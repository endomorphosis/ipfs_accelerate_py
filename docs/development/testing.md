# Testing

**Status:** Current
**Owner:** documentation-governance / package maintainers
**Audience:** developers, contributors, implementation agents
**Scope:** How to select and run focused, valid checks on this tree; how
offline, integration, hardware, and provider suites differ; and how missing
capability must be reported rather than hidden.
**Non-goals:** This page does not invent CI gates, markers, or fixtures that
are not configured. It does not claim optional hardware, network services, or
third-party providers are available on every machine.
**Sources:** `pytest.ini`; `pyproject.toml`; `test/conftest.py`; `test/`;
`tests/` (nearly empty legacy path); focused suites under `test/api/` and
root `test/test_*.py`; console scripts in `pyproject.toml` /
`setup.py`.
**Last-verified:** 2026-08-03 @ `d5f3aa5c6`; paths checked against live modules,
the test tree, and `pytest.ini` `testpaths`.
**Freshness triggers:** changes to `pytest.ini` discovery; relocation of
unified-CLI or agent-supervisor modules/tests; new hardware/provider markers
or skip policy in `test/conftest.py`.

This document is the **TestSelectionGuide@1** for contributors. Prefer the
smallest deterministic suite that covers the change; escalate only when the
changed surface requires network, hardware, or external providers.

## Install test dependencies

```bash
python -m pip install -e ".[dev]"
```

For broader local coverage, install the extras that match the features under
test (`testing`, `mcp`, `mcp-p2p`, `webnn`, `full`, and similar names from
`pyproject.toml`). An import success for the base package is **not** proof that
optional extras, CUDA builds, browsers, IPFS, or LLM/prover CLIs are present.

## Pytest discovery boundaries

Default discovery is **not** “everything under `test/`”. Root `pytest.ini`
sets:

| Setting | Current value (see `pytest.ini`) |
| --- | --- |
| `testpaths` | `ipfs_accelerate_py/mcp/tests`, `test/api`, `test/distributed_testing` |
| `python_files` | `test_*.py` |
| Common markers | `cuda`, `rocm`, `mps`, `webgpu`, `webnn`, `hardware`, `browser`, `integration`, `slow`, `model`, … |

Implications:

1. **Bare `python -m pytest`** runs only the configured `testpaths` (plus
   explicit paths you pass). Many useful modules under `test/test_*.py` are
   **outside** default discovery and must be named explicitly.
2. The near-empty top-level `tests/` directory is **not** the maintained suite.
   Prefer `test/` (singular). Historical docs that point only at `tests/` are
   stale.
3. Model suites under `test/improved/` and similar trees may be gated by
   flags such as `--run-model-tests` (see comments in `pytest.ini`). Do not
   treat a default green run as full model coverage.
4. Use `-m` only for markers declared in `pytest.ini` / `test/conftest.py`.
   Do not assume `pytest -m cuda` selects every CUDA-related file; open the
   module when in doubt.

## Test layout (maintained areas)

| Path | Role |
| --- | --- |
| `test/api/` | Contract, API, and agent-supervisor tests (default discovery). |
| `test/distributed_testing/` | Distributed test and CI service integrations (default discovery). |
| `ipfs_accelerate_py/mcp/tests` | MCP package tests (default discovery). |
| `test/test_*.py` (repo root of `test/`) | Focused integration and CLI modules; **path must be named** (e.g. `test/test_unified_cli_integration.py`). |
| `test/integration/` | Cross-component integration; name paths explicitly. |
| `test/hardware/`, `test/hardware_detection/` | Hardware discovery and backend behavior; capability-gated. |
| `test/integration/browser/`, `test/fixed_web_tests/`, `test/web_platform_tests/` | Browser and web accelerator behavior; capability-gated. |
| `test/ipfs_accelerate_py/` | Package-specific regression tests. |
| `examples/` | Executable examples and smoke demonstrations (not pytest discovery). |

There is no single authoritative `test/unit/` tree. Older documents that use
that layout are historical examples.

## Capability classes (select the right suite)

| Class | Intent | Typical command shape | Missing capability policy |
| --- | --- | --- | --- |
| **Offline / deterministic** | Schemas, parsers, pure logic, fixture-backed contracts | Explicit paths under `test/api/` or named root modules | Must **not** require network, GPU, browser, or external CLIs. Fail hard on logic bugs; do not skip away contract failures. |
| **Integration (local process)** | Multi-module wiring, CLI entry, in-process services | Named modules such as `test/test_unified_cli_integration.py` | Report import/install gaps clearly. Prefer explicit skip reasons over silent pass. |
| **Network / service** | IPFS, libp2p, remote MCP, live APIs | Integration paths that open sockets or daemons | If the service or credential is absent, **skip or fail with an explicit reason**. Never treat absence as a green “not applicable” without a message. |
| **Hardware** | CUDA, ROCm, MPS, OpenVINO, WebNN/WebGPU | Marker-selected or hardware trees; see `test/conftest.py` | Markers (`cuda`, `webgpu`, …) skip when detection reports unavailable. Record that skip in CI logs; do not rewrite the suite to hide hardware debt. |
| **Provider / external CLI** | LLM CLIs, provers, Leanstral, Codex, cloud backends | Supervisor and router suites that shell out or call APIs | Missing binary, auth, or model must surface as skip/fail with a clear reason. Green without the provider does **not** prove provider health. |

### Report, do not hide, missing capability

Contributors and agents must:

1. **Separate** offline contract results from capability-gated results in any
   report or PR note.
2. **Keep skip reasons visible** (`-rs` is useful). A skipped CUDA test is
   evidence of missing hardware, not proof that CUDA paths work.
3. **Never delete or broaden skips** solely to make a matrix look greener.
4. **Probe before claiming** optional stacks (examples below). An installed
   driver is not the same as a CUDA-capable PyTorch build; a package import
   is not the same as a healthy MCP or IPFS daemon.

```bash
# CUDA / torch (optional)
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", getattr(torch.version, "cuda", None))
PY

# Show skip/xfail reasons for a selected path
python -m pytest test/api/test_agent_supervisor_objective_graph.py -q -rs
```

`test/conftest.py` skips marked `cuda` / `webgpu` / `webnn` / `rocm` / `mps`
(and related) tests when hardware detection reports the platform unavailable.
That is intentional reporting, not a silent green for those backends.

## Fast deterministic checks

Start with focused tests that do not require external services:

```bash
# Unified CLI integration lives at test root (not under test/api/).
python -m pytest test/test_unified_cli_integration.py -q

python -m pytest test/api/test_agent_supervisor_objective_graph.py -q
python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py -q
```

For the supervisor control plane (still offline when fixtures mock providers):

```bash
python -m pytest \
  test/api/test_agent_supervisor_analysis_ast_index.py \
  test/api/test_agent_supervisor_analysis_cache.py \
  test/api/test_agent_supervisor_analysis_contracts.py \
  test/api/test_agent_supervisor_analysis_retrieval.py -q

python -m pytest \
  test/api/test_agent_supervisor_lease_coordination.py \
  test/api/test_agent_supervisor_resource_scheduler.py \
  test/api/test_agent_supervisor_scheduler_metrics.py -q

python -m pytest \
  test/api/test_agent_supervisor_leanstral_goal_benchmark.py \
  test/api/test_agent_supervisor_leanstral_goal_lifecycle_e2e.py -q
```

These exercises typed contracts and deterministic fixtures. They do **not**
prove that a local Leanstral, Codex, CUDA, IPFS, or external prover
installation is healthy.

## Full and selective runs

```bash
# Default discovery only (see testpaths above).
python -m pytest

# Named subsystem trees (always pass the path when outside testpaths).
python -m pytest test/api/ -q
python -m pytest test/integration/ -q
python -m pytest test/hardware_detection/ -q
python -m pytest test/test_unified_cli_integration.py -q

# Single node id.
python -m pytest test/api/test_agent_supervisor_objective_graph.py::test_name -q

# Stop at first failure; show local output and skip reasons.
python -m pytest -x -vv -s -rs
```

## Hardware and provider tests

Hardware tests belong in a different report column from deterministic
contracts. Before a CUDA run, record torch/CUDA availability (snippet above).
Similar care applies to OpenVINO, MPS, WebNN/WebGPU, IPFS, libp2p, MCP, and
external LLM/prover commands.

| Expectation | Do |
| --- | --- |
| Optional stack missing | Accept explicit skip/fail with reason; note it in the PR. |
| Optional stack claimed working | Run the matching marked suite and attach environment probes. |
| Force CUDA despite auto-skip | Only when intentionally testing CUDA paths: set `IPFS_ACCELERATE_PY_TEST_FORCE_CUDA=1` (see `test/conftest.py`). Failure then means the path broke, not that skip policy failed. |

## Agent supervisor smoke checks

Module paths after the layout cutover live under **domain packages**. Use the
importable module names (and console scripts from `pyproject.toml`), not the
old package-root basenames.

| Surface | Import / module path | Console script (when installed) |
| --- | --- | --- |
| Objective daemon | `ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon` | `ipfs-accelerate-agent-objective-daemon` |
| Bundle supervisor | `ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor` | `ipfs-accelerate-agent-bundle-supervisor` |
| Todo / implementation | `ipfs_accelerate_py.agent_supervisor.todo_daemon` / `.implementation_daemon` / `.implementation_supervisor` | matching `ipfs-accelerate-agent-*` scripts |

Command parsers are safe to inspect without starting workers:

```bash
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon --help
python -m ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon --help
python -m ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor --help
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor --help
```

For a real smoke, use temporary state and a fixture objective/todo board.
Run the bundle supervisor **without** `--start` first. Prefer `--once`
before enabling a long-running lane. Inspect status, heartbeat, event, and
manifest artifacts after every stage. Missing provider, lease backend, or
state path must appear in those artifacts or process exit status—not as an
empty success.

## Coverage and performance

Coverage helps regression analysis but is **not** a substitute for capability
validation:

```bash
python -m pytest --cov=ipfs_accelerate_py --cov-report=term-missing
```

Performance and hardware benchmarks live under `data/benchmarks/` and related
test directories. Benchmark numbers are workload- and hardware-specific; do
not copy a historical report into current documentation without recording
commit, model, provider, device, and test configuration.

## Writing tests

New tests should:

- use deterministic fixtures for schemas, identity, ordering, and bounded
  output;
- isolate optional provider or hardware dependencies behind explicit fixtures
  or markers so absence is **reported**;
- assert provenance and failure behavior, not only the happy path;
- keep large source/model/provider payloads out of durable status records;
  and
- add a focused test beside the module changed before running broad suites.

For supervisor changes, test the trust boundary: model proposals remain
non-authoritative until independent validators or authoritative receipts
accept them.

## Related documents

| Document | Role |
| --- | --- |
| [DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md) | Doc review checklist and honest PR-gate expectations |
| [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md) | Status vocabulary and revalidation procedure |
| [DOCUMENTATION_CURRENT_STATE.md](DOCUMENTATION_CURRENT_STATE.md) | Maintained surfaces and short audit checklist |
| [DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | Known path/drift findings (Historical inventory) |
