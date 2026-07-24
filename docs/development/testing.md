# Testing

The repository uses pytest over a large, capability-oriented test tree. This
page documents the current commands and the boundaries that make tests
repeatable. It does not claim that optional hardware, network services, or
third-party providers are available on every machine.

## Install test dependencies

```bash
python -m pip install -e ".[dev]"
```

For the broadest local coverage, install the `testing`, `mcp`, `mcp-p2p`,
`webnn`, and `full` extras that match the features under test.

## Test layout

The most important maintained areas are:

| Path | Coverage |
| --- | --- |
| `test/api/` | Python API, contract, integration, and subsystem tests. |
| `test/integration/` | Cross-component and service integration tests. |
| `test/hardware/` and `test/hardware_detection/` | Hardware discovery and backend behavior. |
| `test/browser/`, `test/fixed_web_tests/`, and WebNN/WebGPU fixtures | Browser and web accelerator behavior. |
| `test/distributed_testing/` | Distributed test and CI service integrations. |
| `test/ipfs_accelerate_py/` | Package-specific regression tests. |
| `test/api/test_agent_supervisor_*.py` | Objective graph, evidence, scheduling, daemon, prover, and Leanstral contracts. |
| `examples/` | Executable examples and smoke-oriented demonstrations. |

The repository does not have one authoritative `test/unit/` tree. Older
documents that use that layout are historical examples.

## Fast deterministic checks

Start with focused tests that do not require external services:

```bash
python -m pytest test/api/test_unified_cli_integration.py -q
python -m pytest test/api/test_agent_supervisor_objective_graph.py -q
python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py -q
```

For the supervisor control plane:

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

These tests exercise typed contracts and deterministic fixtures. They do not
prove that a local Leanstral, Codex, CUDA, or external prover installation is
healthy.

## Full and selective runs

```bash
# Entire repository (may require optional dependencies and services).
python -m pytest

# One subsystem.
python -m pytest test/api/ -q
python -m pytest test/integration/ -q
python -m pytest test/hardware_detection/ -q

# Run a selected test by node id.
python -m pytest test/api/test_agent_supervisor_objective_graph.py::test_name -q

# Stop at the first failure and show local output.
python -m pytest -x -vv -s
```

Use `-m` only for markers declared by the repository's pytest configuration.
Do not assume that `pytest -m cuda` or `pytest -m browser` selects every test
in that area; inspect a test module when in doubt.

## Hardware and provider tests

Hardware tests should be separated from deterministic contract tests. Before a
CUDA run, record:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

An available driver is not the same as a CUDA-capable PyTorch build. Similar
care applies to OpenVINO, MPS, WebNN/WebGPU, IPFS, libp2p, MCP, and external
LLM/prover commands. Tests that need them should fail with a clear missing
capability or be explicitly skipped by their fixture policy.

## Agent supervisor smoke checks

The command parsers are safe to inspect without starting workers:

```bash
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon --help
python -m ipfs_accelerate_py.agent_supervisor.objective_daemon --help
python -m ipfs_accelerate_py.agent_supervisor.bundle_supervisor --help
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor --help
```

For a real smoke, use temporary state and a fixture objective/todo board. Run
the bundle supervisor without `--start` first. Add `--once` before enabling a
long-running lane. Inspect the emitted status, heartbeat, event, and manifest
artifacts after every stage.

## Coverage and performance

Coverage is useful for regression analysis but is not a substitute for
capability validation:

```bash
python -m pytest --cov=ipfs_accelerate_py --cov-report=term-missing
```

Performance and hardware benchmarks live in `data/benchmarks/` and related test
directories. Benchmark numbers are workload- and hardware-specific; do not
copy a historical report into current documentation without recording the
commit, model, provider, device, and test configuration.

## Writing tests

New tests should:

- use deterministic fixtures for schemas, identity, ordering, and bounded
  output;
- isolate optional provider or hardware dependencies behind explicit fixtures;
- assert provenance and failure behavior, not only the happy-path result;
- keep large source/model/provider payloads out of durable status records; and
- add a focused test beside the module changed before running broad suites.

For supervisor changes, test the trust boundary: model proposals must remain
non-authoritative until independent validators or authoritative receipts accept
them.
