# Contributing to IPFS Accelerate Python

Thank you for contributing. This repository contains a Python inference
library, optional MCP/IPFS/P2P integrations, a HuggingFace model server, and an
objective-driven agent supervisor. Keep changes scoped to the subsystem they
actually affect and make optional capability boundaries explicit.

## Before changing code

1. Read the [current documentation audit](docs/development/DOCUMENTATION_CURRENT_STATE.md).
2. Check the relevant module, tests, and current CLI help before copying an
   example from a historical document.
3. Open an issue or design note for a cross-cutting change to the public API,
   persistence schema, scheduler contracts, or trust boundary.

## Development setup

```bash
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install feature extras only when the work needs them. The extras are defined
in `pyproject.toml`; common profiles are `full`, `mcp`, `mcp-p2p`, `libp2p`,
`webnn`, `llama_cpp`, `analysis`, `monitoring`, `testing`, and `all`.

Verify that the shell is using the intended checkout:

```bash
python - <<'PY'
import ipfs_accelerate_py
from ipfs_accelerate_py import get_instance

print("version:", ipfs_accelerate_py.__version__)
print(get_instance().get_capabilities(detail=True))
PY
```

Optional dependencies may be absent. A base-package import is not evidence
that CUDA, Transformers, MCP, IPFS, P2P, browser, or prover integrations are
available.

## Repository boundaries

- `ipfs_accelerate_py/`: library, providers, routers, MCP runtimes, and model
  server code.
- `ipfs_accelerate_py/mcp_server/`: canonical MCP runtime.
- `ipfs_accelerate_py/mcp/`: compatibility facade and older adapters.
- `ipfs_accelerate_py/agent_supervisor/`: objective analysis, scheduling,
  implementation, evidence, and formal-assurance infrastructure.
- `test/`: deterministic, API, integration, optional-provider, and hardware
  tests.
- `docs/`: maintained guides plus clearly labeled historical records.

The package exposes `ipfs_accelerate_py` and `get_instance()` as the primary
Python API. Do not introduce new examples using `IPFSAccelerator`; that class
is not a current package export.

## Testing

Run the smallest relevant test first:

```bash
python -m pytest test/test_unified_cli_integration.py -q
python -m pytest test/api/test_agent_supervisor_objective_graph.py -q
python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py -q
```

Useful subsystem checks include:

```bash
python -m pytest test/test_model_manager.py -q
python -m pytest test/test_hf_model_server_endpoint_contract.py -q
python -m pytest test/test_error_handler.py -q
python -m pytest test/test_docker_executor.py -q
```

Run the full suite only after the required optional dependencies and external
services are installed:

```bash
python -m pytest
python -m pytest --cov=ipfs_accelerate_py --cov-report=html
```

Hardware, network, model-download, Docker, P2P, and provider tests must be
identified as such. Do not make a live service or a GPU mandatory for a
deterministic contract test.

## Code quality

The development extra includes the repository's primary local tools:

```bash
python -m black ipfs_accelerate_py test
python -m flake8 ipfs_accelerate_py
python -m mypy ipfs_accelerate_py
```

Use the repository's existing style in the touched module. Add type hints at
public boundaries, keep imports lazy for optional capabilities, and document
fallback behavior rather than hiding import failures.

## Supervisor extension points

These are maintainer/developer boundaries for extending the agent supervisor;
they are not a required next step for ordinary users:

- Add an evidence-producing scanner or validator behind a versioned receipt.
- Add a prover through the capability registry and conformance fixtures,
  rather than calling a CLI directly from scheduling code.
- Add a task source through `objective_graph`/`backlog_refinery` so identity,
  deduplication, dependencies, and bundle metadata are preserved.
- Add an LLM through the existing router/provider boundary and keep its output
  in the proposal tier until deterministic checks accept it.
- Add scheduler policy through typed resource/lease contracts and metrics,
  not by mutating daemon state ad hoc.
- Add persistence through an artifact or projection store with a schema
  version, canonical identity, and migration behavior.

The [Agent Supervisor Guide](docs/guides/AGENT_SUPERVISOR_GUIDE.md) explains
the operational sequence and trust model.

## Documentation changes

Update documentation when a public import, CLI option, environment variable,
configuration field, persistence schema, test command, or operational behavior
changes. Prefer links to maintained pages:

- [Documentation index](docs/INDEX.md)
- [API overview](docs/api/overview.md)
- [Architecture overview](docs/architecture/overview.md)
- [Testing](docs/development/testing.md)
- [Current-state audit](docs/development/DOCUMENTATION_CURRENT_STATE.md)

Keep implementation reports, phase summaries, benchmark snapshots, and
session notes labeled as historical. Do not use a fixed benchmark, model
count, or test count as a current product guarantee without reproducible
hardware and commit context.

## Pull requests

Use a focused branch and commit:

```bash
git switch -c fix/short-description
git diff --check
git status --short
git commit -m "docs: describe current MCP boundary"
git push origin fix/short-description
```

Pull requests should include:

- the problem and affected boundary;
- tests run, including optional capability requirements;
- documentation updates for changed public behavior; and
- migration or rollback notes for schema, scheduler, or service changes.

Use conventional prefixes such as `feat:`, `fix:`, `docs:`, `test:`,
`refactor:`, and `chore:`. Keep generated logs, credentials, model weights,
and local state out of commits.

## Bug reports and security

For bugs, include the first traceback, command, Python executable, package
version, platform, hardware capability report, and a minimal reproduction.
Do not include tokens, private model data, or large raw artifacts.

For vulnerabilities, follow [SECURITY.md](SECURITY.md) rather than opening a
public issue with exploit details.
