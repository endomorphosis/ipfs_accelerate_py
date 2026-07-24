# IPFS Accelerate Python Documentation

The documentation is organized around the code that is maintained today. The
[documentation index](INDEX.md) is the canonical navigation page; this file is
the shorter orientation page for new readers.

## Choose a path

| You want to... | Read |
| --- | --- |
| Install the package | [Installation guide](guides/getting-started/installation.md) |
| Run inference or start MCP | [Getting started](guides/getting-started/README.md) and [Quick start](guides/QUICKSTART.md) |
| Operate the objective-driven agent supervisor | [Agent Supervisor Guide](guides/AGENT_SUPERVISOR_GUIDE.md) |
| Understand system structure | [Architecture overview](architecture/overview.md) |
| Use Python APIs | [API overview](api/overview.md) |
| Configure hardware | [Hardware guide](guides/hardware/overview.md) |
| Run tests | [Testing guide](development/testing.md) |
| Find a feature guide | [Documentation index](INDEX.md) |

## Current system boundaries

The repository contains several related but distinct runtimes:

- `ipfs_accelerate_py`: the Python library, model/inference integrations, and
  optional IPFS/P2P services;
- `ipfs_accelerate_py.mcp_server`: the canonical MCP runtime;
- `ipfs_accelerate_py.agent_supervisor`: objective analysis, task generation,
  scheduling, implementation, and assurance infrastructure;
- `examples/`, `test/`, and `scripts/`: executable examples, validation suites,
  and operational tooling.

The package supports optional dependencies. A successful import of the base
package does not imply that CUDA, Transformers, IPFS, MCP, P2P, browser, or
formal-prover integrations are installed or healthy. Use the relevant
capability/status command or test for those integrations.

## Agent supervisor documentation

The supervisor is a maintainer/operator subsystem, not a prerequisite for
ordinary inference. Start with the [operator guide](guides/AGENT_SUPERVISOR_GUIDE.md),
then use the design documents for the trust and assurance model:

- [Agent supervisor architecture](architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
- [Formal planning and prover matrix](architecture/AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md)
- [Formal verification plan](architecture/AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md)
- [Leanstral goal-development benchmark](architecture/AGENT_SUPERVISOR_LEANSTRAL_GOAL_DEVELOPMENT.md)

The four documents above are normative design/current-state references. Files
whose names include `summary`, `history`, `complete`, or `todo` are records of a
particular delivery or backlog tranche and should be read in that context.

## Project information

- [Root README](../README.md)
- [Changelog](../CHANGELOG.md)
- [Contributing](../CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
- [Project records](project/README.md)
- [MCP++ records](../mcpplusplus/README.md)

**Current documentation baseline:** 2026-07-24.
