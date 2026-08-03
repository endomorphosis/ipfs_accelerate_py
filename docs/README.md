# IPFS Accelerate Python Documentation

**Status:** Current
**Owner:** documentation-governance
**Audience:** New readers choosing a first path; agents and maintainers routing
to maintained surfaces
**Scope:** Short orientation and labelled entrypoints. The
[documentation index](INDEX.md) is the canonical navigation page.
**Non-goals:** Restating subsystem architecture; classifying every file under
`docs/`; resolving code-owned packaging contradictions in prose.
**Sources:** [DOCUMENTATION_MANIFEST.md](development/DOCUMENTATION_MANIFEST.md);
[DOCUMENTATION_LIFECYCLE.md](development/DOCUMENTATION_LIFECYCLE.md);
[DOCUMENTATION_CURRENT_STATE.md](development/DOCUMENTATION_CURRENT_STATE.md);
maintained guides linked below.
**Last verified:** 2026-08-03; navigation routes checked against the DOC-027
manifest and Current guide headers on this tree. See the
[validation closeout](development/DOCUMENTATION_VALIDATION_2026_08.md) for the
exact commit and offline check results.
**Interface:** DocumentationNavigation@2 (orientation half)

This file is the **shorter orientation page**. Prefer **Current** guides for
operational contracts. Prefer **Reference** pages for vocabulary and intent.
**Plan** and **Historical** material is labelled and is never a substitute for
landed behavior. Live code, package metadata, and executable help override
prose when they disagree.

## Choose a path

| You want to... | Read | Lifecycle |
| --- | --- | --- |
| Install the package | [Installation guide](guides/getting-started/installation.md) | Current |
| Run inference or start MCP | [Getting started](guides/getting-started/README.md) and [Quick start](guides/QUICKSTART.md) | Current |
| Use Python APIs | [API overview](api/overview.md) | Current |
| Use the product CLI | [CLI guide](guides/cli/README_CLI.md) | Current |
| Operate the agent supervisor | [Agent Supervisor Guide](guides/AGENT_SUPERVISOR_GUIDE.md) | Current |
| Orient in supervisor docs by audience | [Agent Supervisor doc hub](architecture/agent_supervisor/README.md) | Current |
| Understand product architecture | [Architecture hub](architecture/README.md) → [overview](architecture/overview.md) | Reference → Current |
| Learn shared product vocabulary | [Product glossary](architecture/GLOSSARY.md) | Reference |
| Configure hardware | [Hardware guide](guides/hardware/overview.md) | Current |
| Run tests | [Testing guide](development/testing.md) | Current |
| See what is maintained | [Documentation current state](development/DOCUMENTATION_CURRENT_STATE.md) | Current |
| Find any maintained entry by topic | [Documentation index](INDEX.md) | Current |
| Review classification inventory | [Documentation manifest](development/DOCUMENTATION_MANIFEST.md) | Current |
| Reproduce the closeout checks | [Validation closeout](development/DOCUMENTATION_VALIDATION_2026_08.md) | Reference |

## Current system boundaries

The repository contains several related but distinct runtimes:

- `ipfs_accelerate_py`: the Python library, model/inference integrations, and
  optional IPFS/P2P services;
- `ipfs_accelerate_py.mcp_server`: the **canonical** MCP runtime;
- `ipfs_accelerate_py.mcp`: the MCP **compatibility facade** (not the primary
  operator entry);
- `ipfs_accelerate_py.agent_supervisor`: objective analysis, task generation,
  scheduling, implementation, and assurance infrastructure;
- `examples/`, `test/`, and `scripts/`: executable examples, validation suites,
  and operational tooling.

The package supports optional dependencies. A successful import of the base
package does not imply that CUDA, Transformers, IPFS, MCP, P2P, browser, or
formal-prover integrations are installed or healthy. Use the relevant
capability/status command or test for those integrations.

Two installed CLI entry points are **not interchangeable**:

| Command | Module | Role |
| --- | --- | --- |
| `ipfs-accelerate` | `cli_entry.py` / `cli.py` | Unified product CLI |
| `ipfs_accelerate` | `ai_inference_cli.py` | Separate AI-inference CLI |

Use each command's own `--help` output. Do not merge their flag sets in prose.

## Agent supervisor documentation

The supervisor is a maintainer/operator subsystem, not a prerequisite for
ordinary inference. Start with the [doc hub](architecture/agent_supervisor/README.md)
(**Current**) for audience routing, then the
[operator guide](guides/AGENT_SUPERVISOR_GUIDE.md) (**Current**) for day-to-day
use. Design vocabulary lives in the
[philosophy](architecture/AGENT_SUPERVISOR_PHILOSOPHY.md) page (**Reference**).

Maintained Current architecture depth (control, planning, execution, prompt):

- [Control plane](architecture/agent_supervisor/CONTROL_PLANE.md)
- [Planning and assurance](architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md)
- [Execution and recovery](architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md)
- [Prompt-first runtime](architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md)
- [Developer guide](architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Agent capsule](architecture/agent_supervisor/FOR_AGENTS.md)

Reference maps (vocabulary and deep orientation, not sole runtime authority):

- [Design philosophy](architecture/AGENT_SUPERVISOR_PHILOSOPHY.md)
- [Architecture map](architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
- [Package map](architecture/agent_supervisor/PACKAGE_MAP.md)
- [Program glossary](architecture/agent_supervisor/PROGRAMS.md)
- [Contributor guide](architecture/agent_supervisor/FOR_CONTRIBUTORS.md)

Files whose names include `summary`, `history`, `complete`, `todo`,
`objectives`, or `*_PLAN*` are **Plan** or **Historical** records of a
particular delivery or backlog tranche. They are not Current API contracts.

## Documentation governance

| Document | Role | Lifecycle |
| --- | --- | --- |
| [INDEX.md](INDEX.md) | Canonical topic and audience index | Current |
| [DOCUMENTATION_CURRENT_STATE.md](development/DOCUMENTATION_CURRENT_STATE.md) | Maintained-surface matrix and audit checklist | Current |
| [DOCUMENTATION_MANIFEST.md](development/DOCUMENTATION_MANIFEST.md) | Status inventory with owners and baselines | Current |
| [DOCUMENTATION_LIFECYCLE.md](development/DOCUMENTATION_LIFECYCLE.md) | Status vocabulary and authority order | Current |
| [DOCUMENTATION_MAINTENANCE.md](development/DOCUMENTATION_MAINTENANCE.md) | PR review checklist without suppressing drift | Current |
| [DOCUMENTATION_VALIDATION_2026_08.md](development/DOCUMENTATION_VALIDATION_2026_08.md) | Offline validation closeout receipt | Reference |
| [GUIDE_CONVENTIONS.md](architecture/GUIDE_CONVENTIONS.md) | Architecture guide writing contract | Current |

## Project information

- [Root README](../README.md)
- [Changelog](../CHANGELOG.md)
- [Contributing](../CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
- [Project records](project/README.md) (**Historical** / project hub)
- [MCP++ records](../mcpplusplus/README.md) (**Plan** / evidence — not
  `mcp_server` runtime API)

**Documentation baseline:** 2026-08-03 (documentation-refresh closeout). Update
this page when a maintained entry point changes. Exact commit and check results:
[DOCUMENTATION_VALIDATION_2026_08.md](development/DOCUMENTATION_VALIDATION_2026_08.md).
