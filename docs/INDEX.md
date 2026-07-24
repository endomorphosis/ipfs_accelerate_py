# IPFS Accelerate Python Documentation Index

This index separates current user/developer guidance from historical project
records. Current documentation is maintained against the checked-out Python
package; documents under `archive/` and `development_history/` preserve the
context of earlier implementations and are not normative API references.

## Start here

- [Getting started](guides/getting-started/README.md): install the package and
  run a first inference or MCP server.
- [Quick start](guides/QUICKSTART.md): short command-line and Python examples.
- [Agent Supervisor Guide](guides/AGENT_SUPERVISOR_GUIDE.md): objective heaps,
  bundle lanes, implementation daemons, evidence, and Leanstral boundaries.
- [API reference](api/overview.md): current Python exports and supported entry
  points.
- [Architecture overview](architecture/overview.md): current runtime layers
  and data flow.

## User guides

- [Installation](guides/getting-started/installation.md)
- [Hardware support and tuning](guides/hardware/overview.md)
- [MCP setup](guides/MCP_SETUP_GUIDE.md)
- [P2P workflows](guides/p2p/README.md)
- [Deployment](guides/deployment/README.md)
- [Docker](guides/docker/README.md)
- [Troubleshooting](guides/troubleshooting/faq.md)
- [Examples](../examples/README.md)

## Developer and operator references

- [Testing](development/testing.md)
- [Documentation current state](development/DOCUMENTATION_CURRENT_STATE.md)
- [Contributing](../CONTRIBUTING.md)
- [LLM router](LLM_ROUTER.md)
- [IPFS backend router](IPFS_BACKEND_ROUTER.md)
- [Canonical MCP server README](../ipfs_accelerate_py/mcp_server/README.md)
- [MCP++ records](../mcpplusplus/README.md)

## Agent supervisor architecture

- [Architecture and assurance model](architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
- [Formal planning and prover matrix](architecture/AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md)
- [Formal verification plan](architecture/AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md)
- [Leanstral goal development and benchmark](architecture/AGENT_SUPERVISOR_LEANSTRAL_GOAL_DEVELOPMENT.md)
- [Objective graph implementation notes](agent_supervisor_objective_graph.md)
- [Completed supervisor task records](architecture/)

## Feature areas

- [IPFS integration](features/ipfs/IPFS.md)
- [WebNN/WebGPU](features/webnn-webgpu/WEBNN_WEBGPU_README.md)
- [Auto-healing](features/auto-healing/README.md)
- [HuggingFace model server](features/hf-model-server/README.md)
- [GitHub cache integration](features/github-cache/overview.md)

## Project records and archives

- [Project documentation hub](project/README.md)
- [Status records](project/status/)
- [Dashboard records](project/dashboard/)
- [Migration records](project/migration/MIGRATION_GUIDE.md)
- [Historical session summaries](archive/sessions/)
- [Documentation audit history](development_history/README.md)

Historical reports may contain point-in-time scores, paths, test counts, or
planned work. Use the current guides and source code for present behavior.

## By task

| Need | Start with |
| --- | --- |
| Install or verify the package | [Installation](guides/getting-started/installation.md) |
| Run inference | [Quick start](guides/QUICKSTART.md) |
| Start MCP | [MCP setup](guides/MCP_SETUP_GUIDE.md) |
| Operate objective-driven agent lanes | [Agent Supervisor Guide](guides/AGENT_SUPERVISOR_GUIDE.md) |
| Understand assurance and provers | [Formal verification plan](architecture/AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md) |
| Run tests | [Testing](development/testing.md) |
| Audit documentation drift | [Documentation current state](development/DOCUMENTATION_CURRENT_STATE.md) |
| Troubleshoot | [FAQ](guides/troubleshooting/faq.md) |

**Documentation baseline:** 2026-07-24. Update this page when a maintained
entry point or canonical architecture document changes.
