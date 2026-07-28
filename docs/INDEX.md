# IPFS Accelerate Python Documentation Index

This index separates current user/developer guidance from historical project
records. Current documentation is maintained against the checked-out Python
package; documents under `archive/` and `development_history/` preserve the
context of earlier implementations and are not normative API references.

## Start here

- [Getting started](guides/getting-started/README.md): install the package and
  run a first inference or MCP server.
- [Quick start](guides/QUICKSTART.md): short command-line and Python examples.
- [Agent Supervisor Guide](guides/AGENT_SUPERVISOR_GUIDE.md): operate the
  stable Python, CLI, and MCP interfaces; choose a resource and rollout
  profile; migrate standalone scripts; and recover failed work.
- [API reference](api/overview.md): current Python exports and supported entry
  points.
- [Architecture overview](architecture/overview.md): current runtime layers
  and data flow.
- [AI Service Catalog architecture](architecture/AI_SERVICE_CATALOG.md):
  canonical service identities, resolution, source precedence, security,
  migration, and rollout.
- [Endpoint usage-aware routing plan](architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md):
  planned endpoint/account limit accounting, atomic reservations,
  ModelManager planning, router fallback, and supervisor capacity governance.
- [Codebase-aware plan creation and steering plan](architecture/AGENT_SUPERVISOR_PLAN_CREATE_AND_STEER_PLAN.md):
  planned create/steer tools, registry-backed code and logic queries,
  revision-safe taskboard deltas, and parallel schedulability contracts.

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
- [AI Service Catalog](architecture/AI_SERVICE_CATALOG.md)
- [Endpoint usage-aware routing plan](architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md)
- [LLM router](LLM_ROUTER.md)
- [MCP server AI catalog and router tools](MCP_SERVER.md)
- [IPFS backend router](IPFS_BACKEND_ROUTER.md)
- [Canonical MCP server README](../ipfs_accelerate_py/mcp_server/README.md)
- [MCP++ records](../mcpplusplus/README.md)

## Agent supervisor

Use the operator guide for supported entry points and day-to-day workflows,
the architecture document for control-plane contracts and trust boundaries,
and the self-improvement plan for the rollout roadmap and acceptance evidence.

- [Operator guide, profiles, and migration](guides/AGENT_SUPERVISOR_GUIDE.md)
- [Architecture, contracts, and assurance model](architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
- [Endpoint usage-aware routing and supervisor capacity plan](architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md)
- [Codebase-aware plan creation and steering](architecture/AGENT_SUPERVISOR_PLAN_CREATE_AND_STEER_PLAN.md)
- [Self-improvement rollout plan](architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md)
- [Formal planning and prover matrix](architecture/AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md)
- [Formal verification plan](architecture/AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md)
- [Leanstral goal development and benchmark](architecture/AGENT_SUPERVISOR_LEANSTRAL_GOAL_DEVELOPMENT.md)
- [Supervisor self-improvement objective heap](architecture/agent_supervisor_self_improvement.objectives.md)
- [Supervisor self-improvement task board](architecture/agent_supervisor_self_improvement.todo.md)
- [Objective graph implementation notes](agent_supervisor_objective_graph.md)
- [Architecture documentation](architecture/overview.md)

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
| Discover or resolve AI services | [AI Service Catalog](architecture/AI_SERVICE_CATALOG.md) |
| Plan endpoint usage limits and intelligent fallback | [Endpoint usage-aware routing](architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md) |
| Use catalog or router tools over MCP | [MCP server AI tools](MCP_SERVER.md) |
| Migrate legacy model or MCP APIs | [Catalog migration and compatibility](architecture/AI_SERVICE_CATALOG.md#migration-and-compatibility) |
| Operate or migrate agent-supervisor workflows | [Agent Supervisor Guide](guides/AGENT_SUPERVISOR_GUIDE.md) |
| Understand supervisor services, contracts, and trust boundaries | [Agent Supervisor Architecture](architecture/AGENT_SUPERVISOR_ARCHITECTURE.md) |
| Create or steer a codebase-aware supervisor plan | [Plan creation and steering](architecture/AGENT_SUPERVISOR_PLAN_CREATE_AND_STEER_PLAN.md) |
| Follow the supervisor rollout and self-improvement roadmap | [Self-Improvement Plan](architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md) |
| Understand assurance and provers | [Formal verification plan](architecture/AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md) |
| Run tests | [Testing](development/testing.md) |
| Audit documentation drift | [Documentation current state](development/DOCUMENTATION_CURRENT_STATE.md) |
| Troubleshoot | [FAQ](guides/troubleshooting/faq.md) |

**Documentation baseline:** 2026-07-28. Update this page when a maintained
entry point or canonical architecture document changes.
