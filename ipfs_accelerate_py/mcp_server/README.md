# IPFS Accelerate MCP++ Server

<!-- markdownlint-disable MD013 -->

The canonical MCP server runtime for this repository lives in `ipfs_accelerate_py.mcp_server`.

## Current status

- **Canonical runtime:** `ipfs_accelerate_py/mcp_server`
- **Compatibility facade:** `ipfs_accelerate_py/mcp`
- **Default startup path:** unified runtime is now the default behind `create_mcp_server()`
- **Cutover state:** approved for canonical-default operation
- **Release-candidate validation:** focused matrix passed with `120 passed`

This package is the stable import target for the MCP server unification work. Public entrypoints remain source-compatible where practical, while the canonical runtime owns the MCP++ feature surface, transport hardening, and conformance tracking.

## What the server provides

The MCP++ server combines the general IPFS Accelerate tool surface with a unified MCP control plane:

- Unified discovery and dispatch meta-tools
- Native migrated categories for `ipfs`, `workflow`, and `p2p`
- MCP++ profile negotiation metadata
- CID-native execution artifacts
- UCAN capability validation
- Temporal/deontic policy evaluation with auditable decisions
- Event DAG provenance tracking
- Risk scoring and frontier execution hooks
- Monitoring, tracing, Prometheus, and audit-to-metrics bridging
- Compatibility-tested FastAPI, process-level, and MCP+p2p transport paths

## Recommended entrypoints

| Entry point | Use when | Notes |
| --- | --- | --- |
| `ipfs-accelerate mcp start` | You want the main user-facing CLI | Best general-purpose command for operators already using the project CLI |
| `python -m ipfs_accelerate_py.mcp.cli --host 0.0.0.0 --port 9000` | You want direct control over MCP + TaskQueue/p2p worker options | Supports built-in task worker, autoscaling, libp2p service, and queue-path controls |
| `python -m ipfs_accelerate_py.mcp_server.fastapi_service` | You want a standalone FastAPI-hosted MCP endpoint | Uses `IPFS_MCP_*` environment variables and mounts MCP at `/mcp` by default |
| `from ipfs_accelerate_py.mcp_server import create_server` | You are embedding the server from Python | Stable package import target for programmatic startup |
| `python -m ipfs_accelerate_py.mcp_server` | You need the package module entrypoint | Preserved as a compatibility facade while migration entrypoints stay stable |

## Quick start

### Programmatic startup

```python
from ipfs_accelerate_py.mcp_server import create_server

server = create_server(name="ipfs-accelerate")
server.run(host="0.0.0.0", port=9000)
```

### Standalone FastAPI service

```bash
export IPFS_MCP_HOST=0.0.0.0
export IPFS_MCP_PORT=9000
export IPFS_MCP_MOUNT_PATH=/mcp
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

### Direct MCP CLI with libp2p task service

```bash
python -m ipfs_accelerate_py.mcp.cli \
  --host 0.0.0.0 \
  --port 9000 \
  --mcp-p2p-port 9100 \
  --p2p-task-worker \
  --p2p-service \
  --p2p-queue ~/.cache/ipfs_datasets_py/task_queue.duckdb
```

## Unified meta-tools

The canonical control-plane tool names exposed by the unified runtime are:

- `tools_list_categories`
- `tools_list_tools`
- `tools_get_schema`
- `tools_dispatch`
- `tools_runtime_metrics`

These tools provide category discovery, schema inspection, canonical dispatch, and runtime telemetry surfaces for the migrated tool registry.

## Supported MCP++ profiles

The unified bootstrap currently advertises the following supported profiles:

- `mcp++/profile-a-idl`
- `mcp++/profile-b-cid-artifacts`
- `mcp++/profile-c-ucan`
- `mcp++/profile-d-temporal-policy`
- `mcp++/profile-e-mcp-p2p`

## Feature areas

### 1. Unified registry and dispatch

The server centralizes registration through the canonical runtime package and exposes a single dispatch path for discovery, schema lookup, and execution. The migrated Wave A categories are:

- `ipfs`
- `workflow`
- `p2p`

### 2. Security, authorization, and policy

Feature flags exist for enabling:

- UCAN delegation validation
- temporal/deontic policy evaluation
- policy audit logging
- secrets-vault integration
- risk scoring and risk-frontier execution

These features are exercised by deterministic coverage in the MCP++ chapter suites and the unified bootstrap regression corpus.

### 3. Artifacts and provenance

When enabled, the server can emit and persist deterministic artifact chains including:

- `intent_cid`
- `decision_cid`
- `receipt_cid`
- `event_cid`

Artifact persistence supports in-memory mode and JSON-backed durability for replay and restart validation.

### 4. Observability

The canonical runtime supports optional:

- runtime metrics collection
- audit-to-metrics bridging
- OpenTelemetry tracing
- Prometheus exporter wiring
- transport/runtime monitoring counters surfaced via `tools_runtime_metrics`

### 5. Transport and process integration

Validated surfaces include:

- process-level helper startup
- FastAPI sub-application mounting
- standalone uvicorn hosting
- MCP+p2p initialization and tool-call parity
- mixed-version negotiation hardening and malformed-negotiation sanitization

## Key environment variables

### Cutover and compatibility controls

| Variable | Purpose |
| --- | --- |
| `IPFS_MCP_FORCE_LEGACY_ROLLBACK` | Force the compatibility facade to remain on the legacy wrapper |
| `IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN` | Validate unified startup while intentionally falling back to legacy behavior |
| `IPFS_MCP_ENABLE_UNIFIED_BRIDGE` | Explicitly request the unified bridge when entering through compatibility-facade paths |

### Unified feature flags

| Variable | Purpose |
| --- | --- |
| `IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP` | Attach unified bootstrap metadata and services |
| `IPFS_MCP_SERVER_ENABLE_CID_ARTIFACTS` | Enable CID-native artifact emission |
| `IPFS_MCP_SERVER_ARTIFACT_STORE_BACKEND` | Select artifact backend: `memory` or `json` |
| `IPFS_MCP_SERVER_ARTIFACT_STORE_PATH` | JSON artifact store location when durable storage is enabled |
| `IPFS_MCP_SERVER_ENABLE_UCAN_VALIDATION` | Enable UCAN validation during dispatch |
| `IPFS_MCP_SERVER_ENABLE_POLICY_EVALUATION` | Enable temporal/deontic policy evaluation |
| `IPFS_MCP_SERVER_ENABLE_POLICY_AUDIT` | Enable policy audit logging |
| `IPFS_MCP_SERVER_ENABLE_MONITORING` | Enable runtime monitoring collectors |
| `IPFS_MCP_SERVER_ENABLE_OTEL_TRACING` | Enable OpenTelemetry tracing hooks |
| `IPFS_MCP_SERVER_ENABLE_PROMETHEUS_EXPORTER` | Enable Prometheus exporter integration |
| `IPFS_MCP_SERVER_ENABLE_PROMETHEUS_HTTP_SERVER` | Enable Prometheus HTTP server support |
| `IPFS_MCP_SERVER_ENABLE_SECRETS_VAULT` | Enable secrets vault support |
| `IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_AUTOLOAD` | Autoload secrets into the runtime environment |
| `IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_OVERWRITE` | Permit secrets autoload to overwrite existing env values |
| `IPFS_MCP_SERVER_ENABLE_RISK_SCORING` | Enable risk scoring during dispatch |
| `IPFS_MCP_SERVER_ENABLE_RISK_FRONTIER_EXECUTION` | Enable frontier execution binding |
| `IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES` | Preload migrated categories (`ipfs`, `workflow`, `p2p`, or `all`) |

### FastAPI service configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `IPFS_MCP_HOST` | FastAPI bind host | `0.0.0.0` |
| `IPFS_MCP_PORT` | FastAPI bind port | `8000` |
| `IPFS_MCP_MOUNT_PATH` | MCP mount path | `/mcp` |
| `IPFS_MCP_NAME` | Service name | `ipfs-accelerate-mcp` |
| `IPFS_MCP_DESCRIPTION` | Service description | `IPFS Accelerate MCP Server` |
| `IPFS_MCP_VERBOSE` | Verbose server logging | `false` |

Legacy fallback keys `HOST`, `PORT`, `MOUNT_PATH`, `APP_NAME`, `APP_DESCRIPTION`, and `DEBUG` remain accepted for compatibility.

## Remote libp2p task pickup

For remote machines that should host the MCP server and also service `ipfs_datasets_py` task submissions, use the direct MCP CLI with the task worker and libp2p service enabled:

```bash
python -m ipfs_accelerate_py.mcp.cli \
  --host 0.0.0.0 --port 9000 \
  --p2p-task-worker --p2p-service --p2p-listen-port 9710 \
  --p2p-queue ~/.cache/ipfs_datasets_py/task_queue.duckdb
```

Optional public announcement override:

```bash
export IPFS_DATASETS_PY_TASK_P2P_PUBLIC_IP="YOUR_PUBLIC_IP"
```

## Validation and evidence

The current server state is backed by deterministic repo-local evidence.

Primary references:

- `../../mcpplusplus/CUTOVER_CHECKLIST.md`
- `../../mcpplusplus/SERVER_UNIFICATION_PLAN.md`
- `../../mcpplusplus/CONFORMANCE_CHECKLIST.md`
- `../../mcpplusplus/SPEC_GAP_MATRIX.md`

Highlights:

- Canonical-default startup is approved and documented
- Rollback and dry-run behavior are covered by focused cutover tests
- Process/FastAPI/MCP+p2p entrypoints remain stable
- Representative MCP++ chapter suites are included in the focused RC matrix

## Migration notes

- Prefer imports from `ipfs_accelerate_py.mcp_server` for new code.
- Treat `ipfs_accelerate_py.mcp` as the compatibility facade during the remaining deprecation phases.
- Keep rollback and dry-run controls available for operational safety and validation.
- Update the conformance checklist and spec gap matrix whenever MCP++ capability status changes.
