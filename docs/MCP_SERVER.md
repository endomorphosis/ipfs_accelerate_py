# MCP Server Reference

**Status:** Current

**Audience:** Operators, integrators, and agents selecting or embedding the MCP
runtime

**Scope:** Canonical package identity, startup entry points, install extras,
configuration, transports, tool discovery, AI catalog/router tools, security
gates, compatibility migration, and auto-install caveats

**Non-goals:** Full MCP++ chapter checklists (`mcpplusplus/`); architecture
internals beyond operator-needed maps (see
[MCP runtime architecture](architecture/MCP_RUNTIME.md)); agent-supervisor
control-plane design

**Last verified:** `49c76b69f` (2026-08-03); entrypoints, extras, env keys, and
side-effect notes checked against `ipfs_accelerate_py/mcp_server/`,
`ipfs_accelerate_py/mcp/`, `pyproject.toml`, and
`docs/architecture/MCP_RUNTIME.md`

## Package identity

| Package | Role | Prefer for |
| --- | --- | --- |
| `ipfs_accelerate_py.mcp_server` | **Canonical** registry, server construct, FastAPI host, MCP++ primitives | New integrations, embedding, production HTTP hosting |
| `ipfs_accelerate_py.mcp` | **Compatibility facade** for historical imports and the task-worker CLI | Legacy callers mid-migration only |
| `ipfs_accelerate_py.mcplusplus_module` | Alternate Trio-first MCP++ surface | Existing Trio hosts; keep business logic in `mcp_server` |
| `mcpplusplus/` | Spec/conformance records | Evidence — not an importable runtime |

**Selection rule:** start and embed through `ipfs_accelerate_py.mcp_server`.
Do not treat `ipfs_accelerate_py.mcp` as the ownership target. Compatibility
paths default to the unified runtime when not forced onto the legacy wrapper,
but they still carry import-time auto-install side effects and facade
telemetry.

Deep runtime, dispatch, and trust-ladder detail lives in
[MCP runtime architecture](architecture/MCP_RUNTIME.md). Package module notes
are in the [canonical server README](../ipfs_accelerate_py/mcp_server/README.md).

## Install extras

MCP is optional. Base `ipfs-accelerate-py` does not install an MCP host.

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
```

| Extra | What it adds | What it does **not** prove |
| --- | --- | --- |
| `mcp` | FastMCP, Flask/Werkzeug, related HTTP helpers, PyGithub | A live peer, GPU, IPFS node, or auth boundary |
| `mcp-p2p` / `libp2p` | libp2p, protobuf, multihash, dnspython | Reachable mesh, queue durability, or tool authority |
| `all` | Broad application deps including FastMCP, **without** native P2P by default | Full production readiness |

Install extras explicitly in production images. Do not rely on import-time
auto-install (see [Auto-install and optional dependencies](#auto-install-and-optional-dependencies)).

## Canonical startup (preferred)

Use one of these paths first. They stay inside the canonical package and do
not require the historical `mcp` CLI.

### 1. Standalone FastAPI host (HTTP MCP)

```bash
export IPFS_MCP_HOST=127.0.0.1
export IPFS_MCP_PORT=8000
export IPFS_MCP_MOUNT_PATH=/mcp
# Optional: attach hierarchical meta-tools and MCP++ services
export IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP=1
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

Health endpoints on this host:

| Path | Purpose |
| --- | --- |
| `GET /healthz` | Process liveness |
| `GET /mcp/health` | MCP service + tool-count summary |

MCP protocol routes mount under `IPFS_MCP_MOUNT_PATH` (default `/mcp`). Keep
development servers on localhost unless authentication, TLS, firewall, and
resource limits are configured.

### 2. Programmatic construction

```python
from ipfs_accelerate_py.mcp_server import create_server

server = create_server(
    name="ipfs-accelerate",
    host="127.0.0.1",
    port=8000,
    mount_path="/mcp",
)
# Mount a functional transport (for example the FastAPI service) or call
# server.run(...) only when the returned object exposes a verified run path.
```

`create_server` builds the canonical wrapper and, when
`IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP=1`, attaches the hierarchical registry,
meta-tools, and MCP++ service factories. Bootstrap is **off by default**;
without it you get the base server without the unified meta-tool control plane.

### 3. Product CLI (operator surface)

```bash
python -m ipfs_accelerate_py.cli mcp start --host 127.0.0.1 --port 9000
python -m ipfs_accelerate_py.cli mcp status --host 127.0.0.1 --port 9000
```

When the `ipfs-accelerate` console script is installed, the same commands are
`ipfs-accelerate mcp start|status|dashboard`. Default bind port for this CLI is
**9000**. The start path prefers the Flask dashboard when present and falls
back to an integrated HTTP dashboard. It is the product operator entry, not the
lowest-level transport host.

Inspect flags from the live tree:

```bash
python -m ipfs_accelerate_py.cli mcp --help
python -m ipfs_accelerate_py.cli mcp start --help
```

### Entrypoint selection table

| Entry | Package | When to use | Caveat |
| --- | --- | --- | --- |
| `python -m ipfs_accelerate_py.mcp_server.fastapi_service` | canonical | Functional HTTP MCP host | Uses `IPFS_MCP_*` env; bootstrap flag separate |
| `from ipfs_accelerate_py.mcp_server import create_server` | canonical | Embed or test registry/lifecycle | Not a complete client-facing host by itself |
| `python -m ipfs_accelerate_py.mcp_server --fastapi` | canonical | Standalone wrapper that runs the FastAPI path | Without `--fastapi`, do not assume full MCP routes |
| `python -m ipfs_accelerate_py.cli mcp start` | product CLI | Dashboard-oriented operator start | Pulls dashboard/autoscaler options; not pure transport |
| `python -m ipfs_accelerate_py.mcp.cli …` | **compatibility** | TaskQueue/libp2p worker host options | Imports `mcp` (auto-install risk); bridges to unified by default |
| `from ipfs_accelerate_py.mcp.server import create_mcp_server` | **compatibility** | Legacy factory | Defaults to unified bridge; may fall back under rollback flags |

**Do not** point MCP clients at a bare lifecycle shell that only serves
`/healthz`. Prefer `fastapi_service` or an explicitly mounted transport over
`python -m ipfs_accelerate_py.mcp_server` without `--fastapi`. Canonical stdio
transport is not currently implemented under `mcp_server`.

## Configuration map

### FastAPI host (`UnifiedFastAPIConfig`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `IPFS_MCP_HOST` | `0.0.0.0` | Bind host |
| `IPFS_MCP_PORT` | `8000` | Bind port |
| `IPFS_MCP_MOUNT_PATH` | `/mcp` | MCP mount path |
| `IPFS_MCP_NAME` | `ipfs-accelerate-mcp` | Service name |
| `IPFS_MCP_DESCRIPTION` | `IPFS Accelerate MCP Server` | Service description |
| `IPFS_MCP_VERBOSE` | off | Verbose logging (`1`/`true`/`yes`/`on`) |

Legacy fallback keys (`HOST`, `PORT`, `MOUNT_PATH`, `APP_NAME`,
`APP_DESCRIPTION`, `DEBUG`) are accepted only when the canonical key is unset.
Prefer `IPFS_MCP_*`.

### Unified runtime / MCP++ gates (`UnifiedMCPServerConfig`)

All feature gates default **off** unless noted. Setting a flag advertises or
enables a code path; it does not install missing packages or prove a peer is
alive.

| Variable | Purpose |
| --- | --- |
| `IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP` | Attach hierarchical meta-tools and MCP++ service factories |
| `IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES` | Preload `ipfs`, `workflow`, `p2p`, or `all` |
| `IPFS_MCP_SERVER_ENABLE_CID_ARTIFACTS` | CID-native artifact emission |
| `IPFS_MCP_SERVER_ARTIFACT_STORE_BACKEND` | `memory` or `json` |
| `IPFS_MCP_SERVER_ARTIFACT_STORE_PATH` | JSON store path when durable |
| `IPFS_MCP_SERVER_ENABLE_UCAN_VALIDATION` | UCAN validation on guarded dispatch |
| `IPFS_MCP_SERVER_ENABLE_POLICY_EVALUATION` | Temporal/deontic policy evaluation |
| `IPFS_MCP_SERVER_ENABLE_POLICY_AUDIT` | Policy audit log |
| `IPFS_MCP_SERVER_ENABLE_MONITORING` | Runtime monitoring collectors |
| `IPFS_MCP_SERVER_ENABLE_OTEL_TRACING` | OpenTelemetry hooks |
| `IPFS_MCP_SERVER_ENABLE_PROMETHEUS_EXPORTER` | Prometheus exporter |
| `IPFS_MCP_SERVER_ENABLE_PROMETHEUS_HTTP_SERVER` | Prometheus HTTP server |
| `IPFS_MCP_SERVER_ENABLE_SECRETS_VAULT` | Secrets vault |
| `IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_AUTOLOAD` | Autoload secrets into the environment |
| `IPFS_MCP_SERVER_ENABLE_SECRETS_ENV_OVERWRITE` | Allow secrets autoload to overwrite env |
| `IPFS_MCP_SERVER_ENABLE_RISK_SCORING` | Risk scoring on guarded dispatch |
| `IPFS_MCP_SERVER_ENABLE_RISK_FRONTIER_EXECUTION` | Frontier execution binding |

### Compatibility facade runtime selection

| Variable | Effect |
| --- | --- |
| `IPFS_MCP_FORCE_LEGACY_ROLLBACK` | Force `create_mcp_server` onto the legacy wrapper |
| `IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN` | Probe unified `create_server`, then stay on legacy |
| `IPFS_MCP_ENABLE_UNIFIED_BRIDGE` | Explicit bridge telemetry; unified is already the default when not forced legacy |

These flags affect **facade factory selection**, not install policy. New code
should call `ipfs_accelerate_py.mcp_server.create_server` and avoid needing
them.

## Transports

| Transport | Primary modules | Operator note |
| --- | --- | --- |
| FastAPI / HTTP JSON-RPC | `mcp_server.fastapi_service`, `fastapi_config` | Preferred functional HTTP host |
| Process / standalone shell | `mcp_server.standalone_server` | Use with `--fastapi` for MCP routes; bare shell is not a full MCP client host |
| Product CLI dashboard | `cli.py` → dashboard / integrated HTTP | Operator UX; optional autoscaler and browser open |
| Compatibility CLI + TaskQueue | `mcp.cli` + `p2p_tasks` | Optional libp2p/queue worker host; compatibility import path |
| MCP+p2p | `mcp_server.mcp_p2p_transport` → `p2p_tasks.mcp_p2p` | Requires `mcp-p2p` extra and network identity |
| gRPC | `mcp_server.grpc_transport` | Compatibility facade when source module present |
| Canonical stdio | — | **Not implemented** under `mcp_server` today |

Transport success (HTTP 200, stream open) is delivery only. It is not
authorization. Guarded admission for category/tool dispatch lives on the
`tools_dispatch` meta-tool path when unified bootstrap is attached; ordinary
direct tool routes may not yet share every UCAN/policy gate. See the runtime
architecture guide for the direct-call gap.

## Tool discovery and meta-tools

When unified bootstrap is enabled, the control-plane meta-tools are:

| Meta-tool | Purpose |
| --- | --- |
| `tools_list_categories` | List registered categories |
| `tools_list_tools` | List tools in a category |
| `tools_get_schema` | Fetch a tool schema |
| `tools_dispatch` | Guarded category/tool dispatch |
| `tools_runtime_metrics` | Runtime telemetry surface |

Category loaders are lazy: listing or dispatching a category triggers
registration. Optional preload uses `IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES`.

Do not assume a tool exists because an older guide listed it. Query the runtime
manifest or meta-tools:

```python
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print(report.get("mcp", {}))
```

Wave A native categories include `ipfs`, `workflow`, and `p2p`. Many additional
native categories register under `ipfs_accelerate_py/mcp_server/tools/*` when
their loaders run and optional dependencies are present.

## AI catalog and router tools

Catalog tools query the `ModelManager` facade. They never invoke a provider.
Invocation tools resolve one immutable revision, then call the owning LLM,
embeddings, multimodal, or voice router. MCP++ `ai.catalog.v1` publishes the
same eleven canonical operations with bounded schemas and separate read,
refresh, and invoke authorities.

### Catalog query tools

| Tool | Purpose |
| --- | --- |
| `model_catalog_list_services` | Page provider/service descriptors |
| `model_catalog_list_models` | Page canonical model descriptors |
| `model_catalog_get` | Get one provider, model, deployment, or binding |
| `model_catalog_resolve` | Rank bindings for typed constraints |
| `model_catalog_health` | Read already-published source and record health |
| `model_catalog_refresh` | Explicitly refresh named sources with authority |

### Router invocation tools

| Tool | Owner | Operation |
| --- | --- | --- |
| `llm_generate` | `llm_router` | `text.generate` |
| `embeddings_generate` | `embeddings_router` | `embedding.generate` |
| `multimodal_generate` | `multimodal_router` | `vision.generate` |
| `voice_transcribe` | `voice_router` | `audio.transcribe` |
| `voice_synthesize` | `voice_router` | `audio.synthesize` |

Compatibility names `generate_text`, `generate_embeddings`, and
`generate_embedding` remain registered for existing MCP callers. They project
to the canonical text/embedding implementation. Historical `model_*` tools
also remain available while callers migrate to `model_catalog_*`.

### Revisions, pagination, and receipts

Successful catalog and invocation envelopes include `schema_version` and
`catalog_revision`. Invocation output also includes the selected binding and a
bounded receipt. A client that needs reproducibility should send the revision
it resolved; a changed revision fails with a typed mismatch instead of routing
against different content.

List pagination uses revision-bound cursors with at most 1,000 items per page.
If content changes between pages, restart from the first page at the new
revision. Do not combine pages from different revisions.

The selected binding exposes stable IDs, not credentials. Deployment endpoint
URIs are redacted from MCP responses. Receipts include selection facts and
revision but exclude prompts, media, model output, headers, credentials, and
raw endpoints.

### Bounded schemas and errors

Input schemas reject unknown properties and bound free text, arrays, pages,
timeouts, output sizes, embedding dimensions, stream chunks, and media. Media
must use one of the explicit inline, URI, or artifact-reference variants.
Remote media is denied unless the operation and authority explicitly allow it.

Common error codes include:

- `invalid_filter` or `invalid_request` for malformed bounded inputs;
- `no_match` for an empty constraint intersection;
- `ambiguous_identifier` for an alias that maps to multiple canonical IDs;
- `catalog_revision_mismatch` or `cursor_revision_mismatch`;
- `refresh_denied`, `invalid_sources`, or `source_refresh_failed`;
- typed invocation, timeout, media, and output-bound failures.

Errors retain the safe catalog revision when available and do not echo
credential values, private endpoint details, prompts, or media bodies.

### Authority and refresh

MCP++ separates authority by effect:

| Authority | Permitted operations |
| --- | --- |
| `ai.catalog/read` | list, get, resolve, health |
| `ai.catalog/refresh` | named explicit refresh |
| `ai.catalog/invoke` | five modality invocation tools |

Read authority cannot refresh or invoke. Refresh requires a non-empty source
list and an explicit true authority field at the local MCP boundary.
Side-effecting sources are additionally checked by catalog refresh policy.
Invocation authority does not grant catalog refresh.

Remote advertisements and catalog pages are untrusted. Signatures, issuers,
expiry, replay state, catalog revision, page bounds, URL/media policy, and
capabilities are validated before records enter a peer source. A remote record
cannot override a trusted local identity.

### Migration (AI tools)

Migrate discovery first, invocation second:

1. replace historical model enumeration with
   `model_catalog_list_services`/`model_catalog_list_models`;
2. persist stable IDs and `catalog_revision`, not list positions;
3. replace registry heuristics with `model_catalog_resolve`;
4. change `generate_text` to `llm_generate`;
5. change `generate_embeddings` or `generate_embedding` to
   `embeddings_generate`;
6. adopt the canonical multimodal and voice tool names;
7. validate output against the MCP++ `ai.catalog.v1` schema when interoperating
   with MCP++ peers.

Compatibility aliases have no scheduled removal date. See the
[catalog sunset policy](architecture/AI_SERVICE_CATALOG.md#compatibility-sunset-policy)
for the gates required before a future removal.

## Security and policy prerequisites

MCP tools may expose inference, storage, GitHub, Docker, P2P, or operational
actions depending on installed capabilities and policy. A registered tool is
not automatically authorized for an untrusted caller.

Operator checklist:

1. Bind development servers to `127.0.0.1` until auth and network policy exist.
2. Keep secrets out of client configs, prompts, and checked-in MCP client files.
3. Enable UCAN/policy/risk gates deliberately for high-assurance dispatch; use
   `tools_dispatch` when those gates must apply.
4. Treat peer advertisements and remote catalog pages as untrusted input.
5. Do not interpret transport success as authorization.
6. Production images: pin extras, disable auto-install, and verify capabilities
   after deploy.

## Auto-install and optional dependencies

| Path | Side effect | Production guidance |
| --- | --- | --- |
| Bare `import ipfs_accelerate_py.mcp_server` | Lazy package facade only | Preferred package boundary for discovery |
| Resolving `mcp_server.create_server` / importing `mcp_server.server` | Configures logging; eagerly imports native registrars; inference registrar currently reaches compatibility `mcp` | May trigger auto-install policy; not a pure discovery import |
| `import ipfs_accelerate_py.mcp` | Best-effort `ensure_packages({fastapi, uvicorn, fastmcp})` | **May install packages** when policy allows |
| `ipfs_accelerate_py.mcp.server` | Same class of best-effort ensure for fastapi/uvicorn/fastmcp | Compatibility surface |

Auto-install is controlled by `IPFS_ACCEL_AUTO_INSTALL`:

- unset: enabled inside a virtualenv, skipped outside;
- set to `0` / `false` / `no`: never auto-install;
- other truthy values: allow best-effort `pip install`.

**Production:** install `mcp` / `mcp-p2p` (and host deps) explicitly, set
`IPFS_ACCEL_AUTO_INSTALL=0`, then verify the capability report. Auto-install is
a convenience for local development, not a deploy contract.

Presence of an optional extra is install-time capability only. It does not prove
a GPU, IPFS daemon, libp2p peer, provider credential, or external queue is
available.

## Compatibility migration

1. Prefer imports from `ipfs_accelerate_py.mcp_server` for new code.
2. Replace `create_mcp_server` with `create_server` at call sites when practical.
3. Replace `python -m ipfs_accelerate_py.mcp.cli` with
   `python -m ipfs_accelerate_py.mcp_server.fastapi_service` for pure HTTP
   hosting; keep the compatibility CLI only when you need its TaskQueue/libp2p
   worker flags.
4. Keep rollback/dry-run flags available for operational safety during cutover;
   do not document them as the preferred steady-state path.
5. Treat facade success as a bridge result, not a second ownership model.

## Testing and rollout

Default tests use fake routers/providers and require no network:

```bash
python -m pytest \
  test/mcp_server/test_ai_catalog_tools.py \
  test/mcp_server/test_ai_router_text_embedding_tools.py \
  test/mcp_server/test_ai_router_vision_voice_tools.py \
  test/test_mcplusplus_ai_catalog_idl.py \
  test/test_mcplusplus_ai_catalog.py \
  test/test_ai_catalog_conformance.py -q
```

Usage control tools (`model_catalog_usage`, `model_catalog_usage_metrics`,
`route_preview`) share reason codes and authorities with the Python control
service and MCP++ `ai.usage.v1` IDL. Offline parity and rollout proofs:

```bash
python -m pytest \
  test/test_endpoint_usage_controls.py \
  test/test_endpoint_usage_conformance.py \
  test/test_endpoint_usage_faults.py \
  test/test_endpoint_usage_rollout.py -q
```

Read/preview paths are side-effect free and never reserve capacity. Admin
mutations require `ai.usage/admin`, expected revision, lease/fence, and
idempotency. Distributed admission without a fenced coordinator fails closed.

Roll out read-only catalog tools first and compare their stable IDs/revisions
with Python and legacy views. Then enable refresh for named sources, run
per-modality live canaries, publish MCP++ advertisements, and finally promote
invocation traffic. Monitor catalog conflicts, no matches, stale records,
source latency, cache hit/miss rates, resolution outcomes, and health
transitions.

For opt-in live smoke configuration, see
[Testing and live smoke](architecture/AI_SERVICE_CATALOG.md#testing-and-live-smoke).
Unselected modalities skip, so a deployment can test only providers it has.

## Rollback and troubleshooting

On identity/revision drift, unexpected side effects, schema incompatibility,
security failure, or elevated routing errors:

1. stop new catalog-selected invocation traffic;
2. disable peer federation and side-effecting refresh;
3. keep the last immutable snapshot for read-only diagnosis;
4. withdraw or stop routing to the new MCP++ advertisement;
5. pin the previous server release;
6. restore compatible tool names and rerun offline parity tests.

| Symptom | Action |
| --- | --- |
| Import / missing FastMCP or Flask | Install `ipfs-accelerate-py[mcp]` explicitly; set `IPFS_ACCEL_AUTO_INSTALL=0` in production |
| Server starts but tools missing | Enable `IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP=1`; query meta-tools / capability report |
| Only `/healthz` responds | Use `fastapi_service` (or `--fastapi`); bare standalone shell is not a full MCP host |
| Status cannot connect | Confirm host/port, process still running, and which entrypoint default port you used (CLI `9000` vs FastAPI `8000`) |
| P2P / queue fails | Install `mcp-p2p`, configure queue path and ports, verify identity/firewall |
| Unexpected `pip install` on import | Compatibility auto-install; set `IPFS_ACCEL_AUTO_INSTALL=0` and preinstall extras |
| Facade lands on legacy runtime | Check `IPFS_MCP_FORCE_LEGACY_ROLLBACK` and dry-run flags; prefer direct `mcp_server` imports |
| `no_match` | Inspect safe resolution reasons and required operational state |
| `ambiguous_identifier` | Retry with a stable ID rather than an alias |
| revision mismatch | Re-list or re-resolve and bind the request to the returned revision |
| refresh denied | Use refresh—not read—authority and specify named sources |
| redacted endpoint | Expected; invoke by binding/deployment ID |
| oversized media/output | Reduce the bounded request or select a compatible binding |

## Related documentation

- [MCP setup guide](guides/MCP_SETUP_GUIDE.md)
- [MCP quick start](guides/QUICK_START_MCP.md)
- [MCP runtime architecture](architecture/MCP_RUNTIME.md)
- [Canonical server README](../ipfs_accelerate_py/mcp_server/README.md)
- [MCP dashboard guide](MCP_DASHBOARD_GUIDE.md)
- [AI Service Catalog architecture](architecture/AI_SERVICE_CATALOG.md)
- [LLM Router](LLM_ROUTER.md)
- [MCP++ records](../mcpplusplus/README.md)
- [Installation](guides/getting-started/installation.md)
