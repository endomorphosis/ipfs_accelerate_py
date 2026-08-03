# MCP and MCP++ runtime architecture

**Status:** Current  
**Audience:** Integrators, operators, and implementation agents selecting an MCP
runtime, registering tools, or tracing dispatch through validation and policy  
**Scope:** Canonical MCP server ownership, compatibility and alternate packages,
tool registry and hierarchical dispatch, transport boundaries, MCP++ primitives
(IDL, CID artifacts, UCAN, temporal policy, Event DAG, risk/frontier), and
import side effects that affect process startup  
**Non-goals:** Operator install and journey steps (see `docs/MCP_SERVER.md` and
setup guides when refreshed under DOC-023); full MCP++ chapter checklists
(`mcpplusplus/`); sibling-repo ownership maps (DOC-010); model catalog vs
router plane detail (DOC-007); agent-supervisor control-plane design beyond the
MCP tool category boundary  
**Last verified:** `f279353053fe41593d76a95245416933d08e8999` (2026-08-03);
paths and symbols checked against `ipfs_accelerate_py/mcp_server/`,
`ipfs_accelerate_py/mcp/`, `ipfs_accelerate_py/mcplusplus_module/`,
`mcpplusplus/`, and focused unified-bootstrap / transport test modules

## Source anchors

| Concern | Path / symbol | Notes |
| --- | --- | --- |
| Canonical package | `ipfs_accelerate_py/mcp_server/` | Authoritative runtime |
| Package exports | `ipfs_accelerate_py/mcp_server/__init__.py` | Lazy `_EXPORT_MAP` re-exports |
| Server construct | `mcp_server.server.create_server` | Canonical builder + optional bootstrap |
| Unified bootstrap | `mcp_server.server._attach_unified_bootstrap` | Meta-tools, services, policy hooks |
| Config / feature flags | `mcp_server.configs.UnifiedMCPServerConfig` | Env-driven MCP++ gates |
| Hierarchical registry | `mcp_server.hierarchical_tool_manager.HierarchicalToolManager` | Category loaders + dispatch |
| Flat registry | `mcp_server.tool_registry.ToolRegistry` | Class/function tool store |
| Dispatch pipeline | `mcp_server.dispatch_pipeline` | Param coercion + intent CID |
| Input validation | `mcp_server.validators.validate_dispatch_inputs` | Fail-closed category/tool_name |
| Runtime router | `mcp_server.runtime_router.RuntimeRouter` | fastapi / trio execution routing |
| MCP++ primitives | `mcp_server/mcplusplus/` | Artifacts, UCAN, policy, Event DAG, risk |
| FastAPI transport | `mcp_server.fastapi_service` | HTTP JSON-RPC without legacy facade |
| Standalone process | `mcp_server.standalone_server` | Process entry; `python -m ipfs_accelerate_py.mcp_server` |
| MCP+p2p facade | `mcp_server.mcp_p2p_transport` | Delegates to `p2p_tasks.mcp_p2p` |
| Compatibility facade | `ipfs_accelerate_py/mcp/` | Legacy import path; bridge to unified |
| Facade factory | `mcp.server.create_mcp_server` | Default unified bridge; rollback flags |
| Compatibility auto-install | `mcp/__init__.py`, `mcp/server.py` | Best-effort `ensure_packages` on import |
| Shared compat helpers | `mcp_server.compatibility` | Storage / P2P registrar resolvers |
| Trio alternate package | `ipfs_accelerate_py/mcplusplus_module/` | Trio-native MCP++ surface |
| Spec / conformance records | `mcpplusplus/` | Checklist and gap matrix (not runtime) |
| Wave A loaders | `mcp_server.wave_a_loaders.configure_wave_a_loaders` | `ipfs`, `workflow`, `p2p` |
| Focused tests | `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` | Bootstrap + dispatch contracts |
| Transport matrix | `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py` | HTTP / p2p parity of policy shape |

## Context and component map

MCP is the **tool and protocol plane**: clients discover tools, call them, and
receive structured results. It is not the model catalog, not endpoint usage
accounting, and not the agent-supervisor objective heap. Those planes may expose
tools *through* MCP, but authority for inventory, quota, and completion remains
with their owning packages.

```text
  MCP clients (CLI, IDE, agents, HTTP, p2p peers)
                    |
        +-----------+-----------+
        |     Transports        |   (delivery only — not policy authority)
        |  stdio / FastAPI      |
        |  Trio / gRPC / p2p    |
        +-----------+-----------+
                    |
        +-----------v-----------+
        | Canonical runtime     |
        | ipfs_accelerate_py    |
        |   .mcp_server         |
        |  registry + meta-tools|
        |  validation / UCAN /  |
        |  policy / risk        |
        |  tools/* execution    |
        +-----------+-----------+
                    |
     optional services (capability-gated)
     inference routers, IPFS, p2p_tasks, secrets vault
```

### Package roles

| Package / tree | Role | Prefer for |
| --- | --- | --- |
| `ipfs_accelerate_py.mcp_server` | **Canonical** runtime: server construct, hierarchical tools, unified dispatch, MCP++ primitives, FastAPI/standalone/p2p facades | New integrations, embedding, operators |
| `ipfs_accelerate_py.mcp` | **Compatibility facade**: historical import path; `create_mcp_server` bridges to unified by default; some modules auto-install optional deps | Legacy callers mid-migration only |
| `ipfs_accelerate_py.mcplusplus_module` | **Alternate Trio-first** MCP++ implementation (trio server, p2p helpers); shims often *delegate* to canonical tools | Trio-native hosts already on this package |
| `mcpplusplus/` | Spec conformance, gap matrix, unification plan | Evidence and chapter status — not importable runtime |
| `ipfs_accelerate_py.p2p_tasks` | Framed MCP-over-p2p protocol implementation | Transport peer path; still dispatches into server tools |

**Integrator selection rule:** import and start
`ipfs_accelerate_py.mcp_server` (`create_server`, `fastapi_service`, or
`standalone_server`). Treat `mcp` as a compatibility entry that may still work
but is not the ownership target. Treat `mcplusplus_module` as a specialized
Trio surface that should not fork unique business logic.

## Entrypoints

| Entry | Package | When to use |
| --- | --- | --- |
| `from ipfs_accelerate_py.mcp_server import create_server` | canonical | Embed or script the server in-process |
| `python -m ipfs_accelerate_py.mcp_server` | canonical | Module entry → `standalone_server.main` |
| `python -m ipfs_accelerate_py.mcp_server.fastapi_service` | canonical | HTTP service using `IPFS_MCP_*` env |
| `python -m ipfs_accelerate_py.mcp.cli …` | compatibility CLI surface | Host options for task worker / libp2p; still expected to land on unified runtime when bridge is enabled |
| `from ipfs_accelerate_py.mcp.server import create_mcp_server` | compatibility | Legacy factory; **defaults to unified bridge** unless rollback flags force legacy |
| `ipfs_accelerate_py.mcplusplus_module` Trio server | alternate | Trio-native deployments; prefer delegating tools to `mcp_server` |

Optional install extras (see `pyproject.toml`): `mcp` (FastMCP and related),
`mcp-p2p` (p2p transport deps). Presence of an extra is a install-time
capability, not proof that a peer, GPU, or IPFS node is available.

## Flows

### 1. Startup and registration

```text
create_server()
    |
    v
MCPServerWrapper base construct
    |
    +-- UnifiedMCPServerConfig.from_env()
    |
    +-- if IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP:
            _attach_unified_bootstrap()
                |
                +-- RuntimeRouter + HierarchicalToolManager
                +-- Wave A category loaders (ipfs, workflow, p2p)
                +-- many native category loaders under tools/*
                +-- MCP++ services as lazy factories
                +-- register meta-tools:
                      tools_list_categories
                      tools_list_tools
                      tools_get_schema
                      tools_dispatch
                      tools_runtime_metrics
                +-- optional monitoring / OTEL / Prometheus / secrets vault
```

Category loaders are **lazy**: listing or dispatching a category triggers
registration. Preload is optional via `IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES`
(`ipfs`, `workflow`, `p2p`, or `all`). Agent-supervisor tools use a dedicated
lazy loader so merely importing the server does not start a supervisor.

### 2. Tool invocation (authority path)

Transport delivers a `tools/call` (or HTTP JSON-RPC equivalent). Authority for
whether the tool may run is **not** the transport. The canonical control path
is unified meta-tool dispatch:

```text
Client tools/call
        |
        v
Transport adapter (stdio | FastAPI | Trio | gRPC facade | MCP+p2p)
        |  unwrap framing / HTTP / stream only
        v
Registered tool handler
        |
        +-- discovery meta-tools --> HierarchicalToolManager list/schema APIs
        |
        +-- tools_dispatch(category, tool_name, parameters)
                    |
                    v
            validate_dispatch_inputs   # non-empty strings; normalize params
                    |
                    v
            compute_dispatch_intent_cid  # deterministic intent identity
                    |
                    v
            optional control keys stripped from payload
              (__enforce_ucan, __ucan_proof_chain, __enforce_policy,
               __policy_clauses, __enforce_risk, __emit_artifacts, …)
                    |
                    v
            [if enforce_risk] RiskScorer — may return risk_denied
                    |
                    v
            [if enforce_ucan] validate_raw_delegation_chain
                    |            may return authorization_denied
                    v
            [if enforce_policy] evaluate_with_ipfs_datasets_policy
                    |            may return policy_denied
                    v
            HierarchicalToolManager / RuntimeRouter execute tool
                    |
                    v
            optional result cache, peer probe, frontier schedule
                    |
                    v
            artifact / Event DAG / audit / metrics (best-effort or config-gated)
                    |
                    v
            structured response {ok, result|error, …}
```

**Transport is not authority.** A successful HTTP or p2p delivery only means the
request reached the server. Admission is validation + optional risk + UCAN +
temporal policy, then execution. The same `tools_dispatch` path is what
transport matrix tests use to prove policy decision shape parity across HTTP and
MCP+p2p.

### 3. Compatibility facade bridge

```text
create_mcp_server()   # ipfs_accelerate_py.mcp.server
        |
        +-- IPFS_MCP_FORCE_LEGACY_ROLLBACK?
        |         yes --> legacy MCPServerWrapper only
        |
        +-- IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN?
        |         yes --> probe unified create_server, then stay on legacy
        |
        +-- else (default)
                  create_server() from mcp_server  --> unified runtime
                  attach facade telemetry (_mcp_facade_telemetry)
```

Deprecation telemetry labels the facade path; do not treat facade success as a
second runtime ownership model.

## State and identity

| Artifact / ID | Owner | Meaning |
| --- | --- | --- |
| `intent_cid` | `dispatch_pipeline.compute_dispatch_intent_cid` | Content-addressed dispatch identity over category, tool, parameters |
| `decision_cid` | MCP++ `artifacts` + policy path | Persisted temporal-policy decision payload |
| `receipt_cid` / `event_cid` | artifact emission + Event DAG | Execution receipt and provenance node |
| Tool name + category | `HierarchicalToolManager` / `ToolRegistry` | Discovery and dispatch keys |
| `catalog_revision` (AI tools) | model catalog plane via tools | Bound to catalog/router tools; not invented by transport |
| Peer multiaddrs / peer_id | peer registry / bootstrap services | Discovery metadata; untrusted until validated |
| Policy audit entries | `policy_audit_log.PolicyAuditLog` | In-memory decision trail when audit enabled |
| Artifact store | memory or JSON path from config | Optional durable chain for replay |

Feature flags that gate durable or security state are read from environment
through `UnifiedMCPServerConfig` (for example
`IPFS_MCP_SERVER_ENABLE_UCAN_VALIDATION`,
`IPFS_MCP_SERVER_ENABLE_POLICY_EVALUATION`,
`IPFS_MCP_SERVER_ENABLE_CID_ARTIFACTS`,
`IPFS_MCP_SERVER_ARTIFACT_STORE_BACKEND` = `memory` | `json`).

### Supported MCP++ profile strings (bootstrap advertisement)

`get_unified_supported_profiles()` currently advertises:

- `mcp++/idl`
- `mcp++/cid-envelope`
- `mcp++/ucan`
- `mcp++/deontic-policy`
- `mcp++/p2p-transport`
- `mcp++/risk-scheduling`

Advertisement is negotiation metadata. A profile string in a snapshot does not
prove every optional dependency (libp2p, Prometheus, secrets backend) is live.

## Trust, authorization, and failure semantics

### Trust ladder

| Signal | Means | Does not mean |
| --- | --- | --- |
| Import of `mcp_server` | Package present | Tools, GPUs, or peers work |
| Import of `mcp` | Compatibility package present | Safe for production; may auto-install packages |
| Meta-tool list succeeds | Bootstrap attached and registry wired | Authorization configured |
| Capability / profile list | Server *claims* profile support | Peer negotiated that profile |
| UCAN allow / policy allow | Dispatch admission for that call | Future calls free without re-check |
| Transport connected | Bytes can flow | Caller is authorized |
| Conformance checklist PASS in `mcpplusplus/` | Documented evidence at last update | Current tree without re-run |

**Import success is never a capability signal. Capability is never proof.
Transport is never authorization.**

### Fail-closed conditions

When the corresponding enforcement flag is on (config default or per-call
`__enforce_*` control key):

| Condition | Dispatch error | Effect |
| --- | --- | --- |
| Missing/invalid category or tool_name | `invalid_dispatch_parameter` | No execution |
| Malformed control booleans/lists/dicts | `invalid_dispatch_parameter` | No execution |
| Risk score above threshold | `risk_denied` | No execution; audit if enabled |
| UCAN chain fails validation | `authorization_denied` | No execution; audit if enabled |
| Temporal policy decision `deny` | `policy_denied` | No execution; decision may still be persisted |
| Circuit breaker open (category) | tool execution error from manager | Reject until recovery timeout |

UCAN and policy evaluation are independent gates: either deny stops the call
even if the other would allow. Disabled enforcement skips that gate; it does
not invent an allow decision for audit.

### Degradation

| Missing optional piece | Behavior |
| --- | --- |
| FastAPI / uvicorn not installed | Canonical package import may succeed; HTTP entrypoints fail at use |
| Trio not installed | `TRIO_AVAILABLE` false; trio runtime routing unavailable |
| gRPC source module unavailable | `grpc_transport` facade raises / stubs unavailability |
| IPFS kit / audit storage unavailable | Event DAG may still record in-memory; durable audit storage skipped |
| Unified bootstrap disabled | Base server without meta-tool MCP++ control plane |
| Unified bridge failure from `mcp` facade | Logs warning and falls back to legacy wrapper |
| Native category loader exception | Warning log; other categories remain |

### Recovery

- Retry after fixing inputs or providing UCAN/policy material.
- Circuit breakers move OPEN → HALF_OPEN after recovery timeout.
- JSON artifact store can reload at bootstrap when path is configured.
- Force-legacy rollback is an explicit operator escape hatch, not the default
  long-term path.

### Non-authoritative signals

- Chat or agent prose claiming a tool ran
- Board / ticket status
- Cache hits without re-admission when enforcement is required
- Transport-level HTTP 200 alone
- Historical plans under `docs/architecture/MCP_SERVER_UNIFICATION_PLAN.md` or
  `mcpplusplus/SERVER_UNIFICATION_PLAN.md` (intent / history, not live API)

## Compatibility imports and side effects

Integrators must know which imports are pure and which mutate the process.

| Import path | Side effects at import time | Notes |
| --- | --- | --- |
| `ipfs_accelerate_py.mcp_server` | Lazy export resolution only; no auto-install of FastAPI/FastMCP | Preferred |
| `ipfs_accelerate_py.mcp_server.server` | Configures root logging via `configure_root_logging()` | Logging only |
| `ipfs_accelerate_py.mcp` package init | Best-effort `ensure_packages({fastapi, uvicorn, fastmcp})` | **May install packages** when auto-install policy allows |
| `ipfs_accelerate_py.mcp.server` | Same class of best-effort `ensure_packages` for fastapi/uvicorn/fastmcp | Compatibility surface |
| Some `mcp` tools (e.g. inference helpers) | May call `ensure_packages` on optional third-party libs | Capability path, not discovery |
| `mcplusplus_module` | Resolves stubs for missing optional symbols; may import compatibility helpers | Prefer canonical for new code |
| `mcp_server.compatibility` resolvers | Import-time module probes across historical locations | Used by shims; not a second registry |

Environment flags that change facade **runtime selection** (not install):

| Variable | Effect |
| --- | --- |
| `IPFS_MCP_FORCE_LEGACY_ROLLBACK` | Keep compatibility factory on legacy wrapper |
| `IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN` | Exercise unified create then continue legacy |
| `IPFS_MCP_ENABLE_UNIFIED_BRIDGE` | Explicit bridge telemetry; unified is already default when not forced legacy |
| `IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP` | Attach hierarchical meta-tools and MCP++ services on `create_server` |

Do not rely on auto-install for production images: install `mcp` / `mcp-p2p`
extras (and host deps) explicitly, then verify capabilities.

## Transports (delivery plane)

| Transport | Primary modules | Role |
| --- | --- | --- |
| Process / standalone | `standalone_server.run_server` | Local process lifecycle around `create_server` |
| FastAPI / HTTP JSON-RPC | `fastapi_service`, `fastapi_config` | Mount at `IPFS_MCP_MOUNT_PATH` (default `/mcp`) |
| Compatibility FastAPI | historical paths under `mcp/` | Prefer canonical `fastapi_service` |
| Trio adapter | `trio_adapter.TrioMCPServerAdapter` | Optional Trio lifecycle facade over `create_server` |
| MCP+p2p | `mcp_p2p_transport` → `p2p_tasks.mcp_p2p` + `mcplusplus.p2p_framing` | Framed JSON-RPC over libp2p streams |
| gRPC | `grpc_transport` | Compatibility facade delegating to source module when present |

Transports **normalize** requests into tool calls. They must not:

- skip `validate_dispatch_inputs` / UCAN / policy when enforcement is enabled;
- treat peer advertisement as catalog or credential authority;
- invent tool results on framing success.

Profile-G style REST routes under FastAPI (goals, risk, schedule) are optional
HTTP surfaces attached by the FastAPI service; they still sit on the same
runtime services and do not replace meta-tool dispatch ownership.

## Extension and compatibility

### Safe extension

1. Add tools under `ipfs_accelerate_py/mcp_server/tools/<category>/` with a
   native registrar.
2. Register a category loader on `HierarchicalToolManager` (lazy).
3. Prefer meta-tool discovery (`tools_list_*` / `tools_get_schema`) over forcing
   clients to load the full schema set.
4. Keep business logic out of transport modules and out of `mcp` shims.
5. For MCP++ descriptors, use `mcplusplus.idl_registry` / `tools/idl` so
   interface CIDs stay deterministic.

### Compatibility boundary

- `mcp` remains for source-compatible imports and CLI habits during migration.
- Deferred modules (enterprise API, investigation client, NL-UCAN helpers, etc.)
  may exist as thin adapters; they are not alternate authority planes.
- `mcplusplus_module` tool registrars should **delegate** to canonical native
  tools; unique scheduler logic belongs in `mcp_server`.
- Unification and chapter evidence live in `mcpplusplus/` documents; runtime
  claims in this guide follow live `mcp_server` code first.

## Rationale

1. **Single dispatch authority** — One hierarchical manager and `tools_dispatch`
   path keep validation, UCAN, policy, risk, and artifacts coherent across
   transports.
2. **Transport neutrality** — stdio, HTTP, Trio, and p2p are adapters so peer
   connectivity cannot upgrade privileges by itself.
3. **Lazy categories** — Large tool surfaces load on demand; importing the
   server stays a discovery-safe operation for most categories.
4. **Optional security features** — UCAN, policy, CID artifacts, and risk are
   env-gated so minimal deployments stay light while high-assurance paths exist.
5. **Compatibility without dual ownership** — The `mcp` facade defaults to the
   unified runtime so old imports do not permanently fork behavior.

## Alternatives

| Alternative | Breakage |
| --- | --- |
| Keep `mcp` as long-term canonical package | Split ownership; auto-install side effects on import; dual registries |
| Authorize at transport only (API key / peer ID) | Bypasses tool-scoped UCAN and temporal policy; peers become superusers |
| Eager-load every tool category at import | Heavy startup; accidental supervisor / network side effects |
| Collapse catalog + usage + MCP into one registry | Mixes inventory, quota, and invocation authority (rejected under DOC-007) |
| Implement MCP++ only in `mcplusplus_module` | Trio-only island; asyncio/FastAPI hosts lose parity |

## Consequences

**Positive**

- Integrators have one clear import target (`mcp_server`).
- Dispatch, policy, and artifacts share identity (`intent_cid` and related CIDs).
- Transports can be added without reimplementing admission.
- Lazy loaders and feature flags keep optional cost off the critical path.

**Negative**

- Dual packages (`mcp` vs `mcp_server`) confuse readers until guides point here.
- Compatibility auto-install can surprise locked-down environments.
- Many env flags increase operator configuration surface.
- Facade fallback to legacy can hide incomplete cutover if telemetry is ignored.
- Some deferred modules still proxy to external/source packages when present.

## Operational signals

| Signal | Where |
| --- | --- |
| Facade selection telemetry | `_mcp_facade_telemetry` on server from `create_mcp_server` |
| Runtime metrics meta-tool | `tools_runtime_metrics` |
| Metrics collector | `monitoring.EnhancedMetricsCollector` / P2P metrics |
| Prometheus | `prometheus_exporter.PrometheusExporter` when enabled |
| OTEL | `otel_tracing.configure_tracing` when enabled |
| Policy audit stats | `PolicyAuditLog.stats()` in dispatch responses when audit on |
| MCP+p2p counters | `get_mcp_p2p_stats()` |
| Logs | `mcp_server.logger` / package loggers |

Health is process- and dependency-specific. Prefer probing meta-tools and a
known no-op or schema call over assuming a listening port implies a full MCP++
stack.

## Verification

```bash
# Structural
test -f docs/architecture/MCP_RUNTIME.md
rg -q 'mcp_server' docs/architecture/MCP_RUNTIME.md
rg -qi 'compatib' docs/architecture/MCP_RUNTIME.md
git diff --check

# Source anchors present
test -d ipfs_accelerate_py/mcp_server
test -f ipfs_accelerate_py/mcp_server/server.py
test -f ipfs_accelerate_py/mcp_server/configs.py
test -f ipfs_accelerate_py/mcp/server.py
rg -n 'def create_server' ipfs_accelerate_py/mcp_server/server.py
rg -n 'def tools_dispatch' ipfs_accelerate_py/mcp_server/server.py
rg -n 'def validate_dispatch_inputs' ipfs_accelerate_py/mcp_server/validators.py
rg -n 'ensure_packages' ipfs_accelerate_py/mcp/__init__.py
rg -n 'FORCE_LEGACY_ROLLBACK|create_mcp_server' ipfs_accelerate_py/mcp/server.py

# Focused contract tests (optional; deterministic, no live network required)
# python -m pytest ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py -q
# python -m pytest ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py -q
```

Review checklist: status/audience/scope present; source anchors current;
integrators can pick `mcp_server` as canonical; compatibility auto-install and
rollback flags disclosed; dispatch flow shows validation/policy before
execution; transports labelled as non-authoritative; rationale, alternatives,
consequences, and failure semantics present; no invented public APIs.

## Related guides and records

| Document | Relation |
| --- | --- |
| [Architecture overview](overview.md) | System layers; points here for MCP |
| [Guide conventions](GUIDE_CONVENTIONS.md) | ArchitectureGuideContract@1 |
| [Canonical server README](../../ipfs_accelerate_py/mcp_server/README.md) | Operator-oriented package notes and env tables |
| [MCP++ workspace records](../../mcpplusplus/README.md) | Conformance checklist and gap matrix (Plan/evidence) |
| [MCP server unification plan](MCP_SERVER_UNIFICATION_PLAN.md) | Historical / planned migration narrative |
| [MCP server user doc](../MCP_SERVER.md) | Product-facing tool catalog notes (journeys refreshed under DOC-023) |
| `docs/architecture/MODEL_SERVICE_ROUTING.md` | Planned sibling: catalog vs usage vs invocation planes |
| `docs/architecture/DISTRIBUTED_RUNTIME.md` | Planned sibling: IPFS / P2P task execution beyond MCP framing |
