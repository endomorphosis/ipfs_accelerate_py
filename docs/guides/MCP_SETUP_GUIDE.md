# MCP Setup Guide

**Status:** Current

**Owner:** MCP maintainers

**Audience:** Operators and developers starting a local or embedded MCP server

**Scope:** Install extras, choose a canonical entrypoint, configure host/bootstrap
flags, run health checks, and recognize compatibility/auto-install pitfalls

**Non-goals:** Full tool catalog contracts (see [MCP server reference](../MCP_SERVER.md));
MCP++ chapter evidence (`mcpplusplus/`); production auth product design

**Sources:** `requirements.txt`; `pyproject.toml`;
`ipfs_accelerate_py/mcp_server/`; `ipfs_accelerate_py/mcp/`;
`ipfs_accelerate_py/cli.py`; `docs/architecture/MCP_RUNTIME.md`

**Last-verified:** 2026-08-03 @ `d5f3aa5c6`; install metadata, commands, fixed
routes, transports, and flags rechecked

**Freshness triggers:** MCP dependency, route, transport, configuration,
entrypoint, compatibility, or auto-install changes

The canonical MCP runtime is `ipfs_accelerate_py.mcp_server`. Start there.
The `ipfs_accelerate_py.mcp` package is a **compatibility facade** for older
imports and the TaskQueue/libp2p worker CLI. It is not the preferred ownership
path for new setups.

## 1. Choose an entrypoint (canonical first)

| Goal | Command / API | Notes |
| --- | --- | --- |
| Functional HTTP MCP host | `python -m ipfs_accelerate_py.mcp_server.fastapi_service` | **Preferred** transport host; `IPFS_MCP_*` env |
| Embed in Python | `from ipfs_accelerate_py.mcp_server import create_server` | Canonical builder; mount a transport yourself |
| Operator CLI + dashboard | `python -m ipfs_accelerate_py.cli mcp start` | Product surface; default port **9000** |
| TaskQueue / libp2p worker host | `python -m ipfs_accelerate_py.mcp.cli …` | **Compatibility** path; may auto-install deps on import |

Avoid routing new work through the compatibility package unless you need its
worker flags. Bare `python -m ipfs_accelerate_py.mcp_server` without
`--fastapi` is a lifecycle shell, not a full MCP client host—use
`fastapi_service` instead.

## 2. Install

The MCP runtime is optional to start, but current base packaging already names
FastMCP, Flask, Flask-CORS, Werkzeug, and PyGithub through `requirements.txt`.
The `mcp` extra repeats those dependencies and adds `async-timeout`; use it to
state deployment intent, not as evidence that the base package excludes MCP.
FastAPI and Uvicorn are imported by the preferred HTTP entrypoint but are not
direct members of the `mcp` extra, so locked deployments must verify or pin
them explicitly.

Published package:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
```

Editable checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[mcp]"
```

Optional networking extra (separate from basic MCP HTTP):

```bash
python -m pip install "ipfs-accelerate-py[mcp-p2p]"
```

| Extra | Installs | Does not prove |
| --- | --- | --- |
| Base dependencies | Currently include FastMCP and the Flask/PyGithub stack | A configured or running MCP service |
| `mcp` | Repeats that MCP/Flask set and adds `async-timeout` | A complete isolation boundary, live providers, GPUs, or auth |
| `mcp-p2p` / `libp2p` | libp2p and related wire deps | Reachable mesh or durable queue |
| `all` | Broad app deps (no native P2P by default) | Production readiness |

Install extras **before** first import in locked-down or production
environments. See [Auto-install caveats](#6-auto-install-caveats).

## 3. Start the canonical FastAPI host

```bash
export IPFS_MCP_HOST=127.0.0.1
export IPFS_MCP_PORT=8000
export IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP=1
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

| Variable | Default | Role |
| --- | --- | --- |
| `IPFS_MCP_HOST` | `0.0.0.0` | Bind host — use `127.0.0.1` for local-only |
| `IPFS_MCP_PORT` | `8000` | Bind port |
| `IPFS_MCP_MOUNT_PATH` | `/mcp` | Wrapper sub-app mount only; does **not** relocate fixed canonical `/mcp` protocol/tool routes |
| `IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP` | off | Attach hierarchical meta-tools and MCP++ services |

The canonical FastAPI service hardcodes JSON-RPC and tool endpoints at `/mcp`,
`/mcp/health`, and `/mcp/tools/*`. A custom `IPFS_MCP_MOUNT_PATH` only moves
the wrapper's minimal sub-application and is not a protocol-prefix setting.

Without the bootstrap flag, `create_server` still builds a base server, but the
unified meta-tool control plane is not attached.

Health checks:

```bash
curl -sS "http://127.0.0.1:8000/healthz"
curl -sS "http://127.0.0.1:8000/mcp/health"
```

Expect JSON with `"status": "ok"`. `/mcp/health` also reports a tool count for
the running server.

Programmatic equivalent:

```python
from ipfs_accelerate_py.mcp_server import create_server

server = create_server(name="ipfs-accelerate", host="127.0.0.1", port=8000)
```

Inspect the versioned module and tests before embedding private attributes.
Prefer mounting through `fastapi_service` when you need HTTP MCP routes.

## 4. Product CLI (dashboard-oriented)

```bash
python -m ipfs_accelerate_py.cli mcp start --host 127.0.0.1 --port 9000
python -m ipfs_accelerate_py.cli mcp status --host 127.0.0.1 --port 9000
```

If the `ipfs-accelerate` console script is on `PATH`, the same verbs work as
`ipfs-accelerate mcp start|status|dashboard`.

Useful start options (see live help for the authoritative list):

| Flag | Effect |
| --- | --- |
| `--dashboard` | Request dashboard integration (start path enables dashboard integration) |
| `--open-browser` | Open the dashboard URL after startup |
| `--disable-autoscaler` | Disable GitHub Actions autoscaler |
| `--no-p2p` | Disable P2P workflow monitoring in the autoscaler |

```bash
python -m ipfs_accelerate_py.cli mcp --help
python -m ipfs_accelerate_py.cli mcp start --help
```

Default product CLI port is **9000**, not the FastAPI service default of
**8000**. Match host/port when running status checks.

## 5. Compatibility paths (legacy — labeled)

Use these only when migrating existing automation or when you need TaskQueue
worker/libp2p host flags that the product CLI does not expose the same way.

```bash
# Compatibility CLI — imports ipfs_accelerate_py.mcp (auto-install side effects)
python -m ipfs_accelerate_py.mcp.cli \
  --host 127.0.0.1 \
  --port 9000 \
  --no-p2p-task-worker \
  --no-p2p-service
```

That CLI calls `create_mcp_server` from the compatibility facade. By default the
factory **bridges to** `ipfs_accelerate_py.mcp_server.create_server` (unified
runtime). Explicit rollback stays available for cutover:

| Variable | Effect |
| --- | --- |
| `IPFS_MCP_FORCE_LEGACY_ROLLBACK` | Stay on the legacy wrapper |
| `IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN` | Probe unified, then continue legacy |
| `IPFS_MCP_ENABLE_UNIFIED_BRIDGE` | Explicit bridge request / telemetry |

For pure HTTP hosting of the canonical runtime, prefer section 3 over this
section. Full flag lists for P2P workers belong in live `--help` output and the
[server reference](../MCP_SERVER.md).

## 6. Auto-install caveats

Importing `ipfs_accelerate_py.mcp` (and some paths that reach it through tool
registrars) may run best-effort `ensure_packages` for `fastapi`, `uvicorn`, and
`fastmcp`. Policy is controlled by `IPFS_ACCEL_AUTO_INSTALL`:

| Setting | Behavior |
| --- | --- |
| unset in a virtualenv | auto-install **enabled** |
| unset outside a virtualenv | auto-install **skipped** |
| `0` / `false` / `no` | never auto-install |
| other truthy values | allow best-effort `pip install` |

Resolving `mcp_server.create_server` is not a pure discovery import: native
registrars load, and the inference registrar currently reaches the
compatibility package. Production images should:

1. install `mcp` / `mcp-p2p` extras explicitly;
2. set `IPFS_ACCEL_AUTO_INSTALL=0`;
3. verify capabilities after deploy.

Optional dependency presence is not proof that a peer, GPU, IPFS node, or
provider credential exists.

## 7. Capability and tool inspection

```python
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print(report.get("mcp", {}))
```

With unified bootstrap enabled, meta-tools
(`tools_list_categories`, `tools_list_tools`, `tools_get_schema`,
`tools_dispatch`, `tools_runtime_metrics`) are the control-plane discovery
surface. Do not assume a historical tool name is still registered.

## 8. P2P and remote task workers

P2P is optional and must be enabled deliberately:

1. install `mcp-p2p` (or `libp2p`);
2. configure queue path, listen ports, and identity;
3. open only the ports you intend;
4. treat remote `call_tool` as a security-sensitive capability.

The product CLI may start TaskQueue p2p services from environment toggles used
by systemd units. The compatibility `mcp.cli` exposes fine-grained
`--p2p-*` / `--no-p2p-*` flags. Always read live help before copying flags
across environments:

```bash
python -m ipfs_accelerate_py.mcp.cli --help
```

External queue durability, firewall rules, and peer authentication are out of
band of the Python extra install.

## 9. VS Code and other clients

Start the canonical HTTP host as a separate process, then configure an
HTTP/Streamable HTTP client endpoint. For clients using the VS Code-style
`mcp.json` shape, the transport pairing is:

```json
{
  "servers": {
    "ipfs-accelerate": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Client configuration wrappers vary, but the transport URL is the fixed
`http://127.0.0.1:8000/mcp` endpoint for the startup example above. Do not
commit credentials. If a client cannot connect to an HTTP MCP URL, it cannot
use `mcp_server.fastapi_service` as a command/stdio substitute.

Note: a full MCP **stdio** transport is not currently implemented under
`mcp_server`. HTTP hosts and product CLI paths are the supported local
startup surfaces today.

## 10. Troubleshooting

| Symptom | Check |
| --- | --- |
| `ipfs-accelerate` missing on PATH | Use `python -m ipfs_accelerate_py.cli …` or install the package so console scripts are available |
| MCP dependency import fails | Inspect base plus `[mcp]` resolution and verify FastAPI/Uvicorn explicitly; inspect the first traceback |
| Unexpected package install on import | Compatibility auto-install; set `IPFS_ACCEL_AUTO_INSTALL=0` |
| Status cannot connect | Confirm the process is up and you used the same host/port (CLI **9000** vs FastAPI **8000**) |
| Tools missing | Enable `IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP=1`; query the capability report / meta-tools |
| Only `/healthz` works | Switch to `fastapi_service` or `--fastapi`; bare module entry is incomplete for MCP clients |
| P2P startup fails | Install `mcp-p2p`, configure queue/ports/identity, verify firewall |
| Browser dashboard fails | Start without browser open first; confirm the Flask/dashboard dependencies (currently present in the base set) |
| Forced onto legacy runtime | Unset `IPFS_MCP_FORCE_LEGACY_ROLLBACK` / dry-run flags, or import `mcp_server` directly |

## Related documentation

- [MCP server reference](../MCP_SERVER.md)
- [MCP quick start](QUICK_START_MCP.md)
- [MCP runtime architecture](../architecture/MCP_RUNTIME.md)
- [Canonical server README](../../ipfs_accelerate_py/mcp_server/README.md)
- [MCP++ records](../../mcpplusplus/README.md)
- [Installation](getting-started/installation.md)
- [MCP dashboard](../MCP_DASHBOARD_GUIDE.md)
- [Testing](../development/testing.md)
