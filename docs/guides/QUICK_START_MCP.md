# MCP Quick Start

**Status:** Current

**Owner:** MCP maintainers

**Audience:** Operators and developers starting the canonical local HTTP host

**Sources:** `requirements.txt`; `pyproject.toml`;
`ipfs_accelerate_py/mcp_server/fastapi_service.py`;
`ipfs_accelerate_py/mcp_server/server.py`; `ipfs_accelerate_py/cli.py`

**Last-verified:** 2026-08-03 @ `d5f3aa5c6`; install metadata, startup routes,
and compatibility behavior rechecked

**Freshness triggers:** MCP dependency, route, entrypoint, transport,
compatibility, or auto-install changes

This is the short path for starting the optional MCP server from the
**canonical** package `ipfs_accelerate_py.mcp_server`. MCP is an integration
boundary; it is not required for direct Python inference or the unified CLI.

For full configuration, compatibility labels, and auto-install policy, see the
[setup guide](MCP_SETUP_GUIDE.md) and [server reference](../MCP_SERVER.md).

## Install

Starting MCP is optional, but current base metadata already installs the
server code and names FastMCP plus the Flask/PyGithub stack in
`requirements.txt`. The `mcp` extra repeats that set and adds `async-timeout`;
it records explicit feature intent rather than isolating MCP from the base
wheel. The preferred FastAPI host also imports FastAPI and Uvicorn, which are
not direct members of that extra, so verify or pin those imports in locked
environments.

```bash
python -m pip install -e ".[mcp]"
```

Published package equivalent:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
```

For production or locked environments, install extras explicitly and disable
import-time auto-install:

```bash
export IPFS_ACCEL_AUTO_INSTALL=0
```

The compatibility package `ipfs_accelerate_py.mcp` may otherwise attempt
best-effort installs of `fastapi` / `uvicorn` / `fastmcp` when auto-install
policy allows (default on inside a virtualenv).

## Start the canonical HTTP host

```bash
export IPFS_MCP_HOST=127.0.0.1
export IPFS_MCP_PORT=8000
export IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP=1
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

Health check:

```bash
curl -sS "http://127.0.0.1:8000/healthz"
curl -sS "http://127.0.0.1:8000/mcp/health"
```

Programmatic construction:

```python
from ipfs_accelerate_py.mcp_server import create_server

server = create_server(name="ipfs-accelerate")
```

## Product CLI (optional)

Dashboard-oriented operator start (default port **9000**):

```bash
python -m ipfs_accelerate_py.cli mcp start --host 127.0.0.1 --port 9000
python -m ipfs_accelerate_py.cli mcp status --host 127.0.0.1 --port 9000
```

Use `ipfs-accelerate mcp …` when that console script is installed.

## Do not use as the primary path

| Path | Why it is secondary |
| --- | --- |
| `python -m ipfs_accelerate_py.mcp.cli` | Compatibility facade; TaskQueue/libp2p worker CLI only |
| `from ipfs_accelerate_py.mcp.server import create_mcp_server` | Legacy factory (bridges to unified by default) |
| `python -m ipfs_accelerate_py.mcp_server` without `--fastapi` | Incomplete lifecycle shell for MCP clients |

## Inspect tools

```bash
curl -sS "http://127.0.0.1:8000/mcp/tools/list"
```

Do not assume a tool exists merely because an older guide listed it.

## Auto-healing (optional)

Auto-healing is opt-in proposal generation for MCP errors. It does not bypass
deterministic validation, repository policy, or human review. See
[Auto-Healing](../features/auto-healing/README.md).

## Related documentation

- [MCP setup](MCP_SETUP_GUIDE.md)
- [MCP server reference](../MCP_SERVER.md)
- [MCP runtime architecture](../architecture/MCP_RUNTIME.md)
- [MCP dashboard](../MCP_DASHBOARD_GUIDE.md)
- [Testing guide](../development/testing.md)
