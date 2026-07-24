# MCP Dashboard

The repository contains an optional Flask dashboard around the MCP tool
registry. It is a presentation and observability surface, not a replacement
for the canonical MCP runtime or its capability manifest.

## Start it

Install the MCP extra and use the product CLI:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp dashboard --host 127.0.0.1 --port 9000
```

Alternatively, start the MCP server with the dashboard enabled:

```bash
ipfs-accelerate mcp start \
  --host 127.0.0.1 \
  --port 9000 \
  --dashboard
```

Use `--open-browser` only on a local interactive machine. Keep unauthenticated
development listeners on localhost.

## What the dashboard exposes

The `MCPDashboard` class in `ipfs_accelerate_py/mcp_dashboard.py` provides:

- an HTML dashboard at `/`, `/mcp`, and `/dashboard`;
- feature views for GraphRAG, analytics, RAG, investigation, and Copilot SDK;
- status and observability routes under `/api/mcp/`; and
- tool discovery/dispatch through the configured unified registry when the
  optional registry dependencies are available.

The canonical MCP server also registers dashboard tools such as runtime
metrics, peer status, cache statistics, user information, and TDFOL dashboard
generation. Query the runtime manifest rather than assuming a fixed tool count
or category list.

## Health and inspection

```bash
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
ipfs-accelerate mcp --help
```

For programmatic inspection:

```python
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print(report.get("mcp", {}))
```

## Security and operations

- Do not expose the dashboard or MCP server directly to the public internet
  without authentication, TLS, firewall policy, and a process manager.
- Treat Docker, GitHub, IPFS, P2P, Copilot, and model-provider tools as
  optional capabilities with separate credentials and resource limits.
- Avoid placing tokens or large model/cache payloads in browser configuration.
- Capture the server log and capability report when a tool is missing.

## Related documentation

- [MCP setup](guides/MCP_SETUP_GUIDE.md)
- [Canonical MCP server README](../ipfs_accelerate_py/mcp_server/README.md)
- [Deployment](guides/deployment/README.md)
- [Testing](development/testing.md)
- [Current documentation state](development/DOCUMENTATION_CURRENT_STATE.md)
