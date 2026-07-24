# MCP Quick Start

This is the maintained short path for starting the optional MCP server. MCP
is an integration boundary; it is not required for direct Python inference or
for the unified CLI.

## Install

```bash
python -m pip install -e ".[mcp]"
```

For the full optional dependency set:

```bash
python -m pip install -e ".[full]"
```

## Start and Check the Server

```bash
ipfs-accelerate mcp start --host 127.0.0.1 --port 8000
ipfs-accelerate mcp status --host 127.0.0.1 --port 8000
```

The server exposes the current MCP tool and resource surface through the
package's capability manifest. The dashboard is available with
`ipfs-accelerate mcp dashboard` when that optional component is installed.

## Auto-Healing

Auto-healing is opt-in proposal generation for MCP errors. It does not bypass
deterministic validation, repository policy, or human review. See
[Auto-Healing](../features/auto-healing/README.md) for configuration and
evidence requirements.

## Related Documentation

- [MCP setup](MCP_SETUP_GUIDE.md)
- [MCP dashboard](../MCP_DASHBOARD_GUIDE.md)
- [MCP integration](../features/mcp-integration/README.md)
- [Current documentation state](../development/DOCUMENTATION_CURRENT_STATE.md)
- [Testing guide](../development/testing.md)
