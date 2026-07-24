# MCP Setup

The canonical MCP runtime is `ipfs_accelerate_py.mcp_server`. The
`ipfs_accelerate_py.mcp` package remains a compatibility facade. This guide
covers local development startup and basic health checks; production
authentication, policy, and transport choices belong in the MCP++ records.

## Install

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
```

For P2P TaskQueue support, install the `mcp-p2p` extra and configure the
external queue/service separately.

## Start the server

The product CLI is the normal entry point:

```bash
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
```

Useful options include `--dashboard`, `--open-browser`,
`--disable-autoscaler`, and `--no-p2p`. Keep development servers on localhost
unless authentication and network policy are configured.

Check health from another terminal:

```bash
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

The exact server process and configured paths can be inspected with:

```bash
ipfs-accelerate mcp --help
ipfs-accelerate mcp start --help
```

## Direct module entry points

Use these when embedding or testing a specific transport:

```bash
python -m ipfs_accelerate_py.mcp.cli --host 127.0.0.1 --port 9000
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

Programmatic construction uses the canonical package:

```python
from ipfs_accelerate_py.mcp_server import create_server

server = create_server()
```

The concrete server object and optional dependency behavior are versioned with
the package; inspect its module README and tests before embedding private
attributes.

## Capability and tool inspection

The accelerator capability report includes an MCP manifest when a server is
available:

```python
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print(report.get("mcp", {}))
```

The MCP server exposes tool/schema/runtime inspection through its registered
meta-tools. Do not assume a tool exists merely because an older guide listed
it; query the manifest or use the server's schema endpoint.

## P2P and remote task workers

P2P operation is optional and should be enabled explicitly. The direct MCP CLI
may host TaskQueue/libp2p services when the P2P extras and the corresponding
configuration are present. Use:

```bash
python -m ipfs_accelerate_py.mcp.cli --help
```

before copying deployment flags from another environment. Remote networking
also requires firewall, identity, queue, and authentication configuration.

## VS Code and other clients

Configure a client with the command that is known to work in the target
environment. For a local checkout, the command is typically `ipfs-accelerate`
or the direct Python module path above. Use an absolute working directory and
do not expose credentials in a checked-in client configuration.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `ipfs-accelerate` is missing | Activate the environment and install the package in editable or published mode. |
| `mcp` extra import fails | Install `ipfs-accelerate-py[mcp]` and inspect the first traceback. |
| Status cannot connect | Confirm host/port and that the start command is still running. |
| Tools are missing | Query the runtime manifest; optional categories may be unavailable. |
| P2P startup fails | Install `mcp-p2p`/`libp2p`, configure the queue, and verify ports/identity. |
| Browser dashboard fails | Start the server without `--dashboard` first, then inspect the dashboard-specific logs. |

## Related documentation

- [Canonical MCP server README](../../ipfs_accelerate_py/mcp_server/README.md)
- [MCP++ records](../../mcpplusplus/README.md)
- [Architecture overview](../architecture/overview.md)
- [Installation](getting-started/installation.md)
- [Testing](../development/testing.md)
