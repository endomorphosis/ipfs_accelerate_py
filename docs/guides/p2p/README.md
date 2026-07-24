# P2P and Distributed Workflows

IPFS, libp2p, TaskQueue, and distributed workflow support are optional
integrations. They are separate from the base local inference path and require
their own dependencies, identities, ports, credentials, and failure handling.

## Install the optional capabilities

```bash
python -m pip install "ipfs-accelerate-py[mcp-p2p]"
# Or install the lower-level libp2p extra when MCP is not needed.
python -m pip install "ipfs-accelerate-py[libp2p]"
```

The P2P extras include dependencies, not a running peer network. Configure the
queue/service and network policy separately.

## MCP-backed P2P

The canonical MCP runtime owns the current server and P2P integration boundary:

```bash
ipfs-accelerate mcp --help
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
```

Use `--no-p2p` when the optional P2P service should be disabled. Before
enabling it, inspect the installed manifest and direct module help:

```bash
python -m ipfs_accelerate_py.mcp.cli --help
python - <<'PY'
from ipfs_accelerate_py import get_instance
print(get_instance().get_capabilities(detail=True).get("mcp", {}))
PY
```

There is no general-purpose `ipfs-accelerate p2p start` command in the current
product CLI. Older examples using that command are historical.

## Code and test boundaries

The implementation is distributed across optional modules rather than one
public P2P facade. Relevant code includes:

- `ipfs_accelerate_py/p2p_workflow_scheduler.py` for workflow scheduling;
- `ipfs_accelerate_py/p2p_tasks/` for TaskQueue/libp2p runtime pieces;
- `ipfs_accelerate_py/mcp_server/` for MCP transport and service integration;
- `ipfs_accelerate_py/mcp/tools/` for registered P2P tool adapters.

Use the MCP and P2P tests as the conformance surface. Networked tests may need
optional packages, ports, peer identity, and an explicit opt-in:

```bash
python -m pytest ipfs_accelerate_py/mcp/tests -q
```

Start with the import/manifest checks and add live network tests only in an
isolated environment.

## Operational checklist

- Pin the optional dependency set and record the peer/runtime versions.
- Configure peer identity, bootstrap addresses, queue limits, and timeouts.
- Keep control-plane and data-plane ports private until authenticated.
- Bound task payloads, concurrency, retries, and cache retention.
- Record content identifiers and receipts for artifacts shared between peers.
- Test degraded operation with IPFS/P2P disabled.
- Monitor memory and shutdown behavior; distributed caches can amplify
  persistence and artifact sizes.

## Related documentation

- [MCP setup](../MCP_SETUP_GUIDE.md)
- [Deployment](../deployment/README.md)
- [MCP/P2P architecture](../../features/mcp-integration/p2p-integration.md)
- [Testing](../../development/testing.md)
