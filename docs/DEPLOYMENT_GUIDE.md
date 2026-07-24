# Multi-Protocol Deployment

This compatibility path used to describe a single “unified inference service.”
The current repository has several independently deployable surfaces, and the
right one depends on the workload:

- [Deployment guide](guides/deployment/README.md) for the maintained selection
  and operational checklist;
- [MCP setup](guides/MCP_SETUP_GUIDE.md) for the canonical FastAPI/Uvicorn
  server and health checks;
- [HuggingFace model server](features/hf-model-server/README.md) for the
  optional OpenAI-compatible model-serving surface;
- [Docker guide](guides/docker/README.md) for the repository's container
  configurations; and
- [P2P guide](guides/p2p/README.md) for optional TaskQueue/libp2p services.

The package does not silently enable every protocol, provider, or hardware
backend. Install the required extras, inspect `get_capabilities(detail=True)`,
and validate the selected service in the target environment.

## Minimal local deployment

```bash
python -m pip install "ipfs-accelerate-py[minimal]"
python - <<'PY'
from ipfs_accelerate_py import get_instance
print(get_instance().get_capabilities(detail=True))
PY
```

For the canonical MCP service:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Keep development listeners on localhost. Remote exposure requires an
authenticated network boundary, TLS, resource limits, logging, and a process
manager. GPU, IPFS, P2P, external-provider, and supervisor deployments need
their own capability and failure-mode checks.

## Related documentation

- [Installation](guides/getting-started/installation.md)
- [Quick start](guides/QUICKSTART.md)
- [Architecture overview](architecture/overview.md)
- [Testing](development/testing.md)
