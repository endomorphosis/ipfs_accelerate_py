# Deployment

IPFS Accelerate can run as a local Python library, a CLI-managed MCP server,
or an application-specific service. Optional IPFS, P2P, model-provider, and
agent-supervisor integrations should be enabled only when their dependencies
and operational controls are available.

## Choose a deployment surface

- [Installation](../getting-started/installation.md) for package and extra
  selection.
- [Quick start](../QUICKSTART.md) for a local inference or MCP smoke.
- [MCP setup](../MCP_SETUP_GUIDE.md) for the canonical server entry point.
- [P2P guide](../p2p/README.md) for optional queue and libp2p services.
- [Docker guide](../docker/README.md) for the repository's container files.
- [Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md) for maintainer/operator
  workloads.

## Local or managed process

Install the package and verify capabilities before selecting a model or
provider:

```bash
python -m pip install "ipfs-accelerate-py[minimal]"
python - <<'PY'
from ipfs_accelerate_py import get_instance
print(get_instance().get_capabilities(detail=True))
PY
```

For a local MCP service, bind to localhost during development:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

For a production service, put the selected process behind the deployment's
process manager and network boundary. Configure authentication, TLS, logging,
resource limits, health checks, and shutdown behavior in that environment;
the package CLI is not a substitute for those controls.

## Containers in this checkout

The repository contains several Docker configurations for different workflows:

```bash
docker build -f Dockerfile -t ipfs-accelerate-py .
docker compose -f docker-compose.yml config
docker compose -f docker-compose.yml up
```

The `deployments/` and `install/` directories contain additional, workflow-
specific Docker files. Inspect the compose services and environment variables
before using them in a shared or production environment; image names and
service ports are not universal package defaults.

## Scaling and parallelism

Scale only after measuring the selected model/provider and confirming the
resource report. More processes can duplicate model memory, compete for one
GPU, or overload a provider. For the agent supervisor, admission is controlled
by leases, CPU/memory, provider capacity, dependencies, conflicts, and
validation receipts rather than by an arbitrary worker count.

## Deployment checklist

- Verify the installed version and capability report.
- Pin model/provider versions and cache policy for reproducible deployments.
- Keep development MCP listeners on localhost.
- Configure authentication and TLS before remote exposure.
- Set explicit CPU, memory, GPU, disk, and concurrency limits.
- Capture health, error, latency, and shutdown metrics.
- Exercise the same focused tests used by CI in the target environment.
- Keep optional P2P and supervisor services separate from ordinary inference
  until their dependencies and trust boundaries are verified.

## Related references

- [Architecture overview](../../architecture/overview.md)
- [Hardware guide](../hardware/overview.md)
- [Testing guide](../../development/testing.md)
- [Troubleshooting FAQ](../troubleshooting/faq.md)
