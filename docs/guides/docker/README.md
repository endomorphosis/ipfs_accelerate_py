# Docker

This repository includes Docker files for reproducible development, CI, and
selected service workflows. The image and compose files are deployment
artifacts, not a promise that every optional backend is installed in every
image.

## Inspect before building

From the repository root:

```bash
docker build -f Dockerfile -t ipfs-accelerate-py .
docker compose -f docker-compose.yml config
```

Use the compose configuration only after reviewing its services, mounts,
environment variables, ports, credentials, and model/cache paths:

```bash
docker compose -f docker-compose.yml up
docker compose -f docker-compose.yml ps
docker compose -f docker-compose.yml logs -f
docker compose -f docker-compose.yml down
```

The `deployments/`, `install/`, and `docker/` directories contain additional
workflow-specific files. Do not assume that a file in one of those directories
uses the same image name, port, or entry point as the root compose file.

## GPU containers

GPU access requires a compatible host driver, container runtime, and framework
wheel inside the image. Validate the host and container separately:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

The [hardware guide](../hardware/overview.md) explains capability discovery;
the [installation guide](../getting-started/installation.md) covers CUDA
requirements. A container being able to see `/dev/nvidia*` is not by itself a
successful inference check.

## MCP in a container

The canonical service is started through the installed product CLI. Keep the
listener private until authentication and TLS are provided by the deployment:

```bash
ipfs-accelerate mcp start --host 0.0.0.0 --port 9000
```

See [MCP setup](../MCP_SETUP_GUIDE.md) and [deployment guidance](../deployment/README.md)
for the runtime boundary and health check.

## CI and cache files

Some Docker documents in this directory describe GitHub runner or cache
workflows. Treat those as workflow-specific runbooks and verify their
referenced compose files and secrets before use. They do not define the core
Python API.

## Related documentation

- [Deployment](../deployment/README.md)
- [Installation](../getting-started/installation.md)
- [Testing](../../development/testing.md)
- [Hardware](../hardware/overview.md)
