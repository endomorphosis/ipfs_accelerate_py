# Frequently Asked Questions

This page answers current setup and runtime questions. For a complete
installation sequence, see [Getting started](../getting-started/README.md) and
the [installation guide](../getting-started/installation.md).

## Installation

### What are the minimum requirements?

Use Python 3.8 or newer. The base package runs on CPU; GPU, browser, IPFS,
MCP, and analysis features are optional extras. Install only the capability
sets needed by the deployment, for example:

```bash
python -m pip install "ipfs-accelerate-py[minimal]"
python -m pip install "ipfs-accelerate-py[mcp]"
```

The available extras are defined in `pyproject.toml`; they are not a promise
that every optional backend is present in every environment.

### Can I install from a checkout?

Yes:

```bash
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
python -m pip install -e .
```

The [installation guide](../getting-started/installation.md) also covers
source builds, CUDA requirements, and optional extras.

### Can I run it offline?

Yes, when models and dependencies are already available locally. Model
downloads, IPFS, remote providers, and P2P services require network access;
they are optional and should be disabled or configured explicitly for an
offline deployment.

## API and inference

### What is the main Python API?

The package-level API is the stable starting point:

```python
import ipfs_accelerate_py

accelerator = ipfs_accelerate_py.get_instance()
print(accelerator.get_capabilities(detail=True))
```

The runtime also exposes `ipfs_accelerate_py.run_model` for the common
inference path. Do not use the retired `IPFSAccelerator` class name from older
examples; inspect [API overview](../../api/overview.md) for the current
exports.

### Which models and providers are supported?

Support depends on the installed backend, model format, provider, and local
hardware. There is no fixed "300 models" guarantee. Query capabilities and
run a small model-specific smoke before planning production capacity.

### Why is inference slow?

Check the following in order:

1. Confirm the intended backend with `get_capabilities(detail=True)`.
2. Confirm that the model is already cached or locally available.
3. Use batching, an appropriate model size, and a supported precision.
4. Measure first-run download time separately from steady-state inference.
5. Check provider-specific logs before changing scheduler settings.

The [hardware guide](../hardware/overview.md) documents capability checks and
CUDA smoke tests.

### Why does a GPU not appear?

The installed framework wheel must match the driver and architecture. Check
the backend directly, then inspect the package capability report:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

If the check is positive but the accelerator report is not, compare the
environment used by the shell, service, and worker. Optional backends should
fail with a capability report rather than be assumed available.

## IPFS, P2P, and MCP

### Do I need a local IPFS daemon?

No. Core local inference does not require IPFS. IPFS, libp2p, and P2P task
queues are optional integrations and have their own binaries, credentials,
ports, and lifecycle requirements.

### How do I start MCP?

Install the MCP extra and use the product CLI:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

See [MCP setup](../MCP_SETUP_GUIDE.md) for direct module entry points,
capability inspection, and P2P notes.

### Is my data sent to a remote service?

Local execution stays local unless a remote model/provider, IPFS, P2P, or
other network integration is enabled. Review provider and deployment
configuration before processing sensitive material.

## CLI and supervisor

### Which CLI commands are current?

Run `ipfs-accelerate --help` for the installed command set. The current
top-level groups include `mcp`, `github`, `copilot`, `copilot-sdk`, `text`,
`audio`, `vision`, `multimodal`, `specialized`, and `models`. Older examples
using `inference`, `hardware`, `workflow`, or `network` as top-level groups
are not current product documentation.

### How do I run the agent supervisor?

The supervisor is an optional maintainer/operator surface, not required for
ordinary inference. Start with the
[Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md), which documents the
objective daemon, bundle supervisor, implementation supervisor, lifecycle
wrappers, evidence queries, and formal-assurance workflow.

## Browser support

Browser execution depends on the browser, runtime, and WebNN/WebGPU support
available on the target machine. Consult the
[WebNN/WebGPU feature guide](../../features/webnn-webgpu/WEBNN_WEBGPU_README.md)
and test the target browser rather than relying on a hard-coded compatibility
table.

## Testing and troubleshooting

Run the focused current checks from the
[testing guide](../../development/testing.md):

```bash
python -m pytest test/test_unified_cli_integration.py -q
python -m pytest test/api/test_agent_supervisor_objective_graph.py -q
```

When a command fails, capture the first traceback, the Python executable, the
installed package version, and the capability report. This is more useful than
retrying with unrelated optional extras.
