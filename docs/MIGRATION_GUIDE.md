# Migration Guide

This guide maps older IPFS Accelerate examples to the current package
boundaries. The executable help and `pyproject.toml` remain authoritative for
commands, extras, and console scripts.

## CLI entry points

The installed scripts have different roles:

- `ipfs-accelerate` runs the current unified CLI from `cli_entry.py`;
- `ipfs_accelerate` runs the separate AI inference parser from
  `ai_inference_cli.py`; and
- `python -m ipfs_accelerate_py.cli` is useful for testing the unified parser
  from a checkout.

Do not mix command groups between these parsers:

```bash
ipfs-accelerate --help
ipfs_accelerate --help
python -m ipfs_accelerate_py.cli --help
```

The current unified groups include `mcp`, `github`, `copilot`, `copilot-sdk`,
`text`, `audio`, `vision`, `multimodal`, `specialized`, and `models`. Older
examples using `inference`, `hardware`, `workflow`, `network`, `queue`, or a
top-level `docker` group are not current unified-CLI commands.

## Common migrations

| Older pattern | Current surface |
| --- | --- |
| `ipfs-accelerate inference ...` | `ipfs-accelerate text --ai-help`, the direct AI CLI, or the Python API. |
| `ipfs-accelerate github list-repos` | `ipfs-accelerate github repos --owner OWNER --limit 10`. |
| `ipfs-accelerate hardware info` | `HardwareKit().get_hardware_info(include_detailed=True)` or `get_instance().get_capabilities(detail=True)`. |
| `ipfs-accelerate docker ...` | `DockerKit`/`docker_executor` from Python or the registered MCP Docker tools. |
| `ipfs-accelerate runner ...` | `github autoscaler`, the GitHub runner kit, or the agent-supervisor tools, depending on the workload. |
| `ipfs_accelerate_py.mcp` as the server | `ipfs_accelerate_py.mcp_server` is canonical; `mcp` is a compatibility facade. |
| `IPFSAccelerator` | `ipfs_accelerate_py`, `get_instance()`, or the documented router/helper export. |

## Python kit modules

The kit modules remain useful for typed, direct operations:

```python
from ipfs_accelerate_py import get_instance
from ipfs_accelerate_py.kit.github_kit import GitHubKit
from ipfs_accelerate_py.kit.hardware_kit import HardwareKit

accelerator = get_instance()
print(accelerator.get_capabilities(detail=True))
print(GitHubKit().list_repos(limit=10))
print(HardwareKit().get_hardware_info(include_detailed=True))
```

For Docker, prefer the core `DockerExecutor` or the MCP wrappers when the
Docker daemon is deliberately enabled. For IPFS, use the backend router and
check which backend was selected before treating a CID as network-persistent.

## MCP and supervisor migration

Start the canonical MCP service through the product CLI:

```bash
python -m pip install -e ".[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

The agent supervisor is a separate maintainer/operator control plane. Its
objective graph, evidence artifacts, leases, implementation lanes, and proof
receipts are not prerequisites for ordinary inference. Start with the
[supervisor guide](guides/AGENT_SUPERVISOR_GUIDE.md) and the
[current-state documentation audit](development/DOCUMENTATION_CURRENT_STATE.md).

## Compatibility policy

Compatibility facades are retained where practical, but they are not a reason
to copy old command examples into new integrations. When migrating a script:

1. identify the installed script or Python module it actually invokes;
2. inspect its `--help` or public signature;
3. install only the matching optional extra;
4. run a capability or deterministic contract check; and
5. keep external credentials, network services, and model/provider output
   outside the authoritative application state until validated.

See [CLI guide](guides/cli/README_CLI.md), [API overview](api/overview.md),
[installation](guides/getting-started/installation.md), and
[architecture overview](architecture/overview.md).
