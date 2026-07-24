# Best Practices

These practices apply to the current Python, CLI, MCP, optional P2P, and agent
supervisor surfaces. They favor runtime capability checks and bounded evidence
over static claims about hardware or model support.

## Discover before selecting

Start with the package capability report:

```python
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print(report)
```

The report describes the current process environment. It does not guarantee
that a model can load, that a remote provider is authenticated, or that a
service is reachable. Run a small model-specific smoke after discovery.

For CUDA, verify the framework build and an actual operation separately from
`nvidia-smi`:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

## Use stable package boundaries

Use `ipfs_accelerate_py`/`get_instance()` for the main Python API and
`ipfs_accelerate_py.mcp_server` for the canonical MCP runtime. Keep optional
imports behind capability checks. The `ipfs_accelerate_py.mcp` package remains
for compatibility; new MCP integrations should target `mcp_server`.

The hyphenated `ipfs-accelerate` script is the current unified CLI. The
underscore `ipfs_accelerate` script is a separate AI-inference CLI with a
different parser. Always inspect the relevant `--help` output.

## Prefer typed kit modules over shell calls

The kit modules provide typed result objects and centralized error handling:

```python
from ipfs_accelerate_py.kit.docker_kit import run_container
from ipfs_accelerate_py.kit.github_kit import GitHubKit
from ipfs_accelerate_py.kit.hardware_kit import HardwareKit

docker_result = run_container(
    image="python:3.12",
    command=["python", "-c", "print('ok')"],
    network="none",
    memory="512m",
    timeout=60,
)
github_result = GitHubKit().list_repos(limit=10)
hardware = HardwareKit().get_hardware_info(include_detailed=True)
```

Check `success` and `error` before consuming result data. External commands,
credentials, network access, and Docker availability are all runtime inputs.

## Bound resources and payloads

Set timeouts, memory/CPU limits, queue bounds, retry budgets, and cache limits
for Docker, model serving, P2P, and supervisor work. Keep large decoded text,
logs, and nested artifact graphs out of persistent evidence; store a bounded
summary plus a content-addressed artifact reference instead.

For parallel work, measure the actual bottleneck. Additional workers can
duplicate model memory, contend for one GPU, overload a provider, or increase
checkpoint/persistence cost.

## MCP and P2P

Start MCP locally during development:

```bash
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Enable P2P only with the `mcp-p2p` or `libp2p` extra and an explicit network
configuration. Keep identities, queue paths, ports, and authentication out of
source control. The P2P guide documents discovery and degraded operation.

## Model manager and serving

The model manager supports JSON or DuckDB metadata storage, optional IPFS
artifacts, model search, serving records, and cache-aware loading. Use an
isolated storage path in tests:

```python
from ipfs_accelerate_py.model_manager import ModelManager

with ModelManager(storage_path="/tmp/models.json", use_database=False) as manager:
    print(manager.list_models())
```

The optional HF model server exposes health/readiness, OpenAI-compatible
completion/chat/embedding routes, model management, and metrics. Configure
authentication and rate limiting before remote exposure.

## Supervisor and formal assurance

LLMs and external providers produce proposals. Deterministic parsers,
validators, capability probes, lease contracts, and authoritative proof
receipts decide admission and completion. Keep retries bounded and make every
repair explainable through an artifact or receipt.

Use the [Agent Supervisor Guide](guides/AGENT_SUPERVISOR_GUIDE.md) for the
operational workflow and the [formal verification plan](architecture/AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md)
for assurance boundaries.

## Tests and documentation

Run the smallest deterministic test that exercises the change, then expand to
optional integration tests:

```bash
python -m pytest test/test_unified_cli_integration.py -q
python -m pytest test/api/test_agent_supervisor_objective_graph.py -q
```

Document public changes in the maintained guides. Historical summaries and
benchmark reports must include their commit, environment, and date if they
remain useful.

## Related documentation

- [Current documentation state](development/DOCUMENTATION_CURRENT_STATE.md)
- [API overview](api/overview.md)
- [Architecture overview](architecture/overview.md)
- [Installation](guides/getting-started/installation.md)
- [Hardware guide](guides/hardware/overview.md)
- [MCP setup](guides/MCP_SETUP_GUIDE.md)
