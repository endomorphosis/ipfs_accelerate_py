# Examples

The examples directory contains runnable demonstrations of the optional model,
MCP, router, Docker, and model-management integrations. Each example may have
different dependencies; inspect its imports before installing a large profile.

## Examples available in this checkout

| Example | Focus |
| --- | --- |
| `ai_implementation_showcase.py` | Broad AI/provider integration. |
| `ai_mcp_demo.py` | MCP service and AI integration. |
| `ai_model_discovery_example.py` | Model discovery and search. |
| `comprehensive_ai_demo.py` | Combined model operations. |
| `demonstration_example.py` | Basic package demonstration. |
| `model_manager_example.py` | Model manager operations. |
| `sdk_demo.py` | SDK-oriented integration example. |
| `docker_execution_examples.py` | Docker executor and MCP wrappers. |
| `auto_healing_demo.py` | Opt-in error-handler behavior. |
| `llm_router_example.py` | LLM provider selection and caching. |
| `kitchen_sink_demo.py` | Web-based testing/demo surface. |

The file names above are the source of truth for this checkout. A name that
appears in an older report but is absent here should not be copied into a new
runbook.

## Install and run

From the repository root:

```bash
python -m pip install -e ".[dev]"
python examples/demonstration_example.py
python examples/llm_router_example.py
```

For examples that need model providers or MCP:

```bash
python -m pip install -e ".[full]"
python -m pip install -e ".[mcp]"
python examples/ai_mcp_demo.py
```

Do not install every extra by default. CUDA, browser, IPFS, P2P, Docker,
external LLMs, and model downloads are separate runtime capabilities.

## CLI relationship

There are two installed CLI scripts:

```bash
ipfs-accelerate --help
ipfs_accelerate --help
```

`ipfs-accelerate` is the unified MCP/GitHub/model-management CLI. The
underscore `ipfs_accelerate` script is the separate AI inference parser and
contains provider-dependent `text`, `audio`, `vision`, `system`, and related
groups. Use the parser's own help output rather than mixing examples between
them.

## Testing examples

Examples are not a substitute for contract tests. Run the focused deterministic
suite first:

```bash
python -m pytest test/test_unified_cli_integration.py -q
python -m pytest test/test_llm_router_integration.py -q
```

Feature-specific examples may need credentials, a local service, a Docker
daemon, or model files. Keep those resources isolated and avoid putting
secrets in example source or shell history.

## Related documentation

- [Documentation index](../docs/INDEX.md)
- [Getting started](../docs/guides/getting-started/README.md)
- [CLI guide](../docs/guides/cli/README_CLI.md)
- [MCP setup](../docs/guides/MCP_SETUP_GUIDE.md)
- [Testing](../docs/development/testing.md)
- [Docker execution](../docs/DOCKER_EXECUTION.md)
