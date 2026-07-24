# CLI

The supported product entry point is `ipfs-accelerate`. The module form is
useful when testing the checkout directly:

```bash
ipfs-accelerate --help
python -m ipfs_accelerate_py.cli --help
```

The package also installs `ipfs_accelerate`, backed by the separate
`ai_inference_cli.py` parser. It supports a different command surface; use
`ipfs_accelerate --help` for that entry point. The commands documented below
refer to the unified hyphenated CLI.

The parser currently exposes these top-level groups:

| Group | Purpose |
| --- | --- |
| `mcp` | Start, inspect, or stop at the MCP service boundary. |
| `github` | GitHub integration operations. |
| `copilot` | GitHub Copilot CLI operations. |
| `copilot-sdk` | GitHub Copilot SDK operations. |
| `text` | Text generation, classification, and embeddings when providers are installed. |
| `audio` | Audio processing when the required provider is installed. |
| `vision` | Vision processing when the required provider is installed. |
| `multimodal` | Multimodal processing when the required provider is installed. |
| `specialized` | Specialized model tasks. |
| `models` | Model listing, search, details, and IPLD/IPFS model records. |

Top-level flags are:

```bash
ipfs-accelerate --help
ipfs-accelerate --debug --help
ipfs-accelerate --output-json models list
```

## MCP commands

```bash
ipfs-accelerate mcp --help
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
ipfs-accelerate mcp dashboard --help
```

Keep development servers on localhost. Authentication, TLS, firewall rules,
and process supervision belong to the deployment environment. See the
[MCP setup guide](../MCP_SETUP_GUIDE.md).

## Model commands

```bash
ipfs-accelerate models --help
ipfs-accelerate models list
ipfs-accelerate models search "embedding"
ipfs-accelerate models details --help
ipfs-accelerate models ipld-document --help
ipfs-accelerate models ipld-cid --help
ipfs-accelerate models ipld-publish --help
ipfs-accelerate models ipld-load --help
```

IPLD publish/load operations require the relevant IPFS dependencies and
service. A model being listed does not mean that its provider or weights are
installed locally.

## AI processing commands

The text, audio, vision, multimodal, and specialized groups are provider-
dependent. Ask the installed parser for their detailed help:

```bash
ipfs-accelerate text --ai-help
ipfs-accelerate audio --help
ipfs-accelerate vision --help
ipfs-accelerate multimodal --help
ipfs-accelerate specialized --help
```

The command may report an unavailable optional provider rather than installing
one implicitly. Use `get_capabilities(detail=True)` to inspect the same runtime
boundary from Python.

## GitHub and Copilot groups

These groups are optional integrations with their own credentials and SDK
versions:

```bash
ipfs-accelerate github --help
ipfs-accelerate copilot --help
ipfs-accelerate copilot-sdk --help
```

Do not place tokens in shell history or checked-in configuration. The GitHub
guides describe repository-specific workflows; they are not prerequisites for
local model inference.

## Output and diagnostics

Use `--output-json` before the command group when a command supports structured
output. Use `--debug` to increase diagnostic logging. Capture the complete
command, first traceback, Python executable, package version, and capability
report when filing a failure.

## What is not a current CLI command

The current parser does not register generic top-level `inference`, `hardware`,
`workflow`, `network`, `queue`, or `p2p` groups. Some older help text and
historical guides mention them; use the current parser help and the
[documentation index](../../INDEX.md) as the source of truth.

## Related documentation

- [Quick start](../QUICKSTART.md)
- [API overview](../../api/overview.md)
- [Hardware guide](../hardware/overview.md)
- [MCP setup](../MCP_SETUP_GUIDE.md)
- [Testing](../../development/testing.md)
