# CLI

The supported product entry point is `ipfs-accelerate`. The module form is
useful when testing the checkout directly or when the console script is not on
`PATH`:

```bash
ipfs-accelerate --help
python -m ipfs_accelerate_py.cli --help
```

The package also installs `ipfs_accelerate`, backed by the separate
`ai_inference_cli.py` parser. It supports a different command surface; use
`ipfs_accelerate --help` for that entry point. The commands documented below
refer to the unified hyphenated CLI only.

## Registered top-level groups

The parser currently registers these top-level groups (from the live
`choices=` set):

| Group | Purpose |
| --- | --- |
| `agent` | Inspect and control the agent supervisor through typed contracts. |
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

Always prefer `ipfs-accelerate <group> --help` over copied historical examples.
The parser epilog may still show obsolete sample lines for `inference`,
`queue`, or `network`; those are **not** registered groups (see
[What is not a current CLI command](#what-is-not-a-current-cli-command)).

## Agent supervisor commands

The `agent` group is the preferred operator surface for the supervisor. All
commands share the `OperationRequest` / `OperationResult` contract. Real
mutations require a complete request or direct authorization, idempotency,
lease, fencing, and effects.

```bash
ipfs-accelerate agent --help
ipfs-accelerate agent capabilities --help
ipfs-accelerate agent status --help
ipfs-accelerate agent health --help
ipfs-accelerate agent goals --help
ipfs-accelerate agent tasks --help
```

Typical discovery-style invocations (read-oriented when authorization and
mutation fields are omitted):

```bash
ipfs-accelerate agent capabilities --output-json
ipfs-accelerate agent status --request-file request.json --watch-count 5
ipfs-accelerate agent pause --request-file authorized-pause.json --output-json
```

Low-level `ipfs-accelerate-agent-*` console scripts remain available for
daemons and recovery engines. They are not substitutes for the typed
`ipfs-accelerate agent` control API. See the
[Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md).

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
boundary from Python. See the [API overview](../../api/overview.md).

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

The current parser does **not** register top-level `inference`, `hardware`,
`workflow`, `network`, `queue`, or `p2p` groups. Invoking them fails with an
invalid-choice error listing only the groups in the table above.

Some older guides and the CLI epilog still mention examples such as:

```text
ipfs-accelerate inference generate --prompt "Hello world"
ipfs-accelerate queue status
ipfs-accelerate network status
```

Treat those as stale help text, not as supported entry points. Use:

- AI groups (`text`, `audio`, `vision`, …) or the Python routers for inference-like work
- `mcp` for service/dashboard operation (including optional queue/P2P hosting behind MCP flags and env)
- Python APIs and architecture guides for hardware, workflow, network, and P2P details

The live source of truth is `ipfs-accelerate --help` (or
`python -m ipfs_accelerate_py.cli --help`) plus the
[documentation index](../../INDEX.md).

## Related documentation

- [Quick start](../QUICKSTART.md)
- [API overview](../../api/overview.md)
- [Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md)
- [Hardware guide](../hardware/overview.md)
- [MCP setup](../MCP_SETUP_GUIDE.md)
- [Testing](../../development/testing.md)
