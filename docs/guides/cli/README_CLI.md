# CLI

**Status:** Current
**Owner:** CLI and package maintainers
**Audience:** Users, operators, integrators, and maintainer agents
**Sources:** `pyproject.toml`; `ipfs_accelerate_py/cli_entry.py`;
`ipfs_accelerate_py/cli.py`;
`ipfs_accelerate_py/agent_supervisor/control/control_cli.py`; CLI contract tests
under `test/`
**Last-verified:** 2026-08-03 @ 2bf2cebd3 (CLI epilog/docstring revalidated after help-text fix)

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
commands share the `OperationRequest` / `OperationResult` contract. A submitted
operation, including read-oriented discovery, requires complete target
bindings: repository root/state root, repository/tree/objective/policy IDs and
revisions, and caller. Mutations additionally require direct authorization,
idempotency, lease, fencing, and effects.

```bash
ipfs-accelerate agent --help
ipfs-accelerate agent capabilities --help
ipfs-accelerate agent status --help
ipfs-accelerate agent health --help
ipfs-accelerate agent goals --help
ipfs-accelerate agent tasks --help
```

Use `--help` for cold command discovery. Submit actual operations with a
complete request file (or every required binding flag):

```bash
ipfs-accelerate agent capabilities --help
ipfs-accelerate agent status --request-file complete-status-request.json --watch-count 5
ipfs-accelerate agent pause --request-file authorized-pause.json --output-json
```

The bare command `ipfs-accelerate agent capabilities --output-json` is invalid:
omitting the target bindings yields a typed `invalid_request` result.

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
ipfs-accelerate --output-json models list
ipfs-accelerate --output-json models search "embedding"
ipfs-accelerate models details --help
ipfs-accelerate models ipld-document --help
ipfs-accelerate models ipld-cid --help
ipfs-accelerate models ipld-publish --help
ipfs-accelerate models ipld-load --help
```

`--output-json` is a global flag and must appear before `models`. Listing and
search are runtime operations, not cold discovery: they can initialize
`SharedCore` and IPFS storage, contact configured endpoints, attempt optional
provider/package setup or auto-install flows, and write configuration or cache
state. Use `models --help` when those effects are not acceptable. IPLD
publish/load operations require the relevant IPFS dependencies and service. A
listed or matched model does not mean that its provider, credentials, service,
or weights are available locally.

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

The command may report an unavailable optional provider. The Python
`get_instance().get_capabilities(detail=True)` path inspects the same runtime
boundary but constructs a side-effecting coordinator; it is not a cold probe.
See the [API overview](../../api/overview.md).

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

## Codex owner observation during an MCP outage

The separate Codex module adapter supports `read-owner` and
`read-owner-fallback`; these are module commands, not unified CLI groups.
Automation can call `read_codex_owner_after_mcp_failure` from
`ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration` after its
current MCP status call has ended with a transport-send failure. The adapter
accepts only failures of `codex_tui/read_thread` or `codex_tui/wait_threads`
against a loopback HTTP `/mcp` bridge. Supply the original error unchanged, the
explicit existing owned Unix socket, and the exact expected thread and turn:

```bash
python -m ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration \
  read-owner-fallback --socket-path "$CODEX_EXISTING_SOCKET" \
  --thread-id "$OWNER_THREAD_ID" --expected-turn-id "$OWNER_TURN_ID" \
  --mcp-method read_thread --transport-error "$MCP_STATUS_ERROR" \
  --timeout-seconds 5 --max-response-bytes 262144
```

These values come from the caller's configured endpoint and existing owner
binding. The adapter does not discover or start a daemon. It performs one
read-only observation through the existing Unix reader, retaining the original
error and SHA256. Caller-supplied error text is not authenticated provenance.
Policy refusals and application errors do not authorize this fallback.

The JSON schema is `codex-owner-transport-fallback/v1`. Exit 4 means the exact
owner turn was observed in progress, 0 means it is terminal and its saved
result needs reconciliation, 3 means a different turn was observed, 1 means
the observation is unavailable, and 2 means the request was refused. A changed
turn retains the expected handle for explicit reconciliation. An unavailable
observation leaves owner state unknown. Neither outcome authorizes restarting
work. No task message, execution, completion, custody, claim, goal change or
budget reset follows from a status observation.

The error is limited to 8192 UTF8 bytes. One Unix read shares a timeout of at
most 60 seconds and a configured response limit of 4096–4194304 bytes; client
connection close can add at most one second. CLI output is limited to 65536
UTF8 bytes including the newline. Output failure does not retry the read.

Use an interpreter with the existing `websockets.sync.client` dependency.
Python isolated mode (`-I`) excludes user-site packages, so it can make an
otherwise installed client unavailable. This is reported as
`read_client_dependency_unavailable` / `owner_client_dependency_unavailable`;
the adapter never installs packages or changes interpreter isolation. Select
an already configured compatible environment. The module uses the existing
package import path; this adapter is not a package-wide side-effect audit.

## Output and diagnostics

Use `--output-json` before the command group when a command supports structured
output. Use `--debug` to increase diagnostic logging. Capture the complete
command, first traceback, Python executable, package version, and capability
report when filing a failure.

## What is not a current CLI command

The current parser does **not** register top-level `inference`, `hardware`,
`workflow`, `network`, `queue`, or `p2p` groups. Invoking them fails with an
invalid-choice error listing only the groups in the table above.

Historical guides and completion reports may still mention examples such as:

```text
ipfs-accelerate inference generate --prompt "Hello world"
ipfs-accelerate queue status
ipfs-accelerate network status
```

Those top-level groups are **not** registered. Prefer:

- AI groups (`text`, `audio`, `vision`, …) or the Python routers for inference-like work
- `mcp` for service/dashboard operation (including optional queue/P2P hosting behind MCP flags and env)
- Python APIs and architecture guides for hardware, workflow, network, and P2P details

The live source of truth is `ipfs-accelerate --help` (or
`python -m ipfs_accelerate_py.cli --help`) plus the
[documentation index](../../INDEX.md). The unified CLI module docstring and
parser epilog were aligned to registered groups on the post-DOC-028 revalidation
pass so `--help` examples no longer advertise the removed top-level names.

## Related documentation

- [Quick start](../QUICKSTART.md)
- [API overview](../../api/overview.md)
- [Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md)
- [Hardware guide](../hardware/overview.md)
- [MCP setup](../MCP_SETUP_GUIDE.md)
- [Testing](../../development/testing.md)
