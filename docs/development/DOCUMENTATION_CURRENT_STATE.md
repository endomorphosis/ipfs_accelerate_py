# Documentation Current State

This page records which documents are normative for the checked-out code and
how to audit documentation drift. It is intentionally short; the source code
and executable help remain authoritative for behavior.

## Sources of truth

| Surface | Source of truth | Current documentation |
| --- | --- | --- |
| Package metadata and extras | `pyproject.toml`, `setup.py` | [Installation](../guides/getting-started/installation.md) |
| Python exports | `ipfs_accelerate_py/__init__.py`, `ipfs_accelerate_py/ipfs_accelerate.py` | [API overview](../api/overview.md) |
| Unified CLI | `ipfs_accelerate_py/cli.py`, `cli_entry.py` | [CLI guide](../guides/cli/README_CLI.md) |
| Direct AI CLI | `ipfs_accelerate_py/ai_inference_cli.py` | Inspect `ipfs_accelerate --help` |
| Canonical MCP runtime | `ipfs_accelerate_py/mcp_server/` | [MCP setup](../guides/MCP_SETUP_GUIDE.md) |
| HF model server | `ipfs_accelerate_py/hf_model_server/` | [HF model server](../features/hf-model-server/README.md) |
| Agent supervisor | `ipfs_accelerate_py/agent_supervisor/` | [Supervisor guide](../guides/AGENT_SUPERVISOR_GUIDE.md) |
| Test contracts | `test/`, `test/api/`, optional integration suites | [Testing](testing.md) |

The hyphenated `ipfs-accelerate` command and underscore `ipfs_accelerate`
command are separate installed scripts. Their parsers and capabilities are not
interchangeable. Use the command's own `--help` output when working with the
underscore entry point.

## Maintained documentation

The current navigation starts at:

- [Root README](../../README.md)
- [Documentation index](../INDEX.md)
- [Documentation orientation](../README.md)
- [API overview](../api/overview.md)
- [Architecture overview](../architecture/overview.md)
- [Testing guide](testing.md)
- [Agent Supervisor Guide](../guides/AGENT_SUPERVISOR_GUIDE.md)

These pages should describe the current package boundaries, supported commands,
optional dependency behavior, and reproducible validation commands. When a
maintained page and a historical report disagree, the maintained page and live
code win.

## Historical and compatibility records

The following locations contain useful context but are not current API
contracts:

- `docs/archive/` and `docs/development_history/`;
- `docs/summaries/`, `docs/project/status/`, `docs/project/dashboard/`, and
  dated phase/implementation summaries;
- files whose names contain `summary`, `history`, `complete`, `final`,
  `phase`, `plan`, `review`, or `todo` unless they are linked as a maintained
  architecture reference; and
- compatibility paths that explicitly point to a maintained replacement.

Historical records should not be deleted merely because their point-in-time
claims are old. They should be labeled when a reader could mistake them for a
current command, API, benchmark, or deployment guarantee.

## Review checklist

Before merging a documentation change:

1. Confirm imports and public names in the live module.
2. Confirm CLI flags with `--help`; do not infer a command from an old example.
3. Confirm package extras and console scripts in `pyproject.toml`.
4. Confirm relative links from the document's actual directory.
5. Use capability language for optional hardware, providers, IPFS, P2P, and
   formal provers.
6. Avoid fixed model counts, benchmark numbers, or test totals without a date,
   commit, hardware, and reproducible command.
7. Run the smallest relevant deterministic test.

Useful local checks are:

```bash
git diff --check
ipfs-accelerate --help
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon list
python -m pytest test/test_unified_cli_integration.py -q
```

## Known optionality

The base package is deliberately defensive. CUDA, Transformers, MCP, IPFS,
libp2p, browser runtimes, external LLMs, and theorem provers may be absent or
unhealthy even when `import ipfs_accelerate_py` succeeds. Documentation should
show how to discover and report that state instead of claiming universal
availability.
