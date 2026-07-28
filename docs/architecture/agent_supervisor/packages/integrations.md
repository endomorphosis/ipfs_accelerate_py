# agent_supervisor.integrations

**Code:** `ipfs_accelerate_py/agent_supervisor/integrations/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/integrations/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md)


## Purpose

Optional adapters: LLM merge resolver fallback, Goose/Meta Spark runners, datasets analysis/logic providers, and other provider-backed bridges.

## When to use this package

You are wiring an external tool or model backend that must stay off the cold-import path of control/proof contracts.

## Public modules

| Module | Role |
| --- | --- |
| `llm_merge_resolver_fallback` | LLM-assisted merge repair fallback |
| `meta_spark_goose_runner` | Goose / Meta Spark runner |
| `grok_cli_runner` | Grok Build CLI runner |
| `ipfs_datasets_analysis_provider` | Datasets analysis provider |
| `ipfs_datasets_logic_provider` | Datasets logic provider |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.integrations import ...
# or
from ipfs_accelerate_py.agent_supervisor.integrations.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Explicit implementation/merge paths that opt into providers. |
| **Outbound** | External CLIs and optional packages; must not be imported by control cold paths. |
| **Forbidden** | Making control discovery import these modules. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.