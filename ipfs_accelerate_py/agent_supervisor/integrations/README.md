# agent_supervisor.integrations

**Layer:** Edge · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Optional external bridges: dataset logic providers, LLM merge fallback, Goose/meta runners. Lazy; not required for cold import.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | domain packages it bridges; remain optional |
| **Typical dependents** | todo_daemon, control fallbacks |

## Modules

| Module | Path |
| --- | --- |
| `ipfs_datasets_analysis_provider` | `integrations/ipfs_datasets_analysis_provider.py` |
| `ipfs_datasets_logic_provider` | `integrations/ipfs_datasets_logic_provider.py` |
| `llm_merge_resolver_fallback` | `integrations/llm_merge_resolver_fallback.py` |
| `meta_spark_goose_runner` | `integrations/meta_spark_goose_runner.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.integrations.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/integrations.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
