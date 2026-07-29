# agent_supervisor.analysis

**Layer:** Mid · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Integrated analysis pipeline: AST index, cache, retrieval, consensus, and transport adapters used by proof and planning.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core` |
| **Typical dependents** | proof, planning, integrations |

## Modules

| Module | Path |
| --- | --- |
| `analysis_ast_index` | `analysis/analysis_ast_index.py` |
| `analysis_cache` | `analysis/analysis_cache.py` |
| `analysis_consensus` | `analysis/analysis_consensus.py` |
| `analysis_contracts` | `analysis/analysis_contracts.py` |
| `analysis_operation_registry` | `analysis/analysis_operation_registry.py` |
| `analysis_pipeline` | `analysis/analysis_pipeline.py` |
| `analysis_retrieval` | `analysis/analysis_retrieval.py` |
| `analysis_transport` | `analysis/analysis_transport.py` |
| `analyzer_health` | `analysis/analyzer_health.py` |
| `audit_scanner` | `analysis/audit_scanner.py` |
| `cache_coordinator` | `analysis/cache_coordinator.py` |
| `code_evidence_graph` | `analysis/code_evidence_graph.py` |
| `semantic_dependency_graph` | `analysis/semantic_dependency_graph.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.analysis.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/analysis.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
