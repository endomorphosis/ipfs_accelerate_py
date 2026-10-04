# agent_supervisor.prompt

**Layer:** Mid · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Prompt workflow surfaces: directory scanning, goal planning hooks, plan admission, and bootstrap/rescue prompt pipelines.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control` contracts, planning/objectives as needed |
| **Typical dependents** | todo_daemon, runtime, control surfaces |

## Modules

| Module | Path |
| --- | --- |
| `prompt_directory_scanner` | `prompt/prompt_directory_scanner.py` |
| `prompt_goal_planner` | `prompt/prompt_goal_planner.py` |
| `intent_plan_coverage` | `prompt/intent_plan_coverage.py` |
| `plan_create_service` | `prompt/plan_create_service.py` |
| `prompt_plan_admission` | `prompt/prompt_plan_admission.py` |
| `prompt_workflow` | `prompt/prompt_workflow.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.prompt.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

`intent_plan_coverage` wraps the existing planner request with a source requirement
ledger and independently authored output, validation, and dependency groundings.
Its candidate coverage receipt binds the unchanged graph and canonical task CIDs.
The receipt compares declared obligations; source translation, exact validation
commands, execution authorization, and completion remain checked by their owners.
Rich compounds, native action/control context, and native statement roles other
than goals remain explicit unsupported execution scopes in this first bounded
integration.

Requirement contract version 2 additionally requires exact reviewed native
operation bindings and selects the existing symbolic planner. Version 1 keeps
the provider graph-plus-bindings route. `plan_create_service` snapshots complete
supplied semantic materials in version 2 identities, checks mutation before
cache reuse or persistence, and disables reuse for opaque live dependencies.
Request-only snapshot version 1 stays compatible.

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/prompt.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
