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
| `prompt_plan_admission` | `prompt/prompt_plan_admission.py` |
| `prompt_workflow` | `prompt/prompt_workflow.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.prompt.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/prompt.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
