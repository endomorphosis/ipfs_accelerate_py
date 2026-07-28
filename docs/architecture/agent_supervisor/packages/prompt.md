# agent_supervisor.prompt

## Purpose

Prompt workflow: directory scanning, goal planning from prompts, plan admission, bootstrap/rescue rollout and benchmarks.

## When to use this package

You are changing how free-form prompts become admitted tasks or how rescue prompts are gated.

## Public modules

| Module | Role |
| --- | --- |
| `prompt_workflow` | Bootstrap / rescue workflow |
| `prompt_directory_scanner` | Scan prompt directories |
| `prompt_goal_planner` | Plan goals from prompts |
| `prompt_plan_admission` | Admit or reject planned work |
| `prompt_workflow_benchmark` | Parity / adversarial benchmarks |
| `prompt_workflow_rollout` | Rollout modes and promotion gates |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.prompt import ...
# or
from ipfs_accelerate_py.agent_supervisor.prompt.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Operators and control surfaces that bootstrap work from prompts. |
| **Outbound** | `context`, `task_sources`, validation helpers. |
| **Forbidden** | Bypassing admission to write boards or mutate repos directly. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.
