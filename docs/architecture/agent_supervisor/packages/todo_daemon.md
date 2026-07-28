# agent_supervisor.todo_daemon

## Purpose

Executable implementation and supervisor process loops: worktree management, git helpers, proposal/validation engine pieces, and CLI entrypoints that drain Markdown boards.

## When to use this package

You are changing how a lane implements a task in a worktree, not the abstract control operation vocabulary.

## Public modules

| Module | Role |
| --- | --- |
| `implementation_daemon` | Drain a taskboard in a worktree |
| `implementation_supervisor` | Watchdog / recover implementation daemon |
| `engine` | Task/proposal/validation mechanics |
| `worktrees` | Worktree pool operations |
| `git_utils` | Git helpers |
| `auto_commit` | Auto-commit policies |
| `cli / runner / app` | Composition entrypoints |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.todo_daemon import ...
# or
from ipfs_accelerate_py.agent_supervisor.todo_daemon.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Runtime multi-supervisor; operator scripts. |
| **Outbound** | Most domain packages as needed; keep provider selection via env/integrations. |
| **Forbidden** | Becoming the public control API (use `control` instead). |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.
