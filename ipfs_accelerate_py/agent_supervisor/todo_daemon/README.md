# agent_supervisor.todo_daemon

**Layer:** Edge · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Executable implementation and supervisor daemons, git/worktree helpers, and board-drain loops.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | most domain packages it orchestrates (top of DAG) |
| **Typical dependents** | ops scripts, multi-supervisor entrypoints |

## Modules

| Module | Path |
| --- | --- |
| `app` | `todo_daemon/app.py` |
| `artifacts` | `todo_daemon/artifacts.py` |
| `auto_commit` | `todo_daemon/auto_commit.py` |
| `cli` | `todo_daemon/cli.py` |
| `context` | `todo_daemon/context.py` |
| `core` | `todo_daemon/core.py` |
| `deterministic_fallback` | `todo_daemon/deterministic_fallback.py` |
| `diagnostics` | `todo_daemon/diagnostics.py` |
| `engine` | `todo_daemon/engine.py` |
| `file_replacement` | `todo_daemon/file_replacement.py` |
| `git_utils` | `todo_daemon/git_utils.py` |
| `history` | `todo_daemon/history.py` |
| `implementation_daemon` | `todo_daemon/implementation_daemon.py` |
| `implementation_daemon_runner` | `todo_daemon/implementation_daemon_runner.py` |
| `implementation_supervisor` | `todo_daemon/implementation_supervisor.py` |
| `implementation_supervisor_runner` | `todo_daemon/implementation_supervisor_runner.py` |
| `legal_parser` | `todo_daemon/legal_parser.py` |
| `legal_parser_daemon` | `todo_daemon/legal_parser_daemon.py` |
| `lifecycle_wrapper` | `todo_daemon/lifecycle_wrapper.py` |
| `llm` | `todo_daemon/llm.py` |
| `llm_defaults` | `todo_daemon/llm_defaults.py` |
| `logic_port` | `todo_daemon/logic_port.py` |
| `plans` | `todo_daemon/plans.py` |
| `registry` | `todo_daemon/registry.py` |
| `runner` | `todo_daemon/runner.py` |
| `specs` | `todo_daemon/specs.py` |
| `status` | `todo_daemon/status.py` |
| `supervisor` | `todo_daemon/supervisor.py` |
| `supervisor_loop` | `todo_daemon/supervisor_loop.py` |
| `supervisor_runtime` | `todo_daemon/supervisor_runtime.py` |
| `task_board` | `todo_daemon/task_board.py` |
| `typescript` | `todo_daemon/typescript.py` |
| `worktrees` | `todo_daemon/worktrees.py` |
| `wrapper` | `todo_daemon/wrapper.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.todo_daemon.<module> import ...
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
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/todo_daemon.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
