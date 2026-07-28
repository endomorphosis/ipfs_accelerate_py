# Domain package reference

Semantic descriptions of each `ipfs_accelerate_py.agent_supervisor` domain
package. Use these pages together with:

- [Package map](../PACKAGE_MAP.md) (DAG & placement)
- [Developer guide](../DEVELOPER_GUIDE.md) (how to extend)
- Code-tree `ipfs_accelerate_py/agent_supervisor/<package>/README.md` (module tables)

These pages describe **product ownership**. Board tickets that funded a module
are historical; they do not rename the package.

| Package | Layer | Doc |
| --- | --- | --- |
| `core/` | Foundation | [core.md](core.md) |
| `control/` | Foundation | [control.md](control.md) |
| `task_sources/` | Foundation | [task_sources.md](task_sources.md) |
| `context/` | Foundation | [context.md](context.md) |
| `prompt/` | Mid | [prompt.md](prompt.md) |
| `analysis/` | Mid | [analysis.md](analysis.md) |
| `proof/` | Mid | [proof.md](proof.md) |
| `objectives/` | Mid | [objectives.md](objectives.md) |
| `planning/` | Mid | [planning.md](planning.md) |
| `validation/` | Mid | [validation.md](validation.md) |
| `merge/` | Ops | [merge.md](merge.md) |
| `rescue/` | Ops | [rescue.md](rescue.md) |
| `runtime/` | Ops | [runtime.md](runtime.md) |
| `self_improvement/` | Ops | [self_improvement.md](self_improvement.md) |
| `integrations/` | Edge | [integrations.md](integrations.md) |
| `todo_daemon/` | Edge | [todo_daemon.md](todo_daemon.md) |

## Reading a package page

Each page should answer:

1. **Purpose** — what problem this package owns  
2. **Key modules** — primary entrypoints  
3. **Dependencies** — what it may import / who may import it  
4. **Extension tips** — where new code goes  

If a semantic page lags the code-tree README, trust the code-tree module list
and file a doc update in the same change that moves modules.
