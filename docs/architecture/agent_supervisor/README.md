# Agent Supervisor documentation hub

This hub is the product-facing entry point for `ipfs_accelerate_py.agent_supervisor`.

The supervisor is a **proof- and policy-bounded control plane** for objective-driven
software work. Models propose plans and edits; deterministic validation, leases,
allowlists, and typed evidence decide whether work may advance.

## Read by audience

| Audience | Start here | Then |
| --- | --- | --- |
| New reader / design | [Design philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) | [Package map](PACKAGE_MAP.md) |
| Operator / user | [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) | Philosophy (mental model) |
| Contributor | [Contributor guide](FOR_CONTRIBUTORS.md) | [Package map](PACKAGE_MAP.md), domain package READMEs |
| Implementation agent | [Agent capsule](FOR_AGENTS.md) | Philosophy fail-closed sections |
| Deep implementation | [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md) | Domain package READMEs under the code tree |
| Program / board audit | [Program glossary](PROGRAMS.md) | Sealed boards under `docs/architecture/*.{todo,objectives}.md` |

## Two namespaces (important)

| Namespace | Meaning | Examples |
| --- | --- | --- |
| **Product / domain** | What the system *is* and how to use it | `proof/`, control plane, codebase-proof pipeline |
| **Program / board** | How work was *scheduled and evidenced* | task headers like `## <PREFIX>-###` on boards |

Primary documentation teaches the **product** namespace. Board IDs stay in taskboards,
objective heaps, and optional “Program evidence” footers—not in API names or intro prose.

## Package layout (domain map)

On current `main`, code lives in domain packages (not a flat module warehouse):

- `core/`, `control/`, `task_sources/`, `context/`, `prompt/`
- `analysis/`, `proof/`, `objectives/`, `planning/`
- `validation/`, `merge/`, `rescue/`, `runtime/`
- `self_improvement/`, `integrations/`, `todo_daemon/`

See [PACKAGE_MAP.md](PACKAGE_MAP.md) and the code tree README at
`ipfs_accelerate_py/agent_supervisor/README.md` (when present on your branch).

## Programs layered on the control plane

Self-improvement, codebase-proof, domain layout, AI service catalog, and related
efforts are **programs** that use the supervisor—they are not alternate supervisors.
Map board prefixes to semantic names in [PROGRAMS.md](PROGRAMS.md).

## Domain package reference pages

Per-package semantic READMEs (purpose, modules, dependency rules):

- [packages/README.md](packages/README.md)

## Inventory


Machine-generated prefix inventories used during the semantic refresh live under
[`_inventory/`](_inventory/).
