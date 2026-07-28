# Agent Supervisor developer guide

Practical guide for engineers and agents who **build on or inside**
`ipfs_accelerate_py.agent_supervisor`.

| Role | Prefer |
| --- | --- |
| New developer | This guide → [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) → [Package map](PACKAGE_MAP.md) |
| Operator | [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) |
| Deep design | [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md) |
| Implementation agent | [FOR_AGENTS.md](FOR_AGENTS.md) |

Package entry README: [`ipfs_accelerate_py/agent_supervisor/README.md`](../../../ipfs_accelerate_py/agent_supervisor/README.md).

---

## 1. Mental model (five minutes)

The supervisor is a **constrained feedback controller**:

1. **Intent** — objective heap (durable goals, evidence expectations).
2. **Projection** — taskboard (drainable tasks with stable `## PREFIX-###` IDs).
3. **Actuation** — implementation agent in an isolated worktree.
4. **Measurement** — validation commands, typed receipts, event log.
5. **Admission** — control plane / policy decides progress, merge, or rescue.

Models never complete work by eloquence alone. Completion is a **policy +
evidence** decision.

Read the seven pillars in [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) before
changing authority, evidence classes, or isolation behavior.

---

## 2. Repository orientation

```text
ipfs_accelerate_py/agent_supervisor/
  core/ control/ task_sources/ context/ prompt/
  analysis/ proof/ objectives/ planning/ validation/
  merge/ rescue/ runtime/ self_improvement/
  integrations/ todo_daemon/
  __init__.py          # reviewed public re-exports + layout inventories
  README.md            # package developer entry

docs/architecture/agent_supervisor/
  DEVELOPER_GUIDE.md   # this file
  PACKAGE_MAP.md
  packages/*.md        # semantic package pages
  programs/            # long-running program indexes
  LAYOUT_CUTOVER_EVIDENCE.md  # historical layout cutover only

docs/guides/AGENT_SUPERVISOR_GUIDE.md   # operator / CLI / MCP
docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md
test/api/test_agent_supervisor_*.py
scripts/ops/agent_supervisor/           # launch & ops entrypoints
```

**Two namespaces:**

| Namespace | Meaning | Examples |
| --- | --- | --- |
| Product / domain | What the system is | `proof/`, control operations |
| Program / board | How work was scheduled | `## ASI-170`, `## ASREF-G020` |

Do not put board prefixes into public API names. Boards record *work*; packages
record *ownership*.

---

## 3. Public API and imports

### Prefer domain imports

```python
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    SupervisorControlService,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    Operation,
    OperationRequest,
    OperationResult,
)
```

### Package-root inventories (semantic)

```python
from ipfs_accelerate_py.agent_supervisor import (
    AGENT_SUPERVISOR_PUBLIC_API_EXPORTS,
    AGENT_SUPERVISOR_DOMAIN_PACKAGES,
    AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE,
    AGENT_SUPERVISOR_CORE_PACKAGES,
    AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS,
)
```

| Prefer | Avoid as primary name |
| --- | --- |
| `AGENT_SUPERVISOR_CORE_PACKAGES` | `AGENT_SUPERVISOR_G020_PACKAGES` |
| `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE` | `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS` |
| `AGENT_SUPERVISOR_PUBLIC_API_EXPORTS` | only `V2_STABLE_EXPORTS` |
| `AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS` | `AGENT_SUPERVISOR_EVIDENCE_CLUSTER_G020_G050` |

Deprecated aliases still resolve for compatibility. Board IDs remain **string
values** (e.g. `"ASREF-G020"`) for scanners.

### Cold import rules

Importing the package root, control contracts, or discovery helpers **must not**:

- start subprocesses,
- open network connections,
- load optional prover/dataset providers,
- mutate the repository.

Accessing a lazy provider attribute may import that adapter; it still does not
prove the provider is configured or healthy. **Discovery → capability probe →
work.**

---

## 4. Control plane usage

### Operation path

```text
Transport (Python / CLI / MCP)
    → OperationRequest (immutable, typed)
    → SupervisorControlService
    → backend + policy (allowlists, authz, leases, idempotency)
    → OperationResult (status, effects, receipts)
```

All three transports share one operation vocabulary. They differ in **how roots
and credentials are configured**, not in what operations mean.

### Adding or changing an operation

1. Update `control/control_contracts.py` (catalog, schemas, status codes).
2. Implement dispatch in `control/control_plane.py` (or the owning backend).
3. Keep CLI (`control_cli`) and MCP adapters in parity where the op is public.
4. Extend discovery / capability reporting if the op is advertised.
5. Add conformance tests under `test/api/`.
6. Do **not** casually expand `AGENT_SUPERVISOR_PUBLIC_API_EXPORTS` — that set is
   a closed generation-2 manifest; treat additions as API review work.

See architecture: *Stable control surface and operating model*.

---

## 5. Where to put new code

Use [PACKAGE_MAP.md](PACKAGE_MAP.md). Quick table:

| Change | Package |
| --- | --- |
| New operation, CLI binding, permit | `control/` |
| Board parse, queue, DuckDB source | `task_sources/` |
| Prover, attestation, proof cache | `proof/` |
| Context capsule / decision runtime | `context/` |
| Formal plan compile/validate | `planning/` |
| Proposal validation / pre-merge gate | `validation/` |
| Merge train, checkout lock, lease | `merge/` |
| Multi-lane runner, event log | `runtime/` |
| Implementation daemon loop | `todo_daemon/` |
| Optional external tool bridge | `integrations/` |
| Self-improvement epoch / refill | `self_improvement/` |
| Program board only (no new domain) | `docs/architecture/*.{todo,objectives}.md` |

### Dependency rules

- Edges flow **up** the DAG (core at bottom; daemons at top).
- No cycles.
- `core` must not import `todo_daemon`, `runtime`, `merge`, or `rescue`.
- Prefer absolute imports:
  `from ipfs_accelerate_py.agent_supervisor.<pkg>.<mod> import …`
- Relative imports stay within a package (`from .x import y`).

### Landed module resolution

`AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE` maps historical flat stems to domain
packages. Root resolution may alias `agent_supervisor.<stem>` →
`agent_supervisor.<package>.<stem>` for compatibility. **New callers must import
the domain path.**

---

## 6. Programs vs packages

A **program** is a long-running effort with boards and objectives (self-improvement,
codebase-proof, domain layout, Goose, catalog, …). It is **not** a second
supervisor and usually **not** a top-level package named after the prefix.

1. Keep/add `docs/architecture/<program>.{objectives,todo}.md`.
2. Register the glossary row in [PROGRAMS.md](PROGRAMS.md).
3. Implement code in existing domain packages.
4. Document operator launch (protected paths, providers) in the plan or guide.
5. Preserve task headers (`## PREFIX-###`) for the daemon.

---

## 7. Implementation daemons and lanes

High-level loop (see also architecture *Execution daemons*):

1. Parse taskboard + apply shard / ready / dependency filters.
2. Claim a task; create or reuse a fenced worktree.
3. Run the implementation provider with a bounded prompt and timeout.
4. Run validation; apply completion / merge-queue policy.
5. Emit events; on failure, rescue or release with retry budget.

When debugging:

| Symptom | Check |
| --- | --- |
| Task never selected | Board status, deps, shard index, protected-path contention |
| Stuck `implementing` | Child provider process, attempt log, timeout policy |
| Validation failed | Task `Validation` command, worktree tip, protected paths |
| Merge blocked | Target branch, lease, dirty tree, concurrent claims |

Ops entrypoints: `scripts/ops/agent_supervisor/` and the
[operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md).

Default provider env (ops may override):

| Path | Typical setting |
| --- | --- |
| Grok | `IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok` |
| Codex | `IPFS_ACCELERATE_AGENT_CODEX_MODEL=…` |

---

## 8. Evidence and validation

Typed evidence classes must not be collapsed:

- unit/integration tests  
- runtime observations / metrics  
- static analysis findings  
- solver candidates  
- kernel-checked proofs  
- cryptographic attestations  

Cache hits **re-derive** assurance; they do not invent it.

For a change, expect:

1. Targeted pytest for the modules you touch.
2. No unprotected rewrites of foreign boards or sealed plans.
3. Import paths in docs/examples use domain packages.
4. If you changed control ops: conformance / discovery tests still pass.

Useful layout/API suites:

```bash
python -m pytest \
  test/api/test_agent_supervisor_asref_layout_evidence.py \
  test/api/test_agent_supervisor_semantic_layout_exports.py -q
```

---

## 9. Documentation obligations

Non-trivial changes should update:

| Change type | Docs to touch |
| --- | --- |
| New/moved module | Owning package `README.md` + `packages/<name>.md` if semantic page lags |
| New public operation | Architecture control section, operator guide examples |
| Authority / evidence rule | Philosophy + agent capsule |
| Package placement rule | PACKAGE_MAP + this guide |
| New program | PROGRAMS.md + program plan |

Checklist: [FOR_CONTRIBUTORS.md](FOR_CONTRIBUTORS.md).

---

## 10. Common pitfalls

1. **Flat imports** after domain layout — use domain paths.
2. **Treating import success as capability** — always probe.
3. **Encoding ticket IDs in APIs** — keep board IDs on boards.
4. **Editing protected paths** “to help the daemon” — fail closed.
5. **Expanding v2 export manifests casually** — closed sets need review.
6. **Putting daemon logic in `core`/`proof`** — wrong DAG layer.
7. **Assuming local tests equal kernel proof** — different evidence tiers.

---

## 11. Where to go next

| Topic | Document |
| --- | --- |
| Design pillars & authority ladder | [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) |
| Subsystems, leases, formal planning | [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md) |
| CLI / MCP / profiles / recovery | [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) |
| Package ownership | [PACKAGE_MAP.md](PACKAGE_MAP.md) |
| Board prefix glossary | [PROGRAMS.md](PROGRAMS.md) |
| Historical layout cutover tables | [LAYOUT_CUTOVER_EVIDENCE.md](LAYOUT_CUTOVER_EVIDENCE.md) |
| Nested trees policy | [NESTED_PACKAGES.md](../../NESTED_PACKAGES.md) |
