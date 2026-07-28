# Agent Supervisor Module Refactor Plan

**Branch:** `refactor/agent-supervisor-layout`  
**Prefix:** `ASREF` (Agent Supervisor REFactor)  
**Objective heap:** [`agent_supervisor_module_refactor.objectives.md`](./agent_supervisor_module_refactor.objectives.md)  
**Todo board:** [`agent_supervisor_module_refactor.todo.md`](./agent_supervisor_module_refactor.todo.md)  
**Implementation agent:** Grok 4.6 (and existing implementation-daemon / multi-lane supervisor)

## Problem statement

Today `ipfs_accelerate_py/agent_supervisor/` is effectively a **flat module warehouse**:

| Location | Approx. size | Pain |
|---|---:|---|
| Top-level `*.py` | ~153 modules | No domain boundaries; discovery by filename only |
| `todo_daemon/` | ~34 modules | Mixed lifecycle, git, LLM, legal, supervisor, implementation |
| Package root | ~50 entries + nested sibling trees | Runtime junk, mixed products, dual `test/`/`tests/`, nested package checkouts |

The flat layout forces every autonomous worker to load a large import surface, makes protected-path and conflict domains hard to declare, and buries ownership. The monorepo root also mixes installable package surfaces, nested git checkouts, ad-hoc scripts, and ephemeral process files.

## Non-goals

- Rewriting supervisor algorithms, proof systems, or self-improvement completion gates (except where moves force import/path updates).
- Introducing **thin re-export compatibility wrappers** that freeze the old flat import paths indefinitely.
- Turning internal Python packages into **git submodules**.
- Force-pushing or rewriting `main` history; all work lands on a dedicated branch and merges via normal review/CI.

## Goals (program-level)

1. **Package submodules** under `agent_supervisor/` with clear domain ownership and a `README.md` in each package.
2. **Explicit public API** via a thin root `__init__.py` that re-exports only intentional symbols (not the entire tree).
3. **Root directory hygiene** for the `ipfs_accelerate_py` repo: keep installable surface, move or ignore junk, document nested packages.
4. **Hard cut of import paths**: update entry points, scripts, tests, docs, and protected-path configs to the new locations **in the same change** as each move. No long-lived shims.
5. **Supervisor-native execution**: objective heap + bundles that multi-lane implementation daemons (Grok 4.6) can drain autonomously.

## Design principles

1. **File-ownership lanes.** Each child goal owns a disjoint path set (`Outputs:`) so parallel lanes do not thrash the same files.
2. **Move then fix callers.** For package `P`, one task moves modules into `P/`, updates every import and entry point, and runs a focused pytest selection. Status stays open until imports and tests pass on the current tree.
3. **No silent aliases.** Temporary imports of old paths must not exist after the goal’s validation gate. Prefer `rg`-enforced absence of old module paths.
4. **README as contract.** Each package README states purpose, public modules, inbound/outbound dependencies, and forbidden dependencies (to prevent cycles).
5. **Root hygiene is a separate track** with its own bundles so agent_supervisor moves are not blocked by doc/data cleanup.
6. **Grok 4.6 readiness.** Validation commands are copy-pasteable; acceptance criteria are path- and test-bound; gap tasks name the next atomic move.

## Target layout

```text
ipfs_accelerate_py/agent_supervisor/
  README.md                          # map of packages + public API
  __init__.py                        # explicit re-exports only
  py.typed                           # if packaging requires

  core/                              # identity, events, shared contracts, task_source protocol
  control/                           # control plane, CLI, lifecycle, permits, authz surface
  objectives/                        # objective heap parse, daemon, tracker, janitor, goal_*
  planning/                          # adaptive planning, formal plan compile/validate (non-proof runtime)
  proof/                             # formal verification, provers, leanstral, attestation
  analysis/                          # analysis pipeline, AST, cache, retrieval, consensus
  context/                           # context compiler/contracts, decision runtime
  prompt/                            # prompt workflow, directory scanner, plan admission, goal planner
  task_sources/                      # markdown, duckdb, taskboard, queues, vector index
  validation/                        # proposal validation, validation scheduler/runtime/commands
  merge/                             # merge queue/train/resolver, checkout lock, git_gc, leases
  runtime/                           # multi_supervisor, runners, resource/scheduler, runtime CAS
  rescue/                            # rescue planner/orchestrator, recovery, watchdog hooks
  self_improvement/                  # self-improvement + supervisor_v2 contracts/metrics
  integrations/                      # llm merge fallback, meta_spark/goose, datasets providers
  todo_daemon/                       # executable daemons only; split internal packages below
    implementation/                  # implementation_daemon, implementation_supervisor
    loop/                            # supervisor_loop, runner, core engine pieces
    git/                             # worktrees, git_utils, auto_commit
    ...
```

Exact file→package maps live in the objective heap children (`ASREF-G1xx`). The map may be refined by the heap-refiner when evidence shows a better cut, but **must keep package DAGs acyclic**.

### Package dependency DAG (allowed direction)

```text
core
  ↑
control, task_sources, context, analysis, proof
  ↑
objectives, planning, validation, prompt
  ↑
merge, rescue, runtime, self_improvement
  ↑
todo_daemon.implementation, integrations
```

Forbidden: `core` importing `todo_daemon`; `proof` importing `todo_daemon.implementation`; cross-package cycles.

## Branch and execution model

### Branch bootstrap

```bash
cd /path/to/ipfs_accelerate_py
git fetch origin
git checkout -b refactor/agent-supervisor-layout origin/main
# land objectives + plan first (this document + heap + empty/seed todo)
```

### Supervisor launch (Grok 4.6 / implementation lanes)

Treat the heap as the durable intent and the todo board as the drainable projection:

```bash
# 1) Refine broad goals into children if needed
ipfs-accelerate-agent-objective-daemon \
  --objective-path docs/architecture/agent_supervisor_module_refactor.objectives.md \
  --todo-path docs/architecture/agent_supervisor_module_refactor.todo.md \
  --refine-objective-heap \
  --discovery-dir data/agent_supervisor/discovery/asref \
  --objective-bundle-dir data/agent_supervisor/bundles/asref

# 2) Generate / refresh todos from missing evidence
ipfs-accelerate-agent-objective-daemon \
  --objective-path docs/architecture/agent_supervisor_module_refactor.objectives.md \
  --todo-path docs/architecture/agent_supervisor_module_refactor.todo.md \
  --discovery-dir data/agent_supervisor/discovery/asref \
  --objective-bundle-dir data/agent_supervisor/bundles/asref

# 3) Run multi-lane implementation against bundle shards
#    (use Grok 4.6 as the implementation provider when wired)
ipfs-accelerate-agent-bundle-supervisor \
  --objective-bundle-dir data/agent_supervisor/bundles/asref \
  --start
```

**Protected paths for this program** (must appear on every implementation supervisor):

- `docs/architecture/agent_supervisor_module_refactor.objectives.md`
- `docs/architecture/agent_supervisor_module_refactor.todo.md`
- `docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`

### Lane / bundle matrix

| Bundle | Owns | Parallel with |
|---|---|---|
| `asref/bootstrap` | Branch, inventory, import graph, move map | alone first |
| `asref/core` | `core/` package | after bootstrap |
| `asref/control` | `control/` | after core |
| `asref/task-sources` | `task_sources/` | after core |
| `asref/context` | `context/` | after core |
| `asref/analysis` | `analysis/` | after core |
| `asref/proof` | `proof/` | after core |
| `asref/objectives` | `objectives/` | after core+task_sources |
| `asref/planning` | `planning/` | after core+proof |
| `asref/prompt` | `prompt/` | after context |
| `asref/validation` | `validation/` | after core |
| `asref/merge` | `merge/` | after core |
| `asref/rescue` | `rescue/` | after core |
| `asref/runtime` | `runtime/` | after control+merge |
| `asref/self-improvement` | `self_improvement/` | after objectives+validation |
| `asref/todo-daemon` | `todo_daemon/**` re-packaging | after runtime+merge |
| `asref/integrations` | `integrations/` | after runtime |
| `asref/public-api` | root `__init__.py`, package README | after all packages |
| `asref/root-hygiene` | monorepo root cleanup | parallel after bootstrap |
| `asref/scripts-entrypoints` | pyproject/setup/scripts | with each package; final sweep |
| `asref/docs-tests` | docs + test path alignment | final |
| `asref/cutover` | CI, no old imports, merge readiness | last |

## Move procedure (every package goal)

For package `P` with modules `M1..Mn`:

1. Create `ipfs_accelerate_py/agent_supervisor/P/` with `__init__.py` and `README.md`.
2. `git mv` modules into `P/` (preserve history).
3. Fix **all** imports:
   - in-package relative imports
   - cross-package absolute imports
   - `pyproject.toml` / `setup.py` entry points
   - `scripts/**`
   - `test/**` and `tests/**`
   - docs/architecture protected paths and validation commands
4. Update any string path references (validation commands, discovery evidence paths, protected-path lists).
5. Run package validation from the goal’s `Validation:` line.
6. Assert absence of old paths:  
   `rg -n 'agent_supervisor\.(OLD_MODULE)\b' --glob '!**/__pycache__/**'`
7. Do **not** leave `agent_supervisor/OLD_MODULE.py` re-export stubs.

## Root hygiene targets

Keep at monorepo root only what packaging and humans need:

| Keep | Action for others |
|---|---|
| `pyproject.toml`, `setup.py`, `pytest.ini`, `MANIFEST.in`, `LICENSE`, `README.md`, `SECURITY.md`, `CHANGELOG.md`, `requirements*.txt` (consolidated later) | |
| `ipfs_accelerate_py/` (installable package) | |
| `docs/`, `scripts/`, `deployments/`, `install/`, `examples/`, `config/` | |
| `test/` (canonical test tree) | Merge or deprecate root `tests/` → `test/` |
| Nested product trees (`ipfs_accelerate_js`, nested kits) | Document in `docs/NESTED_PACKAGES.md`; do not delete without ownership review |
| `dashboard.out`, `dashboard.pid`, `err.txt`, root one-off `test_*.py` | `.gitignore` + move tests under `test/`; delete ephemeral runtime files |
| `MCP_SERVER_UNIFICATION_PLAN.md`, `SDK_PLAYGROUND_PREVIEW.html` | Move under `docs/` |
| `state/`, local DBs under `data/` | Ensure gitignored where ephemeral |

## Risk register

| Risk | Mitigation |
|---|---|
| Mass import breakage | Per-package moves with focused pytest; final full `test/api/test_agent_supervisor_*.py` gate |
| Active supervisors thrashing `main` | Dedicated branch; protected objective/todo paths; worktrees for agents |
| Cyclic imports after split | Package DAG + import-linter or custom test ASREF-G080 |
| Hidden string imports / dynamic import paths | `rg` inventory in bootstrap; dynamic-import allowlist test |
| Todo board regenerates moving targets | Freeze move map in ASREF-G010 evidence before bulk moves |
| Nested monorepo confusion | Root hygiene docs; no accidental deletion of nested products |

## Success criteria (program)

- [ ] Branch `refactor/agent-supervisor-layout` exists from current `origin/main`.
- [ ] Every domain package under `agent_supervisor/` has `README.md` and `__init__.py`.
- [ ] Zero production modules remain in the flat root of `agent_supervisor/` except `__init__.py` (and optional `py.typed`).
- [ ] `rg` finds no imports of pre-move module paths.
- [ ] All console entry points in `pyproject.toml`/`setup.py` resolve to new modules.
- [ ] Full agent-supervisor API test suite passes on the branch.
- [ ] Root directory no longer contains ephemeral process files or misplaced one-off tests.
- [ ] Objective heap children are `verified_complete` with current-tree validation receipts (supervisor lifecycle).

## Operator notes for Grok 4.6

- Prefer **one package move per task** (or one tightly coupled pair) so context stays small.
- Always include the package README update in the same commit as the move.
- Prefer `git mv` over copy/delete.
- After each move, run the goal’s Validation line before marking the todo complete.
- Never “fix” import failures by adding a compatibility stub at the old path.
- If two lanes touch the same file, stop and refine the heap into smaller ownership boundaries rather than racing merges.

## Related documents

- [`docs/agent_supervisor_objective_graph.md`](../agent_supervisor_objective_graph.md) — how heaps become bundles and todos
- [`docs/architecture/ai_service_catalog.objectives.md`](./ai_service_catalog.objectives.md) — style reference for goal fields
- [`docs/architecture/agent_supervisor_self_improvement.objectives.md`](./agent_supervisor_self_improvement.objectives.md) — completion/evidence rigor reference

## Execution log (bootstrap)

- Branch `refactor/agent-supervisor-layout` created from main tip including plan seed commit.
- ASREF-001 inventory committed: `docs/architecture/asref/move_map.json` and `import_inventory.md` (tracked; `data/.../discovery` is gitignored).
- Isolated worktree for continued work: `/home/barberb/.local/share/ipfs_accelerate_py/manual-worktrees/asref-layout` (avoids concurrent main-checkout thrash).
- First objective scan (`--no-reconcile-goal-completion --max-findings 12`) appended ASREF-008..014 gap tasks and wrote bundle shards under `data/agent_supervisor/bundles/asref/`.
- Next operator actions: run ASREF-002 launch wiring, then package moves starting with ASREF-003 (`core/`) using Grok 4.6 multi-lane supervisors **bound to this worktree/branch only**.

