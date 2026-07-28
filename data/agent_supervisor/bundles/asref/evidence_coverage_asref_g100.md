# ASREF-G100 Evidence Coverage Receipt

**Task:** ASREF-010
**Goal:** ASREF-G100 — Autonomous supervisor execution with Grok 4.6
**Date:** 2026-07-28
**Discovery input:** `data/agent_supervisor/discovery/asref/2026-07-27-asref-010-objective-gap-6eb7af222181.md`
**Track:** autonomous-execution

## Missing evidence terms (from objective scan)

| Evidence term | Role | Coverage produced by ASREF-010 |
|---|---|---|
| `docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md` | Human plan: branch model, lane/bundle matrix, protected paths, Grok operator notes, multi-lane launch sketch | Wired as **read-only plan source** in `launch_recipe.json`, `lane_matrix.json`, `protected_paths.json`, and every `--implementation-protected-path` emission from `scripts/ops/agent_supervisor/asref_multi_lane.py`. Content is **not rewritten** (operator-protected). |
| `docs/architecture/agent_supervisor_module_refactor.todo.md` | Executable todo board drained by implementation supervisors | Wired as **todo-path** for objective scan and multi-lane launch; listed as protected path; present on disk as the supervisor-fed backlog. Content is **not rewritten** by this task. |

## Present supporting evidence (already accepted)

- `scripts/ops/agent_supervisor/` — implementation entry, provider wrappers, ASREF multi-lane launcher
- Plan § “Supervisor launch (Grok 4.6 / implementation lanes)” — command shape mirrored by the launcher
- Objective heap ASREF-G100 — acceptance criteria mapped below

## How the terms are “covered” without editing protected files

The objective gap scanner treats the plan and todo paths as evidence *requirements* for ASREF-G100. Those files already exist and are **operator-protected**. Closing the gap means proving the autonomous-execution wiring *uses* them correctly:

1. **Todo board alignment** — launch recipe `paths.todo_board` and objective-scan CLI bind exclusively to `docs/architecture/agent_supervisor_module_refactor.todo.md`.
2. **Plan alignment** — lane matrix is projected from the plan’s lane/bundle table; protected-path list matches the plan’s protected-path section exactly.
3. **Protected-path fence** — every launch argv includes all three architecture files as `--implementation-protected-path`.
4. **Provider selection** — Grok 4.6 (or successor) is selected via `IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER` at launch time; goal/todo text stays provider-agnostic. Provider bridges remain under integrations/runtime ownership and do not gate package moves.
5. **Bundle isolation** — repo-local `data/agent_supervisor/bundles/asref` is the objective-bundle-dir; shards assign lanes by `Bundle` fields.

## Acceptance subset checklist (ASREF-G100)

| Criterion | Proof artifact |
|---|---|
| Objective daemon can scan this heap into the todo board | `asref_multi_lane.py objective-scan` + recipe `commands.objective_scan` |
| Bundle index assigns lanes by Bundle fields | `lane_matrix.json`, `seed_manifest.json`, bootstrap shard headers |
| Implementation supervisor launch docs/scripts protect the three architecture files | `protected_paths.json`, launcher `_common_args()`, ops README |
| Grok 4.6 (or successor) selectable without changing goal text | `launch_recipe.json` `implementation_provider`, launcher `--implementation-provider` |
| Workers follow Validation lines and the no-shim rule | Recipe `invariants`; plan protected as evidence of the no-shim rule |

## Validation

```bash
test -f docs/architecture/agent_supervisor_module_refactor.todo.md \
  && test -d data/agent_supervisor/bundles/asref
python scripts/ops/agent_supervisor/asref_multi_lane.py preflight
python scripts/ops/agent_supervisor/asref_multi_lane.py verify-evidence
```

## No-shim / provider policy

- Package moves must not wait on provider choice.
- Provider wiring stays in integrations/runtime surfaces; this seed only documents env/flags.
- Never leave thin re-export stubs at retired flat `agent_supervisor/*.py` paths.
