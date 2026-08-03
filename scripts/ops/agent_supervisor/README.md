# `scripts/ops/agent_supervisor`

Stable ops entry points for agent-supervisor daemons. These scripts only adjust
`sys.path` for a source checkout and/or assemble multi-lane launch argv; core
logic lives under `ipfs_accelerate_py/agent_supervisor/` (runtime, todo_daemon,
objectives, integrations).

## Scripts

| Script | Role |
|---|---|
| `implementation_supervisor_entry.py` | Thin entry for multi-supervisor implementation tracks |
| `asref_multi_lane.py` | **ASREF-G100** preflight, objective scan, multi-lane launch |
| `meta_spark_goose_runner.py` | Ops wrapper for Meta Spark + goose implementation runner |
| `prompt_workflow.py` | Ops wrapper for prompt-workflow CLI |

## ASREF module-refactor multi-lane (ASREF-G100)

Program documents (operator-protected — never rewrite from workers):

- `docs/architecture/agent_supervisor_module_refactor.objectives.md`
- `docs/architecture/agent_supervisor_module_refactor.todo.md`
- `docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md`

Bundle directory (lane shards + launch recipe):

- `data/agent_supervisor/bundles/asref/`

### Exact launch sequence

```bash
# Preflight (heap, board, protected paths, open tasks)
python scripts/ops/agent_supervisor/asref_multi_lane.py preflight

# Prove plan/todo evidence terms are wired (ASREF-G100)
python scripts/ops/agent_supervisor/asref_multi_lane.py verify-evidence

# Optional: refine broad goals, then scan missing evidence into the todo board
python scripts/ops/agent_supervisor/asref_multi_lane.py objective-scan --refine-objective-heap
python scripts/ops/agent_supervisor/asref_multi_lane.py objective-scan

# Multi-lane implementation: pinned Grok 4.5, quota-only Terra/medium fallback
export IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=auto
python scripts/ops/agent_supervisor/asref_multi_lane.py launch \
  --implementation-provider auto \
  --enable-objective-refill \
  --dry-run   # remove --dry-run to start
```

### Provider selection

| Env / flag | Meaning |
|---|---|
| `IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=auto` | Default: Grok 4.5 first; `gpt-5.6-terra` at medium only after verified Grok quota exhaustion |
| `=grok` / `goose` / `codex` / `copilot` | Explicit provider routes; forced Grok does not fall back |
| `IMPLEMENTATION_DAEMON_COMMAND=...` | Full command override |

Provider wiring stays in **integrations/runtime**. Package-move tasks must not
block on provider choice; goal text remains provider-agnostic.

### Protected-path fence

Every launch emits:

```text
--implementation-protected-path docs/architecture/agent_supervisor_module_refactor.objectives.md
--implementation-protected-path docs/architecture/agent_supervisor_module_refactor.todo.md
--implementation-protected-path docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md
```

### Bundle isolation

Objective refill and objective-scan write under:

- `--objective-bundle-dir data/agent_supervisor/bundles/asref`
- `--discovery-dir data/agent_supervisor/discovery/asref`

Lanes are assigned by each task’s `Bundle` field (`lane_matrix.json`).

### Worker invariants

1. Run the task `Validation:` command on the current tree.
2. No thin compatibility wrappers at retired flat `agent_supervisor/*.py` paths.
3. Honor `Bundle` / `Conflict policy` ownership.
4. Prefer one package move per task after the freeze map exists.

See also `data/agent_supervisor/bundles/asref/README.md` and
`data/agent_supervisor/bundles/asref/evidence_coverage_asref_g100.md`.
