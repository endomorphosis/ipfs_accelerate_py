# `scripts/ops/agent_supervisor`

Stable ops entry points for agent-supervisor daemons. These scripts only adjust
`sys.path` for a source checkout and/or assemble multi-lane launch argv; core
logic lives under `ipfs_accelerate_py/agent_supervisor/` (runtime, todo_daemon,
objectives, integrations).

## Scripts

| Script | Role |
|---|---|
| `implementation_supervisor_entry.py` | Thin entry for multi-supervisor implementation tracks |
| `configured_board_scheduler.py` | Fail-closed adapter from sealed scheduler JSON to multi-lane supervisors |
| `asref_multi_lane.py` | **ASREF-G100** preflight, objective scan, multi-lane launch |
| `meta_spark_goose_runner.py` | Ops wrapper for Meta Spark + goose implementation runner |
| `prompt_workflow.py` | Ops wrapper for prompt-workflow CLI |
| `ipfs_kit_vfs_symbolic_assurance.py` | Thin facade for the IPFS Kit VFS symbolic-assurance job |
| `ipfs_kit_vfs_symbolic_assurance_control.sh` | Two-shard supervisor control for the VFS assurance board |
| `validate_ipfs_kit_vfs_symbolic_assurance.py` | Fail-closed preflight for the VFS objective heap and board |

## Sealed scheduler configs

Use the shared adapter when a reviewed `scheduler_config@1` document owns
lane count, paths, retry policy, sharding, and protection:

```bash
python scripts/ops/agent_supervisor/configured_board_scheduler.py \
  --config config/<board>_scheduler.json preflight
python scripts/ops/agent_supervisor/configured_board_scheduler.py \
  --config config/<board>_scheduler.json launch --implement --dry-run
```

Preflight and dry-run do not probe providers. A real launch requires the
explicit `--implement` flag and re-runs the fail-closed Git, submodule, and
declared-validator checks.

## IPFS Kit VFS symbolic assurance (generalized engines)

Reusable assurance engines live under semantic packages
(`analysis/`, `validation/`, `runtime/`, `control/`). Domain vocabulary and
the locked job profile live only in:

- `config/ipfs_kit_vfs_symbolic_assurance.json`
- `ipfs_accelerate_py/agent_supervisor/integrations/ipfs_kit_vfs_assurance.py`

There is **no** root `agent_supervisor/vfs_*.py` module. Do not add
compatibility shims.

### Thin ops facade

```bash
# Project operation/invariant/error/canonical-vector mappings
python scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance.py contracts

# Adversarial gates + shadow rollout (default mutation disabled)
python scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance.py rollout --mode assist

# Verify gates and rollout decision
python scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance.py verify

# Closed adapters: inventory | contracts | differential | parity |
#                  benchmark | pilot | rollout | verify
python scripts/ops/agent_supervisor/ipfs_kit_vfs_symbolic_assurance.py --help
```

The wrapper only parses arguments, loads the locked config, lazy-loads the
integration, and delegates. It must not embed scan, proof, comparison, gate,
repair, or mutation logic. Cold import and `--help` start no process, open no
database, and load no optional providers.

### Cutover / placement guards

Equivalence is a content-addressed fixed point over
`VfsGeneralizationEquivalenceReceipt` + `VfsCallerMigrationReceipt` (locked
source → generic engines, caller impact closure, Tactician/Hammer dispositions),
`VfsRootLayoutGuard` (no root `vfs_*.py`, no `agent_supervisor.vfs_*` imports,
domain-free generic modules, thin ops), and `AssuranceTwoProfileConformance`
(VFS profile and hermetic non-VFS fixture share the same engine modules).

- `test/api/test_agent_supervisor_vfs_generalization_equivalence.py`
- `test/api/test_agent_supervisor_vfs_root_layout_guard.py`
- `test/api/test_agent_supervisor_assurance_two_profile_end_to_end.py`
- Plan section **Assurance generalization cutover** in
  `docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md`
- Map: `docs/architecture/agent_supervisor/VFS_ASSURANCE_GENERALIZATION_MAP.md`

Unsupported dispositions retained (not claimed as proved): source-blob byte
equivalence and unresolved dynamic/native public-API diff.
### Board preflight

```bash
python scripts/ops/agent_supervisor/validate_ipfs_kit_vfs_symbolic_assurance.py
```

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

# Multi-lane implementation with Grok (or successor) as provider
export IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok
python scripts/ops/agent_supervisor/asref_multi_lane.py launch \
  --implementation-provider grok \
  --enable-objective-refill \
  --dry-run   # remove --dry-run to start
```

### Provider selection

| Env / flag | Meaning |
|---|---|
| `IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok` | Prefer Grok CLI (ASREF-G100 default) |
| `=auto` / `goose` / `codex` / `copilot` | Other selectable providers |
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
