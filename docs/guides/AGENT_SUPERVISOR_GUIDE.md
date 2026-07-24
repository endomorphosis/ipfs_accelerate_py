# Agent Supervisor Guide

This guide documents the current `ipfs_accelerate_py.agent_supervisor` control
plane. It is for maintainers and operators who want to generate objective-driven
work, run isolated implementation lanes, inspect evidence, or add a new
provider. It is not required for ordinary model inference.

The authoritative design description is
[Agent Supervisor Architecture](../architecture/AGENT_SUPERVISOR_ARCHITECTURE.md).
The formal planning and assurance details live in the
[formal planning/prover matrix](../architecture/AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md),
[formal verification plan](../architecture/AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md),
and [Leanstral goal-development benchmark](../architecture/AGENT_SUPERVISOR_LEANSTRAL_GOAL_DEVELOPMENT.md).

## What the supervisor does

The supervisor is a bounded control plane around implementation agents. Its
responsibilities are deliberately separated:

1. An objective heap describes desired outcomes and required evidence.
2. Deterministic scanners build objective, AST, dependency, and proof-gap
   projections.
3. The objective daemon turns missing evidence into typed todo records and
   bundle shards.
4. The bundle supervisor schedules isolated lanes with leases, resource
   limits, conflict metadata, and separate state roots.
5. The implementation daemon asks an LLM to propose edits in an ephemeral
   worktree, runs configured validation, and records receipts.
6. The implementation supervisor watches heartbeats, retries bounded failures,
   reconciles worktrees, and creates follow-up tasks when a retry budget is
   exhausted.

LLM output is proposal material. A model response does not itself prove a task
complete, authorize a merge, or change the canonical objective graph. Those
decisions require deterministic validation and the configured assurance gates.

## Installation and entry points

The supervisor tools are installed with the package. From a source checkout:

```bash
python -m pip install -e ".[dev]"
```

The installed console scripts are:

| Command | Purpose |
| --- | --- |
| `ipfs-accelerate-agent-objective-daemon` | Scan an objective heap and generate tasks, graph artifacts, datasets, and bundle shards. |
| `ipfs-accelerate-agent-backlog-refinery` | Refill a low backlog and turn code, retry, and dependency findings into bounded follow-up work. |
| `ipfs-accelerate-agent-bundle-supervisor` | Plan or launch isolated supervisors for bundle shards. |
| `ipfs-accelerate-agent-implementation-daemon` | Drain a Markdown task board with the implementation loop. |
| `ipfs-accelerate-agent-implementation-supervisor` | Watch and repair an implementation daemon. |
| `ipfs-accelerate-agent-artifact-query` | Query JSON/DuckDB evidence artifacts without loading large payloads into prompts. |
| `ipfs-accelerate-agent-merge-resolver` | Inspect a failed merge and build a bounded resolver prompt. |
| `ipfs-accelerate-agent-llm-merge-resolver-fallback` | Run the packaged Codex/Copilot merge-repair fallback. |

The package dispatcher is also available as:

```bash
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon list
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon --help
```

The dispatcher currently registers the reusable `legal-parser` and `logic-port`
daemon families. The objective and implementation tools above are separate
entry points because they require different project-specific paths and state
contracts.

## Generate objective work

An objective heap is a Markdown document containing `## GOAL-...` records. The
scanner preserves goal identity, evidence references, dependencies, and bundle
metadata. A minimal source-checkout invocation is:

```bash
ipfs-accelerate-agent-objective-daemon \
  --repo-root "$PWD" \
  --objective-path docs/objectives.md \
  --todo-path data/agent_supervisor/tasks.todo.md \
  --discovery-dir data/agent_supervisor/discovery \
  --bundle-dir data/agent_supervisor/objective_bundles \
  --dataset-dir data/agent_supervisor/objective_datasets \
  --graph-path data/agent_supervisor/objective_graph.json
```

The objective file must exist unless `--ensure-tracking-document` is supplied
with `--ultimate-goal`. Useful optional stages include:

```bash
# Refine broad goals into bounded child goals before task generation.
ipfs-accelerate-agent-objective-daemon ... --refine-objective-heap

# Ask the router for bounded plan alternatives; deterministic validation still
# selects or rejects them.
ipfs-accelerate-agent-objective-daemon ... \
  --generate-plan-branches --plan-branch-count 3

# Submit generated bundle shards to the local task queue.
ipfs-accelerate-agent-objective-daemon ... --submit-bundles
```

The normal outputs are:

- `objective_graph.json`: goal nodes, dependencies, depth, and priorities;
- `objective_bundles/index.json`: bundle and task projections;
- `objective_bundles/*.todo.md`: one worker-facing shard per bundle;
- `objective_bundles/todo_vector_index.json`: bounded lexical, vector, and AST
  relationships for task selection;
- `objective_datasets/`: JSONL and optional dataset-manager projections of
  larger AST/symbol evidence;
- discovery and summary receipts under the configured state directory.

These are projections and evidence artifacts. The objective heap remains the
source of intent; a generated todo file is not a second authority.

## Run isolated lanes

Plan lanes first. Planning is the default and does not launch an LLM:

```bash
ipfs-accelerate-agent-bundle-supervisor \
  --bundle-index-path data/agent_supervisor/objective_bundles/index.json \
  --repo-root "$PWD" \
  --state-root data/agent_supervisor/bundles \
  --worktree-root data/agent_supervisor/worktrees \
  --log-dir data/agent_supervisor/logs \
  --once
```

Launch only after reviewing the generated manifest:

```bash
ipfs-accelerate-agent-bundle-supervisor \
  --bundle-index-path data/agent_supervisor/objective_bundles/index.json \
  --repo-root "$PWD" \
  --state-root data/agent_supervisor/bundles \
  --worktree-root data/agent_supervisor/worktrees \
  --log-dir data/agent_supervisor/logs \
  --start --max-lanes 4
```

`--max-lanes` is an admission limit, not a promise to start that many
processes. Dependency readiness, conflicting paths, CPU/memory/disk budgets,
provider capacity, and active leases can reduce the admitted width.

For a single board, the implementation daemon and supervisor can be run
directly:

```bash
# One pass, no implementation agent.
ipfs-accelerate-agent-implementation-daemon \
  --once --todo-path data/agent_supervisor/tasks.todo.md \
  --state-dir data/agent_supervisor/implementation

# A long-running daemon that may invoke the configured implementation command.
ipfs-accelerate-agent-implementation-daemon \
  --implement --interval 300 \
  --todo-path data/agent_supervisor/tasks.todo.md \
  --state-dir data/agent_supervisor/implementation

# One health/reconciliation pass.
ipfs-accelerate-agent-implementation-supervisor \
  --once --todo-path data/agent_supervisor/tasks.todo.md \
  --state-dir data/agent_supervisor/implementation
```

For unattended operation, use the lifecycle wrappers exposed by the registered
daemon families and verify health before assuming that a process is making
progress:

```bash
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon \
  legal-parser check
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon \
  legal-parser ensure
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon \
  legal-parser spec
```

`check` reports heartbeat age and process liveness. `ensure` starts the
wrapper/supervisor only when health is not fresh. `stop` terminates the owned
process group, and `spec` prints the resolved paths and launch environment.

## Backlog maintenance

The backlog refinery is intentionally bounded and can run in a scheduled
maintenance pass:

```bash
ipfs-accelerate-agent-backlog-refinery \
  --repo-root "$PWD" \
  --todo-path data/agent_supervisor/tasks.todo.md \
  --state-path data/agent_supervisor/state.json \
  --events-path data/agent_supervisor/events.jsonl \
  --objective-path docs/objectives.md \
  --objective-scan --codebase-scan --retry-budget --dependency-guardrail
```

The four modes have different evidence sources:

- `--objective-scan` finds missing objective evidence;
- `--codebase-scan` finds bounded static findings in tracked code and worktrees;
- `--retry-budget` converts repeated implementation, validation, or merge
  failures into repair tasks;
- `--dependency-guardrail` repairs missing, self-referential, or cyclic task
  dependencies.

When no mode flag is supplied, the refinery runs all available modes. Generated
tasks are deduplicated by canonical identity and discovery fingerprint.

## Evidence and artifacts

The supervisor keeps large evidence out of task prompts. JSONL events and
versioned artifacts retain identities, verdicts, paths, counts, and bounded
diagnostics; raw provider output and large source bodies stay in their dedicated
artifact locations. Use the query tool for inspection:

```bash
ipfs-accelerate-agent-artifact-query \
  data/agent_supervisor/objective_bundles/index.json --schema

ipfs-accelerate-agent-artifact-query \
  data/agent_supervisor/objective_bundles/index.json \
  --table bundles --limit 20
```

The artifact store supports bundle indexes, scheduler manifests, code-evidence
graphs, proof attestations, and proof metrics. Each artifact carries a schema
and canonical identity so projections can be rebuilt or migrated.

## Leanstral and formal assurance

Leanstral is an optional proposal provider for goal development and proof
candidates. It can suggest refinements, translations, and candidate proofs for
the supported logic families, but its output remains untrusted until the typed
contracts, capability checks, and authoritative prover receipts accept it.

The relevant operational sequence is:

1. Discover effective provider context limits and prover capabilities.
2. Build a bounded goal-development context from immutable goal and evidence
   records.
3. Generate a proposal or proof candidate.
4. Validate its schema, logic vocabulary, scope, and translation semantics.
5. Route independent obligations to the registered prover portfolio.
6. Persist bounded receipts and use them for planning, merge, and completion
   gates.

Use the architecture and benchmark documents linked at the top of this guide
for rollout policy. Do not treat model confidence, package discovery, or a
syntactically valid candidate as proof.

## Extending the supervisor

The extension-point guidance in the architecture document is for maintainers
adding integrations; it is not a list of mandatory next steps for every user.
New integrations should normally:

- expose evidence through a versioned receipt;
- register provers through capability/conformance contracts;
- add task sources through the objective graph and backlog refinery;
- route LLM calls through the existing provider boundary;
- express scheduler policy with typed resource and lease contracts; and
- persist projections through versioned artifact stores.

This keeps proposal, admission, execution, and evidence responsibilities
separate and prevents a new integration from silently becoming an authority
source.

## Tests

The supervisor API tests are grouped under `test/api/`:

```bash
python -m pytest \
  test/api/test_agent_supervisor_objective_graph.py \
  test/api/test_agent_supervisor_todo_daemon_port.py \
  test/api/test_agent_supervisor_bundle_plan_cache.py -q

python -m pytest \
  test/api/test_agent_supervisor_leanstral_goal_benchmark.py \
  test/api/test_agent_supervisor_leanstral_goal_lifecycle_e2e.py -q
```

Optional provider, prover, IPFS, and P2P tests require their corresponding
dependencies and should be run separately from the deterministic contract
tests.
