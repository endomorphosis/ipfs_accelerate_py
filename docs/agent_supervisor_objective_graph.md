# Objective-Driven Agent Supervisor

`ipfs_accelerate_py.agent_supervisor` is the acceleration-layer home for
objective-driven autonomous agent execution.  It ports the repo-local objective
scanner and the reusable todo-daemon concepts into `ipfs_accelerate_py` without
modifying `ipfs_datasets_py`.

## Purpose

The supervisor starts from a long-range objective, such as "make the monorepo
operate as a virtual AI operating system", and parses it into a graph of
Fibonacci-prioritized subgoals.  The scanner then compares each goal against the
repository and submodules using:

- repository path evidence,
- exact text evidence,
- AST/symbol evidence from Python, JavaScript, TypeScript, Markdown, and JSON,
- deterministic token-embedding evidence for semantic near matches.

Missing evidence becomes todo work.  Each generated task is mirrored into a
bundle-local todo shard so multiple Codex daemons can work independent lanes in
parallel.

When a goal has an explicit `Bundle:` field, that bundle assignment is honored.
When it does not, the scanner assigns a bundle using the goal track, the
conflict-domain path root, AST/query text, present evidence paths, and
deterministic sentence-embedding similarity. This keeps semantically similar
work that is likely to touch the same files in one lane while allowing unrelated
roots to run in parallel. The implicit clustering threshold is controlled by
`IPFS_ACCELERATE_AGENT_BUNDLE_CLUSTER_MIN_SCORE`, which defaults to `0.42`.

The objective heap is the tracking document and stays separate from executable
todo boards. Use `--ensure-tracking-document` to create it from an ultimate goal
when it is missing, and `--refine-objective-heap` to append child goals for broad
missing-evidence gaps before generating todos. A JSON graph artifact is written
on each objective-daemon run so supervisors can inspect roots, edges, depths,
priorities, and active goal counts without reparsing markdown.

When the backlog needs more raw material than one aggregate objective gap,
enable surplus generation with `--surplus-findings-per-goal` or the supervisor's
`--objective-surplus-findings-per-goal`. The first generated todo remains the
aggregate gap, while additional todos split individual missing evidence terms.
Those surplus todos carry `Goal id`, `Missing evidence`, `Surplus group`,
`Merge key`, `Candidate kind`, `Embedding query`, `AST query`, and
`Todo vector key` metadata so related candidates can be bundled, merged, or
collapsed without asking the LLM to rediscover the structure from prose.

Large AST and symbol payloads are dataset artifacts.  The accelerator writes a
JSONL artifact and manifest for every objective scan, and when `ipfs_datasets_py`
plus its dataset dependencies are importable it also saves the rows through
`ipfs_datasets_py.dataset_manager.DatasetManager` and writes a parquet dataset.
This keeps bulky scan evidence out of markdown todos while still making it
available to dataset tooling, provenance systems, and future vector indexes.

Each objective todo generation pass also writes a compact todo vector/AST index
at `objective_bundles/todo_vector_index.json` unless disabled with
`--no-todo-vector-index` or `--no-objective-todo-vector-index`. The index stores
deterministic token embeddings, AST symbols gathered from task outputs, nearest
related task ids, merge keys, surplus groups, cluster summaries, and explicit
`merge_candidates`.  Merge candidates are compact groups derived from exact merge
keys, surplus groups, and vector/AST clusters; they include active task ids,
shared outputs, missing evidence, AST symbols, and estimated prompt-token weight.
Bundle indexes are backfilled with `todo_vector_summary` and per-task vector
metadata, which lets bundle supervisors select cohesive lanes and keep worker
prompts focused on a shard rather than the whole todo board.

Implementation daemons also read that index while selecting the next ready task.
Existing priority, retry, and dependency guardrails still win, but comparable
ready tasks are ranked by merge key, vector cluster, related-task ids, surplus
group, goal id, and ready-cluster density. This keeps a daemon on adjacent work
after a successful task, so follow-up prompts can reuse compact vector context
instead of spending tokens reloading unrelated goal and todo material.

## Bundle Flow

1. Keep the objective heap in markdown with `## GOAL-ID Title` records.
2. Run `ipfs-accelerate-agent-objective-daemon` or
   `generate_objective_todos(...)` with a main todo file, discovery directory,
   and objective bundle directory.
3. Optionally refine broad missing evidence into child goals with
   `--refine-objective-heap`.
4. The scanner appends daemon-parseable tasks to the main todo file.
5. It writes per-bundle shards under `objective_bundles/*.todo.md`.
6. It writes `objective_bundles/index.json`.
7. It writes `objective_bundles/todo_vector_index.json` and annotates the bundle
   index with merge/vector metadata.
8. It writes `objective_graph.json`.
9. It writes AST/symbol datasets under `objective_datasets/`.
10. `build_bundle_task_payloads(...)` converts the index into task queue payloads.
11. `submit_bundle_tasks(...)` submits each bundle to the existing
   `ipfs_accelerate_py.p2p_tasks.task_queue.TaskQueue` as `codex.todo_bundle`.
12. `ipfs-accelerate-agent-bundle-supervisor` plans one isolated
    implementation-supervisor lane per bundle shard and, with `--start`, launches
    those lanes with separate state directories and worktree roots.

This keeps the concurrency concern in `ipfs_accelerate_py`: agent lanes can be
scheduled locally, through P2P task queues, or by future accelerated agent
workers.

## Runnable Daemons

The reusable todo daemon and implementation supervisor are ported into
`ipfs_accelerate_py.agent_supervisor.todo_daemon`.  Installable entry points are:

- `ipfs-accelerate-agent-objective-daemon` for objective-heap scanning,
  AST/dataset persistence, bundle writing, and optional task-queue submission.
- `ipfs-accelerate-agent-backlog-refinery` for script-port backlog feed logic:
  refill low todo queues from an objective heap, scan tracked code for focused
  bug/improvement findings, and convert repeated validation or merge failures
  into evidence-backed follow-up tasks instead of indefinite retry loops.
- `ipfs-accelerate-agent-bundle-supervisor` for turning
  `objective_bundles/index.json` into isolated daemon lanes. Dry-run planning is
  the default; `--start` launches the lane supervisors.
- `ipfs-accelerate-agent-implementation-daemon` for draining markdown todo
  tasks with the Codex/Copilot implementation loop.
- `ipfs-accelerate-agent-implementation-supervisor` for monitoring, restarting,
  and repairing the implementation daemon.
- `ipfs-accelerate-agent-merge-resolver` for building a merge-conflict prompt
  from daemon events and, with `--apply`, invoking an external LLM resolver
  command.

The objective daemon suppresses duplicate work by reading existing discovery
fingerprints unless `--repeat-existing` is set.

## Backlog Refinery

`ipfs_accelerate_py.agent_supervisor.backlog_refinery` is the reusable port of
the repo-local supervisor feed behavior. It is separate from the implementation
daemon so product-specific wrappers can decide when to run it, while the
accelerator owns the actual policy.

The objective heap parser is package-neutral: callers can provide their own goal
ids, task prefixes, paths, discovery output roots, and summary prefixes. The
default tracking-document creator uses `OBJ-G###` ids, while existing heaps that
already use another numeric prefix keep that prefix when child goals are
refined.

The refinery has four modes:

- Objective scan: calls the objective graph scanner and appends missing-evidence
  tasks only when the backlog is low, forced, or fully drained. It updates the
  strategy file with seen fingerprints and writes the same bundle shards and AST
  dataset artifacts as the objective daemon.
- Codebase scan: scans tracked files across the repo and nested worktrees for
  small actionable findings such as unfenced TODO/FIXME annotations, swallowed
  exception paths, and placeholder runtime paths. Findings become parseable todo
  tasks with discovery evidence.
- Retry budget: reads daemon events and blocks source tasks that repeatedly fail
  implementation setup/runtime, validation, or merge reconciliation, then
  appends a follow-up task with the relevant logs, failed command, and
  repair/unblock instructions.
- Dependency guardrail: scans todo dependency metadata for missing task ids,
  self-references, and open-task cycles, then appends ready repair tasks with
  discovery evidence.

When no mode flag is passed, `ipfs-accelerate-agent-backlog-refinery` runs all
available modes. `--objective-scan`, `--codebase-scan`, `--retry-budget`, and
`--dependency-guardrail` select individual modes. The refill thresholds are controlled by
`IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_MIN_OPEN_TASKS`,
`IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MIN_OPEN_TASKS`, and the matching
`*_MAX_FINDINGS` and `*_COOLDOWN_SECONDS` environment variables. Use
`--discovery-output-path`, `--objective-summary-prefix`, `--task-prefix`, and
`--task-header-prefix` when embedding the refinery in another package's todo
format.

## Implementation Worktrees

The implementation daemon is package-neutral by default. It does not assume that
the host repository has Hallucinate-specific submodules. Callers that need
submodule-aware worktrees can pass `worktree_submodule_paths` to
`PortalImplementationDaemon`, repeat `--worktree-submodule-path` on the CLI, or
set `IPFS_ACCELERATE_AGENT_WORKTREE_SUBMODULE_PATHS` to a comma-separated list.
`PortalImplementationSupervisor` accepts the same `worktree_submodule_paths`
configuration and forwards it to managed implementation daemons.
The Hallucinate wrapper supplies its own `hallucinate_app`, `ipfs_datasets_py`,
and `swissknife` paths as adapter configuration.

## Drained-Backlog Refill

Supervisors can keep an implementation loop fed after the todo board drains by
enabling `--codebase-refill-scan`. On each supervisor health check the reusable
backlog refinery inspects the board state and only appends work when the open
task count is at or below `--codebase-scan-min-open-tasks`. When the count is
zero, the scan mode becomes `drained_exhaustive`, which walks the root checkout
and discovered git worktrees/submodules even if the normal cooldown has not
elapsed. The scan writes discovery reports and daemon-parseable follow-up tasks
for code annotations, swallowed exceptions, and placeholder runtime paths.

Enable `--objective-refill-scan` when the supervisor should also maintain the
durable goal graph. In that mode, a low or drained backlog causes the supervisor
to scan the objective heap, append bounded child goals for missing evidence, and
then generate bundle-local todos from the refined graph. Use
`--objective-scan-min-open-tasks`, `--objective-scan-cooldown-seconds`,
`--objective-max-refinement-children`, and `--objective-max-refinement-depth` to
control graph growth, and use `--objective-surplus-findings-per-goal` to create
extra mergeable todo candidates per goal when workers need a larger surplus
queue. `--no-objective-goal-refinement` keeps the scan in todo-only mode for
callers that want a fixed goal graph. `--no-objective-todo-vector-index` disables
the compact todo vector/AST index if a caller only wants plain markdown output.

The implementation supervisor also runs the retry-budget guardrail by default.
It converts repeated implementation exceptions, implementation timeouts,
validation failures, and merge failures into normal follow-up todos and adds the
source task to `blocked_tasks` so the daemon does not keep retrying the same
blocker. Use `--implementation-retry-budget`, `--validation-retry-budget`, and
`--merge-retry-budget` to tune the thresholds, or
`--no-retry-budget-guardrail` to disable this behavior for a lane.
Skipped merge reconciliation events, such as missing implementation branches,
are treated as merge retry-budget evidence. Cleanup failures after an
implementation commit already reached the target branch are not marked resolved;
they stay eligible for the next reconciliation pass until the worktree and
temporary branches are removed or a follow-up blocker is filed.

## Merge Conflicts

`ipfs_accelerate_py.agent_supervisor.merge_resolver` reads daemon JSONL events
and builds a dry-run LLM prompt for the latest failed merge.  The prompt includes
the task id, branch, target branch, dirty paths, unmerged paths, and compact
stdout/stderr excerpts.

Set `IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND` to invoke an external LLM
resolver command.  The default resolver path is dry-run only, so supervisors can
record the conflict payload before deciding whether to apply a semantic merge.
When that environment variable is present for the implementation daemon, failed
content merges invoke the resolver against the conflicted merge workspace before
the daemon aborts the merge. If the resolver clears all unmerged paths, the
daemon commits the resolved merge and records `llm_merge_resolver_invoked` plus
the merge result in the event log.

The same resolver path is also used for `main_checkout_dirty_conflict` blockers:
if dirty files overlap the implementation branch, the daemon records the dirty
paths, invokes the configured resolver, and retries the merge immediately when
the dirty overlap is cleared. Managed main-merge worktrees under the daemon's
worktree root also invoke the resolver for `main_merge_worktree_dirty` and retry
workspace preparation after the dirty state is cleared. Supervisors can pass
resolver settings directly to managed daemons with
`--llm-merge-resolver-command` and `--llm-merge-resolver-timeout-seconds`; the
timeout also honors
`IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS` and defaults to 600
seconds, with values less than or equal to zero disabling the timeout.

Submodule blockers use the same mechanism. During parent merge reconciliation,
the daemon enters the submodule checkout and invokes the resolver for
`submodule_checkout_dirty`, `submodule_default_branch_checkout_failed`, and
`submodule_merge_conflict` failures. It retries the blocked checkout or merge
after the resolver returns, records the prompt result on the submodule merge
entry, and accepts resolver-created merge commits when the implementation branch
is an ancestor of the submodule `HEAD`.

Implementation worktree setup failures and unexpected merge-reconciliation
exceptions are recorded as durable `implementation_exception` or
`merge_reconcile_exception` events instead of killing the daemon pass. This lets
the supervisor keep the loop running, rewrite strategy if progress stalls, and
feed retry-budget follow-up tasks with concrete blocker evidence.

If the supervisor sees stale active-task state that is no longer an active
implementation attempt, it rewrites strategy first, then records
`blocked_progress_state_repaired` and clears the stale active fields. This keeps
old active selections, abandoned worktree paths, or stale heartbeats from
blocking refill scans and later daemon passes.

Empty or missing todo boards are also non-fatal. The daemon records
`daemon_no_tasks` and updates state with zero ready/waiting/blocked tasks instead
of exiting. When objective or codebase refill is enabled, the supervisor creates
a skeleton todo board with `todo_board_created` before running refill scans, so a
fresh lane can bootstrap itself from the objective heap or codebase scan.

Validation commands are bounded and non-interactive. The daemon still records
timeouts as validation failures for retry-budget handling, but validation
subprocesses receive stdin from `/dev/null` so accidental prompts or commands
that read from stdin do not consume the daemon's own input stream.

Strategy state is repaired before supervisor guardrails run. If the strategy
file is missing, unreadable, invalid JSON, or has malformed list metadata, the
supervisor rewrites it to a normalized object, records `strategy_file_repaired`,
and continues. The daemon has the same repair path when it loads strategy
directly, so a corrupt `strategy.json` does not block every later pass.

Daemon state is repaired the same way. If `task_state.json` is unreadable,
invalid JSON, not an object, or contains malformed numeric/dictionary metadata,
the daemon or supervisor rewrites it to a valid empty state, records
`state_file_repaired`, and continues with a clean pass instead of repeatedly
crashing or losing progress visibility on the same corrupt state file.

Event logs are also self-repaired. If `events.jsonl` contains malformed JSONL
or non-object events, the valid events are preserved, invalid lines are moved to
a quarantine file, `event_log_repaired` is recorded, and retry-budget scans can
continue using the remaining evidence. If the configured event-log path is
accidentally a directory, it is moved aside and replaced with a writable JSONL
file before the daemon or supervisor records more events.

Todo boards are repaired before refill scans. If the configured todo path is a
directory, or if its text cannot be decoded as UTF-8, and objective/codebase
refill is enabled, the supervisor moves the bad path aside, writes a skeleton
todo board, records `todo_board_repaired`, and then allows refill to populate
new work. The daemon also treats decode failures as `todo_read_failed` instead
of crashing the pass.

Managed daemon PID files are repaired before adoption or restart. Invalid PID
text is moved aside, stale PID files are removed, command-line mismatches are
unlinked, and an accidental PID directory is moved to a timestamped backup. The
supervisor records `managed_daemon_pid_file_repaired`, so stale process metadata
does not block starting a fresh managed daemon.

The shared supervisor launcher also repairs generated marker paths before it
starts a child process. A directory in place of the child PID marker, child log,
or latest-log symlink is moved aside before launch, and stale symlinks are
removed. This prevents malformed runtime artifacts from crashing the supervisor
loop before the watchdog can supervise the daemon.

Shared JSON output writers also move directory-shaped targets aside before
writing supervisor status, ensure/check status, strategy files, or JSONL event
records. This keeps a bad generated artifact from breaking the next health
write, startup check, or retry-budget pass.

Stop cleanup is also tolerant of malformed PID markers. If the supervisor or
child PID marker path is a directory during stop/restart recovery, it is moved
aside instead of raising during cleanup, so restart orchestration can continue.

Child launch failures are supervised instead of crashing the supervisor loop.
If the daemon command cannot be launched, the loop records `launch_failed`,
waits according to the restart policy, and retries until the configured restart
budget is exhausted. Status-write failures are also isolated from the process
monitor so a transient status artifact problem cannot kill an otherwise healthy
supervisor loop.

Lock cleanup handles malformed lock paths. Stale implementation or merge lock
files are removed as before, and if a lock path is accidentally a directory, it
is moved aside to a timestamped backup path before retrying acquisition. The
daemon records the normal lock-cleared event with the moved directory path so a
bad lock cannot permanently return `lock_cleanup_failed`.

The supervisor also runs a dependency and board-metadata guardrail by default.
If open tasks depend on task ids that are not present on the board, on
themselves, on a closed cycle of open tasks, or if the board contains duplicate
task ids, it appends a ready repair task and writes discovery evidence with
`dependency_guardrail`. This turns no-ready-task dependency deadlocks and
ambiguous task metadata into normal backlog work. The malformed source task is
also added to `blocked_tasks` until the generated repair task is completed, so
workers do not act on known-bad metadata. Use
`--no-dependency-guardrail`, `--dependency-guardrail-discovery-dir`, and
`--dependency-guardrail-max-findings` to tune that behavior.

Guardrail blocks are released automatically once their generated repair task is
marked completed. The supervisor reads `retry_budget_findings` and
`dependency_guardrail_findings`, removes completed sources from `blocked_tasks`,
and records `guardrail_blocks_released`. This prevents source tasks from staying
blocked after the repair work has landed.

The same release pass also cleans stale strategy blocks that can no longer be
valid: duplicate entries in `blocked_tasks`, blocked task ids that no longer
exist on the todo board, and blocked source tasks already marked completed. This
prevents abandoned strategy state from permanently hiding ready work.

Merge-lock contention is treated as a deferred merge rather than a terminal
implementation failure. If a task validates and commits an implementation branch
but cannot merge because another daemon owns the repository merge lock, the
result remains eligible for the next reconciliation pass until the branch is
merged or explicitly abandoned.

## Relationship To `ipfs_datasets_py`

`ipfs_datasets_py` remains the upstream source of the existing todo daemon
framework. This module does not edit that package. The accelerator port now
contains its own runnable copy of the todo daemon/supervisor runtime plus the
objective graph, bundle planning, task-queue payloads, and merge-resolution
bridge that are specific to accelerating autonomous agent systems.

When `ipfs_datasets_py` is available, `ObjectiveDatasetStore` uses its
`DatasetManager` to register the AST dataset.  When it is not available, the same
scan still succeeds with JSONL and manifest artifacts, so accelerator workers can
run in minimal environments.

Some dataset-specific daemon families, such as the legal parser and logic port,
still bridge to `ipfs_datasets_py` logic modules when those specialized runtimes
are invoked. Core objective scanning and the generic todo daemon import without
that optional package.
