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

Large AST and symbol payloads are dataset artifacts.  The accelerator writes a
JSONL artifact and manifest for every objective scan, and when `ipfs_datasets_py`
plus its dataset dependencies are importable it also saves the rows through
`ipfs_datasets_py.dataset_manager.DatasetManager` and writes a parquet dataset.
This keeps bulky scan evidence out of markdown todos while still making it
available to dataset tooling, provenance systems, and future vector indexes.

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
7. It writes `objective_graph.json`.
8. It writes AST/symbol datasets under `objective_datasets/`.
9. `build_bundle_task_payloads(...)` converts the index into task queue payloads.
10. `submit_bundle_tasks(...)` submits each bundle to the existing
   `ipfs_accelerate_py.p2p_tasks.task_queue.TaskQueue` as `codex.todo_bundle`.
11. `ipfs-accelerate-agent-bundle-supervisor` plans one isolated
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

The refinery has three modes:

- Objective scan: calls the objective graph scanner and appends missing-evidence
  tasks only when the backlog is low, forced, or fully drained. It updates the
  strategy file with seen fingerprints and writes the same bundle shards and AST
  dataset artifacts as the objective daemon.
- Codebase scan: scans tracked files across the repo and nested worktrees for
  small actionable findings such as unfenced TODO/FIXME annotations, swallowed
  exception paths, and placeholder runtime paths. Findings become parseable todo
  tasks with discovery evidence.
- Retry budget: reads daemon events and blocks source tasks that repeatedly fail
  validation or merge reconciliation, then appends a follow-up task with the
  relevant logs, failed command, and merge-resolution instructions.

When no mode flag is passed, `ipfs-accelerate-agent-backlog-refinery` runs all
available modes. `--objective-scan`, `--codebase-scan`, and `--retry-budget`
select individual modes. The refill thresholds are controlled by
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
The Hallucinate wrapper supplies its own `hallucinate_app`, `ipfs_datasets_py`,
and `swissknife` paths as adapter configuration.

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
