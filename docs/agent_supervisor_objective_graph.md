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

1. Keep the objective heap in markdown with `## VAIOS-G*` records.
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

This keeps the concurrency concern in `ipfs_accelerate_py`: agent lanes can be
scheduled locally, through P2P task queues, or by future accelerated agent
workers.

## Runnable Daemons

The reusable todo daemon and implementation supervisor are ported into
`ipfs_accelerate_py.agent_supervisor.todo_daemon`.  Installable entry points are:

- `ipfs-accelerate-agent-objective-daemon` for objective-heap scanning,
  AST/dataset persistence, bundle writing, and optional task-queue submission.
- `ipfs-accelerate-agent-implementation-daemon` for draining markdown todo
  tasks with the Codex/Copilot implementation loop.
- `ipfs-accelerate-agent-implementation-supervisor` for monitoring, restarting,
  and repairing the implementation daemon.
- `ipfs-accelerate-agent-merge-resolver` for building a merge-conflict prompt
  from daemon events and, with `--apply`, invoking an external LLM resolver
  command.

The objective daemon suppresses duplicate work by reading existing discovery
fingerprints unless `--repeat-existing` is set.

## Merge Conflicts

`ipfs_accelerate_py.agent_supervisor.merge_resolver` reads daemon JSONL events
and builds a dry-run LLM prompt for the latest failed merge.  The prompt includes
the task id, branch, target branch, dirty paths, unmerged paths, and compact
stdout/stderr excerpts.

Set `IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND` to invoke an external LLM
resolver command.  The default resolver path is dry-run only, so supervisors can
record the conflict payload before deciding whether to apply a semantic merge.

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
