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

## Bundle Flow

1. Keep the objective heap in markdown with `## VAIOS-G*` records.
2. Run `generate_objective_todos(...)` with a main todo file, discovery
   directory, and objective bundle directory.
3. The scanner appends daemon-parseable tasks to the main todo file.
4. It writes per-bundle shards under `objective_bundles/*.todo.md`.
5. It writes `objective_bundles/index.json`.
6. `build_bundle_task_payloads(...)` converts the index into task queue payloads.
7. `submit_bundle_tasks(...)` submits each bundle to the existing
   `ipfs_accelerate_py.p2p_tasks.task_queue.TaskQueue` as `codex.todo_bundle`.

This keeps the concurrency concern in `ipfs_accelerate_py`: agent lanes can be
scheduled locally, through P2P task queues, or by future accelerated agent
workers.

## Merge Conflicts

`ipfs_accelerate_py.agent_supervisor.merge_resolver` reads daemon JSONL events
and builds a dry-run LLM prompt for the latest failed merge.  The prompt includes
the task id, branch, target branch, dirty paths, unmerged paths, and compact
stdout/stderr excerpts.

Set `IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND` to invoke an external LLM
resolver command.  The default resolver path is dry-run only, so supervisors can
record the conflict payload before deciding whether to apply a semantic merge.

## Relationship To `ipfs_datasets_py`

`ipfs_datasets_py` remains the source of the existing todo daemon framework. This
module does not edit that package and does not import from it at runtime.  The
ported code provides the objective graph, bundle planning, task-queue payloads,
and merge-resolution bridge that are specific to accelerating autonomous agent
systems in `ipfs_accelerate_py`.
