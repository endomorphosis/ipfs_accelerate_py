# Prompt entrypoint baseline

Status: executable inventory for `ASE-001`
Scope: current checkout before prompt-only entrypoint implementation

This baseline records the friction that a prompt-only supervisor entrypoint
must remove. The companion test,
`test/api/test_agent_supervisor_prompt_entrypoint_baseline.py`, derives each
claim from the current parsers, control catalog, service capability report, and
packaging metadata. It is deliberately an inventory of the current tree, not a
recommended interface.

## Current operator path

The installed path is:

```text
ipfs-accelerate
  -> ipfs_accelerate_py.cli_entry:main
  -> ipfs_accelerate_py.cli:main
  -> ipfs-accelerate agent COMMAND
  -> control.control_cli.run_agent_cli
  -> SupervisorControlService
```

`ipfs-accelerate agent` publishes all 31 canonical control operations and 15
usage-governance commands. Prompt workflow names are present, but an ordinary
direct request must bind all nine target fields:

1. `repository_root`
2. `state_root`
3. `repository_id`
4. `tree_id`
5. `objective_id`
6. `objective_revision`
7. `policy_id`
8. `policy_revision`
9. `caller`

A real mutation additionally requires a complete authorization decision,
idempotency key, lease ID, fencing epoch, and expected effects. A canonical
request file can carry those fields, but there is no current inference layer
that constructs them from a prompt and repository state.

## Default runtime gap

The product CLI constructs `SupervisorControlService` with
`RepositorySupervisorBackend` unless an embedding injects another service or
factory. Its capability report contains only the 12 local read operations:

```text
artifact_query, bundles, cache_inspect, capabilities, events, goals, health,
lanes, metrics, receipts, status, tasks
```

Neither `workflow_preview` nor `workflow_materialize` is registered. The
existing `PromptSupervisorService` therefore does not become the product CLI's
prompt handler by default. Supplying a prompt-shaped command name is not enough
to construct the scanner, planner, admission service, materializers, lifecycle
runtime, or implementation supervisor.

## Prompt-body handoff gap

The unified CLI correctly keeps raw prompts out of durable
`OperationRequest.parameters`. It hashes an inline, file, or stdin body and
retains only:

```json
{"kind": "inline", "content_cid": "<CIDv1>"}
```

That is the correct persistence boundary, but the default dispatch path has no
transient body broker or artifact loader that can resolve the CID for
`PromptSupervisorService`. The body is discarded before dispatch and no prompt
handler is installed. The new entrypoint needs a bounded, process-local or
content-addressed handoff while continuing to keep the body out of receipts,
logs, and durable control parameters.

## `--start` mismatch

Two prompt CLI adapters currently disagree:

- `ipfs-accelerate agent workflow-create --start` parses the flag but does not
  add `start_after_materialize` to the operation parameters.
- The standalone
  `python -m ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow`
  adapter does add `start_after_materialize`.
- The closed `workflow_materialize` control schema does not admit that field.

Consequently one adapter silently drops start intent while the other constructs
a parameter that the shared request contract rejects. Start needs to be a
first-class saga step with its own authorization and receipt, not an
adapter-specific parameter overlay.

## Low-level launch burden

The current legacy implementation entrypoints expose the following numbers of
long options:

| Entry module | Operator options | Including `--help` |
| --- | ---: | ---: |
| `todo_daemon.implementation_supervisor` | 139 | 140 |
| `todo_daemon.implementation_daemon` | 56 | 57 |

These remain useful expert interfaces. A prompt-only facade should infer a
reviewable launch profile and call the package-level builders instead of
requiring an operator to reproduce these options.

## State-root divergence

Current defaults do not identify one run namespace:

| Surface | Default |
| --- | --- |
| Objective daemon | `<repo>/data/agent_supervisor` |
| Bundle scheduler | `<repo>/data/agent_supervisor/bundle_lanes` |
| Implementation supervisor/daemon | `data/portal_implementation/state` relative to CWD |
| Unified control and prompt CLIs | no default; `--state-root` is required |

Automatic resolution must select one repository-bound state namespace, record
the source and confidence of that selection, and refuse ambiguous existing
runs rather than quietly starting another namespace.

## Installed surfaces

Packaging installs the product CLI plus eight low-level
`ipfs-accelerate-agent-*` commands:

```text
ipfs-accelerate
ipfs-accelerate-agent-objective-daemon
ipfs-accelerate-agent-backlog-refinery
ipfs-accelerate-agent-bundle-supervisor
ipfs-accelerate-agent-artifact-query
ipfs-accelerate-agent-implementation-daemon
ipfs-accelerate-agent-implementation-supervisor
ipfs-accelerate-agent-merge-resolver
ipfs-accelerate-agent-llm-merge-resolver-fallback
```

The prompt workflow has a `python -m` surface and an operations wrapper, but no
installed prompt-first console script. More importantly, the installed product
CLI does not wire a standard live prompt runtime.

## Executable acceptance probes

Run:

```bash
python -m pytest \
  test/api/test_agent_supervisor_prompt_entrypoint_baseline.py -q
```

The probes fail if the inventory silently drifts. When later tasks intentionally
close a gap, they should update the corresponding assertion and this document
in the same change so the baseline becomes an implementation progress ledger.
