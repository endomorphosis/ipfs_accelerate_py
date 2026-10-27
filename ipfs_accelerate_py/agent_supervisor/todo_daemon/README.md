# agent_supervisor.todo_daemon

**Layer:** Edge · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Executable implementation and supervisor daemons, git/worktree helpers, and board-drain loops.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | most domain packages it orchestrates (top of DAG) |
| **Typical dependents** | ops scripts, multi-supervisor entrypoints |

## Modules

| Module | Path |
| --- | --- |
| `app` | `todo_daemon/app.py` |
| `artifacts` | `todo_daemon/artifacts.py` |
| `auto_commit` | `todo_daemon/auto_commit.py` |
| `campaign_resume` | `todo_daemon/campaign_resume.py` |
| `cli` | `todo_daemon/cli.py` |
| `context` | `todo_daemon/context.py` |
| `core` | `todo_daemon/core.py` |
| `deterministic_fallback` | `todo_daemon/deterministic_fallback.py` |
| `diagnostics` | `todo_daemon/diagnostics.py` |
| `engine` | `todo_daemon/engine.py` |
| `file_replacement` | `todo_daemon/file_replacement.py` |
| `git_utils` | `todo_daemon/git_utils.py` |
| `history` | `todo_daemon/history.py` |
| `implementation_daemon` | `todo_daemon/implementation_daemon.py` |
| `implementation_daemon_runner` | `todo_daemon/implementation_daemon_runner.py` |
| `implementation_supervisor` | `todo_daemon/implementation_supervisor.py` |
| `implementation_supervisor_runner` | `todo_daemon/implementation_supervisor_runner.py` |
| `legal_parser` | `todo_daemon/legal_parser.py` |
| `legal_parser_daemon` | `todo_daemon/legal_parser_daemon.py` |
| `lifecycle_wrapper` | `todo_daemon/lifecycle_wrapper.py` |
| `llm` | `todo_daemon/llm.py` |
| `llm_defaults` | `todo_daemon/llm_defaults.py` |
| `logic_port` | `todo_daemon/logic_port.py` |
| `plans` | `todo_daemon/plans.py` |
| `process_tree_fencing` | `todo_daemon/process_tree_fencing.py` |
| `registry` | `todo_daemon/registry.py` |
| `runner` | `todo_daemon/runner.py` |
| `specs` | `todo_daemon/specs.py` |
| `status` | `todo_daemon/status.py` |
| `supervisor` | `todo_daemon/supervisor.py` |
| `supervisor_loop` | `todo_daemon/supervisor_loop.py` |
| `supervisor_runtime` | `todo_daemon/supervisor_runtime.py` |
| `task_board` | `todo_daemon/task_board.py` |
| `typescript` | `todo_daemon/typescript.py` |
| `worktrees` | `todo_daemon/worktrees.py` |
| `wrapper` | `todo_daemon/wrapper.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.todo_daemon.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

## Ordered provider fallback policies

`grok-codex` provider aliases use the no-shell
`agent_supervisor/provider_fallback_runner.py`, but the canonical
`ipfs_accelerate_py.llm_router` owns readiness, failure classification, and the
fallback decision. The fixed agent policy is
`IPFS_ACCELERATE_AGENT_PROVIDER_FALLBACK_POLICY=grok_quota_auth_or_unavailable`:
Grok remains primary, while Codex may run only for a typed quota exhaustion,
authentication failure, or launch unavailability before side effects and with
an unchanged candidate workspace. Timeout, transport, generic nonzero,
malformed-output, ordinary task failure, unknown activity, or workspace
mutation remains terminal. The adapter sanitizes both output streams and emits
bounded, task/attempt/stage-bound route telemetry with
`completion_authority=false`; normal proposal, validation, and merge gates
remain authoritative.

This explicit agent-CLI route is separate from `llm_router.generate_text`.
Generic agent/tool requests continue to prohibit cross-provider retries after
side effects.

## Consumed attempts after a runtime reload

An ordinary validation failure, test timeout, proposal failure or changed
control-plane source revision does not grant another attempt budget for the
same canonical task. Historical `ordinary_task_validation_obsolescence_rearmed`
events and `validation_obsolescence_rearm_receipts` remain evidence; the generic
source-revision reset entrypoint no longer clears either attempt-count map.
Completed board-bound retry repairs and their once-only durable receipts,
generated-repair recovery, and separately verified stale-proposal or
undispatched-provider recovery retain their existing owners and gates.

After the whole lane is quiescent, attempt-limit partitioning can restore a
minimum known consumed count for an exact CID with nonempty persisted
`validation_obsolescence_rearm_receipts`. It uses the existing strict lifecycle
journal verifier, matching each persisted fingerprint, canonical CID, policy
revision and exact integer count fields. Protected same-attempt reuse is checked
through its existing latch/finish owners using those same verified rows, before
any floor mutation. A later exact completed-repair reset supersedes older floors
and clears only its exact-CID derived fingerprint cache, recording the superseded
values in its original reset event so future repaired retries survive archive
eviction. Restoration changes the existing count maps only;
it never relabels the historical local ordinal, reconstructs a lifetime total,
refunds an attempt or admits a candidate. Active task identities, phases and
in-progress owners are exempt from this acquisition and mutation.

Only this floor caller supplies a total evidence limit of **64 selected files
and 32 MiB**, including an existing journal manifest. The optional verifier
preflight runs before manifest rebuilding, body replay or cache reuse and
rejects unsafe aliases and growth beyond the limit. Existing callers keep
their previous verification semantics and independent limits. This evidence
acquisition budget does not change CPU, memory, retry, provider, validation
runtime or model budgets. The motivating retained lane journal was
15,914,019 bytes plus a 1,504-byte manifest; its eventual post-closeout size must
be checked again rather than inferred from that earlier observation.

Missing, malformed, mismatched or over-budget proof returns the affected CID
as `validation_rearm_floor_evidence_unverified` through the existing partition,
with no counter mutation or provider dispatch. Other receipt-free task
identities do not scan or depend on that legacy history. The persisted field
belongs to `PortalTaskState` and its original DuckDB/Quack save/load owners;
an absent plain JSON snapshot does not prove that the native field is absent.
If an affected legacy CID genuinely lacks its receipt, a separate original-owned
quiescent accounting migration is required before its release. This path does
not infer an absent receipt or manufacture that migration authority.

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/todo_daemon.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
