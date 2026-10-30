# Isolated supervisor repair

Campaign repair is one deduplicated task for one reproduced native defect. Grok reaches that task only through `ipfs_accelerate_py.llm_router`. The edit lands only in a campaign-owned repair worktree, and only on the new reproducer plus the source frames named by the defect. Regressions and a canary run before promotion. Promotion is a native merge, followed by a reload and a fence. Rollback evidence stays with the incident.

The adapter is `ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_repair`. It spends the WWL-100 progress budgets and uses the WWL-003 explicit agent route. It does not launch a hosted model and it does not grant completion authority.

## Deduplicated repair task

The repair task id is `WWL-REPAIR-` plus the first sixteen hex digits of the durable incident id. That id binds the run, the task, the canonical stage, and the normalized defect. A new PID, attempt number, heartbeat, or log line does not open another task.

One incident has one open writer. A second action while that writer is open is a duplicate and does not spend another repair. Replaying the same action id returns the stored decision. After a reloaded promotion, a later action is already promoted and is not merged again.

Unknown effects, a hold (`HOLD`, `STOP`, `OPERATOR_STOP`), or a deadline that has already passed admit no repair mutation and do not spend a budget.

## Grok through llm_router

`bind_repair_route` calls `bind_explicit_agent_route`. The admitted route is `grok_cli` / `grok-4.7` through `provider_fallback_runner.py`, with emergency policy `grok_quota_auth_or_unavailable`. The argv is built and not launched. `paid_provider_calls` and `hosted_model_invocations` stay 0. The interpreter on that argv is `/usr/bin/python3.12`.

Every workspace flag on the runner, the Grok command, and the Codex command must be the campaign repair worktree. A route that points at the live checkout is refused. Sandbox bypass and a generic `generate_text` call are refused. A plan that is not the canonical route is refused before a writer opens.

## Campaign-owned repair worktree

The worktree is `<campaign-root>/repair-worktrees/<repair-task-id>`. The campaign root and the live checkout are disjoint directories. The worktree carries `.wwl-repair-owner.json` with this run id and task id. A symlink, a missing marker, or another run's marker is refused. Edits are written only under that directory, with symlink leaves rejected.

## Permitted paths and the reproducer

The campaign watchdog scope is necessary and not sufficient. An edit is admitted only when it is one of:

- the new reproducer, matching `test/api/test_worldwide_legal_search*.py`, absent from the baseline, and naming the defect
- a source frame from the reproduced failure whose path matches the admitted scope and whose baseline digest still matches
- the evidence file the session itself writes under `artifacts/worldwide-legal-search/repairs/`

The reproducer is written before any source frame changes. The source text must differ from the baseline. Existing tests, completion modules, watchdog policy, product acceptance, credentials, and law text are protected. An edit that grants completion authority or skips the reproducer is refused. A validation or canary step that changes any other worktree byte is refused and is not merged.

## Regressions and canary

The regression report has to show that the reproducer failed on the baseline, that the candidate and the focused regressions passed, that protected hashes were unchanged, and that the argv and exit status are the admitted reproducer command. The canary has to verify that same red/green pair inside the isolated worktree. The canary cannot set `validated_promotion`, spend a hosted call, or grant completion authority. Either failure stops the pipeline before merge. The validation budget stays spent.

## Native merge

The merger must return `merged: true`, a new git commit, a `sha256` merge receipt, a `sha256` tree, and `effects_unknown: false`. A zero exit without that receipt is not promotion. Unknown merge effects fence the worktree, skip reload, and keep the baseline as the rollback revision. The merge budget is spent when the merger is invoked.

## Reload and fence

After an accepted merge, the reloader must report the lane reloaded and fenced at that merge commit. The session then fences the worktree marker and increments the fence generation. A failed reload keeps the merge receipt and the baseline rollback, and it blocks another repair writer for that incident. A later action with phase `reload` may retry the reload. It does not open a second edit and it does not refund the repair budget.

## Rollback evidence

Every terminal decision appends a `wwl.repair.rollback@1` record. The record keeps the baseline revision, the worktree, the fence generation, and, when a merge was accepted, the promoted revision and merge receipt. A successful reload also records the baseline and the failure receipt as invalidated evidence. Records are not deleted. A checkpoint that lowers the spent repair count, or drops the rollback list for a finished task, is rejected.

## Budgets

| Budget | Limit | When it is spent |
| --- | --- | --- |
| repair | 2 per incident | a writer opens |
| provider | 4 per incident | the routed edit is applied |
| validation | 3 per incident | regressions and the canary run |
| merge | 3 per incident | native merge is attempted |
| campaign repairs | 8 | each opened writer, across incidents |

These repair, provider, validation, and merge limits are the WWL-100 ledger limits. A bound ledger with any other limit is refused. Noise, a checkpoint, and a reload retry do not refund a spent incident. Exhaustion is a blocker. The campaign cap is eight opened writers.

## Limitations

The binding constructs the canonical explicit-agent argv and applies admitted edits inside the repair worktree. It does not start Grok, Codex, or a hosted model. Native merge and reload are injected authorities; this module checks their receipts and does not invent a git commit or a lane restart. A checkpoint reloads spent counts, fences, and rollback records. It does not resume an open writer. This document does not grant completion authority. `artifacts/worldwide-legal-search/receipts/WWL-102.json` does not grant campaign completion either.
