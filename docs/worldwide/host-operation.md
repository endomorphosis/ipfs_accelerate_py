# Host operation for the watchdog loop

The watchdog loop and the native supervisor are separate processes. The loop keeps its JSON state. The supervisor keeps the published recovery, the incident claim, and a restart journal. Either process can be restarted while a recovery is still in flight. The restarted pair loads those records and does not open a second writer.

The adapter is `ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog`. It does not open the task database and it does not grant completion authority.

## In-flight recovery

A recovery is in flight when the supervisor has published the action and has not settled it. The loop's copy of that action sits in the outbox, the inflight map, or the uncertain map. `dispatch` writes the action before it calls the host lifecycle. It then seals `watchdog/restart-journal.json` to the loop file `watchdog/state.json` when that file is already present.

The seal stores the action identity, the incident's reconcile, recover, diagnose and repair counts, the campaign action and repair counts, `next_action_at`, and any `HOLD`, `STOP`, or `OPERATOR_STOP` marker that exists at that moment.

## What survives restart

Restarting the supervisor means constructing the bridge again on the same state root. Restarting the loop means reading the same `watchdog/state.json`. `resume_after_restart` checks both and does not charge another attempt.

| Record | Survival rule |
| --- | --- |
| Deduplication | The first action id for the incident remains the only writer. Replaying it returns the stored result. A different action id is refused. |
| Cooldown | `next_action_at` must be the sealed time or later. An earlier time is refused. |
| Attempt accounting | Incident and campaign counters must be the sealed counts or higher. A lower count is a refund and is refused. |
| Stop and hold | A marker that was present at the seal must still be on disk. A marker that appears later still blocks dispatch. Restart does not delete either one. |
| Admitted tasks | The loop's required task set cannot drop a task the seal recorded. |

A healthy observation may reset the confirmation streak inside the loop. That reset does not lower these counters and it does not move the cooldown earlier. The resume report leaves `attempts_refunded`, `cooldown_reset`, `hold_cleared`, and `completion_authority` false.

## Optional systemd guardian

The guardian is optional. Rendering a unit does not install it, enable it, or start it. The only unit it is allowed to start is `worldwide-legal-search-watchdog.service`.

The unit runs `watchdog_loop.py` with this campaign's adapter argv, this campaign's `watchdog/state.json`, and `--execute`. `ExecStartPre` runs the adapter operation `guardian-start-check`. There is one `ExecStart`. The supervisor module and `worldwide-legal-search-supervisor.service` are not in the unit. `Restart=on-failure` and `StartLimitBurst=3` over `StartLimitIntervalSec=1800` bound systemd's own restarts. A successful exit is not a failure, so a verified completion does not come back as another loop.

`revive` starts that unit only after the start check passes. The check fails closed when:

- `HOLD`, `STOP`, or `OPERATOR_STOP` exists under the state root
- three guardian starts already occurred inside the current 1800 second window
- the shown unit id is not `worldwide-legal-search-watchdog.service`
- the fragment is missing, unowned, a symlink, or different from the rendered unit
- the registered loop already has a live main process

A start that is refused does not spend the restart budget. A start that is attempted spends one entry before `systemctl --user start` runs, including when that command fails. The spent entry is not refunded. The command is only `systemctl --user start worldwide-legal-search-watchdog.service`.

The canonical interpreter on the rendered unit is `/usr/bin/python3.12`.

## Limitations

The journal seals a loop file that the loop has already written. It does not invent attempt counts for a dispatch that has no loop file yet. Resume then still requires the loop file and the published action.

A sealed hold marker has to remain on disk. Deleting the file is not treated as a survived hold. This adapter does not clear operator markers.

The guardian does not install or enable the user unit. An operator can place the rendered text at `~/.config/systemd/user/worldwide-legal-search-watchdog.service`. Until that fragment matches this registration, the start check refuses to revive anything.

Native host recovery remains an injected authority in tests. A published unknown recovery is in flight; it is not a settled fence and it is not task success. This document does not grant completion authority. `artifacts/worldwide-legal-search/receipts/WWL-105.json` does not grant campaign completion either.
