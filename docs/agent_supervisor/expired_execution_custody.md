# Expired production execution custody

A running production or native Portal attempt requires settlement before a
generic expiry pass can retire it. Elapsed coordination authority and missing
provider/effect result rows do not establish that its callback never ran.
This runtime has no durable pre-dispatch journal that could prove that case.

The existing exact native completion and terminal-failure reconciliation paths
run first. If an unresolved real execution remains running with expired
authority or missing claim history, the expiry pass raises a typed custody
diagnostic. `run_once` ends that tick with
`selection_idle_reason=expired_attempt_settlement_unavailable`, retains the
attempt, and grants no retry, coordination mutation, or completion authority.
Historical provider dispatch remains `unknown`. Completed reconciliation
results from earlier in the tick remain visible in `recovery_prefix`.

Each subsequent tick rereads the native settlement state. A proved native
terminal settlement may resolve its exact attempt through the existing path.
Synthetic execution without a native Portal callback binding retains its
existing expiry behavior. A provider or effect result by itself does not grant
a production expiry retry while the attempt is still unresolved.

The regression tests execute a real disposable callback effect, expire its
lease before the outer result is persisted, and verify repeated ticks neither
repeat the callback nor alter the retained native rows. Additional cases cover
each unfinished phase, accepted and expired claim rows, missing claim history,
explicit Portal binding, malformed diagnostics, and completed earlier
reconciliation reporting.

This guard does not infer settlement for SAWM's retained attempt 4. It prevents
a historical attempt-2 repair from exposing that unknown callback to generic
expiry and retry. Live source adoption and exact callback diagnosis remain
separate operational steps.
