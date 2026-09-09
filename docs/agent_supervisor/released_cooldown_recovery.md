# Recover a blocked task with an older released cooldown

`QuackStateClient.recover_blocked_task_retry` accepts an optional `expected_released_cooldown` containing the complete observed typed cooldown row. It handles a pre-provider failure on a newer attempt while an older retry cooldown remains retained.

This path requires fresh Portal validation. The client and exclusive owner validate the complete prior row and typed receipt, same task and owner, released state, zero expiry, older attempt and task revision, and bounded prior fences. The owner compares the full persisted row to the sealed observation before effects. An exact revision/attempt CAS advances the queue revision in the same transaction as blocked-to-retrying, task history, generation, and idempotency records. Active, foreign, stale, malformed, or newer rows cannot be replaced. Missing rows continue through the original expected-absence insertion.

The retry preserves terminal evidence, refunds no attempt, admits exactly one fresh attempt, and installs the existing mandatory Portal validation obligation. It does not establish acceptance. Native operators retain their content-addressed authorization and prior row before submission. Ordinary executor grants cannot invoke operator recovery.

An operator can also rearm an exact unconsumed typed-deferral-budget block after repairing its prerequisite. That path requires fresh validation and preserves the failed budget in task history. The daemon independently binds the operator retry to the historical receipt and validation obligation before suppressing replay of that old exhausted budget. The native operator must establish the repaired prerequisite before requesting this task-scoped transition.
