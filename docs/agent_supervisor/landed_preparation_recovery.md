# Prepared completion after landed recovery

An independently admitted typed legacy-orphan landed completion can finish the
canonical control task while its former lane still retains an expired PREPARED
barrier. The ordinary prepared-CAS receipt and the landed recovery receipt are
different protocols.

`DatabaseCoordinator.recover_prepared_task_completion` accepts the landed
receipt only from authoritative control truth, for the exact expired claim,
lease, attempt, fences, task alias and immediate admission revision. It checks
the closed recovery receipt, dead historical admission, landed proof digest and
proof attempt/validation bindings. It retains the original preparation and its
evidence digest, and records the distinct recovery digest in the control summary.
It never adds a fabricated preparation to the canonical task receipt.

Live ordinary completion still requires its original preparation binding.
Cross-store guards and current ownership checks remain mandatory. Unknown
receipts, newer revisions and foreign attempts leave the barrier unsettled.
This is reconciliation of existing accepted truth; it cannot independently
accept a task, retry a provider, or certify board completion.

Regression coverage is in `test_agent_supervisor_database_coordination.py`,
including stale identity/revision/proof rejection, guard enforcement, immutable
receipt preservation and promoted replay. Existing daemon prepared/promoted
restart tests cover the surrounding execution projection protocol.
