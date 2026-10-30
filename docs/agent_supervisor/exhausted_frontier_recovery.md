# Exhausted frontier recovery

An unchanged ready frontier whose canonical paid attempts have reached the
configured limit is a blocker. A live supervisor heartbeat does not show that
this work can progress. Native first performs its ordinary maintenance pass,
including preserved-candidate reconciliation and release of completed repair
guardrails. If the same frontier remains spent, it reports
`task_attempt_budget_exhausted` and stops its exact owned child through the
existing sealed-child termination protocol. Active work, validators, protected
transactions, source changes and completion callbacks defer that stop. The
diagnostic preserves task CIDs and attempt counts; it authorizes no provider
call, attempt refund, task completion or replacement generation.

The supervisor's configured restart allowance covers all shared loops within
that supervisor lifetime. Typed child failures and later clean exits consume
the same allowance. Outer maintenance cannot obtain a fresh allowance by
constructing another shared loop. Exhaustion publishes `max_restarts_reached`
with the cumulative count and exits before another recovery restart.

Release requirements continue to require real evidence. On an exhausted
frontier, where an operator has explicitly enabled objective completion
reconciliation and configured an artifact producer plus gate/evidence paths,
Native tries that fixed argv once per supervisor incident and source identity,
under its ordinary maintenance lease and configured timeout. It then
independently reconciles the gate and evidence artifacts and re-reads the task
frontier. An unchanged spent frontier still stops. A timeout, partial artifact
or failed gate cannot restart the producer for the same incident. The default
unconfigured path executes no producer. The command comes from trusted configuration,
never from artifact contents, and executes with `shell=False`. Failure is a
failed refresh rather than completion. This does not enable objective refill,
generate extra tasks for a fixed campaign, alter gate flags, reset budgets or
turn an external dependency into an indefinite retry. A campaign's declared
final acceptance producer and Native completion verifier must still prove its
release requirements against the current source.

The focused regression module
`test/api/test_supervisor_exhausted_frontier_terminal.py` exercises the real
control-plane state store, ordinary maintenance and harmless sealed children.
Its source-change fixture checks the race with a completion status update; it
does not claim to validate a product candidate or satisfy a release gate.
