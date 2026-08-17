# MCPP-086 Validation Retry-Budget Finding: MCPP-005

Date: 2026-08-15
Source task: MCPP-005
Follow-up task: MCPP-086
Retry budget: 3
Observed consecutive validation failures: 3

## Evidence

- Failed command: `cd ipfs_accelerate_py/mcplusplus/tests-rs && cargo test`
- Attempts: 1, 2, 3
- Logs: lane-5 implementation_logs mcpp-005-attempt-1.log .. attempt-3.log
- Validation attempted: `True`
- Validation return code: `101`
- Validation error: `validation_command_failed`
- Validation reason: `declared_validation_failed`

Root cause: hermetic cargo used repository `.cargo/config.toml` default
`build.target = aarch64-pc-windows-msvc` without `CARGO_BUILD_TARGET`, so the
Windows target was selected and `core` was missing.

## Guardrail Result

The accelerator backlog refinery classified this as backlog work and filed
MCPP-086 to repair the validation blocker and deliver the baseline receipt.

## Repair resolution (MCPP-086)

- Status: completed (repair attempt 2; attempt 1 lost commit handoff)
- Declared gate: `cd ipfs_accelerate_py/mcplusplus/tests-rs && cargo test` → pass (191/0)
- Receipt: `docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-rust.json`
- Detail: `2026-08-15-mcpp-086-repair-complete.md`
