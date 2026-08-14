# SHQ-027 Objective Goal Gap

Date: 2026-08-14
Fingerprint: 7e7ccb2d115ebb66e0df5bebc93c9ebe0cbc3a0e
Goal id: SHQ-G006B
Goal title: Add the isolated live receipt/cache terminal chain
Objective heap: docs/architecture/self_hosting_qualification.objectives.md
Priority: P0
Track: prerequisite-observation-terminal-chain
Status: todo
Schedulable: true
Review only: false
Parent goals: SHQ-G005
Graph depth: 2
Objective heap index: 1
Bundle: agent-supervisor/self-hosting/prerequisite-observer-terminal-chain-bounded-v12
Parallel lane: prerequisite-observer-terminal-chain-bounded-v12
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: isolated live process runner compiler direct execution observation test receipt production cache terminal chain
AST query: VerificationProcessRunner VerificationIdentityCompiler DirectExecutionObservation TestReceipt VerificationReceiptCache
Conflict policy: Edit only the clean merged observer/test predecessor. Do not edit `.gitignore`, add publication or CLI artifact writes, generate the artifact, consume predecessor runtime, or access any v11 attempt/quarantine material.
Predicted files: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
AST symbols: VerificationProcessRunner VerificationIdentityCompiler DirectExecutionObservation TestReceipt VerificationReceiptCache
Interfaces: verification.contracts.VerificationIdentityCompiler, verification.process_runner.PROCESS_RUNNER_SCHEMA, verification.process_runner.VerificationProcessRunner, verification.process_runner.VerificationCommand, verification.process_runner.VerificationStreamArtifact, validation.validation_runtime.build_hermetic_validation_runtime, validation.validation_runtime.hermetic_validation_command, verification.contracts.TestReceipt@1, verification.contracts.DirectExecutionObservation@1, verification.receipt_cache.VerificationReceiptCache
Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/c9eede0b3b23682bbf5931fdcd7027fe2cbaccd5d6f2c033fe698cf19bc62ffc
Acceptance subset: This bounded task is neither resumable nor long-running, the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`, deliberately forward their values as task input or tool arguments, or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. Consume G006A only as the exact clean merged tracked bounded-v12 predecessor selected by the freshly generated dependency CID, no predecessor log, worktree, runtime, coordination record, claim, lease, receipt file, temp, cache or other outside-checkout state is input. Apply the same absolute v11 prohibition as G006A to every SHQ-024 attempt, SHQ-025 registration, log, runtime/checkpoint/coordination record, receipt, quarantine bundle, rejected proposal and derived byte. Enter the terminal chain only after clean complete `S0`: derive inner argv exactly `(sealed_python, '-m', 'pytest', '-q', *selectors)` and bind its exact `shlex.join` value, require that inner argv to equal the exact suffix after `--` in the outer Bubblewrap argv. Build the runtime exactly once and only through `build_hermetic_validation_runtime` and `hermetic_validation_command`, requiring `--unshare-net`, read-only host binding, a bounded writable checkout, private `/tmp`, and no fallback. Bind actual bwrap bytes `b"bubblewrap 0.9.0\n"` and live isolated Python/pytest probe bytes. Call exactly one live same-process `VerificationProcessRunner.run(VerificationCommand)`, require `schema == PROCESS_RUNNER_SCHEMA`, `process_started is true`, `disposition == completed`, `exit_code == 0`, `result.ok is true`, and `publication_allowed is true`, with no timed-out, cancelled, unavailable, simulated, or replayed result. Before any structural projection, require the live result's `executable`, `cwd`, `environment`, `sandbox`, `network_policy`, `timeout_seconds`, `disposition`, `command_argv`, process/lease identity, and stdout/stderr stream fields to equal their corresponding authoritative values from the exact `VerificationCommand`, hermetic runtime, and observed process. Only after that run call `VerificationIdentityCompiler.compile_key` with `receipt_kind=TEST`, `adapter_schema=PROCESS_RUNNER_SCHEMA`, `selector_argv` equal to the exact outer argv, the resolved bwrap executable, and `tool_name='bwrap'` (never pytest), and require the compiled key to match its corresponding live command/tool/environment identities. Construct `DirectExecutionObservation` only from its actual contract fields: `receipt_key_cid`, `repository_tree_cid`, `environment_cid`, `repository_tree_observation`, `environment_observation`, `terminal_status`, `duration_ms`, `exit_code`, `stdout_artifact_cid`, `stderr_artifact_cid`, `artifact_cids`, and `reason_codes`, require those receipt/tree/environment identities, status, argv, duration, exit code, output CIDs, artifacts, and reasons to equal the compiled key and live result. Each stream must be nontruncated with `captured_byte_count == byte_count == len(preview.encode('utf-8'))`, and the preview bytes must rehash to both the declared digest and CID. Require exact `TestReceipt.from_dict(receipt.to_record()).to_record() == receipt.to_record()`, `admit(..., for_production=True, require_production_eligible=True)`, and `lookup(key, for_production=True)`, then require `S1 == S0`. Reject missing isolation, injected phase reports, cache-only authority, simulated results, and replayed results. Missing isolation or degraded forest remains typed nonterminal and skips the chain. Positive terminal proof is confined to a self-contained clean temporary Git fixture whose authority never upgrades the actual checkout.
Preconditions: objective goal SHQ-G006B is schedulable
Effects: satisfy evidence requirement: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, satisfy evidence requirement: test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
Evidence subset: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
Dependencies: SHQ-G006A
Resource class: cpu-small
Token class: medium
Estimated tokens: 0
Resources: cpu-small
Merge fate: objective/SHQ-G006B
Rejection reasons: none (accepted)

## Goal

From only the clean merged bounded-v12 G006A tracked predecessor, implement and independently test the terminal-only isolated live process-runner to identity-compiler to direct-observation to canonical `TestReceipt` and production receipt-cache chain. Do not add atomic publication, the ignore exception, CLI artifact writing, or the observation artifact in this stage.

## Missing Evidence

- scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py
- test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py

## Present Evidence

- none found for this goal

## Suggested Handling

Add and independently test only the exact terminal isolated live verification/receipt/cache chain over the merged G006A core.
