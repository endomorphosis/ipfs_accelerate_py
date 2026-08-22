# Objective Bundle: agent-supervisor/self-hosting/prerequisite-observation-snapshot-bounded-v12

Source todo: docs/architecture/self_hosting_qualification.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## SHQ-029 Close objective gap: Generate the post-merge prerequisite observation snapshot

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prerequisite-observation
- Depends on: SHQ-028
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode verify-existing --artifact artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json --quiet; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode require-terminal --quiet && exit 99 || test "$?" -eq 1; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode verify-existing --artifact artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json --quiet
- Board namespace: self_hosting_qualification.todo.md
- Evidence inputs: data/agent_supervisor/self_hosting_qualification/discovery
- Discovery evidence: data/agent_supervisor/self_hosting_qualification/discovery/2026-08-14-shq-029-objective-gap-925ecd59554c.md
- Bundle: agent-supervisor/self-hosting/prerequisite-observation-snapshot-bounded-v12
- Bundle shard: data/agent_supervisor/self_hosting_qualification/projections/v12/bundles/agent-supervisor-self-hosting-prerequisite-observation-snapshot-bounded-v12.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 3
- Parallel lane: prerequisite-observation-snapshot-bounded-v12
- Conflict policy: Do not edit `.gitignore`, observer implementation, tests, prerequisite owners, release admission, policies, keys or generated supervisor state; never read arbitrary host paths.
- Predicted files: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Changed paths:
- Context paths: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- AST symbols: PrerequisiteObservation observation_to_json write_observation_artifact
- Interfaces: PrerequisiteObservation@1
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Generated artifacts:
- Allow concurrent with:
- Goal id: SHQ-G007
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/2c78e15403f9df96456ebf815ddd1ba7c6f9c6af2a42be521f584fbb59c30a0d
- Canonical task CID: baguqeerafr4ocvad7hpzmrlox6av3xi3u7dptrvpfjbl4uq7lbh3wwodbigq
- Semantic identity: objective-evidence-obligation/v1/2b17e60a3516287ce21a56894825629c819e5da2113bd1262eb6e0cbd8e55d83
- Acceptance subset: This bounded task is neither resumable nor long-running, the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`, deliberately forward their values as task input or tool arguments, or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. Consume G006 only as the exact clean merged tracked bounded-v12 predecessor selected by the freshly generated SHQ-028 dependency CID, no predecessor runtime, receipt, log, worktree or coordination state is input. Every bounded-v11 SHQ-024 attempt 1 through 4 and the SHQ-025 registration, plus every v11 display ID/key/CID offered as a dependency, implementation log, disposable or sibling worktree, supervisor/checkpoint/runtime/coordination record, claim, lease, receipt as bytes, rejected code/test proposal, rescue or quarantine ref, operator quarantine bundle, cache and derived byte is a prohibited non-input and must not be inspected, enumerated, restored, copied, seeded, validated, cited as evidence, or used for retry. Change only `artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`, never edit `.gitignore`, observer code, tests, prerequisite owners, policies, keys or generated supervisor state. From the freshly committed clean v12 launch HEAD/tree, execute the merged ordinary-observe CLI exactly once as `python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode observe --quiet`, require its default canonical target to equal the sole declared output, and publish one structurally complete deterministic repository-relative ten-row snapshot through the already-tested durable dirfd/no-clobber publisher. Bind the pre-observation outer HEAD/tree, complete recursive gitlinks and matching submodule HEAD/trees, indexes, tracked-content digests and admitted current receipt authorities while excluding only the artifact path, require the stable observation manifest and degraded reasons satisfy `S1 == S0` through final source revalidation and canonical readback. Validation must never invoke ordinary observe again: run default read-only `verify-existing` before and after the no-output `require-terminal` rc1 probe, forbid the projection-child flag precommit, and prove all artifact bytes/stat and repository state remain unchanged. Refuse an existing target/symlink, dirty or source-raced input, any path/identity mismatch, partial/noncanonical rows, short I/O or publication failure, and leave no temp or partial target. The artifact declares that it is neither completion, proof nor release authority and that its later evidence-projection commit is not the observed source. On this host incomplete recursive closure or unavailable namespace isolation produces exact typed limitations, all ten rows, no receipt authority and truthful `terminal:false`, ordinary observe remains valid, while require-terminal returns 1 and performs no write unless the complete live terminal chain succeeds. Never initialize omitted submodules, access the network, manufacture closure, upgrade a clean fixture result into current-tree evidence, or consume v11 bytes.
- Preconditions: objective goal SHQ-G007 is schedulable
- Effects: satisfy evidence requirement: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Evidence subset: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Resource class: cpu-small
- Token class: small
- Estimated tokens: 0
- Resources: cpu-small
- Merge fate: objective/SHQ-G007
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/2b17e60a3516287ce21a56894825629c819e5da2113bd1262eb6e0cbd8e55d83
- Missing evidence: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Embedding query: post merge prerequisite observation snapshot clean source projection recursive gitlinks
- AST query: PrerequisiteObservation observation_to_json write_observation_artifact
- Surplus group: objective/SHQ-G007
- Merge key: fee3d856a0d71724
- Merge family: objective/SHQ-G007
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: bdca5f32aa0623f1
- Acceptance: Objective scan filed this gap for SHQ-G007. Use evidence in data/agent_supervisor/self_hosting_qualification/discovery/2026-08-14-shq-029-objective-gap-925ecd59554c.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json), and keep the supervisor-fed backlog aligned with the objective heap.  This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. This is the fourth autonomous bounded-v12 stage. The implementation actor is exactly the same pinned direct Terra/high route and the validator remains the deterministic CLI/readback matrix plus operator boundary review. The initial scheduler cap is `--max-task-attempts 1`, so it gets one semantic implementation attempt; attempt 2 requires exact changed typed transient setup/provider/resource/process evidence and an operator pre-invocation gate proving the prior receipt is typed transient with null output, coordination inactive/released, no active claim/lease/process/worktree/ref/lock, `implementation_attempts_by_cid[<exact canonical task CID>] == 1`, `selection_idle_reason == all_selectable_ready_tasks_reached_max_task_attempts`, no `implementation_retry_deferred:*` state or retry-budget-repair receipt, and fresh matching v12 HEAD/tree/route/protected envelope. Semantic rejection freezes and migrates rather than retrying, switching actors, broadening prompts, resetting counters or auto-reopening. The freshly projected SHQ-028 canonical task CID is the sole executable predecessor identity. Generate only the real snapshot from the clean merged G006 tracked tree, verify exact canonical readback and keep every implementation/runtime/history authority boundary closed.
