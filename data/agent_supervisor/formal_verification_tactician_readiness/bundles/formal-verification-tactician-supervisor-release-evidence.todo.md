# Objective Bundle: formal-verification-tactician/supervisor-release-evidence

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-063 Close objective gap: Bind durable supervisor evidence and enforce expected outputs

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: supervisor-integrity
- Depends on: FVT-035, FVT-064
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/merge/leased_lane.py, ipfs_accelerate_py/agent_supervisor/release_evidence.py, test/api/test_agent_supervisor_release_evidence_binding.py
- Validation: python -m pytest test/api/test_agent_supervisor_release_evidence_binding.py test/api/test_agent_supervisor_todo_daemon_port.py -k 'expected_output or completion_receipt or release_evidence' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-063-objective-gap-a119b5b3e02b.md
- Bundle: formal-verification-tactician/supervisor-release-evidence
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-supervisor-release-evidence.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 9
- Parallel lane: formal-verification-tactician/supervisor-release-evidence
- Conflict policy: Own proposal-output enforcement and read-only release evidence; preserve path fences, never broadly force-add ignored files, and never synthesize a missing terminal receipt.
- Predicted files: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/merge/leased_lane.py, ipfs_accelerate_py/agent_supervisor/release_evidence.py, test/api/test_agent_supervisor_release_evidence_binding.py
- Changed paths:
- Context paths: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/merge/leased_lane.py, ipfs_accelerate_py/agent_supervisor/release_evidence.py, test/api/test_agent_supervisor_release_evidence_binding.py
- AST symbols: test/api/test_agent_supervisor_release_evidence_binding.py
- Interfaces: AgentSupervisorReleaseEvidence@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G212
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/bef3dbab319e0d753a7462b7f34b17af31c058fe682056b8dd9b61087c8eadae
- Canonical task CID: baguqeerax3z5xkzrtygxkotumk37gsyxv4y4awh6naqfnog5tnqqq7eovwxa
- Semantic identity: objective-evidence-obligation/v1/9330d7f16bbafa77a3b5eb2e73f11097b1ddcf437c263437f9bd469cc39ae1b5
- Acceptance subset: Declared outputs are compared with filesystem, proposed paths, staged paths, and ignore rules, an exact allowed ignored output is force-added only by its explicit path or the proposal fails `expected_output_ignored_or_unstaged`, a regression proves an ignored JSON and tracked source both enter the commit, the exporter reads committed bundle/task metadata, lane manifest, scheduler snapshot, task state, event manifest/JSONL, and durable member_completion receipts once and hashes raw bytes, output binds canonical task CID/key, dependency CIDs, baseline and merged trees/gitlinks, attempt/phase, continuous event sequence, validation and merge outcomes, freshness, authority, and publication state, it never edits live state and cannot treat metrics-module presence as completion.
- Preconditions: objective goal FVT-G212 is schedulable
- Effects: satisfy evidence requirement: test/api/test_agent_supervisor_release_evidence_binding.py
- Evidence subset: test/api/test_agent_supervisor_release_evidence_binding.py
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-validation
- Merge fate: objective/FVT-G212
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/9330d7f16bbafa77a3b5eb2e73f11097b1ddcf437c263437f9bd469cc39ae1b5
- Missing evidence: test/api/test_agent_supervisor_release_evidence_binding.py
- Embedding query: Export a read-only, content-addressed execution snapshot and reject proposals whose declared outputs are ignored, absent, or unstaged.
- AST query: test/api/test_agent_supervisor_release_evidence_binding.py
- Surplus group: objective/FVT-G212
- Merge key: df7f31de539425c0
- Merge family: objective/FVT-G212
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
- Todo vector key: bc412e83e38acc38
- Acceptance: Objective scan filed this gap for FVT-G212. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-063-objective-gap-a119b5b3e02b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/api/test_agent_supervisor_release_evidence_binding.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
