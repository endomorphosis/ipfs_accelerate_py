# Objective Bundle: formal-verification-tactician/toolchain-probe-integrity

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-062 Close objective gap: Repair exact probes and managed artifact identities

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: dependency-integrity
- Depends on: FVT-041, FVT-042, FVT-044, FVT-049
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, tools/logic/certify_formal_verification_toolchains.py, test/integration/toolchains/test_formal_verification_probe_integrity.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_formal_verification_probe_integrity.py test/integration/toolchains/test_state_model_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-062-objective-gap-ed170e90f655.md
- Bundle: formal-verification-tactician/toolchain-probe-integrity
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-toolchain-probe-integrity.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 8
- Parallel lane: formal-verification-tactician/toolchain-probe-integrity
- Conflict policy: Own exact probe commands, reviewed identities, and atomic publication; never mutate system Java, accept an unbound artifact, trust arbitrary nonempty output, or validate a different path than the public launcher.
- Predicted files: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, tools/logic/certify_formal_verification_toolchains.py, test/integration/toolchains/test_formal_verification_probe_integrity.py
- Changed paths:
- Context paths: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, tools/logic/certify_formal_verification_toolchains.py, test/integration/toolchains/test_formal_verification_probe_integrity.py
- AST symbols: test/integration/toolchains/test_formal_verification_probe_integrity.py
- Interfaces: FormalVerificationProbeIntegrity@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G202
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/769254077b08376c347d2ca258bb917ac1df1b4888bbeade4392ee8a8641c94b
- Canonical task CID: baguqeerao2jfib33ba3wynd5fsrfro4rpla56g2irc56vxsdslxivbsbzffq
- Semantic identity: objective-evidence-obligation/v1/78d6fd2ca7b3971a065b2f7c828393e8e9ee70bfcfb36e7dc585d31a39c28c82
- Acceptance subset: Java identity is parsed only from the quoted java/openjdk version banner after hostile Java option variables are neutralized, bare names resolve only through PATH and dry-run executes nothing, Apalache uses `version`, Isabelle uses `version`, ProVerif uses a valid identity command, and nonzero error banners cannot prove usability, TLC 1.8.0 binds SHA-256 `e22f8ffb4bacdea0a871f444dd94fe5fb0d8013b3388ae39e82e26f852c735d5` plus manifest tag `v1.8.0` and revision `30cc360`, genuine TLC help is recognized despite exit 1 only with required markers, returned launchers execute through the validated Java 17+ runtime, TLC and Apalache artifact plus launcher repair is staged, atomic, and rollback-safe, failed repair preserves a prior good install.
- Preconditions: objective goal FVT-G202 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_formal_verification_probe_integrity.py
- Evidence subset: test/integration/toolchains/test_formal_verification_probe_integrity.py
- Resource class: exclusive-jvm-toolchain
- Token class: medium
- Estimated tokens: 0
- Resources: exclusive-jvm-toolchain
- Merge fate: objective/FVT-G202
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/78d6fd2ca7b3971a065b2f7c828393e8e9ee70bfcfb36e7dc585d31a39c28c82
- Missing evidence: test/integration/toolchains/test_formal_verification_probe_integrity.py
- Embedding query: Make generic and state-model identity probing command-correct, return-code-aware, digest-bound, hostile-environment-safe, and atomic.
- AST query: test/integration/toolchains/test_formal_verification_probe_integrity.py
- Surplus group: objective/FVT-G202
- Merge key: be4ad75cc5476b37
- Merge family: objective/FVT-G202
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
- Todo vector key: c265f0f3419c3a74
- Acceptance: Objective scan filed this gap for FVT-G202. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-062-objective-gap-ed170e90f655.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_formal_verification_probe_integrity.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
