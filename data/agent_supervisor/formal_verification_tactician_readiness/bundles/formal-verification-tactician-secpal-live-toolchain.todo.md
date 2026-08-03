# Objective Bundle: formal-verification-tactician/secpal-live-toolchain

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-086 FVT:: Implement the genuine SecPAL external-toolchain path

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-055, FVT-073, FVT-087
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_secpal_live_toolchain_contract.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_secpal_live_toolchain_contract.py test/integration/toolchains/test_external_authorization_vendor_certification.py test/integration/toolchains/test_external_authorization_toolchain_certification.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-086-objective-gap-d9aca6db2ede.md
- Bundle: formal-verification-tactician/secpal-live-toolchain
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-secpal-live-toolchain.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 2
- Parallel lane: formal-verification-tactician/secpal-live-toolchain
- Conflict policy: Own SecPAL artifact provenance, platform matrix, installer, and external semantics; never invent an upstream release, accept an unreviewed mirror, bypass license terms, or label the in-process engine as the external vendor binary.
- Predicted files: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_secpal_live_toolchain_contract.py
- Changed paths:
- Context paths: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/authorization.py, tools/logic/certification/authorization_external.py, test/integration/toolchains/test_secpal_live_toolchain_contract.py
- AST symbols: test/integration/toolchains/test_secpal_live_toolchain_contract.py
- Interfaces: SecPALLiveToolchainContract@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G217
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/673ddca40390b034f58919f5226f741b2125329a9949e689dc59c27aec8c82b0
- Canonical task CID: baguqeeram465zjadscydj5mjdh2se33udmqskmu2tfe6nco4lhbhv3emqkya
- Semantic identity: objective-evidence-obligation/v1/d0a201df52ad6c2dc470b370be114e87721612b0dd93527a071046a825aa86ff
- Acceptance subset: Every supported SecPAL target binds an official publisher URL or operator-supplied reviewed artifact, immutable version and digest, redistribution and execution terms, architecture and OS, runtime dependencies, install plan, executable identity, and rollback behavior, unsupported hosts are derived from the reviewed lock and cannot install, certify, or count as complete, real allow, deny, unknown, delegation, conflict, rule/scope mutation, replay, malformed, timeout, and disagreement cases execute through the selected external engine, the in-process Datalog/SecPAL family and any hermetic adapter remain separately named and cannot impersonate the vendor tool.
- Preconditions: objective goal FVT-G217 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_secpal_live_toolchain_contract.py
- Evidence subset: test/integration/toolchains/test_secpal_live_toolchain_contract.py
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok-implement, codex-review
- Resources: cpu-validation
- Merge fate: objective/FVT-G217
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/d0a201df52ad6c2dc470b370be114e87721612b0dd93527a071046a825aa86ff
- Missing evidence: test/integration/toolchains/test_secpal_live_toolchain_contract.py
- Embedding query: Replace ambiguous SecPAL acquisition and adapter behavior with an official-artifact, license-aware, host-specific lazy installer and live semantic runner.
- AST query: test/integration/toolchains/test_secpal_live_toolchain_contract.py
- Surplus group: objective/FVT-G217
- Merge key: 66eff0af4b1533dd
- Merge family: objective/FVT-G217
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
- Todo vector key: 27c4c3f6a87f0134
- Acceptance: Objective scan filed this gap for FVT-G217. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-086-objective-gap-d9aca6db2ede.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_secpal_live_toolchain_contract.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
