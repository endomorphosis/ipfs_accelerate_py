# Objective Bundle: formal-verification-tactician/toolchain-locks

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-041 Close objective gap: Replace declared external-tool gaps with reviewed deployment locks

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: toolchain-governance
- Depends on: FVT-037
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, test/packaging/test_formal_verification_external_tool_locks.py
- Validation: python -m pytest test/packaging/test_formal_verification_external_tool_locks.py ipfs_datasets_py/tests/integration/logic/test_verification_toolchain_security.py -k 'pin or checksum or gap or install' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-041-objective-gap-9e357d3d8d15.md
- Bundle: formal-verification-tactician/toolchain-locks
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-toolchain-locks.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 40
- Parallel lane: formal-verification-tactician/toolchain-locks
- Conflict policy: Sole owner for the shared lock and installer registry; add per-family installer plugins for downstream tasks and do not install tools as part of this metadata task.
- Predicted files: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, test/packaging/test_formal_verification_external_tool_locks.py
- Changed paths:
- Context paths: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, test/packaging/test_formal_verification_external_tool_locks.py
- AST symbols: config/formal_verification_toolchains.lock.json, test/packaging/test_formal_verification_external_tool_locks.py
- Interfaces: FormalVerificationDeploymentLock@2
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G110
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/d96b41b55c9705ca0a76544249de0a4526cb4a173bd090e22d8bb486ac79bb90
- Canonical task CID: baguqeera3fvudnk4s4c4uctwkrbetxqkiutmwsqxhpijbyrnro2inldzxoia
- Semantic identity: objective-evidence-obligation/v1/2e094180a3c08389c764cf7a34f9431c1a393f61e5d1b1b69d69345368c60a17
- Acceptance subset: TLC, HyperLTL/AutoHyper/MCHyper, Souffle/SecPAL, external Runtime MTL, Vampire, Lean, Rocq, Isabelle, OPAM, SymbolicAI, and ErgoAI have reviewed version/license/platform/source/checksum or immutable package-lock identities and installer entries, ZKP has a secret-safe deployment-artifact schema, unsupported platforms fail explicitly, installs are user-local and require explicit opt-in, imports, discovery, tests, and offline certification never install, download, access the network, or mutate a system package manager.
- Preconditions: objective goal FVT-G110 is schedulable
- Effects: satisfy evidence requirement: test/packaging/test_formal_verification_external_tool_locks.py
- Evidence subset: test/packaging/test_formal_verification_external_tool_locks.py
- Resource class: cpu-install-test
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-install-test
- Merge fate: objective/FVT-G110
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/2e094180a3c08389c764cf7a34f9431c1a393f61e5d1b1b69d69345368c60a17
- Missing evidence: test/packaging/test_formal_verification_external_tool_locks.py
- Embedding query: Turn every remaining declared installation gap or incomplete managed pin into a reviewed, licensed, per-platform, explicitly invoked deployment contract.
- AST query: config/formal_verification_toolchains.lock.json, test/packaging/test_formal_verification_external_tool_locks.py
- Surplus group: objective/FVT-G110
- Merge key: 4cb8b1f49fb7920c
- Merge family: objective/FVT-G110
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
- Todo vector key: 5c502f83d4389245
- Acceptance: Objective scan filed this gap for FVT-G110. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-041-objective-gap-9e357d3d8d15.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/packaging/test_formal_verification_external_tool_locks.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
