# Objective Bundle: formal-verification-tactician/state-model-toolchains

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-042 Close objective gap: Install and certify TLC and Apalache state-model checking

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, tools/logic/certification/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py test/integration/toolchains/test_state_model_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'tlc or apalache or state_model' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-042-objective-gap-5780f0b302af.md
- Bundle: formal-verification-tactician/state-model-toolchains
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-state-model-toolchains.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 41
- Parallel lane: formal-verification-tactician/state-model-toolchains
- Conflict policy: Own the state-model installer plugin, handler, and test; consume the shared lock without editing it or the central certificate.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, tools/logic/certification/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, tools/logic/certification/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py
- Interfaces: StateModelToolchainCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G120
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/095faa3453e0d45498aceec91ea00b0ea672e44293ae13a084db849bcb7f8d98
- Canonical task CID: baguqeerabfp2unct4dkfjgfm53er5ialb2thfzccsoxbhiee3ocjxs37rwma
- Semantic identity: objective-evidence-obligation/v1/803de22f31db84c21a29ac24567feb2e091c15e4982046d5e493f7a237056ac4
- Acceptance subset: Explicit strict installation makes both exact tools usable, invariant-holds, violation trace, mutated Next/invariant, replay, malformed model, timeout, and bound behavior pass, model/config/constants/bounds/tool identities are bound, Java remains support only and bounded model-checking never promotes to theorem authority.
- Preconditions: objective goal FVT-G120 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, satisfy evidence requirement: test/integration/toolchains/test_state_model_toolchain_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py
- Resource class: jvm-proof-solver
- Token class: medium
- Estimated tokens: 0
- Resources: jvm-proof-solver
- Merge fate: objective/FVT-G120
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/803de22f31db84c21a29ac24567feb2e091c15e4982046d5e493f7a237056ac4
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py
- Embedding query: Complete the pinned user-local installation and semantic certification of TLC and Apalache for distributed workflows and state machines.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py
- Surplus group: objective/FVT-G120
- Merge key: da9c5f21d696bfe9
- Merge family: objective/FVT-G120
- Merge role: aggregate
- Work item count: 2
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: ffa7511b3bda16d6
- Acceptance: Objective scan filed this gap for FVT-G120. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-042-objective-gap-5780f0b302af.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/state_model.py, test/integration/toolchains/test_state_model_toolchain_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
