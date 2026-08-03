# Objective Bundle: formal-verification-tactician/lazy-installer-facade

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-087 FVT:: Bind the public Logic API to transactional lazy installers

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: dependency-integrity
- Depends on: FVT-084
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/lazy_installer.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py ipfs_datasets_py/tests/unit/test_lazy_dependency_installation.py ipfs_datasets_py/tests/unit_tests/logic/external_provers/test_lazy_native_solver_installation.py test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-087-objective-gap-b46c3afbede6.md
- Bundle: formal-verification-tactician/lazy-installer-facade
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-lazy-installer-facade.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 3
- Parallel lane: formal-verification-tactician/lazy-installer-facade
- Conflict policy: Own the public install facade, registry, and lifecycle; never infer permission from a probe, dispatch an unreviewed shell command, silently fall back to a shim, or let installation occur inside certification.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/lazy_installer.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/lazy_installer.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
- AST symbols: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py, ipfs_datasets_py/tests/unit_tests/logic/external_provers/test_lazy_native_solver_installation.py
- Interfaces: LogicVerificationLazyInstaller@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G216
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/acde7f9caf2500740711b070b10fa4cadc5f4d314c7a1097ea28395521cda592
- Canonical task CID: baguqeeravtph7hfpeuahibyrwbylcd5ezlof6tjrjr5bbf7kfa4vkionuwja
- Semantic identity: objective-evidence-obligation/v1/bbf2e78467aeba4045dddaef815fd78c4f48767e9e1c3105c8f5bab7a39f226b
- Acceptance subset: LogicVerificationAPI.install_provider resolves reviewed family plugins for SMT, kernels, state models, authorization, protocols, ATP, hyperproperties, Runtime MTL, advisors, and ZKP, probe, inventory, import, dry-run, and offline certification execute no installer command and perform no network access, installation requires an explicit allow flag and produces a bounded plan before mutation, platform, dependency, license, checksum, artifact, executable, rollback, and post-install semantic-probe results are returned as structured evidence, interrupted or failed publication preserves the previous good installation and cannot promote capability or semantic authority.
- Preconditions: objective goal FVT-G216 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
- Evidence subset: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
- Resource class: io-artifact
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok-implement, codex-review
- Resources: io-artifact
- Merge fate: objective/FVT-G216
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/bbf2e78467aeba4045dddaef815fd78c4f48767e9e1c3105c8f5bab7a39f226b
- Missing evidence: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
- Embedding query: Replace placeholder and stale installer dispatch with one explicit, platform-aware, transactional lazy-install lifecycle for every reviewed prover family.
- AST query: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py, ipfs_datasets_py/tests/unit_tests/logic/external_provers/test_lazy_native_solver_installation.py
- Surplus group: objective/FVT-G216
- Merge key: b7c231b32f4acdbc
- Merge family: objective/FVT-G216
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
- Todo vector key: 22412d403776657e
- Acceptance: Objective scan filed this gap for FVT-G216. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-087-objective-gap-b46c3afbede6.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
