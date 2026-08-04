# FVT-087 Objective Goal Gap

Date: 2026-08-03
Fingerprint: b46c3afbede6ff7608e983f6d1f265951a933df7
Goal id: FVT-G216
Goal title: Bind the public Logic API to transactional lazy installers
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: dependency-integrity
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 3
Bundle: formal-verification-tactician/lazy-installer-facade
Parallel lane: formal-verification-tactician/lazy-installer-facade
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: path
Embedding query: Replace placeholder and stale installer dispatch with one explicit, platform-aware, transactional lazy-install lifecycle for every reviewed prover family.
AST query: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py, ipfs_datasets_py/tests/unit_tests/logic/external_provers/test_lazy_native_solver_installation.py
Conflict policy: Own the public install facade, registry, and lifecycle; never infer permission from a probe, dispatch an unreviewed shell command, silently fall back to a shim, or let installation occur inside certification.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/lazy_installer.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
AST symbols: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py, ipfs_datasets_py/tests/unit_tests/logic/external_provers/test_lazy_native_solver_installation.py
Interfaces: LogicVerificationLazyInstaller@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/bbf2e78467aeba4045dddaef815fd78c4f48767e9e1c3105c8f5bab7a39f226b
Acceptance subset: LogicVerificationAPI.install_provider resolves reviewed family plugins for SMT, kernels, state models, authorization, protocols, ATP, hyperproperties, Runtime MTL, advisors, and ZKP, probe, inventory, import, dry-run, and offline certification execute no installer command and perform no network access, installation requires an explicit allow flag and produces a bounded plan before mutation, platform, dependency, license, checksum, artifact, executable, rollback, and post-install semantic-probe results are returned as structured evidence, interrupted or failed publication preserves the previous good installation and cannot promote capability or semantic authority.
Preconditions: objective goal FVT-G216 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
Evidence subset: ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py
Dependencies: FVT-084
Resource class: io-artifact
Token class: medium
Estimated tokens: 0
Resources: io-artifact
Merge fate: objective/FVT-G216
Rejection reasons: none (accepted)

## Goal

Replace placeholder and stale installer dispatch with one explicit, platform-aware, transactional lazy-install lifecycle for every reviewed prover family.

## Missing Evidence

- ipfs_datasets_py/tests/unit/logic/test_verification_api_lazy_installation.py

## Present Evidence

- ipfs_datasets_py/tests/unit_tests/logic/external_provers/test_lazy_native_solver_installation.py: ipfs_datasets_py/tests/unit_tests/logic/external_provers/test_lazy_native_solver_installation.py (path)

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
