# Objective Bundle: formal-verification-tactician/public-provider-role-closure

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-095 Close objective gap: Close Logic API and installer-role gaps

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-surface
- Depends on: FVT-087, FVT-094, FVT-088
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py, test/api/test_formal_verification_provider_role_closure.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/api/test_formal_verification_provider_role_closure.py ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py test/api/test_logic_verification_api_install_provider.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-095-objective-gap-5764c22457a5.md
- Bundle: formal-verification-tactician/public-provider-role-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-public-provider-role-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 3
- Parallel lane: formal-verification-tactician/public-provider-role-closure
- Conflict policy: Own public dispatch and role classification; do not change provider semantics, promote advisors/support tools, or expose restricted SecPAL bytes.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py, test/api/test_formal_verification_provider_role_closure.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/registry.py, ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py, test/api/test_formal_verification_provider_role_closure.py
- AST symbols: test/api/test_formal_verification_provider_role_closure.py, ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py
- Interfaces: LogicVerificationProviderRoleClosure@1, LogicVerificationLazyInstaller@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G227
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/3b3215b6610f461e53c8b10c3b62713d4daffbbcc1e8742a6179e8d0c4ce189f
- Canonical task CID: baguqeerahmzblntbb5db4u6iwegdwytrhvg27654yhuhiktbphunbrgodcpq
- Semantic identity: objective-evidence-obligation/v1/b586437971bb1dcd40c6cbb4fe4b1a9da27f382e5f915ab7855bc841936440b5
- Acceptance subset: ErgoAI, external Runtime MTL, Souffle, and SymbolicAI have real typed inventory, probe, explicit install, and verification/advisor dispatch surfaces, legacy Microsoft SecPAL exposes only reviewed non-executable artifact intake and compatibility receipt lookup, never a live verification provider, Stack, Temurin JDK, Maude, and OPAM are explicitly support-only with semantic, authority, and public-verification axes not applicable, every installer registry entry resolves to a real bounded implementation rather than placeholder dispatch, import, inventory, dry-run, and offline certification are side-effect free, unsupported roles and provider/role confusion fail with typed non-success responses.
- Preconditions: objective goal FVT-G227 is schedulable
- Effects: satisfy evidence requirement: test/api/test_formal_verification_provider_role_closure.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py
- Evidence subset: test/api/test_formal_verification_provider_role_closure.py, ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: cpu-validation
- Merge fate: objective/FVT-G227
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/b586437971bb1dcd40c6cbb4fe4b1a9da27f382e5f915ab7855bc841936440b5
- Missing evidence: test/api/test_formal_verification_provider_role_closure.py, ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py
- Embedding query: Make every runnable prover/advisor reachable through the stable Logic API while keeping support dependencies and archival SecPAL intake out of verification dispatch.
- AST query: test/api/test_formal_verification_provider_role_closure.py, ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py
- Surplus group: objective/FVT-G227
- Merge key: 376223237519b30f
- Merge family: objective/FVT-G227
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
- Todo vector key: 7653fe29c363acba
- Acceptance: Objective scan filed this gap for FVT-G227. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-095-objective-gap-5764c22457a5.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/api/test_formal_verification_provider_role_closure.py, ipfs_datasets_py/tests/unit/logic/test_provider_role_installation_closure.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
