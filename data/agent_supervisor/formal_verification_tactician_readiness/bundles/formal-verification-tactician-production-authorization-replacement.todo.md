# Objective Bundle: formal-verification-tactician/production-authorization-replacement

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-099 Close objective gap: Implement a separately named production-candidate SecPAL-style authorization provider

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authorization-engine
- Depends on: FVT-101, FVT-092, FVT-093, FVT-095
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/secpal_style_authorization.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, docs/architecture/formal_verification_production_authorization_replacement_receipt.json, test/integration/toolchains/test_production_authorization_replacement.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_production_authorization_replacement.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-099-objective-gap-3721985e57d6.md
- Bundle: formal-verification-tactician/production-authorization-replacement
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-production-authorization-replacement.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 7
- Parallel lane: formal-verification-tactician/production-authorization-replacement
- Conflict policy: Own the separately named replacement provider and its evidence; never edit or vendor restricted Microsoft bytes, reuse the `secpal` external provider id, or claim legal/deployment approval.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/secpal_style_authorization.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, docs/architecture/formal_verification_production_authorization_replacement_receipt.json, test/integration/toolchains/test_production_authorization_replacement.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/secpal_style_authorization.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, docs/architecture/formal_verification_production_authorization_replacement_receipt.json, test/integration/toolchains/test_production_authorization_replacement.py
- AST symbols: docs/architecture/formal_verification_production_authorization_replacement_receipt.json, test/integration/toolchains/test_production_authorization_replacement.py
- Interfaces: ProductionAuthorizationReplacement@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G231
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/ca00feac74c55ea228dbf882a89f427e4d975ab43dbf63ace63578a976cc384e
- Canonical task CID: baguqeeraziap5lduyvpkekg37cbkrh2cpzgzowvuhw7whlhggv4ks5wmhbha
- Semantic identity: objective-evidence-obligation/v1/d80082413c09f2005d21d11e6a3bd56a531d86ab61e5e8e8c453bed4593d9fde
- Acceptance subset: The provider uses a new provider id and project-owned implementation derived only from public formal specifications and independently reviewed clean-room design records, no restricted MSI, decompiled code, sample source, trademark implication, or Microsoft vendor-compatibility claim enters the implementation, the typed policy/query language covers principal identity, delegation depth/scope, can-say/can-act-as, roles, exclusions, revocation/time validity, conflict, unknown/no-proof, constraints, and deterministic proof or counterexample witnesses, positive, negative, mutation, replay, malformed, cycle/resource-bound, differential, fuzz/property, and denial-safety cases pass against an executable formal semantics, public API, packaging, lazy dependencies, caches, proof tactician, Hammer/advisors, and receipts bind the new identity and authority ceiling, the provider cannot satisfy FVT-G219 or claim Microsoft SecPAL authority.
- Preconditions: objective goal FVT-G231 is schedulable
- Effects: satisfy evidence requirement: docs/architecture/formal_verification_production_authorization_replacement_receipt.json, satisfy evidence requirement: test/integration/toolchains/test_production_authorization_replacement.py
- Evidence subset: docs/architecture/formal_verification_production_authorization_replacement_receipt.json, test/integration/toolchains/test_production_authorization_replacement.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G231
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/d80082413c09f2005d21d11e6a3bd56a531d86ab61e5e8e8c453bed4593d9fde
- Missing evidence: docs/architecture/formal_verification_production_authorization_replacement_receipt.json, test/integration/toolchains/test_production_authorization_replacement.py
- Embedding query: Provide a production-candidate authorization prover with SecPAL-style typed delegation semantics under a project-owned, license-clear identity, without copying or impersonating the retired Microsoft implementation.
- AST query: docs/architecture/formal_verification_production_authorization_replacement_receipt.json, test/integration/toolchains/test_production_authorization_replacement.py
- Surplus group: objective/FVT-G231
- Merge key: fcec2865efe09b57
- Merge family: objective/FVT-G231
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
- Todo vector key: f0290cdd9a104255
- Acceptance: Objective scan filed this gap for FVT-G231. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-099-objective-gap-3721985e57d6.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (docs/architecture/formal_verification_production_authorization_replacement_receipt.json, test/integration/toolchains/test_production_authorization_replacement.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
