# Objective Bundle: formal-verification-tactician/authoritative-vendor-release

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-089 FVT:: Reissue deployment certification with authoritative vendor evidence

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: completion
- Depends on: FVT-067, FVT-082, FVT-G219, FVT-088
- Outputs: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, docs/architecture/formal_verification_authoritative_vendor_release.json, docs/architecture/formal_verification_tactician_readiness_completion_receipt.json, test/integration/test_formal_verification_authoritative_vendor_release.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/test_formal_verification_authoritative_vendor_release.py test/integration/toolchains/test_formal_verification_end_to_end_assurance_matrix.py test/integration/toolchains/test_secpal_ergoai_authoritative_live_evidence.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-089-objective-gap-c74869305ba9.md
- Bundle: formal-verification-tactician/authoritative-vendor-release
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-authoritative-vendor-release.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 5
- Parallel lane: formal-verification-tactician/authoritative-vendor-release
- Conflict policy: Sole owner of the authoritative vendor release after every dependency closes; never manufacture external evidence, weaken a platform or authority gate, or attest the current task's future merge.
- Predicted files: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, docs/architecture/formal_verification_authoritative_vendor_release.json, docs/architecture/formal_verification_tactician_readiness_completion_receipt.json, test/integration/test_formal_verification_authoritative_vendor_release.py
- Changed paths:
- Context paths: tools/logic/certify_formal_verification_toolchains.py, tools/logic/build_formal_verification_tactician_receipt.py, docs/architecture/formal_verification_authoritative_vendor_release.json, docs/architecture/formal_verification_tactician_readiness_completion_receipt.json, test/integration/test_formal_verification_authoritative_vendor_release.py
- AST symbols: docs/architecture/formal_verification_authoritative_vendor_release.json, test/integration/test_formal_verification_authoritative_vendor_release.py
- Interfaces: FormalVerificationAuthoritativeVendorRelease@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G221
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/74ebf612ce3fedbc89b9b6234685fd7ab4b1c5b373683f8273512a47261d660d
- Canonical task CID: baguqeeraotv7mewoh7w3zcnzwyrunbp5pk2ldrntonud7attkeveojq5mygq
- Semantic identity: objective-evidence-obligation/v1/25c959347d4cad30ccbc2794dc7f5bfa56571aad78f440f179dcf50deed52b15
- Acceptance subset: The release binds clean-wheel evidence, explicit lazy-install receipts, exact dependency and platform identities, complete specialized semantic cases, SecPAL and ErgoAI authoritative live receipts, authority ceilings, disagreement quarantines, public-safe envelopes, durable supervisor completion, source and merged trees, recursive gitlinks, and origin publication, every dependency is reachable and fresh, fixture, shim, unsupported, proposal-only, or externally blocked lanes remain disclosed and prevent deployment-ready status.
- Preconditions: objective goal FVT-G221 is schedulable
- Effects: satisfy evidence requirement: docs/architecture/formal_verification_authoritative_vendor_release.json, satisfy evidence requirement: test/integration/test_formal_verification_authoritative_vendor_release.py
- Evidence subset: docs/architecture/formal_verification_authoritative_vendor_release.json, test/integration/test_formal_verification_authoritative_vendor_release.py
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: cpu-validation
- Merge fate: objective/FVT-G221
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/25c959347d4cad30ccbc2794dc7f5bfa56571aad78f440f179dcf50deed52b15
- Missing evidence: docs/architecture/formal_verification_authoritative_vendor_release.json, test/integration/test_formal_verification_authoritative_vendor_release.py
- Embedding query: Reissue the role-aware release and post-merge attestation only after packaging, lazy installers, every readiness axis, and genuine SecPAL and ErgoAI live evidence are closed.
- AST query: docs/architecture/formal_verification_authoritative_vendor_release.json, test/integration/test_formal_verification_authoritative_vendor_release.py
- Surplus group: objective/FVT-G221
- Merge key: 9fe21bef9d5b4ae6
- Merge family: objective/FVT-G221
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
- Todo vector key: a70fb0fe53a6dfc2
- Acceptance: Objective scan filed this gap for FVT-G221. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-089-objective-gap-c74869305ba9.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (docs/architecture/formal_verification_authoritative_vendor_release.json, test/integration/test_formal_verification_authoritative_vendor_release.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
