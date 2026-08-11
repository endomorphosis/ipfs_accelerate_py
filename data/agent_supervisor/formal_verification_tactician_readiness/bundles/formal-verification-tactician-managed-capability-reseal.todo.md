# Objective Bundle: formal-verification-tactician/managed-capability-reseal

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: parallel managed-tool residual repair for unavailable deployment tools.
Conflict policy: own only this tool family; never greenwash missing sealed roots.
Unavailable tools covered: fan-in/reseal.
Discovery: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md

## FVT-113 Residual: Reseal matrix+deployment after managed capability closure

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-103, FVT-104, FVT-105, FVT-106, FVT-107, FVT-108, FVT-109, FVT-110, FVT-111, FVT-112
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-reseal
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-reseal.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-reseal
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {deployment-receipt,matrix}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G245
- Completion authority: local
- Canonical task key: task/v1/9d0c4855ac26bfbbcc04147942848eee856a57ff503ec7b8472e0bee2b5713a7
- Canonical task CID: baguqeeratugeqvnme273xtaecr4ufbeo52cwuv77ka7mpochfyf64k2xcotq
- Semantic identity: objective-evidence-obligation/v1/69f1dc5759133b6964438f97073881ac352b2ad4a9250be6340d0626e1c78b04
- Acceptance subset: For tools {deployment-receipt,matrix}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G245 is schedulable
- Effects: clear managed residual for deployment-receipt,matrix
- Resource class: cpu-validation
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G245
- Acceptance: Close the managed-capability residual for {deployment-receipt,matrix} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.

