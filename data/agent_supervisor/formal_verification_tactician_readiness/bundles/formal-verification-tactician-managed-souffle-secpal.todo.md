# Objective Bundle: formal-verification-tactician/managed-souffle-secpal

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: parallel managed-tool residual repair for unavailable deployment tools.
Conflict policy: own only this tool family; never greenwash missing sealed roots.
Unavailable tools covered: secpal,souffle.
Discovery: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md

## FVT-109 Residual: Reinstall+certify Souffle/SecPAL vendor shadows

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-094
- Outputs: data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-souffle-secpal/install-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-souffle-secpal/certification-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-souffle-secpal/tools/souffle/managed-root.marker, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-souffle-secpal/tools/secpal/managed-root.marker
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-souffle-secpal
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-souffle-secpal.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-souffle-secpal
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {souffle,secpal}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Allow concurrent with: formal-verification-tactician/toolchain-release-candidate, formal-verification-tactician/secpal-live-toolchain, formal-verification-tactician/managed-tlc-apalache, formal-verification-tactician/managed-vampire-eprover, formal-verification-tactician/managed-tamarin-maude-stack, formal-verification-tactician/managed-proverif-opam, formal-verification-tactician/managed-rocq-coq, formal-verification-tactician/managed-isabelle, formal-verification-tactician/managed-ergoai-temurin, formal-verification-tactician/managed-runtime-mtl-external, formal-verification-tactician/managed-elevation-datalog-secpal
- Predicted files: data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-souffle-secpal/install-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-souffle-secpal/certification-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-souffle-secpal/tools/souffle/managed-root.marker, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-souffle-secpal/tools/secpal/managed-root.marker
- Goal id: FVT-G241
- Completion authority: local
- Canonical task key: task/v1/efd9135f8f25ada5a6dbeea78e76a076c9a457cf3286ea56c22a9551098d9dbc
- Canonical task CID: baguqeera57mrgx4peww2ljw352ty45vao3e2iv6pgkdouvwcfkkvccmntw6a
- Semantic identity: objective-evidence-obligation/v1/0276e6a07a72341c6a7ef7230feafeec954ded3dd1d670051f3555227d8a5549
- Acceptance subset: For tools {souffle,secpal}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G241 is schedulable
- Effects: clear managed residual for souffle,secpal
- Resource class: cpu-validation
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G241
- Acceptance: Close the managed-capability residual for {souffle,secpal} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.

