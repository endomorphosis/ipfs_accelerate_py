# Objective Bundle: formal-verification-tactician/managed-elevation-datalog-secpal

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: parallel managed-tool residual repair for unavailable deployment tools.
Conflict policy: own only this tool family; never greenwash missing sealed roots.
Unavailable tools covered: datalog-authorization,secpal-authorization.
Discovery: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md

## FVT-112 Residual: Elevate datalog/secpal-authorization production semantics

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-094
- Outputs: data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-elevation-datalog-secpal/install-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-elevation-datalog-secpal/certification-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-elevation-datalog-secpal/tools/datalog-authorization/managed-root.marker, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-elevation-datalog-secpal/tools/secpal-authorization/managed-root.marker
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-elevation-datalog-secpal
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-elevation-datalog-secpal.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-elevation-datalog-secpal
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {datalog-authorization,secpal-authorization}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Allow concurrent with: formal-verification-tactician/toolchain-release-candidate, formal-verification-tactician/secpal-live-toolchain, formal-verification-tactician/managed-tlc-apalache, formal-verification-tactician/managed-vampire-eprover, formal-verification-tactician/managed-tamarin-maude-stack, formal-verification-tactician/managed-proverif-opam, formal-verification-tactician/managed-rocq-coq, formal-verification-tactician/managed-isabelle, formal-verification-tactician/managed-souffle-secpal, formal-verification-tactician/managed-ergoai-temurin, formal-verification-tactician/managed-runtime-mtl-external
- Predicted files: data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-elevation-datalog-secpal/install-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-elevation-datalog-secpal/certification-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-elevation-datalog-secpal/tools/datalog-authorization/managed-root.marker, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-elevation-datalog-secpal/tools/secpal-authorization/managed-root.marker
- Goal id: FVT-G244
- Completion authority: local
- Canonical task key: task/v1/684b9f0322314ab5df0e080bb78d7ed3a70e39f674a73b512dafc0e6c4e3fc1a
- Canonical task CID: baguqeeranbfz6azcgffllxyobaf3pdl62otq4opwosttwujnv7aonrhd7qna
- Semantic identity: objective-evidence-obligation/v1/7ecba9263634218bc76b0574ec16c3b6c5f0c4d6eaea65ce514d72f187f4e92f
- Acceptance subset: For tools {datalog-authorization,secpal-authorization}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G244 is schedulable
- Effects: clear managed residual for datalog-authorization,secpal-authorization
- Resource class: cpu-validation
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G244
- Acceptance: Close the managed-capability residual for {datalog-authorization,secpal-authorization} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.

