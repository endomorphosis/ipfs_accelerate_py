# Objective Bundle: formal-verification-tactician/managed-vampire-eprover

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: parallel managed-tool residual repair for unavailable deployment tools.
Conflict policy: own only this tool family; never greenwash missing sealed roots.
Unavailable tools covered: vampire,eprover.
Discovery: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md

## FVT-104 Residual: Reinstall+certify Vampire/E ATP managed

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-094
- Outputs: data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-vampire-eprover/install-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-vampire-eprover/certification-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-vampire-eprover/tools/vampire/managed-root.marker, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-vampire-eprover/tools/eprover/managed-root.marker
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-vampire-eprover
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-vampire-eprover.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-vampire-eprover
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {vampire,eprover}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Allow concurrent with: formal-verification-tactician/toolchain-release-candidate, formal-verification-tactician/secpal-live-toolchain, formal-verification-tactician/managed-tlc-apalache, formal-verification-tactician/managed-tamarin-maude-stack, formal-verification-tactician/managed-proverif-opam, formal-verification-tactician/managed-rocq-coq, formal-verification-tactician/managed-isabelle, formal-verification-tactician/managed-souffle-secpal, formal-verification-tactician/managed-ergoai-temurin, formal-verification-tactician/managed-runtime-mtl-external, formal-verification-tactician/managed-elevation-datalog-secpal
- Predicted files: data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-vampire-eprover/install-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-vampire-eprover/certification-receipt.json, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-vampire-eprover/tools/vampire/managed-root.marker, data/agent_supervisor/formal_verification_tactician_readiness/managed-residuals/managed-vampire-eprover/tools/eprover/managed-root.marker
- Goal id: FVT-G236
- Completion authority: local
- Canonical task key: task/v1/e08aaa402286fb7c17ee6cdb69622b19ea73c899bf071c49f613d2dc2bc9a72b
- Canonical task CID: baguqeera4cfkuqbcq35xyf7ontnwsyrldhvhhsezx4dryspwcpjnyk6ju4vq
- Semantic identity: objective-evidence-obligation/v1/534962db21d4fe15cce41a37041e647e402fa9d8ba25f3c4fbec34262b567203
- Acceptance subset: For tools {vampire,eprover}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G236 is schedulable
- Effects: clear managed residual for vampire,eprover
- Resource class: cpu-validation
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G236
- Acceptance: Close the managed-capability residual for {vampire,eprover} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.

