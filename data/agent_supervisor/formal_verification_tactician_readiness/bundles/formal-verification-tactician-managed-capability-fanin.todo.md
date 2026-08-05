# Objective Bundle: formal-verification-tactician/managed-capability-fanin

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: parallel managed-tool residual repair for unavailable deployment tools.
Conflict policy: own only this tool family; never greenwash missing sealed roots.
Unavailable tools covered: fan-in/reseal.
Discovery: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md

## FVT-102 Residual: managed-capability residual fan-in

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-094
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-fanin
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-fanin.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-fanin
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {apalache,tlc,coq,isabelle,vampire,eprover,tamarin,maude,stack,proverif,opam,secpal,souffle,ergoai,temurin-jdk,runtime-mtl-external}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G234
- Completion authority: local
- Canonical task key: task/v1/125d200781399cc22f3e6f3620a4ba8c6f33b0524a595f8cc01f6f5c3e1e16da
- Canonical task CID: baguqeeracjosab4bhgomelz6n43cbjf2rrxthmcsjjmv7dgad5xvypq6c3na
- Semantic identity: objective-evidence-obligation/v1/83758f91d6ad568364168964f66c55f3c0c08fe114e4552b789a2c5ae94df90b
- Acceptance subset: For tools {apalache,tlc,coq,isabelle,vampire,eprover,tamarin,maude,stack,proverif,opam,secpal,souffle,ergoai,temurin-jdk,runtime-mtl-external}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G234 is schedulable
- Effects: clear managed residual for apalache,tlc,coq,isabelle,vampire,eprover,tamarin,maude,stack,proverif,opam,secpal,souffle,ergoai,temurin-jdk,runtime-mtl-external
- Resource class: cpu-validation
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G234
- Acceptance: Close the managed-capability residual for {apalache,tlc,coq,isabelle,vampire,eprover,tamarin,maude,stack,proverif,opam,secpal,souffle,ergoai,temurin-jdk,runtime-mtl-external} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.

