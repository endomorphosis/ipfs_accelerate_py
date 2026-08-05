# Objective Bundle: formal-verification-tactician/managed-capability-closure

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: parallel residual repair for managed tools still unavailable on the deployment receipt.
Conflict policy: one tool family per task; reseal only in FVT-113.

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
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
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
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G234
- Acceptance: Close the managed-capability residual for {apalache,tlc,coq,isabelle,vampire,eprover,tamarin,maude,stack,proverif,opam,secpal,souffle,ergoai,temurin-jdk,runtime-mtl-external} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-103 Residual: Reinstall+certify TLC/Apalache managed state-model

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {tlc,apalache}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G235
- Completion authority: local
- Canonical task key: task/v1/d4baf86bb57b2e8e3536e436c426a511b624b38ae9db50089a94fba4657d3e78
- Canonical task CID: baguqeera2s5pq25vpmxi4njw4q3mijvfcg3cjm4k5hnvace2st52izl5hz4a
- Semantic identity: objective-evidence-obligation/v1/2951414198570da7b425c7c7915511cce2c1ef103600f4efe8de97f66495b44a
- Acceptance subset: For tools {tlc,apalache}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G235 is schedulable
- Effects: clear managed residual for tlc,apalache
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G235
- Acceptance: Close the managed-capability residual for {tlc,apalache} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-104 Residual: Reinstall+certify Vampire/E ATP managed

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {vampire,eprover}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G236
- Completion authority: local
- Canonical task key: task/v1/e08aaa402286fb7c17ee6cdb69622b19ea73c899bf071c49f613d2dc2bc9a72b
- Canonical task CID: baguqeera4cfkuqbcq35xyf7ontnwsyrldhvhhsezx4dryspwcpjnyk6ju4vq
- Semantic identity: objective-evidence-obligation/v1/534962db21d4fe15cce41a37041e647e402fa9d8ba25f3c4fbec34262b567203
- Acceptance subset: For tools {vampire,eprover}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G236 is schedulable
- Effects: clear managed residual for vampire,eprover
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G236
- Acceptance: Close the managed-capability residual for {vampire,eprover} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-105 Residual: Reinstall+certify Tamarin/Maude/stack managed

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {tamarin,maude,stack}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G237
- Completion authority: local
- Canonical task key: task/v1/3671f17f1ee6c00b3e1bc4bf40953a22f77093336c3838aacddf769309474986
- Canonical task CID: baguqeeragzy7c7y643aawpq3ys7ubfj2el3xbeztnq4drkwn353jgckhjgda
- Semantic identity: objective-evidence-obligation/v1/cd9f23e26d9acc6182dc17e132dadf6fc8d2a5f520af5fc9a6d112e5004739bc
- Acceptance subset: For tools {tamarin,maude,stack}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G237 is schedulable
- Effects: clear managed residual for tamarin,maude,stack
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G237
- Acceptance: Close the managed-capability residual for {tamarin,maude,stack} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-106 Residual: Reinstall+certify ProVerif/opam managed

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {proverif,opam}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G238
- Completion authority: local
- Canonical task key: task/v1/e4463131f365d80f1f94574066b1b9446632cac060192c87322850700b97cbad
- Canonical task CID: baguqeera4rddcmptmxma6h4uk5agnmnzirtdfswamamszbzsfbihac4xzowq
- Semantic identity: objective-evidence-obligation/v1/4ad5a25575795e94fb65fa4168e10b459cf11569e8204d32a72d1c32c5820fbc
- Acceptance subset: For tools {proverif,opam}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G238 is schedulable
- Effects: clear managed residual for proverif,opam
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G238
- Acceptance: Close the managed-capability residual for {proverif,opam} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-107 Residual: Reinstall+certify Rocq/Coq managed kernel

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {coq}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G239
- Completion authority: local
- Canonical task key: task/v1/bfbb2fd0118cb2d8c7e8df7bde13cab9f3d65580a2fa96488745bf114420bd77
- Canonical task CID: baguqeerax65s7uarrsznrr7i35554e6kxhz5mvmaul5jmsehiw7rcrbaxv3q
- Semantic identity: objective-evidence-obligation/v1/9ee37f020a4ce329919262d473f430b04d85d4259f6664709f7bfcad72f30fd0
- Acceptance subset: For tools {coq}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G239 is schedulable
- Effects: clear managed residual for coq
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G239
- Acceptance: Close the managed-capability residual for {coq} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-108 Residual: Reinstall+certify Isabelle managed kernel

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {isabelle}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G240
- Completion authority: local
- Canonical task key: task/v1/19b61914bb16206d90bd344d245b9dac179461923939f227aa71d8127a18c196
- Canonical task CID: baguqeeradg3bsff3cyqg3ef5grgsiw45vqlziymshe47ej5kohmbe6qyygla
- Semantic identity: objective-evidence-obligation/v1/d6f327332a0886643baada518338404230ddd4fdb1729baa410473ee7920f402
- Acceptance subset: For tools {isabelle}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G240 is schedulable
- Effects: clear managed residual for isabelle
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G240
- Acceptance: Close the managed-capability residual for {isabelle} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-109 Residual: Reinstall+certify Souffle/SecPAL vendor shadows

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {souffle,secpal}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G241
- Completion authority: local
- Canonical task key: task/v1/efd9135f8f25ada5a6dbeea78e76a076c9a457cf3286ea56c22a9551098d9dbc
- Canonical task CID: baguqeera57mrgx4peww2ljw352ty45vao3e2iv6pgkdouvwcfkkvccmntw6a
- Semantic identity: objective-evidence-obligation/v1/0276e6a07a72341c6a7ef7230feafeec954ded3dd1d670051f3555227d8a5549
- Acceptance subset: For tools {souffle,secpal}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G241 is schedulable
- Effects: clear managed residual for souffle,secpal
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G241
- Acceptance: Close the managed-capability residual for {souffle,secpal} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-110 Residual: Reinstall+certify ErgoAI + Temurin JDK managed

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {ergoai,temurin-jdk}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G242
- Completion authority: local
- Canonical task key: task/v1/fb71fee4640de2b944ce9452335d7b835b4209c974d7e02ec94e7822c30c2b78
- Canonical task CID: baguqeera7ny75zdebxrlsrgosrjdgxl3qnnuecojotl6alwjjz4cfqymfn4a
- Semantic identity: objective-evidence-obligation/v1/0bb0c3dd8aa99c3d48a4ef2c534dd404dd1f2ff0b69c3a74c9483c900ba125f4
- Acceptance subset: For tools {ergoai,temurin-jdk}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G242 is schedulable
- Effects: clear managed residual for ergoai,temurin-jdk
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G242
- Acceptance: Close the managed-capability residual for {ergoai,temurin-jdk} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-111 Residual: Reinstall+certify runtime-mtl-external sealed vendor

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {runtime-mtl-external}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G243
- Completion authority: local
- Canonical task key: task/v1/471b362c96f7b42b062d62c58a0ab489438123962a5ba7e5afcd9411221d13d0
- Canonical task CID: baguqeerai4ntmlew662cwbrnmlcyucvurfbyci4wfjn2pznpzwkbciq5cpia
- Semantic identity: objective-evidence-obligation/v1/020f02fa8c9e4a37318f0d4fddff5071b253b080755d77b2b3f12208f2b0612f
- Acceptance subset: For tools {runtime-mtl-external}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G243 is schedulable
- Effects: clear managed residual for runtime-mtl-external
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G243
- Acceptance: Close the managed-capability residual for {runtime-mtl-external} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


## FVT-112 Residual: Elevate datalog/secpal-authorization production semantics

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: managed-capability-residual
- Depends on: FVT-102
- Outputs: docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_toolchain_certificate.json, config/formal_verification_toolchains.lock.json
- Validation: PYTHONPATH=ipfs_datasets_py IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT=/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers python -m pytest test/integration/test_formal_verification_role_aware_post_merge_attestation.py::test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/formal_verification_tactician_readiness/discovery
- Discovery evidence: data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-05-fvt-managed-tool-residual.md
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
- Conflict policy: Own managed installer roots, lock pins, and live certification receipts for tools {datalog-authorization,secpal-authorization}; never mark deployment_ready true while managed blockers remain; never greenwash missing sealed roots.
- Predicted files: config/formal_verification_toolchains.lock.json, tools/logic/certify_formal_verification_toolchains.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
- Goal id: FVT-G244
- Completion authority: local
- Canonical task key: task/v1/684b9f0322314ab5df0e080bb78d7ed3a70e39f674a73b512dafc0e6c4e3fc1a
- Canonical task CID: baguqeeranbfz6azcgffllxyobaf3pdl62otq4opwosttwujnv7aonrhd7qna
- Semantic identity: objective-evidence-obligation/v1/7ecba9263634218bc76b0574ec16c3b6c5f0c4d6eaea65ce514d72f187f4e92f
- Acceptance subset: For tools {datalog-authorization,secpal-authorization}, produce a lock-matching managed installation under the approved sealed/user root, live positive/negative/mutation/replay certification receipts, and clear the corresponding managed:* deployment blockers and elevation gaps without forging digests. FVT-113 additionally reseals the role-aware certificate, matrix, and deployment receipt and records residual honest blockers only.
- Preconditions: sealed vendor root env IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT is set when required; objective goal FVT-G244 is schedulable
- Effects: clear managed residual for datalog-authorization,secpal-authorization
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G244
- Acceptance: Close the managed-capability residual for {datalog-authorization,secpal-authorization} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.


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
- Bundle: formal-verification-tactician/managed-capability-closure
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-managed-capability-closure.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/managed-capability-closure
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
- Resource class: cpu-proof-solver
- Token class: medium
- Provider role: grok, codex-review
- Merge fate: residual/FVT-G245
- Acceptance: Close the managed-capability residual for {deployment-receipt,matrix} that still blocks RoleAwareFormalVerificationRelease@1 after prior FVT install tasks lease-completed without host closure. Discovery: 2026-08-05-fvt-managed-tool-residual.md.

