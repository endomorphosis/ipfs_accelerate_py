# Objective Bundle: formal-verification-tactician/zkp-attestation-toolchain

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-047 Close objective gap: Bind and certify a production ZKP circuit deployment

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-041
- Outputs: config/formal_verification_zkp_deployment.lock.json, tools/logic/certification/zkp.py, test/integration/toolchains/test_zkp_deployment_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/api/test_agent_supervisor_program_analysis_zkp_conformance.py ipfs_datasets_py/tests/integration/logic/test_proof_receipt_attestation.py test/integration/toolchains/test_zkp_deployment_certification.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-047-objective-gap-657e17ea093d.md
- Bundle: formal-verification-tactician/zkp-attestation-toolchain
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-zkp-attestation-toolchain.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 46
- Parallel lane: formal-verification-tactician/zkp-attestation-toolchain
- Conflict policy: Own the deployment binding, ZKP handler, and test; reference private artifacts only by digest and configured secret-safe location.
- Predicted files: config/formal_verification_zkp_deployment.lock.json, tools/logic/certification/zkp.py, test/integration/toolchains/test_zkp_deployment_certification.py
- Changed paths:
- Context paths: config/formal_verification_zkp_deployment.lock.json, tools/logic/certification/zkp.py, test/integration/toolchains/test_zkp_deployment_certification.py
- AST symbols: config/formal_verification_zkp_deployment.lock.json, test/integration/toolchains/test_zkp_deployment_certification.py
- Interfaces: ZKPDeploymentCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G190
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/743e4169341aea847a33fbe937bf3c893e08c1c80b53ec2eb9ac40d0016b52e1
- Canonical task CID: baguqeeraoq7ec2judlvii6rt7putppz4re7arqoibnj6ylvzvranaallklqq
- Semantic identity: objective-evidence-obligation/v1/26052d1f339cd8de934f1c6d9247fa4c0f756352bd0db1d2dbe0858c3855f3b5
- Acceptance subset: Circuit, ceremony, proving-key and verification-key digests, public-input schema, backend, expiry, freshness, and revocation are exact and reviewable, live positive verification and corrupted proof/key/public-input, circuit mismatch, mutation, replay, stale, and revoked cases pass, private witnesses and secrets never enter Git, logs, caches, public receipts, or model context, ZKP authority attests an underlying receipt and never replaces semantic theorem authority.
- Preconditions: objective goal FVT-G190 is schedulable
- Effects: satisfy evidence requirement: config/formal_verification_zkp_deployment.lock.json, satisfy evidence requirement: test/integration/toolchains/test_zkp_deployment_certification.py
- Evidence subset: config/formal_verification_zkp_deployment.lock.json, test/integration/toolchains/test_zkp_deployment_certification.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G190
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/26052d1f339cd8de934f1c6d9247fa4c0f756352bd0db1d2dbe0858c3855f3b5
- Missing evidence: config/formal_verification_zkp_deployment.lock.json, test/integration/toolchains/test_zkp_deployment_certification.py
- Embedding query: Replace the production-circuit gap with a reviewed, secret-safe deployment binding and live verifier attestation certification.
- AST query: config/formal_verification_zkp_deployment.lock.json, test/integration/toolchains/test_zkp_deployment_certification.py
- Surplus group: objective/FVT-G190
- Merge key: 33f854a81ae2b9b3
- Merge family: objective/FVT-G190
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
- Todo vector key: 7c07d2498b72122f
- Acceptance: Objective scan filed this gap for FVT-G190. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-047-objective-gap-657e17ea093d.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (config/formal_verification_zkp_deployment.lock.json, test/integration/toolchains/test_zkp_deployment_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
