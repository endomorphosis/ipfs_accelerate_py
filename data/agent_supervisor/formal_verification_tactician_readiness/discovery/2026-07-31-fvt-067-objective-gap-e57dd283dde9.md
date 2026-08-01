# FVT-067 Objective Goal Gap

Date: 2026-07-31
Fingerprint: e57dd283dde93c82db7725d2d3ae49dccf0a5f72
Goal id: FVT-G214
Goal title: Publish the post-merge deployment attestation
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: completion
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 13
Bundle: formal-verification-tactician/toolchain-release-finalizer
Parallel lane: formal-verification-tactician/toolchain-release-finalizer
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: After the release-candidate merge, bind its durable terminal supervisor receipt and publish the final deployment attestation without circular tree identity.
AST query: test/integration/test_formal_verification_role_aware_post_merge_attestation.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
Conflict policy: Sole post-merge finalizer; read live state without mutation, never attest the current task's future event, and never weaken a missing terminal receipt or publication gate.
Predicted files: tools/logic/finalize_formal_verification_deployment.py, test/integration/test_formal_verification_role_aware_post_merge_attestation.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json, docs/architecture/formal_verification_tactician_readiness_completion_receipt.json
AST symbols: test/integration/test_formal_verification_role_aware_post_merge_attestation.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
Interfaces: RoleAwareFormalVerificationRelease@1
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/064cc3158783ceeab9e8c3bee194f4b5ad98ed47c149c8cc822e3ff0e0cf6d6e
Acceptance subset: The finalizer runs only after FVT-G213 has a successful, durable, canonical member completion receipt and reachable merged commit, it verifies event-chain continuity, expected outputs, validation result, source tree, merged tree, datasets gitlink, origin publication, candidate digest, supported-capability closure, hard-zero gates, authority boundaries, quarantines, and public surfaces, it publishes either a receipt commit whose parent is the certified release commit with a strictly limited generated-artifact diff or an external content-addressed attestation, mutating any event, tree, artifact, check, binding, or publication fact invalidates the receipt, absent or stale terminal evidence remains partial and can never be called deployment-ready.
Preconditions: objective goal FVT-G214 is schedulable
Effects: satisfy evidence requirement: test/integration/test_formal_verification_role_aware_post_merge_attestation.py, satisfy evidence requirement: docs/architecture/formal_verification_role_aware_deployment_receipt.json
Evidence subset: test/integration/test_formal_verification_role_aware_post_merge_attestation.py, docs/architecture/formal_verification_role_aware_deployment_receipt.json
Dependencies: FVT-G213
Resource class: cpu-validation
Token class: medium
Estimated tokens: 0
Resources: cpu-validation
Merge fate: objective/FVT-G214
Rejection reasons: none (accepted)

## Goal

After the release-candidate merge, bind its durable terminal supervisor receipt and publish the final deployment attestation without circular tree identity.

## Missing Evidence

- test/integration/test_formal_verification_role_aware_post_merge_attestation.py
- docs/architecture/formal_verification_role_aware_deployment_receipt.json

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
