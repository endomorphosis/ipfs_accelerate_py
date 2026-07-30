# CVESIR-010 Objective Goal Gap

Date: 2026-07-29
Fingerprint: b6a4cf6a4106f3175088cb9d5491110cd5329b6d
Goal id: CVESIR-G030
Goal title: Licensing, privacy, poisoning, and release policy
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: governance
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 9
Bundle: cvefixes-security-ir/source
Parallel lane: cvefixes-security-ir/source
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Define public/internal body profiles, license provenance, PII/secret scanning, prompt-injection treatment, redaction receipts, and publication admission.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/release_policy.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_release_policy.py
Conflict policy: Own release policy only; do not upload or change shared Security IR authority.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/release_policy.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_release_policy.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/release_policy.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_release_policy.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/7a014d0e6331197ff4ef7f020ee3a5eb5c30673014ab606489b603592f6d6e7b
Acceptance subset: Default public profile excludes unrestricted full bodies, provenance is retained, source instructions remain inert, detected secrets, unsafe paths, unreviewed licenses, and policy drift block release.
Preconditions: objective goal CVESIR-G030 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/release_policy.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_release_policy.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/release_policy.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_release_policy.py
Dependencies: none
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G030
Rejection reasons: none (accepted)

## Goal

Define public/internal body profiles, license provenance, PII/secret scanning, prompt-injection treatment, redaction receipts, and publication admission.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/release_policy.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_release_policy.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
