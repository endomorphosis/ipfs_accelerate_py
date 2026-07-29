# CVESIR-001 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 7cc4e4483c6891fc38458f88d483dbaad1598bd5
Goal id: CVESIR-G090
Goal title: Idempotent publish and remote verification command
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P1
Track: publication-tooling
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 0
Bundle: cvefixes-security-ir/release
Parallel lane: cvefixes-security-ir/release
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Provide an authenticated, secret-safe, idempotent upload command and remote Dataset Viewer verification that emits a proposed publication receipt.
AST query: ipfs_datasets_py/scripts/ops/security_ir/publish_cvefixes_security_ir.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_publish_command.py
Conflict policy: Own publication command and tests; tests must not mutate the Hub.
Predicted files: ipfs_datasets_py/scripts/ops/security_ir/publish_cvefixes_security_ir.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_publish_command.py
AST symbols: ipfs_datasets_py/scripts/ops/security_ir/publish_cvefixes_security_ir.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_publish_command.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/e8397eea8558df30579e6b7dd8ea3b36b86d5eb31d498ee027d550b74ffde600
Acceptance subset: Dry-run is default, target/source/release tuple is idempotent, credentials never persist, remote revision and shard/schema verification are mandatory before a receipt is proposed.
Preconditions: objective goal CVESIR-G090 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/scripts/ops/security_ir/publish_cvefixes_security_ir.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_publish_command.py
Evidence subset: ipfs_datasets_py/scripts/ops/security_ir/publish_cvefixes_security_ir.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_publish_command.py
Dependencies: CVESIR-G080
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G090
Rejection reasons: none (accepted)

## Goal

Provide an authenticated, secret-safe, idempotent upload command and remote Dataset Viewer verification that emits a proposed publication receipt.

## Missing Evidence

- ipfs_datasets_py/scripts/ops/security_ir/publish_cvefixes_security_ir.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_publish_command.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
