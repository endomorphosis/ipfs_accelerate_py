# CVESIR-002 Objective Goal Gap

Date: 2026-07-29
Fingerprint: b1aa6d2f7cac89e0ef8af3f27e594a40fbeb9164
Goal id: CVESIR-G080
Goal title: Reproducible Hugging Face release builder
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: release
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 1
Bundle: cvefixes-security-ir/release
Parallel lane: cvefixes-security-ir/release
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Build deterministic Parquet configs, dataset card, manifest, evaluation report, and bounded query client for Publicus/cvefixes-security-ir-graphrag.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_release.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_release.py
Conflict policy: Own local release packaging; do not perform the Hub mutation.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_release.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_release.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_release.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_release.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/7be1a9ce7d7217673f8b94b0e539c37067e35f00e26d364e49d660e2d61f6740
Acceptance subset: Bounded shards and schemas validate, card documents source/license/limitations, release root is stable, validate-only requires no credentials, no secrets/caches/internal bodies enter staging.
Preconditions: objective goal CVESIR-G080 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_release.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_release.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_release.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_release.py
Dependencies: CVESIR-G070
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G080
Rejection reasons: none (accepted)

## Goal

Build deterministic Parquet configs, dataset card, manifest, evaluation report, and bounded query client for Publicus/cvefixes-security-ir-graphrag.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_release.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_release.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
