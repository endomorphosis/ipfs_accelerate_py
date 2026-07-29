# CVESIR-003 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 6d0717c71e5170b9748c1aa40d1adcac99640963
Goal id: CVESIR-G130
Goal title: Pinned Hugging Face Security IR source adapter
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: hf-integration
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 2
Bundle: cvefixes-security-ir/security-ir
Parallel lane: cvefixes-security-ir/security-ir
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Load the derived dataset by exact repo and revision, verify its manifest/shards/row identities, and expose bounded canonical Security IR declarations and policy lookup.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_source.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_source.py
Conflict policy: Own CVEfixes HF adapter and package exports; reuse existing huggingface snapshot contracts.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_source.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_source.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_source.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_source.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/03c9893388e5a705115d00865b721935ff0d27327f2de1466319073a68ad2292
Acceptance subset: Floating revisions, manifest drift, missing shards, row tampering, unknown schema, and candidate-as-authority fail closed, offline cache preserves revision identity.
Preconditions: objective goal CVESIR-G130 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_source.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_source.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_source.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_source.py
Dependencies: CVESIR-G080, CVESIR-G110
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G130
Rejection reasons: none (accepted)

## Goal

Load the derived dataset by exact repo and revision, verify its manifest/shards/row identities, and expose bounded canonical Security IR declarations and policy lookup.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/hf_source.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_hf_source.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
