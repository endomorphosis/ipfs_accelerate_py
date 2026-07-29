# CVESIR-007 Objective Goal Gap

Date: 2026-07-29
Fingerprint: f856e630549d9ae9b6d4f62bc4e267888a758d5b
Goal id: CVESIR-G070
Goal title: Leakage-safe evaluation and promotion gates
Objective heap: docs/architecture/cvefixes_security_ir.objectives.md
Priority: P0
Track: evaluation
Status: todo
Schedulable: true
Review only: false
Parent goals: CVESIR-G000
Graph depth: 1
Objective heap index: 6
Bundle: cvefixes-security-ir/release
Parallel lane: cvefixes-security-ir/release
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Implement repository/CVE/body-isolated splits, vulnerable-positive/fixed-negative metrics, calibration, adversarial injection tests, and explicit review/promotion decisions.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/evaluation.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_evaluation.py
Conflict policy: Own evaluation and split policy; do not rewrite source, graph, or Security IR.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/evaluation.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_evaluation.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/evaluation.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_evaluation.py
Interfaces: none
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/9575509a828036d4f2a03b342e14f19c03c9c81c2b20634b6c09b9d5cda582ce
Acceptance subset: Leakage checks cover repo, commit, body hash, and near duplicates, metrics are stratified, fixed negatives cannot inherit vulnerable labels, thresholds are measured, failed gates cannot promote candidates.
Preconditions: objective goal CVESIR-G070 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/evaluation.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_evaluation.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/evaluation.py, ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_evaluation.py
Dependencies: CVESIR-G040, CVESIR-G050, CVESIR-G060, CVESIR-G110, CVESIR-G120
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/CVESIR-G070
Rejection reasons: none (accepted)

## Goal

Implement repository/CVE/body-isolated splits, vulnerable-positive/fixed-negative metrics, calibration, adversarial injection tests, and explicit review/promotion decisions.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/cvefixes/evaluation.py
- ipfs_datasets_py/tests/unit/logic/security_ir/cvefixes/test_evaluation.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
