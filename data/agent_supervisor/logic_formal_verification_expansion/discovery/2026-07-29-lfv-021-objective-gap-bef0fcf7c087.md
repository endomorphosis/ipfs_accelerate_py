# LFV-021 Objective Goal Gap

Date: 2026-07-29
Fingerprint: bef0fcf7c087fa32caf53a03876e6a1b67afc052
Goal id: LFV-G045
Goal title: Build generic runtime MTL with portable TypeScript parity
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P1
Track: monitor
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 20
Bundle: logic-formal-verification/runtime-monitoring
Parallel lane: logic-formal-verification/runtime-monitoring
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Extract a generic Python finite-trace MTL/LTLf monitor, define portable formula/trace/result schemas, and provide a TypeScript reference implementation over the same golden fixtures.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
Conflict policy: Own the new monitor packages and parity test; leave crypto-exchange/supervisor monitors as compatibility consumers.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/package-lock.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
Interfaces: RuntimeMTLMonitor@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/4dc0fc3b6fa86709f1e5e0fcc7b4387ec2c0d1e841c5218cbc62b8174dbeb199
Acceptance subset: Python and TypeScript agree on interval boundaries, clocks, missing/late events, violations, inconclusive prefixes, and serialization, results always have monitor authority, no-violation-observed never becomes proof.
Preconditions: objective goal LFV-G045 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, satisfy evidence requirement: ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, satisfy evidence requirement: ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, satisfy evidence requirement: ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, satisfy evidence requirement: ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
Dependencies: LFV-G013, LFV-G023
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/LFV-G045
Rejection reasons: none (accepted)

## Goal

Extract a generic Python finite-trace MTL/LTLf monitor, define portable formula/trace/result schemas, and provide a TypeScript reference implementation over the same golden fixtures.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py
- ipfs_datasets_py/typescript/logic-runtime-mtl/package.json
- ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json
- ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts
- ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts
- ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
