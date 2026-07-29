# Objective Bundle: logic-formal-verification/runtime-monitoring

Source todo: docs/architecture/logic_formal_verification_expansion.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## LFV-021 Close objective gap: Build generic runtime MTL with portable TypeScript parity

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: monitor
- Depends on: LFV-006, LFV-010
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/package-lock.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py -q && npm --prefix ipfs_datasets_py/typescript/logic-runtime-mtl test
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/logic_formal_verification_expansion/discovery/2026-07-29-lfv-021-objective-gap-bef0fcf7c087.md
- Bundle: logic-formal-verification/runtime-monitoring
- Bundle shard: data/agent_supervisor/logic_formal_verification_expansion/bundles/logic-formal-verification-runtime-monitoring.todo.md
- Bundle strategy: explicit
- Graph parents: LFV-G000
- Graph depth: 1
- Objective heap index: 20
- Parallel lane: logic-formal-verification/runtime-monitoring
- Conflict policy: Own the new monitor packages and parity test; leave crypto-exchange/supervisor monitors as compatibility consumers.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/package-lock.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
- Changed paths:
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
- Interfaces: RuntimeMTLMonitor@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: LFV-G045
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/01f25cf924866bdbe1f9954dcef6294ef24fbde01886b863deb91d5df05622e1
- Canonical task CID: baguqeeraahzfz6jeqzv5xypzsvg455rjj3ze7ppadcdlqy66xeov34cwelqq
- Semantic identity: objective-evidence-obligation/v1/4dc0fc3b6fa86709f1e5e0fcc7b4387ec2c0d1e841c5218cbc62b8174dbeb199
- Acceptance subset: Python and TypeScript agree on interval boundaries, clocks, missing/late events, violations, inconclusive prefixes, and serialization, results always have monitor authority, no-violation-observed never becomes proof.
- Preconditions: objective goal LFV-G045 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, satisfy evidence requirement: ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, satisfy evidence requirement: ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, satisfy evidence requirement: ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, satisfy evidence requirement: ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
- Resource class: cpu-medium
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-medium
- Merge fate: objective/LFV-G045
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/4dc0fc3b6fa86709f1e5e0fcc7b4387ec2c0d1e841c5218cbc62b8174dbeb199
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
- Embedding query: Extract a generic Python finite-trace MTL/LTLf monitor, define portable formula/trace/result schemas, and provide a TypeScript reference implementation over the same golden fixtures.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py
- Surplus group: objective/LFV-G045
- Merge key: 86961a39ad5d1d84
- Merge family: objective/LFV-G045
- Merge role: aggregate
- Work item count: 6
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: db953340d084b8a5
- Acceptance: Objective scan filed this gap for LFV-G045. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/logic_formal_verification_expansion/discovery/2026-07-29-lfv-021-objective-gap-bef0fcf7c087.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/monitoring/runtime_mtl.py, ipfs_datasets_py/typescript/logic-runtime-mtl/package.json, ipfs_datasets_py/typescript/logic-runtime-mtl/tsconfig.json, ipfs_datasets_py/typescript/logic-runtime-mtl/src/index.ts, ipfs_datasets_py/typescript/logic-runtime-mtl/test/runtime_mtl.test.ts, ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
