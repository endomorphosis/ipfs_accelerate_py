# FVT-015 Objective Goal Gap

Date: 2026-07-30
Fingerprint: 99bf5f408e4f53adf286ed5751b64956901ff2f2
Goal id: FVT-G041
Goal title: Make every counterexample exactly replayable
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: counterexample-replay
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 14
Bundle: formal-verification-tactician/counterexamples
Parallel lane: formal-verification-tactician/counterexamples
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Define safe replay recipes and receipts that reconstruct the exact property violation from immutable source/model/tool/policy/bound identities without exposing private material.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/replay.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py
Conflict policy: Own replay contracts/runtime and tests; use the universal bounded runner and do not reinterpret provider syntax outside its adapter.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/replay.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/replay.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py
Interfaces: CounterexampleReplay@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/f0f0755058fc658bda66da7e8d810dc4afccd1db3cbcd710d2f7055c4acb0ef4
Acceptance subset: Corpus witnesses replay under their exact identities and fail binding on changed tree/property/assumption/tool/bound, unavailable tools return unavailable rather than success, raw private artifacts remain out of public recipes, replay result is content addressed.
Preconditions: objective goal FVT-G041 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/replay.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/replay.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py
Dependencies: FVT-G011, FVT-G040
Resource class: cpu-proof-replay
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-replay
Merge fate: objective/FVT-G041
Rejection reasons: none (accepted)

## Goal

Define safe replay recipes and receipts that reconstruct the exact property violation from immutable source/model/tool/policy/bound identities without exposing private material.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/replay.py
- ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
