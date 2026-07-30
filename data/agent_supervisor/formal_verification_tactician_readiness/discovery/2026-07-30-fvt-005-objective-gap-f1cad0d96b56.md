# FVT-005 Objective Goal Gap

Date: 2026-07-30
Fingerprint: f1cad0d96b56bdcbed369dc1d2486bd531d6569d
Goal id: FVT-G009
Goal title: Route every external tool through one bounded lifecycle
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: tool-runtime
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 4
Bundle: formal-verification-tactician/runtime
Parallel lane: formal-verification-tactician/runtime
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Remove direct unbounded subprocess execution from concrete backends and version probes and enforce one injected process lifecycle across native, JVM, OCaml/opam, kernel, and WASM tools.
AST query: ipfs_datasets_py/tests/integration/logic/backends/test_all_backend_process_isolation.py
Conflict policy: Own shared process lifecycle integration and isolation tests; change backend invocation mechanics without changing their formula semantics or result authority.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/process.py, ipfs_datasets_py/tests/integration/logic/backends/test_all_backend_process_isolation.py
AST symbols: ipfs_datasets_py/tests/integration/logic/backends/test_all_backend_process_isolation.py
Interfaces: UniversalBoundedToolLifecycle@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/5128548586682af1fd3ac96e250fc387227cc62b6d729db7279ea3e443dcf39f
Acceptance subset: SMT/differential and every other adapter and probe use argument arrays, private workspaces, process-tree termination, wall/memory/CPU/output bounds, cancellation, redaction, and cleanup, adversarial fake tools cannot escape paths, leave children, flood output, or trigger installation/network access.
Preconditions: objective goal FVT-G009 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/backends/test_all_backend_process_isolation.py
Evidence subset: ipfs_datasets_py/tests/integration/logic/backends/test_all_backend_process_isolation.py
Dependencies: FVT-G005
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/FVT-G009
Rejection reasons: none (accepted)

## Goal

Remove direct unbounded subprocess execution from concrete backends and version probes and enforce one injected process lifecycle across native, JVM, OCaml/opam, kernel, and WASM tools.

## Missing Evidence

- ipfs_datasets_py/tests/integration/logic/backends/test_all_backend_process_isolation.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
