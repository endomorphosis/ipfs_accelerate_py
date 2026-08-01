# FVT-013 Objective Goal Gap

Date: 2026-07-30
Fingerprint: dc61c76c11c948b669d570c0dc731cbb359946c3
Goal id: FVT-G012
Goal title: Execute the full lazy provider matrix through stable surfaces
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: provider-execution
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 12
Bundle: formal-verification-tactician/provider-surface
Parallel lane: formal-verification-tactician/provider-surface
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Register every LFV provider lazily behind the shared protocol, make portfolio execution real rather than plan-only, and expose equivalent availability and execution semantics through Python, datasets MCP, and parent MCP.
AST query: ipfs_datasets_py/tests/integration/logic/test_verification_provider_matrix_api.py, test/api/test_root_mcp_formal_verification_parity.py
Conflict policy: Own registry/public execution wiring and parity tests; do not install providers during discovery or weaken property-specific routing and authority policy.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/registry.py, ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/integration/logic/test_verification_provider_matrix_api.py, test/api/test_root_mcp_formal_verification_parity.py
AST symbols: ipfs_datasets_py/tests/integration/logic/test_verification_provider_matrix_api.py, test/api/test_root_mcp_formal_verification_parity.py
Interfaces: ExecutableProviderMatrix@1, FormalVerificationMCPParity@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/d08a220a092526ee5809ca3dba35657b629657ba7bd3caf7a8bb0bec1c391472
Acceptance subset: SMT, state-model, runtime, authorization, protocol, hyperproperty, ATP, Hammer, and kernel providers are discoverable without import side effects, available lanes execute, absent lanes report unavailable, portfolios preserve typed authority and quarantine contradiction, both MCP roots match the stable schema.
Preconditions: objective goal FVT-G012 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/test_verification_provider_matrix_api.py, satisfy evidence requirement: test/api/test_root_mcp_formal_verification_parity.py
Evidence subset: ipfs_datasets_py/tests/integration/logic/test_verification_provider_matrix_api.py, test/api/test_root_mcp_formal_verification_parity.py
Dependencies: FVT-G006, FVT-G009, FVT-G011
Resource class: cpu-proof-portfolio
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-portfolio
Merge fate: objective/FVT-G012
Rejection reasons: none (accepted)

## Goal

Register every LFV provider lazily behind the shared protocol, make portfolio execution real rather than plan-only, and expose equivalent availability and execution semantics through Python, datasets MCP, and parent MCP.

## Missing Evidence

- ipfs_datasets_py/tests/integration/logic/test_verification_provider_matrix_api.py
- test/api/test_root_mcp_formal_verification_parity.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
