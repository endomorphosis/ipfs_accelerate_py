# Objective Bundle: formal-verification-tactician/runtime-monitor-toolchains

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-052 Close objective gap: Install and certify external Runtime MTL parity

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: external-capability
- Depends on: FVT-039, FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py test/integration/toolchains/test_external_runtime_mtl_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k 'runtime_mtl or mtl' -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-052-objective-gap-80e15787341b.md
- Bundle: formal-verification-tactician/runtime-monitor-toolchains
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-runtime-monitor-toolchains.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 51
- Parallel lane: formal-verification-tactician/runtime-monitor-toolchains
- Conflict policy: Own the external monitor installer plugin, parity handler, and test; do not edit the in-process semantic reference lane.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, test/integration/toolchains/test_external_runtime_mtl_certification.py
- Interfaces: ExternalRuntimeMTLCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G181
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/cc16054b9c6bfe2d94ed67b6fe342474899ed5b36bedba966c6bcb38b808c891
- Canonical task CID: baguqeerazqlaks44np7c3fhnm63p4nbeosez5vntnpw3vftmnpftroaizciq
- Semantic identity: objective-evidence-obligation/v1/328d0e74fd7c9c414d3cf0377c90689e8204aa82f13000ae835d9ce3add37029
- Acceptance subset: Explicit strict installation selects an exact external monitor, Python, TypeScript, and external implementations agree on satisfied/violated golden traces, boundary intervals, mutations, shortest-prefix replay, malformed input, and bounds or quarantine disagreement, finite-trace authority is preserved and no global correctness claim is inferred.
- Preconditions: objective goal FVT-G181 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, satisfy evidence requirement: test/integration/toolchains/test_external_runtime_mtl_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, test/integration/toolchains/test_external_runtime_mtl_certification.py
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-validation
- Merge fate: objective/FVT-G181
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/328d0e74fd7c9c414d3cf0377c90689e8204aa82f13000ae835d9ce3add37029
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, test/integration/toolchains/test_external_runtime_mtl_certification.py
- Embedding query: Replace the external Runtime MTL gap with a pinned parity engine and cross-runtime semantic disagreement checks.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, test/integration/toolchains/test_external_runtime_mtl_certification.py
- Surplus group: objective/FVT-G181
- Merge key: 22004bb1f06336f7
- Merge family: objective/FVT-G181
- Merge role: aggregate
- Work item count: 2
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: fb038642244fb4f4
- Acceptance: Objective scan filed this gap for FVT-G181. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-052-objective-gap-80e15787341b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, test/integration/toolchains/test_external_runtime_mtl_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
