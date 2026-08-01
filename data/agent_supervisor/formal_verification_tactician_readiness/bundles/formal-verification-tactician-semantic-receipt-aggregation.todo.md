# Objective Bundle: formal-verification-tactician/semantic-receipt-aggregation

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-065 Close objective gap: Aggregate full specialized receipts with composite lane handlers

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: certification-integrity
- Depends on: FVT-040, FVT-038, FVT-039, FVT-042, FVT-043, FVT-044, FVT-048, FVT-045, FVT-049, FVT-050, FVT-046, FVT-051, FVT-052, FVT-047, FVT-064, FVT-062
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, tools/logic/certify_formal_verification_toolchains.py, test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py test/integration/test_formal_verification_real_tool_matrix.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-065-objective-gap-0bbb37a543ad.md
- Bundle: formal-verification-tactician/semantic-receipt-aggregation
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-semantic-receipt-aggregation.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 11
- Parallel lane: formal-verification-tactician/semantic-receipt-aggregation
- Conflict policy: Own role registration and lossless aggregation; do not run installers, collapse by check kind, discard raw receipt identity, or let one tool overwrite a sibling handler.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, tools/logic/certify_formal_verification_toolchains.py, test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/toolchain_roles.py, tools/logic/certify_formal_verification_toolchains.py, test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- AST symbols: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- Interfaces: FormalVerificationSpecializedReceiptAggregation@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G203
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/aa62a2917020434ac44a13363c8ab96e7e70b356e352664b0ffd3f49ff0e4924
- Canonical task CID: baguqeeravjrkfelqebbuvrckcm3dzcvznz7hbm2w4njgmsyp7u7ut7yojesa
- Semantic identity: objective-evidence-obligation/v1/8d7d04f775a3f8b4e0fdc64c6b9480ee02eb0cad857628eed078ed8b604dba24
- Acceptance subset: Handlers are keyed by `(lane_id, tool_id)` or a composite lane returns distinct per-tool receipts, kernel retains Lean, Rocq, and Isabelle evidence and protocol retains Tamarin and ProVerif evidence, state, protocol, kernel, ATP, hyperproperty, advisor, in-process and external authorization, in-process and external Runtime MTL, and ZKP certifiers are all represented, every check, case, binding, executable, artifact, dependency, source, authority ceiling, and raw receipt digest participates in the top-level digest, a second failed check of an already-present kind blocks promotion, mutating any retained check or identity changes the certificate digest.
- Preconditions: objective goal FVT-G203 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- Evidence subset: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-validation
- Merge fate: objective/FVT-G203
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/8d7d04f775a3f8b4e0fdc64c6b9480ee02eb0cad857628eed078ed8b604dba24
- Missing evidence: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- Embedding query: Replace first-check and one-handler-per-lane fan-in with lossless, per-tool specialized evidence aggregation.
- AST query: test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py
- Surplus group: objective/FVT-G203
- Merge key: 4faac722330b5af9
- Merge family: objective/FVT-G203
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: f14b9b92fb5e91f6
- Acceptance: Objective scan filed this gap for FVT-G203. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-065-objective-gap-0bbb37a543ad.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
