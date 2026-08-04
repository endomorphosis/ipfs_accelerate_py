# Objective Bundle: formal-verification-tactician/replayed-core-external-semantics

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-096 Close objective gap: Certify replayed state, protocol, kernel, and ATP semantics

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: semantic-certification
- Depends on: FVT-060, FVT-076, FVT-058, FVT-075, FVT-057, FVT-074, FVT-054, FVT-071, FVT-094
- Outputs: tools/logic/certify_formal_verification_replayed_core_semantics.py, docs/architecture/formal_verification_replayed_core_external_semantics.json, test/integration/toolchains/test_replayed_core_external_semantics.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_replayed_core_external_semantics.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-096-objective-gap-8a708a76a37d.md
- Bundle: formal-verification-tactician/replayed-core-external-semantics
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-replayed-core-external-semantics.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 4
- Parallel lane: formal-verification-tactician/replayed-core-external-semantics
- Conflict policy: Own the cross-family replay aggregator and receipt; reuse family certifiers without changing their authority ceilings or installers.
- Predicted files: tools/logic/certify_formal_verification_replayed_core_semantics.py, docs/architecture/formal_verification_replayed_core_external_semantics.json, test/integration/toolchains/test_replayed_core_external_semantics.py
- Changed paths:
- Context paths: tools/logic/certify_formal_verification_replayed_core_semantics.py, docs/architecture/formal_verification_replayed_core_external_semantics.json, test/integration/toolchains/test_replayed_core_external_semantics.py
- AST symbols: docs/architecture/formal_verification_replayed_core_external_semantics.json, test/integration/toolchains/test_replayed_core_external_semantics.py
- Interfaces: ReplayedCoreExternalSemantics@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G228
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/502e1abaa4ba466503e33627e3b01fafa9d0daffd180d80bf40b0c5fed6ffdcc
- Canonical task CID: baguqeerakaxbvovexjdgka7dgyt6hma7v6u5bwx72ganqc7ubmgf73lp7xga
- Semantic identity: objective-evidence-obligation/v1/91b7a47d3752901036fc6127ac6d9c2b1b205958667db8e40bc3e63db0093bde
- Acceptance subset: TLC and Apalache execute bounded state/safety/liveness cases, Tamarin and ProVerif execute protocol secrecy/authentication and mutation cases, Rocq/Coq and Isabelle check accepted/rejected proof objects in their genuine kernels, Vampire and E execute theorem/non-theorem/resource-bound cases, each provider has independent positive, negative, mutation, replay, malformed, timeout, and disagreement evidence bound to the managed identity, Maude and OPAM remain support-only, no fixture, parser, wrapper, advisor, or other provider can supply a missing engine's semantic or authority axis.
- Preconditions: objective goal FVT-G228 is schedulable
- Effects: satisfy evidence requirement: docs/architecture/formal_verification_replayed_core_external_semantics.json, satisfy evidence requirement: test/integration/toolchains/test_replayed_core_external_semantics.py
- Evidence subset: docs/architecture/formal_verification_replayed_core_external_semantics.json, test/integration/toolchains/test_replayed_core_external_semantics.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G228
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/91b7a47d3752901036fc6127ac6d9c2b1b205958667db8e40bc3e63db0093bde
- Missing evidence: docs/architecture/formal_verification_replayed_core_external_semantics.json, test/integration/toolchains/test_replayed_core_external_semantics.py
- Embedding query: Re-execute and bind genuine managed semantics for the supported state-model, protocol, proof-kernel, and automated-theorem-prover families in the unified environment.
- AST query: docs/architecture/formal_verification_replayed_core_external_semantics.json, test/integration/toolchains/test_replayed_core_external_semantics.py
- Surplus group: objective/FVT-G228
- Merge key: c1c6d47806dd58fd
- Merge family: objective/FVT-G228
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
- Todo vector key: cc0d1324c3b8da2f
- Acceptance: Objective scan filed this gap for FVT-G228. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-096-objective-gap-8a708a76a37d.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (docs/architecture/formal_verification_replayed_core_external_semantics.json, test/integration/toolchains/test_replayed_core_external_semantics.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
