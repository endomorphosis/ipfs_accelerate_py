# Objective Bundle: logic-formal-verification/state-model-checking

Source todo: docs/architecture/logic_formal_verification_expansion.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## LFV-023 Close objective gap: Generalize TLA+, TLC, and Apalache state-model checking

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: prover
- Depends on: LFV-007, LFV-012, LFV-010, LFV-018
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py
- Validation: python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py -q
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/logic_formal_verification_expansion/discovery/2026-07-29-lfv-023-objective-gap-8e466002b28b.md
- Bundle: logic-formal-verification/state-model-checking
- Bundle shard: data/agent_supervisor/logic_formal_verification_expansion/bundles/logic-formal-verification-state-model-checking.todo.md
- Bundle strategy: explicit
- Graph parents: LFV-G000
- Graph depth: 1
- Objective heap index: 22
- Parallel lane: logic-formal-verification/state-model-checking
- Conflict policy: Own the new TLA backend package/test; port generic behavior without deleting or breaking the supervisor-local facade.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py
- Changed paths:
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py
- Interfaces: TLABackend@1, TLCBackend@1, ApalacheBackend@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: LFV-G044
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/b78dfccba841cc4e15c63731d972f91da32183d9bd6a7b77166c04ad49853adb
- Canonical task CID: baguqeeraw6g7zs5iihge4fogg4y5s4xzdwrsda6zxvvhw5ywnqck2smfhlnq
- Semantic identity: objective-evidence-obligation/v1/991b9bb1f42a73fd7556dc05231d2f9d132c7a86af499904db883141f6a2efbd
- Acceptance subset: Generated modules/configs are deterministic and source mapped, state, concurrency, rely/guarantee, and refinement projections disclose losses, TLC and Apalache capabilities/bounds differ explicitly, counterexamples parse and replay, liveness/fairness limitations are disclosed, absent JVM/tools return unavailable.
- Preconditions: objective goal LFV-G044 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-proof-solver
- Merge fate: objective/LFV-G044
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/991b9bb1f42a73fd7556dc05231d2f9d132c7a86af499904db883141f6a2efbd
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py
- Embedding query: Extract the supervisor state-model implementation into reusable TLA translation plus distinct bounded TLC and Apalache backends.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py
- Surplus group: objective/LFV-G044
- Merge key: a27cf4d51ebaaea2
- Merge family: objective/LFV-G044
- Merge role: aggregate
- Work item count: 3
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: b555be299598e086
- Acceptance: Objective scan filed this gap for LFV-G044. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/logic_formal_verification_expansion/discovery/2026-07-29-lfv-023-objective-gap-8e466002b28b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/compiler.py, ipfs_datasets_py/ipfs_datasets_py/logic/backends/tla/runners.py, ipfs_datasets_py/tests/integration/logic/backends/test_tla_model_checkers.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
