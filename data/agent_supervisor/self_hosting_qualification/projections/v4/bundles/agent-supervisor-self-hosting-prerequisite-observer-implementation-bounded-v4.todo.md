# Objective Bundle: agent-supervisor/self-hosting/prerequisite-observer-implementation-bounded-v4

Source todo: docs/architecture/self_hosting_qualification.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## SHQ-006 Close objective gap: Install and test the prerequisite-state observer

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prerequisite-observation
- Depends on:
- Outputs: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py; python scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode observe --quiet; python scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode require-terminal --quiet && exit 99 || test "$?" -eq 1
- Board namespace: self_hosting_qualification.todo.md
- Evidence inputs: data/agent_supervisor/self_hosting_qualification/discovery
- Discovery evidence: data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-006-objective-gap-c8da90812f65.md
- Bundle: agent-supervisor/self-hosting/prerequisite-observer-implementation-bounded-v4
- Bundle shard: data/agent_supervisor/self_hosting_qualification/projections/v4/bundles/agent-supervisor-self-hosting-prerequisite-observer-implementation-bounded-v4.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 0
- Parallel lane: prerequisite-observer-implementation-bounded-v4
- Conflict policy: Never modify prerequisite implementation or completion evidence from this task; never read sibling worktrees, operator state, hidden evaluator data, or arbitrary host paths.
- Predicted files: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Changed paths:
- Context paths: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- AST symbols: IncrementalSemanticIndex SemanticCapsuleCompiler ContextPacker ContextPackBuilder VerificationReceiptCache IncrementalVerificationPlanner ModelRoutePlanner VerifiedGuiOptimizer IncrementalProofSealer SemanticCompressionGovernor AdversarialAssuranceEngine
- Interfaces: observe_prerequisite_releases, PrerequisiteObservation
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Generated artifacts:
- Allow concurrent with:
- Goal id: SHQ-G006
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/a327007501d6235cfdd2dd70b96b3c4d2c32b79af4dacd0539dcd2e4942bbf3c
- Canonical task CID: baguqeeraumtqa5ib2yrvz7os3vyls2z4juwdfn426tnm2bjz3tjojfblx46a
- Semantic identity: objective-evidence-obligation/v1/7c534d2f622c24baaac5c98252408f5cbb50b6fa59e35323c5c1c1be04eda68b
- Acceptance subset: `.gitignore` contains the exact narrow exception `!artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`, ordinary observation succeeds when an upstream is incomplete, while `require-terminal` validates the complete snapshot before any output write and leaves no artifact on failure, every row binds a clean repository commit/tree, the exact superproject gitlink and matching submodule HEAD, a complete module-level API or explicit versioned compatibility resolution, current focused-test execution receipts, a fully parsed owner board, evidence time and limitations, selector presence alone is never release evidence, malformed or unreadable modules, receipts or boards are unverifiable, every recognized task block has exactly one recognized status and an unrecognized/missing/duplicate status prevents terminal classification, datasets boards `ipfs_datasets_py/docs/architecture/incremental_semantic_index.todo.md` and `ipfs_datasets_py/docs/architecture/semantic_state_contract.todo.md` are bound, no row claims release from prompt text, a branch name, path presence or a partial symbol match. All discovery and validation reads stay within the disposable task worktree and its declared submodules, recursive search of `/home`, sibling worktrees, supervisor state, or any other host path is prohibited. This goal must not create or change `prerequisite_observation.json`.
- Preconditions: objective goal SHQ-G006 is schedulable
- Effects: satisfy evidence requirement: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, satisfy evidence requirement: test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Evidence subset: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Resource class: cpu-small
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-small
- Merge fate: objective/SHQ-G006
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/7c534d2f622c24baaac5c98252408f5cbb50b6fa59e35323c5c1c1be04eda68b
- Missing evidence: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Embedding query: self hosting qualification prerequisite completion release board commit API focused tests observer
- AST query: IncrementalSemanticIndex SemanticCapsuleCompiler ContextPacker ContextPackBuilder VerificationReceiptCache IncrementalVerificationPlanner ModelRoutePlanner VerifiedGuiOptimizer IncrementalProofSealer SemanticCompressionGovernor AdversarialAssuranceEngine
- Surplus group: objective/SHQ-G006
- Merge key: 9ad3a4d635830842
- Merge family: objective/SHQ-G006
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
- Todo vector key: 10e4ecaca8968d25
- Acceptance: Objective scan filed this gap for SHQ-G006. Use evidence in data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-006-objective-gap-c8da90812f65.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py), and keep the supervisor-fed backlog aligned with the objective heap.  Resolve every required symbol and interface by AST/module inspection in its declared module, recognizing versioned functional interfaces such as `ContextPacker` only through an explicit compatibility map and never manufacturing missing facades. Keep the ten-name result order stable, but discover future releases of currently missing systems through constrained package exports, release manifests and declared owner-board candidates so a new released module does not require another hard-coded missing result. Work only from the current disposable checkout. As bounded prior-attempt context, inspect exactly `git show 63ea88e41227d4d2d424f41051b9e9390c1a1c32 -- scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py`; audit and repair it against this goal rather than trusting or copying it blindly. Do not enumerate unrelated refs or search outside the checkout. Tests include independent dirty-tree, gitlink/HEAD mismatch, partial/malformed API, missing/duplicate/unknown board status, stale-or-presence-only test receipt, failed `require-terminal` no-write and two-phase source-binding counterexamples.
