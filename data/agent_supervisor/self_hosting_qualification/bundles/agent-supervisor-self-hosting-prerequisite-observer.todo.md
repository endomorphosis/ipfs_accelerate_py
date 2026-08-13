# Objective Bundle: agent-supervisor/self-hosting/prerequisite-observer

Source todo: docs/architecture/self_hosting_qualification.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## SHQ-001 Close objective gap: Install the prerequisite-state observer and bind current facts

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prerequisite-observation
- Depends on:
- Outputs: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py, artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Board namespace: self_hosting_qualification.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/.worktrees/ipfs-accelerate-self-hosting-qualification/data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-001-objective-gap-06ca583436ae.md
- Bundle: agent-supervisor/self-hosting/prerequisite-observer
- Bundle shard: data/agent_supervisor/self_hosting_qualification/bundles/agent-supervisor-self-hosting-prerequisite-observer.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 0
- Parallel lane: prerequisite-observer
- Conflict policy: Never modify prerequisite implementation or completion evidence from this task.
- Predicted files: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py, artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Changed paths:
- Context paths: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py, artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- AST symbols: IncrementalSemanticIndex SemanticCapsuleCompiler ContextPacker ContextPackBuilder VerificationReceiptCache IncrementalVerificationPlanner ModelRoutePlanner VerifiedGuiOptimizer IncrementalProofSealer SemanticCompressionGovernor AdversarialAssuranceEngine
- Interfaces: observe_prerequisite_releases, PrerequisiteObservation
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: SHQ-G006
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/2c8b1abad5687f355cfe872b2b1f8a8cd27b9ffe9e7c7c86dd0556be741e6ae0
- Canonical task CID: baguqeerafsfrvowvnb7tkxh6q4vswh4krtjhxh76tz6hzbw5avll45a6nlqa
- Semantic identity: objective-evidence-obligation/v1/0b7a94a549d5acdd90c422f9da902ef94cb70e5fa89c912fb692737ad82b1532
- Acceptance subset: Ordinary observation succeeds even when an upstream system is incomplete, require-terminal mode fails closed, every row binds repository, commit, API mapping, test selector, board state and evidence time, no row claims release from prompt text or branch name alone.
- Preconditions: objective goal SHQ-G006 is schedulable
- Effects: satisfy evidence requirement: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, satisfy evidence requirement: test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py, satisfy evidence requirement: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Evidence subset: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py, artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Resource class: cpu-small
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-small
- Merge fate: objective/SHQ-G006
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/0b7a94a549d5acdd90c422f9da902ef94cb70e5fa89c912fb692737ad82b1532
- Missing evidence: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py, artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Embedding query: self hosting qualification prerequisite completion release board commit API focused tests observer
- AST query: IncrementalSemanticIndex SemanticCapsuleCompiler ContextPacker ContextPackBuilder VerificationReceiptCache IncrementalVerificationPlanner ModelRoutePlanner VerifiedGuiOptimizer IncrementalProofSealer SemanticCompressionGovernor AdversarialAssuranceEngine
- Surplus group: objective/SHQ-G006
- Merge key: 62c9ae9c5db9e72a
- Merge family: objective/SHQ-G006
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
- Todo vector key: 0485db696ba3a322
- Acceptance: Objective scan filed this gap for SHQ-G006. Use evidence in /home/barberb/lift_coding/.worktrees/ipfs-accelerate-self-hosting-qualification/data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-001-objective-gap-06ca583436ae.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py, artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json), and keep the supervisor-fed backlog aligned with the objective heap.  Recognize versioned functional interfaces such as ContextPacker only through an explicit compatibility map; do not manufacture missing facade classes.
