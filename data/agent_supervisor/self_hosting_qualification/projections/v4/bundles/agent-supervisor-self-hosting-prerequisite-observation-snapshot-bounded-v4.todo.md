# Objective Bundle: agent-supervisor/self-hosting/prerequisite-observation-snapshot-bounded-v4

Source todo: docs/architecture/self_hosting_qualification.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## SHQ-007 Close objective gap: Generate the post-merge prerequisite observation snapshot

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prerequisite-observation
- Depends on: SHQ-006
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py; python scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode observe --quiet; python scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode require-terminal --quiet && exit 99 || test "$?" -eq 1
- Board namespace: self_hosting_qualification.todo.md
- Evidence inputs: data/agent_supervisor/self_hosting_qualification/discovery
- Discovery evidence: data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-007-objective-gap-925ecd59554c.md
- Bundle: agent-supervisor/self-hosting/prerequisite-observation-snapshot-bounded-v4
- Bundle shard: data/agent_supervisor/self_hosting_qualification/projections/v4/bundles/agent-supervisor-self-hosting-prerequisite-observation-snapshot-bounded-v4.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 1
- Parallel lane: prerequisite-observation-snapshot-bounded-v4
- Conflict policy: Do not edit `.gitignore`, observer implementation, tests, prerequisite owners, release admission, policies, keys or generated supervisor state; never read arbitrary host paths.
- Predicted files: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Changed paths:
- Context paths: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- AST symbols: PrerequisiteObservation observation_to_json write_observation_artifact
- Interfaces: PrerequisiteObservation@1
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Generated artifacts:
- Allow concurrent with:
- Goal id: SHQ-G007
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/d6bb59bcadd2313cd400ac5b11a52d310b2a4cbc68f28732fb2ae0857d6eeffe
- Canonical task CID: baguqeera225vtpfn2iytzvaavrnrdjjngefsutf4ndziomx3flqik7lo577a
- Semantic identity: objective-evidence-obligation/v1/2b17e60a3516287ce21a56894825629c819e5da2113bd1262eb6e0cbd8e55d83
- Acceptance subset: The only changed path is `artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`, the task begins from the clean merged current bounded G006 successor, never the retired SHQ-002 worktree or rescue branch, and the snapshot binds exactly that pre-observation outer commit/tree, recursive gitlinks and matching submodule HEADs while excluding only its own artifact path, all ten rows and limitations are current and deterministic, all reads stay within the disposable task worktree and its three declared gitlinks, the artifact declares that it is neither completion nor proof nor release authority, its later artifact commit is an evidence projection and never claimed as the observed source, native validation and local two-pass completion receipts independently bind the clean post-artifact tree.
- Preconditions: objective goal SHQ-G007 is schedulable
- Effects: satisfy evidence requirement: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Evidence subset: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Resource class: cpu-small
- Token class: small
- Estimated tokens: 0
- Resources: cpu-small
- Merge fate: objective/SHQ-G007
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/2b17e60a3516287ce21a56894825629c819e5da2113bd1262eb6e0cbd8e55d83
- Missing evidence: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Embedding query: post merge prerequisite observation snapshot clean source projection recursive gitlinks
- AST query: PrerequisiteObservation observation_to_json write_observation_artifact
- Surplus group: objective/SHQ-G007
- Merge key: fee3d856a0d71724
- Merge family: objective/SHQ-G007
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
- Todo vector key: bdca5f32aa0623f1
- Acceptance: Objective scan filed this gap for SHQ-G007. Use evidence in data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-007-objective-gap-925ecd59554c.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json), and keep the supervisor-fed backlog aligned with the objective heap.  Refuse dirty input, gitlink/HEAD mismatch, a retired predecessor identity or an already modified output; never read sibling worktrees or operator state, repair or upgrade a prerequisite, or turn an observe result into terminal admission.
