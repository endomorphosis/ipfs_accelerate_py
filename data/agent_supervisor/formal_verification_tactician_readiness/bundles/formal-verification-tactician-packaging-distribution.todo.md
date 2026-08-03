# Objective Bundle: formal-verification-tactician/packaging-distribution

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-084 FVT:: Close formal-verification packaging and distribution coverage

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: dependency-integrity
- Depends on: FVT-036, FVT-064
- Outputs: setup.py, pyproject.toml, requirements.txt, ipfs_datasets_py/setup.py, ipfs_datasets_py/pyproject.toml, ipfs_datasets_py/requirements.txt, ipfs_datasets_py/requirements-lazy.txt, ipfs_datasets_py/requirements-theorem-provers.txt, test/packaging/test_formal_verification_distribution_contract.py
- Validation: python -m pytest test/packaging/test_logic_verification_clean_install.py test/packaging/test_formal_verification_distribution_contract.py test/test_pip_install_simulation.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-084-objective-gap-e66440ea4b08.md
- Bundle: formal-verification-tactician/packaging-distribution
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-packaging-distribution.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/packaging-distribution
- Conflict policy: Own packaging metadata and clean-install gates; do not make heavyweight prover binaries mandatory Python dependencies, hide missing wheel content with PYTHONPATH, or install anything during import and inventory.
- Predicted files: setup.py, pyproject.toml, requirements.txt, ipfs_datasets_py/setup.py, ipfs_datasets_py/pyproject.toml, ipfs_datasets_py/requirements.txt, ipfs_datasets_py/requirements-lazy.txt, ipfs_datasets_py/requirements-theorem-provers.txt, test/packaging/test_formal_verification_distribution_contract.py
- Changed paths:
- Context paths: setup.py, pyproject.toml, requirements.txt, ipfs_datasets_py/setup.py, ipfs_datasets_py/pyproject.toml, ipfs_datasets_py/requirements.txt, ipfs_datasets_py/requirements-lazy.txt, ipfs_datasets_py/requirements-theorem-provers.txt, test/packaging/test_formal_verification_distribution_contract.py
- AST symbols: test/packaging/test_logic_verification_clean_install.py, test/packaging/test_formal_verification_distribution_contract.py
- Interfaces: FormalVerificationDistributionContract@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G215
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/8f638d914faec5658b32e744609c26a7f3f13e903b13cb8d264562c83e1e4537
- Canonical task CID: baguqeerar5ry3ekpv3cwlczs45cgbhbgu7z7cpuqhmj4xdjgivrmqpq6iu3q
- Semantic identity: objective-evidence-obligation/v1/8787575a47c3eb507479b38b3d10a32ab8d69405ec8bcfd27ad1fcd3ff3d808f
- Acceptance subset: Root and datasets setup.py, pyproject metadata, requirements files, and extras have one machine-checked dependency inventory, namespace-package discovery includes logic backends, software verification, installer plugins, and runtime assets in built wheels, every declared plugin module exists, a clean isolated wheel install imports and inventories the Logic API without network access, downloads, builds, user-site leakage, editable-source leakage, or installation side effects, optional native and external provers remain optional and are surfaced as unavailable rather than breaking base installation.
- Preconditions: objective goal FVT-G215 is schedulable
- Effects: satisfy evidence requirement: test/packaging/test_formal_verification_distribution_contract.py
- Evidence subset: test/packaging/test_formal_verification_distribution_contract.py
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok-implement, codex-review
- Resources: cpu-validation
- Merge fate: objective/FVT-G215
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/8787575a47c3eb507479b38b3d10a32ab8d69405ec8bcfd27ad1fcd3ff3d808f
- Missing evidence: test/packaging/test_formal_verification_distribution_contract.py
- Embedding query: Make every public formal-verification module, reviewed installer plugin, runtime asset, and optional dependency declaration survive both source and clean wheel installations.
- AST query: test/packaging/test_logic_verification_clean_install.py, test/packaging/test_formal_verification_distribution_contract.py
- Surplus group: objective/FVT-G215
- Merge key: 0c67d2994a2a48b0
- Merge family: objective/FVT-G215
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
- Todo vector key: bc5fe50251ecc038
- Acceptance: Objective scan filed this gap for FVT-G215. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-084-objective-gap-e66440ea4b08.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/packaging/test_formal_verification_distribution_contract.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
