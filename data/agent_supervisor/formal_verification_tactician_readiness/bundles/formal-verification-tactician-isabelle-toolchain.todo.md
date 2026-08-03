# Objective Bundle: formal-verification-tactician/isabelle-toolchain

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-049 Close objective gap: Install and semantically certify Isabelle

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: external-capability
- Depends on: FVT-041
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/backends/test_kernel_backends.py test/integration/toolchains/test_isabelle_toolchain_certification.py test/integration/test_formal_verification_real_tool_matrix.py -k isabelle -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-049-objective-gap-985a2c93ea4e.md
- Bundle: formal-verification-tactician/isabelle-toolchain
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-isabelle-toolchain.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 48
- Parallel lane: formal-verification-tactician/isabelle-toolchain
- Conflict policy: Own the Isabelle installer plugin, handler, and test; observe an explicit large-download/storage budget and do not edit the shared lock or central certificate.
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
- Changed paths:
- Context paths: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, tools/logic/certification/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
- AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
- Interfaces: IsabelleToolchainCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G151
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/f9138f4bf10c77e00f798d4d2455fc9be4b14229a0a04224bba59240114babac
- Canonical task CID: baguqeera7ejy6s7rbr36ad3zrvgsivp4tpslcqrjucqeejf3uwjeaeklvowa
- Semantic identity: objective-evidence-obligation/v1/0c06bcc700647549f133f76ae385c2a83227eda44b8e647d4f6bf152268cb569
- Acceptance subset: Explicit strict installation selects Isabelle2025-2, a checked theory/session passes while bad proof, mutated assumptions/conclusion, replay mismatch, malformed output, timeout, and wrong installation fail, theory heap, session, imports, source, property, and exact tool identity are bound, Hammer remains proposal-only until kernel reconstruction.
- Preconditions: objective goal FVT-G151 is schedulable
- Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, satisfy evidence requirement: test/integration/toolchains/test_isabelle_toolchain_certification.py
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
- Resource class: large-kernel-toolchain
- Token class: medium
- Estimated tokens: 0
- Resources: large-kernel-toolchain
- Merge fate: objective/FVT-G151
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/0c06bcc700647549f133f76ae385c2a83227eda44b8e647d4f6bf152268cb569
- Missing evidence: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
- Embedding query: Complete the pinned Isabelle installation and real session/kernel certification used for reconstruction and Hammer validation.
- AST query: ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py
- Surplus group: objective/FVT-G151
- Merge key: c07f3ab5b106d297
- Merge family: objective/FVT-G151
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
- Todo vector key: b58ccc64b23f6396
- Acceptance: Objective scan filed this gap for FVT-G151. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-049-objective-gap-985a2c93ea4e.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/isabelle.py, test/integration/toolchains/test_isabelle_toolchain_certification.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
