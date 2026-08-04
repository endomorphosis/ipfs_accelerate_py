# Objective Bundle: formal-verification-tactician/ergoai-live-toolchain

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-085 FVT:: Implement the genuine ErgoAI advisor-toolchain path

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-064, FVT-087
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, tools/logic/certification/advisors.py, test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_ergoai_live_toolchain_contract.py test/integration/toolchains/test_advisor_role_certification.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-085-objective-gap-fdb5dfbe2504.md
- Bundle: formal-verification-tactician/ergoai-live-toolchain
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-ergoai-live-toolchain.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 1
- Parallel lane: formal-verification-tactician/ergoai-live-toolchain
- Conflict policy: Own ErgoAI provenance, dependencies, lazy installer, wrapper, and bounded semantics; never scrape an unauthoritative artifact, download during certification, treat wrapper fixtures as live execution, or elevate an advisor verdict to theorem authority.
- Predicted files: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, tools/logic/certification/advisors.py, test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- Changed paths:
- Context paths: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, tools/logic/certification/advisors.py, test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- AST symbols: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- Interfaces: ErgoAILiveToolchainContract@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G218
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/4e83e3b9c072fd1356fa3c4294673ca851a98770a5d7832973343fd6aa5817e3
- Canonical task CID: baguqeeraj2b6hooaol6rgvx2hrbjizz4vbi2tb3quxlygkltgq75nksyc7rq
- Semantic identity: objective-evidence-obligation/v1/8bea67970a56a02025411e374670d7c59ebb920ea7df71b904344f9089f1abc4
- Acceptance subset: The lock binds the official ErgoAI distribution or reviewed source revision, license and acquisition conditions, archive/source digests, XSB and every runtime/build dependency, supported OS/architecture matrix, entry point, and exact identity probe, explicit lazy installation is staged, checksum-verified, atomic, relocatable, and offline after acquisition, live entailment, non-entailment, contradiction, rule/query mutation, deterministic replay, malformed input, timeout, and resource-bound cases execute through ErgoAI, results remain proposal or candidate evidence until reconstructed or checked by an independent proof authority.
- Preconditions: objective goal FVT-G218 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- Evidence subset: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- Resource class: cpu-proof-solver
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: cpu-proof-solver
- Merge fate: objective/FVT-G218
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/8bea67970a56a02025411e374670d7c59ebb920ea7df71b904344f9089f1abc4
- Missing evidence: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- Embedding query: Replace ErgoAI wrapper and proposal-only assumptions with a locked official distribution, dependency-complete lazy installer, and bounded live semantic adapter while preserving advisor authority ceilings.
- AST query: test/integration/toolchains/test_ergoai_live_toolchain_contract.py
- Surplus group: objective/FVT-G218
- Merge key: c3c1e3e9a1a67258
- Merge family: objective/FVT-G218
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
- Todo vector key: 4e37cb79c79de031
- Acceptance: Objective scan filed this gap for FVT-G218. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-085-objective-gap-fdb5dfbe2504.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_ergoai_live_toolchain_contract.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
