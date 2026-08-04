# Objective Bundle: formal-verification-tactician/platform-support-classifier

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-064 Close objective gap: Derive exact host support and platform exceptions from the lock

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: deployment-integrity
- Depends on: FVT-006, FVT-041
- Outputs: tools/logic/certification/platform_support.py, test/integration/toolchains/test_formal_verification_platform_support.py
- Validation: python -m pytest test/integration/toolchains/test_formal_verification_platform_support.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-064-objective-gap-64fd42d3cc7f.md
- Bundle: formal-verification-tactician/platform-support-classifier
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-platform-support-classifier.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 10
- Parallel lane: formal-verification-tactician/platform-support-classifier
- Conflict policy: Own platform normalization and classification only; never probe or install tools, infer support from PATH, or convert unavailability into unsupported status.
- Predicted files: tools/logic/certification/platform_support.py, test/integration/toolchains/test_formal_verification_platform_support.py
- Changed paths:
- Context paths: tools/logic/certification/platform_support.py, test/integration/toolchains/test_formal_verification_platform_support.py
- AST symbols: test/integration/toolchains/test_formal_verification_platform_support.py
- Interfaces: FormalVerificationPlatformSupport@1
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G201
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/5dcd6b710e781dc299e9bf3bcca6e5a2afb257ebe2241f9bdacbdcc0000a7192
- Canonical task CID: baguqeeralxgww4iopao4fgpjx454zjxfukx3ev7l4isb7g62zpomaaakogja
- Semantic identity: objective-evidence-obligation/v1/7140fe479d3e8b202596942b2806965d383aa7a0ce9d22a50f075a7a3457dd10
- Acceptance subset: The normalized host key is derived from the running OS and architecture, each tool reports supported_here, unsupported_here, or ambiguous from its own pins and deployment contract, `any` support is honored, absent, contradictory, or ambiguous metadata is a blocker, only an explicit host exclusion can produce a narrow platform exception, linux-aarch64 classifies HyperLTL, AutoHyper, MCHyper, Souffle, and external Runtime MTL as supported under the current lock, external SecPAL as unsupported, and ZKP as a platform-independent deployment binding, a lock mutation that adds or removes linux-aarch64 changes the classification and final digest.
- Preconditions: objective goal FVT-G201 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_formal_verification_platform_support.py
- Evidence subset: test/integration/toolchains/test_formal_verification_platform_support.py
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-validation
- Merge fate: objective/FVT-G201
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/7140fe479d3e8b202596942b2806965d383aa7a0ce9d22a50f075a7a3457dd10
- Missing evidence: test/integration/toolchains/test_formal_verification_platform_support.py
- Embedding query: Give every locked tool an auditable host-platform classification so missing supported capabilities can never be relabeled as exceptions.
- AST query: test/integration/toolchains/test_formal_verification_platform_support.py
- Surplus group: objective/FVT-G201
- Merge key: 925cd5a43b3cc250
- Merge family: objective/FVT-G201
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
- Todo vector key: 2507a9ad47898be8
- Acceptance: Objective scan filed this gap for FVT-G201. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-064-objective-gap-64fd42d3cc7f.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_formal_verification_platform_support.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
