# Objective Bundle: formal-verification-tactician/runtime-mtl-external-runtime

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-056 Close objective gap: Build and certify an independent external Runtime MTL engine

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: external-capability
- Depends on: FVT-039, FVT-052, FVT-064, FVT-062
- Outputs: ipfs_datasets_py/typescript/logic-runtime-mtl, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py test/integration/toolchains/test_external_runtime_mtl_certification.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-056-objective-gap-110d71dbeb6c.md
- Bundle: formal-verification-tactician/runtime-mtl-external-runtime
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-runtime-mtl-external-runtime.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 2
- Parallel lane: formal-verification-tactician/runtime-mtl-external-runtime
- Conflict policy: Own TypeScript monitor, reproducible package build, installer, and cross-runtime certifier; do not change the Python reference or infer global proof from finite traces.
- Predicted files: ipfs_datasets_py/typescript/logic-runtime-mtl, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- Changed paths:
- Context paths: ipfs_datasets_py/typescript/logic-runtime-mtl, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- AST symbols: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- Interfaces: ExternalRuntimeMTLVendorCertification@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G210
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/aa6e4c5b8b105722bea5e7f968122b8325d85476b7f8e8c4bf124ea259c0f45d
- Canonical task CID: baguqeeravjxeyw4lcblsfpvf474wqerlqms5qvdww74orrf7cjhkewoa6roq
- Semantic identity: objective-evidence-obligation/v1/a60b8e3ac481146abf4cc428012737f1f50987f9290eafd544b680c7f8c66017
- Acceptance subset: A locked TypeScript dependency graph builds an independent Node package/executable without importing or dispatching to the Python reference, package, source, lockfile, runtime, executable, and artifact digests are bound, positive, negative, interval/event mutation, timestamp boundary, shortest-prefix replay, malformed input, timeout, bounds, and disagreement cases execute out of process, finite-trace authority and inconclusive-prefix semantics are preserved, generated Python parity wrappers remain non-production shadow evidence.
- Preconditions: objective goal FVT-G210 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, satisfy evidence requirement: docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- Evidence subset: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- Resource class: cpu-validation
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-validation
- Merge fate: objective/FVT-G210
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/a60b8e3ac481146abf4cc428012737f1f50987f9290eafd544b680c7f8c66017
- Missing evidence: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- Embedding query: Replace the Python-backed parity wrapper with a reproducibly built TypeScript/Node monitor and honest cross-runtime evidence.
- AST query: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
- Surplus group: objective/FVT-G210
- Merge key: c172d551569e89a8
- Merge family: objective/FVT-G210
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
- Todo vector key: 81f3dd2e2ff32a8d
- Acceptance: Objective scan filed this gap for FVT-G210. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-07-31-fvt-056-objective-gap-110d71dbeb6c.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
