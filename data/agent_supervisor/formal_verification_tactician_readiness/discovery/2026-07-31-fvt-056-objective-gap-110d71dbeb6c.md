# FVT-056 Objective Goal Gap

Date: 2026-07-31
Fingerprint: 110d71dbeb6ce826df90c5930b2658935b4c9274
Goal id: FVT-G210
Goal title: Build and certify an independent external Runtime MTL engine
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: external-capability
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 2
Bundle: formal-verification-tactician/runtime-mtl-external-runtime
Parallel lane: formal-verification-tactician/runtime-mtl-external-runtime
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Replace the Python-backed parity wrapper with a reproducibly built TypeScript/Node monitor and honest cross-runtime evidence.
AST query: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
Conflict policy: Own TypeScript monitor, reproducible package build, installer, and cross-runtime certifier; do not change the Python reference or infer global proof from finite traces.
Predicted files: ipfs_datasets_py/typescript/logic-runtime-mtl, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
AST symbols: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
Interfaces: ExternalRuntimeMTLVendorCertification@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/a60b8e3ac481146abf4cc428012737f1f50987f9290eafd544b680c7f8c66017
Acceptance subset: A locked TypeScript dependency graph builds an independent Node package/executable without importing or dispatching to the Python reference, package, source, lockfile, runtime, executable, and artifact digests are bound, positive, negative, interval/event mutation, timestamp boundary, shortest-prefix replay, malformed input, timeout, bounds, and disagreement cases execute out of process, finite-trace authority and inconclusive-prefix semantics are preserved, generated Python parity wrappers remain non-production shadow evidence.
Preconditions: objective goal FVT-G210 is schedulable
Effects: satisfy evidence requirement: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, satisfy evidence requirement: docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
Evidence subset: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
Dependencies: FVT-G103, FVT-G181, FVT-G201, FVT-G202
Resource class: cpu-validation
Token class: medium
Estimated tokens: 0
Resources: cpu-validation
Merge fate: objective/FVT-G210
Rejection reasons: none (accepted)

## Goal

Replace the Python-backed parity wrapper with a reproducibly built TypeScript/Node monitor and honest cross-runtime evidence.

## Missing Evidence

- test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py
- docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
