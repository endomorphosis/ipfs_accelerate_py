# FVT-072 Objective Goal Gap

Date: 2026-08-01
Fingerprint: d0fd68e4de5bdcd398036ad5cae5485e33f9554e
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
Objective heap index: 4
Bundle: formal-verification-tactician/runtime-mtl-external-runtime
Parallel lane: formal-verification-tactician/runtime-mtl-external-runtime
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: path
Embedding query: Replace the Python-backed parity wrapper with a reproducibly built TypeScript/Node monitor, enforce the install-versus-offline-certification boundary, and produce honest cross-runtime evidence.
AST query: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
Conflict policy: Own TypeScript monitor, reproducible package build, installer, offline install boundary, and cross-runtime certifier; do not silently build during certification, change the Python reference semantics, or infer global proof from finite traces.
Predicted files: ipfs_datasets_py/typescript/logic-runtime-mtl, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/runtime_mtl.py, tools/logic/certification/runtime_mtl.py, tools/logic/certification/runtime_mtl_external.py, test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
AST symbols: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py, test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py, docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json
Interfaces: ExternalRuntimeMTLVendorCertification@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/fe04c5e60b0f760900b57604ed92a615e8e31797078176da594e29f991ef0af2
Acceptance subset: A locked TypeScript dependency graph builds an independent Node package/executable without importing or dispatching to the Python reference, the explicit opt-in user-local installation phase may run the locked build, but every offline semantic-certification path, including the in-process Runtime MTL parity helper, consumes only a preinstalled digest-verified artifact and never runs `npm install`, `npm ci`, `npm run build`, downloads, or network access, a missing or stale prebuilt artifact blocks certification instead of rebuilding, the authoritative private-HOME validation environment receives an explicit approved immutable deployment root rather than discovering mutable user paths, package, source, lockfile, runtime, launcher, launcher target, executable, and artifact digests are bound, positive, negative, interval/event mutation, timestamp boundary, shortest-prefix replay, malformed input, timeout, bounds, and disagreement cases execute out of process, finite-trace authority and inconclusive-prefix semantics are preserved, generated Python parity wrappers remain non-production shadow evidence.
Preconditions: objective goal FVT-G210 is schedulable
Effects: satisfy evidence requirement: objective validation repair
Evidence subset: objective validation repair
Dependencies: FVT-G103, FVT-G181, FVT-G201, FVT-G202
Resource class: cpu-validation
Token class: medium
Estimated tokens: 0
Resources: cpu-validation
Merge fate: objective/FVT-G210
Rejection reasons: none (accepted)

## Goal

Replace the Python-backed parity wrapper with a reproducibly built TypeScript/Node monitor, enforce the install-versus-offline-certification boundary, and produce honest cross-runtime evidence.

## Missing Evidence

- objective validation repair

## Present Evidence

- test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py: test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py (path)
- test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py: test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py (path)
- docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json: docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json (path)

## Suggested Handling

Run and repair the objective validation command until it passes, then record the evidence.
