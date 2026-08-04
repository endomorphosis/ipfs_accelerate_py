# FVT-091 Objective Goal Gap

Date: 2026-08-03
Fingerprint: 2db42c69ed9bbc7f7487a8f6ea3d357e890d680b
Goal id: FVT-G223
Goal title: Live-certify/harden managed ErgoAI Java/JDK path
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: certification-integrity
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 0
Bundle: formal-verification-tactician/ergoai-java-api-live-certification
Parallel lane: formal-verification-tactician/ergoai-java-api-live-certification
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Convert the managed ErgoAI Java/JDK capability from contract and fixture coverage into a live, adversarially hardened, replayable deployment path using the exact official ErgoAI 3.0 and Eclipse Temurin 17.0.20+8 identities, without coupling core ErgoAI availability to Java or raising advisor output above its authority ceiling.
AST query: docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
Conflict policy: Own live Java/JDK lifecycle hardening, the ErgoAI Java consumer binding, and focused public-safe evidence; never depend on SecPAL or FVT-G219, trust ambient Java state, substitute fixtures for live vendor execution, publish an unverified or partially installed tree, delete outside the bounded HOME descendant, weaken rollback or identity checks to pass, elevate advisor evidence to proof authority, or make core ErgoAI depend on this optional lane.
Predicted files: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, tools/logic/certification/advisors.py, docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
AST symbols: docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
Interfaces: ErgoAIJavaAPILiveCertification@1, ErgoAIJavaAPIToolchainContract@1, ErgoAILiveToolchainContract@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/734049b8034a98243053e37302a179380a0267e295f57708e0b1a14ea19b14ad
Acceptance subset: On each reviewed host, an explicit opt-in installs the immutable official Eclipse Temurin 17.0.20+8 artifact, verifies its pinned archive digest, size, release checksum, detached signature or equivalently strong publisher evidence, vendor, license, OS, and architecture, and records exact `java`, `javac`, and `jar` versions, paths, and byte digests, the pinned official ErgoAI 3.0 tree then compiles and executes at least one real vendor Java consumer against real ErgoAI/XSB, and a synthetic HelloWorld or JDK-only probe cannot satisfy the semantic case, a real timeout terminates the complete child process tree within a bounded grace period and removes transient consumer workspaces, locks, and partial outputs, relocating the validated JDK and ErgoAI pair beneath a fresh HOME preserves identity binding and produces deterministic replay evidence without ambient `JAVA_HOME`, PATH, caches, or source-tree leakage, byte mutations of each selected Java tool and every identity-bound ErgoAI Java launcher, archive, or classpath artifact are rejected before execution and cannot be repaired into success by stale evidence, concurrent processes use a bounded cross-process single-flight lock so only one acquisition/publication occurs, all waiters observe the same validated identity, abandoned locks are recovered safely, and no partial tree is visible, a failed `force=True` replacement, including download, publisher-evidence, extraction, probe, or publication failure, rolls back to the exact previous-good tree and receipt while quarantining or deleting only bounded staged state, every mutable install, cache, staging, lock, backup, quarantine, and temporary path is a strict canonical descendant of the selected HOME, while HOME itself, sibling or prefix-collision paths, lexical traversal, and symlinked ancestors or targets fail closed, manifests and receipts bind the official publisher evidence, artifact and executable digests, platform tuple, canonical paths, ErgoAI release identity, test inputs, raw-output digests, cleanup result, and source/merged trees, the Java capability remains advisor-only, cannot discharge a proof obligation without an independent kernel or solver, and its absence or failure never makes the already certified core ErgoAI advisor unavailable.
Preconditions: objective goal FVT-G223 is schedulable
Effects: satisfy evidence requirement: docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, satisfy evidence requirement: test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
Evidence subset: docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
Dependencies: FVT-G222, FVT-G218
Resource class: io-artifact
Token class: medium
Estimated tokens: 0
Resources: io-artifact
Merge fate: objective/FVT-G223
Rejection reasons: none (accepted)

## Goal

Convert the managed ErgoAI Java/JDK capability from contract and fixture coverage into a live, adversarially hardened, replayable deployment path using the exact official ErgoAI 3.0 and Eclipse Temurin 17.0.20+8 identities, without coupling core ErgoAI availability to Java or raising advisor output above its authority ceiling.

## Missing Evidence

- docs/architecture/formal_verification_ergoai_java_api_live_receipt.json
- test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py
- ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
