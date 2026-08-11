# Objective Bundle: formal-verification-tactician/ergoai-java-api-live-certification

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-091 FVT::: Live-certify/harden managed ErgoAI Java/JDK path

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: certification-integrity
- Depends on: FVT-090, FVT-085
- Outputs: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, tools/logic/certification/advisors.py, docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
- Validation: IPFS_DATASETS_PY_TEST_LIVE_ERGOAI_JAVA=1 IPFS_DATASETS_PY_ALLOW_LAZY_INSTALL=1 PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py test/integration/toolchains/test_ergoai_live_toolchain_contract.py ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-091-objective-gap-2db42c69ed9b.md
- Bundle: formal-verification-tactician/ergoai-java-api-live-certification
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-ergoai-java-api-live-certification.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/ergoai-java-api-live-certification
- Conflict policy: Own live Java/JDK lifecycle hardening, the ErgoAI Java consumer binding, and focused public-safe evidence; never depend on SecPAL or FVT-G219, trust ambient Java state, substitute fixtures for live vendor execution, publish an unverified or partially installed tree, delete outside the bounded HOME descendant, weaken rollback or identity checks to pass, elevate advisor evidence to proof authority, or make core ErgoAI depend on this optional lane.
- Predicted files: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, tools/logic/certification/advisors.py, docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
- Changed paths:
- Context paths: config/formal_verification_toolchains.lock.json, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, tools/logic/certification/advisors.py, docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
- AST symbols: docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
- Interfaces: ErgoAIJavaAPILiveCertification@1, ErgoAIJavaAPIToolchainContract@1, ErgoAILiveToolchainContract@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G223
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/4bb559bdd6f57e5063ad41698aff92375629a3ed701831f58873954fd2bc2cb6
- Canonical task CID: baguqeerajo2vtpow6v7fay5nifuyv74sg5lcti7noamdd5miooku7uv4fs3a
- Semantic identity: objective-evidence-obligation/v1/734049b8034a98243053e37302a179380a0267e295f57708e0b1a14ea19b14ad
- Acceptance subset: On each reviewed host, an explicit opt-in installs the immutable official Eclipse Temurin 17.0.20+8 artifact, verifies its pinned archive digest, size, release checksum, detached signature or equivalently strong publisher evidence, vendor, license, OS, and architecture, and records exact `java`, `javac`, and `jar` versions, paths, and byte digests, the pinned official ErgoAI 3.0 tree then compiles and executes at least one real vendor Java consumer against real ErgoAI/XSB, and a synthetic HelloWorld or JDK-only probe cannot satisfy the semantic case, a real timeout terminates the complete child process tree within a bounded grace period and removes transient consumer workspaces, locks, and partial outputs, relocating the validated JDK and ErgoAI pair beneath a fresh HOME preserves identity binding and produces deterministic replay evidence without ambient `JAVA_HOME`, PATH, caches, or source-tree leakage, byte mutations of each selected Java tool and every identity-bound ErgoAI Java launcher, archive, or classpath artifact are rejected before execution and cannot be repaired into success by stale evidence, concurrent processes use a bounded cross-process single-flight lock so only one acquisition/publication occurs, all waiters observe the same validated identity, abandoned locks are recovered safely, and no partial tree is visible, a failed `force=True` replacement, including download, publisher-evidence, extraction, probe, or publication failure, rolls back to the exact previous-good tree and receipt while quarantining or deleting only bounded staged state, every mutable install, cache, staging, lock, backup, quarantine, and temporary path is a strict canonical descendant of the selected HOME, while HOME itself, sibling or prefix-collision paths, lexical traversal, and symlinked ancestors or targets fail closed, manifests and receipts bind the official publisher evidence, artifact and executable digests, platform tuple, canonical paths, ErgoAI release identity, test inputs, raw-output digests, cleanup result, and source/merged trees, the Java capability remains advisor-only, cannot discharge a proof obligation without an independent kernel or solver, and its absence or failure never makes the already certified core ErgoAI advisor unavailable.
- Preconditions: objective goal FVT-G223 is schedulable
- Effects: satisfy evidence requirement: docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, satisfy evidence requirement: test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
- Evidence subset: docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
- Resource class: io-artifact
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: io-artifact
- Merge fate: objective/FVT-G223
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/734049b8034a98243053e37302a179380a0267e295f57708e0b1a14ea19b14ad
- Missing evidence: docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
- Embedding query: Convert the managed ErgoAI Java/JDK capability from contract and fixture coverage into a live, adversarially hardened, replayable deployment path using the exact official ErgoAI 3.0 and Eclipse Temurin 17.0.20+8 identities, without coupling core ErgoAI availability to Java or raising advisor output above its authority ceiling.
- AST query: docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py
- Surplus group: objective/FVT-G223
- Merge key: bb7c08e54a7f353b
- Merge family: objective/FVT-G223
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
- Todo vector key: eef9a1cd056e3b49
- Acceptance: Objective scan filed this gap for FVT-G223. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-091-objective-gap-2db42c69ed9b.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (docs/architecture/formal_verification_ergoai_java_api_live_receipt.json, test/integration/toolchains/test_ergoai_java_api_live_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_transactional_hardening.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
