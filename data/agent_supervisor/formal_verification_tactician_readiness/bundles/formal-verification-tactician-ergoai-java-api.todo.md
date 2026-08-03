# Objective Bundle: formal-verification-tactician/ergoai-java-api

Source todo: docs/architecture/formal_verification_tactician_readiness.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## FVT-090 FVT:: Complete the optional ErgoAI Java API capability

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: dependency-integrity
- Depends on: FVT-084, FVT-087, FVT-085
- Outputs: config/formal_verification_toolchains.lock.json, setup.py, requirements.txt, ipfs_datasets_py/setup.py, ipfs_datasets_py/pyproject.toml, ipfs_datasets_py/requirements.txt, ipfs_datasets_py/requirements-lazy.txt, ipfs_datasets_py/requirements-theorem-provers.txt, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/lazy_installer.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py
- Validation: PYTHONPATH=ipfs_datasets_py python -m pytest test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py test/integration/toolchains/test_ergoai_live_toolchain_contract.py ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py test/packaging/test_formal_verification_distribution_contract.py -q
- Board namespace: formal_verification_tactician_readiness.todo.md
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-090-objective-gap-3a5d407335d4.md
- Bundle: formal-verification-tactician/ergoai-java-api
- Bundle shard: data/agent_supervisor/formal_verification_tactician_readiness/bundles/formal-verification-tactician-ergoai-java-api.todo.md
- Bundle strategy: explicit
- Graph parents: FVT-G000
- Graph depth: 1
- Objective heap index: 0
- Parallel lane: formal-verification-tactician/ergoai-java-api
- Conflict policy: Own the optional JDK lock, lazy lifecycle, ErgoAI Java binding, and its focused evidence; never download a moving `latest` target without resolving and pinning its immutable identity, trust ambient `JAVA_HOME`, make a JDK a mandatory pip dependency, elevate advisor output to proof authority, or couple core ErgoAI availability to this optional capability.
- Predicted files: config/formal_verification_toolchains.lock.json, setup.py, requirements.txt, ipfs_datasets_py/setup.py, ipfs_datasets_py/pyproject.toml, ipfs_datasets_py/requirements.txt, ipfs_datasets_py/requirements-lazy.txt, ipfs_datasets_py/requirements-theorem-provers.txt, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/lazy_installer.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py
- Changed paths:
- Context paths: config/formal_verification_toolchains.lock.json, setup.py, requirements.txt, ipfs_datasets_py/setup.py, ipfs_datasets_py/pyproject.toml, ipfs_datasets_py/requirements.txt, ipfs_datasets_py/requirements-lazy.txt, ipfs_datasets_py/requirements-theorem-provers.txt, ipfs_datasets_py/ipfs_datasets_py/logic/backends/installers/advisors.py, ipfs_datasets_py/ipfs_datasets_py/logic/external_provers/lazy_installer.py, ipfs_datasets_py/ipfs_datasets_py/logic/flogic/ergoai_wrapper.py, test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py
- AST symbols: test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py
- Interfaces: ErgoAIJavaAPIToolchainContract@1, LogicVerificationLazyInstaller@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: FVT-G222
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/ca01c66ed6c1cfd334cd87ff77c34ffa22cf2385d484d6466efcb084b41e21e8
- Canonical task CID: baguqeerazia4m3wwyhh5gngnq77xpq2p7irm6i4f2scnmrto7syijna6ehua
- Semantic identity: objective-evidence-obligation/v1/cf62c255281892a7cc9274e63f32bc6cccbfe8d267545dc3fd6e559445032fa4
- Acceptance subset: The lock selects reviewed Eclipse Temurin or another authoritative OpenJDK JDK artifact by exact version, publisher, immutable URL, SHA-256, signature or equivalent publisher evidence, license, OS, architecture, archive size, and required `java`, `javac`, and `jar` identities, only an explicit allow flag may perform a bounded acquisition and symlink-safe extraction into the validated user-local transaction root, interruption, checksum mismatch, archive traversal, unsupported hosts, stale dependency identities, or failed post-install probes roll back and remain unavailable, the managed ErgoAI wrapper binds the exact JDK identity used by its Java consumers and executes positive, negative, malformed, timeout, replay, relocation, and dependency-mutation cases, capability, dependency, semantic, platform, packaging, and authority axes remain independently visible, setup.py, pyproject, and requirements inventories classify the JDK as a reviewed external lazy dependency rather than an invalid mandatory Python package, import, probe, wheel installation, dry-run, and offline certification never download or install, core ErgoAI remains independently usable when the Java API capability is absent.
- Preconditions: objective goal FVT-G222 is schedulable
- Effects: satisfy evidence requirement: test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, satisfy evidence requirement: ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py
- Evidence subset: test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py
- Resource class: io-artifact
- Token class: medium
- Estimated tokens: 0
- Context budget tokens: 4096
- Provider role: grok, codex-review
- Resources: io-artifact
- Merge fate: objective/FVT-G222
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/cf62c255281892a7cc9274e63f32bc6cccbfe8d267545dc3fd6e559445032fa4
- Missing evidence: test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py
- Embedding query: Turn the currently detected Java-runtime-only ErgoAI installation into a separately certified Java API capability by providing an authoritative, checksum-pinned, platform-bound, user-local JDK lifecycle without weakening the already working core ErgoAI lane.
- AST query: test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py
- Surplus group: objective/FVT-G222
- Merge key: 89e16b721d0b0828
- Merge family: objective/FVT-G222
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
- Todo vector key: faea0e67f4c892d8
- Acceptance: Objective scan filed this gap for FVT-G222. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/formal_verification_tactician_readiness/discovery/2026-08-03-fvt-090-objective-gap-3a5d407335d4.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (test/integration/toolchains/test_ergoai_java_api_toolchain_contract.py, ipfs_datasets_py/tests/unit/logic/backends/test_ergoai_java_api_lazy_installation.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
