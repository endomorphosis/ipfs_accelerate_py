# SHQ-020 Objective Goal Gap

Date: 2026-08-13
Fingerprint: 2dc7ed8e8734da29b4f5d36d1bfcb15f900520ab
Goal id: SHQ-G005A
Goal title: Add the reviewed bwrap banner-name compatibility
Objective heap: docs/architecture/self_hosting_qualification.objectives.md
Priority: P0
Track: prerequisite-compatibility
Status: todo
Schedulable: true
Review only: false
Parent goals: SHQ-G005
Graph depth: 2
Objective heap index: 0
Bundle: agent-supervisor/self-hosting/verification-banner-alias-compatibility-bounded-v10
Parallel lane: verification-banner-alias-compatibility-bounded-v10
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: path
Embedding query: verification identity compiler bwrap bubblewrap exact banner token alias
AST query: VerificationIdentityCompiler compile_key tool version probe banner token
Conflict policy: Edit only the existing contracts authority and its full focused test; do not add a wrapper, provider, schema, receipt type, runtime facade, configuration knob, or caller-controlled alias.
Predicted files: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, test/api/test_agent_supervisor_verification_contracts.py
AST symbols: VerificationIdentityCompiler compile_key tool version probe banner token
Interfaces: verification.contracts.VerificationIdentityCompiler
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/4aca181471ea85d9ebba0dc97b3b54706cca36eab5ea6d46fb328806b6a4063b
Acceptance subset: `contracts.py` alone owns a private immutable closed alias constant with the sole reviewed pair `{('bwrap', 'bubblewrap')}`, the special bwrap banner grammar is applied only after every existing capability, reviewed `ToolIdentity`, locator, resolved executable, executable-byte hash, selector executable, probe executable, and invocation-prefix binding succeeds. The declared and keyed `tool_name` and executable basename remain exact `bwrap`, no caller argument, environment value, configuration, adapter, subclass, or module-global constant rebinding can add or replace aliases, replacing the constant with a caller-extension or superset must not expand the hard-coded closed canonical pair or permitted raw-byte set. For already-bound exact bwrap only, accept raw probe bytes if and only if they are one canonical line equal to either `b"bwrap " + version_ascii + b"\n"` or the sole reviewed alias form `b"bubblewrap " + version_ascii + b"\n"`, where the independently normalized claimed version is one nonempty ASCII `[A-Za-z0-9._+\-]+` token and the existing independent whole-token version predicate succeeds for that same token. The live host positive uses the actually observed raw bytes exactly `b"bubblewrap 0.9.0\n"`, with no rewrite or synthesis. Canonical exact-name `b"bwrap 0.9.0\n"` is accepted only in a bounded pure-compiler fixture that binds the actual executable bytes and SHA-256, that fixture is not live execution evidence or authority. For exact bwrap, reject paths, help/usage/error/diagnostic prose, split-line name/version, extra lines or text, prefixes/suffixes, case variants, CR/CRLF, tabs, doubled/leading/trailing spaces, missing or extra LF, embedded whitespace or non-ASCII claimed versions, `notbubblewrap`, `bubblewrap-helper`, `not-bwrap`, unknown aliases, wrong/missing/subtoken versions, selector/probe executable mismatch, reviewed locator/bytes/identity mismatch, fake or changed executable bytes including `b"reviewed-launcher:bwrap"`, and every alias-extension attempt, including module-global rebinding to a caller extension or superset. Non-bwrap legacy exact-name behavior, including pytest and mypy, remains unchanged. Exact raw probe bytes, probe-output CID, executable bytes and identity remain key-bound, any output mutation changes the key. No banner form bypasses executable/version identity or permits a helper, wrapper, rewritten banner, or synthetic probe. The bounded-v9 SHQ-017 attempt is hard-rejected, its dirty worktree, proposed code/tests, implementation log, supervisor/checkpoint/runtime state, receipts and derived bytes are prohibited non-inputs with no discovery, completion, evidence, or retry authority.
Preconditions: objective goal SHQ-G005A is schedulable
Effects: satisfy evidence requirement: objective validation repair
Evidence subset: objective validation repair
Dependencies: none
Resource class: cpu-small
Token class: small
Estimated tokens: 0
Resources: cpu-small
Merge fate: objective/SHQ-G005A
Rejection reasons: none (accepted)

## Goal

Add and independently test the one reviewed executable-to-banner token compatibility needed for the existing `VerificationIdentityCompiler` to bind the real `/usr/bin/bwrap --version` output without weakening any executable, version, selector, byte, CID, or tool-identity check.

## Missing Evidence

- objective validation repair

## Present Evidence

- ipfs_accelerate_py/agent_supervisor/verification/contracts.py: ipfs_accelerate_py/agent_supervisor/verification/contracts.py (path)
- test/api/test_agent_supervisor_verification_contracts.py: test/api/test_agent_supervisor_verification_contracts.py (path)

## Suggested Handling

Run and repair the objective validation command until it passes, then record the evidence.
