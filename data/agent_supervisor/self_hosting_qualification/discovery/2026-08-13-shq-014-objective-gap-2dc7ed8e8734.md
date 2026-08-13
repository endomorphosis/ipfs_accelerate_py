# SHQ-014 Objective Goal Gap

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
Bundle: agent-supervisor/self-hosting/verification-banner-alias-compatibility-bounded-v8
Parallel lane: verification-banner-alias-compatibility-bounded-v8
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
Acceptance subset: `contracts.py` alone owns a private immutable closed alias constant with the sole reviewed pair `{('bwrap', 'bubblewrap')}`, it is applied only inside the bounded version-banner name-token predicate and only after every existing capability, reviewed `ToolIdentity`, locator, resolved executable, executable-byte hash, selector executable, probe executable, and invocation-prefix binding succeeds. The declared and keyed `tool_name` remains exact `bwrap`, executable basename remains exact `bwrap`, no caller argument, environment value, configuration, adapter, or subclass can add or replace aliases. Accept both the existing exact `bwrap` name token and, only for exact tool name `bwrap`, the exact standalone `bubblewrap` token in actual raw probe bytes `bubblewrap 0.9.0\n`, matching is case-normalized only as the existing predicate permits and is token-bound, never substring, prefix, suffix, path, or helper-name matching. The exact version token check, exact raw probe bytes, probe-output CID, executable bytes and identity remain unchanged and affect the compiled key. Reject `notbubblewrap`, `bubblewrap-helper`, `not-bwrap`, unknown aliases, wrong/missing/subtoken versions, selector/probe executable mismatch, reviewed locator/bytes/identity mismatch, and every attempt to extend aliases. Existing pytest, mypy, and all ordinary exact-name behavior remain unchanged. Probe-output mutation changes the key. The alias never bypasses executable or version identity and never permits a wrapper or synthetic probe output.
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
