# Objective Bundle: agent-supervisor/self-hosting/verification-banner-alias-compatibility-bounded-v8

Source todo: docs/architecture/self_hosting_qualification.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## SHQ-014 Close objective gap: Add the reviewed bwrap banner-name compatibility

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prerequisite-compatibility
- Depends on:
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, test/api/test_agent_supervisor_verification_contracts.py
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_contracts.py -k bwrap_banner_alias; python3 -m pytest -q test/api/test_agent_supervisor_verification_contracts.py
- Board namespace: self_hosting_qualification.todo.md
- Evidence inputs: data/agent_supervisor/self_hosting_qualification/discovery
- Discovery evidence: data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-014-objective-gap-2dc7ed8e8734.md
- Bundle: agent-supervisor/self-hosting/verification-banner-alias-compatibility-bounded-v8
- Bundle shard: data/agent_supervisor/self_hosting_qualification/projections/v8/bundles/agent-supervisor-self-hosting-verification-banner-alias-compatibility-bounded-v8.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 0
- Parallel lane: verification-banner-alias-compatibility-bounded-v8
- Conflict policy: Edit only the existing contracts authority and its full focused test; do not add a wrapper, provider, schema, receipt type, runtime facade, configuration knob, or caller-controlled alias.
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, test/api/test_agent_supervisor_verification_contracts.py
- Changed paths:
- Context paths: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, test/api/test_agent_supervisor_verification_contracts.py
- AST symbols: VerificationIdentityCompiler compile_key tool version probe banner token
- Interfaces: verification.contracts.VerificationIdentityCompiler
- Submodules:
- Generated artifacts:
- Allow concurrent with:
- Goal id: SHQ-G005A
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/6574b965c62b6258f970bb9147bab9c43cee86a74e60151603696f4a98d6fd50
- Canonical task CID: baguqeeramv2lszogfnrfr6lqxoiupovzyq6o5bvhjzqbkfqdnfxuvggw7via
- Semantic identity: objective-evidence-obligation/v1/4aca181471ea85d9ebba0dc97b3b54706cca36eab5ea6d46fb328806b6a4063b
- Acceptance subset: `contracts.py` alone owns a private immutable closed alias constant with the sole reviewed pair `{('bwrap', 'bubblewrap')}`, it is applied only inside the bounded version-banner name-token predicate and only after every existing capability, reviewed `ToolIdentity`, locator, resolved executable, executable-byte hash, selector executable, probe executable, and invocation-prefix binding succeeds. The declared and keyed `tool_name` remains exact `bwrap`, executable basename remains exact `bwrap`, no caller argument, environment value, configuration, adapter, or subclass can add or replace aliases. Accept both the existing exact `bwrap` name token and, only for exact tool name `bwrap`, the exact standalone `bubblewrap` token in actual raw probe bytes `bubblewrap 0.9.0\n`, matching is case-normalized only as the existing predicate permits and is token-bound, never substring, prefix, suffix, path, or helper-name matching. The exact version token check, exact raw probe bytes, probe-output CID, executable bytes and identity remain unchanged and affect the compiled key. Reject `notbubblewrap`, `bubblewrap-helper`, `not-bwrap`, unknown aliases, wrong/missing/subtoken versions, selector/probe executable mismatch, reviewed locator/bytes/identity mismatch, and every attempt to extend aliases. Existing pytest, mypy, and all ordinary exact-name behavior remain unchanged. Probe-output mutation changes the key. The alias never bypasses executable or version identity and never permits a wrapper or synthetic probe output.
- Preconditions: objective goal SHQ-G005A is schedulable
- Effects: satisfy evidence requirement: objective validation repair
- Evidence subset: objective validation repair
- Resource class: cpu-small
- Token class: small
- Estimated tokens: 0
- Resources: cpu-small
- Merge fate: objective/SHQ-G005A
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/4aca181471ea85d9ebba0dc97b3b54706cca36eab5ea6d46fb328806b6a4063b
- Missing evidence: objective validation repair
- Embedding query: verification identity compiler bwrap bubblewrap exact banner token alias
- AST query: VerificationIdentityCompiler compile_key tool version probe banner token
- Surplus group: objective/SHQ-G005A
- Merge key: ba74ca463f79bed2
- Merge family: objective/SHQ-G005A
- Merge role: validation_gate
- Work item count: 1
- Work scope: objective_validation_repair
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: validation_gate
- Todo vector key: ea7c99d52602f2ee
- Acceptance: Objective scan filed this gap for SHQ-G005A. Use evidence in data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-014-objective-gap-2dc7ed8e8734.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Define one private immutable closed constant in `verification/contracts.py`, semantically exactly `frozenset({("bwrap", "bubblewrap")})`, and expose no mutation or injection seam. Preserve the existing exact normalized `tool_name == executable.name`, `selector[0] == resolved_tool_executable`, `probe_argv[0] == resolved_tool_executable`, invocation-prefix, capability snapshot, reviewed locator, executable-byte SHA-256, lock, environment, exact version-token, raw probe-byte, and CID checks. Only replace the banner-name substring predicate with a tokenizing predicate whose candidate-name set is ordinarily `{tool_name}` and is exactly `{"bwrap", "bubblewrap"}` only when the already-bound exact tool name is `bwrap`; require one whole candidate token using the same identifier boundaries as the version check, not substring containment. The canonical positive is resolved `/usr/bin/bwrap`, selector and probe beginning with that exact path, tool name `bwrap`, version `0.9.0`, and actually observed bytes `b"bubblewrap 0.9.0\n"`; do not add a wrapper and do not synthesize or rewrite output. Tests prove exact `bwrap` and exact `bubblewrap` tokens pass only for bwrap; `notbubblewrap`, `bubblewrap-helper`, `not-bwrap`, unknown alias, caller extension, wrong/missing/subtoken version, selector/probe executable mismatch, reviewed locator/bytes/identity mismatch fail; pytest/mypy and ordinary exact-name cases are unchanged; output-byte mutation changes the key; no alias bypasses executable/version identity.
