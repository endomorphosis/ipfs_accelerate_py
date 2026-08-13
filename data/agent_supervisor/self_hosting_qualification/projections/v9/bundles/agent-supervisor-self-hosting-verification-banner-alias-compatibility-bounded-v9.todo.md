# Objective Bundle: agent-supervisor/self-hosting/verification-banner-alias-compatibility-bounded-v9

Source todo: docs/architecture/self_hosting_qualification.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## SHQ-017 Close objective gap: Add the reviewed bwrap banner-name compatibility

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
- Discovery evidence: data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-017-objective-gap-2dc7ed8e8734.md
- Bundle: agent-supervisor/self-hosting/verification-banner-alias-compatibility-bounded-v9
- Bundle shard: data/agent_supervisor/self_hosting_qualification/projections/v9/bundles/agent-supervisor-self-hosting-verification-banner-alias-compatibility-bounded-v9.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 0
- Parallel lane: verification-banner-alias-compatibility-bounded-v9
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
- Canonical task key: task/v1/4f6899f93c817124a1c9a4d6d63ba1ac886e20c17ca53dfcf9b3837e24f2ec27
- Canonical task CID: baguqeeraj5ujt6j4qfysjiojutlnmo5bvseg4igbpsst37hzwobx4jhs5qtq
- Semantic identity: objective-evidence-obligation/v1/4aca181471ea85d9ebba0dc97b3b54706cca36eab5ea6d46fb328806b6a4063b
- Acceptance subset: `contracts.py` alone owns a private immutable closed alias constant with the sole reviewed pair `{('bwrap', 'bubblewrap')}`, it is applied only inside the bounded version-banner name predicate and only after every existing capability, reviewed `ToolIdentity`, locator, resolved executable, executable-byte hash, selector executable, probe executable, and invocation-prefix binding succeeds. The declared and keyed `tool_name` remains exact `bwrap`, executable basename remains exact `bwrap`, no caller argument, environment value, configuration, adapter, or subclass can add or replace aliases. Ordinary exact tool-name behavior, including exact bwrap, pytest and mypy banners, remains unchanged. Only for already-bound exact tool name `bwrap`, accept the exceptional exact lower-case `bubblewrap` alias when the unmodified raw probe output is exactly one canonical banner line pairing that alias with the independently token-bound claimed version: alias, exactly one ASCII space, claimed version, and exactly one terminal LF, the actual positive is exactly `b"bubblewrap 0.9.0\n"`. Reject paths, help/usage/error/diagnostic prose, a name and version separated across lines, extra lines or text, prefixes/suffixes, CR, tabs, doubled spaces, leading/trailing whitespace, a missing LF, multiple LFs, `notbubblewrap`, `bubblewrap-helper`, `not-bwrap`, unknown aliases, wrong/missing/subtoken versions, selector/probe executable mismatch, reviewed locator/bytes/identity mismatch, and every attempt to extend aliases. The exact version-token check, exact raw probe bytes, probe-output CID, executable bytes and identity remain unchanged and affect the compiled key. Probe-output mutation changes the key. The alias never bypasses executable or version identity and never permits a wrapper, rewritten banner, or synthetic probe output. The bounded-v8 SHQ-014 attempt is hard-rejected, its dirty worktree, proposed code/tests, implementation log, supervisor/checkpoint/runtime state, receipts and any derived bytes are non-inputs and provide no discovery, completion, or retry authority.
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
- Acceptance: Objective scan filed this gap for SHQ-G005A. Use evidence in data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-017-objective-gap-2dc7ed8e8734.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Define one private immutable closed constant in `verification/contracts.py`, semantically exactly `frozenset({("bwrap", "bubblewrap")})`, and expose no mutation or injection seam. Preserve the existing exact normalized `tool_name == executable.name`, `selector[0] == resolved_tool_executable`, `probe_argv[0] == resolved_tool_executable`, invocation-prefix, capability snapshot, reviewed locator, executable-byte SHA-256, lock, environment, exact version-token, raw probe-byte, and CID checks. Preserve the existing ordinary exact-name predicate byte-for-byte in behavior; do not impose the exceptional alias grammar on exact tool-name banners or any non-bwrap tool. Add a separate exceptional path only when the already-bound exact tool name and executable basename are `bwrap` and the candidate alias is the closed exact lower-case `bubblewrap`. First require the normalized claimed version itself to be exactly one nonempty ASCII `[A-Za-z0-9._+\-]+` token, containing no space, tab, CR, LF, or non-ASCII code point. Then require `tool_version_probe_output_bytes == f"bubblewrap {normalized_tool_version}\n".encode("ascii")` exactly, and require the existing independent whole-token version predicate to succeed with that same version and identifier boundaries. The canonical positive must resolve and actually read `/usr/bin/bwrap`, bind its actual executable bytes and SHA-256 into the reviewed `ToolIdentity`, use that same resolved path for selector and probe, keep keyed tool name `bwrap`, independently claim version `0.9.0`, and bind actually observed raw probe bytes exactly `b"bubblewrap 0.9.0\n"`; synthetic bytes such as `b"reviewed-launcher:bwrap"`, a helper locator, wrapper, rewrite, or synthesized probe is forbidden. Reject fake or changed executable bytes, helper locator, path banner, help/usage/error/diagnostic prose, cross-line separated alias/version, any additional line, any prefix or suffix, upper-case or mixed-case alias, CR or CRLF, tab, doubled-space, leading-space, trailing-space, missing-LF or extra-LF variants, embedded CR/LF/tab/space or non-ASCII claimed versions, `notbubblewrap`, `bubblewrap-helper`, `not-bwrap`, unknown alias, caller extension, wrong/missing/subtoken version, selector/probe executable mismatch, and reviewed locator/bytes/identity mismatch. Tests prove those cases and prove exact bwrap, pytest, mypy and all other ordinary exact-name cases are unchanged; output-byte mutation changes the key; no alias bypasses executable/version identity. Before any edit or validation, hard-reject the bounded-v8 SHQ-014 attempt and treat its dirty disposable worktree, code, tests, implementation log, supervisor/checkpoint/runtime state, receipts and derived bytes as prohibited non-inputs; inspect and implement only from the clean bounded-v9 task checkout.
