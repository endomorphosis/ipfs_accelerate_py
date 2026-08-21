# Objective Bundle: agent-supervisor/self-hosting/verification-banner-alias-compatibility-bounded-v10

Source todo: docs/architecture/self_hosting_qualification.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## SHQ-020 Close objective gap: Add the reviewed bwrap banner-name compatibility

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
- Discovery evidence: data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-020-objective-gap-2dc7ed8e8734.md
- Bundle: agent-supervisor/self-hosting/verification-banner-alias-compatibility-bounded-v10
- Bundle shard: data/agent_supervisor/self_hosting_qualification/projections/v10/bundles/agent-supervisor-self-hosting-verification-banner-alias-compatibility-bounded-v10.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 0
- Parallel lane: verification-banner-alias-compatibility-bounded-v10
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
- Canonical task key: task/v1/f94c9cab7fc4ffbd2ad76bd8e54fa94dfc004f773f8cc753b63a094aa9a7fc9c
- Canonical task CID: baguqeera7fgjzk37yt732kwxnpmokt5jjx6aat3xh6gmou5whieuvknh7soa
- Semantic identity: objective-evidence-obligation/v1/4aca181471ea85d9ebba0dc97b3b54706cca36eab5ea6d46fb328806b6a4063b
- Acceptance subset: `contracts.py` alone owns a private immutable closed alias constant with the sole reviewed pair `{('bwrap', 'bubblewrap')}`, the special bwrap banner grammar is applied only after every existing capability, reviewed `ToolIdentity`, locator, resolved executable, executable-byte hash, selector executable, probe executable, and invocation-prefix binding succeeds. The declared and keyed `tool_name` and executable basename remain exact `bwrap`, no caller argument, environment value, configuration, adapter, subclass, or module-global constant rebinding can add or replace aliases, replacing the constant with a caller-extension or superset must not expand the hard-coded closed canonical pair or permitted raw-byte set. For already-bound exact bwrap only, accept raw probe bytes if and only if they are one canonical line equal to either `b"bwrap " + version_ascii + b"\n"` or the sole reviewed alias form `b"bubblewrap " + version_ascii + b"\n"`, where the independently normalized claimed version is one nonempty ASCII `[A-Za-z0-9._+\-]+` token and the existing independent whole-token version predicate succeeds for that same token. The live host positive uses the actually observed raw bytes exactly `b"bubblewrap 0.9.0\n"`, with no rewrite or synthesis. Canonical exact-name `b"bwrap 0.9.0\n"` is accepted only in a bounded pure-compiler fixture that binds the actual executable bytes and SHA-256, that fixture is not live execution evidence or authority. For exact bwrap, reject paths, help/usage/error/diagnostic prose, split-line name/version, extra lines or text, prefixes/suffixes, case variants, CR/CRLF, tabs, doubled/leading/trailing spaces, missing or extra LF, embedded whitespace or non-ASCII claimed versions, `notbubblewrap`, `bubblewrap-helper`, `not-bwrap`, unknown aliases, wrong/missing/subtoken versions, selector/probe executable mismatch, reviewed locator/bytes/identity mismatch, fake or changed executable bytes including `b"reviewed-launcher:bwrap"`, and every alias-extension attempt, including module-global rebinding to a caller extension or superset. Non-bwrap legacy exact-name behavior, including pytest and mypy, remains unchanged. Exact raw probe bytes, probe-output CID, executable bytes and identity remain key-bound, any output mutation changes the key. No banner form bypasses executable/version identity or permits a helper, wrapper, rewritten banner, or synthetic probe. The bounded-v9 SHQ-017 attempt is hard-rejected, its dirty worktree, proposed code/tests, implementation log, supervisor/checkpoint/runtime state, receipts and derived bytes are prohibited non-inputs with no discovery, completion, evidence, or retry authority.
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
- Acceptance: Objective scan filed this gap for SHQ-G005A. Use evidence in data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-020-objective-gap-2dc7ed8e8734.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (objective validation repair), and keep the supervisor-fed backlog aligned with the objective heap.  Define one private immutable closed constant in `verification/contracts.py`, semantically exactly `frozenset({("bwrap", "bubblewrap")})`, and expose no mutation or injection seam; the bwrap predicate must validate the hard-coded closed canonical pair and exact two-value raw-byte set so monkeypatching/rebinding that module-global constant to a caller extension or superset still cannot authorize another name or form. Preserve the existing exact normalized `tool_name == executable.name`, `selector[0] == resolved_tool_executable`, `probe_argv[0] == resolved_tool_executable`, invocation-prefix, capability snapshot, reviewed locator, executable-byte SHA-256, lock, environment, raw probe-byte, CID, and independent whole-token version checks. Preserve legacy behavior for every non-bwrap exact tool name; do not route pytest, mypy, or any other tool through the special grammar. Only when the already-bound exact tool name and executable basename are `bwrap`, require the normalized claimed version itself to be exactly one nonempty ASCII `[A-Za-z0-9._+\-]+` token with no space, tab, CR, LF, or non-ASCII code point. Construct exactly two permitted raw values from that independently bound token: `f"bwrap {normalized_tool_version}\n".encode("ascii")` and `f"bubblewrap {normalized_tool_version}\n".encode("ascii")`; require `tool_version_probe_output_bytes` to equal one of those two byte strings exactly, then also require the existing independent whole-token version predicate to succeed for the same version and identifier boundaries. The live host positive must resolve and actually read `/usr/bin/bwrap`, bind its actual executable bytes and SHA-256 into the reviewed `ToolIdentity`, use that same path for selector and probe, retain keyed tool name `bwrap`, claim version `0.9.0`, and bind the actually observed raw bytes exactly `b"bubblewrap 0.9.0\n"`; rewriting or synthesizing `bwrap` output is forbidden. Separately prove canonical exact-name bytes `b"bwrap 0.9.0\n"` pass only in a bounded pure-compiler fixture that binds those same actual executable bytes and SHA-256; the fixture is not execution evidence or authority. Reject fake/changed executable bytes including `b"reviewed-launcher:bwrap"`, helper locator, path banner, help/usage/error/diagnostic prose, cross-line separated name/version, any extra line/text/prefix/suffix, upper/mixed-case name, CR/CRLF, tab, doubled/leading/trailing spaces, missing/extra LF, embedded CR/LF/tab/space or non-ASCII claimed versions, helper/subtoken names, unknown aliases, caller extension, wrong/missing/subtoken version, selector/probe mismatch, and reviewed locator/bytes/identity mismatch. The exact-name negative matrix must literally reject `b"bwrap\n0.9.0\n"`, `b"bwrap  0.9.0\n"`, `b"bwrap 0.9.0 extra\n"`, and `b"bwrap 0.9.0"`; replace only the leading name with `bubblewrap` and require the identical malformed matrix to reject. Also reject concrete path/help/error/extra-line forms such as `b"/usr/bin/bwrap 0.9.0\n"`, `b"bwrap 0.9.0\nUsage: bwrap ...\n"`, `b"error: bwrap 0.9.0\n"`, and `b"bwrap 0.9.0\nextra\n"`, with parallel bubblewrap cases. Tests explicitly exercise malformed exact `bwrap` as well as malformed `bubblewrap` forms and prove all non-bwrap legacy exact-name cases unchanged; output-byte mutation changes the key; no accepted banner bypasses executable/version identity. Before any edit or validation, hard-reject bounded-v9 SHQ-017 and treat its dirty disposable worktree, proposed code/tests, log, supervisor/checkpoint/runtime state, receipts and derived bytes as prohibited non-inputs; inspect and implement only from the clean bounded-v10 task checkout.
