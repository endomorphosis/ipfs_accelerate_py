# Self-Hosting Qualification Objective Heap (SHQ)

Machine-ingestible goal hierarchy for the bounded
`SelfHostingQualificationHarness` capstone and its narrow
`GovernedCodingAgentRuntime` facade. After the reviewed observer-task migrations, the
executable task board is
`docs/architecture/self_hosting_qualification.todo.md` with task prefix
`SHQ-`; the current bounded-retry bundle, graph and dataset projections live
below `data/agent_supervisor/self_hosting_qualification/projections/v12/`. The
retired v1, cancelled v2, prelaunch v3, rejected/cancelled v4, never-launched
prelaunch-corrected v5, rejected/cancelled v6, v7, v8, v9 and v10 cards, and
the settled/rejected/unlaunched v11 cards live
only in versioned history boards,
which are never task sources. This heap is fail-closed. `SHQ-G010` is an
externally governed release-admission
gate. Until it has typed, current-tree completion receipts for every prerequisite
system, the objective daemon must not project any descendant implementation work.

## North star

Determine, with reproducible and signed evidence, whether one bounded Python
package can safely maintain part of itself with less model context, less frontier
inference, exact incremental verification reuse, omission detection, interruption
recovery, independent evaluation, and controlled human escalation.

## Bounded target

The planned target is `endomorphosis/ipfs_kit_py:ipfs_kit_py/core/wal` at the
eventual frozen qualification revision. `core.operation_contracts` is a read-only
dependency. The target can change only through an operator-reviewed amendment to
this protected heap before corpus construction.

## Goal tree

```text
SHQ-G000  Bounded self-hosting qualification and truthful release decision
├── SHQ-G005  Prerequisite observation
│   ├── SHQ-G005A Add the reviewed bwrap banner-name compatibility
│   ├── SHQ-G006A Build the catalog, forest and deterministic nonterminal observer core
│   ├── SHQ-G006B Add the isolated live receipt/cache terminal chain
│   ├── SHQ-G006  Integrate and harden atomic observer publication
│   └── SHQ-G007  Generate the post-merge prerequisite observation snapshot
├── SHQ-G010  Externally admit all ten prerequisite releases
├── SHQ-G020  Reproducible baseline and environment freeze
│   ├── SHQ-G021  Inventory exact revisions, versions, schemas, routes and proofs
│   ├── SHQ-G022  Run subsystem tests and import-safety probes
│   └── SHQ-G023  Prove the WAL target green and freeze the environment
├── SHQ-G030  Shared wire contracts and task corpus
│   ├── SHQ-G031  Add narrow MCP++ shared schemas and canonical vectors
│   ├── SHQ-G032  Define datasets-owned task, split and result contracts
│   ├── SHQ-G033  Build history-firewalled replay tasks
│   ├── SHQ-G034  Build controlled synthetic tasks
│   ├── SHQ-G035  Adapt bounded adversarial-assurance tasks
│   ├── SHQ-G036  Implement independent semantic outcome evaluation
│   ├── SHQ-G037  Build, stratify, split and persist at least 50 tasks
│   └── SHQ-G038  Bind datasets semantic state and ContextPack construction
├── SHQ-G040  Immutable evidence, CAS recovery and release storage
│   ├── SHQ-G041  Bind immutable artifacts and content identities
│   ├── SHQ-G042  Persist typed task, model, test, proof and manifest receipts
│   ├── SHQ-G043  Implement fenced CAS and ambiguous-outcome recovery
│   └── SHQ-G044  Sign, verify and roll back qualification releases
├── SHQ-G050  Governed execution and five-configuration harness
│   ├── SHQ-G051  Implement provider-neutral tier dispatch and accounting
│   ├── SHQ-G052  Compose the stage-resumable GovernedCodingAgentRuntime
│   ├── SHQ-G053  Implement the SelfHostingQualificationHarness plan
│   ├── SHQ-G054  Execute configurations A and B
│   ├── SHQ-G055  Execute configuration C
│   ├── SHQ-G056  Execute configuration D
│   ├── SHQ-G057  Execute configuration E
│   └── SHQ-G058  Expose the required CLI and resume/status operations
├── SHQ-G060  Independent analysis, crash safety and qualification policy
│   ├── SHQ-G061  Compare outcomes and evaluate preregistered noninferiority
│   ├── SHQ-G062  Compute economics and the model-substitution matrix
│   ├── SHQ-G063  Qualify all twelve crash and recovery boundaries
│   ├── SHQ-G064  Control a bounded disposable longitudinal pilot
│   ├── SHQ-G065  Determine the qualification level and project the manifest
│   ├── SHQ-G066  Enforce fail-closed CI and current-release verification
│   ├── SHQ-G068  Implement preregistration and complete metric schemas
│   └── SHQ-G067  Implement the integrated release-candidate freeze
└── SHQ-G070  Execute the evidence program
    ├── SHQ-G071  Freeze the release candidate, then run development and calibration tasks
    └── SHQ-G072  Externally freeze margins, policies, prices, routes and seeds
        └── SHQ-G073  Run held-out configurations A through E
            └── SHQ-G074  Analyze held-out, assurance and recovery evidence
                └── SHQ-G075  Run or truthfully decline the longitudinal pilot
                    └── SHQ-G076  Emit the final report and conditionally signed release
```

## SHQ-G000 Bounded self-hosting qualification and truthful release decision

- Status: active
- Parent:
- Depends on:
- Fib priority: 100
- Track: self-hosting-qualification
- Priority: P0
- Bundle: agent-supervisor/self-hosting/root
- Parallel lane: release
- Resource class: cpu-medium
- Token class: medium
- Goal: Qualify exactly one bounded target through five comparable configurations and emit a truthful evidence-bound decision without expanding the portfolio architecture.
- Evidence:
- Outputs:
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_plan.py
- Acceptance: Every mandatory child reaches a typed terminal state; failed gates become explicit negative evidence; no partial or simulated run is published as qualification success.
- Refinement: Children own disjoint repository authorities and evidence stages; the root owns no aggregate implementation edit.
- Conflict policy: Do not create another agent framework, semantic analyzer, capsule format, proof system, provider, transport, backend, GUI, dataset, or MCP++ profile.

## SHQ-G005 Observe prerequisite release convergence

- Status: active
- Parent: SHQ-G000
- Depends on:
- Fib priority: 100
- Track: prerequisite-observation
- Priority: P0
- Bundle: agent-supervisor/self-hosting/prerequisite-observer
- Parallel lane: prerequisite-observer
- Resource class: cpu-small
- Token class: small
- Goal: Maintain a non-authoritative, current observation of all named prerequisite systems while their owning supervisors finish.
- Evidence:
- Outputs:
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Acceptance: The observer reports exact repositories, revisions, public symbols, focused tests, board states and limitations without upgrading in-flight work to released status.
- Refinement: Observation is not completion authority and is safe to execute before the release gate.
- Conflict policy: Read other supervisors and repositories only; never edit their boards, worktrees, receipts, branches, state databases, keys, or policies.

## SHQ-G005A Add the reviewed bwrap banner-name compatibility

- Status: blocked
- Review only: true
- Blocked reason: SHQ-023 completion and its clean tracked merge are accepted only as the reviewed G006A source-baseline precondition; `external_goal_completion_authoritative=false`, so G005A may not be automatically reprojected or reimplemented before formal operator reconciliation.
- Parent: SHQ-G005
- Depends on:
- Fib priority: 89
- Track: prerequisite-compatibility
- Priority: P0
- Bundle: agent-supervisor/self-hosting/verification-banner-alias-compatibility-bounded-v11
- Parallel lane: verification-banner-alias-compatibility-bounded-v11
- Resource class: cpu-small
- Token class: small
- Goal: Add and independently test the one reviewed executable-to-banner token compatibility needed for the existing `VerificationIdentityCompiler` to bind the real `/usr/bin/bwrap --version` output without weakening any executable, version, selector, byte, CID, or tool-identity check.
- Reviewed predecessor producing task: SHQ-023
- Reviewed predecessor binding: v11-clean-merge-and-coordination-settlement; source-baseline precondition only; not objective lifecycle completion authority
- Reviewed predecessor implementation merge commit: 0200be041e1c154660ade9c44a552df97b84dec1
- Reviewed predecessor frozen target commit: 17e19a8e5db327a18dc9437a8de2be299599ecf2
- Reviewed predecessor frozen target tree: 389048a0ee4d39b24dc68289e21a78da9ca1c4c9
- Reviewed predecessor canonical task CID: baguqeerag67a4omevn536zn5wbdtzrvpipp7yym7uptusjxe4vroojgx5bea
- Reviewed predecessor coordination task CID: baguqeera5o6wzpnwezcacp5oiwycvzk5uhvrvadr7e6m6x3qdzh65ff5nktq
- Reviewed predecessor attempt/fence/token: 3/3/3
- Reviewed predecessor receipt CID: baguqeerayifbixgmh227xewfgwza77itadvtynj5oaavccihrdxh5ftkbuoq
- Reviewed predecessor output CID: baguqeerahs3er2kphhbtexrifryshplgoxyzprzgy5bdk2qfdfezrfzh62ma
- Evidence: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, test/api/test_agent_supervisor_verification_contracts.py
- Outputs: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, test/api/test_agent_supervisor_verification_contracts.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/verification/contracts.py, test/api/test_agent_supervisor_verification_contracts.py
- Interfaces: verification.contracts.VerificationIdentityCompiler
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_verification_contracts.py -k bwrap_banner_alias; python3 -m pytest -q test/api/test_agent_supervisor_verification_contracts.py
- Acceptance: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke the task's separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions to this checkpoint/temp-state prohibition are supervisor/runner-private lifecycle operations outside the implementation agent and fresh transient temp/stream objects automatically owned by the listed validation/test runner, including pytest fixture internals; G006/G007 additionally permit only the required process-runner stream capture and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. `contracts.py` alone owns a private immutable closed alias constant with the sole reviewed pair `{('bwrap','bubblewrap')}`; the special bwrap banner grammar is applied only after every existing capability, reviewed `ToolIdentity`, locator, resolved executable, executable-byte hash, selector executable, probe executable, and invocation-prefix binding succeeds. The declared and keyed `tool_name` and executable basename remain exact `bwrap`; no caller argument, environment value, configuration, adapter, subclass, or module-global constant rebinding can add or replace aliases; replacing the constant with a caller-extension or superset must not expand the hard-coded closed canonical pair or permitted raw-byte set. For already-bound exact bwrap only, accept raw probe bytes if and only if they are one canonical line equal to either `b"bwrap " + version_ascii + b"\n"` or the sole reviewed alias form `b"bubblewrap " + version_ascii + b"\n"`, where the independently normalized claimed version is one nonempty ASCII `[A-Za-z0-9._+\-]+` token and the existing independent whole-token version predicate succeeds for that same token. The live host positive uses the actually observed raw bytes exactly `b"bubblewrap 0.9.0\n"`, with no rewrite or synthesis. Canonical exact-name `b"bwrap 0.9.0\n"` is accepted only in a bounded pure-compiler fixture that binds the actual executable bytes and SHA-256; that fixture is not live execution evidence or authority. For exact bwrap, reject paths, help/usage/error/diagnostic prose, split-line name/version, extra lines or text, prefixes/suffixes, case variants, CR/CRLF, tabs, doubled/leading/trailing spaces, missing or extra LF, embedded whitespace or non-ASCII claimed versions, `notbubblewrap`, `bubblewrap-helper`, `not-bwrap`, unknown aliases, wrong/missing/subtoken versions, selector/probe executable mismatch, reviewed locator/bytes/identity mismatch, fake or changed executable bytes including `b"reviewed-launcher:bwrap"`, and every alias-extension attempt, including module-global rebinding to a caller extension or superset. Non-bwrap legacy exact-name behavior, including pytest and mypy, remains unchanged. Exact raw probe bytes, probe-output CID, executable bytes and identity remain key-bound; any output mutation changes the key. No banner form bypasses executable/version identity or permits a helper, wrapper, rewritten banner, or synthetic probe. The cancelled bounded-v10 SHQ-020 attempt and bounded-v9 SHQ-017 attempt are hard-rejected; their worktrees, proposed code/tests, logs, generic checkpoint instructions, supervisor/checkpoint/runtime state, coordination state, claims, leases, receipts and derived bytes are prohibited non-inputs with no discovery, completion, evidence, or retry authority.
- Gap task: Repair only the verified tool-banner name predicate so the real bwrap executable and its real bubblewrap banner can share one exact reviewed identity.
- Refinement: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke the task's separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions to this checkpoint/temp-state prohibition are supervisor/runner-private lifecycle operations outside the implementation agent and fresh transient temp/stream objects automatically owned by the listed validation/test runner, including pytest fixture internals; G006/G007 additionally permit only the required process-runner stream capture and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. Define one private immutable closed constant in `verification/contracts.py`, semantically exactly `frozenset({("bwrap", "bubblewrap")})`, and expose no mutation or injection seam; the bwrap predicate must validate the hard-coded closed canonical pair and exact two-value raw-byte set so monkeypatching/rebinding that module-global constant to a caller extension or superset still cannot authorize another name or form. Preserve the existing exact normalized `tool_name == executable.name`, `selector[0] == resolved_tool_executable`, `probe_argv[0] == resolved_tool_executable`, invocation-prefix, capability snapshot, reviewed locator, executable-byte SHA-256, lock, environment, raw probe-byte, CID, and independent whole-token version checks. Preserve legacy behavior for every non-bwrap exact tool name; do not route pytest, mypy, or any other tool through the special grammar. Only when the already-bound exact tool name and executable basename are `bwrap`, require the normalized claimed version itself to be exactly one nonempty ASCII `[A-Za-z0-9._+\-]+` token with no space, tab, CR, LF, or non-ASCII code point. Construct exactly two permitted raw values from that independently bound token: `f"bwrap {normalized_tool_version}\n".encode("ascii")` and `f"bubblewrap {normalized_tool_version}\n".encode("ascii")`; require `tool_version_probe_output_bytes` to equal one of those two byte strings exactly, then also require the existing independent whole-token version predicate to succeed for the same version and identifier boundaries. The live host positive must resolve and actually read `/usr/bin/bwrap`, bind its actual executable bytes and SHA-256 into the reviewed `ToolIdentity`, use that same path for selector and probe, retain keyed tool name `bwrap`, claim version `0.9.0`, and bind the actually observed raw bytes exactly `b"bubblewrap 0.9.0\n"`; rewriting or synthesizing `bwrap` output is forbidden. Separately prove canonical exact-name bytes `b"bwrap 0.9.0\n"` pass only in a bounded pure-compiler fixture that binds those same actual executable bytes and SHA-256; the fixture is not execution evidence or authority. Reject fake/changed executable bytes including `b"reviewed-launcher:bwrap"`, helper locator, path banner, help/usage/error/diagnostic prose, cross-line separated name/version, any extra line/text/prefix/suffix, upper/mixed-case name, CR/CRLF, tab, doubled/leading/trailing spaces, missing/extra LF, embedded CR/LF/tab/space or non-ASCII claimed versions, helper/subtoken names, unknown aliases, caller extension, wrong/missing/subtoken version, selector/probe mismatch, and reviewed locator/bytes/identity mismatch. The exact-name negative matrix must literally reject `b"bwrap\n0.9.0\n"`, `b"bwrap  0.9.0\n"`, `b"bwrap 0.9.0 extra\n"`, and `b"bwrap 0.9.0"`; replace only the leading name with `bubblewrap` and require the identical malformed matrix to reject. Also reject concrete path/help/error/extra-line forms such as `b"/usr/bin/bwrap 0.9.0\n"`, `b"bwrap 0.9.0\nUsage: bwrap ...\n"`, `b"error: bwrap 0.9.0\n"`, and `b"bwrap 0.9.0\nextra\n"`, with parallel bubblewrap cases. Tests explicitly exercise malformed exact `bwrap` as well as malformed `bubblewrap` forms and prove all non-bwrap legacy exact-name cases unchanged; output-byte mutation changes the key; no accepted banner bypasses executable/version identity. Before any edit or validation, hard-reject cancelled bounded-v10 SHQ-020 and bounded-v9 SHQ-017 and treat their disposable worktrees, proposed code/tests, logs, generic checkpoint instructions, supervisor/checkpoint/runtime state, coordination state, claims, leases, receipts and derived bytes as prohibited non-inputs; this historical contract is preserved for reconciliation only; no bounded-v12 implementation task or checkout is authorized.
- Embedding query: verification identity compiler bwrap bubblewrap exact banner token alias
- AST query: VerificationIdentityCompiler compile_key tool version probe banner token
- Conflict policy: Edit only the existing contracts authority and its full focused test; do not add a wrapper, provider, schema, receipt type, runtime facade, configuration knob, or caller-controlled alias.

## SHQ-G006A Build the catalog, forest and deterministic nonterminal observer core

- Status: active
- Parent: SHQ-G005
- Depends on:
- Fib priority: 144
- Track: prerequisite-observation-core
- Priority: P0
- Bundle: agent-supervisor/self-hosting/prerequisite-observer-catalog-bounded-v12
- Parallel lane: prerequisite-observer-catalog-bounded-v12
- Resource class: cpu-small
- Token class: medium
- Goal: Implement and independently test only the fixed ten-owner catalog, complete compatibility-map/API and board parsing, path confinement, replay-valid recursive `RepositoryForest` observation, stable `S0 == S1`, and deterministic structurally complete nonterminal rows. Do not implement the isolated terminal receipt chain, atomic publisher, ignore exception, CLI publication, or observation artifact in this stage.
- Evidence: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Outputs: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Predicted files: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Interfaces: observe_prerequisite_releases, PrerequisiteObservation, RepositoryForest
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py -k "catalog or compatibility or api or path or board or forest or nonterminal"
- Acceptance: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. V12 isolation is absolute: every bounded-v11 SHQ-024 attempt 1 through 4 and the SHQ-025 registration, plus all v11 implementation logs, disposable/sibling worktrees, supervisor/checkpoint/runtime/coordination state, claims, leases, cancellation receipts, rescue or quarantine refs, operator quarantine bundles, rejected code/tests, scratch/cache material and derived bytes are prohibited non-inputs and may not be inspected, enumerated, restored, copied, seeded, cited as evidence, or used for retry. The retained functional predecessor anchor is commit `17e19a8e5db327a18dc9437a8de2be299599ecf2`, tree `389048a0ee4d39b24dc68289e21a78da9ca1c4c9`, which contains reviewed SHQ-023 implementation merge `0200be041e1c154660ade9c44a552df97b84dec1`, merge tree `aea528d467450cf6a70efa36d5ab6f34b4947fc7`, reviewed follow-up `bbf8039a67bf2f4dafdd19ef289638d023825e22`, and follow-up tree `00c76524f2f9e1273b89816103a27130a551de85`; its exact settled coordination tuple is documented only as an operator-reviewed source precondition and grants no authority to consume receipt files or other v11 bytes. Those reviewed tracked bytes must be carried forward into the freshly committed clean bounded-v12 launch HEAD/tree recorded by operator preflight. The implementation agent reads only that current launch checkout and its declared submodules; it must not `git show`, check out, or otherwise reopen the frozen anchor. From that clean baseline implement the exact fixed ordered ten-owner catalog; reject catalog mutation, every absolute/`..`/symlink escape and incomplete public API/compatibility map; parse boards with the exact closed grammar: heading only `^##[ \t]+(?P<task_id>[A-Z][A-Z0-9_]*-[0-9]+)(?:[ \t]+.*)?$`; a block ends only at the next `^##(?:[ \t]|$)` or EOF; status only `^[ \t]*-[ \t]+Status:[ \t]*(?P<value>[^\r\n]*)[ \t]*$`; require nonzero unique task headings, exactly one status per block, closed tokens `completed`, `todo`, `blocked`, `in_progress`, `review`, and `cancelled`, and terminal iff every status is `completed`; prose and deeper headings do not count; bind complete recursive outer/gitlink/submodule HEAD/tree/index/tracked-content and recursive porcelain identity in replay-valid `RepositoryForest`; make degraded closure honest, typed and nonterminal; and require stable observation manifest `S1 == S0`. Ordinary core observation may return a deterministic structurally complete ten-row `terminal:false` value. This stage must expose no terminal receipt authority and must not write `prerequisite_observation.json` or any same-directory publication temp. The exact ordered mapping is literal: (1) `IncrementalSemanticIndex`: root `ipfs_datasets_py`, module `ipfs_datasets_py/logic/software_contracts/semantic_index/index.py`, export `ipfs_datasets_py/logic/software_contracts/semantic_index/__init__.py`, API class plus `scan_repository`/`diff_repository_states`/`calculate_invalidation`/`explain_symbol`/`explain_impact`/`watch_repository`, release `docs/software_contracts/INCREMENTAL_SEMANTIC_INDEX.md`, board `docs/architecture/incremental_semantic_index.todo.md`, selector `tests/unit/logic/software_contracts/semantic_index/test_api.py`; (2) `SemanticCapsuleCompiler`: root `ipfs_datasets_py`, module `ipfs_datasets_py/logic/software_contracts/semantic_state/capsules.py`, exact `SemanticCapsuleCompiler@1`, module APIs `compile_semantic_capsule`/`compile_semantic_capsules`/`verify_capsule_compile_result`, package `ipfs_datasets_py/logic/software_contracts/semantic_state/__init__.py` exports only singular compile, release `docs/software_contracts/SEMANTIC_STATE_CONTRACT.md`, board `docs/architecture/semantic_state_contract.todo.md`, selector `tests/unit/logic/software_contracts/semantic_state/test_capsules.py`; (3) requested `ContextPackBuilder` is a compatibility label only: root `.`, module-public `ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py`, parent package exports `ContextPack` only; builder operations remain module-public and there is no `ContextPackBuilder` facade, interface `ContextPack@1`, release `docs/semantic_state/SEMANTIC_COMPRESSION_HARNESS.md`, seal `config/semantic_state_dependencies.seal.json` at schema `ipfs-accelerate.agent-supervisor.semantic-state-dependency-seal@2`, board `docs/architecture/semantic_compression_harness.todo.md`, selector `test/api/semantic_state/test_context_pack.py`, corroborating benchmark `docs/benchmarks/semantic_compression_harness_results.json` at schema `ipfs_accelerate_py/semantic-state/benchmark-report@1`; (4) `VerificationReceiptCache`: root `.`, module `ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py`, exact lazy package export from `ipfs_accelerate_py/agent_supervisor/verification/__init__.py` and interface `VerificationReceiptCache@1`, common release `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md` at binding schema `ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@2`, board `docs/architecture/incremental_verification_planner.todo.md`, selector `test/api/test_agent_supervisor_verification_receipt_cache.py`; (5) `IncrementalVerificationPlanner`: root `.`, module `ipfs_accelerate_py/agent_supervisor/verification/planner.py`, exact lazy export from `ipfs_accelerate_py/agent_supervisor/verification/__init__.py` and interface `IncrementalVerificationPlanner@1`, release `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md` at binding schema `ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@2`, board `docs/architecture/incremental_verification_planner.todo.md`, selector `test/api/test_agent_supervisor_incremental_verification_planner.py`; (6) `ModelRoutePlanner`: root `.`, module `ipfs_accelerate_py/agent_supervisor/verification/model_route.py`, exact lazy export from `ipfs_accelerate_py/agent_supervisor/verification/__init__.py` and interface `ModelRoutePlanner@1`, release `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md` at binding schema `ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@2`, board `docs/architecture/incremental_verification_planner.todo.md`, selector `test/api/test_agent_supervisor_verification_model_route.py`; rows 4 through 6 may bind corroborating `artifacts/agent_supervisor/incremental_verification/benchmark.json` only at schema `ipfs_accelerate_py/agent-supervisor/incremental-verification-benchmark@2`; (7) `VerifiedGuiOptimizer`: exact `expected_absent`, checkout root/interface and all module/export/release/board/selector/receipt paths null, limitation `owner_contract_not_declared_on_launch_tree`, never terminal and no guessed facade/path; (8) `IncrementalProofSealer`: the same exact `expected_absent` null-root/null-interface/null-path policy and limitation, never terminal; (9) `SemanticCompressionGovernor`: root `.`, present board `docs/architecture/semantic_compression_governor.todo.md`, expected module `ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py`, export `ipfs_accelerate_py/agent_supervisor/semantic_governor/__init__.py`, selector-path metadata `test/api/semantic_governor/test_public_api.py`, and release `artifacts/agent_supervisor/semantic_compression_governor/release.json`, all four expected paths absent and non-executable metadata, `interface` and `receipt_path` are null, exact `expected_absent_pending_owner`, no receipt authority and never terminal; (10) `AdversarialAssuranceEngine`: exact `expected_absent`, checkout root/interface and all module/export/release/board/selector/receipt paths null, limitation `owner_contract_not_declared_on_launch_tree`, never terminal and no guessed facade/path. There is no authoritative filesystem receipt path for any row. Declare exact `TestReceipt@1`, `ipfs_accelerate_py/agent-supervisor/verification-test-receipt@1`, `DirectExecutionObservation@1`, `ipfs_accelerate_py/agent-supervisor/direct-verification-observation@1`, and `ipfs_accelerate_py/agent-supervisor/verification-process-runner@1`; G006A sets `receipt_id`, `key_id`, `observation_content_id` null with `terminal_chain_not_run`; static files are only `corroboration_paths`. The complete ContextPack compatibility map is: construct `ContextPacker(budget=ContextBudget(), policy=ContextCoveragePolicy(), estimator_version=TOKEN_ESTIMATOR_VERSION)`; build through both `ContextPacker.pack` and `pack_context`; project through `project_admission_to_reference(CapsuleAdmission, token_count=0)`; require common keyword-only inputs exactly `objective`, `target_source_cid`, `surrounding_source_cid`, `test_source_cid`, `dependency_admissions=()`, `obligation_cids=()`, `counterexample_cids=()`, `delta_cid`, `interface_cids=()`, `assumptions=()`, `exclusions=None`, `raw_source_regions=()`, `production_slice=None`, and `production_slice_builder=None`, with functional-only `budget=None`, `policy=None`, and `estimator_version=TOKEN_ESTIMATOR_VERSION`; require exact in-memory `ContextPackResult` fields `pack`, `pack_cid`, `references`, `token_estimate`, `coverage_satisfied`, `production_slice`, `production_slice_cid`, `budget_exceeded`, and `decisions`. The embedded `ContextPack` fields are exactly `objective`, `target_source_cid`, `surrounding_source_cid`, `test_source_cid`, `dependency_capsule_cids`, `obligation_cids`, `counterexample_cids`, `delta_cid`, `interface_cids`, `assumptions`, `exclusions`, `token_totals`, `estimator_version`, `risk`, `route`, and `escalation_recommendation`. `ContextPackResult.to_dict()` serializes exactly `schema` and `interface` plus `pack`, `pack_cid`, `references`, `token_estimate`, `coverage_satisfied`, `production_slice_cid`, `budget_exceeded`, and `decisions`; it never serializes the optional `production_slice` object itself. Require result schema `ipfs-accelerate.context-pack-result@1`, policy schema `ipfs-accelerate.context-coverage-policy@1`, interface `ContextPack@1`, and estimator `context-compiler-calibrated_utf8@1`; preserve exact never-compressed target/surrounding/test `INVARIANT` sources, substitution only on policy plus admission, otherwise raw source plus explained exclusion, required obligation/delta evidence, deterministic ordering/CID, nontruncating budget/coverage escalation, datasets-owned capsule facts, and production-slice CID binding. Tests cover every mapping mutation, partial compatibility map, package/module export distinction, expected-absent null policy and attempted guessed facade/path.
- Gap task: Build and independently test the bounded observer's catalog, confinement, API/board and recursive forest/nonterminal core only.
- Refinement: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. This is the first autonomous bounded-v12 stage. Its implementation actor and validator route are exactly the plan's direct Codex `gpt-5.6-terra`, 49152-window, high-reasoning, one-thread, depth-zero, subagents-disabled path. The initial scheduler cap is `--max-task-attempts 1`, so it gets one semantic implementation attempt; only exact changed typed transient setup/provider/resource/process evidence plus an operator pre-invocation gate proving the prior receipt is typed transient with null output, coordination is inactive/released, no active claim/lease/process/worktree/ref/lock exists, `implementation_attempts_by_cid[<exact canonical task CID>] == 1`, `selection_idle_reason == all_selectable_ready_tasks_reached_max_task_attempts`, no `implementation_retry_deferred:*` state or retry-budget-repair receipt, and the fresh v12 launch HEAD/tree/route/protected envelope match preflight may authorize attempt 2. A semantic or contract rejection freezes the stage and requires a new reviewed migration, never an actor switch, prompt broadening, counter reset, or automatic reopen. No executable task dependency on retired SHQ-023 is emitted; the exact clean merge/settlement tuple above is a reviewed source-baseline precondition only.
- Embedding query: prerequisite observer fixed catalog board API path recursive repository forest deterministic nonterminal
- AST query: observe_prerequisite_releases PrerequisiteObservation RepositoryForest board status catalog
- Conflict policy: Edit only the observer script and its focused test. Do not touch `.gitignore`, execute or manufacture the terminal receipt/cache chain, add publication code, generate the artifact, or read any v11 runtime/quarantine material.

## SHQ-G006B Add the isolated live receipt/cache terminal chain

- Status: active
- Parent: SHQ-G005
- Depends on: SHQ-G006A
- Fib priority: 233
- Track: prerequisite-observation-terminal-chain
- Priority: P0
- Bundle: agent-supervisor/self-hosting/prerequisite-observer-terminal-chain-bounded-v12
- Parallel lane: prerequisite-observer-terminal-chain-bounded-v12
- Resource class: cpu-small
- Token class: medium
- Goal: From only the clean merged bounded-v12 G006A tracked predecessor, implement and independently test the terminal-only isolated live process-runner to identity-compiler to direct-observation to canonical `TestReceipt` and production receipt-cache chain. Do not add atomic publication, the ignore exception, CLI artifact writing, or the observation artifact in this stage.
- Evidence: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Outputs: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Predicted files: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Interfaces: verification.contracts.VerificationIdentityCompiler, verification.process_runner.PROCESS_RUNNER_SCHEMA, verification.process_runner.VerificationProcessRunner, verification.process_runner.VerificationCommand, verification.process_runner.VerificationStreamArtifact, validation.validation_runtime.build_hermetic_validation_runtime, validation.validation_runtime.hermetic_validation_command, verification.contracts.TestReceipt@1, verification.contracts.DirectExecutionObservation@1, verification.receipt_cache.VerificationReceiptCache
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py -k "runner or compiler or receipt or cache or terminal"
- Acceptance: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. Consume G006A only as the exact clean merged tracked bounded-v12 predecessor selected by the freshly generated dependency CID; no predecessor log, worktree, checkpoint, runtime, coordination record, claim, lease, receipt file, temp, cache or other outside-checkout state is input. Apply the same absolute v11 prohibition as G006A to every SHQ-024 attempt, SHQ-025 registration, log, worktree, runtime/checkpoint/coordination record, receipt, quarantine bundle, rejected proposal and derived byte. Enter the terminal chain only after clean complete `S0`: derive inner argv exactly `(sealed_python, '-m', 'pytest', '-q', *selectors)` and bind its exact `shlex.join` value; require that inner argv to equal the exact suffix after `--` in the outer Bubblewrap argv. Build the runtime exactly once and only through `build_hermetic_validation_runtime` and `hermetic_validation_command`, requiring `--unshare-net`, read-only host binding, a bounded writable checkout, private `/tmp`, and no fallback. Bind actual bwrap bytes `b"bubblewrap 0.9.0\n"` and live isolated Python/pytest probe bytes. Call exactly one live same-process `VerificationProcessRunner.run(VerificationCommand)`; require `schema == PROCESS_RUNNER_SCHEMA`, `process_started is true`, `disposition == completed`, `exit_code == 0`, `result.ok is true`, and `publication_allowed is true`, with no timed-out, cancelled, unavailable, simulated, or replayed result. Before any structural projection, require the live result's `executable`, `cwd`, `environment`, `sandbox`, `network_policy`, `timeout_seconds`, `disposition`, `command_argv`, process/lease identity, and stdout/stderr stream fields to equal their corresponding authoritative values from the exact `VerificationCommand`, hermetic runtime, and observed process. Only after that run call `VerificationIdentityCompiler.compile_key` with `receipt_kind=TEST`, `adapter_schema=PROCESS_RUNNER_SCHEMA`, `selector_argv` equal to the exact outer argv, the resolved bwrap executable, and `tool_name='bwrap'` (never pytest), and require the compiled key to match its corresponding live command/tool/environment identities. Construct `DirectExecutionObservation` only from its actual contract fields: `receipt_key_cid`, `repository_tree_cid`, `environment_cid`, `repository_tree_observation`, `environment_observation`, `terminal_status`, `command_argv`, `duration_ms`, `exit_code`, `stdout_artifact_cid`, `stderr_artifact_cid`, `artifact_cids`, and `reason_codes`; require those receipt/tree/environment identities, status, argv, duration, exit code, output CIDs, artifacts, and reasons to equal the compiled key and live result. Each stream must be nontruncated with `captured_byte_count == byte_count == len(preview.encode('utf-8'))`, and the preview bytes must rehash to both the declared digest and CID. Require exact `TestReceipt.from_dict(receipt.to_record()).to_record() == receipt.to_record()`, `admit(..., for_production=True, require_production_eligible=True)`, and `lookup(key, for_production=True)`; then require `S1 == S0`. Reject missing isolation, injected phase reports, cache-only authority, simulated results, and replayed results. Missing isolation or degraded forest remains typed nonterminal and skips the chain. Positive terminal proof is confined to a self-contained clean temporary Git fixture whose authority never upgrades the actual checkout.
- Gap task: Add and independently test only the exact terminal isolated live verification/receipt/cache chain over the merged G006A core.
- Refinement: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. This is the second autonomous bounded-v12 stage and uses the identical pinned Terra/high actor and independent validator route. The initial scheduler cap is `--max-task-attempts 1`, so it gets one semantic implementation attempt; attempt 2 requires exact changed typed transient setup/provider/resource/process evidence and an operator pre-invocation gate proving the prior receipt is typed transient with null output, coordination inactive/released, no active claim/lease/process/worktree/ref/lock, `implementation_attempts_by_cid[<exact canonical task CID>] == 1`, `selection_idle_reason == all_selectable_ready_tasks_reached_max_task_attempts`, no `implementation_retry_deferred:*` state or retry-budget-repair receipt, and fresh matching v12 HEAD/tree/route/protected envelope. Semantic rejection freezes and migrates rather than retrying. Validation must prove runner-before-compiler ordering and reject injected, stale, simulated, replayed, cache-only, mismatched-field, truncated-stream, digest/CID, namespace and fallback counterexamples.
- Embedding query: isolated live process runner compiler direct execution observation test receipt production cache terminal chain
- AST query: VerificationProcessRunner VerificationIdentityCompiler DirectExecutionObservation TestReceipt VerificationReceiptCache
- Conflict policy: Edit only the clean merged observer/test predecessor. Do not edit `.gitignore`, add publication or CLI artifact writes, generate the artifact, consume predecessor runtime, or access any v11 attempt/quarantine material.

## SHQ-G006 Integrate and harden atomic prerequisite-observer publication

- Status: active
- Parent: SHQ-G005
- Depends on: SHQ-G006B
- Fib priority: 377
- Track: prerequisite-observation
- Priority: P0
- Bundle: agent-supervisor/self-hosting/prerequisite-observer-integration-bounded-v12
- Parallel lane: prerequisite-observer-integration-bounded-v12
- Resource class: cpu-small
- Token class: medium
- Goal: From only the clean merged bounded-v12 G006B tracked predecessor, add the exact `.gitignore` exception and durable no-clobber publisher, then complete the full integration and negative matrix entirely in isolated test fixtures without generating the repository observation artifact in this goal.
- Evidence: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Outputs: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Predicted files: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Interfaces: observe_prerequisite_releases, PrerequisiteObservation, verification.contracts.VerificationIdentityCompiler, verification.process_runner.PROCESS_RUNNER_SCHEMA, verification.process_runner.VerificationProcessRunner, verification.process_runner.VerificationCommand, verification.process_runner.VerificationStreamArtifact, validation.validation_runtime.build_hermetic_validation_runtime, validation.validation_runtime.hermetic_validation_command, verification.contracts.TestReceipt@1, verification.contracts.DirectExecutionObservation@1, verification.receipt_cache.VerificationReceiptCache
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Acceptance: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. Consume G006B only as the exact clean merged tracked bounded-v12 predecessor selected by the freshly generated SHQ-027 dependency CID; no predecessor log, worktree, checkpoint, runtime, coordination state, claim, lease, receipt file, temp, cache or other outside-checkout state is input. Every bounded-v11 SHQ-024 attempt 1 through 4 and the SHQ-025 registration, plus every v11 display ID/key/CID offered as a dependency, implementation log, disposable or sibling worktree, supervisor/checkpoint/runtime/coordination record, claim, lease, receipt as bytes, rejected code/test proposal, rescue or quarantine ref, operator quarantine bundle, scratch, cache and derived byte is a prohibited non-input and must not be inspected, enumerated, restored, copied, seeded, validated, cited as evidence, or used for retry. Preserve and revalidate the complete G006A catalog/path/API/board/recursive-forest/nonterminal contract and the complete G006B isolated live runner→compiler→DirectExecutionObservation→canonical TestReceipt/production-cache terminal chain; final integration may not weaken, bypass, simulate or replace either stage. Add the exact narrow `.gitignore` exception `!artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json` only after the final applicable `*.json` rule. Prove by last-rule parsing and exact `git check-ignore -q --no-index --` results that the target returns 1, an unpredictable same-directory `.prerequisite_observation.<nonce>.json` temp returns 0, an isolated Git fixture reports exactly the target as `??`, and recursive porcelain omits the owned temp while its fd remains open. Implement publication only through a validated parent dirfd opened with `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC` where available; create the unpredictable ignored temp through that dirfd with `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC` at mode 0600; require stable full-source `S1 == S0`; `fchmod` 0644; loop until all canonical bytes are written; reject short/zero writes; `fsync` temp; revalidate the full source and `S1 == S0`; no-clobber link temp to target with source/destination dirfds and `follow_symlinks=False`; `fsync` parent; unlink temp; `fsync` parent; reopen target with `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`; read through EOF and require exact canonical bytes. Never use `os.replace` or direct target writes. On post-link failure unlink target through the dirfd and `fsync` parent; on every failure remove the temp and leave no partial or ambiguous artifact. Full tests cover exact catalog/mappings, every path/API/board/forest mutation, degraded nonterminal behavior, runner/compiler/receipt/cache ordering and field/stream/CID negatives, ignore-rule order, nonignored or exception-matching temp, existing file/symlink and concurrent link races, short/zero I/O, open/fchmod/fsync/link/unlink/readback failure, source races, cleanup durability and canonical mismatch. Implement and test strictly read-only `--mode verify-existing --artifact <path>`: nofollow-open only the exact canonical target, strictly decode `PrerequisiteObservation@1`, require canonical reserialization byte equality, exact ten-row/order/null/terminal policy, rederive embedded `RepositoryForest` and all CIDs, recapture the complete current source excluding only the artifact and match every claimed source/gitlink/submodule/index/tracked-input/degraded-reason binding, make zero runner/compiler/cache calls, create no temp or write, and require artifact bytes/stat and repository state unchanged before/after. Default verification requires current HEAD/tree equal the claimed observed source and is the only form allowed during G007 precommit validation. A separately explicit `--allow-exact-evidence-projection-child` is post-merge only and accepts no general descendant: either a clean one-parent HEAD whose parent commit/tree is the claim and whose sole regular-file diff is the canonical artifact blob/mode, or a clean exact two-parent supervisor merge whose first parent is the claim, whose second parent has the claim as its sole parent, whose claim-to-second and first-parent-to-merge diffs both change only that same regular artifact to identical blob/mode, and whose merge tree equals the implementation-child tree; reject any other parent count/order/ancestry/diff/mode/type/symlink or changed recursive identity. This task tests ordinary-observe publication only in isolated fixtures; it must not invoke the real ordinary-observe CLI or create/change the checkout's real `prerequisite_observation.json`. The real target is absent before and after G006.
- Gap task: Implement and test the read-only prerequisite observer and exact ignore exception without generating the tree-bound observation artifact.
- Refinement: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. This is the third autonomous bounded-v12 stage. The implementation actor is exactly direct Codex `gpt-5.6-terra`, total context 49152, high reasoning, one thread, depth zero and subagents disabled; independent authority comes from declared deterministic validation and operator boundary review, not model self-approval. The initial scheduler cap is `--max-task-attempts 1`, so it gets one semantic implementation attempt. Attempt 2 is authorized only by exact changed typed transient setup/provider/resource/process evidence plus an operator pre-invocation gate proving the prior receipt is typed transient with null output, coordination inactive/released, no active claim/lease/process/worktree/ref/lock, `implementation_attempts_by_cid[<exact canonical task CID>] == 1`, `selection_idle_reason == all_selectable_ready_tasks_reached_max_task_attempts`, no `implementation_retry_deferred:*` state or retry-budget-repair receipt, and fresh matching v12 HEAD/tree/route/protected envelope; semantic or contract rejection freezes and migrates the stage, never retries, switches actor, broadens the prompt, resets counters, auto-reopens the board, or continues a repair loop. Edit only `.gitignore`, the clean merged observer and its focused test. Consume G006B solely as tracked source, preserve the exact G006A/G006B interfaces, run the full focused test in this stage, and leave the real artifact absent for G007.
- Embedding query: self hosting qualification prerequisite completion release board commit API focused tests observer
- AST query: IncrementalSemanticIndex SemanticCapsuleCompiler ContextPacker ContextPackBuilder VerificationReceiptCache IncrementalVerificationPlanner ModelRoutePlanner VerifiedGuiOptimizer IncrementalProofSealer SemanticCompressionGovernor AdversarialAssuranceEngine
- Conflict policy: Never modify prerequisite implementation or completion evidence from this task; never read sibling worktrees, operator state, hidden evaluator data, or arbitrary host paths.

## SHQ-G007 Generate the post-merge prerequisite observation snapshot

- Status: active
- Parent: SHQ-G005
- Depends on: SHQ-G006
- Fib priority: 610
- Track: prerequisite-observation
- Priority: P0
- Bundle: agent-supervisor/self-hosting/prerequisite-observation-snapshot-bounded-v12
- Parallel lane: prerequisite-observation-snapshot-bounded-v12
- Resource class: cpu-small
- Token class: small
- Goal: From the clean merged `SHQ-G006` implementation commit, execute the observer and persist the non-authoritative current prerequisite snapshot without changing observer code, tests or ignore policy.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Predicted files: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Interfaces: PrerequisiteObservation@1
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode verify-existing --artifact artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json --quiet; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode require-terminal --quiet && exit 99 || test "$?" -eq 1; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode verify-existing --artifact artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json --quiet
- Acceptance: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. Consume G006 only as the exact clean merged tracked bounded-v12 predecessor selected by the freshly generated SHQ-028 dependency CID; no predecessor runtime, receipt, log, cache, worktree or coordination state is input. Every bounded-v11 SHQ-024 attempt 1 through 4 and the SHQ-025 registration, plus every v11 display ID/key/CID offered as a dependency, implementation log, disposable or sibling worktree, supervisor/checkpoint/runtime/coordination record, claim, lease, receipt as bytes, rejected code/test proposal, rescue or quarantine ref, operator quarantine bundle, scratch, cache and derived byte is a prohibited non-input and must not be inspected, enumerated, restored, copied, seeded, validated, cited as evidence, or used for retry. Change only `artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`; never edit `.gitignore`, observer code, tests, prerequisite owners, policies, keys or generated supervisor state. From the freshly committed clean v12 launch HEAD/tree, execute the merged ordinary-observe CLI exactly once as `python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode observe --quiet`, require its default canonical target to equal the sole declared output, and publish one structurally complete deterministic repository-relative ten-row snapshot through the already-tested durable dirfd/no-clobber publisher. Bind the pre-observation outer HEAD/tree, complete recursive gitlinks and matching submodule HEAD/trees, indexes, tracked-content digests and admitted current receipt authorities while excluding only the artifact path; require the stable observation manifest and degraded reasons satisfy `S1 == S0` through final source revalidation and canonical readback. Validation must never invoke ordinary observe again: run default read-only `verify-existing` before and after the no-output `require-terminal` rc1 probe, forbid the projection-child flag precommit, and prove all artifact bytes/stat and repository state remain unchanged. Refuse an existing target/symlink, dirty or source-raced input, any path/identity mismatch, partial/noncanonical rows, short I/O or publication failure, and leave no temp or partial target. The artifact declares that it is neither completion, proof nor release authority and that its later evidence-projection commit is not the observed source. On this host incomplete recursive closure or unavailable namespace isolation produces exact typed limitations, all ten rows, no receipt authority and truthful `terminal:false`; ordinary observe remains valid, while require-terminal returns 1 and performs no write unless the complete live terminal chain succeeds. Never initialize omitted submodules, access the network, manufacture closure, upgrade a clean fixture result into current-tree evidence, or consume v11 bytes.
- Gap task: Generate and validate the non-authoritative observation artifact only after the merged G006 implementation tree is clean.
- Refinement: This bounded task is neither resumable nor long-running; the generic durable-checkpoint clause is inapplicable and expressly revoked for the implementation agent. No autonomous model-issued shell/file-tool may reference or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`, `authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`; deliberately forward their values as task input or tool arguments; or use them to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy, source, execute, create, modify, redirect, tee, save, cache, checkpoint, materialize, or reread the named checkpoint directory, any alias, resolution, or descendant of it, or other supervisor/checkpoint/runtime state outside the workspace. The implementation agent may not use those paths or bytes as discovery, scratch, evidence, completion, or retry input. This checkpoint revocation does not revoke separately explicit read/execute authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools and bwrap argv, or declared gitlinks. The only permitted exceptions are supervisor/runner-private lifecycle operations outside the implementation agent, fresh transient temp/stream objects automatically owned by the listed validation/test runner including pytest fixture internals, the required process-runner stream capture, and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery or scratch, persisted evidence, or prior/private-state input. This is the fourth autonomous bounded-v12 stage. The implementation actor is exactly the same pinned direct Terra/high route and the validator remains the deterministic CLI/readback matrix plus operator boundary review. The initial scheduler cap is `--max-task-attempts 1`, so it gets one semantic implementation attempt; attempt 2 requires exact changed typed transient setup/provider/resource/process evidence and an operator pre-invocation gate proving the prior receipt is typed transient with null output, coordination inactive/released, no active claim/lease/process/worktree/ref/lock, `implementation_attempts_by_cid[<exact canonical task CID>] == 1`, `selection_idle_reason == all_selectable_ready_tasks_reached_max_task_attempts`, no `implementation_retry_deferred:*` state or retry-budget-repair receipt, and fresh matching v12 HEAD/tree/route/protected envelope. Semantic rejection freezes and migrates rather than retrying, switching actors, broadening prompts, resetting counters or auto-reopening. The freshly projected SHQ-028 canonical task CID is the sole executable predecessor identity. Generate only the real snapshot from the clean merged G006 tracked tree, verify exact canonical readback and keep every implementation/runtime/history authority boundary closed.
- Embedding query: post merge prerequisite observation snapshot clean source projection recursive gitlinks
- AST query: PrerequisiteObservation observation_to_json write_observation_artifact
- Conflict policy: Do not edit `.gitignore`, observer implementation, tests, prerequisite owners, release admission, policies, keys or generated supervisor state; never read arbitrary host paths.

## SHQ-G010 Externally admit all ten prerequisite releases

- Status: active
- Parent: SHQ-G000
- Depends on: SHQ-G007
- Fib priority: 100
- Track: prerequisite-admission
- Priority: P0
- Bundle: agent-supervisor/self-hosting/prerequisite-admission
- Parallel lane: operator-gate
- Resource class: operator-review
- Token class: small
- Completion authority: external
- External completion required: true
- Goal: Admit exact released commits and compatible public interfaces for every named prerequisite only after current focused tests and terminal owner evidence verify.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json
- Validation: python scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode require-terminal --output artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json
- Acceptance: All ten rows are terminal and current; exact commit and interface bindings are explicit; the clean capstone integration branch and all three gitlinks are converged to those admitted revisions; focused tests are current and green; simulated or historical-only evidence is rejected; any missing, in-flight, dirty, unmerged or mismatched subsystem keeps this goal open.
- Gap task: Operator admission only; local implementation receipts cannot complete this goal.
- Refinement: This external gate fences every descendant goal. Rerun the objective daemon only after a typed external completion receipt has been independently validated.
- Embedding query: externally governed terminal prerequisite release admission current tree receipt
- AST query: ExternalCompletionReceipt CompletionEvidence prerequisite_release_admission
- Conflict policy: No agent may edit this goal, its admission receipt, trusted keys, or the prerequisite owners' completion state.

## SHQ-G020 Reproducible baseline and environment freeze

- Status: active
- Parents: SHQ-G000, SHQ-G010
- Depends on:
- Fib priority: 100
- Track: baseline
- Priority: P0
- Bundle: agent-supervisor/self-hosting/baseline
- Parallel lane: baseline
- Resource class: cpu-large
- Token class: medium
- Goal: Establish a clean, exact and reproducibly green four-repository baseline before any qualification implementation or corpus construction.
- Evidence:
- Outputs:
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_baseline.py
- Acceptance: Exact immutable roots, current tests, import safety, target proof checkpoint and frozen environment all pass; otherwise downstream work remains ineligible.
- Refinement: Three serial children prevent implementation from racing an unproven baseline.
- Conflict policy: Do not repair prerequisite failures in the capstone branch; return them to the owning subsystem.

## SHQ-G021 Inventory exact revisions, versions, schemas, routes and proofs

- Status: active
- Parent: SHQ-G020
- Depends on:
- Fib priority: 100
- Track: baseline-inventory
- Priority: P0
- Bundle: agent-supervisor/self-hosting/baseline
- Parallel lane: baseline
- Resource class: cpu-small
- Token class: medium
- Goal: Record exact four-repository commits, package versions, canonical schema/CID/canonicalization versions, model routes, proof systems, test selectors, seal formats and known limitations.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/baseline_inventory.json, docs/architecture/SELF_HOSTING_QUALIFICATION_BASELINE.md
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/baseline_inventory.json, docs/architecture/SELF_HOSTING_QUALIFICATION_BASELINE.md, test/api/test_agent_supervisor_self_hosting_baseline.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/baseline.py, artifacts/agent_supervisor/self_hosting_qualification/baseline_inventory.json, docs/architecture/SELF_HOSTING_QUALIFICATION_BASELINE.md, test/api/test_agent_supervisor_self_hosting_baseline.py
- Interfaces: create_baseline_inventory, BaselineInventory
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_baseline.py -k inventory
- Acceptance: Every identity is exact and immutable; dirty or detached-unbound inputs fail; missing versions are unknown rather than guessed; limitations identify name/API adapters and unavailable capabilities.
- Gap task: Implement the immutable inventory projection and bind all current baseline facts.
- Refinement: This task records facts but does not yet assert that tests or imports are green.
- Embedding query: exact repository commits versions schema CID canonicalization model routes proofs selectors seals limitations
- AST query: version schema_version cid_version canonicalization model route proof seal
- Conflict policy: Inventory existing authorities; never fork their version or identity schemes.

## SHQ-G022 Run subsystem tests and import-safety probes

- Status: active
- Parent: SHQ-G020
- Depends on: SHQ-G021
- Fib priority: 100
- Track: baseline-validation
- Priority: P0
- Bundle: agent-supervisor/self-hosting/baseline
- Parallel lane: baseline
- Resource class: cpu-large
- Token class: medium
- Goal: Run current focused tests for every admitted prerequisite and prove ordinary imports do not install, access the network, alter the environment or report simulated success.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/subsystem_test_receipts.json, artifacts/agent_supervisor/self_hosting_qualification/import_safety_receipts.json
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/import_safety.py, test/api/test_agent_supervisor_self_hosting_import_safety.py, artifacts/agent_supervisor/self_hosting_qualification/subsystem_test_receipts.json, artifacts/agent_supervisor/self_hosting_qualification/import_safety_receipts.json
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/import_safety.py, test/api/test_agent_supervisor_self_hosting_import_safety.py, artifacts/agent_supervisor/self_hosting_qualification/subsystem_test_receipts.json, artifacts/agent_supervisor/self_hosting_qualification/import_safety_receipts.json
- Interfaces: run_import_safety_probe, run_focused_subsystem_checks
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_import_safety.py
- Acceptance: Each selector runs at the bound commit; import probes fence package installation, sockets, subprocess package managers and environment mutation; unavailable and simulated remain non-pass; receipts contain argv, environment CID, timing, exit status and output CID.
- Gap task: Implement hermetic probes and execute the admitted focused subsystem matrix.
- Refinement: A historical green report is context only and cannot satisfy this goal.
- Embedding query: focused prerequisite tests hermetic imports no network install environment mutation simulated success
- AST query: import safety socket pip install subprocess environ simulated live receipt
- Conflict policy: Do not add dependencies or loosen required selectors to make the baseline green.

## SHQ-G023 Prove the WAL target green and freeze the environment

- Status: active
- Parent: SHQ-G020
- Depends on: SHQ-G022
- Fib priority: 100
- Track: target-freeze
- Priority: P0
- Bundle: agent-supervisor/self-hosting/baseline
- Parallel lane: baseline
- Resource class: cpu-proof-solver
- Token class: large
- Goal: Verify `ipfs_kit_py/core/wal` at the selected revision, create a full initial proof checkpoint and freeze locks, SBOM, container, toolchain, seeds and environment identity before corpus execution.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/target_baseline_receipt.json, artifacts/agent_supervisor/self_hosting_qualification/initial_proof_checkpoint.json, artifacts/agent_supervisor/self_hosting_qualification/environment_manifest.json
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/target_baseline_receipt.json, artifacts/agent_supervisor/self_hosting_qualification/initial_proof_checkpoint.json, artifacts/agent_supervisor/self_hosting_qualification/environment_manifest.json, test/api/test_agent_supervisor_self_hosting_environment_freeze.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/environment.py, test/api/test_agent_supervisor_self_hosting_environment_freeze.py, artifacts/agent_supervisor/self_hosting_qualification/target_baseline_receipt.json, artifacts/agent_supervisor/self_hosting_qualification/initial_proof_checkpoint.json, artifacts/agent_supervisor/self_hosting_qualification/environment_manifest.json
- Interfaces: freeze_qualification_environment, TargetBaselineReceipt
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_environment_freeze.py
- Acceptance: All focused WAL tests and declared static/integration/proof checks are green; checkpoint binds exact source and proof-system roots; locks, SBOM and container digest are immutable; target failure terminates qualification at Level 0 rather than spawning repair work.
- Gap task: Run the selected-target baseline and construct deterministic freeze artifacts, failing closed on any required check.
- Refinement: The target is eight bounded WAL modules; `core.operation_contracts` and qualification infrastructure remain read-only.
- Embedding query: ipfs kit core wal green proof checkpoint environment lock SBOM container digest seed freeze
- AST query: core.wal kit-modern-wal checkpoint environment manifest dependency lock SBOM
- Conflict policy: Never alter the WAL target while establishing its baseline.

## SHQ-G030 Shared wire contracts and task corpus

- Status: active
- Parents: SHQ-G000, SHQ-G010
- Depends on:
- Fib priority: 90
- Track: corpus
- Priority: P0
- Bundle: agent-supervisor/self-hosting/corpus-root
- Parallel lane: corpus
- Resource class: cpu-medium
- Token class: medium
- Goal: Define only the shared wire surface and datasets-owned benchmark semantics needed to build a sealed, separated and history-firewalled corpus.
- Evidence:
- Outputs:
- Validation: python -m pytest -q test/api/test_agent_supervisor_self_hosting_corpus_adapter.py
- Acceptance: MCP++ remains a narrow wire authority and datasets owns task semantics; at least 50 stratified tasks are sealed before experimental execution.
- Refinement: Schema, source builders and evaluator use disjoint files so eligible work can proceed in parallel.
- Conflict policy: No new MCP++ profile, general dataset, agent framework or duplicated storage authority.

## SHQ-G031 Add narrow MCP++ shared schemas and canonical vectors

- Status: active
- Parent: SHQ-G030
- Depends on: SHQ-G023
- Fib priority: 90
- Track: shared-contracts
- Priority: P0
- Bundle: mcplusplus/self-hosting/schemas
- Parallel lane: mcplusplus-contracts
- Resource class: cpu-small
- Token class: medium
- Submodules: ipfs_accelerate_py/mcplusplus
- Goal: Add provider-neutral invocation, receipt, qualification and narrow runtime-interface schemas with canonical valid/invalid vectors, without defining a new MCP++ profile.
- Evidence: ipfs_accelerate_py/mcplusplus/schemas/self-hosting-qualification/1.0/qualification.schema.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/self_hosting_qualification_valid.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/self_hosting_qualification_invalid.json
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/self-hosting-qualification/1.0, ipfs_accelerate_py/mcplusplus/conformance/vectors/self_hosting_qualification_valid.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/self_hosting_qualification_invalid.json, ipfs_accelerate_py/mcplusplus/tests-py/test_self_hosting_qualification_schemas.py
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/self-hosting-qualification/1.0/qualification.schema.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/self_hosting_qualification_valid.json, ipfs_accelerate_py/mcplusplus/conformance/vectors/self_hosting_qualification_invalid.json, ipfs_accelerate_py/mcplusplus/tests-py/test_self_hosting_qualification_schemas.py
- Interfaces: ModelInvocationReceipt@1, QualificationManifest@1, QualificationRuntimePort@1
- Validation: python -m pytest -q ipfs_accelerate_py/mcplusplus/tests-py/test_self_hosting_qualification_schemas.py
- Acceptance: Canonical vectors cover all tier, token, price, retry, CID, live/replay, seal and decision fields; unknown statuses reject; provider identity remains data; schemas do not authorize execution, storage or self-approval.
- Gap task: Implement only shared wire schemas, validators and vectors required by the capstone.
- Refinement: Reuse existing canonical JSON and Profile-G invocation primitives through references where compatible.
- Embedding query: MCP++ shared self hosting invocation receipt qualification manifest canonical vectors runtime port
- AST query: ModelInvocationReceipt QualificationManifest QualificationRuntimePort canonical JSON schema
- Conflict policy: No new profile, transport, provider, execution policy or semantic authority in MCP++.

## SHQ-G032 Define datasets-owned task, split and result contracts

- Status: active
- Parent: SHQ-G030
- Depends on: SHQ-G031
- Fib priority: 100
- Track: corpus-contracts
- Priority: P0
- Bundle: datasets/self-hosting/corpus
- Parallel lane: datasets-contracts
- Resource class: cpu-small
- Token class: medium
- Submodules: ipfs_datasets_py
- Goal: Define immutable `SelfHostingTaskCorpus`, task constraints, source kind, classification, split, expected behavior, configuration result and comparison-input schemas under datasets authority.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/metrics.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/metrics.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_metrics.py
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/contracts.py, ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/metrics.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_contracts.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_metrics.py
- Interfaces: SelfHostingTaskCorpus, SelfHostingTask, PatchConstraint, ExpectedEffectClass, CorpusSplit, ExpectedBehavior, BenchmarkTaskResult, QualificationMetrics
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_contracts.py ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_metrics.py
- Acceptance: Every task declares allowed symbols/files/lines/effects/interfaces/tests/proofs/review and hidden evaluator references; effect class is closed over code_patch, no_patch_reject, context_expand, verification_broaden, route_escalate and human_review; split membership is immutable; hidden artifacts cannot serialize into model-visible views; canonical identity changes with every authority field; the aggregate schema explicitly contains every named context, route, quality, verification, compression-safety, assurance, economics and performance metric.
- Gap task: Implement the datasets-owned immutable contracts without copying MCP++ wire or kit storage models.
- Refinement: Keep package imports lazy and free of network, installation and environment mutation.
- Embedding query: SelfHostingTaskCorpus task constraints classification split expected behavior result schema hidden evaluator
- AST query: SelfHostingTaskCorpus SelfHostingTask PatchConstraint CorpusSplit ExpectedBehavior BenchmarkTaskResult
- Conflict policy: Datasets owns task meaning, not execution, provider dispatch, artifact persistence or signing.

## SHQ-G033 Build history-firewalled replay tasks

- Status: active
- Parent: SHQ-G030
- Depends on: SHQ-G023, SHQ-G032
- Fib priority: 90
- Track: historical-corpus
- Priority: P0
- Bundle: datasets/self-hosting/corpus
- Parallel lane: datasets-historical
- Resource class: io-medium
- Token class: large
- Submodules: ipfs_datasets_py
- Context paths: ipfs_kit_py/ipfs_kit_py/core/wal
- Goal: Derive real WAL maintenance tasks from parent revisions, requirements and failures while making future patches and later history evaluator-only and inaccessible to proposing models.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/historical.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/historical.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_historical.py, ipfs_datasets_py/tests/fixtures/self_hosting/historical_manifest.json
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/historical.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_historical.py, ipfs_datasets_py/tests/fixtures/self_hosting/historical_manifest.json
- Interfaces: build_historical_replay_tasks, HistoricalReplayFirewall
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_historical.py
- Acceptance: Every replay starts at the real parent, derives visible requirements from contemporaneous evidence, hides the future patch and later refs, blocks public-issue/provider browsing, retains evaluator evidence separately and scores semantic outcome rather than textual similarity; isolation tests cover refs, reflogs, alternates, remotes, unreachable objects, cat-file batch/all-object enumeration and build artifacts.
- Gap task: Build deterministic historical replay fixtures from qualifying WAL commits and taskboard requirements.
- Refinement: Use all qualifying real history; do not fabricate commits merely to hit a category count.
- Embedding query: historical replay parent commit hidden future patch history firewall semantic outcome WAL
- AST query: build_historical_replay_tasks HistoricalReplayFirewall expected_patch evaluator_only
- Conflict policy: Hidden patches, later Git objects and evaluator metadata never enter a ContextPack or model worktree.

## SHQ-G034 Build controlled synthetic tasks

- Status: active
- Parent: SHQ-G030
- Depends on: SHQ-G023, SHQ-G032
- Fib priority: 90
- Track: synthetic-corpus
- Priority: P0
- Bundle: datasets/self-hosting/corpus
- Parallel lane: datasets-synthetic
- Resource class: cpu-medium
- Token class: large
- Submodules: ipfs_datasets_py
- Context paths: ipfs_kit_py/ipfs_kit_py/core/wal
- Goal: Build controlled WAL-centered fixtures covering all eighteen required bug, test, type, schema, contract, refactor, performance, invalidation, proof, context, dynamic, selection and recovery classes.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/synthetic.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/synthetic.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_synthetic.py, ipfs_datasets_py/tests/fixtures/self_hosting/synthetic_manifest.json
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/synthetic.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_synthetic.py, ipfs_datasets_py/tests/fixtures/self_hosting/synthetic_manifest.json
- Interfaces: build_controlled_synthetic_tasks
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_synthetic.py
- Acceptance: Each required class has deterministic setup and hidden checks; stale capsule/receipt, insufficient context, selection and proof-cache cases declare the correct no-patch rejection, context expansion, verification broadening or escalation effect instead of editing control-plane code; no fixture changes qualification infrastructure or weakens its own oracle; performance and recovery thresholds are explicit; opaque-dependency tasks preserve uncertainty.
- Gap task: Implement the full controlled synthetic factory with bounded patches and independently testable oracles.
- Refinement: Prefer multiple risk/dependency-cone strata where total corpus capacity permits.
- Embedding query: synthetic self hosting bug unit integration type schema exception adapter refactor performance stale invalidation proof context recovery
- AST query: build_controlled_synthetic_tasks SyntheticTaskFactory hidden tests
- Conflict policy: Fixtures may mutate only disposable target copies, never the frozen source or benchmark policy.

## SHQ-G035 Adapt bounded adversarial-assurance tasks

- Status: active
- Parent: SHQ-G030
- Depends on: SHQ-G023, SHQ-G032
- Fib priority: 90
- Track: adversarial-corpus
- Priority: P0
- Bundle: datasets/self-hosting/corpus
- Parallel lane: datasets-adversarial
- Resource class: cpu-medium
- Token class: large
- Submodules: ipfs_datasets_py
- Goal: Adapt the admitted `AdversarialAssuranceEngine` to generate bounded tasks for every required authority, stale-evidence, false-success, timeout, fencing, CAS, omission, assertion, vacuity and seal-manifest failure.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/adversarial.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/adversarial.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_adversarial.py, ipfs_datasets_py/tests/fixtures/self_hosting/adversarial_manifest.json
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/adversarial.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_adversarial.py, ipfs_datasets_py/tests/fixtures/self_hosting/adversarial_manifest.json
- Interfaces: build_adversarial_assurance_tasks, AdversarialAssuranceTaskAdapter
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_adversarial.py
- Acceptance: All thirteen required assurance classes appear; mutants carry expected kill conditions and risk; no new mutation engine is introduced; unavailable engine capability fails closed rather than substituting hand-waved success.
- Gap task: Implement a narrow task adapter over the released assurance engine and deterministic bounded fixtures.
- Refinement: Mutation generation belongs to the existing engine; datasets owns task projection and expected behavior.
- Embedding query: adversarial assurance authorization stale proof missing test false success timeout simulation fence CAS omission weak assertion vacuity seal
- AST query: AdversarialAssuranceEngine AdversarialAssuranceTaskAdapter mutant kill condition
- Conflict policy: Do not rebuild the assurance engine or allow a proposing model to define its own oracle.

## SHQ-G036 Implement independent semantic outcome evaluation

- Status: active
- Parent: SHQ-G030
- Depends on: SHQ-G032
- Fib priority: 100
- Track: semantic-evaluation
- Priority: P0
- Bundle: datasets/self-hosting/corpus
- Parallel lane: datasets-evaluator
- Resource class: cpu-medium
- Token class: large
- Submodules: ipfs_datasets_py
- Goal: Independently compare declared behavior, semantic diffs, hidden tests, static/type/proof/performance/assurance results and patch scope without using the proposing model as sole evaluator.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/evaluator.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/evaluator.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_evaluator.py
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/evaluator.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_evaluator.py
- Interfaces: compare_semantic_outcome, compare_task_outcomes, IndependentSemanticOutcomeEvaluator, TaskComparisonReport
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_evaluator.py
- Acceptance: Textually different correct patches can pass; patch-applied alone cannot; all ten acceptance conditions are recomputed from independent evidence; stale, simulated, missing, self-approved and out-of-scope evidence fail.
- Gap task: Implement the pure independent evaluator and adversarial fixtures for every acceptance gate.
- Refinement: Evaluation consumes typed results but does not execute models, tests or storage operations.
- Embedding query: independent semantic outcome evaluator hidden tests accepted patch proof assurance performance scope
- AST query: compare_semantic_outcome IndependentSemanticOutcomeEvaluator AcceptedPatchDecision
- Conflict policy: The proposing model cannot author, modify, select or approve hidden evaluation evidence.

## SHQ-G037 Build, stratify, split and persist at least 50 tasks

- Status: active
- Parent: SHQ-G030
- Depends on: SHQ-G033, SHQ-G034, SHQ-G035, SHQ-G036, SHQ-G041
- Fib priority: 100
- Track: corpus-release
- Priority: P0
- Bundle: datasets/self-hosting/corpus
- Parallel lane: datasets-release
- Resource class: cpu-medium
- Token class: medium
- Submodules: ipfs_datasets_py
- Goal: Create and version a minimum 50-task initial corpus, stratify all required dimensions, separate development/calibration/held-out outcomes and persist immutable model-visible, evaluator-only and longitudinal-eligibility views through the kit artifact port.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/task_corpus_manifest.json, artifacts/agent_supervisor/self_hosting_qualification/task_split_manifest.json, artifacts/agent_supervisor/self_hosting_qualification/longitudinal_eligibility_manifest.json
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/corpus.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_corpus.py, artifacts/agent_supervisor/self_hosting_qualification/task_corpus_manifest.json, artifacts/agent_supervisor/self_hosting_qualification/task_split_manifest.json, artifacts/agent_supervisor/self_hosting_qualification/longitudinal_eligibility_manifest.json
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/corpus.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_corpus.py, artifacts/agent_supervisor/self_hosting_qualification/task_corpus_manifest.json, artifacts/agent_supervisor/self_hosting_qualification/task_split_manifest.json, artifacts/agent_supervisor/self_hosting_qualification/longitudinal_eligibility_manifest.json
- Interfaces: create_task_corpus, persist_task_corpus
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_corpus.py
- Acceptance: Total tasks are at least 50; all source and class floors hold; each task has risk/cone/context/dynamic/tier/verification strata; split is deterministic and kit-CID-bound; held-out outcome access is denied until policy freeze; model views cannot resolve hidden patches or public links; longitudinal candidates declare composition order, preconditions, rebase semantics and stop conditions without assuming later acceptance.
- Gap task: Assemble the versioned corpus and immutable split from the three independently tested builders.
- Refinement: A stronger release may use at least 100 tasks; the report must qualify inference limits for smaller samples.
- Embedding query: corpus build stratify split development calibration held out CID hidden firewall fifty tasks
- AST query: create_task_corpus seal_task_corpus SelfHostingTaskCorpus CorpusSplit
- Conflict policy: Datasets creates canonical corpus meaning while kit alone persists immutable bytes; do not tune policies from held-out outcomes or count duplicate semantic tasks as independent merely to reach the floor.

## SHQ-G038 Bind datasets semantic state and ContextPack construction

- Status: active
- Parent: SHQ-G030
- Depends on: SHQ-G032
- Fib priority: 100
- Track: context-pack
- Priority: P0
- Bundle: datasets/self-hosting/corpus
- Parallel lane: datasets-corpus
- Resource class: cpu-medium
- Token class: large
- Submodules: ipfs_datasets_py
- Goal: Expose the datasets-owned semantic-state and ContextPack construction boundary for qualification tasks while adapting the admitted capsule/compiler/packer implementation instead of inventing another representation.
- Evidence: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/context_pack.py
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/context_pack.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_context_pack.py
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/self_hosting/context_pack.py, ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_context_pack.py
- Interfaces: SelfHostingContextPackPort, build_self_hosting_context_pack
- Validation: python -m pytest -q ipfs_datasets_py/tests/unit/logic/software_contracts/self_hosting/test_context_pack.py
- Acceptance: The port binds task/source/semantic/capsule/policy roots, records raw cone, packed/expanded tokens, sufficiency and fallback, rejects stale/heuristic capsules, preserves opaque dependencies and delegates actual capsule/pack construction to the admitted versioned API.
- Gap task: Implement the qualification-specific datasets port and compatibility adapter for the released ContextPack implementation.
- Refinement: The port owns benchmark semantics and expected fields; it does not create a new capsule or packing algorithm.
- Embedding query: datasets self hosting semantic state ContextPack construction capsule sufficiency fallback tokens
- AST query: SelfHostingContextPackPort build_self_hosting_context_pack SemanticCapsuleCompiler ContextPacker ContextPackBuilder
- Conflict policy: Reuse the admitted semantic state and packer; no duplicated semantic index, compressor or context format.

## SHQ-G040 Immutable evidence, CAS recovery and release storage

- Status: active
- Parents: SHQ-G000, SHQ-G010
- Depends on:
- Fib priority: 90
- Track: durable-evidence
- Priority: P0
- Bundle: kit/self-hosting/evidence-root
- Parallel lane: kit-evidence
- Resource class: io-medium
- Token class: medium
- Goal: Use ipfs_kit_py as the sole authority for immutable benchmark artifacts, worktree state, receipts, manifests, fencing, release evidence and rollback.
- Evidence:
- Outputs:
- Validation: python -m pytest -q ipfs_kit_py/tests/test_self_hosting_qualification_store.py
- Acceptance: Bytes and roots are exact; CAS is fenced and recoverable; signatures and release verification fail closed; no parallel storage authority appears elsewhere.
- Refinement: Artifact, receipt, recovery and signing children build serially where authority overlaps.
- Conflict policy: Reuse kit CID, canonicalization, WAL and immutable-store primitives; do not fork them.

## SHQ-G041 Bind immutable artifacts and content identities

- Status: active
- Parent: SHQ-G040
- Depends on: SHQ-G023
- Fib priority: 90
- Track: immutable-artifacts
- Priority: P0
- Bundle: kit/self-hosting/artifacts
- Parallel lane: kit-artifacts
- Resource class: io-medium
- Token class: medium
- Submodules: ipfs_kit_py
- Goal: Provide a narrow immutable artifact/CID port for tasks, worktrees, model responses, test/proof evidence and qualification files using existing kit canonicalization.
- Evidence: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/artifacts.py
- Outputs: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/artifacts.py, ipfs_kit_py/tests/test_self_hosting_qualification_artifacts.py
- Predicted files: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/artifacts.py, ipfs_kit_py/tests/test_self_hosting_qualification_artifacts.py
- Interfaces: QualificationArtifactStore, put_immutable, get_verified
- Validation: python -m pytest -q ipfs_kit_py/tests/test_self_hosting_qualification_artifacts.py
- Acceptance: Stored bytes are immutable and CID-verified on every read; canonicalization versions are bound; corrupt/missing data is typed failure; arbitrary remote paths and network fallback are rejected.
- Gap task: Implement a narrow adapter over existing kit artifact and canonicalization primitives.
- Refinement: This port stores bytes and identities, not benchmark semantics or orchestration policy.
- Embedding query: ipfs kit immutable qualification artifact CID canonical bytes verified read
- AST query: QualificationArtifactStore put_immutable get_verified CID canonicalization
- Conflict policy: No new backend, mutable overwrite, simulated storage or duplicate CID algorithm.

## SHQ-G042 Persist typed task, model, test, proof and manifest receipts

- Status: active
- Parent: SHQ-G040
- Depends on: SHQ-G031, SHQ-G041
- Fib priority: 100
- Track: qualification-receipts
- Priority: P0
- Bundle: kit/self-hosting/receipts
- Parallel lane: kit-receipts
- Resource class: io-medium
- Token class: large
- Submodules: ipfs_kit_py
- Goal: Persist canonical model-call, task-execution, test, proof, comparison, pilot and qualification-manifest receipts with exact live/replay and evidence provenance.
- Evidence: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/receipts.py
- Outputs: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/receipts.py, ipfs_kit_py/tests/test_self_hosting_qualification_receipts.py
- Predicted files: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/receipts.py, ipfs_kit_py/tests/test_self_hosting_qualification_receipts.py
- Interfaces: TaskExecutionReceiptStore, ModelCallReceipt, QualificationManifestStore
- Validation: python -m pytest -q ipfs_kit_py/tests/test_self_hosting_qualification_receipts.py
- Acceptance: Every requested model accounting field is required; replayed output is distinguishable and excluded from live evidence; stale/simulated/missing proof cannot serialize as accepted; receipt CIDs bind policy, environment, roots and artifacts.
- Gap task: Implement typed persistence against the shared schemas and immutable artifact port.
- Refinement: Reuse existing VerificationReceiptCache and proof receipt schemas through adapters, not copies.
- Embedding query: task model test proof qualification receipt store live replay tokens latency price cost CID
- AST query: TaskExecutionReceiptStore ModelCallReceipt QualificationManifestStore VerificationReceipt
- Conflict policy: Storage never upgrades evaluator status and never treats existence as validity.

## SHQ-G043 Implement fenced CAS and ambiguous-outcome recovery

- Status: active
- Parent: SHQ-G040
- Depends on: SHQ-G042
- Fib priority: 100
- Track: qualification-recovery
- Priority: P0
- Bundle: kit/self-hosting/recovery
- Parallel lane: kit-recovery
- Resource class: io-medium
- Token class: large
- Submodules: ipfs_kit_py
- Goal: Maintain compare-and-swap qualification roots with generation/fencing tokens, idempotent stage recovery, immutable completion discovery and explicit unknown-outcome repair.
- Evidence: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/state.py, ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/recovery.py
- Outputs: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/state.py, ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/recovery.py, ipfs_kit_py/tests/test_self_hosting_qualification_recovery.py
- Predicted files: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/state.py, ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/recovery.py, ipfs_kit_py/tests/test_self_hosting_qualification_recovery.py
- Interfaces: QualificationStateStore, resume_qualification_stage, recover_ambiguous_outcome
- Validation: python -m pytest -q ipfs_kit_py/tests/test_self_hosting_qualification_recovery.py
- Acceptance: Stale fences and lost-update races fail; repeated recovery causes no duplicate effects; immutable completed artifacts are reused; uncertain model billing or effect outcome requires repair; partial tasks cannot count as accepted.
- Gap task: Implement WAL-backed fenced CAS and explicit recovery decisions over existing kit contracts.
- Refinement: Recovery identifies evidence; it does not guess that an interrupted operation succeeded.
- Embedding query: qualification CAS generation fencing token WAL recovery idempotent ambiguous outcome
- AST query: QualificationStateStore compare_and_swap fence generation recovery WAL
- Conflict policy: Do not weaken WAL, current-root or durability contracts and do not introduce a second state authority.

## SHQ-G044 Sign, verify and roll back qualification releases

- Status: active
- Parent: SHQ-G040
- Depends on: SHQ-G042, SHQ-G043
- Fib priority: 100
- Track: qualification-release
- Priority: P0
- Bundle: kit/self-hosting/release
- Parallel lane: kit-release
- Resource class: cpu-crypto
- Token class: large
- Submodules: ipfs_kit_py
- Goal: Build and verify a signed, content-addressed qualification release with explicit classification, verification instructions, known limitations, blockers and rollback.
- Evidence: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/release.py
- Outputs: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/release.py, ipfs_kit_py/tests/test_self_hosting_qualification_release.py
- Predicted files: ipfs_kit_py/ipfs_kit_py/self_hosting_qualification/release.py, ipfs_kit_py/tests/test_self_hosting_qualification_release.py
- Interfaces: OperatorSigningPort, create_qualification_manifest, verify_qualification_release, build_rollback_manifest
- Validation: python -m pytest -q ipfs_kit_py/tests/test_self_hosting_qualification_release.py
- Acceptance: Signature binds every required artifact and exact revision; an injected operator-controlled signing port may sign only an already gated manifest and never exposes private key bytes to a worktree or model; untrusted/stale/missing keys, denied signing, or missing artifacts fail; partial failure forbids publication; rollback is complete; research/alpha/pilot labels cannot exceed the decision evidence.
- Gap task: Implement release assembly, signing-port invocation, verification and rollback without modifying trusted keys.
- Refinement: Tests use ephemeral fixture keys; production-admissible trusted identities remain operator managed.
- Embedding query: signed content addressed qualification release manifest verify trusted keys rollback classification
- AST query: OperatorSigningPort create_qualification_manifest verify_qualification_release build_rollback_manifest
- Conflict policy: Agents cannot modify trusted keys, approve their own patch or publish after a failed gate.

## SHQ-G050 Governed execution and five-configuration harness

- Status: active
- Parents: SHQ-G000, SHQ-G010
- Depends on:
- Fib priority: 100
- Track: governed-execution
- Priority: P0
- Bundle: agent-supervisor/self-hosting/runtime-root
- Parallel lane: runtime
- Resource class: cpu-large
- Token class: large
- Goal: Compose existing authorities into one stage-resumable runtime and run an identical eligible task set through configurations A through E.
- Evidence:
- Outputs:
- Validation: python -m pytest -q test/api/self_hosting
- Acceptance: Every canonical lifecycle stage is explicit and resumable; configurations isolate only their declared variables; provider resolution remains downstream of capability routing.
- Refinement: Runtime, plan and configurations use dedicated modules; no child becomes another agent framework.
- Conflict policy: Integrate admitted systems through dependency-injected ports and do not duplicate their semantic, routing, verification, assurance, proof or storage authority.

## SHQ-G051 Implement provider-neutral tier dispatch and accounting

- Status: active
- Parent: SHQ-G050
- Depends on: SHQ-G031, SHQ-G042
- Fib priority: 90
- Track: model-dispatch
- Priority: P0
- Bundle: agent-supervisor/self-hosting/model-runner
- Parallel lane: model-runner
- Resource class: model-provider
- Token class: large
- Goal: Resolve deterministic, small-local, medium, frontier and human-review capability tiers through injected providers and persist complete call/cost/latency/retry/live-or-replay accounting.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/model_runner.py, ipfs_accelerate_py/agent_supervisor/self_hosting/context_authorization.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/model_runner.py, ipfs_accelerate_py/agent_supervisor/self_hosting/context_authorization.py, test/api/self_hosting/test_model_runner.py, test/api/self_hosting/test_context_authorization.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/model_runner.py, ipfs_accelerate_py/agent_supervisor/self_hosting/context_authorization.py, test/api/self_hosting/test_model_runner.py, test/api/self_hosting/test_context_authorization.py
- Interfaces: ProviderNeutralModelRunner, ProviderContextAuthorizationReceipt, authorize_provider_context, invoke_model_tier
- Validation: python -m pytest -q test/api/self_hosting/test_model_runner.py test/api/self_hosting/test_context_authorization.py
- Acceptance: Route class never hardcodes a provider; all requested invocation fields persist; pricing is frozen configuration data; replay is excluded from live quality/cost; a mandatory pre-invocation receipt proves secret scan/redaction, approved-source allowlist, hidden-store exclusion, disabled browsing/tools, and approved endpoint; unknown or failed authorization rejects/escalates; missing required tier triggers escalation or human review, never silent downgrade.
- Gap task: Implement the provider-neutral dispatch port and deterministic fixture providers.
- Refinement: This is a runner interface, not a new model provider or routing planner.
- Embedding query: provider neutral deterministic small local medium frontier human model runner accounting price tokens retry replay
- AST query: ProviderNeutralModelRunner invoke_model_tier ModelRoutePlanner ModelCallReceipt
- Conflict policy: Do not add provider SDKs, credentials, network defaults or provider-specific policy.

## SHQ-G052 Compose the stage-resumable GovernedCodingAgentRuntime

- Status: active
- Parent: SHQ-G050
- Depends on: SHQ-G036, SHQ-G038, SHQ-G043, SHQ-G051
- Fib priority: 100
- Track: governed-runtime
- Priority: P0
- Bundle: agent-supervisor/self-hosting/runtime
- Parallel lane: runtime
- Resource class: cpu-large
- Token class: xlarge
- Goal: Implement a narrow dependency-injected `GovernedCodingAgentRuntime` that executes every canonical lifecycle stage, checkpoints after durable boundaries and delegates all authority to released components.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/runtime.py, ipfs_accelerate_py/agent_supervisor/self_hosting/integrations.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/runtime.py, ipfs_accelerate_py/agent_supervisor/self_hosting/integrations.py, test/api/self_hosting/test_runtime.py, test/api/self_hosting/test_runtime_lifecycle.py, test/api/self_hosting/test_prerequisite_integrations.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/runtime.py, ipfs_accelerate_py/agent_supervisor/self_hosting/integrations.py, test/api/self_hosting/test_runtime.py, test/api/self_hosting/test_runtime_lifecycle.py, test/api/self_hosting/test_prerequisite_integrations.py
- Interfaces: GovernedCodingAgentRuntime, execute_task_configuration
- Validation: python -m pytest -q test/api/self_hosting/test_runtime.py test/api/self_hosting/test_runtime_lifecycle.py
- Acceptance: Every user-declared lifecycle stage plus mandatory provider-context authorization appears and no stage silently disables in qualification mode; all ten admitted prerequisites are bound and invoked when applicable; the non-GUI WAL target receives a typed evidence-bound `VerifiedGuiOptimizer` not-applicable decision rather than silent omission; scope is validated before apply; context insufficiency expands or escalates; acceptance is independently recomputed; cancellation and restart resume safely; hidden patches and policies are inaccessible.
- Gap task: Compose the admitted scanner, capsule/context governor, route planner, worktree executor, incremental verifier, assurance engine, proof sealer, human gate and kit stores through narrow ports.
- Refinement: The facade parses no competing semantic graph, chooses no tests independently and owns no provider or persistent-store implementation.
- Embedding query: GovernedCodingAgentRuntime stage resumable canonical lifecycle semantic capsule context routing verification assurance proof human
- AST query: GovernedCodingAgentRuntime execute_task_configuration IncrementalSemanticIndex SemanticCapsuleCompiler ContextPacker SemanticCompressionGovernor ModelRoutePlanner IncrementalVerificationPlanner VerifiedGuiOptimizer AdversarialAssuranceEngine IncrementalProofSealer
- Conflict policy: Extend no existing prerequisite internals; adapters fail typed-unavailable when an admitted capability is absent or incompatible.

## SHQ-G053 Implement the SelfHostingQualificationHarness plan

- Status: active
- Parent: SHQ-G050
- Depends on: SHQ-G037, SHQ-G052
- Fib priority: 100
- Track: experiment-plan
- Priority: P0
- Bundle: agent-supervisor/self-hosting/harness
- Parallel lane: harness
- Resource class: cpu-medium
- Token class: large
- Goal: Implement `SelfHostingQualificationHarness` and a deterministic experiment plan that binds the same eligible tasks, randomization, policies, environment and independent evaluation across A through E.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/harness.py, ipfs_accelerate_py/agent_supervisor/self_hosting/experiment.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/harness.py, ipfs_accelerate_py/agent_supervisor/self_hosting/experiment.py, test/api/self_hosting/test_harness.py, test/api/self_hosting/test_experiment_plan.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/harness.py, ipfs_accelerate_py/agent_supervisor/self_hosting/experiment.py, test/api/self_hosting/test_harness.py, test/api/self_hosting/test_experiment_plan.py
- Interfaces: SelfHostingQualificationHarness, create_experiment_plan
- Validation: python -m pytest -q test/api/self_hosting/test_harness.py test/api/self_hosting/test_experiment_plan.py
- Acceptance: Plan identity binds corpus/split/environment/model/price/policy/seeds; eligibility is identical; order is reproducible and balanced; configuration isolation tests reject leaked capsules, routing or reuse; replay development cannot count as live evaluation.
- Gap task: Implement the capstone harness and immutable five-arm experiment plan.
- Refinement: The harness orchestrates existing systems and does not become another coding-agent framework.
- Embedding query: SelfHostingQualificationHarness experiment plan configurations A B C D E randomization isolation
- AST query: SelfHostingQualificationHarness create_experiment_plan SelfHostingExperimentPlan
- Conflict policy: No arm may change its task set, hidden evaluator, environment or acceptance criteria.

## SHQ-G054 Execute configurations A and B

- Status: active
- Parent: SHQ-G050
- Depends on: SHQ-G053
- Fib priority: 90
- Track: baseline-configurations
- Priority: P0
- Bundle: agent-supervisor/self-hosting/config-ab
- Parallel lane: config-ab
- Resource class: model-frontier
- Token class: large
- Goal: Implement frontier ordinary-retrieval configuration A and frontier stateful-retrieval configuration B with full/normal verification and no semantic-capsule substitution.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/baselines.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/baselines.py, test/api/self_hosting/test_configurations_ab.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/baselines.py, test/api/self_hosting/test_configurations_ab.py
- Interfaces: FrontierBaselineConfiguration, StatefulRetrievalConfiguration
- Validation: python -m pytest -q test/api/self_hosting/test_configurations_ab.py
- Acceptance: A uses ordinary retrieval, frontier only, full required tests and no proof reuse; B adds only persistent task state and ordinary lexical/semantic retrieval; isolation tests fail on capsule or smaller-tier use.
- Gap task: Implement the two baseline strategy adapters over the common runtime.
- Refinement: A is the principal cost/quality baseline and B isolates state persistence.
- Embedding query: frontier baseline ordinary retrieval stateful retrieval full tests no semantic capsules
- AST query: FrontierBaselineConfiguration StatefulRetrievalConfiguration
- Conflict policy: Do not optimize baseline context, reuse incremental proofs or route to smaller models.

## SHQ-G055 Execute configuration C

- Status: active
- Parent: SHQ-G050
- Depends on: SHQ-G053
- Fib priority: 90
- Track: compression-configuration
- Priority: P0
- Bundle: agent-supervisor/self-hosting/config-c
- Parallel lane: config-c
- Resource class: model-frontier
- Token class: medium
- Goal: Implement frontier semantic-compression configuration C using admitted capsules and ContextPack construction with normal verification and no smaller-tier routing.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/compression.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/compression.py, test/api/self_hosting/test_configuration_c.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/compression.py, test/api/self_hosting/test_configuration_c.py
- Interfaces: SemanticCompressionConfiguration
- Validation: python -m pytest -q test/api/self_hosting/test_configuration_c.py
- Acceptance: C changes only context construction; model stays frontier; normal verification runs; context tokens, fallback, expansion and insufficiency are recorded; stale capsules reject.
- Gap task: Implement the semantic-compression-only strategy and isolation tests.
- Refinement: This arm isolates compression from routing and incremental verification savings.
- Embedding query: configuration C semantic compression ContextPack frontier normal verification no routing
- AST query: SemanticCompressionConfiguration ContextPacker SemanticCapsuleCompiler
- Conflict policy: Do not enable smaller models or incremental proof reuse in C.

## SHQ-G056 Execute configuration D

- Status: active
- Parent: SHQ-G050
- Depends on: SHQ-G053
- Fib priority: 90
- Track: routing-configuration
- Priority: P0
- Bundle: agent-supervisor/self-hosting/config-d
- Parallel lane: config-d
- Resource class: model-mixed
- Token class: medium
- Goal: Implement configuration D with semantic compression, provider-neutral model routing, incremental test/proof reuse and required frontier escalation.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/routed.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/routed.py, test/api/self_hosting/test_configuration_d.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/routed.py, test/api/self_hosting/test_configuration_d.py
- Interfaces: RoutedIncrementalConfiguration
- Validation: python -m pytest -q test/api/self_hosting/test_configuration_d.py
- Acceptance: D records every route and exact reused receipt; stale/uncovered/opaque impact broadens verification or escalates; missing tier cannot downgrade; no assurance sampling or governed release claim leaks from E.
- Gap task: Implement the compression-plus-routing strategy and exact reuse guards.
- Refinement: This arm measures direct inference savings without complete governed-system assurance.
- Embedding query: configuration D compression routing small medium frontier escalation incremental verification reuse
- AST query: RoutedIncrementalConfiguration ModelRoutePlanner IncrementalVerificationPlanner VerificationReceiptCache
- Conflict policy: Reuse only exact admitted receipts and never convert unavailable verification to pass.

## SHQ-G057 Execute configuration E

- Status: active
- Parent: SHQ-G050
- Depends on: SHQ-G044, SHQ-G053
- Fib priority: 100
- Track: governed-configuration
- Priority: P0
- Bundle: agent-supervisor/self-hosting/config-e
- Parallel lane: config-e
- Resource class: model-mixed-proof
- Token class: xlarge
- Goal: Implement the complete governed configuration E with compression auditing, context sufficiency expansion, routing, incremental verification, assurance sampling, proof sealing, shadow evaluation, human escalation and signed receipts.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/governed.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/governed.py, test/api/self_hosting/test_configuration_e.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/configurations/governed.py, test/api/self_hosting/test_configuration_e.py
- Interfaces: CompleteGovernedConfiguration
- Validation: python -m pytest -q test/api/self_hosting/test_configuration_e.py
- Acceptance: All admitted systems participate through executable or typed applicability receipts, including the WAL-specific VerifiedGuiOptimizer decision; critical omission, stale capsule/proof, surviving blocker, ambiguous recovery or incomplete human approval rejects; shadow samples are bounded; incremental seal verifies before acceptance; complete signed task receipt persists.
- Gap task: Compose the full governed strategy from existing components and exercise every fail-closed boundary.
- Refinement: The strategy contains policy wiring only and does not reimplement any prerequisite.
- Embedding query: configuration E governed compression audit assurance sampling proof sealing shadow human signed receipt
- AST query: CompleteGovernedConfiguration SemanticCompressionGovernor AdversarialAssuranceEngine IncrementalProofSealer
- Conflict policy: No simulated proof, self-approval, optional lifecycle step or silent policy bypass.

## SHQ-G058 Expose the required CLI and resume/status operations

- Status: active
- Parent: SHQ-G050
- Depends on: SHQ-G044, SHQ-G054, SHQ-G055, SHQ-G056, SHQ-G057, SHQ-G062, SHQ-G064, SHQ-G065
- Fib priority: 80
- Track: qualification-cli
- Priority: P0
- Bundle: agent-supervisor/self-hosting/cli
- Parallel lane: cli
- Resource class: cpu-small
- Token class: large
- Goal: Expose corpus, benchmark, economics, pilot, qualify, verify-release and report commands as thin projections of the required APIs with safe resume and stop semantics.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/cli.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/cli.py, test/api/self_hosting/test_cli.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/cli.py, test/api/self_hosting/test_cli.py, pyproject.toml
- Interfaces: self-hosting corpus build, self-hosting benchmark run, self-hosting benchmark resume, self-hosting qualify, self-hosting verify-release
- Validation: python -m pytest -q test/api/self_hosting/test_cli.py
- Acceptance: All requested commands exist; dry inspection is read-only; run/resume preserve exact bindings; stop cancels through typed control; verify-release performs current validation; nonzero exits enforce failed gates; CLI owns no duplicated business logic.
- Gap task: Add the thin command tree and test every required command and failure exit.
- Refinement: Existing interfaces can consume machine reports later; no GUI is added.
- Embedding query: self hosting CLI corpus inspect benchmark plan run resume compare economics pilot qualify verify release report
- AST query: cli corpus benchmark pilot qualify verify_release report
- Conflict policy: Do not hide failures with continue-on-error, warning exits or simulated provider success.

## SHQ-G060 Independent analysis, crash safety and qualification policy

- Status: active
- Parents: SHQ-G000, SHQ-G010
- Depends on:
- Fib priority: 100
- Track: qualification-analysis
- Priority: P0
- Bundle: agent-supervisor/self-hosting/analysis-root
- Parallel lane: analysis
- Resource class: cpu-large
- Token class: large
- Goal: Independently compare quality, context, routing, verification, safety, economics and longitudinal behavior and map evidence to a bounded qualification level.
- Evidence:
- Outputs:
- Validation: python -m pytest -q test/api/self_hosting/test_qualification_analysis.py
- Acceptance: Noninferiority is preregistered, uncertainty is reported, crashes recover or fail safely, economics separate observation from projection and level cannot exceed evidence.
- Refinement: Analysis, recovery, pilot and release gates remain independently testable.
- Conflict policy: No observed/hypothetical conflation, small-sample equivalence claim or level inflation.

## SHQ-G061 Consume outcome comparisons and evaluate preregistered noninferiority

- Status: active
- Parent: SHQ-G060
- Depends on: SHQ-G036, SHQ-G054, SHQ-G055, SHQ-G056, SHQ-G057
- Fib priority: 100
- Track: noninferiority
- Priority: P0
- Bundle: agent-supervisor/self-hosting/noninferiority
- Parallel lane: noninferiority
- Resource class: cpu-medium
- Token class: large
- Goal: Consume datasets-owned `TaskComparisonReport` records across A through E and evaluate E versus A using a margin frozen before held-out access, paired estimates, confidence intervals and zero-tolerance safety gates.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/analysis/noninferiority.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/analysis/noninferiority.py, test/api/self_hosting/test_noninferiority.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/analysis/noninferiority.py, test/api/self_hosting/test_noninferiority.py
- Interfaces: evaluate_noninferiority
- Validation: python -m pytest -q test/api/self_hosting/test_noninferiority.py
- Acceptance: The lower confidence bound for paired accepted-patch difference is compared with the frozen 2–5 point margin; critical regressions/security failures/stale evidence are zero-tolerance; hidden tests and reviewer outcomes are explicit; insufficient power returns inconclusive, never equivalent.
- Gap task: Implement deterministic noninferiority aggregation over independently produced datasets comparison reports with preregistration enforcement.
- Refinement: Default initial margin is five percentage points, but SHQ-G072 must freeze the exact value before held-out evaluation.
- Embedding query: accepted patch paired noninferiority margin confidence interval frontier governed hidden tests regression
- AST query: compare_task_outcomes evaluate_noninferiority NoninferiorityReport
- Conflict policy: Do not select the margin after viewing held-out outcomes or suppress unfavorable strata.

## SHQ-G062 Compute economics and the model-substitution matrix

- Status: active
- Parent: SHQ-G060
- Depends on: SHQ-G061
- Fib priority: 90
- Track: economics
- Priority: P0
- Bundle: agent-supervisor/self-hosting/economics
- Parallel lane: economics
- Resource class: cpu-small
- Token class: large
- Goal: Compute complete observed cost per task/accepted patch and task-class substitution evidence, then separately project five deployment scenarios at four annual volumes using explicit assumptions.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/analysis/economics.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/analysis/economics.py, test/api/self_hosting/test_economics.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/analysis/economics.py, test/api/self_hosting/test_economics.py
- Interfaces: calculate_cost_per_accepted_patch, create_model_substitution_matrix
- Validation: python -m pytest -q test/api/self_hosting/test_economics.py
- Acceptance: Model, verification, proof, shadow, human and failed-attempt costs are included; cached/local compute prices are explicit; API-only/local+API/enterprise/high-context/moderate-context scenarios cover 10k/100k/500k/1m tasks; projected values are labeled non-observed; matrix includes every required per-class field.
- Gap task: Implement exact aggregation, uncertainty and projection without fabricating savings.
- Refinement: Division by zero, missing prices and replay-only evidence produce typed insufficient-data outcomes.
- Embedding query: economics cost accepted patch model verification proof shadow human failed attempt annual volume substitution matrix
- AST query: calculate_cost_per_accepted_patch create_model_substitution_matrix EconomicAnalysis
- Conflict policy: Do not exclude failed attempts or verification cost to improve the savings claim.

## SHQ-G063 Implement and fixture-test all twelve crash boundaries

- Status: active
- Parent: SHQ-G060
- Depends on: SHQ-G043, SHQ-G052, SHQ-G057
- Fib priority: 100
- Track: crash-recovery
- Priority: P0
- Bundle: agent-supervisor/self-hosting/recovery-matrix
- Parallel lane: recovery-matrix
- Resource class: cpu-large
- Token class: xlarge
- Goal: Implement deterministic fault injection at repository scan, ContextPack persistence, model invocation, patch apply, state rescan, tests, proofs, receipt persistence, proof forest, seal, root CAS and result persistence and fixture-test exact recovery semantics without claiming a live qualification report.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/faults.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/faults.py, test/api/self_hosting/test_crash_recovery.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/faults.py, test/api/self_hosting/test_crash_recovery.py
- Interfaces: CrashRecoveryQualification, inject_stage_failure
- Validation: python -m pytest -q test/api/self_hosting/test_crash_recovery.py
- Acceptance: Fixtures cover every declared injection and prove recovery or typed repair; completed immutable artifacts are discovered; duplicate billing/effects are avoided where knowable; ambiguity never becomes success; partial tasks never count accepted; no fixture report is admissible as a current live qualification run.
- Gap task: Implement deterministic stage fault injection and exhaustive recovery fixtures; defer the current-tree live matrix to SHQ-G074.
- Refinement: Infrastructure failures may retry within bounds; benchmark task rejection is terminal evidence, not a retry trigger.
- Embedding query: crash recovery repository scan context persistence model patch tests proof receipt forest seal CAS result
- AST query: CrashRecoveryQualification inject_stage_failure resume_qualification_stage
- Conflict policy: Do not kill unrelated processes, edit state manually or convert unknown outcomes into pass.

## SHQ-G064 Implement a bounded disposable longitudinal pilot controller

- Status: active
- Parent: SHQ-G060
- Depends on: SHQ-G057, SHQ-G061, SHQ-G063
- Fib priority: 90
- Track: longitudinal-pilot
- Priority: P0
- Bundle: agent-supervisor/self-hosting/pilot
- Parallel lane: pilot
- Resource class: cpu-large
- Token class: xlarge
- Goal: Implement and fixture-test a controller for at most 20–50 accepted WAL maintenance changes on a disposable integration branch; perform no live pilot in this implementation goal.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/pilot.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/pilot.py, test/api/self_hosting/test_pilot.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/pilot.py, test/api/self_hosting/test_pilot.py
- Interfaces: run_longitudinal_pilot, start_pilot, stop_pilot, pilot_status
- Validation: python -m pytest -q test/api/self_hosting/test_pilot.py
- Acceptance: No protected-branch merge or production effect is possible; task/commit limit is exact; only sealed longitudinal-eligible accepted tasks enter; preconditions and rebase semantics are checked before each change; fewer than 20 composable tasks yields not-eligible; schema/circuit/key/canonicalization changes force full checkpoints; critical invariant stops; semantic/proof/cache/capsule/context/policy/chain/cost growth metrics persist; rollback verifies.
- Gap task: Implement the bounded pilot controller, composition queue and disposable-branch safety fixtures without starting a live pilot.
- Refinement: If initial held-out gates fail, pilot returns a terminal not-eligible report instead of blocking overall diagnosis.
- Embedding query: longitudinal self hosting pilot disposable branch checkpoint stop rollback 20 50 changes
- AST query: run_longitudinal_pilot LongitudinalPilotReport start_pilot stop_pilot
- Conflict policy: Zero automatic merges to protected branches, production deployment, sensitive data or unrestricted network effects.

## SHQ-G065 Determine the qualification level and project manifest inputs

- Status: active
- Parent: SHQ-G060
- Depends on: SHQ-G044, SHQ-G061, SHQ-G062, SHQ-G063, SHQ-G064
- Fib priority: 100
- Track: qualification-decision
- Priority: P0
- Bundle: agent-supervisor/self-hosting/decision
- Parallel lane: decision
- Resource class: cpu-small
- Token class: large
- Goal: Map exact evidence to Level 0–5, cap unsupported claims, project complete decision inputs for kit-owned manifest creation and issue explicit go/no-go recommendations for research, internal, external supervised and production stages.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/qualification.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/qualification.py, test/api/self_hosting/test_qualification_decision.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/qualification.py, test/api/self_hosting/test_qualification_decision.py
- Interfaces: determine_qualification_level, project_qualification_manifest_inputs
- Validation: python -m pytest -q test/api/self_hosting/test_qualification_decision.py
- Acceptance: Baseline/safety/reproducibility failures yield Level 0; small/inconclusive evidence caps at research; Levels 3–5 require every declared prerequisite; Level 4 requires independent review/reproduction/licensing/isolation/access; Level 5 cannot result from this one-package run.
- Gap task: Implement the deterministic decision table, required artifact inventory and recommendation projection; kit remains the sole manifest creator.
- Refinement: The decision applies only to exact target, release, tasks, configurations and policy.
- Embedding query: qualification level decision manifest research alpha internal pilot external supervised production candidate
- AST query: determine_qualification_level create_qualification_manifest QualificationDecision
- Conflict policy: Never infer readiness from component implementation alone or omit blockers from the manifest.

## SHQ-G066 Enforce fail-closed CI and current-release verification

- Status: active
- Parent: SHQ-G060
- Depends on: SHQ-G058, SHQ-G065
- Fib priority: 100
- Track: qualification-ci
- Priority: P0
- Bundle: agent-supervisor/self-hosting/ci
- Parallel lane: ci
- Resource class: cpu-large
- Token class: large
- Goal: Add qualification CI and verifier gates that reject skipped/failed/currentness-incomplete evidence and never publish after partial failure.
- Evidence: .github/workflows/self-hosting-qualification.yml, scripts/verify_self_hosting_qualification_release.py
- Outputs: .github/workflows/self-hosting-qualification.yml, scripts/verify_self_hosting_qualification_release.py, test/api/self_hosting/test_release_ci.py
- Predicted files: .github/workflows/self-hosting-qualification.yml, scripts/verify_self_hosting_qualification_release.py, test/api/self_hosting/test_release_ci.py
- Interfaces: verify_qualification_release
- Validation: python -m pytest -q test/api/self_hosting/test_release_ci.py
- Acceptance: Required jobs fail on any gate; dependencies are immutable; no continue-on-error, ignored exit, skipped check, historical current evidence, simulation, missing proof or setup warning is accepted; incomplete artifacts prevent release publication.
- Gap task: Implement a fail-closed workflow and current-tree release verifier with adversarial workflow tests.
- Refinement: CI may generate a failure report but never a signed success release after partial failure.
- Embedding query: fail closed CI qualification release verify current evidence no continue error skipped simulated proof
- AST query: verify_qualification_release workflow release gate
- Conflict policy: Models cannot modify qualification policy or trusted keys in benchmark worktrees.

## SHQ-G068 Implement preregistration and complete metric schemas

- Status: active
- Parent: SHQ-G060
- Depends on: SHQ-G032, SHQ-G062, SHQ-G063, SHQ-G064, SHQ-G065
- Fib priority: 100
- Track: preregistration-contracts
- Priority: P0
- Bundle: agent-supervisor/self-hosting/preregistration
- Parallel lane: preregistration
- Resource class: cpu-small
- Token class: large
- Goal: Implement canonical calibration-only policy proposals, external-freeze verification and complete aggregate metric validation without granting model workers authority to write the final preregistered policy.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/preregistration.py, ipfs_accelerate_py/agent_supervisor/self_hosting/metrics.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/preregistration.py, ipfs_accelerate_py/agent_supervisor/self_hosting/metrics.py, test/api/self_hosting/test_preregistration.py, test/api/self_hosting/test_qualification_metrics.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/preregistration.py, ipfs_accelerate_py/agent_supervisor/self_hosting/metrics.py, test/api/self_hosting/test_preregistration.py, test/api/self_hosting/test_qualification_metrics.py
- Interfaces: prepare_qualification_policy_proposal, verify_frozen_qualification_policy, validate_aggregate_metrics
- Validation: python -m pytest -q test/api/self_hosting/test_preregistration.py test/api/self_hosting/test_qualification_metrics.py
- Acceptance: Proposals consume development/calibration only and contain every margin, CI, compression, routing, assurance, shadow, human, model, price, resource and seed field; held-out access is impossible; frozen-policy verification requires authenticated external completion and exact source/environment/corpus bindings; aggregate validation enumerates every required metric and rejects missing strata.
- Gap task: Implement policy-proposal and metric-validation code while leaving the final protected policy exclusively operator controlled.
- Refinement: A model may propose bounded values from calibration; only the external SHQ-G072 authority freezes them.
- Embedding query: qualification preregistration proposal external freeze aggregate metrics complete schema calibration only
- AST query: prepare_qualification_policy_proposal verify_frozen_qualification_policy validate_aggregate_metrics
- Conflict policy: Never write the protected final policy, read held-out outcomes or redefine datasets metric semantics.

## SHQ-G067 Implement the integrated release-candidate freeze

- Status: active
- Parent: SHQ-G060
- Depends on: SHQ-G037, SHQ-G066, SHQ-G068
- Fib priority: 100
- Track: release-candidate-freeze
- Priority: P0
- Bundle: agent-supervisor/self-hosting/release-candidate-freeze
- Parallel lane: release-candidate-freeze
- Resource class: cpu-proof-solver
- Token class: large
- Goal: After every harness, corpus, storage, CLI, analysis and CI implementation lands, provide the operation that can bind a clean committed four-repository source projection, rerun current focused and WAL proof checks, and freeze the exact release-candidate environment without a self-referential manifest.
- Evidence: ipfs_accelerate_py/agent_supervisor/self_hosting/release_candidate.py, test/api/self_hosting/test_release_candidate_freeze.py
- Outputs: ipfs_accelerate_py/agent_supervisor/self_hosting/release_candidate.py, test/api/self_hosting/test_release_candidate_freeze.py
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_hosting/release_candidate.py, test/api/self_hosting/test_release_candidate_freeze.py
- Interfaces: freeze_release_candidate, ReleaseCandidateFreeze
- Validation: python -m pytest -q test/api/self_hosting/test_release_candidate_freeze.py
- Acceptance: Tests prove that the operation rejects dirty or mutable inputs, binds outer commit and recursive gitlinks, checks current harness/corpus/runtime/schema/store/CLI/CI versions, reruns focused prerequisite and target checks, requires unchanged WAL source or a new full checkpoint, regenerates lock/SBOM/container/toolchain/environment roots, and writes authoritative bytes only through kit ports; it cannot include its own evidence projection in the source identity.
- Gap task: Implement and fixture-test the post-integration release-candidate freeze operation; do not run the evidence program or manufacture final freeze artifacts in this implementation task.
- Refinement: SHQ-G071 invokes this committed implementation as its first operation, freezes the executable source before any experiment, then runs only detached worktrees from that root. Later evidence projections live on a distinct evidence branch and never redefine the qualified source commit.
- Embedding query: integrated release candidate exact commits gitlinks harness corpus runtime current tests WAL proof environment freeze
- AST query: freeze_release_candidate ReleaseCandidateFreeze environment manifest proof checkpoint source roots
- Conflict policy: Do not reuse the pre-implementation environment manifest as current evidence and do not repair target or prerequisite failures here.

## SHQ-G070 Execute the evidence program

- Status: active
- Parents: SHQ-G000, SHQ-G010
- Depends on:
- Fib priority: 100
- Track: qualification-execution
- Priority: P0
- Bundle: agent-supervisor/self-hosting/evidence-program
- Parallel lane: evidence-program
- Resource class: model-mixed-proof
- Token class: xlarge
- Goal: Execute development, calibration, held-out, crash and longitudinal stages in strict order and publish only the terminal evidence permitted by policy.
- Evidence:
- Outputs:
- Validation: python scripts/verify_self_hosting_qualification_release.py --check-current
- Acceptance: Held-out remains sealed until freeze; all eligible tasks run A–E; negative outcomes persist honestly; a release is signed only if publication gates pass.
- Refinement: Evidence execution is serial after parallel implementation converges.
- Conflict policy: No policy tuning from held-out data, cherry-picked task removal or partial-run success claim.

## SHQ-G071 Freeze the release candidate, then run development and calibration tasks

- Status: active
- Parent: SHQ-G070
- Depends on: SHQ-G067
- Fib priority: 90
- Track: development-calibration
- Priority: P0
- Bundle: agent-supervisor/self-hosting/evidence-program
- Parallel lane: evidence-program
- Resource class: model-mixed-proof
- Token class: xlarge
- Goal: First freeze the committed executable source and environment, then execute development and calibration splits through all five arms, validate instrumentation, tune only permitted compression/routing/assurance policies and emit a non-authoritative policy proposal for operator review.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/release_candidate_source.json, artifacts/agent_supervisor/self_hosting_qualification/release_candidate_environment.json, artifacts/agent_supervisor/self_hosting_qualification/release_candidate_proof_checkpoint.json, artifacts/agent_supervisor/self_hosting_qualification/development_results.json, artifacts/agent_supervisor/self_hosting_qualification/calibration_results.json, artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy_proposal.json
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/release_candidate_source.json, artifacts/agent_supervisor/self_hosting_qualification/release_candidate_environment.json, artifacts/agent_supervisor/self_hosting_qualification/release_candidate_proof_checkpoint.json, artifacts/agent_supervisor/self_hosting_qualification/development_results.json, artifacts/agent_supervisor/self_hosting_qualification/calibration_results.json, artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy_proposal.json
- Predicted files: artifacts/agent_supervisor/self_hosting_qualification/release_candidate_source.json, artifacts/agent_supervisor/self_hosting_qualification/release_candidate_environment.json, artifacts/agent_supervisor/self_hosting_qualification/release_candidate_proof_checkpoint.json, artifacts/agent_supervisor/self_hosting_qualification/development_results.json, artifacts/agent_supervisor/self_hosting_qualification/calibration_results.json, artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy_proposal.json
- Interfaces: freeze_release_candidate, benchmark run development, benchmark run calibration
- Validation: python -m ipfs_accelerate_py.agent_supervisor.self_hosting.cli benchmark compare --split calibration
- Acceptance: Before any task runs, a kit-persisted freeze binds a clean executable commit, recursive gitlinks, current toolchain/environment and full WAL checkpoint; every experiment uses a detached worktree at that frozen root; every eligible task has A–E receipts or explicit terminal infrastructure exclusion; replay results are labeled; instrumentation captures all required metric fields; no held-out outcome is read; the proposal is calibration-derived, non-authoritative and cannot overwrite the protected final policy.
- Gap task: Run the development/calibration experiment and persist complete current receipts.
- Refinement: Kit ports own authoritative immutable bytes and receipts. Listed repository artifacts are CID-verified projections on an evidence branch distinct from the frozen source; only calibration evidence may inform the next policy freeze.
- Embedding query: development calibration A E benchmark instrumentation policy tuning no held out
- AST query: benchmark calibration TaskExecutionReceipt
- Conflict policy: Benchmark rejection is an outcome, not a reason to retry until accepted.

## SHQ-G072 Freeze margins, policies, prices, routes and seeds

- Status: active
- Parent: SHQ-G070
- Depends on: SHQ-G071
- Fib priority: 100
- Track: preregistration
- Priority: P0
- Bundle: agent-supervisor/self-hosting/evidence-program
- Parallel lane: evidence-program
- Resource class: operator-review
- Token class: medium
- Completion authority: external
- External completion required: true
- Goal: Preregister the exact accepted-patch noninferiority margin, confidence method, compression/routing/assurance/shadow/human policies, model revisions, prices, resource rates and random seeds before held-out access.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy.json
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy.json
- Predicted files: artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy.json
- Interfaces: freeze_qualification_policy
- Validation: python -m ipfs_accelerate_py.agent_supervisor.self_hosting.cli benchmark plan --verify-frozen
- Acceptance: A typed external receipt proves the policy CID predates held-out access; margin is 2–5 points and exact; models/prices/seeds/resources are immutable; subsequent drift invalidates results; authenticated human approval is recorded.
- Gap task: Operator admission only; local task receipts cannot create or complete the preregistration.
- Refinement: This task requires controlled human approval but does not expose held-out outcomes. The operator admits policy bytes through the kit artifact port; the listed file is only a protected CID-verified evidence-branch projection.
- Embedding query: preregister noninferiority margin freeze policy model prices seed held out
- AST query: freeze_qualification_policy PreregisteredQualificationPolicy
- Conflict policy: Never amend the frozen policy after held-out access; a new policy requires a new qualification run.

## SHQ-G073 Run held-out configurations A through E

- Status: active
- Parents: SHQ-G070, SHQ-G072
- Depends on: SHQ-G072
- Fib priority: 100
- Track: held-out-execution
- Priority: P0
- Bundle: agent-supervisor/self-hosting/evidence-program
- Parallel lane: evidence-program
- Resource class: model-mixed-proof
- Token class: xlarge
- Goal: Execute every eligible held-out task through A–E under the frozen plan, retaining raw immutable receipts, failures, costs, latencies, context, verification and assurance evidence.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/held_out_results.json
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/held_out_results.json
- Predicted files: artifacts/agent_supervisor/self_hosting_qualification/held_out_results.json
- Interfaces: benchmark run held-out
- Validation: python -m ipfs_accelerate_py.agent_supervisor.self_hosting.cli benchmark compare --split held-out --verify-complete
- Acceptance: Same eligible set runs in every arm; no hidden-task omission; live versus replay is exact; cancellations/retries follow frozen policy; missing required evidence makes the run incomplete; all negative outcomes remain.
- Gap task: Run the sealed held-out experiment without policy mutation.
- Refinement: Kit ports own authoritative immutable bytes and receipts; the listed file is a CID-verified evidence-branch projection. Model-provider access may be required, but unavailable access produces an incomplete qualification rather than replayed success.
- Embedding query: held out benchmark configurations A B C D E frozen complete receipts
- AST query: benchmark held-out TaskExecutionReceipt
- Conflict policy: Do not replace live model-quality evidence with replay or remove expensive/failed tasks.

## SHQ-G074 Analyze held-out, assurance and recovery evidence

- Status: active
- Parents: SHQ-G070, SHQ-G073
- Depends on: SHQ-G062, SHQ-G063, SHQ-G073
- Fib priority: 100
- Track: held-out-analysis
- Priority: P0
- Bundle: agent-supervisor/self-hosting/evidence-program
- Parallel lane: evidence-program
- Resource class: cpu-large
- Token class: xlarge
- Goal: Produce results by configuration and task class, noninferiority, all required metrics/economics, substitution matrix and assurance findings, and execute the live twelve-boundary crash matrix on the frozen current tree.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/aggregate_metrics.json, artifacts/agent_supervisor/self_hosting_qualification/noninferiority_report.json, artifacts/agent_supervisor/self_hosting_qualification/economic_analysis.json, artifacts/agent_supervisor/self_hosting_qualification/model_substitution_matrix.json, artifacts/agent_supervisor/self_hosting_qualification/assurance_report.json, artifacts/agent_supervisor/self_hosting_qualification/crash_recovery_report.json
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/aggregate_metrics.json, artifacts/agent_supervisor/self_hosting_qualification/noninferiority_report.json, artifacts/agent_supervisor/self_hosting_qualification/economic_analysis.json, artifacts/agent_supervisor/self_hosting_qualification/model_substitution_matrix.json, artifacts/agent_supervisor/self_hosting_qualification/assurance_report.json, artifacts/agent_supervisor/self_hosting_qualification/crash_recovery_report.json
- Predicted files: artifacts/agent_supervisor/self_hosting_qualification/aggregate_metrics.json, artifacts/agent_supervisor/self_hosting_qualification/noninferiority_report.json, artifacts/agent_supervisor/self_hosting_qualification/economic_analysis.json, artifacts/agent_supervisor/self_hosting_qualification/model_substitution_matrix.json, artifacts/agent_supervisor/self_hosting_qualification/assurance_report.json, artifacts/agent_supervisor/self_hosting_qualification/crash_recovery_report.json
- Interfaces: benchmark compare, benchmark economics
- Validation: python -m ipfs_accelerate_py.agent_supervisor.self_hosting.cli benchmark compare --split held-out --verify-all-metrics
- Acceptance: Context/routing/quality/verification/compression/assurance/economics/performance metric families are schema-complete; CIs and task counts accompany claims; target misses are reported; the current-tree crash report covers all twelve boundaries and no fixture report substitutes for it; stale/simulated/critical omissions accepted remain zero or force failure.
- Gap task: Analyze the immutable held-out and recovery receipts and persist complete aggregate evidence.
- Refinement: Kit ports own authoritative immutable bytes and receipts; listed files are CID-verified evidence-branch projections. Hypothetical annual projections remain separate from observed costs.
- Embedding query: aggregate held out metrics noninferiority economics substitution assurance crash recovery
- AST query: AggregateQualificationMetrics NoninferiorityReport EconomicAnalysis AssuranceReport
- Conflict policy: Do not suppress strata, outliers, failures or uncertainty to meet initial targets.

## SHQ-G075 Run or truthfully decline the longitudinal pilot

- Status: active
- Parents: SHQ-G070, SHQ-G074
- Depends on: SHQ-G064, SHQ-G074
- Fib priority: 90
- Track: pilot-execution
- Priority: P0
- Bundle: agent-supervisor/self-hosting/evidence-program
- Parallel lane: evidence-program
- Resource class: model-mixed-proof
- Token class: xlarge
- Goal: If and only if initial gates pass, execute 20–50 accepted sequential maintenance changes; otherwise emit a terminal not-eligible report with the exact failed gates.
- Evidence: artifacts/agent_supervisor/self_hosting_qualification/longitudinal_pilot_report.json
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/longitudinal_pilot_report.json
- Predicted files: artifacts/agent_supervisor/self_hosting_qualification/longitudinal_pilot_report.json
- Interfaces: pilot start, pilot status, pilot stop
- Validation: python -m ipfs_accelerate_py.agent_supervisor.self_hosting.cli pilot status --verify-terminal
- Acceptance: Eligible pilot selects 20–50 composable tasks from the sealed longitudinal-eligible set that also passed acceptance, rechecks preconditions/rebases before each change and satisfies branch, route, count, checkpoint, review, invariant and rollback policy; fewer than 20 composable accepted tasks or any failed gate produces ineligible with no model/repository effects; neither case remains an infinite retry loop.
- Gap task: Evaluate pilot eligibility and either execute the bounded pilot or persist a truthful not-eligible outcome.
- Refinement: Kit ports own the authoritative pilot receipt; the listed file is a CID-verified evidence-branch projection. A negative benchmark result is valid qualification evidence and should terminate this task diagnostically.
- Embedding query: longitudinal pilot eligibility execute not eligible terminal evidence rollback
- AST query: LongitudinalPilotReport PilotEligibilityDecision
- Conflict policy: Never bypass failed held-out or crash gates merely to obtain pilot data.

## SHQ-G076 Emit the final report and conditionally signed release

- Status: active
- Parents: SHQ-G070, SHQ-G075
- Depends on: SHQ-G065, SHQ-G066, SHQ-G074, SHQ-G075
- Fib priority: 100
- Track: final-release
- Priority: P0
- Bundle: agent-supervisor/self-hosting/evidence-program
- Parallel lane: evidence-program
- Resource class: cpu-crypto
- Token class: xlarge
- Goal: Emit the complete machine-readable evidence, human-readable report, qualification decision and exact go/no-go recommendations; publish a signed release only when every publication gate passes.
- Evidence: docs/architecture/SELF_HOSTING_QUALIFICATION_REPORT.md, artifacts/agent_supervisor/self_hosting_qualification/qualification_decision.json
- Outputs: docs/architecture/SELF_HOSTING_QUALIFICATION_REPORT.md, artifacts/agent_supervisor/self_hosting_qualification/qualification_decision.json, artifacts/agent_supervisor/self_hosting_qualification/release
- Predicted files: docs/architecture/SELF_HOSTING_QUALIFICATION_REPORT.md, artifacts/agent_supervisor/self_hosting_qualification/qualification_decision.json, artifacts/agent_supervisor/self_hosting_qualification/release
- Interfaces: OperatorSigningPort, self-hosting qualify, self-hosting report, self-hosting verify-release
- Validation: python scripts/verify_self_hosting_qualification_release.py --check-current
- Acceptance: Report contains every requested final field, actual misses and limitations; decision applies only to exact release/task/model/policy; a complete valid run may request an operator signature and publish a signed research/alpha/negative qualification release even when targets miss; the model lane never receives private key bytes and denial is not bypassed; incomplete, stale, simulated, unverified or partially failed evidence yields a diagnostic report but no published release; every published release verifies from operator-admitted public keys.
- Gap task: Generate the terminal report and decision, then conditionally assemble/sign/verify the release without claiming beyond evidence.
- Refinement: Kit ports own authoritative decision/release bytes; listed files are CID-verified evidence-branch projections. This capstone cannot assign Level 5; Level 4 additionally requires independent review and reproduction outside this run.
- Embedding query: final qualification report signed release go no go research internal external production evidence
- AST query: QualificationDecision QualificationManifest verify_qualification_release
- Conflict policy: Do not publish incomplete artifacts, hide blockers or claim production readiness from implemented components.
