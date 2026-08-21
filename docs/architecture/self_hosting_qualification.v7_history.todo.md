# Self-Hosting Qualification v7 Task History

Historical record only. This board is never a scheduler source and supplies no
completion, receipt, proof, release, or retry authority.

Disposition:

- SHQ-012: rejected/cancelled retryable after attempt 1 following independent
  contract review at exact target HEAD
  `b1ea78f66073b5ceb6c22375cafc4bd80d0e1eec`. Coordination internal CID
  `baguqeeraokrailmmvgz3vc5tm6lcj2ttovwg6lxmtvgldvfrmlxkicucxsbq` was
  released by cancellation receipt CID
  `baguqeerakoa6upvffhceogv5rolwg4bxdwdcdnfwphni6fqsn7nffgt2z2za` as
  `cancelled:retryable` with null output. The attempt log SHA-256 is
  `575c48e0d4ade5b7f38dc330499d7e62a00dcb104a768f7051473c2995ab014a`;
  final validation reported 1 failed/29 passed. The core semantic failure
  was deliberate return of an unavailable result with an empty receipt key
  after a successful live process-runner execution, so the direct-execution
  receipt/cache path could never become authoritative. No
  `implementation_finished`, implementation commit, or merge occurred. Its
  dirty disposable attempt contained only the three declared output paths and
  is not completion evidence; SHQ-012 is superseded by the bounded-v8 G006
  projection.
- SHQ-013: never leased or launched because SHQ-012 did not merge; it is
  superseded by the bounded-v8 G007 projection.

The stopped attempt's worktree and supervisor/checkpoint state are not task
inputs, completion evidence, or retry authority. The canonical task blocks
below are preserved byte-for-byte from the retired active board; only this
preamble records their disposition.

## SHQ-012 Close objective gap: Install and test the prerequisite-state observer

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prerequisite-observation
- Depends on:
- Outputs: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode observe --quiet; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode require-terminal --quiet && exit 99 || test "$?" -eq 1
- Board namespace: self_hosting_qualification.todo.md
- Evidence inputs: data/agent_supervisor/self_hosting_qualification/discovery
- Discovery evidence: data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-012-objective-gap-c8da90812f65.md
- Bundle: agent-supervisor/self-hosting/prerequisite-observer-implementation-bounded-v7
- Bundle shard: data/agent_supervisor/self_hosting_qualification/projections/v7/bundles/agent-supervisor-self-hosting-prerequisite-observer-implementation-bounded-v7.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 0
- Parallel lane: prerequisite-observer-implementation-bounded-v7
- Conflict policy: Never modify prerequisite implementation or completion evidence from this task; never read sibling worktrees, operator state, hidden evaluator data, or arbitrary host paths.
- Predicted files: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Changed paths:
- Context paths: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- AST symbols: IncrementalSemanticIndex SemanticCapsuleCompiler ContextPacker ContextPackBuilder VerificationReceiptCache IncrementalVerificationPlanner ModelRoutePlanner VerifiedGuiOptimizer IncrementalProofSealer SemanticCompressionGovernor AdversarialAssuranceEngine
- Interfaces: observe_prerequisite_releases, PrerequisiteObservation, verification.contracts.VerificationIdentityCompiler, verification.process_runner.PROCESS_RUNNER_SCHEMA, verification.process_runner.VerificationProcessRunner, verification.process_runner.VerificationCommand, verification.process_runner.VerificationStreamArtifact, validation.validation_runtime.build_hermetic_validation_runtime, validation.validation_runtime.hermetic_validation_command, verification.contracts.TestReceipt@1, verification.contracts.DirectExecutionObservation@1, verification.receipt_cache.VerificationReceiptCache
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Generated artifacts:
- Allow concurrent with:
- Goal id: SHQ-G006
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/c5ce9eb20c4b360c276d1075d4a6399139e5c76d77e4596271b48a4a011a6d6d
- Canonical task CID: baguqeerayxhj5mqmjm3ayj3ncb25jjrzse46lr3no7sfsytrwsfeuai2nvwq
- Semantic identity: objective-evidence-obligation/v1/7c534d2f622c24baaac5c98252408f5cbb50b6fa59e35323c5c1c1be04eda68b
- Acceptance subset: Prior attempt 1 was hard-rejected for redirecting permitted `git show` stdout to host `/tmp` and rereading it. Any outside-checkout redirect, tee, copy, save, cache, checkpoint, materialization, or read is an immediate hard rejection, stop before validation. The only exception is writing an exact authorized blob seed directly to its matching declared output path inside the checkout, followed by audit and repair with no intermediate. `.gitignore` contains the exact narrow exception `!artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`. The prerequisite catalog is the exact non-empty ordered list of ten unique requested systems, empty, duplicate, omitted, added, or reordered catalogs fail closed. Every row binds the clean outer repository `HEAD` and tree, the exact recursive gitlink and matching submodule `HEAD` and tree, and deterministic digests/CIDs of the tracked source and evidence blobs used to decide it, dirty state or any mismatch is unverifiable. Every configured module, package export, release manifest, owner board, and receipt path is non-empty and repository-relative, contains no absolute root or `..` component, and after resolving every existing parent and symlink remains beneath its one declared checkout/submodule root, a missing, malformed, or escaping path fails closed. Interfaces are proved by exact module-level definition or assignment in the declared module and, when public, by its exact package export, aliases such as `ContextPacker` require an explicit complete versioned compatibility map covering every required operation and semantic constraint, never a partial symbol/name match or manufactured facade. Focused-test evidence is admitted only through the existing live direct-execution path: `VerificationIdentityCompiler.compile_key` with `receipt_kind=TEST` and `PROCESS_RUNNER_SCHEMA`, `build_hermetic_validation_runtime` plus `hermetic_validation_command` must produce an exact pinned Bubblewrap argv with `--unshare-net`, read-only host binding, bounded writable workspace, private `/tmp`, and no fallback, an actual same-process `VerificationProcessRunner.run(VerificationCommand)` executes that exact argv and returns schema `PROCESS_RUNNER_SCHEMA`, that live result is projected into its canonically keyed `DirectExecutionObservation` and `TestReceipt`, canonically round-tripped, then admitted and exact-key looked up through `VerificationReceiptCache` with production eligibility required. Before projection, compare the live result's executable, cwd, environment, sandbox, network policy, timeout, disposition, command argv, process/lease identity, and stream fields to the exact `VerificationCommand`, compiled key, and observed run, require the projected observation's stdout/stderr CIDs to equal the live result and include both CIDs in its artifacts. A declarative `deny_all` or sandbox label is not enforcement. Missing Bubblewrap, namespace denial, isolation startup failure, changed isolation argv, or any unisolated fallback makes the test evidence unverifiable. Constructing or caching the structural contracts does not authenticate execution origin, authority comes only from the observer's live in-process isolated runner call. The observer must not define a receipt schema, accept receipt JSON from disk/cache as execution proof, or accept a caller-supplied/injected phase report. Admission requires a present real run result, process started, completed disposition, `ok` and publication allowed, exact observed argv/selectors/tool/version, zero exit, no timeout/cancel/unavailable/simulation/replay, stdout and stderr content evidence, freshness, and identical clean pre/post `RepositoryForest` identities matching the outer/tree/gitlink/submodule/tracked-content source identity. Because `VerificationStreamArtifact` exposes only a preview at this boundary, each stream must be non-truncated and satisfy `captured_byte_count == byte_count == len(preview.encode("utf-8"))`, rehash the exact preview bytes to both declared digest and CID. Any larger, invalid/non-round-trippable UTF-8, partial, count-mismatched, digest-mismatched, or CID-mismatched stream is unverifiable, do not claim the discarded runner temporary bytes were persisted. Existing proof or semantic receipts are optional corroboration and insufficient without that current isolated direct run. Selector presence, stale evidence, or structurally valid but untrusted evidence is never proof or release evidence. Every owner board is fully parsed: every recognized task block has exactly one recognized status, and an unrecognized, missing, or duplicate status prevents terminal classification, the datasets boards `ipfs_datasets_py/docs/architecture/incremental_semantic_index.todo.md` and `ipfs_datasets_py/docs/architecture/semantic_state_contract.todo.md` are bound. Serialized roots and paths are canonical repository-relative values with deterministic row/field ordering and contain no host-absolute prefix. Ordinary `observe` succeeds for incomplete upstream releases but may serialize only a structurally complete ten-row observation, `require-terminal` additionally requires every row terminal. Before any output publication, recompute and compare the entire in-memory observation, outer `HEAD`/tree, recursive gitlinks/submodule identities, tracked-content digests, and all module/export/manifest/board/receipt inputs. Publish only through a same-directory exclusive temporary file plus durable atomic no-clobber replacement, refuse an existing target and remove temporary state on every failure. No incomplete, stale, partially validated, or source-raced artifact may be written. Prior-attempt access is limited to the two named blobs at commit `63ea88e41227d4d2d424f41051b9e9390c1a1c32`: use the exact authorized two-path diff command, or when a seed is necessary use exact `git show 63ea88e41227d4d2d424f41051b9e9390c1a1c32:<authorized-path>`. Consume stdout directly, or write each blob only to its matching declared output path and then audit and repair it, those two declared outputs are the sole persistence targets for code derived from the bounded input. No intermediate or scratch copy is evidence or authority, and every other revision or path is prohibited. The implementation agent must not use a host path outside the disposable checkout as a discovery source or scratch sink, specifically it must neither materialize the authorized bytes into nor read them from host `/tmp`, supervisor/checkpoint/state directories, or sibling worktrees. Generic checkpoint instructions grant no task-input authority. The required process runner's internal ephemeral stream capture and the validation namespace's private `/tmp` are permitted execution internals, but neither is a discovery source nor persisted evidence. This goal must not create or change `prerequisite_observation.json`.
- Preconditions: objective goal SHQ-G006 is schedulable
- Effects: satisfy evidence requirement: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, satisfy evidence requirement: test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Evidence subset: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Resource class: cpu-small
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-small
- Merge fate: objective/SHQ-G006
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/7c534d2f622c24baaac5c98252408f5cbb50b6fa59e35323c5c1c1be04eda68b
- Missing evidence: scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Embedding query: self hosting qualification prerequisite completion release board commit API focused tests observer
- AST query: IncrementalSemanticIndex SemanticCapsuleCompiler ContextPacker ContextPackBuilder VerificationReceiptCache IncrementalVerificationPlanner ModelRoutePlanner VerifiedGuiOptimizer IncrementalProofSealer SemanticCompressionGovernor AdversarialAssuranceEngine
- Surplus group: objective/SHQ-G006
- Merge key: 9ad3a4d635830842
- Merge family: objective/SHQ-G006
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
- Todo vector key: 10e4ecaca8968d25
- Acceptance: Objective scan filed this gap for SHQ-G006. Use evidence in data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-012-objective-gap-c8da90812f65.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py), and keep the supervisor-fed backlog aligned with the objective heap.  Prior attempt 1 was hard-rejected for redirecting permitted `git show` stdout to host `/tmp` and rereading it. Any outside-checkout redirect, tee, copy, save, cache, checkpoint, materialization, or read is an immediate hard rejection; stop before validation. The only exception is an exact authorized seed written straight to its matching declared output path inside the checkout and then audited and repaired with no intermediate. Resolve every required symbol and interface by exact AST/module and public-export inspection in its declared module, recognizing versioned functional interfaces such as `ContextPacker` only through the explicit complete compatibility map and never manufacturing missing facades or receipt authorities. Keep the ten-name catalog fixed and ordered; future releases are discovered only through its constrained per-entry package exports, release manifests, owner-board candidates, and live isolated direct-execution receipt path. Work only from the current disposable checkout. As bounded prior-attempt context, inspect only the two named blobs at commit `63ea88e41227d4d2d424f41051b9e9390c1a1c32`: run exactly `git show 63ea88e41227d4d2d424f41051b9e9390c1a1c32 -- scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py`; if a full-file seed is required, run only `git show 63ea88e41227d4d2d424f41051b9e9390c1a1c32:<authorized-path>` for one of those same two paths. Consume stdout directly or write it only to the corresponding declared output path, then audit and repair it against this goal rather than trusting or copying it blindly. Do not persist it anywhere else; no intermediate or scratch copy is evidence or authority. Do not access any other revision or path. Do not use host `/tmp`, a supervisor/checkpoint/state directory, a sibling worktree, or another outside-checkout host path as implementation-agent discovery input or scratch storage, even if an earlier attempt or operator created files there; generic checkpoint instructions grant no authority to consume them. This restriction does not prohibit the required runner's own ephemeral stream capture or its isolation namespace's private `/tmp`, which are execution internals and not discovery or persisted evidence. Do not enumerate unrelated refs. Independent tests cover: empty/duplicate/reordered catalogs; absolute, `..`, and symlink escapes for every configured path class; dirty outer state; outer `HEAD`/tree, recursive gitlink/submodule `HEAD`/tree, and tracked-blob digest mismatches; partial/module-local-but-unexported/malformed APIs and incomplete compatibility maps; missing/duplicate/unknown board status; pytest-adapter or other wrong runner schema, wrong canonical key, wrong command or selectors, missing real in-process run result, injected phase report, process-not-started, non-completed disposition, nonzero exit/non-pass status, wrong tool/version, mismatched live-result executable/cwd/environment/sandbox/network/timeout/disposition/process/lease fields, projected/live stream-CID mismatch, missing Bubblewrap, namespace denial, isolation startup failure, changed isolation argv or unisolated fallback, missing stream evidence, truncated or partially represented streams, byte-count/preview mismatch, digest/CID mismatch, timeout/cancel/unavailable, stale, simulated/replayed, cache-only, and untrusted structural receipt evidence; optional proof/semantic receipt without the current isolated direct run; repo-relative deterministic serialization; source mutation after initial scan; concurrent/existing destination races; failed `require-terminal` no-write; ordinary observe refusing partial output; and final whole-snapshot two-phase revalidation counterexamples.

## SHQ-013 Close objective gap: Generate the post-merge prerequisite observation snapshot

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prerequisite-observation
- Depends on: SHQ-012
- Outputs: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Validation: python3 -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode observe --quiet; python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode require-terminal --quiet && exit 99 || test "$?" -eq 1
- Board namespace: self_hosting_qualification.todo.md
- Evidence inputs: data/agent_supervisor/self_hosting_qualification/discovery
- Discovery evidence: data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-013-objective-gap-925ecd59554c.md
- Bundle: agent-supervisor/self-hosting/prerequisite-observation-snapshot-bounded-v7
- Bundle shard: data/agent_supervisor/self_hosting_qualification/projections/v7/bundles/agent-supervisor-self-hosting-prerequisite-observation-snapshot-bounded-v7.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 1
- Parallel lane: prerequisite-observation-snapshot-bounded-v7
- Conflict policy: Do not edit `.gitignore`, observer implementation, tests, prerequisite owners, release admission, policies, keys or generated supervisor state; never read arbitrary host paths.
- Predicted files: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Changed paths:
- Context paths: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- AST symbols: PrerequisiteObservation observation_to_json write_observation_artifact
- Interfaces: PrerequisiteObservation@1
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Generated artifacts:
- Allow concurrent with:
- Goal id: SHQ-G007
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/59545bc3fe7f5404a1e363511a0924574f1864864b0c484aace352408624eedb
- Canonical task CID: baguqeeralfkfxq76p5kajipdmnirucjek5hrqzegjmgeqsvm4njebbre53nq
- Semantic identity: objective-evidence-obligation/v1/2b17e60a3516287ce21a56894825629c819e5da2113bd1262eb6e0cbd8e55d83
- Acceptance subset: The only changed path is `artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`, the task begins from the clean merged current bounded-v7 G006 successor, never a retired task worktree or rescue branch. The observer exclusively and atomically publishes one structurally complete, deterministic, repository-relative ten-row snapshot only after its final whole-source revalidation, it refuses an existing output and leaves no partial artifact on failure. The snapshot binds exactly that pre-observation outer `HEAD`/tree, recursive gitlinks and matching submodule `HEAD`/trees, tracked-content digests, and the admitted existing receipt authorities while excluding only its own artifact path, all reads stay within the disposable task worktree and its three declared gitlinks. The artifact declares that it is neither completion nor proof nor release authority, its later artifact commit is an evidence projection and never claimed as the observed source, native validation and local two-pass completion receipts independently bind the clean post-artifact tree.
- Preconditions: objective goal SHQ-G007 is schedulable
- Effects: satisfy evidence requirement: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Evidence subset: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Resource class: cpu-small
- Token class: small
- Estimated tokens: 0
- Resources: cpu-small
- Merge fate: objective/SHQ-G007
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/2b17e60a3516287ce21a56894825629c819e5da2113bd1262eb6e0cbd8e55d83
- Missing evidence: artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
- Embedding query: post merge prerequisite observation snapshot clean source projection recursive gitlinks
- AST query: PrerequisiteObservation observation_to_json write_observation_artifact
- Surplus group: objective/SHQ-G007
- Merge key: fee3d856a0d71724
- Merge family: objective/SHQ-G007
- Merge role: aggregate
- Work item count: 1
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: bdca5f32aa0623f1
- Acceptance: Objective scan filed this gap for SHQ-G007. Use evidence in data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-013-objective-gap-925ecd59554c.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json), and keep the supervisor-fed backlog aligned with the objective heap.  Require the freshly projected bounded-v7 G006 canonical task CID as the sole predecessor identity; a retired display ID, alias, canonical key, CID, worktree, receipt, or merge cannot satisfy this dependency. Refuse dirty or source-raced input, any outer/tree/gitlink/submodule/tracked-content mismatch, a retired predecessor identity, an already present output, or an incomplete/non-deterministic snapshot; never read sibling worktrees or operator state, repair or upgrade a prerequisite, invent a receipt authority, or turn an observe result into terminal admission.
