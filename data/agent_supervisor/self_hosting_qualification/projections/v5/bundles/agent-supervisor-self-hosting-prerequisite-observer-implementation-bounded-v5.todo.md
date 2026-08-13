# Objective Bundle: agent-supervisor/self-hosting/prerequisite-observer-implementation-bounded-v5

Source todo: docs/architecture/self_hosting_qualification.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## SHQ-008 Close objective gap: Install and test the prerequisite-state observer

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prerequisite-observation
- Depends on:
- Outputs: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Validation: /usr/bin/python3.12 -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py; /usr/bin/python3.12 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode observe --quiet; /usr/bin/python3.12 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode require-terminal --quiet && exit 99 || test "$?" -eq 1
- Board namespace: self_hosting_qualification.todo.md
- Evidence inputs: data/agent_supervisor/self_hosting_qualification/discovery
- Discovery evidence: data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-008-objective-gap-c8da90812f65.md
- Bundle: agent-supervisor/self-hosting/prerequisite-observer-implementation-bounded-v5
- Bundle shard: data/agent_supervisor/self_hosting_qualification/projections/v5/bundles/agent-supervisor-self-hosting-prerequisite-observer-implementation-bounded-v5.todo.md
- Bundle strategy: explicit
- Graph parents: SHQ-G005
- Graph depth: 2
- Objective heap index: 0
- Parallel lane: prerequisite-observer-implementation-bounded-v5
- Conflict policy: Never modify prerequisite implementation or completion evidence from this task; never read sibling worktrees, operator state, hidden evaluator data, or arbitrary host paths.
- Predicted files: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- Changed paths:
- Context paths: .gitignore, scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
- AST symbols: IncrementalSemanticIndex SemanticCapsuleCompiler ContextPacker ContextPackBuilder VerificationReceiptCache IncrementalVerificationPlanner ModelRoutePlanner VerifiedGuiOptimizer IncrementalProofSealer SemanticCompressionGovernor AdversarialAssuranceEngine
- Interfaces: observe_prerequisite_releases, PrerequisiteObservation, verification.contracts.VerificationIdentityCompiler, verification.process_runner.VerificationProcessRunner, verification.process_runner.VerificationCommand, verification.contracts.TestReceipt@1, verification.contracts.DirectExecutionObservation@1, verification.receipt_cache.VerificationReceiptCache
- Submodules: ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus
- Generated artifacts:
- Allow concurrent with:
- Goal id: SHQ-G006
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/aa887128298a38a0a483b40e1e9f25109ad94df3e4eba79c6cee64c467d364eb
- Canonical task CID: baguqeeravkehckbjri4kbjedwqhb5hzfccnnstpt4tv2phdm5zsmiz6tmtvq
- Semantic identity: objective-evidence-obligation/v1/7c534d2f622c24baaac5c98252408f5cbb50b6fa59e35323c5c1c1be04eda68b
- Acceptance subset: `.gitignore` contains the exact narrow exception `!artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`. The prerequisite catalog is the exact non-empty ordered list of ten unique requested systems, empty, duplicate, omitted, added, or reordered catalogs fail closed. Every row binds the clean outer repository `HEAD` and tree, the exact recursive gitlink and matching submodule `HEAD` and tree, and deterministic digests/CIDs of the tracked source and evidence blobs used to decide it, dirty state or any mismatch is unverifiable. Every configured module, package export, release manifest, owner board, and receipt path is non-empty and repository-relative, contains no absolute root or `..` component, and after resolving every existing parent and symlink remains beneath its one declared checkout/submodule root, a missing, malformed, or escaping path fails closed. Interfaces are proved by exact module-level definition or assignment in the declared module and, when public, by its exact package export, aliases such as `ContextPacker` require an explicit complete versioned compatibility map covering every required operation and semantic constraint, never a partial symbol/name match or manufactured facade. Focused-test evidence is admitted only through the existing direct-execution authority: `VerificationIdentityCompiler.compile_key` with `receipt_kind=TEST` and `PYTEST_VERIFICATION_ADAPTER_SCHEMA`, an actual `VerificationProcessRunner.run(VerificationCommand)`, trusted in-process construction of its canonically keyed `DirectExecutionObservation` and `TestReceipt`, canonical `TestReceipt.from_dict(receipt.to_record())` round-trip, then `VerificationReceiptCache.admit(..., require_production_eligible=True)` plus exact-key lookup. The observer must not define a receipt schema, accept a receipt JSON supplied from disk or cache as execution proof, or treat a caller-supplied/injected pytest phase report as live execution. Admission requires a present real run result, process started, completed disposition, `ok` and publication allowed, exact observed argv/selectors/tool/version, zero exit, no timeout/cancel/unavailable/simulation/replay, stdout and stderr CIDs, freshness, and identical clean pre/post `RepositoryForest` identities matching the outer/tree/gitlink/submodule/tracked-content source identity. Existing proof `TestPassReceipt`/`SignedTestPassReceiptV2` and semantic `CompiledReceipt` may be recorded only as optional corroboration and are insufficient without that current direct run. Selector presence, stale evidence, or structurally valid but untrusted evidence is never release evidence. Every owner board is fully parsed: every recognized task block has exactly one recognized status, and an unrecognized, missing, or duplicate status prevents terminal classification, the datasets boards `ipfs_datasets_py/docs/architecture/incremental_semantic_index.todo.md` and `ipfs_datasets_py/docs/architecture/semantic_state_contract.todo.md` are bound. Serialized roots and paths are canonical repository-relative values with deterministic row/field ordering and contain no host-absolute prefix. Ordinary `observe` succeeds for incomplete upstream releases but may serialize only a structurally complete ten-row observation, `require-terminal` additionally requires every row terminal. Before any output publication, recompute and compare the entire in-memory observation, outer `HEAD`/tree, recursive gitlinks/submodule identities, tracked-content digests, and all module/export/manifest/board/receipt inputs. Publish only through a same-directory exclusive temporary file plus durable atomic no-clobber replacement, refuse an existing target and remove temporary state on every failure. No incomplete, stale, partially validated, or source-raced artifact may be written. All discovery and validation reads stay within the disposable task worktree and its declared submodules, recursive search of `/home`, sibling worktrees, supervisor state, or any other host path is prohibited. This goal must not create or change `prerequisite_observation.json`.
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
- Acceptance: Objective scan filed this gap for SHQ-G006. Use evidence in data/agent_supervisor/self_hosting_qualification/discovery/2026-08-13-shq-008-objective-gap-c8da90812f65.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py, test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py), and keep the supervisor-fed backlog aligned with the objective heap.  Resolve every required symbol and interface by exact AST/module and public-export inspection in its declared module, recognizing versioned functional interfaces such as `ContextPacker` only through the explicit complete compatibility map and never manufacturing missing facades or receipt authorities. Keep the ten-name catalog fixed and ordered; future releases are discovered only through its constrained per-entry package exports, release manifests, owner-board candidates, and direct-execution receipt admission. Work only from the current disposable checkout. As bounded prior-attempt context, inspect exactly `git show 63ea88e41227d4d2d424f41051b9e9390c1a1c32 -- scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py`; audit and repair it against this goal rather than trusting or copying it blindly. Do not enumerate unrelated refs or search outside the checkout. Independent tests cover: empty/duplicate/reordered catalogs; absolute, `..`, and symlink escapes for every configured path class; dirty outer state; outer `HEAD`/tree, recursive gitlink/submodule `HEAD`/tree, and tracked-blob digest mismatches; partial/module-local-but-unexported/malformed APIs and incomplete compatibility maps; missing/duplicate/unknown board status; forged schema, wrong canonical key, wrong command or selectors, missing real run result, injected phase report, process-not-started, non-completed disposition, nonzero exit/non-pass status, wrong tool/version, missing stdout/stderr CID, timeout/cancel/unavailable, stale, simulated/replayed, cache-only, and untrusted receipt evidence; optional proof/semantic receipt without the current direct run; repo-relative deterministic serialization; source mutation after initial scan; concurrent/existing destination races; failed `require-terminal` no-write; ordinary observe refusing partial output; and final whole-snapshot two-phase revalidation counterexamples.
