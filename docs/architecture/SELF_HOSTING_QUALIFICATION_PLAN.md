# Self-Hosting Qualification Harness — Comprehensive Integration Plan

## 1. Outcome and governing decision

Build one capstone subsystem, `SelfHostingQualificationHarness`, and one narrow
facade, `GovernedCodingAgentRuntime`, then use them to qualify exactly one
bounded Python target. The work is an integration and evidence program. It does
not authorize another analyzer, agent framework, capsule format, proof system,
model provider, transport, storage backend, dataset, GUI, or MCP++ profile.

The selected target is:

```text
endomorphosis/ipfs_kit_py
└── ipfs_kit_py/core/wal
```

The target will be bound to an exact clean commit only after the prerequisite
release gate and baseline pass. `ipfs_kit_py/core/operation_contracts.py` is a
read-only dependency for target tasks. Qualification infrastructure, policies,
trusted keys, hidden evaluators, generated evidence, legacy WAL modules, and
unrelated public APIs are prohibited patch targets.

The supervisor-native source of truth is
`docs/architecture/self_hosting_qualification.objectives.md`. It contains the
goal hierarchy, dependencies, parallel bundles, resource classes, evidence,
outputs, validation and acceptance clauses. The objective daemon generates the
todo board, conflict graph, datasets, bundle shards and Profile-G canonical task
payloads. Generated task IDs and CIDs are projections; the stable planning IDs
are the `SHQ-G...` goal IDs.

## 2. Critical prerequisite finding

The assertion that all ten prerequisite systems are complete is not true in the
currently inspected workspace. The plan therefore treats prompt text as a
requirement, not as release evidence.

Point-in-time observations on 2026-08-13:

| System | Observed state | Admission consequence |
|---|---|---|
| `IncrementalSemanticIndex` | Completed implementation found at `b572255d...`, integrated into datasets tip `1330038f...` | Re-run focused tests at the frozen release |
| `SemanticCapsuleCompiler` | Functional `@1` API found in datasets `1330038f...` | Bind exact versioned interface rather than require a cosmetic class |
| `ContextPackBuilder` | Existing implementation is named `ContextPacker` / `pack_context` | Admit an explicit compatibility mapping; do not rebuild it |
| `VerificationReceiptCache` | Completion branch `c1b9980e...`, present in newer accelerate main | Re-run exact focused tests |
| `IncrementalVerificationPlanner` | Completion branch `c1b9980e...`, present in newer accelerate main | Re-run exact focused tests |
| `ModelRoutePlanner` | Completion branch `c1b9980e...`, present in newer accelerate main | Re-run exact focused tests |
| `VerifiedGuiOptimizer` | Owning supervisor still had roughly half its board open | Wait for terminal owner evidence |
| `IncrementalProofSealer` | Owning supervisor still had dozens of todo items and blockers | Wait for terminal owner evidence |
| `SemanticCompressionGovernor` | Owning supervisor was actively implementing an unfinished board | Wait for terminal owner evidence |
| `AdversarialAssuranceEngine` | No exact released symbol or terminal board was found | Require an authoritative released capability; do not substitute a new engine |

Observed repository tips are context only, not the qualification baseline:

| Repository | Observed revision |
|---|---|
| `ipfs_datasets_py` semantic-state tip | `1330038f626ef92993f03d46f21e1a57719e9c25` |
| `ipfs_kit_py` `origin/main` | `05ba9375923cd5fb52e2c9c18b98b530d57d077f` |
| `ipfs_accelerate_py` newer `origin/main` | `bc99663b55a823bd992b777cf826e97b74809a5a` |
| `Mcp-Plus-Plus` | `dc3164653a48d059ae9812078359daeafb451c07` |

The planning branch starts from an in-flight semantic-governor integration
revision so it can use the current objective and bundle supervisor. That branch
is an operator workspace, not baseline evidence. `SHQ-G010` admits final
released prerequisite commits only after an operator has converged the clean
capstone integration branch and all gitlinks to those exact revisions; no local
agent receipt can substitute for that merge/pin authority. `SHQ-G021` then
freezes the admitted baseline.

The reviewed v11 `SHQ-G005A` merge already supplies one narrow compatibility
repair in the existing verification identity compiler: a private immutable closed
`frozenset({("bwrap", "bubblewrap")})` is consulted only by the banner-name
predicate after every existing executable, selector, probe, reviewed tool,
byte, CID and version binding. The keyed tool and executable basename remain
exact `bwrap`. For exact bound bwrap, both the ordinary `bwrap` name and the
reviewed `bubblewrap` alias are accepted only when the unmodified raw probe
output is exactly one canonical banner line formed as that lower-case name,
one ASCII space, the independently token-bound claimed ASCII version, and one
terminal LF. The actual host positive is exactly `b"bubblewrap 0.9.0\n"`, and
must use those actually observed raw bytes without rewrite or synthesis. Exact
`b"bwrap 0.9.0\n"` is the other sole form only in a bounded pure-compiler
fixture that still binds the actual executable bytes and SHA-256; that fixture
is not live execution evidence or authority. Paths,
help/usage/error/diagnostic prose, cross-line name/version separation, extra
lines or text, prefixes/suffixes, CR, tabs, doubled spaces, missing/final extra
whitespace, wrappers, rewritten banners and synthetic probes are rejected.
Only non-bwrap legacy exact-name behavior is unchanged. Rebinding the private
module-global constant to a caller extension or superset cannot expand the
hard-coded canonical pair or two-value raw-byte set and remains rejected.

This bounded task is neither resumable nor long-running; the generic
durable-checkpoint clause is inapplicable and expressly revoked for the
implementation agent. No autonomous model-issued shell/file-tool may reference
or expand `$IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR`,
`authority.durable_checkpoint.directory`, or `scope.checkpoint_directory`;
deliberately forward their values as task input or tool arguments; or use them
to print, list, stat, resolve, hash, read, write, inspect, test, enumerate, copy,
source, execute, create, modify, redirect, tee, save, cache, checkpoint,
materialize, or reread the named checkpoint directory, any alias, resolution,
or descendant of it, or other supervisor/checkpoint/runtime state outside the
workspace. The implementation agent may not use those paths or bytes as
discovery, scratch, evidence, completion, or retry input. This checkpoint
revocation does not revoke the task's separately explicit read/execute
authority for actual `/usr/bin/bwrap`, the listed validation interpreter/tools
and bwrap argv, or declared gitlinks. The only permitted exceptions to this
checkpoint/temp-state prohibition are supervisor/runner-private lifecycle
operations outside the implementation agent and fresh transient temp/stream
objects automatically owned by the listed validation/test runner, including
pytest fixture internals;
G006A/G006B/G006/G007 additionally permit only the required process-runner stream capture
and Bubblewrap namespace-private `/tmp`. None is implementation-agent discovery
or scratch, persisted evidence, or prior/private-state input. This revocation
is repeated for G006A, G006B, G006 and G007 and each task's own injected checkpoint paths;
each task may consume its predecessor only as clean merged tracked repository
content, never through predecessor runtime or checkpoint state.

The bounded-v12 observer is deliberately stricter than a file-presence scan.
G006A begins from the clean migration/projection descendant carrying the
reviewed G005A bytes; G006B begins only after G006A is merged and clean; G006
begins only after G006B is merged and clean; G007 runs only after G006 is
merged and clean.
The prerequisite catalog is the exact non-empty ordered list of ten unique
requested systems; omission, addition, duplication, or reordering fails closed.
For each row the observer binds the clean outer repository `HEAD` and tree,
the exact recursive gitlink and matching submodule `HEAD` and tree, and
deterministic digests/CIDs of every tracked source and evidence blob used in
the decision. Every configured module, package-export, release-manifest,
owner-board, and receipt path must be non-empty and repository-relative, contain
neither an absolute root nor `..`, and remain beneath its one declared
checkout/submodule root after existing-parent and symlink resolution.

Board parsing is self-contained and closed. A task heading matches only
`^##[ \t]+(?P<task_id>[A-Z][A-Z0-9_]*-[0-9]+)(?:[ \t]+.*)?$`; its block ends
only at the next `^##(?:[ \t]|$)` or EOF. A status matches only
`^[ \t]*-[ \t]+Status:[ \t]*(?P<value>[^\r\n]*)[ \t]*$`. Each board must
contain nonzero unique task headings and exactly one status per block;
the closed status tokens are `completed`, `todo`, `blocked`, `in_progress`,
`review`, and `cancelled`, and a board is terminal iff every status is
`completed`. Prose and deeper headings never count as tasks or statuses.

The G006A catalog is normative and ordered. A configured present-path field is
never guessed or inferred; rows explicitly marked `expected_absent` may carry
null path/interface/receipt fields and are necessarily nonterminal.

| # | Requested owner | Exact clean-tree mapping |
|---|---|---|
| 1 | `IncrementalSemanticIndex` | root `ipfs_datasets_py`; module `ipfs_datasets_py/logic/software_contracts/semantic_index/index.py`; public export `ipfs_datasets_py/logic/software_contracts/semantic_index/__init__.py`; API class plus `scan_repository`, `diff_repository_states`, `calculate_invalidation`, `explain_symbol`, `explain_impact`, `watch_repository`; release `docs/software_contracts/INCREMENTAL_SEMANTIC_INDEX.md`; board `docs/architecture/incremental_semantic_index.todo.md`; selector `tests/unit/logic/software_contracts/semantic_index/test_api.py` |
| 2 | `SemanticCapsuleCompiler` | root `ipfs_datasets_py`; module `ipfs_datasets_py/logic/software_contracts/semantic_state/capsules.py`; exact `SEMANTIC_CAPSULE_COMPILER_INTERFACE == "SemanticCapsuleCompiler@1"`; module API `compile_semantic_capsule`, `compile_semantic_capsules`, `verify_capsule_compile_result`; package `ipfs_datasets_py/logic/software_contracts/semantic_state/__init__.py` exports only singular `compile_semantic_capsule`, so constant/plural/verify are module-public only; release `docs/software_contracts/SEMANTIC_STATE_CONTRACT.md`; board `docs/architecture/semantic_state_contract.todo.md`; selector `tests/unit/logic/software_contracts/semantic_state/test_capsules.py` |
| 3 | `ContextPackBuilder` | compatibility label only, root `.`; module-public surface `ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py`; parent package exports `ContextPack` only, while builder operations remain module-public and there is no `ContextPackBuilder` facade; interface `ContextPack@1`; release `docs/semantic_state/SEMANTIC_COMPRESSION_HARNESS.md`; dependency seal `config/semantic_state_dependencies.seal.json` schema `ipfs-accelerate.agent-supervisor.semantic-state-dependency-seal@2`; board `docs/architecture/semantic_compression_harness.todo.md`; selector `test/api/semantic_state/test_context_pack.py`; benchmark `docs/benchmarks/semantic_compression_harness_results.json` schema `ipfs_accelerate_py/semantic-state/benchmark-report@1` is corroboration only |
| 4 | `VerificationReceiptCache` | root `.`; module `ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py`; exact lazy package export from `ipfs_accelerate_py/agent_supervisor/verification/__init__.py`; interface `VerificationReceiptCache@1`; release `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md` schema `ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@2`; board `docs/architecture/incremental_verification_planner.todo.md`; selector `test/api/test_agent_supervisor_verification_receipt_cache.py` |
| 5 | `IncrementalVerificationPlanner` | root `.`; module `ipfs_accelerate_py/agent_supervisor/verification/planner.py`; exact lazy package export from `ipfs_accelerate_py/agent_supervisor/verification/__init__.py`; interface `IncrementalVerificationPlanner@1`; release `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md` schema `ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@2`; board `docs/architecture/incremental_verification_planner.todo.md`; selector `test/api/test_agent_supervisor_incremental_verification_planner.py` |
| 6 | `ModelRoutePlanner` | root `.`; module `ipfs_accelerate_py/agent_supervisor/verification/model_route.py`; exact lazy package export from `ipfs_accelerate_py/agent_supervisor/verification/__init__.py`; interface `ModelRoutePlanner@1`; release `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md` schema `ipfs_accelerate_py/agent-supervisor/incremental-verification-release-report-binding@2`; board `docs/architecture/incremental_verification_planner.todo.md`; selector `test/api/test_agent_supervisor_verification_model_route.py` |
| 7 | `VerifiedGuiOptimizer` | exact `expected_absent`: module/export/release/board/selector/receipt paths are null; no guessed facade or owner path; always nonterminal |
| 8 | `IncrementalProofSealer` | exact `expected_absent`: module/export/release/board/selector/receipt paths are null and the interoperability inventory reports typed unavailable; always nonterminal |
| 9 | `SemanticCompressionGovernor` | root `.` and present board `docs/architecture/semantic_compression_governor.todo.md`; expected future module `ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py`, export `ipfs_accelerate_py/agent_supervisor/semantic_governor/__init__.py`, selector `test/api/semantic_governor/test_public_api.py`, and release `artifacts/agent_supervisor/semantic_compression_governor/release.json` are all absent and non-executable metadata; `interface` and `receipt_path` are null; exact state `expected_absent_pending_owner`, always nonterminal |
| 10 | `AdversarialAssuranceEngine` | exact `expected_absent`: module/export/release/board/selector/receipt paths are null; no guessed facade or owner path; always nonterminal |

Rows 4–6 may bind the corroborating
`artifacts/agent_supervisor/incremental_verification/benchmark.json` only when
its schema is exactly
`ipfs_accelerate_py/agent-supervisor/incremental-verification-benchmark@2`.
For every present row, authoritative current test evidence maps only to the
live in-memory `DirectExecutionObservation@1` → `TestReceipt@1` chain with
`PROCESS_RUNNER_SCHEMA`; reports, manifests, seals and benchmarks are
corroboration, never receipt authority. Expected-absent rows have no receipt
authority.

No row has an authoritative filesystem `receipt_path`. The global declarations
are `receipt_interface="TestReceipt@1"`,
`receipt_schema="ipfs_accelerate_py/agent-supervisor/verification-test-receipt@1"`,
`observation_interface="DirectExecutionObservation@1"`,
`observation_schema="ipfs_accelerate_py/agent-supervisor/direct-verification-observation@1"`,
and runner schema
`ipfs_accelerate_py/agent-supervisor/verification-process-runner@1`. G006A sets
`receipt_id`, `key_id`, and `observation_content_id` to null with typed reason
`terminal_chain_not_run`; reports/manifests/benchmarks appear only under
`corroboration_paths`. G006B may populate those IDs only from the trusted
same-process live result after canonical round-trip, production cache admission
and exact lookup: `TestReceipt.receipt_id`,
`VerificationReceiptKey.key_id`, and
`DirectExecutionObservation.content_id`. Expected-absent rows keep all three
IDs null, no selector, and exact limitation
`owner_contract_not_declared_on_launch_tree`. Every non-null configured path is
confined and validated; null is the only path representation for an undeclared
owner contract. The ten requested names and order are invariant, while absent
metadata can change only through a later reviewed protected-catalog amendment
when an owner contract lands.

`ContextPackBuilder` is never a class or facade. Construction maps exactly to
`ContextPacker(budget=ContextBudget(), policy=ContextCoveragePolicy(),
estimator_version=TOKEN_ESTIMATOR_VERSION)`; build maps both to
`ContextPacker.pack(...)` and module function `pack_context(...)`; projection
maps to `project_admission_to_reference(CapsuleAdmission, token_count=0)`.
Common keyword-only inputs are exactly `objective`, `target_source_cid`,
`surrounding_source_cid`, `test_source_cid`, `dependency_admissions=()`,
`obligation_cids=()`, `counterexample_cids=()`, `delta_cid`,
`interface_cids=()`, `assumptions=()`, `exclusions=None`,
`raw_source_regions=()`, `production_slice=None`, and
`production_slice_builder=None`; the functional entry also accepts
`budget=None`, `policy=None`, and
`estimator_version=TOKEN_ESTIMATOR_VERSION`. Output is exact
`ContextPackResult` with schema `ipfs-accelerate.context-pack-result@1`,
interface `ContextPack@1`, and in-memory fields `pack`, `pack_cid`,
`references`, `token_estimate`, `coverage_satisfied`, `production_slice`,
`production_slice_cid`, `budget_exceeded`, and `decisions`; policy schema is
`ipfs-accelerate.context-coverage-policy@1` and estimator is
`context-compiler-calibrated_utf8@1`. The embedded `ContextPack` fields are
exactly `objective`, `target_source_cid`, `surrounding_source_cid`,
`test_source_cid`, `dependency_capsule_cids`, `obligation_cids`,
`counterexample_cids`, `delta_cid`, `interface_cids`, `assumptions`,
`exclusions`, `token_totals`, `estimator_version`, `risk`, `route`, and
`escalation_recommendation`. `ContextPackResult.to_dict()` serializes exactly
`schema` and `interface` plus `pack`, `pack_cid`, `references`,
`token_estimate`, `coverage_satisfied`, `production_slice_cid`,
`budget_exceeded`, and `decisions`; it never serializes the optional
`production_slice` object itself. The target/surrounding/test source are exact
required `INVARIANT` references and never compressed. Capsule substitution is
allowed only when both `allow_capsule_substitution` and
`capsule_may_substitute`; otherwise bind raw source and an explained exclusion.
Explicit raw regions remain raw, obligations/delta are required evidence,
exclusions require explanations, ordering/CID construction is deterministic,
and budget or coverage failure never truncates required coverage but instead
sets coverage false and escalates to human review. Heuristic/model summaries
never establish coverage or raise confidence; capsule facts remain
datasets-owned; an optional production slice binds its `manifest_cid` or
canonical payload CID.

An interface is present only when exact AST inspection proves its module-level
definition or assignment and, when public, its exact package export. A renamed
functional interface such as `ContextPacker` must have an explicit, complete,
versioned compatibility map covering all required operations and semantic
constraints; a partial symbol/name match never qualifies. The datasets owner
boards are `ipfs_datasets_py/docs/architecture/incremental_semantic_index.todo.md`
and `ipfs_datasets_py/docs/architecture/semantic_state_contract.todo.md`.
Every task block must have exactly one recognized status.

Focused tests are not admitted through a new observer-owned receipt format.
The narrow current-test path is the existing `VerificationIdentityCompiler`
and `VerificationProcessRunner`. Deterministically bind the inner logical
`(sealed_python, '-m', 'pytest', '-q', *selectors)` command, actual isolated
Python/pytest version-probe bytes, runtime command text/argv, selectors,
toolchain, locks and configuration inputs, then cross-check it as the exact
`--` suffix of the outer Bubblewrap argv. Run exactly one actual same-process
`VerificationCommand` over that outer argv. Only after the run, compile the
exact TEST key whose selector is the same outer Bubblewrap argv, whose resolved
tool and `selector_argv[0]` are exact bwrap, whose probe is the actual bwrap
version output, and whose adapter is `PROCESS_RUNNER_SCHEMA`. Construct the
matching `DirectExecutionObservation` and `TestReceipt` in the trusted process,
require `TestReceipt.from_dict(receipt.to_record()).to_record() == receipt.to_record()`,
then admit and exact-key lookup through `VerificationReceiptCache` with
production eligibility required. A deserialized
observation or receipt supplied from disk/cache is structural data, not proof
that execution occurred. An injected pytest phase report is categorically
forbidden. Admission requires a present real run result, process-started and
completed disposition, zero exit, `ok` and publication allowed, exact observed
argv/selectors/tool version, no timeout/cancel/unavailable/simulation/replay,
stdout and stderr CIDs, freshness, and identical clean pre/post repository
forests matching the complete source identity. Both retained streams must be
non-truncated and satisfy
`captured_byte_count == byte_count == len(preview.encode("utf-8"))`; rehash the
exact preview bytes to both digest and CID. Existing proof test receipts or
semantic compiled receipts may corroborate this evidence but are never
sufficient without the current direct run. Self-asserted/ignored JSON, stale
evidence, an absent output identity, or a structurally valid receipt without
trusted execution authority is `unverifiable`. The same result applies to
unreadable/malformed inputs and unknown, missing, or duplicate board status.

The observation is versioned despite the repository-wide JSON ignore rule:
`.gitignore` must contain exactly the narrow exception
`!artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`.
Its position after the final applicable ignore rule is proved by parsing that
last rule, exact
`git check-ignore -q --no-index -- artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`
return code 1, and an isolated repository showing exactly the corresponding
`??` path. The unpredictable same-directory publication temp must still match
the final `*.json` ignore rule while not matching the exact target exception,
for example `.prerequisite_observation.<nonce>.json`; the same check-ignore
probe returns 0 for it, recursive porcelain omits it while its owned fd is
open, and S1 remains identical. A nonignored or exception-matching temp fails.
Serialized paths and roots are deterministic canonical repository-relative
values with no host-absolute prefix. It uses explicit two-phase identity rather than an impossible self-reference.
G006 commits the observer, tests and ignore exception and makes that tree clean;
G007 then generates the observation. Its source binding is that clean G006
commit/tree, including recursive gitlinks and matching submodule heads and
excluding only the observation artifact. Committing the artifact creates an
evidence projection; it does not change what the JSON claims to have observed.
Native validation and completion receipts independently bind the clean
post-artifact tree. The full clean/complete S0-to-live-run-to-receipt/cache
pipeline is a terminal-admission branch only. The current and future task
worktree may have incomplete committed recursive gitlink closure; ordinary
`observe` records the exact typed degraded reasons, skips the terminal runner,
compiler, receipt and cache steps, and returns rc0 with a structurally complete
ten-row `terminal:false` observation. It must not initialize omitted submodules,
access the network, or manufacture closure. `require-terminal` instead returns
rc1 and writes nothing unless that terminal branch and all rows succeed.

The complete positive pipeline is tested in a self-contained clean temporary
Git fixture with complete zero/controlled gitlink closure and a concrete runner
spy/result. Its repository, environment and forest identity—and therefore its
key—are distinct from the current checkout. Its receipt may structurally admit
and exact-key lookup through an isolated, nonpersisted cache solely to test the
existing contracts; it is never serialized, injected, or substituted as current
checkout evidence, and creates no production injection seam. Immediately before
publication both modes recompute and compare the entire in-memory observation,
outer `HEAD`/tree, recursive
gitlinks/submodule identities, tracked-content digests, and every configured
evidence input and require stable observation-level `S1 == S0`, including
identical degraded-closure reasons. Publication opens the validated parent by
dirfd, fully writes and fsyncs an unpredictable same-directory `O_EXCL`/
`O_NOFOLLOW` temporary whose basename is ignored by the final `*.json` rule but
not the exact target exception, proves the temp remains absent from recursive
porcelain and `S1 == S0` while its fd is open, publishes by no-overwrite `os.link`,
fsyncs/unlinks/fsyncs, and requires nofollow canonical readback. It refuses an
existing target or symlink, forbids `os.replace` and direct target writes, and
durably cleans both temp and any post-link target on failure. No partial, stale,
raced, new, or replaced
admission artifact is evidence.

G006 also provides a strictly read-only verifier, invoked as
`--mode verify-existing --artifact <path>`. By default it nofollow-opens only
the canonical observation target, strictly decodes `PrerequisiteObservation@1`,
requires byte-exact canonical reserialization and the closed ten-row policy,
rederives every forest/content identity, and requires current `HEAD`/tree to be
the claimed observed source. It performs no runner/compiler/cache call and no
write or temp creation; before/after artifact bytes, stat and repository state
must be identical. G007 precommit validation uses only this default form before
and after a no-output `require-terminal` rc1 probe; it never invokes ordinary
`observe` a second time.

Post-merge review may add `--allow-exact-evidence-projection-child`. That flag
accepts no general descendant: either the clean current commit has one parent
equal to the claimed source and changes only the regular observation blob/mode,
or it is the exact native two-parent supervisor merge whose first parent is the
claim, whose second parent has the claim as its sole parent, whose two relevant
diffs both change only that same regular artifact to identical blob/mode, and
whose merge tree equals the implementation-child tree. Any other parent count,
order, ancestry, diff, mode/type/symlink change, dirt or recursive identity
change fails closed. The projection-child flag is forbidden during the G007
task's precommit validation.

## 3. Why `ipfs_kit_py/core/wal` is representative

The package matches the requested bounded VFS/WAL contract class and exercises
the safety properties the capstone is meant to measure:

- Python 3.12-compatible and bounded to eight modules, approximately 7.4 kLOC.
- Clear typed/schema-oriented records and explicit state transitions.
- Standard-library, intra-package WAL and `core.operation_contracts` dependency
  surface; no network imports were observed.
- Hermetic local filesystem and threading behavior, with limited reflection.
- Four focused runtime-readiness files contain 56 declared tests; 21 broader
  suites importing `core.wal` contain 338 declared tests.
- Real history covers contracts, writing, recovery, VFS integration,
  performance and authoritative recovery.
- The in-flight proof-sealer workspace was observed to name a
  `kit-modern-wal` proof unit. That is useful integration context, but it is
  not released proof evidence; `SHQ-G010`, `SHQ-G023` and `SHQ-G067` must
  admit and then re-create or revalidate the exact current checkpoint.
- WAL semantics naturally exercise invalidation, stale receipts, fencing, CAS,
  interruption recovery, proof reuse, performance and side-effect declarations.

Alternatives were rejected for the initial pilot:

- MCP++ validators are smaller and mature but provide weaker incremental proof
  and recovery coverage.
- Datasets semantic-state code is central to the system under evaluation and
  would confound evaluator independence.
- `core.operation_contracts` is pure but concentrated in one large file with a
  thinner historical task base.

## 4. Authority boundaries

| Repository | Sole authority in this capstone | Explicit exclusions |
|---|---|---|
| `ipfs_datasets_py` | Task/corpus contracts, semantic state inputs, ContextPack construction adapters, classification, splits, expected behavior, semantic outcome comparison and result schemas | No execution, provider selection, durable state, signing or release publication |
| `ipfs_kit_py` | Immutable bytes/CIDs, worktree and benchmark evidence, model/test/proof receipts, qualification manifests, release artifacts, fenced CAS and rollback | No benchmark semantics, routing or acceptance-policy invention |
| `ipfs_accelerate_py` | Experiment orchestration, resource admission, provider-neutral tier dispatch, worktree execution, retries/cancellation, verification scheduling, shadow evaluation, accounting, crash injection and pilot control | No duplicate semantic graph, proof system, capsule, storage authority or provider |
| `Mcp-Plus-Plus` | Shared invocation/receipt/qualification wire schemas, canonical vectors and narrow runtime ports | No new profile, execution policy, storage, provider or semantic authority |

`GovernedCodingAgentRuntime` is dependency-injected and stage-resumable. It
delegates semantic identity to datasets, persistence and CAS to kit, model route
class to `ModelRoutePlanner`, verification to the incremental planner/cache,
assurance to the admitted assurance engine, and sealing to the admitted proof
sealer. It does not parse a competing graph, independently select tests, approve
its own patches, or become a second agent framework.

`SHQ-G038` supplies the only qualification-facing ContextPack port. It adapts
the admitted versioned `IncrementalSemanticIndex`,
`SemanticCapsuleCompiler` and `ContextPacker`/`pack_context` APIs, binds their
source/schema/policy roots and reports sufficiency, expansion and fallback. It
does not create a second capsule or semantic-state representation. Every one of
the ten prerequisite systems must produce either an invocation receipt or a
typed, evidence-bound applicability decision. In particular, the non-GUI WAL
target will normally receive a `VerifiedGuiOptimizer: not_applicable` receipt;
silently omitting that integration or hard-coding an unbound exception is not
allowed.

## 5. Fail-closed stage gates

```text
observe prerequisites
        ↓
external release admission (all ten systems)
        ↓
exact inventory → focused/import checks → WAL green/proof/environment freeze
        ↓
parallel contracts, corpus, immutable evidence and provider-neutral ports
        ↓
runtime → experiment plan → configurations A–E
        ↓
analysis, crash-injection/pilot controllers, decision and CI
        ↓
implement and test the non-self-referential release-candidate freeze operation
        ↓
commit implementation → kit-persisted source/environment/proof freeze
        ↓
detached-root development/calibration → external policy freeze → held-out A–E
        ↓
analysis → gated longitudinal pilot → report → conditional signed release
```

Rules:

1. `SHQ-G010` uses external completion authority. Local task receipts cannot
   satisfy it. Its descendants are not projected while it is open.
2. Baseline failure produces a Level 0 diagnostic result. It does not authorize
   a repair of a prerequisite or the target.
3. Benchmark patch rejection, insufficiency, escalation or gate failure is a
   terminal experimental outcome, not an infrastructure retry.
4. Only transient setup, provider transport, resource-admission or process
   failures retry, with an exact trigger and bounded budget.
5. An initial-gate failure makes the longitudinal pilot `not_eligible`; it does
   not leave the overall evidence program in an infinite blocked state.
6. A complete, current and reproducible run may publish a signed negative,
   research or alpha qualification even when quality/cost targets miss or the
   pilot is correctly `not_eligible`; those are evidence-backed outcomes, not
   partial execution. Missing, stale, simulated, unverified or incomplete
   required evidence is a partial failure and permits only an unsigned
   diagnostic report.
7. `SHQ-G072` is a second external gate: an authenticated operator freezes the
   preregistered policy after calibration and before held-out access. Model
   workers cannot create or amend that protected artifact.

Throughout `SHQ-G070`, kit remains the evidence authority: tasks write
authoritative immutable bytes, receipts and CAS roots only through the admitted
kit ports. Human-readable files and JSON under `docs/` or `artifacts/` are
CID-verified projections on the evidence branch, never a second store and never
part of the already frozen executable-source identity.

## 6. Goal, subgoal and task projection

The objective heap defines 41 locally governed work goals, including the
blocked review-only G005A source anchor, and two external gates. The objective
daemon generates their task IDs in a
deterministic scan and assigns content IDs after checking repository evidence.

| Goal | Planned work item | Owner | Depends on |
|---|---|---|---|
| `SHQ-G005A` | Reviewed v11 bwrap→bubblewrap compatibility source anchor; outside v12 projection | accelerate | — |
| `SHQ-G006A` | Exact catalog/path/API/board/recursive-forest core and nonterminal rows | accelerate | reviewed clean tracked source precondition |
| `SHQ-G006B` | Isolated runner/compiler/direct-observation/receipt-cache terminal chain | accelerate | G006A |
| `SHQ-G006` | Exact ignore exception, durable publisher and complete integration matrix | accelerate | G006B |
| `SHQ-G007` | Clean post-merge current-fact observation snapshot | accelerate | G006 |
| `SHQ-G010` | External terminal release admission | operator/upstream owners | G007 |
| `SHQ-G021` | Exact revision/version/schema/route/proof inventory | accelerate | external gate |
| `SHQ-G022` | Focused tests and import/no-network/no-install probes | accelerate | G021 |
| `SHQ-G023` | WAL green check, full proof checkpoint, environment freeze | accelerate + kit evidence | G022 |
| `SHQ-G031` | Shared MCP++ schemas and canonical vectors | MCP++ | G023 |
| `SHQ-G032` | Corpus/task/split/result contracts | datasets | G031 |
| `SHQ-G033` | Historical replay builder and history firewall | datasets | G023, G032 |
| `SHQ-G034` | Controlled synthetic factory | datasets | G023, G032 |
| `SHQ-G035` | Assurance-engine task adapter | datasets | G023, G032 |
| `SHQ-G036` | Independent semantic/hidden evaluator | datasets | G032 |
| `SHQ-G037` | Build, stratify, split and seal ≥50 tasks through the kit artifact port | datasets | G033–G036, G041 |
| `SHQ-G038` | Bind the admitted semantic state/capsule/ContextPack implementation behind a datasets port | datasets | G032 |
| `SHQ-G041` | Immutable artifact/CID port | kit | G023 |
| `SHQ-G042` | Model/test/proof/task/manifest receipts | kit | G031, G041 |
| `SHQ-G043` | Generation/fence CAS and ambiguous recovery | kit | G042 |
| `SHQ-G044` | Operator-signing port, manifest creation, verification and rollback | kit | G042, G043 |
| `SHQ-G051` | Provider-neutral tier runner, context authorization and accounting | accelerate | G031, G042 |
| `SHQ-G052` | `GovernedCodingAgentRuntime` canonical lifecycle and per-system applicability receipts | accelerate | G036, G038, G043, G051 |
| `SHQ-G053` | `SelfHostingQualificationHarness` experiment plan | accelerate | G037, G052 |
| `SHQ-G054` | Configurations A and B | accelerate | G053 |
| `SHQ-G055` | Configuration C | accelerate | G053 |
| `SHQ-G056` | Configuration D | accelerate | G053 |
| `SHQ-G057` | Configuration E | accelerate | G044, G053 |
| `SHQ-G058` | Required CLI and safe resume/status | accelerate | G044, G054–G057, G062, G064, G065 |
| `SHQ-G061` | Cross-arm comparison and noninferiority | accelerate + datasets evaluator | G036, G054–G057 |
| `SHQ-G062` | Economics and substitution matrix | accelerate | G061 |
| `SHQ-G063` | Implement and fixture-test twelve-boundary crash/recovery injection | accelerate + kit | G043, G052, G057 |
| `SHQ-G064` | Implement and fixture-test bounded longitudinal pilot controller | accelerate | G057, G061, G063 |
| `SHQ-G065` | Qualification decision and typed projection into the kit-owned manifest | accelerate | G044, G061–G064 |
| `SHQ-G066` | Fail-closed CI and current release verifier | accelerate | G058, G065 |
| `SHQ-G068` | Preregistration proposal/freeze verification and complete metric-schema validation | accelerate | G032, G062–G065 |
| `SHQ-G067` | Implement/test the non-self-referential release-candidate freeze operation | accelerate + kit port | G037, G066, G068 |
| `SHQ-G071` | Persist actual release-candidate freeze, then run detached-root development/calibration | harness + kit authority | G067 |
| `SHQ-G072` | Externally preregister/freeze margin, policies, prices and seeds | operator | G071 |
| `SHQ-G073` | Held-out A–E execution | harness | G072 |
| `SHQ-G074` | Held-out/assurance/economic analysis and live frozen-tree crash matrix | harness | G062, G063, G073 |
| `SHQ-G075` | Run or truthfully decline longitudinal pilot | harness | G064, G074 |
| `SHQ-G076` | Final report, decision and conditional signed release | harness + kit | G065, G066, G074, G075 |

### Parallel waves

```text
W0a  G005A
W0b  G006
W0c  G007
WG   G010 external admission
W1   G021 → G022 → G023
W2   G031 | G041
W3   G032 | G042
W4   serialize datasets bundle [G033, G034, G035, G036, G038] | G043 | G051
W5   G037 | G044
W6   G052
W7   G053
W8   G054 | G055 | G056 | G057
W9   G061 | G063
W10  G062 | G064
W11  G065 → (G058 | G068) → G066 → G067 after every implementation task lands
W12  commit source → G071 actual freeze + dev/cal → G072 external gate → G073 → G074 → G075 → G076
```

All datasets tasks `SHQ-G032` through `SHQ-G038` share
`datasets/self-hosting/corpus`. They serialize in that common bundle even when
their dependency edges would otherwise permit concurrency, because they share
package initializers, contracts, fixtures and one submodule gitlink. Kit and
accelerate work with disjoint authoritative paths may proceed beside that lane.
Initial supervisor concurrency is deliberately capped at one lane while other
prerequisite supervisors are active; it can rise to two after provider, CPU,
proof-solver and merge-path telemetry are healthy.

## 7. Baseline protocol

Before integration code or corpus construction:

1. Resolve and record clean exact commits for all four repositories.
2. Bind every prerequisite API, version and compatibility adapter.
3. Run every owning focused test selector at those commits.
4. Inventory package, schema, CID/canonicalization, model-route, proof, selector
   and seal versions plus limitations.
5. Run imports in a controlled subprocess with package installation, socket
   connection, package-manager subprocesses and environment mutation fenced.
6. Reject simulated success and historical-only receipts.
7. Run the focused and broader declared WAL checks.
8. Create the full `kit-modern-wal` proof checkpoint.
9. Bind dependency locks, SBOM, container digest, toolchain, environment, model
   configuration, price configuration and random seeds.
10. Freeze the environment CID before any task executes.

Any failure stops downstream projection and yields exact owner/action evidence.

After all implementation goals land, `SHQ-G067` implements and fixture-tests a
non-self-referential freeze operation. It does not execute that operation or
write release-candidate evidence while its own source is still changing. Once
the complete implementation is committed, `SHQ-G071` invokes that operation as
its first effect. Kit ports persist authoritative source, environment and proof
bytes/CIDs for the clean executable commit and recursive gitlinks, rerun focused
checks, and regenerate locks, SBOM, container, toolchain and environment roots.
If WAL changed, a new full checkpoint is mandatory; otherwise the receipt proves
byte identity and revalidates the original checkpoint.

Every development, calibration, held-out, crash and pilot execution uses a
detached worktree at that frozen executable root. Repository-visible freeze and
result JSON files are only CID-verified projections on a distinct evidence
branch. Committing those projections never changes the qualified source root,
and no source identity can include its own subsequently generated manifest.

## 8. Corpus design and data separation

The initial corpus floor is 50 tasks. The planned minimum allocation is:

| Source | Initial floor | Coverage |
|---|---:|---|
| Historical replay | 10 | All qualifying direct WAL commits plus independently scoped contemporaneous requirements/issues |
| Controlled synthetic | 24 | All 18 required classes, with additional risk/cone variants |
| Adversarial assurance | 16 | All 13 required assurance classes, with additional high-risk variants |

If fewer than ten genuinely independent historical tasks exist, the corpus uses
all qualifying history and increases synthetic/adversarial counts; it does not
fabricate history or treat dependent subtasks as independent observations.

For exactly 50 tasks, the seeded stratified split is planned as 20 development,
10 calibration and 20 held-out tasks. Larger corpora use approximately 40/20/40
percent while preserving source, task type, risk, dependency-cone size, expected
context, dynamic complexity, required model tier and verification strata.
Closely related task families stay within one split to limit leakage.

Every task contains:

- allowed files and symbols;
- maximum changed files and lines;
- prohibited directories;
- expected interface and effect class;
- required static, unit, integration, hidden, performance and proof checks;
- required assurance and human-review level;
- immutable visible requirement and evaluator-only expected behavior;
- parent source revision and environment root;
- model-visible ContextPack policy and hidden-history firewall.

Historical worktrees are object-level firewalled, not merely checked out at the
parent revision. The proposing process receives an isolated object store or
export containing only the parent-reachable history and explicitly required
source objects. It has no shared alternates, remotes, later refs, reflogs,
replace refs, commit-graph reachability, future loose/packed objects or build
artifacts; `cat-file --batch-all-objects`, unreachable-object enumeration and
equivalent Git plumbing are tested as denied. Public issue links, provider
browsing/tools and outbound network retrieval are also disabled. A separate
evaluator identity and store retain the real future patch and hidden behavior
evidence, and none of their object IDs or metadata enter the task view,
ContextPack or model worktree. Acceptance compares behavior, not text.

## 9. Five comparable configurations

| Arm | Context/state | Model policy | Verification/assurance |
|---|---|---|---|
| A | Ordinary repository retrieval; no capsules | Frontier only | Full required suite; no incremental proof reuse |
| B | A plus persistent task state and ordinary lexical/semantic retrieval | Frontier only | Normal verification; no capsule substitution |
| C | Admitted semantic capsules and ContextPack construction | Frontier only | Normal verification; no smaller-tier routing |
| D | C plus model-route planning | Deterministic/small/medium with frontier escalation | Incremental test and proof reuse; no complete E assurance claim |
| E | Full semantic state, compression audit, insufficiency expansion and governed state | All provider-neutral tiers plus controlled human escalation | Incremental verification, assurance sampling, shadow evaluation, proof sealing and signed receipts |

The task set, split, order policy, source/environment roots, evaluator, acceptance
rules and seeds are identical. Configuration-isolation tests reject accidental
capsules, smaller tiers or reuse in earlier arms.

## 10. Canonical runtime lifecycle

`GovernedCodingAgentRuntime.execute_task_configuration` checkpoints and resumes
these mandatory stages:

1. Load immutable task.
2. Verify repository and environment roots.
3. Create isolated disposable worktree.
4. Scan admitted semantic state.
5. Construct invalidation plan.
6. Build ContextPack.
7. Evaluate context sufficiency.
8. Select route capability.
9. Authorize exactly the provider-visible context, redactions, endpoint and
   disabled browsing/tool capabilities, then persist that receipt.
10. Invoke deterministic tool/model/human port.
11. Validate patch scope.
12. Apply patch.
13. Rescan changed state.
14. Execute incremental verification.
15. Expand context or escalate as required.
16. Run broader/full verification according to policy.
17. Run required assurance sampling.
18. Produce and verify incremental seal.
19. Independently accept, reject or require human review.
20. Persist complete receipt and atomically advance state.

Qualification mode rejects any plan that omits a stage. A configuration can
select a stage policy such as “full verification, no reuse,” but cannot silently
remove the stage or convert unavailable/timeout/unknown into success.
The runtime records executable/applicable/not-applicable status for every
admitted component. The WAL-specific `VerifiedGuiOptimizer` decision is thus a
verified lifecycle input even when no GUI optimization runs.

## 11. Required APIs and CLI projection

The implementation exposes the requested API equivalents:

- `create_task_corpus`
- `create_experiment_plan`
- `execute_task_configuration`
- `compare_task_outcomes`
- `evaluate_noninferiority`
- `run_longitudinal_pilot`
- `create_qualification_manifest`
- `determine_qualification_level`
- `verify_qualification_release`

The names do not imply duplicate ownership: datasets implements
`create_task_corpus` and `compare_task_outcomes`; accelerate orchestrates
experiment execution, noninferiority, pilot control and the decision; kit owns
`create_qualification_manifest`, release bytes/CIDs and
`verify_qualification_release`. Accelerate projects typed manifest inputs into
kit rather than rebuilding the manifest authority.

The `self-hosting` command tree is a thin projection with corpus build/inspect,
benchmark plan/run/resume/compare/economics, pilot start/status/stop, qualify,
verify-release and report operations. Machine-readable output is canonical JSON.
No GUI is built.

## 12. Independent accepted-patch gate

A proposing model is never its sole evaluator. The datasets evaluator and
runtime recompute acceptance from independent static analysis, types, selected
and policy-required full tests, proof obligations, mutation/assurance checks,
performance, semantic diff, expected behavior, hidden tests and authenticated
human review.

A patch is accepted only if all ten user-declared conditions pass. Additional
hard rejections include benchmark/policy/key/evidence tampering, test disabling,
unapproved dependencies/network access, hidden-patch access, unrelated interface
changes and simulation substituted for execution.

## 13. Noninferiority and statistical policy

The exact margin is frozen in `SHQ-G072` before held-out access. The planned
initial range is 2–5 percentage points; five points is the conservative default
for a 50-task research corpus.

Primary comparison:

```text
Configuration E accepted-patch rate − Configuration A accepted-patch rate
```

Use paired task outcomes, report the estimate and a two-sided 95% confidence
interval, and declare noninferiority only when the lower bound exceeds the
negative frozen margin. Report per-stratum counts and intervals. Critical
regressions, new security-boundary failures, stale capsule/proof acceptance,
simulated production evidence and selected-test fixture false negatives are
zero-tolerance gates independent of the rate interval.

If the held-out sample cannot establish the margin with adequate precision, the
result is `analysis_inconclusive`; it is not equivalence. A 100+ task full
qualification is preferred before strong routing-policy claims.

## 14. Metrics and economics

The aggregate schema contains every requested metric in these families:

- context/compression, including raw cone, retrieval, packed/expanded tokens,
  percentiles, capsule replacement, fallback, expansion and insufficiency;
- routes, shares, escalations, retries and class-level outcomes;
- patch quality, hidden tests, regressions, static/proof/assurance failures,
  human approval/correction and scope;
- selected/full verification, false negatives, proof reuse/cache, compute,
  seals, stale rejection and recovery;
- semantic compression sufficiency, omission, opacity, staleness, misuse and
  compressed/expanded differences;
- sampled/killed/surviving mutants, vacuity and remediation;
- inference, local compute, verification, proof, shadow, human and failed-attempt
  economics;
- wall/phase latency, throughput, memory, GPU where applicable and cache growth.

Observed cost per accepted patch is:

```text
model inference
+ verification compute
+ proof compute
+ shadow audit
+ estimated human review
+ failed attempt cost
```

All unit prices and compute-rate assumptions are frozen. Replayed outputs are
excluded from live quality and cost. Hypothetical projections are separate for
API-only, local-small-plus-API, enterprise self-hosted, high-context frontier
and moderate-context frontier deployments at 10k, 100k, 500k and one million
annual tasks. They are labeled projections, never observed savings.

The substitution matrix reports task count, acceptance, context, cost,
expansion, escalation, common failures and assurance level for deterministic,
small, medium, frontier and human routes by task class.

## 15. Crash, recovery and longitudinal safety

`SHQ-G063` implements the fault injector and exhaustively fixture-tests all
twelve required boundaries. Those pre-freeze fixtures prove the mechanism but
are not admissible as the qualification's live recovery report. `SHQ-G071`
first invokes the `SHQ-G067` operation to persist the fully integrated
source/environment/proof freeze. Only after `SHQ-G073` finishes held-out
execution does `SHQ-G074` inject one deterministic failure at each boundary in
detached worktrees at that frozen executable root. After restart it must
discover immutable completed artifacts, fence stale workers, avoid known duplicate
billing/effects, preserve unknown outcomes, resume safe stages and exclude
partial tasks from accepted counts.

Likewise, `SHQ-G064` implements and fixture-tests the bounded pilot controller;
it performs no live self-hosting sequence. `SHQ-G075` is the only live pilot
stage, and is eligible only after held-out, assurance and live crash gates pass.
It uses a new disposable branch, 20–50 composable accepted changes from the
sealed longitudinal-eligible set, one or two admitted routes, precondition and
rebase checks before each change, periodic full checkpoints, an immediate full
checkpoint after schema/circuit/key or canonicalization change, mandatory human
review for public APIs, immediate critical-invariant stop and verified
rollback. It never merges to a protected branch or deploys to production. If
fewer than 20 safe composable tasks exist or any gate fails, it emits a terminal
`not_eligible` report without model or repository effects.

Tracked longitudinal state includes semantic and proof-cache growth, capsule
staleness, invalidation fan-out, context/policy drift, verification-chain depth,
compaction and cumulative cost.

## 16. Security and evidence publication

- Disposable worktrees only; no production credentials, customer/legal data or
  arbitrary remote filesystem paths.
- Network disabled by default except explicitly admitted model endpoints.
- Secrets are redacted and only policy-approved source reaches a provider.
- Expected patches, later history and evaluator metadata remain inaccessible.
- Models cannot change qualification policy, trusted keys or their own approval.
- Human approval identities are authenticated.
- Kit exposes an injected `OperatorSigningPort`; no model lane receives raw
  signing authority. The private key lives at
  `$SHQ_RUN/operator/signing.key`, outside every repository/worktree, with mode
  `0600`, and is supplied only to the operator-controlled final signing step.
  The repository contains only the protected admitted public-key file
  `config/self_hosting_qualification_trusted_keys.json`.
- Every source, corpus, split, environment, lock, container, model/price policy,
  schema/proof version, verification key, seed and harness version is bound in
  the manifest.
- The release includes all requested raw/aggregate, noninferiority, economic,
  crash, assurance, pilot, seal, verification, limitation, blocker and rollback
  artifacts.

Qualification level is computed from evidence. This one-package capstone cannot
reach Level 5. Level 4 additionally requires independent security review,
independent reproduction, licensing, deployment isolation and access-control
work outside this supervisor run.

A signed artifact is not synonymous with a positive qualification. A complete
valid run may sign a content-addressed `not qualified`, research or alpha
decision and all of its negative evidence. A missing artifact, incomplete arm,
stale root, simulated substitute, failed verification or unresolved ambiguous
outcome is instead an incomplete run: it receives an unsigned diagnostic and
is never published as a qualification release.

## 17. Agent-supervisor bootstrap

All paths are isolated from existing supervisor programs. The objective heap,
plan, generated todo and trusted policy/key locations are protected outputs.
The initial scan intentionally does not refine the heap, repeat existing work or
submit to a task queue.

```bash
set -euo pipefail
SHQ_REPO=/home/barberb/lift_coding/.worktrees/ipfs-accelerate-self-hosting-qualification
SHQ_DATA=data/agent_supervisor/self_hosting_qualification
SHQ_PROJECTION="$SHQ_DATA/projections/v12"
SHQ_PYTHON=/usr/bin/python3.12
SHQ_ACTIVE_TODO=docs/architecture/self_hosting_qualification.todo.md
SHQ_V1_HISTORY_TODO=docs/architecture/self_hosting_qualification.v1_history.todo.md
SHQ_V2_HISTORY_TODO=docs/architecture/self_hosting_qualification.v2_history.todo.md
SHQ_V3_HISTORY_TODO=docs/architecture/self_hosting_qualification.v3_history.todo.md
SHQ_V4_HISTORY_TODO=docs/architecture/self_hosting_qualification.v4_history.todo.md
SHQ_V5_HISTORY_TODO=docs/architecture/self_hosting_qualification.v5_history.todo.md
SHQ_V6_HISTORY_TODO=docs/architecture/self_hosting_qualification.v6_history.todo.md
SHQ_V7_HISTORY_TODO=docs/architecture/self_hosting_qualification.v7_history.todo.md
SHQ_V8_HISTORY_TODO=docs/architecture/self_hosting_qualification.v8_history.todo.md
SHQ_V9_HISTORY_TODO=docs/architecture/self_hosting_qualification.v9_history.todo.md
SHQ_V10_HISTORY_TODO=docs/architecture/self_hosting_qualification.v10_history.todo.md
SHQ_V11_HISTORY_TODO=docs/architecture/self_hosting_qualification.v11_history.todo.md
SHQ_RUN=/home/barberb/.local/state/ipfs_accelerate_py/self-hosting-qualification-v12
SHQ_FROZEN_V11_HEAD=17e19a8e5db327a18dc9437a8de2be299599ecf2
# Read-only input from the already-live provider monitor. All mutable v12
# coordination, state, worktrees, logs, manifests, metrics, gates and keys use
# SHQ_RUN above; never reopen or alias the retired v1 supervisor namespace.
SHQ_CAPACITY_PATH=/home/barberb/.local/state/ipfs_accelerate_py/self-hosting-qualification-v1/provider-capacity/capacity.json
SHQ_GATE="$SHQ_RUN/operator/objective_completion_gate.json"
SHQ_EXTERNAL_AUTHORITY="$SHQ_RUN/operator/external_completion_authority.json"
SHQ_SIGNING_KEY="$SHQ_RUN/operator/signing.key"
SHQ_IMPLEMENTATION_COMMAND="/usr/local/bin/codex exec --ephemeral --ignore-user-config --strict-config --dangerously-bypass-approvals-and-sandbox --color never -m gpt-5.6-terra -c model_context_window=49152 -c 'model_reasoning_effort=\"high\"' -c agents.max_threads=1 -c agents.max_depth=0 -"
SHQ_PROVIDER_ENV=(
  env
  -u PYTHONOPTIMIZE
  -u IMPLEMENTATION_DAEMON_COMMAND
  -u IPFS_PROOF_REUSE_STATE_ROOT
  -u IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER
  -u IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER
  -u IPFS_ACCELERATE_AGENT_PROVIDER_FALLBACK_POLICY
  -u IPFS_ACCELERATE_AGENT_COPILOT_MODEL
  -u IPFS_ACCELERATE_AGENT_COPILOT_CONTEXT_TIER
  -u IPFS_ACCELERATE_AGENT_COPILOT_EFFORT
  -u IPFS_ACCELERATE_AGENT_COPILOT_MAX_CONTINUES
  -u IPFS_ACCELERATE_AGENT_GROK_BIN
  -u IPFS_ACCELERATE_AGENT_GROK_MODEL
  -u IPFS_ACCELERATE_AGENT_GROK_MAX_TURNS
  -u IPFS_ACCELERATE_AGENT_GOOSE_BIN
  -u IPFS_ACCELERATE_AGENT_GOOSE_MODEL
  -u IPFS_ACCELERATE_AGENT_GOOSE_MAX_TOKENS
  -u IPFS_ACCELERATE_AGENT_GOOSE_MAX_TURNS
  -u GITHUB_TOKEN
  -u GH_TOKEN
  -u COPILOT_GITHUB_TOKEN
  -u GROK_API_KEY
  IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=codex
  IPFS_ACCELERATE_AGENT_CODEX_MODEL=gpt-5.6-terra
  IPFS_ACCELERATE_AGENT_CODEX_CONTEXT_WINDOW=49152
  IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT=high
  IPFS_ACCELERATE_AGENT_CODEX_MAX_THREADS=1
  IPFS_ACCELERATE_AGENT_CODEX_MAX_DEPTH=0
  IPFS_ACCELERATE_AGENT_DISABLE_SUBAGENTS=1
)
SHQ_G006A_RUNTIME_TODO="$SHQ_RUN/state/agent-supervisor-self-hosting-prerequisite-observer-catalog-bounded-v12/state/agent_agent_supervisor_self_hosting_prerequisite_observer_catalog_bounded_v12_runtime.todo.md"
SHQ_G006B_RUNTIME_TODO="$SHQ_RUN/state/agent-supervisor-self-hosting-prerequisite-observer-terminal-chain-bounded-v12/state/agent_agent_supervisor_self_hosting_prerequisite_observer_terminal_chain_bounded_v12_runtime.todo.md"
SHQ_G006_RUNTIME_TODO="$SHQ_RUN/state/agent-supervisor-self-hosting-prerequisite-observer-integration-bounded-v12/state/agent_agent_supervisor_self_hosting_prerequisite_observer_integration_bounded_v12_runtime.todo.md"
SHQ_G007_RUNTIME_TODO="$SHQ_RUN/state/agent-supervisor-self-hosting-prerequisite-observation-snapshot-bounded-v12/state/agent_agent_supervisor_self_hosting_prerequisite_observation_snapshot_bounded_v12_runtime.todo.md"

# Reviewed migrations: retain SHQ-001 in the v1 history board. Move cancelled
# SHQ-002 and its never-launched dependent SHQ-003 into the v2 history board,
# retain every historical canonical task block byte-for-byte and record outcome
# only in its history-board preamble. SHQ-004/005 are prelaunch v3 projections.
# Archive the current SHQ-006/007 canonical blocks byte-for-byte in the v4
# history board: SHQ-006 was rejected/cancelled retryable after independent
# critical review, and dependent SHQ-007 never launched. Archive the current
# SHQ-008/009 canonical blocks byte-for-byte in the v5 history board: neither
# was launched because prelaunch review corrected the direct-runner receipt
# schema and retained-stream binding before submission. Archive SHQ-010/011
# byte-for-byte in the v6 history board: SHQ-010 attempt 1 was rejected after
# materializing authorized git-show output outside its checkout, and dependent
# SHQ-011 never launched. Archive SHQ-012/013 byte-for-byte in the v7 history
# board: SHQ-012 attempt 1 was rejected/cancelled retryable after independent
# contract review, and dependent SHQ-013 was never leased or launched. Archive
# SHQ-014/015/016 byte-for-byte in the v8 history board: SHQ-014 attempt 1 was
# rejected/cancelled retryable and SHQ-015/016 never launched. Archive
# SHQ-017/018/019 byte-for-byte in the v9 history board: SHQ-017 attempt 1 was
# rejected/cancelled retryable and SHQ-018/019 never launched. Archive
# SHQ-020/021/022 byte-for-byte in the v10 history board: SHQ-020 attempt 1 was
# rejected/cancelled retryable for inspecting its injected external checkpoint
# directory, and SHQ-021/022 never launched.
# Archive SHQ-023/024/025 byte-for-byte in the v11 history board: SHQ-023 is
# retained only as the explicit reviewed clean merge/settlement source anchor;
# all four SHQ-024 attempts were cancelled and independently rejected, and
# SHQ-025 was registered but never leased or launched. Every v11 attempt log,
# worktree, runtime/checkpoint/coordination record, receipt, quarantine bundle
# and derived byte is prohibited task input. Apply that reviewed tracked
# migration before this command, leave SHQ_ACTIVE_TODO title-only, and retain
# all twenty-five discovery files so display IDs SHQ-001 through SHQ-025 stay
# reserved. The v12 generation must allocate SHQ-026, SHQ-027, SHQ-028, and SHQ-029,
# in that order, to G006A, G006B, G006, and G007. G005A remains blocked
# but is deliberately outside this scoped projection; no fresh G005A task or
# dependency edge is materialized and no G005A/SHQ-023
# --assume-completed-task-id flag is used.
test -f "$SHQ_REPO/$SHQ_V1_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V2_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V3_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V4_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V5_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V6_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V7_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V8_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V9_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V10_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V11_HISTORY_TODO"
test -x "$SHQ_PYTHON"
test "$("$SHQ_PYTHON" --version 2>&1)" = 'Python 3.12.3'
test "$(git -C "$SHQ_REPO" symbolic-ref --short HEAD)" = \
  agent/self-hosting-qualification-v1
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"
git -C "$SHQ_REPO" submodule foreach --recursive \
  'test -z "$(git status --porcelain=v1 --untracked-files=all)"'
SHQ_V12_MIGRATION_HEAD=$(git -C "$SHQ_REPO" rev-parse --verify 'HEAD^{commit}')
SHQ_V12_MIGRATION_TREE=$(git -C "$SHQ_REPO" rev-parse --verify 'HEAD^{tree}')
test "$SHQ_V12_MIGRATION_HEAD" != "$SHQ_FROZEN_V11_HEAD"
test "$(git -C "$SHQ_REPO" rev-parse --verify 'HEAD^1')" = "$SHQ_FROZEN_V11_HEAD"
git -C "$SHQ_REPO" merge-base --is-ancestor \
  "$SHQ_FROZEN_V11_HEAD" "$SHQ_V12_MIGRATION_HEAD"
( cd "$SHQ_REPO" && "$SHQ_PYTHON" - \
    "$SHQ_FROZEN_V11_HEAD" "$SHQ_V12_MIGRATION_HEAD" "$SHQ_V12_MIGRATION_TREE" <<'PY'
import json
import subprocess
import sys

frozen, head, tree = sys.argv[1:]
expected_paths = [
    "docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md",
    "docs/architecture/self_hosting_qualification.objectives.md",
    "docs/architecture/self_hosting_qualification.todo.md",
    "docs/architecture/self_hosting_qualification.v11_history.todo.md",
    "test/api/test_agent_supervisor_self_hosting_qualification_plan.py",
]
changed = subprocess.run(
    ["git", "diff", "--name-only", "-z", frozen, head],
    check=True,
    capture_output=True,
).stdout.decode("utf-8").rstrip("\0").split("\0")
assert changed == expected_paths, (changed, expected_paths)
assert subprocess.run(
    ["git", "rev-parse", "--verify", f"{head}^{{tree}}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip() == tree
print(json.dumps({"migration_head": head, "migration_tree": tree}, sort_keys=True))
PY
)
! rg -q '^## SHQ-' "$SHQ_REPO/$SHQ_ACTIVE_TODO"
test ! -e "$SHQ_REPO/$SHQ_PROJECTION"
test ! -L "$SHQ_REPO/$SHQ_PROJECTION"
test ! -e "$SHQ_RUN"
test ! -L "$SHQ_RUN"

( cd "$SHQ_REPO" && "$SHQ_PYTHON" - "$SHQ_DATA/discovery" "$SHQ_ACTIVE_TODO" <<'PY'
from pathlib import Path
import sys
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import task_ids_from_todo
from ipfs_accelerate_py.agent_supervisor.task_sources.taskboard_store import task_ids_from_artifact_names

discovery = Path(sys.argv[1])
todo = Path(sys.argv[2])
expected = {f"SHQ-{number:03d}" for number in range(1, 26)}
actual = task_ids_from_artifact_names(discovery, task_prefix="SHQ-")
assert actual == expected, (sorted(actual), sorted(expected))
assert task_ids_from_todo(todo.read_text(encoding="utf-8"), task_prefix="SHQ-") == []
PY
)

( cd "$SHQ_REPO" && "$SHQ_PYTHON" -m ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon \
  --repo-root "$SHQ_REPO" \
  --objective-path docs/architecture/self_hosting_qualification.objectives.md \
  --todo-path "$SHQ_ACTIVE_TODO" \
  --discovery-dir "$SHQ_DATA/discovery" \
  --discovery-output-path "$SHQ_DATA/discovery" \
  --bundle-dir "$SHQ_PROJECTION/bundles" \
  --dataset-dir "$SHQ_PROJECTION/datasets" \
  --graph-path "$SHQ_PROJECTION/objective_graph.json" \
  --plan-evaluation-path "$SHQ_PROJECTION/plan_evaluations.json" \
  --todo-vector-index-path "$SHQ_PROJECTION/bundles/todo_vector_index.json" \
  --task-prefix SHQ- \
  --max-findings 4 \
  --scope-goal-id SHQ-G006A \
  --scope-goal-id SHQ-G006B \
  --scope-goal-id SHQ-G006 \
  --scope-goal-id SHQ-G007 \
  --force-goal-id SHQ-G006A \
  --force-goal-id SHQ-G006B \
  --force-goal-id SHQ-G006 \
  --force-goal-id SHQ-G007 \
  --surplus-findings-per-goal 1 \
  --no-persist-ast-dataset \
  --no-reconcile-goal-completion \
  --no-generate-bounded-work \
  --scan-exclude-path ipfs_accelerate_py \
  --scan-exclude-path ipfs_datasets_py \
  --scan-exclude-path ipfs_kit_py \
  --scan-exclude-path mcpplusplus \
  --scan-exclude-path docs \
  --scan-exclude-path data \
  --scan-exclude-path artifacts \
  --scan-exclude-path test \
  --scan-exclude-path tests \
  --protected-output-path docs/architecture/self_hosting_qualification.objectives.md \
  --protected-output-path "$SHQ_ACTIVE_TODO" \
  --protected-output-path "$SHQ_V1_HISTORY_TODO" \
  --protected-output-path "$SHQ_V2_HISTORY_TODO" \
  --protected-output-path "$SHQ_V3_HISTORY_TODO" \
  --protected-output-path "$SHQ_V4_HISTORY_TODO" \
  --protected-output-path "$SHQ_V5_HISTORY_TODO" \
  --protected-output-path "$SHQ_V6_HISTORY_TODO" \
  --protected-output-path "$SHQ_V7_HISTORY_TODO" \
  --protected-output-path "$SHQ_V8_HISTORY_TODO" \
  --protected-output-path "$SHQ_V9_HISTORY_TODO" \
  --protected-output-path "$SHQ_V10_HISTORY_TODO" \
  --protected-output-path "$SHQ_V11_HISTORY_TODO" \
  --protected-output-path docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md \
  --protected-output-path artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json \
  --protected-output-path artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy.json \
  --protected-output-path artifacts/agent_supervisor/self_hosting_qualification/hidden_evaluator_manifest.json \
  --protected-output-path config/self_hosting_qualification_policy.json \
  --protected-output-path config/self_hosting_qualification_trusted_keys.json
)

( cd "$SHQ_REPO" && "$SHQ_PYTHON" - "$SHQ_ACTIVE_TODO" <<'PY'
from pathlib import Path
import sys
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_file

tasks = parse_task_file(Path(sys.argv[1]), "SHQ-")
actual = [
    (task.task_id, task.metadata.get("goal id", ""), tuple(task.depends_on), task.metadata.get("bundle", ""))
    for task in tasks
]
expected = [
    ("SHQ-026", "SHQ-G006A", (), "agent-supervisor/self-hosting/prerequisite-observer-catalog-bounded-v12"),
    ("SHQ-027", "SHQ-G006B", ("SHQ-026",), "agent-supervisor/self-hosting/prerequisite-observer-terminal-chain-bounded-v12"),
    ("SHQ-028", "SHQ-G006", ("SHQ-027",), "agent-supervisor/self-hosting/prerequisite-observer-integration-bounded-v12"),
    ("SHQ-029", "SHQ-G007", ("SHQ-028",), "agent-supervisor/self-hosting/prerequisite-observation-snapshot-bounded-v12"),
]
assert actual == expected, (actual, expected)
assert all(task.metadata.get("status") == "todo" for task in tasks)
assert all(task.metadata.get("is schedulable") == "true" for task in tasks)
assert all(task.metadata.get("review only") == "false" for task in tasks)
assert all("SHQ-023" not in task.depends_on for task in tasks)
PY
)
```

The history boards retain `SHQ-001` as an abandoned combined task, `SHQ-002` as
a cancelled unbounded implementation attempt, and `SHQ-003` as its unlaunched
dependent. SHQ-004/005 record a never-launched v3 projection whose snapshot CID
exposed missing retry semantics before scheduling. SHQ-006 is a rejected and
cancelled/retryable v4 observer task after independent critical review found
fail-closed contract gaps; its dependent SHQ-007 never launched. SHQ-008 and
SHQ-009 are never-launched v5 cards retired by a prelaunch correction: G006's
direct process-runner path must bind `PROCESS_RUNNER_SCHEMA` and exact retained
stream bytes, not the pytest adapter schema or unchecked previews. SHQ-010 is a
rejected/cancelled retryable v6 attempt because it redirected the authorized
rescue commit's `git show` output into `/tmp` and read it outside the checkout;
its dependent SHQ-011 never launched. SHQ-012 is a rejected/cancelled retryable
v7 attempt after independent contract review; its clean stop released a null
output with no implementation-finished event, implementation commit, or merge,
and dependent SHQ-013 was never leased or launched. SHQ-014 is a rejected/cancelled retryable v8 attempt whose alias accepted noncanonical prose and whose positive used synthetic executable bytes; SHQ-015 and SHQ-016 never launched. SHQ-017 is a rejected/cancelled retryable v9 attempt because its exact-bwrap branch retained permissive legacy banner parsing; SHQ-018 and SHQ-019 never launched. SHQ-020 is a rejected/cancelled retryable v10 attempt because it inspected the external injected checkpoint directory under generic checkpoint prompt policy before any edit or validation; SHQ-021 and SHQ-022 never launched. SHQ-023 is the reviewed clean v11 merge/settlement source anchor, SHQ-024 attempts 1 through 4 are rejected/cancelled, and SHQ-025 was never leased or launched. Every v4, v5, v6, v7, v8, v9, v10 and v11 canonical task block
is preserved byte-for-byte in its history board, and
only each preamble records disposition. Historical cards are never completion
evidence or scheduler sources. Their discovery records stay
in `$SHQ_DATA/discovery`, reserving `SHQ-001` through `SHQ-025`; the clean active
board therefore receives exactly `SHQ-026`, `SHQ-027`, `SHQ-028`, and `SHQ-029`. New graph, dataset and
bundle projections live under `$SHQ_PROJECTION`; no scheduler may read a v1,
v2, v3, v4, v5, v6, v7, v8, v9, v10, or v11 bundle index.

Review the portable v12 Markdown and JSON projections and add those exact files
with `git add -f`; do not add DuckDB databases, lock files, runtime state or
provider logs. The archived v1 board and both objective-control documents are
exact protected paths for every subsequent implementation lane. Commit the
reviewed portable projection and regenerated active board before entering the
dry-plan block. That clean projection commit is the first
`SHQ_STAGE_HEAD`/`SHQ_STAGE_TREE`; every successor merge supplies the next pair.
The migration commit remains its ancestor, and neither a dirty projection nor
uncommitted generated bytes may be planned or launched.

The `--scan-exclude-path` arguments above are bounded bootstrap-generation
inputs only: they prevent the initial evidence-gap scan from rediscovering the
entire product while it projects `SHQ-G006A`, `SHQ-G006B`, `SHQ-G006`, and `SHQ-G007`. They must not appear in a goal
completion reconciliation. Completion must compute its tree identity over all
source and recursive gitlinks; carrying these exclusions into reconciliation
would create a different, incomplete completion identity.

### Local bounded-v12 prerequisite-observer stages

The reviewed v11 G005A implementation is not reprojected. G005A is explicitly
blocked/review-only because `external_goal_completion_authoritative=false` and
formal objective reconciliation is absent; its sole retained role is a
source-baseline precondition: canonical task CID
`baguqeerag67a4omevn536zn5wbdtzrvpipp7yym7uptusjxe4vroojgx5bea`,
coordination task CID
`baguqeera5o6wzpnwezcacp5oiwycvzk5uhvrvadr7e6m6x3qdzh65ff5nktq`,
attempt/fence/token `3/3/3`, succeeded receipt
`baguqeerayifbixgmh227xewfgwza77itadvtynj5oaavccihrdxh5ftkbuoq`,
output CID
`baguqeerahs3er2kphhbtexrifryshplgoxyzprzgy5bdk2qfdfezrfzh62ma`,
reviewed implementation merge
`0200be041e1c154660ade9c44a552df97b84dec1`, merge tree
`aea528d467450cf6a70efa36d5ab6f34b4947fc7`, follow-up test commit
`bbf8039a67bf2f4dafdd19ef289638d023825e22`, follow-up tree
`00c76524f2f9e1273b89816103a27130a551de85`, and frozen functional
anchor `17e19a8e5db327a18dc9437a8de2be299599ecf2`, tree
`389048a0ee4d39b24dc68289e21a78da9ca1c4c9`. The v12 migration and
projection commits carry those reviewed tracked bytes forward. Each task
starts only from the freshly committed clean v12 launch HEAD/tree recorded by
operator preflight; the implementation agent must not reopen the anchor with
`git show`, a sibling checkout, a ref, or any runtime material. No fresh G005A
task, dependency edge, or G005A/SHQ-023 `--assume-completed-task-id` authority
is emitted.

Every bounded-v11 SHQ-024 attempt 1 through 4 and the SHQ-025 registration is
sealed. Their display IDs, keys and CIDs cannot satisfy a v12 dependency.
Every v11 implementation log, worktree, task/ref, checkpoint, supervisor,
runtime or coordination record, claim, lease, receipt as bytes, rejected
code/test proposal, rescue/quarantine ref, operator quarantine bundle, scratch,
cache and derived byte is a prohibited non-input. Future tasks must not inspect,
enumerate, restore, copy, seed, cite, validate or retry from any of it. Only the
clean tracked bytes carried into the current v12 checkout are readable input.

The observer is split into four serial, merge-gated tasks with disjoint semantic
responsibilities and fresh bounded-v12 bundle identities:

| Goal | Display ID | Bundle suffix | Executable dependency | Scope |
|---|---|---|---|---|
| `SHQ-G006A` | `SHQ-026` | `prerequisite-observer-catalog-bounded-v12` | — | exact catalog/path/API/board/recursive-forest core and deterministic nonterminal rows |
| `SHQ-G006B` | `SHQ-027` | `prerequisite-observer-terminal-chain-bounded-v12` | `SHQ-026` | isolated live runner→compiler→direct observation→`TestReceipt`/cache chain |
| `SHQ-G006` | `SHQ-028` | `prerequisite-observer-integration-bounded-v12` | `SHQ-027` | exact ignore exception, durable no-clobber publisher and complete integration/negative matrix |
| `SHQ-G007` | `SHQ-029` | `prerequisite-observation-snapshot-bounded-v12` | `SHQ-028` | publish only the current deterministic non-authoritative snapshot |

Priorities `144`, `233`, `377` and `610` force that display mapping.
All four remain children of G005; seriality comes only from the explicit
dependency chain. G006A intentionally has no executable dependency on G005A.
The generated index must contain exactly the four mappings above or the
operator rejects it before commit or launch.

The scoped v12 actor decision is explicit and unchanged across all four stages.
Implementation uses only direct Codex `gpt-5.6-terra`, total context window
49152, high reasoning, one thread, depth zero and disabled subagents through the
exact `SHQ_IMPLEMENTATION_COMMAND`. No fallback provider, wrapper, actor
substitution, prompt expansion or operator-authored implementation seal is
authorized. Independent validation is the declared deterministic focused/full
test and CLI matrix plus operator boundary audit; model output never validates
itself. Each task receives one semantic implementation attempt. The initial
dry plan and start use `--max-task-attempts 1`, so no failed first
attempt can be autonomously reselected. A conditional operator rerun with
`--max-task-attempts 2` permits attempt 2 only after exact changed typed
transient setup, provider transport, resource-admission or process evidence and
an operator pre-invocation gate proving the prior receipt is typed transient
with null output, coordination is inactive/released, no active
claim/lease/process/worktree/ref/lock exists, the attempt counter is exactly 1,
`implementation_attempts_by_cid[<exact canonical task CID>] == 1`,
`selection_idle_reason == all_selectable_ready_tasks_reached_max_task_attempts`,
no `implementation_retry_deferred:*` state or retry-budget-repair receipt
exists, and the fresh v12 launch HEAD/tree/route/protected envelope matches
preflight.
Semantic/contract rejection freezes the task and
requires another reviewed migration; it never triggers a retry, actor switch,
counter reset, auto-reopen, repair-loop continuation or later manual rerun.
The implementation daemon's default three-round repair ceiling is not extra
authority because the
stricter initial one-attempt task ceiling governs; the conditional two-attempt
ceiling is exposed only by the operator-gated rerun described above.

The task-specific checkpoint revocation in every G006A/G006B/G006/G007
Acceptance and Refinement supersedes injected generic checkpoint instructions.
No autonomous model-issued tool may reference, expand, forward, print, list,
stat, resolve, hash, read, write, inspect, test, enumerate, copy, source,
execute, create, modify, redirect, tee, save, cache, checkpoint, materialize or
reread the named checkpoint directory, any alias/descendant, or other external
supervisor/runtime state. Separately declared `/usr/bin/bwrap`, validation
tools, declared gitlinks, runner-private streams, pytest fixture temps and the
Bubblewrap namespace-private `/tmp` remain bounded execution authorities, not
discovery or persisted evidence.

G006A edits only the observer and focused test. G006B consumes only G006A's
clean merged tracked task predecessor and edits only those same two files.
G006 consumes only G006B's clean merged tracked task predecessor and may add
`.gitignore`; it performs the full integrated negative matrix but creates no
real observation artifact. G007 consumes only G006's clean merged tracked task
predecessor and changes only
`artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`.
No later task consumes predecessor runtime, receipts, logs, caches or worktrees.
Every outer supervisor launch uses `--start --once`. The initial one-shot may
claim only G006A/SHQ-026. Because `--once` returns after one reconciliation
cycle while the detached child continues, the operator waits for and fences
that child before another dry plan. After each stage has an exact durable
succeeded member receipt, a clean reviewed merge, no active
claim/lease/process/worktree/ref/lock, and a dry manifest showing the predecessor
execution slice empty/nonclaimable and exactly one newly claimable direct
successor, the operator may run one new one-shot to admit G006B/SHQ-027, then
G006/SHQ-028, then G007/SHQ-029. Successor dry plans pass the exact predecessor
`operator-stage-binding@1` path as `SHQ_STAGE_PREDECESSOR_BINDING`; the helper
revalidates that binding id, the predecessor member/TaskSpec/coordination
identities, the succeeded predecessor receipt, git ancestry from the
predecessor launch HEAD, the `task_dependencies` row, and the empty
predecessor execution slice. A dash is valid only for G006A/SHQ-026.
Those later invocations are successor
admissions, not retries of the completed predecessor, and each successor resets
to `--max-task-attempts 1` even if its predecessor needed an authorized
attempt-2 rerun. Every normal cycle must report `started_count == 1` and
`launched_task_cids` equal to the singleton exact Profile-G coordination CID
for its expected stage while the lane's member map retains the exact member
canonical CID and TaskSpec CID; any other launch set fails closed. A repeated invocation
whose next ready canonical task is unchanged is forbidden; the sole exception
is the exact typed-transient gated attempt-2 procedure below. A persistent
outer scheduler and a one-shot while a predecessor is unsettled are forbidden
because board auto-reopen can reset counters.
Do not pause for formal objective reconciliation between task commits. After
all four runtime todos are terminal, all four commits are merged and the target
branch is clean, run the focused suite and no-output CLI probes.
G006 admits focused-test evidence through the existing direct process-runner
path, not through the pytest adapter and not through a new receipt schema. It
must compile the exact test key with `receipt_kind=TEST` and
`PROCESS_RUNNER_SCHEMA`. `build_hermetic_validation_runtime` and
`hermetic_validation_command` must generate the exact pinned Bubblewrap argv
with network namespace isolation, read-only host binding, bounded writable
workspace and private `/tmp`; the observer then invokes
`VerificationProcessRunner.run(VerificationCommand)` live in-process over that
exact argv, requires the result schema to equal `PROCESS_RUNNER_SCHEMA`, and
project that exact result into the canonically keyed
`DirectExecutionObservation` and `TestReceipt`. A canonical
`TestReceipt.from_dict(receipt.to_record())` round trip and
`VerificationReceiptCache.admit(..., require_production_eligible=True)` plus
exact-key lookup are required structural checks, but construction, round trip,
and cache admission do not authenticate execution origin. That authority comes
only from the live in-process isolated runner call. A `deny_all` or sandbox
identity label is not enforcement. Missing Bubblewrap, namespace denial,
isolation startup failure, changed isolation argv, or unisolated fallback is
`unverifiable`, never a degraded success in terminal admission. When the
actual S0 has incomplete recursive gitlink closure, ordinary observe instead
records exact typed degraded reasons, skips the terminal execution/receipt
branch, returns rc0, and may publish only a stable structurally complete
ten-row `terminal:false` snapshot; require-terminal returns rc1 and writes
nothing. It never initializes omitted submodules, reaches the network, or
manufactures closure. This host currently rejects both
Bubblewrap and `unshare -n` network namespaces with `Operation not permitted`;
that is an observed qualification limitation, so focused-test rows must remain
unverifiable here unless a later independently verified environment provides
the required existing isolation runtime.

The positive terminal pipeline is exercised in a self-contained clean
temporary Git fixture with complete zero/controlled gitlink closure and one
concrete runner spy/result. That fixture has a distinct repository,
environment, forest, and compiled key. Its receipt may structurally admit and
exact-key lookup in an isolated nonpersisted cache to test the existing
contracts, but it is never serialized or accepted as current-checkout evidence
and does not add a production injection seam. Both real and degraded branches
recapture the observation manifest and require `S1 == S0`, including identical
degraded reasons, before any serialization.

G007's sole mutating action is exactly
`python3 scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py --repo-root . --mode observe --quiet`;
its default canonical target must equal the sole declared G007 output. The same
publication proof applies in G007: its unpredictable temp basename
matches the final `*.json` ignore rule but not the exact target exception (for
example `.prerequisite_observation.<nonce>.json`). Last-rule parsing and exact
`git check-ignore -q --no-index --` results prove only the target is unignored;
recursive porcelain and S1 remain unchanged while the owned temp fd is open.
Positive tests exercise that identity-preserving temp, and negative tests reject
a nonignored or target-exception-matching temp rather than broadly excluding
untracked files.

Before structural projection, compare the live result's executable, cwd,
environment, sandbox, network policy, timeout, disposition, command argv,
process/lease identity, and stream fields to the exact command, compiled key,
and observed process. The observation's stdout/stderr CIDs must equal the live
result and both must be named among its artifacts; otherwise the projection is
unverifiable. This explicit check prevents the narrower structural observation
record from hiding a mismatched live-run field that it does not serialize.

`VerificationStreamArtifact` exposes only a preview at this boundary. Each
stdout and stderr artifact must therefore be non-truncated and small enough
that `captured_byte_count == byte_count == len(preview.encode("utf-8"))`, and
G006 must rehash those exact preview bytes to both the declared digest and CID.
The current runner discards its temporary capture bytes after returning, so it
must not claim those bytes were persisted or use a hypothetical artifact-port
alternative. Truncation, incomplete byte representation, invalid or
non-round-trippable UTF-8, counter mismatch, digest mismatch, CID mismatch, or a
non-round-trippable preview makes the prerequisite row unverifiable. Existing
`TestPassReceipt`/`SignedTestPassReceiptV2` proof evidence and semantic
`CompiledReceipt` evidence are optional corroboration and insufficient alone.
Neither the structural test receipt nor the whole observation snapshot is
proof, completion, or release authority.

**Current execution boundary:** stop the bootstrap run after those probes and
the immutable G007 snapshot are complete. Do not execute the reconciliation
commands below in this qualification revision. The scoped reconciler checks
tracked bundle shards and paired successful merge events, but it has not yet
been independently qualified against the authoritative Profile-G TaskReceipt,
coordination lease, fencing token, and state-database lineage. Until that
narrow authority binding is implemented, reviewed, and named here by an exact
commit, G006A, G006B, G006 and G007 remain implementation evidence rather than formally
completed goals. This limitation is off the bootstrap path because generation
uses `--no-reconcile-goal-completion`, and it cannot open external gate G010.

The following blocks are retained as a **non-executable future protocol**. Once
the missing authority binding is qualified, create current tree-bound local
gates and formally reconcile G006A, then G006B, then G006, then G007, without
external receipts or scan exclusions. The command fragments below cover only
the latter two historical reconciliation shapes and are deliberately
incomplete; they must not be enabled until analogous exact G006A/G006B
member-receipt/state-lineage transitions are added and independently qualified:

```bash
( cd "$SHQ_REPO" &&
  test -z "$(git status --porcelain=v1 --untracked-files=all)" &&
  git submodule status --recursive &&
  "$SHQ_PYTHON" -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py &&
  "$SHQ_PYTHON" scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py \
    --repo-root . --mode verify-existing \
    --artifact artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json \
    --allow-exact-evidence-projection-child --quiet &&
  { "$SHQ_PYTHON" scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py \
      --repo-root . --mode require-terminal --quiet && exit 99 || test "$?" -eq 1; } &&
  "$SHQ_PYTHON" scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py \
    --repo-root . --mode verify-existing \
    --artifact artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json \
    --allow-exact-evidence-projection-child --quiet &&
  test -f artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json
)
```

```bash
SHQ_RECONCILE_G006=(
  "$SHQ_PYTHON" -m ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon
  --repo-root "$SHQ_REPO"
  --objective-path docs/architecture/self_hosting_qualification.objectives.md
  --todo-path "$SHQ_ACTIVE_TODO"
  --discovery-dir "$SHQ_DATA/discovery"
  --discovery-output-path "$SHQ_DATA/discovery"
  --bundle-dir "$SHQ_PROJECTION/bundles"
  --dataset-dir "$SHQ_PROJECTION/datasets"
  --graph-path "$SHQ_PROJECTION/objective_graph.json"
  --plan-evaluation-path "$SHQ_PROJECTION/plan_evaluations.json"
  --todo-vector-index-path "$SHQ_PROJECTION/bundles/todo_vector_index.json"
  --task-prefix SHQ-
  --max-findings 96
  --scope-goal-id SHQ-G006
  --objective-goal-completion-scope-goal-id SHQ-G006
  --surplus-findings-per-goal 1
  --no-persist-ast-dataset
  --no-generate-bounded-work
  --no-todo-vector-index
  --objective-goal-completion-reconciliation-only
  --objective-goal-completion-board-scope explicit
  --objective-goal-completion-todo-board "$SHQ_G006_RUNTIME_TODO::## SHQ-"
  --objective-goal-completion-member-receipt-state-root "${SHQ_G006_RUNTIME_TODO%/*}"
  --objective-goal-completion-bundle-index-path "$SHQ_PROJECTION/bundles/index.json"
  --objective-goal-completion-gate-path "$SHQ_GATE"
  --protected-output-path docs/architecture/self_hosting_qualification.objectives.md
  --protected-output-path "$SHQ_ACTIVE_TODO"
  --protected-output-path "$SHQ_V1_HISTORY_TODO"
  --protected-output-path "$SHQ_V2_HISTORY_TODO"
  --protected-output-path "$SHQ_V3_HISTORY_TODO"
  --protected-output-path "$SHQ_V4_HISTORY_TODO"
  --protected-output-path "$SHQ_V5_HISTORY_TODO"
  --protected-output-path "$SHQ_V6_HISTORY_TODO"
  --protected-output-path "$SHQ_V7_HISTORY_TODO"
  --protected-output-path "$SHQ_V8_HISTORY_TODO"
  --protected-output-path "$SHQ_V9_HISTORY_TODO"
  --protected-output-path "$SHQ_V10_HISTORY_TODO"
  --protected-output-path "$SHQ_V11_HISTORY_TODO"
  --protected-output-path docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md
)

# FUTURE PROTOCOL ONLY; do not invoke in the current qualification revision.
( cd "$SHQ_REPO" && "${SHQ_RECONCILE_G006[@]}" )  # active -> provisionally_complete
git -C "$SHQ_REPO" add \
  docs/architecture/self_hosting_qualification.objectives.md
git -C "$SHQ_REPO" commit -m 'chore: provisionally complete prerequisite observer'
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"

# Independently refresh $SHQ_GATE against this commit and its parent ledger.
( cd "$SHQ_REPO" && "${SHQ_RECONCILE_G006[@]}" )  # provisional -> verified_complete
git -C "$SHQ_REPO" add \
  docs/architecture/self_hosting_qualification.objectives.md
git -C "$SHQ_REPO" commit -m 'chore: verify prerequisite observer completion'
```

The already-merged G007 task started at the clean merged G006 `HEAD` and changed
only the observation JSON, whose exact
`.gitignore` exception makes it visible. The JSON binds that pre-observation
G006 commit/tree, every recursive gitlink and matching submodule `HEAD`, and
excludes only its own artifact path. Its commit is an evidence projection, not
the source identity claimed by the JSON. Independently verify `require-terminal`
without `--output`; tests prove a failing terminal check cannot create or
replace output.

After G006 reaches verified completion, refresh the local gate against that
commit and its parent ledger, then perform the same two-transition protocol
for the already-merged G007 task:

```bash
SHQ_RECONCILE_G007=(
  "$SHQ_PYTHON" -m ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon
  --repo-root "$SHQ_REPO"
  --objective-path docs/architecture/self_hosting_qualification.objectives.md
  --todo-path "$SHQ_ACTIVE_TODO"
  --discovery-dir "$SHQ_DATA/discovery"
  --discovery-output-path "$SHQ_DATA/discovery"
  --bundle-dir "$SHQ_PROJECTION/bundles"
  --dataset-dir "$SHQ_PROJECTION/datasets"
  --graph-path "$SHQ_PROJECTION/objective_graph.json"
  --plan-evaluation-path "$SHQ_PROJECTION/plan_evaluations.json"
  --todo-vector-index-path "$SHQ_PROJECTION/bundles/todo_vector_index.json"
  --task-prefix SHQ-
  --max-findings 96
  --scope-goal-id SHQ-G007
  --objective-goal-completion-scope-goal-id SHQ-G007
  --surplus-findings-per-goal 1
  --no-persist-ast-dataset
  --no-generate-bounded-work
  --no-todo-vector-index
  --objective-goal-completion-reconciliation-only
  --objective-goal-completion-board-scope explicit
  --objective-goal-completion-todo-board "$SHQ_G007_RUNTIME_TODO::## SHQ-"
  --objective-goal-completion-member-receipt-state-root "${SHQ_G007_RUNTIME_TODO%/*}"
  --objective-goal-completion-bundle-index-path "$SHQ_PROJECTION/bundles/index.json"
  --objective-goal-completion-gate-path "$SHQ_GATE"
  --protected-output-path docs/architecture/self_hosting_qualification.objectives.md
  --protected-output-path "$SHQ_ACTIVE_TODO"
  --protected-output-path "$SHQ_V1_HISTORY_TODO"
  --protected-output-path "$SHQ_V2_HISTORY_TODO"
  --protected-output-path "$SHQ_V3_HISTORY_TODO"
  --protected-output-path "$SHQ_V4_HISTORY_TODO"
  --protected-output-path "$SHQ_V5_HISTORY_TODO"
  --protected-output-path "$SHQ_V6_HISTORY_TODO"
  --protected-output-path "$SHQ_V7_HISTORY_TODO"
  --protected-output-path "$SHQ_V8_HISTORY_TODO"
  --protected-output-path "$SHQ_V9_HISTORY_TODO"
  --protected-output-path "$SHQ_V10_HISTORY_TODO"
  --protected-output-path "$SHQ_V11_HISTORY_TODO"
  --protected-output-path docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md
)

# FUTURE PROTOCOL ONLY; do not invoke in the current qualification revision.
( cd "$SHQ_REPO" && "${SHQ_RECONCILE_G007[@]}" )  # active -> provisionally_complete
git -C "$SHQ_REPO" add \
  docs/architecture/self_hosting_qualification.objectives.md
git -C "$SHQ_REPO" commit -m 'chore: provisionally complete prerequisite snapshot'
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"

# Independently refresh $SHQ_GATE against this commit and its parent ledger.
( cd "$SHQ_REPO" && "${SHQ_RECONCILE_G007[@]}" )  # provisional -> verified_complete
git -C "$SHQ_REPO" add \
  docs/architecture/self_hosting_qualification.objectives.md
git -C "$SHQ_REPO" commit -m 'chore: verify prerequisite snapshot completion'
```

These local gates prove implementation and snapshot completion only. Neither
admits a prerequisite release or can satisfy `SHQ-G010`.

### External `SHQ-G010` admission and two-phase reconciliation

Opening the prerequisite gate is an operator workflow, not an implementation
task. It begins only after `SHQ-G006A`, `SHQ-G006B`, `SHQ-G006`, and
`SHQ-G007` are independently merged, validated and reconciled complete.
First converge the capstone branch and all three gitlinks
to the ten terminal releases, run the terminal observer, review and commit its
admission artifact, and require a completely clean recursive source:

```bash
git -C "$SHQ_REPO" submodule status --recursive
git -C "$SHQ_REPO" diff-index --quiet HEAD --
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"

"$SHQ_PYTHON" "$SHQ_REPO/scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py" \
  --repo-root "$SHQ_REPO" \
  --mode require-terminal \
  --output "$SHQ_REPO/artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json"

git -C "$SHQ_REPO" add -f \
  artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json
git -C "$SHQ_REPO" commit -m 'chore: admit terminal self-hosting prerequisite revisions'
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"
```

An independent producer and validator then create the current, identity-only
external authority at `$SHQ_EXTERNAL_AUTHORITY` and the independent completion
gate input at `$SHQ_GATE`. Both live under the operator-owned run directory,
not in a model worktree. They must bind the exact clean outer commit/tree,
recursive gitlinks, admission artifact CID, run-plan/ledger identities,
different producer and validator identities and a current freshness window.
This external protocol is dormant until the local authority binding above is
qualified and all ten prerequisite releases are independently admitted. Use
both files on every future reconciliation:

```bash
SHQ_RECONCILE_G010=(
  "$SHQ_PYTHON" -m ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon
  --repo-root "$SHQ_REPO"
  --objective-path docs/architecture/self_hosting_qualification.objectives.md
  --todo-path "$SHQ_ACTIVE_TODO"
  --discovery-dir "$SHQ_DATA/discovery"
  --discovery-output-path "$SHQ_DATA/discovery"
  --bundle-dir "$SHQ_PROJECTION/bundles"
  --dataset-dir "$SHQ_PROJECTION/datasets"
  --graph-path "$SHQ_PROJECTION/objective_graph.json"
  --plan-evaluation-path "$SHQ_PROJECTION/plan_evaluations.json"
  --todo-vector-index-path "$SHQ_PROJECTION/bundles/todo_vector_index.json"
  --task-prefix SHQ-
  --max-findings 96
  --scope-goal-id SHQ-G010
  --objective-goal-completion-scope-goal-id SHQ-G010
  --surplus-findings-per-goal 1
  --no-persist-ast-dataset
  --no-generate-bounded-work
  --no-todo-vector-index
  --objective-goal-completion-reconciliation-only
  --objective-goal-completion-board-scope explicit
  --objective-goal-completion-gate-path "$SHQ_GATE"
  --objective-external-completion-receipt-path "$SHQ_EXTERNAL_AUTHORITY"
  --protected-output-path docs/architecture/self_hosting_qualification.objectives.md
  --protected-output-path "$SHQ_ACTIVE_TODO"
  --protected-output-path "$SHQ_V1_HISTORY_TODO"
  --protected-output-path "$SHQ_V2_HISTORY_TODO"
  --protected-output-path "$SHQ_V3_HISTORY_TODO"
  --protected-output-path "$SHQ_V4_HISTORY_TODO"
  --protected-output-path "$SHQ_V5_HISTORY_TODO"
  --protected-output-path "$SHQ_V6_HISTORY_TODO"
  --protected-output-path "$SHQ_V7_HISTORY_TODO"
  --protected-output-path "$SHQ_V8_HISTORY_TODO"
  --protected-output-path "$SHQ_V9_HISTORY_TODO"
  --protected-output-path "$SHQ_V10_HISTORY_TODO"
  --protected-output-path "$SHQ_V11_HISTORY_TODO"
  --protected-output-path docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md
  --protected-output-path artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json
  --protected-output-path artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy.json
  --protected-output-path artifacts/agent_supervisor/self_hosting_qualification/hidden_evaluator_manifest.json
  --protected-output-path config/self_hosting_qualification_policy.json
  --protected-output-path config/self_hosting_qualification_trusted_keys.json
)

( cd "$SHQ_REPO" && "${SHQ_RECONCILE_G010[@]}" )  # transition 1: active -> provisionally_complete
```

Review the exact projection, commit the tracked objective transition, and make
the worktree clean. The first authority cannot be reused because that commit
changes the source identity. The independent producer/validator must replace
both `$SHQ_EXTERNAL_AUTHORITY` and `$SHQ_GATE` with fresh documents bound to the
new commit and parent ledger before the second run:

```bash
git -C "$SHQ_REPO" add \
  docs/architecture/self_hosting_qualification.objectives.md
git -C "$SHQ_REPO" commit -m 'chore: provisionally admit self-hosting prerequisites'
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"

# Independently refresh $SHQ_EXTERNAL_AUTHORITY and $SHQ_GATE here.
( cd "$SHQ_REPO" && "${SHQ_RECONCILE_G010[@]}" )  # transition 2: provisional -> verified_complete

git -C "$SHQ_REPO" add \
  docs/architecture/self_hosting_qualification.objectives.md
git -C "$SHQ_REPO" commit -m 'chore: verify self-hosting prerequisite admission'
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"
```

Refresh the authority and gate once more against that post-verification commit
before removing `--scope-goal-id SHQ-G010` and projecting downstream tasks.
Every later daemon invocation must retain `--surplus-findings-per-goal 1` and
both explicit paths. Omitting the authority, allowing it to expire, or carrying
an authority across a source/ledger change reopens the external goal. At
`SHQ-G072`, repeat the same two-transition/commit/refresh protocol with a
combined current authority that retains `SHQ-G010` and adds the externally
approved frozen-policy receipt; held-out projection starts only after the fresh
post-verification authority proves both external goals. The operator must
review, `git add -f` and commit the protected
`preregistered_policy.json` before constructing the first `SHQ-G072` source
identity, because repository-wide `*.json` ignore rules do not themselves make
an authoritative JSON artifact immutable or versioned.

Dry-plan before starting:

```bash
SHQ_STAGE_GOAL_ID=SHQ-G006A
SHQ_STAGE_TASK_ID=SHQ-026
SHQ_STAGE_PREDECESSOR_BINDING=-
SHQ_STAGE_HEAD=$(git -C "$SHQ_REPO" rev-parse --verify 'HEAD^{commit}')
SHQ_STAGE_TREE=$(git -C "$SHQ_REPO" rev-parse --verify 'HEAD^{tree}')
test "$SHQ_STAGE_HEAD" != "$SHQ_V12_MIGRATION_HEAD"
git -C "$SHQ_REPO" merge-base --is-ancestor \
  "$SHQ_V12_MIGRATION_HEAD" "$SHQ_STAGE_HEAD"
test "$(git -C "$SHQ_REPO" symbolic-ref --short HEAD)" = \
  agent/self-hosting-qualification-v1
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"
git -C "$SHQ_REPO" submodule foreach --recursive \
  'test -z "$(git status --porcelain=v1 --untracked-files=all)"'

# This owner-only record closes the native bundle-manifest HEAD/tree gap only
# as operator detective/preflight evidence. It is never task completion,
# verification, release, proof, successor-launch or external authority.
shq_stage_binding() {
  ( cd "$SHQ_REPO" && "${SHQ_PROVIDER_ENV[@]}" "$SHQ_PYTHON" - "$@" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)

(
    mode,
    repo_text,
    run_text,
    projection_text,
    stage_goal_id,
    stage_task_id,
    expected_head,
    expected_tree,
    migration_head,
    predecessor_binding_text,
    expected_implementation_command,
    expected_max_task_attempts,
    retry_authorization_text,
    binding_text,
) = sys.argv[1:]
assert mode in {"write", "verify"}
assert expected_max_task_attempts in {"1", "2"}
assert sys.flags.optimize == 0
repo = Path(repo_text).resolve()
run_root = Path(run_text).resolve()
projection = (repo / projection_text).resolve()
manifest_path = run_root / "bundle_lanes.json"
bundle_index_path = projection / "bundles/index.json"
operator_dir = run_root / "operator"

protected_paths = [
    "docs/architecture/self_hosting_qualification.objectives.md",
    "docs/architecture/self_hosting_qualification.todo.md",
    "docs/architecture/self_hosting_qualification.v1_history.todo.md",
    "docs/architecture/self_hosting_qualification.v2_history.todo.md",
    "docs/architecture/self_hosting_qualification.v3_history.todo.md",
    "docs/architecture/self_hosting_qualification.v4_history.todo.md",
    "docs/architecture/self_hosting_qualification.v5_history.todo.md",
    "docs/architecture/self_hosting_qualification.v6_history.todo.md",
    "docs/architecture/self_hosting_qualification.v7_history.todo.md",
    "docs/architecture/self_hosting_qualification.v8_history.todo.md",
    "docs/architecture/self_hosting_qualification.v9_history.todo.md",
    "docs/architecture/self_hosting_qualification.v10_history.todo.md",
    "docs/architecture/self_hosting_qualification.v11_history.todo.md",
    "docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md",
    "artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json",
    "artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy.json",
    "artifacts/agent_supervisor/self_hosting_qualification/hidden_evaluator_manifest.json",
    "config/self_hosting_qualification_policy.json",
    "config/self_hosting_qualification_trusted_keys.json",
]
expected_implementation_argv = [
    "/usr/local/bin/codex",
    "exec",
    "--ephemeral",
    "--ignore-user-config",
    "--strict-config",
    "--dangerously-bypass-approvals-and-sandbox",
    "--color",
    "never",
    "-m",
    "gpt-5.6-terra",
    "-c",
    "model_context_window=49152",
    "-c",
    'model_reasoning_effort="high"',
    "-c",
    "agents.max_threads=1",
    "-c",
    "agents.max_depth=0",
    "-",
]
expected_provider_environment = {
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "codex",
    "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
    "IPFS_ACCELERATE_AGENT_CODEX_CONTEXT_WINDOW": "49152",
    "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "high",
    "IPFS_ACCELERATE_AGENT_CODEX_MAX_THREADS": "1",
    "IPFS_ACCELERATE_AGENT_CODEX_MAX_DEPTH": "0",
    "IPFS_ACCELERATE_AGENT_DISABLE_SUBAGENTS": "1",
}
required_unset_environment = [
    "PYTHONOPTIMIZE",
    "IMPLEMENTATION_DAEMON_COMMAND",
    "IPFS_PROOF_REUSE_STATE_ROOT",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER",
    "IPFS_ACCELERATE_AGENT_PROVIDER_FALLBACK_POLICY",
    "IPFS_ACCELERATE_AGENT_COPILOT_MODEL",
    "IPFS_ACCELERATE_AGENT_COPILOT_CONTEXT_TIER",
    "IPFS_ACCELERATE_AGENT_COPILOT_EFFORT",
    "IPFS_ACCELERATE_AGENT_COPILOT_MAX_CONTINUES",
    "IPFS_ACCELERATE_AGENT_GROK_BIN",
    "IPFS_ACCELERATE_AGENT_GROK_MODEL",
    "IPFS_ACCELERATE_AGENT_GROK_MAX_TURNS",
    "IPFS_ACCELERATE_AGENT_GOOSE_BIN",
    "IPFS_ACCELERATE_AGENT_GOOSE_MODEL",
    "IPFS_ACCELERATE_AGENT_GOOSE_MAX_TOKENS",
    "IPFS_ACCELERATE_AGENT_GOOSE_MAX_TURNS",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "COPILOT_GITHUB_TOKEN",
    "GROK_API_KEY",
]
assert shlex.split(expected_implementation_command) == expected_implementation_argv
assert {key: os.environ.get(key) for key in expected_provider_environment} == (
    expected_provider_environment
)
assert all(key not in os.environ for key in required_unset_environment)

def run(*argv: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(argv, cwd=repo, check=False, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError((argv, result.returncode, result.stdout, result.stderr))
    return result

def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")

def file_evidence(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = path.read_bytes()
    parsed = json.loads(raw)
    return raw, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content_cid": content_identity(parsed),
    }

def flag_values(command: list[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(command):
        if token == flag:
            assert index + 1 < len(command), (flag, command)
            values.append(command[index + 1])
    return values

def exact_one(command: list[str], flag: str) -> str:
    values = flag_values(command, flag)
    assert len(values) == 1, (flag, values)
    return values[0]

def ancestor_pids() -> set[int]:
    observed: set[int] = set()
    current = os.getpid()
    while current > 1 and current not in observed:
        observed.add(current)
        fields = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").split()
        current = int(fields[3])
    return observed

def live_scheduler_pids() -> list[int]:
    ignored = ancestor_pids()
    scheduler_tokens = {
        "ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor",
        "ipfs_accelerate_py.agent_supervisor.merge.leased_lane",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
        "implementation_supervisor_entry.py",
        "adversarial_assurance_engine_scheduler.py",
        "multi_supervisor_runner.py",
    }
    roots = (
        str(run_root),
        str(repo.parents[1]),
        "ipfs-accelerate-adversarial-assurance-engine",
        "incremental-proof-sealer",
        "semantic-compression-governor",
    )
    live: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ignored:
            continue
        try:
            argv = [
                item.decode("utf-8", "replace")
                for item in (proc / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not argv or not any(root in token for root in roots for token in argv):
            continue
        if any(
            token in scheduler_tokens or Path(token).name in scheduler_tokens
            for token in argv
        ):
            live.append(int(proc.name))
    return sorted(live)

def common_mutation_locks() -> list[str]:
    raw = run("git", "rev-parse", "--git-common-dir").stdout.decode().strip()
    common = Path(raw)
    if not common.is_absolute():
        common = (repo / common).resolve()
    candidates = [
        common / "implementation-main-merge.lock",
        common / "agent-checkout-mutation.lock",
        common / "index.lock",
        common / "HEAD.lock",
        common / "config.lock",
        common / "packed-refs.lock",
        common / "shallow.lock",
    ]
    candidates.extend((common / "refs").rglob("*.lock"))
    candidates.extend((common / "worktrees").rglob("index.lock"))
    candidates.extend((common / "worktrees").rglob("HEAD.lock"))
    return sorted(str(path) for path in set(candidates) if path.exists() or path.is_symlink())

def service_state(unit: str) -> dict[str, object]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=LoadState,ActiveState,SubState,UnitFileState,MainPID,ConditionResult",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and not result.stderr.strip()
    values = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    assert values == {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "disabled",
        "MainPID": "0",
        "ConditionResult": "no",
    }
    return {"returncode": result.returncode, **values}

def observe(captured_at: str) -> tuple[dict[str, object], bytes]:
    assert run("git", "symbolic-ref", "--short", "HEAD").stdout.decode().strip() == (
        "agent/self-hosting-qualification-v1"
    )
    assert run("git", "status", "--porcelain=v1", "--untracked-files=all").stdout == b""
    run(
        "git",
        "submodule",
        "foreach",
        "--recursive",
        "--quiet",
        'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
    )
    head = run("git", "rev-parse", "--verify", "HEAD^{commit}").stdout.decode().strip()
    tree = run("git", "rev-parse", "--verify", "HEAD^{tree}").stdout.decode().strip()
    assert (head, tree) == (expected_head, expected_tree), ((head, tree), (expected_head, expected_tree))

    recursive_status = run("git", "submodule", "status", "--recursive").stdout
    recursive_lines = recursive_status.decode("utf-8").splitlines()
    assert recursive_lines and all(line and line[0] in {" ", "-"} for line in recursive_lines)
    index_raw, index_evidence = file_evidence(bundle_index_path)
    manifest_raw, manifest_evidence = file_evidence(manifest_path)
    manifest = json.loads(manifest_raw)
    assert manifest.get("started_count") == 0
    claimable = [lane for lane in manifest.get("lanes", []) if lane.get("claimable") is True]
    assert len(claimable) == 1, claimable
    lane = claimable[0]
    assert lane.get("task_ids") == [stage_task_id], lane.get("task_ids")
    expected_by_id = lane.get("expected_task_cids_by_id") or {}
    member_task_cid = str(expected_by_id.get(stage_task_id) or "")
    task_spec_cid = str(lane.get("task_cid") or "")
    queue_payload = lane.get("queue_payload") or {}
    profile_g = queue_payload.get("profile_g") or {}
    coordination_task_cid = str(profile_g.get("canonical_task_cid") or "")
    assert member_task_cid.startswith("baguqeera")
    assert task_spec_cid.startswith("baguqeera")
    assert coordination_task_cid.startswith("baguqeera")
    assert str(profile_g.get("task_cid") or "") == task_spec_cid
    assert str(profile_g.get("task_spec_cid") or "") == task_spec_cid
    assert str(queue_payload.get("canonical_task_cid") or "") == coordination_task_cid

    tasks = parse_task_file(
        repo / "docs/architecture/self_hosting_qualification.todo.md",
        "SHQ-",
    )
    selected = [task for task in tasks if task.task_id == stage_task_id]
    assert len(selected) == 1
    assert selected[0].metadata.get("goal id") == stage_goal_id
    assert selected[0].canonical_task_cid == member_task_cid

    expected_predecessor_by_stage = {
        ("SHQ-G006A", "SHQ-026"): None,
        ("SHQ-G006B", "SHQ-027"): ("SHQ-G006A", "SHQ-026"),
        ("SHQ-G006", "SHQ-028"): ("SHQ-G006B", "SHQ-027"),
        ("SHQ-G007", "SHQ-029"): ("SHQ-G006", "SHQ-028"),
    }
    assert (stage_goal_id, stage_task_id) in expected_predecessor_by_stage
    expected_predecessor = expected_predecessor_by_stage[(stage_goal_id, stage_task_id)]
    predecessor_claimable = [
        item
        for item in manifest.get("lanes", [])
        if item.get("claimable") is True
        and expected_predecessor is not None
        and expected_predecessor[1] in (item.get("task_ids") or [])
    ]
    assert predecessor_claimable == []
    if expected_predecessor is None:
        assert predecessor_binding_text == "-"
        assert tuple(selected[0].depends_on) == ()
        predecessor_lineage: dict[str, object] | None = None
    else:
        assert predecessor_binding_text != "-"
        predecessor_binding_raw_path = Path(predecessor_binding_text)
        assert not predecessor_binding_raw_path.is_symlink()
        predecessor_binding_path = predecessor_binding_raw_path.resolve()
        assert predecessor_binding_path.parent == operator_dir
        assert predecessor_binding_path.is_file() and not predecessor_binding_path.is_symlink()
        predecessor_stat = predecessor_binding_path.stat()
        assert stat.S_IMODE(predecessor_stat.st_mode) == 0o600
        assert predecessor_stat.st_uid == os.getuid()
        predecessor_raw = predecessor_binding_path.read_bytes()
        predecessor_record = json.loads(predecessor_raw)
        assert canonical(predecessor_record) == predecessor_raw
        assert predecessor_record.get("schema") == (
            "ipfs_accelerate_py/agent-supervisor/operator-stage-binding@1"
        )
        predecessor_body = dict(predecessor_record)
        predecessor_binding_id = str(predecessor_body.pop("binding_id"))
        assert content_identity(predecessor_body) == predecessor_binding_id
        predecessor_stage = predecessor_record.get("stage") or {}
        assert predecessor_stage.get("goal_id") == expected_predecessor[0]
        assert predecessor_stage.get("display_task_id") == expected_predecessor[1]
        predecessor_member_cid = str(predecessor_stage.get("canonical_task_cid") or "")
        predecessor_spec_cid = str(predecessor_stage.get("task_spec_cid") or "")
        predecessor_coordination_cid = str(
            predecessor_stage.get("coordination_task_cid") or ""
        )
        assert predecessor_member_cid.startswith("baguqeera")
        assert predecessor_spec_cid.startswith("baguqeera")
        assert predecessor_coordination_cid.startswith("baguqeera")
        assert predecessor_member_cid != member_task_cid
        assert predecessor_spec_cid != task_spec_cid
        assert predecessor_coordination_cid != coordination_task_cid
        assert tuple(selected[0].depends_on) == (expected_predecessor[1],)
        predecessor_target = predecessor_record.get("target") or {}
        predecessor_head = str(predecessor_target.get("head") or "")
        predecessor_tree = str(predecessor_target.get("tree") or "")
        assert re.fullmatch(r"[0-9a-f]{40}", predecessor_head)
        assert re.fullmatch(r"[0-9a-f]{40}", predecessor_tree)
        assert predecessor_head != head
        assert predecessor_tree != tree
        assert run(
            "git", "merge-base", "--is-ancestor", predecessor_head, head, check=False
        ).returncode == 0
        assert run(
            "git", "merge-base", "--is-ancestor", migration_head, predecessor_head,
            check=False,
        ).returncode == 0
        predecessor_lanes = [
            item
            for item in manifest.get("lanes", [])
            if (item.get("expected_task_cids_by_id") or {}).get(expected_predecessor[1])
            == predecessor_member_cid
            or item.get("task_cid") in {predecessor_spec_cid, predecessor_coordination_cid}
        ]
        assert predecessor_lanes
        assert all(item.get("claimable") is not True for item in predecessor_lanes)
        coordination_path = run_root / "state" / "coordination.duckdb"
        assert coordination_path.is_file() and not coordination_path.is_symlink()
        connection = duckdb.connect(str(coordination_path), read_only=True)
        try:
            lease_row = connection.execute(
                "SELECT claim_cid,resolution_cid,attempt,state,release_reason "
                "FROM leases WHERE task_cid=?",
                [predecessor_coordination_cid],
            ).fetchone()
            receipt_rows = connection.execute(
                "SELECT receipt_cid,payload_json FROM receipts "
                "WHERE task_cid=? ORDER BY rowid",
                [predecessor_coordination_cid],
            ).fetchall()
            dependency_rows = connection.execute(
                "SELECT dependency_task_cid FROM task_dependencies WHERE task_cid=?",
                [coordination_task_cid],
            ).fetchall()
        finally:
            connection.close()
        assert lease_row is not None
        claim_cid, resolution_cid, coordination_attempt, lease_state, release_reason = (
            lease_row
        )
        assert lease_state == "completed"
        assert release_reason is None
        assert coordination_attempt >= 1
        assert len(receipt_rows) >= 1
        receipt_cid, receipt_json = receipt_rows[-1]
        receipt = json.loads(receipt_json)
        assert receipt.get("status") == "succeeded"
        assert receipt.get("failure_class") == "none"
        assert receipt.get("output_cid")
        assert receipt.get("task_cid") == predecessor_coordination_cid
        assert receipt.get("claim_cid") == claim_cid
        assert receipt.get("resolution_cid") == resolution_cid
        assert {row[0] for row in dependency_rows} == {predecessor_coordination_cid}
        predecessor_lineage = {
            "binding_path": str(predecessor_binding_path),
            "binding_id": predecessor_binding_id,
            "sha256": hashlib.sha256(predecessor_raw).hexdigest(),
            "content_cid": content_identity(predecessor_record),
            "stage": {
                "goal_id": expected_predecessor[0],
                "display_task_id": expected_predecessor[1],
                "canonical_task_cid": predecessor_member_cid,
                "task_spec_cid": predecessor_spec_cid,
                "coordination_task_cid": predecessor_coordination_cid,
            },
            "target": {
                "head": predecessor_head,
                "tree": predecessor_tree,
            },
            "receipt": {
                "receipt_cid": receipt_cid,
                "claim_cid": claim_cid,
                "resolution_cid": resolution_cid,
                "attempt": coordination_attempt,
                "status": "succeeded",
                "output_cid": receipt.get("output_cid"),
            },
            "dependency_task_cids": [predecessor_coordination_cid],
        }

    command = [str(item) for item in lane.get("command") or []]
    assert flag_values(command, "--implementation-protected-path") == protected_paths
    assert exact_one(command, "--implementation-command") == expected_implementation_command
    assert exact_one(command, "--merge-target-branch") == "agent/self-hosting-qualification-v1"
    assert exact_one(command, "--max-task-attempts") == expected_max_task_attempts
    assert flag_values(command, "--worktree-submodule-path") == [
        "ipfs_datasets_py",
        "ipfs_kit_py",
        "ipfs_accelerate_py/mcplusplus",
    ]
    worktree_root_raw = Path(exact_one(command, "--worktree-root"))
    worktree_root = worktree_root_raw.resolve()
    lane_worktree_raw = Path(str(lane.get("worktree_root") or ""))
    if not lane_worktree_raw.is_absolute():
        lane_worktree_raw = repo / lane_worktree_raw
    assert lane_worktree_raw.resolve() == worktree_root
    assert worktree_root.is_relative_to(run_root / "worktrees")
    state_dir = Path(str(lane.get("state_dir") or ""))
    if not state_dir.is_absolute():
        state_dir = (repo / state_dir).resolve()
    state_prefix = str(lane.get("state_prefix") or "")
    assert state_prefix and state_dir.is_relative_to(run_root)
    state_path = state_dir / f"{state_prefix}_task_state.json"
    retry_authorization: dict[str, object] | None = None
    if expected_max_task_attempts == "1":
        assert retry_authorization_text == "-"
        assert not state_path.exists() and not state_path.is_symlink()
        attempt_policy = {
            "max_task_attempts": 1,
            "expected_prior_attempt_count": 0,
            "expected_next_attempt": 1,
            "expected_repair_round": 0,
            "state_path": str(state_path),
            "state_sha256": None,
            "typed_transient_retry_authorization": None,
        }
    else:
        retry_authorization_raw_path = Path(retry_authorization_text)
        assert not retry_authorization_raw_path.is_symlink()
        retry_authorization_path = retry_authorization_raw_path.resolve()
        assert retry_authorization_path.parent == operator_dir
        assert retry_authorization_path.is_file() and not retry_authorization_path.is_symlink()
        retry_stat = retry_authorization_path.stat()
        assert stat.S_IMODE(retry_stat.st_mode) == 0o600 and retry_stat.st_uid == os.getuid()
        retry_raw = retry_authorization_path.read_bytes()
        retry_authorization = json.loads(retry_raw)
        assert canonical(retry_authorization) == retry_raw
        assert retry_authorization.get("schema") == (
            "ipfs_accelerate_py/agent-supervisor/typed-transient-retry-authorization@1"
        )
        assert retry_authorization.get("stage") == {
            "goal_id": stage_goal_id,
            "display_task_id": stage_task_id,
            "canonical_task_cid": member_task_cid,
            "task_spec_cid": task_spec_cid,
            "coordination_task_cid": coordination_task_cid,
        }
        trigger = retry_authorization.get("trigger") or {}
        assert set(trigger) == {
            "kind",
            "before_evidence",
            "after_evidence",
            "failure_evidence",
            "semantic_or_contract_rejection",
        }
        assert trigger.get("kind") in {"setup", "provider", "resource", "process"}
        assert trigger.get("semantic_or_contract_rejection") is False
        evidence_records: dict[str, dict[str, object]] = {}
        for evidence_name in ("before_evidence", "after_evidence", "failure_evidence"):
            evidence_ref = trigger.get(evidence_name) or {}
            assert set(evidence_ref) == {"path", "sha256", "content_cid"}
            evidence_path = Path(str(evidence_ref["path"])).resolve()
            assert evidence_path.is_relative_to(operator_dir)
            assert evidence_path.is_file() and not evidence_path.is_symlink()
            evidence_stat = evidence_path.stat()
            assert stat.S_IMODE(evidence_stat.st_mode) == 0o600
            assert evidence_stat.st_uid == os.getuid()
            evidence_raw = evidence_path.read_bytes()
            evidence_body = json.loads(evidence_raw)
            assert canonical(evidence_body) == evidence_raw
            assert re.fullmatch(r"[0-9a-f]{64}", str(evidence_ref["sha256"]))
            assert hashlib.sha256(evidence_raw).hexdigest() == evidence_ref["sha256"]
            assert content_identity(evidence_body) == evidence_ref["content_cid"]
            evidence_records[evidence_name] = evidence_body
        assert trigger["before_evidence"]["content_cid"] != trigger["after_evidence"]["content_cid"]
        assert evidence_records["failure_evidence"].get("trigger_kind") == trigger["kind"]
        assert evidence_records["failure_evidence"].get("semantic_or_contract_rejection") is False
        assert retry_authorization.get("authority") == {
            "attempt_2": True,
            "task_completion": False,
            "verification": False,
            "release": False,
            "proof": False,
            "successor_launch": False,
            "external": False,
        }
        assert state_path.is_file() and not state_path.is_symlink()
        state_raw = state_path.read_bytes()
        state = json.loads(state_raw)
        assert state.get("implementation_attempts_by_cid") == {member_task_cid: 1}
        assert state.get("selection_idle_reason") == (
            "all_selectable_ready_tasks_reached_max_task_attempts"
        )
        assert not str(state.get("selection_idle_reason") or "").startswith(
            "implementation_retry_deferred:"
        )
        assert not (state.get("retry_budget_repair_receipts") or {})
        assert state.get("implementation_in_progress") is False
        assert not str(state.get("active_task_id") or "")
        last_worktree_text = str(state.get("last_implementation_worktree_path") or "")
        last_worktree_raw = Path(last_worktree_text) if last_worktree_text else None
        if last_worktree_raw is not None:
            assert not last_worktree_raw.is_symlink()
        last_worktree = last_worktree_raw.resolve() if last_worktree_raw is not None else None
        if last_worktree is not None:
            assert not last_worktree.exists() and not last_worktree.is_symlink()
        last_branch = str(state.get("last_implementation_branch") or "")
        common_raw = run("git", "rev-parse", "--git-common-dir").stdout.decode().strip()
        common_dir = Path(common_raw)
        if not common_dir.is_absolute():
            common_dir = (repo / common_dir).resolve()
        if last_worktree_text:
            assert not any(
                last_worktree_text.encode("utf-8") in path.read_bytes()
                for path in common_dir.rglob("gitdir")
                if path.is_file()
            )
        if last_branch:
            assert run(
                "git", "show-ref", "--verify", "--quiet", f"refs/heads/{last_branch}", check=False
            ).returncode == 1
            branch_needle = f"refs/heads/{last_branch}".encode("utf-8")
            assert not any(
                branch_needle in path.read_bytes()
                for path in common_dir.rglob("packed-refs")
                if path.is_file()
            )
        lane_worktree_root = worktree_root
        assert not any(lane_worktree_root.glob("workspace_*"))
        assert not any((lane_worktree_root / ".pool-state").glob("*.json"))
        coordination_path = run_root / "state" / "coordination.duckdb"
        assert coordination_path.is_file() and not coordination_path.is_symlink()
        connection = duckdb.connect(str(coordination_path), read_only=True)
        try:
            lease_row = connection.execute(
                "SELECT claim_cid,resolution_cid,attempt,state,release_reason "
                "FROM leases WHERE task_cid=?",
                [coordination_task_cid],
            ).fetchone()
            receipt_rows = connection.execute(
                "SELECT receipt_cid,payload_json FROM receipts "
                "WHERE task_cid=? ORDER BY rowid",
                [coordination_task_cid],
            ).fetchall()
        finally:
            connection.close()
        assert lease_row is not None
        claim_cid, resolution_cid, coordination_attempt, lease_state, release_reason = lease_row
        assert lease_state == "released" and coordination_attempt == 1
        assert len(receipt_rows) == 1
        receipt_cid, receipt_json = receipt_rows[0]
        receipt = json.loads(receipt_json)
        prior = retry_authorization.get("prior_coordination") or {}
        assert prior == {
            "task_cid": coordination_task_cid,
            "receipt_cid": receipt_cid,
            "claim_cid": claim_cid,
            "resolution_cid": resolution_cid,
            "attempt": 1,
            "status": receipt.get("status"),
            "failure_class": receipt.get("failure_class"),
            "output_cid": receipt.get("output_cid"),
        }
        assert prior["status"] in {"cancelled", "failed"}
        assert prior["failure_class"] == "retryable" and prior["output_cid"] is None
        assert release_reason == f"receipt:{prior['status']}:retryable"
        attempt_policy = {
            "max_task_attempts": 2,
            "expected_prior_attempt_count": 1,
            "expected_next_attempt": 2,
            "expected_repair_round": 1,
            "state_path": str(state_path),
            "state_sha256": hashlib.sha256(state_raw).hexdigest(),
            "typed_transient_retry_authorization": {
                "path": str(retry_authorization_path),
                "sha256": hashlib.sha256(retry_raw).hexdigest(),
                "content_cid": content_identity(retry_authorization),
            },
            "cleanup": {
                "last_worktree_path": last_worktree_text,
                "last_branch": last_branch,
                "workspace_entries": [],
                "active_pool_records": [],
                "gitdir_registrations": [],
                "managed_branch_refs": [],
            },
        }
    envelope = {
        "implementation_command": expected_implementation_command,
        "lane_command": command,
        "max_task_attempts": int(expected_max_task_attempts),
        "protected_paths": protected_paths,
    }
    envelope_bytes = canonical(envelope)

    stop_path = Path("/home/barberb/.local/lib/aae-supervisor-keep/OPERATOR_STOP")
    stop_stat = stop_path.stat()
    services = {
        "service": service_state("aae-adversarial-assurance-engine.service"),
        "timer": service_state("aae-adversarial-assurance-engine.timer"),
    }
    live = live_scheduler_pids()
    locks = common_mutation_locks()
    assert stat.S_IMODE(stop_stat.st_mode) == 0o600 and stop_stat.st_uid == os.getuid()
    assert live == [] and locks == []

    dry_snapshot = operator_dir / f"dry-manifest-{manifest_evidence['sha256']}.json"
    body: dict[str, object] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/operator-stage-binding@1",
        "version": 1,
        "captured_at": captured_at,
        "stage": {
            "goal_id": stage_goal_id,
            "display_task_id": stage_task_id,
            "canonical_task_cid": member_task_cid,
            "task_spec_cid": task_spec_cid,
            "coordination_task_cid": coordination_task_cid,
        },
        "target": {
            "branch": "agent/self-hosting-qualification-v1",
            "head": head,
            "tree": tree,
        },
        "recursive_gitlink_status": {
            "sha256": hashlib.sha256(recursive_status).hexdigest(),
            "byte_count": len(recursive_status),
            "uninitialized_paths": [
                line[42:]
                for line in recursive_status.decode("utf-8").splitlines()
                if line.startswith("-")
            ],
        },
        "bundle_index": {**index_evidence, "byte_count": len(index_raw)},
        "dry_manifest": {
            **manifest_evidence,
            "byte_count": len(manifest_raw),
            "snapshot_path": str(dry_snapshot),
        },
        "implementation_envelope": {
            "sha256": hashlib.sha256(envelope_bytes).hexdigest(),
            "content_cid": content_identity(envelope),
            "implementation_command": expected_implementation_command,
            "implementation_argv": expected_implementation_argv,
            "lane_command_sha256": hashlib.sha256(canonical(command)).hexdigest(),
            "lane_command_cid": content_identity(command),
            "max_task_attempts": int(expected_max_task_attempts),
            "protected_path_count": len(protected_paths),
            "worktree_root": str(worktree_root),
        },
        "provider_environment": {
            "required": expected_provider_environment,
            "unset": required_unset_environment,
        },
        "attempt_policy": attempt_policy,
        "predecessor_lineage": predecessor_lineage,
        "quiescence": {
            "repository_clean": True,
            "recursive_submodules_clean": True,
            "scheduler_process_pids": live,
            "common_mutation_locks": locks,
            "aae_operator_stop_mode": "0600",
            "aae_units": services,
        },
        "authority": {
            "task_completion": False,
            "verification": False,
            "release": False,
            "proof": False,
            "successor_launch": False,
            "external": False,
        },
    }
    return body, manifest_raw

def persist_no_clobber(path: Path, data: bytes, *, allow_identical: bool = False) -> None:
    if path.exists() or path.is_symlink():
        assert allow_identical and path.is_file() and path.read_bytes() == data
        assert stat.S_IMODE(path.stat().st_mode) == 0o600 and path.stat().st_uid == os.getuid()
        return
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            assert written > 0
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temp, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)

if mode == "write":
    operator_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert operator_dir.is_dir() and not operator_dir.is_symlink()
    os.chmod(operator_dir, 0o700)
    captured_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    body, dry_manifest_raw = observe(captured_at)
    dry_snapshot = Path(str(body["dry_manifest"]["snapshot_path"]))
    persist_no_clobber(dry_snapshot, dry_manifest_raw, allow_identical=True)
    record = {**body, "binding_id": content_identity(body)}
    data = canonical(record)
    binding_path = operator_dir / (
        f"stage-{stage_task_id}-{body['dry_manifest']['sha256'][:16]}.json"
    )
    persist_no_clobber(binding_path, data)
    assert canonical(json.loads(binding_path.read_bytes())) == data
    print(binding_path)
else:
    binding_raw_path = Path(binding_text)
    assert not binding_raw_path.is_symlink()
    binding_path = binding_raw_path.resolve()
    assert binding_path.parent == operator_dir and binding_path.is_file()
    assert stat.S_IMODE(binding_path.stat().st_mode) == 0o600 and binding_path.stat().st_uid == os.getuid()
    raw = binding_path.read_bytes()
    record = json.loads(raw)
    assert canonical(record) == raw
    binding_id = str(record.pop("binding_id"))
    assert content_identity(record) == binding_id
    observed, current_manifest_raw = observe(str(record["captured_at"]))
    assert observed == record
    snapshot = Path(str(record["dry_manifest"]["snapshot_path"]))
    assert snapshot.is_file() and not snapshot.is_symlink()
    assert snapshot.read_bytes() == current_manifest_raw
    assert hashlib.sha256(current_manifest_raw).hexdigest() == record["dry_manifest"]["sha256"]
    print(record["stage"]["coordination_task_cid"])
PY
  )
}

test "$(/usr/local/bin/codex --version)" = 'codex-cli 0.147.0'
test "$(sha256sum /usr/local/lib/node_modules/@openai/codex/bin/codex.js | cut -d' ' -f1)" = \
  134063e133f0b4244fa3b251acf973d4fe4b4aeeacbdc135211bf480f59f1477
test "$(sha256sum /usr/bin/node | cut -d' ' -f1)" = \
  2b0f6efd95c31c5538cc0a9042d5d13b7328cffcfdcc409f2e2ef336c4402086
test -z "${IPFS_PROOF_REUSE_STATE_ROOT:-}"
jq -e '.providers.codex_cli.healthy == true and
       .providers.codex_cli.context_window_tokens == 24576 and
       .providers.codex_cli.quota_remaining > 0 and
       .providers.codex_cli.token_budget_remaining > 0' \
  "$SHQ_CAPACITY_PATH"

SHQ_MAX_TASK_ATTEMPTS=1
SHQ_RETRY_AUTHORIZATION_PATH=-
SHQ_BUNDLE_ARGS=( \
  --bundle-index-path "$SHQ_REPO/$SHQ_PROJECTION/bundles/index.json" \
  --repo-root "$SHQ_REPO" \
  --state-root "$SHQ_RUN/state" \
  --worktree-root "$SHQ_RUN/worktrees" \
  --log-dir "$SHQ_RUN/logs" \
  --manifest-path "$SHQ_RUN/bundle_lanes.json" \
  --metrics-path "$SHQ_RUN/scheduler_metrics.json" \
  --coordination-path "$SHQ_RUN/state/coordination.duckdb" \
  --provider-capacity-path "$SHQ_CAPACITY_PATH" \
  --provider-capacity-max-age-ms 30000 \
  --task-prefix '## SHQ-' \
  --implement \
  --implementation-command "$SHQ_IMPLEMENTATION_COMMAND" \
  --max-lanes 1 \
  --max-task-attempts "$SHQ_MAX_TASK_ATTEMPTS" \
  --poll-interval 5 \
  --check-interval 30 \
  --daemon-interval 45 \
  --stale-seconds 1200 \
  --watchdog-startup-grace-seconds 300 \
  --implementation-timeout 14400 \
  --max-restarts 8 \
  --merge-target-branch agent/self-hosting-qualification-v1 \
  --implementation-protected-path docs/architecture/self_hosting_qualification.objectives.md \
  --implementation-protected-path "$SHQ_ACTIVE_TODO" \
  --implementation-protected-path "$SHQ_V1_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V2_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V3_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V4_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V5_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V6_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V7_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V8_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V9_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V10_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V11_HISTORY_TODO" \
  --implementation-protected-path docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md \
  --implementation-protected-path artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json \
  --implementation-protected-path artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy.json \
  --implementation-protected-path artifacts/agent_supervisor/self_hosting_qualification/hidden_evaluator_manifest.json \
  --implementation-protected-path config/self_hosting_qualification_policy.json \
  --implementation-protected-path config/self_hosting_qualification_trusted_keys.json \
  --worktree-submodule-path ipfs_datasets_py \
  --worktree-submodule-path ipfs_kit_py \
  --worktree-submodule-path ipfs_accelerate_py/mcplusplus \
)

"${SHQ_PROVIDER_ENV[@]}" "$SHQ_PYTHON" -m \
  ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor \
  "${SHQ_BUNDLE_ARGS[@]}"

SHQ_STAGE_BINDING_PATH=$(shq_stage_binding write \
  "$SHQ_REPO" "$SHQ_RUN" "$SHQ_PROJECTION" \
  "$SHQ_STAGE_GOAL_ID" "$SHQ_STAGE_TASK_ID" \
  "$SHQ_STAGE_HEAD" "$SHQ_STAGE_TREE" \
  "$SHQ_V12_MIGRATION_HEAD" "$SHQ_STAGE_PREDECESSOR_BINDING" \
  "$SHQ_IMPLEMENTATION_COMMAND" "$SHQ_MAX_TASK_ATTEMPTS" \
  "$SHQ_RETRY_AUTHORIZATION_PATH" -)
test "$(stat -c '%a' "$SHQ_STAGE_BINDING_PATH")" = 600
SHQ_STAGE_COORDINATION_TASK_CID=$(shq_stage_binding verify \
  "$SHQ_REPO" "$SHQ_RUN" "$SHQ_PROJECTION" \
  "$SHQ_STAGE_GOAL_ID" "$SHQ_STAGE_TASK_ID" \
  "$SHQ_STAGE_HEAD" "$SHQ_STAGE_TREE" \
  "$SHQ_V12_MIGRATION_HEAD" "$SHQ_STAGE_PREDECESSOR_BINDING" \
  "$SHQ_IMPLEMENTATION_COMMAND" "$SHQ_MAX_TASK_ATTEMPTS" \
  "$SHQ_RETRY_AUTHORIZATION_PATH" "$SHQ_STAGE_BINDING_PATH")
```

Protected-path matching is exact, not recursive. The commands therefore name
each governed file instead of relying on a directory such as `trusted_keys` to
protect descendants. `$SHQ_SIGNING_KEY` is deliberately absent from both the
repository and the implementation command; the operator provisions it with
mode `0600` only for the final kit signing port.

`--implement` compiles the exact lane command during this dry plan; the absence
of `--start` guarantees that no process is launched. The lane-affecting timing,
watchdog, implementation-timeout and restart arguments are already in
`SHQ_BUNDLE_ARGS`, so the dry and live lane commands are byte-for-byte equal.
Only `--start --once` changes at admission. Run the bootstrap, projection,
dry-plan, binding, launch and post-start checks in this same strict
`set -euo pipefail` shell. Immediately before the live command, rederive every
binding from the still-clean checkout and unchanged dry manifest; then launch
exactly one reconciliation cycle:

```bash
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"
test "$(git -C "$SHQ_REPO" rev-parse --verify 'HEAD^{commit}')" = \
  "$SHQ_STAGE_HEAD"
test "$(git -C "$SHQ_REPO" rev-parse --verify 'HEAD^{tree}')" = \
  "$SHQ_STAGE_TREE"
SHQ_STAGE_COORDINATION_TASK_CID=$(shq_stage_binding verify \
  "$SHQ_REPO" "$SHQ_RUN" "$SHQ_PROJECTION" \
  "$SHQ_STAGE_GOAL_ID" "$SHQ_STAGE_TASK_ID" \
  "$SHQ_STAGE_HEAD" "$SHQ_STAGE_TREE" \
  "$SHQ_V12_MIGRATION_HEAD" "$SHQ_STAGE_PREDECESSOR_BINDING" \
  "$SHQ_IMPLEMENTATION_COMMAND" "$SHQ_MAX_TASK_ATTEMPTS" \
  "$SHQ_RETRY_AUTHORIZATION_PATH" "$SHQ_STAGE_BINDING_PATH")

"${SHQ_PROVIDER_ENV[@]}" "$SHQ_PYTHON" -m \
  ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor \
  "${SHQ_BUNDLE_ARGS[@]}" --start --once
```

The dynamic manifest's `launched_task_cids` names the Profile-G coordination
identity, not the immutable TaskSpec CID or the member task CID. Verify all
three identities and the native worktree-pool `base_commit` before allowing
the implementation to continue into validation or merge:

```bash
shq_verify_started_stage() {
  ( cd "$SHQ_REPO" && "${SHQ_PROVIDER_ENV[@]}" "$SHQ_PYTHON" - "$@" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

repo = Path(sys.argv[1]).resolve()
run_root = Path(sys.argv[2]).resolve()
binding_raw_path = Path(sys.argv[3])
assert not binding_raw_path.is_symlink()
binding_path = binding_raw_path.resolve()
expected_coordination_cid = sys.argv[4]
assert sys.flags.optimize == 0

def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")

def persist_no_clobber(path: Path, data: bytes) -> None:
    assert not path.exists() and not path.is_symlink()
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            assert written > 0
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temp, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)

def load_regular_json(path: Path) -> tuple[bytes, dict[str, object]]:
    assert path.is_file() and not path.is_symlink()
    raw = path.read_bytes()
    value = json.loads(raw)
    assert isinstance(value, dict)
    return raw, value

def process_identity(pid: int) -> dict[str, object]:
    stat_raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = stat_raw.rsplit(")", 1)[1].split()
    argv = [
        item.decode("utf-8", "replace")
        for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if item
    ]
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "start_time_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(),
        "argv": argv,
    }

def ancestor_pids(pid: int) -> set[int]:
    observed: set[int] = set()
    while pid > 1 and pid not in observed and Path(f"/proc/{pid}/stat").is_file():
        observed.add(pid)
        pid = int(process_identity(pid)["ppid"])
    return observed

binding_raw, binding = load_regular_json(binding_path)
assert canonical(binding) == binding_raw
binding_body = dict(binding)
binding_id = str(binding_body.pop("binding_id"))
assert content_identity(binding_body) == binding_id
stage = binding["stage"]
assert expected_coordination_cid == stage["coordination_task_cid"]

manifest_path = run_root / "bundle_lanes.json"
manifest_raw, manifest = load_regular_json(manifest_path)
assert manifest.get("schema") == "ipfs_accelerate_py.agent_supervisor.dynamic_bundle_scheduler@1"
assert manifest.get("authoritative") is True
assert manifest.get("started_count") == 1
assert manifest.get("running_count") == 1
assert (manifest.get("counts") or {}).get("active") == 1
assert manifest.get("launched_task_cids") == [expected_coordination_cid]
assert len(manifest.get("lanes", [])) == 1
lanes = [
    lane
    for lane in manifest.get("lanes", [])
    if lane.get("task_cid") == expected_coordination_cid
]
assert len(lanes) == 1
lane = lanes[0]
assert lane.get("state") == "running"
lane_pid = int(lane.get("pid") or 0)
assert lane_pid > 1 and Path(f"/proc/{lane_pid}").is_dir()
lane_process = process_identity(lane_pid)
assert "ipfs_accelerate_py.agent_supervisor.merge.leased_lane" in lane_process["argv"]
assert str(run_root / "state" / "coordination.duckdb") in lane_process["argv"]
assert expected_coordination_cid in "\0".join(lane_process["argv"])
lane_environment = dict(
    item.decode("utf-8", "replace").split("=", 1)
    for item in Path(f"/proc/{lane_pid}/environ").read_bytes().split(b"\0")
    if b"=" in item
)
assert {
    key: lane_environment.get(key)
    for key in binding["provider_environment"]["required"]
} == binding["provider_environment"]["required"]
assert all(
    key not in lane_environment for key in binding["provider_environment"]["unset"]
)
assert lane.get("task_ids") == [stage["display_task_id"]]
assert lane.get("expected_task_cids_by_id") == {
    stage["display_task_id"]: stage["canonical_task_cid"]
}

dry_snapshot = Path(binding["dry_manifest"]["snapshot_path"])
dry_raw, dry = load_regular_json(dry_snapshot)
assert hashlib.sha256(dry_raw).hexdigest() == binding["dry_manifest"]["sha256"]
dry_lanes = [
    item
    for item in dry.get("lanes", [])
    if item.get("expected_task_cids_by_id")
    == {stage["display_task_id"]: stage["canonical_task_cid"]}
]
assert len(dry_lanes) == 1
dry_lane = dry_lanes[0]
assert dry_lane.get("task_cid") == stage["task_spec_cid"]
assert lane.get("command") == dry_lane.get("command")
assert hashlib.sha256(canonical(lane["command"])).hexdigest() == (
    binding["implementation_envelope"]["lane_command_sha256"]
)
assert content_identity(lane["command"]) == (
    binding["implementation_envelope"]["lane_command_cid"]
)

worktree_root = Path(binding["implementation_envelope"]["worktree_root"]).resolve()
state_dir = Path(str(lane.get("state_dir") or ""))
if not state_dir.is_absolute():
    state_dir = (repo / state_dir).resolve()
state_prefix = str(lane.get("state_prefix") or "")
portal_state_path = state_dir / f"{state_prefix}_task_state.json"
events_path = state_dir / f"{state_prefix}_events.jsonl"
deadline = time.monotonic() + 300.0
while True:
    pool_records: list[tuple[Path, bytes, dict[str, object]]] = []
    for path in sorted((worktree_root / ".pool-state").glob("*.json")):
        raw, value = load_regular_json(path)
        if value.get("state") in {"initializing", "leased"}:
            pool_records.append((path, raw, value))
    matching_events: list[dict[str, object]] = []
    events_raw = b""
    if events_path.is_file() and not events_path.is_symlink():
        try:
            events_raw = events_path.read_bytes()
            events = [json.loads(line) for line in events_raw.splitlines() if line.strip()]
            matching_events = [
                event
                for event in events
                if event.get("type") == "implementation_started"
                and event.get("task_id") == stage["display_task_id"]
                and event.get("attempt") == binding["attempt_policy"]["expected_next_attempt"]
            ]
        except json.JSONDecodeError:
            matching_events = []
    if len(pool_records) == 1 and portal_state_path.is_file() and len(matching_events) == 1:
        portal_raw, portal = load_regular_json(portal_state_path)
        if (
            pool_records[0][2].get("state") == "leased"
            and portal.get("implementation_in_progress") is True
            and portal.get("active_task_id") == stage["display_task_id"]
        ):
            break
    if time.monotonic() >= deadline:
        raise RuntimeError("timed out before exact stage worktree/base binding")
    time.sleep(1.0)

pool_path, pool_raw, pool = pool_records[0]
implementation_started = matching_events[0]
assert pool.get("schema") == "agent-supervisor-worktree-pool-v1"
assert pool.get("lease_token") == pool_path.stem
assert (pool_path.with_suffix(".lock")).is_file()
assert not (pool_path.with_suffix(".lock")).is_symlink()
assert pool.get("base_commit") == binding["target"]["head"]
assert pool.get("repo_root") == str(repo)
lease_pid = int(pool.get("lease_pid") or 0)
assert lease_pid > 1 and Path(f"/proc/{lease_pid}").is_dir()
lease_process = process_identity(lease_pid)
assert lane_pid in ancestor_pids(lease_pid)
lease_argv = [
    item.decode("utf-8", "replace")
    for item in Path(f"/proc/{lease_pid}/cmdline").read_bytes().split(b"\0")
    if item
]
assert "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon" in lease_argv
assert str(state_dir) in lease_argv

workspace_raw = Path(str(pool.get("path") or ""))
assert not workspace_raw.is_symlink()
workspace = workspace_raw.resolve()
assert workspace.is_dir()
assert workspace.is_relative_to(worktree_root)
assert workspace.parent == worktree_root
assert portal.get("active_task_cid") == stage["canonical_task_cid"]
assert portal.get("active_attempt") == binding["attempt_policy"]["expected_next_attempt"]
assert Path(str(portal.get("active_worktree_path") or "")).resolve() == workspace
assert portal.get("active_branch") == pool.get("branch")
assert implementation_started.get("command") == shlex.split(
    binding["implementation_envelope"]["implementation_command"]
)
assert implementation_started.get("worktree_path") == str(workspace)
assert implementation_started.get("branch") == pool.get("branch")
assert implementation_started.get("baseline_ref") == binding["target"]["head"]
assert (implementation_started.get("workspace_setup") or {}).get("base_commit") == (
    binding["target"]["head"]
)
assert str(implementation_started.get("event_id") or "").startswith("baguqeera")
assert subprocess.run(
    ["git", "-C", str(workspace), "rev-parse", "--verify", "HEAD^{commit}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert subprocess.run(
    ["git", "-C", str(workspace), "rev-parse", "--verify", f"{pool['base_commit']}^{{tree}}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip() == binding["target"]["tree"]
assert subprocess.run(
    ["git", "-C", str(workspace), "merge-base", "--is-ancestor", pool["base_commit"], "HEAD"],
    check=False,
).returncode == 0
assert subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD^{commit}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip() == binding["target"]["head"]

body = {
    "schema": "ipfs_accelerate_py/agent-supervisor/operator-stage-start-verification@1",
    "binding_id": binding_id,
    "stage": stage,
    "target": binding["target"],
    "live_manifest": {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "content_cid": content_identity(manifest),
        "lane_process": lane_process,
    },
    "pool_state": {
        "path": str(pool_path),
        "sha256": hashlib.sha256(pool_raw).hexdigest(),
        "base_commit": pool["base_commit"],
        "lease_pid": lease_pid,
        "lease_process": lease_process,
        "workspace": str(workspace),
    },
    "portal_state": {
        "path": str(portal_state_path),
        "sha256": hashlib.sha256(portal_raw).hexdigest(),
        "active_attempt": portal["active_attempt"],
        "canonical_task_cid": portal["active_task_cid"],
    },
    "implementation_started": {
        "events_path": str(events_path),
        "events_sha256": hashlib.sha256(events_raw).hexdigest(),
        "event_id": implementation_started["event_id"],
        "command": implementation_started["command"],
        "baseline_ref": implementation_started["baseline_ref"],
        "worktree_path": implementation_started["worktree_path"],
        "branch": implementation_started["branch"],
        "workspace_setup_base_commit": implementation_started["workspace_setup"][
            "base_commit"
        ],
    },
    "authority": {
        "task_completion": False,
        "verification": False,
        "release": False,
        "proof": False,
        "successor_launch": False,
        "external": False,
    },
}
record = {**body, "verification_id": content_identity(body)}
data = canonical(record)
receipt_path = binding_path.parent / (
    f"stage-start-{stage['display_task_id']}-{hashlib.sha256(binding_id.encode()).hexdigest()[:16]}.json"
)
persist_no_clobber(receipt_path, data)
assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
assert receipt_path.stat().st_uid == os.getuid()
assert canonical(json.loads(receipt_path.read_bytes())) == data
print(receipt_path)
PY
  )
}

shq_cancel_mismatched_stage() {
  ( cd "$SHQ_REPO" && "${SHQ_PROVIDER_ENV[@]}" "$SHQ_PYTHON" - "$@" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

repo = Path(sys.argv[1]).resolve()
run_root = Path(sys.argv[2]).resolve()
binding_raw_path = Path(sys.argv[3])
expected_coordination_cid = sys.argv[4]
assert not binding_raw_path.is_symlink()
binding_path = binding_raw_path.resolve()
assert sys.flags.optimize == 0

def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")

def persist_no_clobber(path: Path, data: bytes) -> None:
    assert not path.exists() and not path.is_symlink()
    temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            assert written > 0
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temp, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)

def load_regular_json(path: Path) -> tuple[bytes, dict[str, object]]:
    assert path.is_file() and not path.is_symlink()
    raw = path.read_bytes()
    value = json.loads(raw)
    assert isinstance(value, dict)
    return raw, value

def process_identity(pid: int) -> dict[str, object] | None:
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not stat_path.is_file() or not cmdline_path.is_file():
        return None
    try:
        stat_raw = stat_path.read_text(encoding="utf-8")
        argv = [
            item.decode("utf-8", "replace")
            for item in cmdline_path.read_bytes().split(b"\0")
            if item
        ]
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    fields = stat_raw.rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "start_time_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(),
        "argv": argv,
    }

def same_birth(current: dict[str, object] | None, expected: dict[str, object]) -> bool:
    return current is not None and current == {
        "pid": expected["pid"],
        "ppid": expected["ppid"],
        "start_time_ticks": expected["start_time_ticks"],
        "boot_id": expected["boot_id"],
        "argv": expected["argv"],
    }

def send_exact(pid: int, expected: dict[str, object], signum: int) -> str:
    current = process_identity(pid)
    if current is None:
        return "absent"
    if not same_birth(current, expected):
        return "identity_changed"
    os.kill(pid, signum)
    return f"sent:{signum}"

binding_raw, binding = load_regular_json(binding_path)
assert canonical(binding) == binding_raw
binding_body = dict(binding)
binding_id = str(binding_body.pop("binding_id"))
assert content_identity(binding_body) == binding_id
stage = binding["stage"]
assert expected_coordination_cid == stage["coordination_task_cid"]

manifest_path = run_root / "bundle_lanes.json"
manifest_raw, manifest = load_regular_json(manifest_path)
lanes = [
    lane
    for lane in manifest.get("lanes", [])
    if lane.get("task_cid") == expected_coordination_cid
]
lane = lanes[0] if len(lanes) == 1 else None
lane_pid = int((lane or {}).get("pid") or 0)
observed_identity = process_identity(lane_pid) if lane_pid > 1 else None
signals: list[str] = []
if observed_identity is not None:
    assert "ipfs_accelerate_py.agent_supervisor.merge.leased_lane" in observed_identity["argv"]
    assert str(run_root / "state" / "coordination.duckdb") in observed_identity["argv"]
    assert expected_coordination_cid in "\0".join(observed_identity["argv"])
    signals.append(send_exact(lane_pid, observed_identity, signal.SIGTERM))
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and same_birth(process_identity(lane_pid), observed_identity):
        time.sleep(0.2)
    if same_birth(process_identity(lane_pid), observed_identity):
        signals.append(send_exact(lane_pid, observed_identity, signal.SIGKILL))
        kill_deadline = time.monotonic() + 10.0
        while time.monotonic() < kill_deadline and same_birth(
            process_identity(lane_pid), observed_identity
        ):
            time.sleep(0.1)
    assert not same_birth(process_identity(lane_pid), observed_identity)

current_head = subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD^{commit}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
current_tree = subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD^{tree}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert current_head == binding["target"]["head"]
assert current_tree == binding["target"]["tree"]

coordination_path = run_root / "state" / "coordination.duckdb"
lease_state = None
release_reason = None
receipt_status = None
receipt_cid = None
output_cid = None
if coordination_path.is_file() and not coordination_path.is_symlink():
    connection = duckdb.connect(str(coordination_path), read_only=True)
    try:
        lease_row = connection.execute(
            "SELECT state,release_reason FROM leases WHERE task_cid=?",
            [expected_coordination_cid],
        ).fetchone()
        receipt_rows = connection.execute(
            "SELECT receipt_cid,payload_json FROM receipts "
            "WHERE task_cid=? ORDER BY rowid",
            [expected_coordination_cid],
        ).fetchall()
    finally:
        connection.close()
    if lease_row is not None:
        lease_state, release_reason = lease_row
    if receipt_rows:
        receipt_cid, receipt_json = receipt_rows[-1]
        receipt = json.loads(receipt_json)
        receipt_status = receipt.get("status")
        output_cid = receipt.get("output_cid")
assert receipt_status != "succeeded"
assert output_cid is None
assert lease_state != "completed"
if lease_state is not None:
    assert lease_state in {"released", "leased", "accepted"}
    if lease_state == "released":
        assert receipt_status in {None, "cancelled", "failed"}
        if receipt_status is not None:
            assert str(release_reason or "").startswith(f"receipt:{receipt_status}:retryable")

body = {
    "schema": "ipfs_accelerate_py/agent-supervisor/operator-stage-mismatch-cancellation@1",
    "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "binding_id": binding_id,
    "stage": stage,
    "target": binding["target"],
    "live_lane": {
        "present": lane is not None,
        "pid": lane_pid or None,
        "process": observed_identity,
        "signals": signals,
        "alive_after_signals": same_birth(process_identity(lane_pid), observed_identity)
        if observed_identity is not None
        else False,
    },
    "coordination": {
        "task_cid": expected_coordination_cid,
        "lease_state": lease_state,
        "release_reason": release_reason,
        "receipt_cid": receipt_cid,
        "receipt_status": receipt_status,
        "output_cid": output_cid,
    },
    "authority": {
        "task_completion": False,
        "verification": False,
        "release": False,
        "proof": False,
        "successor_launch": False,
        "external": False,
        "attempt_2": False,
    },
}
record = {**body, "cancellation_id": content_identity(body)}
data = canonical(record)
receipt_path = binding_path.parent / (
    f"stage-cancel-{stage['display_task_id']}-{hashlib.sha256(binding_id.encode()).hexdigest()[:16]}.json"
)
persist_no_clobber(receipt_path, data)
assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
assert receipt_path.stat().st_uid == os.getuid()
assert canonical(json.loads(receipt_path.read_bytes())) == data
print(receipt_path)
PY
  )
}

if ! SHQ_STAGE_START_VERIFICATION=$(shq_verify_started_stage \
  "$SHQ_REPO" "$SHQ_RUN" "$SHQ_STAGE_BINDING_PATH" \
  "$SHQ_STAGE_COORDINATION_TASK_CID"); then
  SHQ_STAGE_MISMATCH_CANCELLATION=$(shq_cancel_mismatched_stage \
    "$SHQ_REPO" "$SHQ_RUN" "$SHQ_STAGE_BINDING_PATH" \
    "$SHQ_STAGE_COORDINATION_TASK_CID")
  test "$(stat -c '%a' "$SHQ_STAGE_MISMATCH_CANCELLATION")" = 600
  exit 1
fi
test "$(stat -c '%a' "$SHQ_STAGE_START_VERIFICATION")" = 600
```

Any failed pre-start or post-start assertion is a hard boundary. The
`if ! shq_verify_started_stage` wrapper always invokes
`shq_cancel_mismatched_stage` before exiting: that helper binds the exact live
lane PID, coordination CID, run root and process birth identity, sends
`SIGTERM` then `SIGKILL` only to that birth identity, refuses a succeeded
receipt or target-branch mutation, and writes an owner-only
`operator-stage-mismatch-cancellation@1` record. The operator records start
verification and mismatch cancellation only as detective evidence; neither
grants completion, verification, release, proof, successor-launch or external
authority.

The command above is the only initial launch and terminates through `--once`.
After its detached lane has durably succeeded and merged cleanly, the operator
must repeat the no-start dry plan, require the completed predecessor to be
receipt-backed and nonclaimable, require exactly its direct successor to be the
sole claimable lane, and then run the same `--start --once` envelope. Repeat
that successor-admission gate serially for G006B, G006, and G007; never run a
persistent outer scheduler, and restore `--max-task-attempts 1` for every new
successor canonical task. If and only if the typed-transient attempt-2 gate
above passes after a failed stage is fully quiescent, the operator may instead
repeat the identical dry plan and identical one-shot invocation for that same
task with the sole policy change `--max-task-attempts 1` →
`--max-task-attempts 2`. This exposes exactly attempt 2/repair-round 1 for the
same frozen task input. A semantic/contract rejection, attempt exhaustion,
changed source/envelope, missing quiescence, or any other failure forbids that
same-task rerun and requires a reviewed migration.

For a typed-transient attempt 2 only, set `SHQ_MAX_TASK_ATTEMPTS=2` and point
`SHQ_RETRY_AUTHORIZATION_PATH` at an owner-only canonical mode-`0600`
`typed-transient-retry-authorization@1` record. That record is the sole
operator adjudication authority for the retry: it binds the exact member,
TaskSpec and coordination CIDs, the actual released/null-output receipt, and
three canonical evidence records for the failed trigger, its before state and
its distinct corrected after state. Its trigger kind is exactly one of setup,
provider, resource or process; all authority flags except `attempt_2` are
false. The helper independently rechecks the canonical attempt count and idle
reason, absence of retry-budget repair, released coordination, empty managed
worktree/pool state, absent worktree registration/ref and unchanged
HEAD/tree/envelope. A label or unbound prose assertion is insufficient.

Launch and dry-plan with an explicitly cleared provider environment and these
positive bindings: `IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=codex`,
`IPFS_ACCELERATE_AGENT_CODEX_MODEL=gpt-5.6-terra`,
`IPFS_ACCELERATE_AGENT_CODEX_CONTEXT_WINDOW=49152`,
`IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT=high`,
`IPFS_ACCELERATE_AGENT_CODEX_MAX_THREADS=1`,
`IPFS_ACCELERATE_AGENT_CODEX_MAX_DEPTH=0`, and
`IPFS_ACCELERATE_AGENT_DISABLE_SUBAGENTS=1`. Clear
`IMPLEMENTATION_DAEMON_COMMAND`, provider-fallback variables, Copilot tokens and
`IPFS_PROOF_REUSE_STATE_ROOT`. The explicit `--implementation-command` is the
route authority: it bypasses auto discovery and the Codex-to-Copilot fallback.
Parse it with `shlex.split` during preflight and require the exact direct Codex
argv, no `copilot`, `grok`, `goose` or shell wrapper, and a final stdin marker.

The context-window value is the total provider envelope, not an input-token
budget. The implementation compiler reserves 16,384 output tokens and 8,192
tool tokens. A 49,152-token provider window therefore leaves an exact 24,576
token input allowance. The direct Codex argv and daemon-visible environment
must both report the 49,152-token total envelope. The capacity snapshot's
`context_window_tokens` field is named like a total window but this producer
defines it as the usable input admission budget, so it must remain 24,576. Its
response-token admission budget is a third, separate ceiling. Fail preflight
when any of those values differ. An initial v4 start incorrectly
pinned the total window to 24,576 and consequently failed closed in context
compilation with zero usable input tokens. It created no implementation
worktree, made no model call, incurred no model billing and produced no task
completion evidence; coordination released attempt/fence 1/1 as
`cancelled:retryable` before this corrected retry.

The fresh SHQ-026/027/028/029 planning records do not declare a provider route or a
nonzero provider resource estimate, so bundle admission does not bind the
explicit Codex command to that telemetry. For this bootstrap, the `jq` check
above and the live `implementation_started.command` comparison are operator
preflight/detective controls, not scheduler-enforced route evidence. The
capacity producer must retain `--context-budget-tokens 24576`
before the retry. The implemented capstone must close this gap by emitting
provider-neutral, nonempty route/resource requirements that the resource
scheduler can enforce; none of this bootstrap telemetry counts as
qualification model-route evidence.

The current supervisor constrains edits, not reads: its native Landlock policy
does not prevent a provider from reading other host paths. The content-addressed
four bounded-v12 tasks therefore forbid such reads, use only their current
clean launch checkout and declared submodules, disable subagents, and are
actively monitored. No rescue-commit or prior-attempt revision is readable.
Stop the scheduler
wrapper if the implementation event differs from the pinned argv or any child
starts a command whose resolved arguments leave the disposable worktree. Do not
represent this detective boundary as hard provider sandboxing or qualification
evidence.

Do not pass `--allow-missing-provider-telemetry`. Missing or stale telemetry is
valid backpressure. Do not enable objective refinement, codebase refill or a
second one-shot supervisor against a live lane.

The scoped bootstrap scan intentionally produces only `SHQ-G006A`,
`SHQ-G006B`, `SHQ-G006`, and `SHQ-G007`, in distinct bundle keys with the
exact dependency chain G006A→G006B→G006→G007. G005A remains outside scope and
no G005A/SHQ-023 assumed-completion flag is present. Do not perform the prior
unscoped rerun after G010 while G005A is blocked; downstream projection is
deferred until formal operator reconciliation explicitly reopens and then
verifies G005A. Any later reviewed unscoped invocation retains
`--surplus-findings-per-goal 1` and targeted source exclusions only for
genuinely unrelated/vendored trees.

## 18. Monitoring and anti-stall runbook

Poll every 30–60 seconds while work is active:

```bash
jq '{generated_at,scheduler_state,cycle,counts,backpressure_reasons,discovery_error,
     lanes:[.lanes[]?|{bundle_key,pid,state,task_ids}],
     blocked:[.blocked[]?|{bundle_key,task_cid,blocked_reason,blocking_task_cids}]}' \
  "$SHQ_RUN/bundle_lanes.json"

stat -c '%y %s %n' \
  "$SHQ_RUN/bundle_lanes.json" \
  "$SHQ_RUN/scheduler_metrics.json"

find "$SHQ_RUN/state" -name '*_task_state.json' -type f -print0 | \
  xargs -0 -r jq -c \
  '{heartbeat_at,active_task_id,active_phase,active_phase_started_at,
    implementation_in_progress,ready_count,waiting_count,blocked_count,
    completed_count,last_progress_at,last_implementation_log_path,
    last_merge_returncode,last_merge_error}'

find "$SHQ_RUN/state" -name '*_supervisor_status.json' -type f -print0 | \
  xargs -0 -r jq -c \
  '{updated_at,status,daemon_pid,restart_count,last_exit_code,last_recycle_reason,
    maintenance_phase,backpressure,backpressure_reasons,active_worker_count}'
```

Validate each live PID with `ps`, inspect lane logs/events and check artifact and
branch evidence. A healthy scheduler manifest advances about every five seconds;
lane health advances around every 30 seconds. The 1,200-second stale threshold
is an alarm boundary, not a reason to kill children blindly.

Intervention order:

1. Inspect immutable receipts, events, logs, PIDs, resource/provider telemetry,
   coordination leases and merge state.
2. Distinguish external dependency waiting, intentional resource backpressure,
   an active model/validator child and an actual no-progress condition.
3. Use only a configured typed drain/pause/cancel control capability.
4. Stop the scheduler wrapper, never individual lane children; verify descendants
   and leases settle.
5. Resume the exact command with identical repo, branch, bundle index, state,
   worktree, coordination and provider bindings so lease reconciliation runs.
6. Retry only a changed transient trigger within budget.
7. Quarantine persistent failure, then use reviewed rescue preview/rescue if
   authorized. Never edit todo status, receipts, strategy JSON, keys or the
   coordination database by hand.

The later post-G010 qualification scheduler is intentionally persistent after
queue drain; this does not apply to the bounded-v12 G006A→G007 one-shot chain.
An empty queue with
an open external gate is `waiting_external_admission`, not a stall. Upstream
supervisors are monitored separately; when all are terminal, an operator creates
and validates the external receipt, reconciles `SHQ-G010`, reruns the objective
daemon and resumes this scheduler. No blind restart can bypass that gate.

## 19. Completion and reporting

The terminal report includes all user-requested revisions, target/corpus/split,
baseline, per-arm, context, routing, cost, verification/proof, quality/hidden,
compression/assurance, review, crash, longitudinal, level, blockers and go/no-go
fields. Initial target values are evaluated as goals and reported honestly.

The final claim is limited to the exact release, target, task classes, models,
environment and policy. Implemented components alone never imply production
readiness.
