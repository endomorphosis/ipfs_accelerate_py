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

The `SHQ-G006` observer implementation is deliberately stricter than a
file-presence scan; `SHQ-G007` runs it only after G006 is merged and clean.
The prerequisite catalog is the exact non-empty ordered list of ten unique
requested systems; omission, addition, duplication, or reordering fails closed.
For each row the observer binds the clean outer repository `HEAD` and tree,
the exact recursive gitlink and matching submodule `HEAD` and tree, and
deterministic digests/CIDs of every tracked source and evidence blob used in
the decision. Every configured module, package-export, release-manifest,
owner-board, and receipt path must be non-empty and repository-relative, contain
neither an absolute root nor `..`, and remain beneath its one declared
checkout/submodule root after existing-parent and symlink resolution.

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
and `VerificationProcessRunner`: compile the exact TEST key with the pytest
adapter schema, run an actual `VerificationCommand`, construct the
`DirectExecutionObservation` and `TestReceipt` in the trusted process, round
trip the canonical record, then admit and exact-key lookup through
`VerificationReceiptCache` with production eligibility required. A deserialized
observation or receipt supplied from disk/cache is structural data, not proof
that execution occurred. An injected pytest phase report is categorically
forbidden. Admission requires a present real run result, process-started and
completed disposition, zero exit, `ok` and publication allowed, exact observed
argv/selectors/tool version, no timeout/cancel/unavailable/simulation/replay,
stdout and stderr CIDs, freshness, and identical clean pre/post repository
forests matching the complete source identity. Existing proof test receipts or
semantic compiled receipts may corroborate this evidence but are never
sufficient without the current direct run. Self-asserted/ignored JSON, stale
evidence, an absent output identity, or a structurally valid receipt without
trusted execution authority is `unverifiable`. The same result applies to
unreadable/malformed inputs and unknown, missing, or duplicate board status.

The observation is versioned despite the repository-wide JSON ignore rule:
`.gitignore` must contain exactly the narrow exception
`!artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json`.
Serialized paths and roots are deterministic canonical repository-relative
values with no host-absolute prefix. It uses explicit two-phase identity rather than an impossible self-reference.
G006 commits the observer, tests and ignore exception and makes that tree clean;
G007 then generates the observation. Its source binding is that clean G006
commit/tree, including recursive gitlinks and matching submodule heads and
excluding only the observation artifact. Committing the artifact creates an
evidence projection; it does not change what the JSON claims to have observed.
Native validation and completion receipts independently bind the clean
post-artifact tree. Immediately before publication both modes recompute and
compare the entire in-memory observation, outer `HEAD`/tree, recursive
gitlinks/submodule identities, tracked-content digests, and every configured
evidence input. `observe` may record non-terminal rows but may publish only a
structurally complete ten-row snapshot; `require-terminal` additionally
requires every row terminal. Publication uses a same-directory exclusive
temporary plus durable atomic no-clobber operation, refuses an existing target,
and cleans up on every failure. No partial, stale, raced, new, or replaced
admission artifact is evidence.

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

The objective heap defines 38 autonomous work goals and
two external gates. The objective daemon generates their task IDs in a
deterministic scan and assigns content IDs after checking repository evidence.

| Goal | Planned work item | Owner | Depends on |
|---|---|---|---|
| `SHQ-G006` | Prerequisite observer implementation, tests and exact ignore exception | accelerate | — |
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
W0a  G006
W0b  G007
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
SHQ_REPO=/home/barberb/lift_coding/.worktrees/ipfs-accelerate-self-hosting-qualification
SHQ_DATA=data/agent_supervisor/self_hosting_qualification
SHQ_PROJECTION="$SHQ_DATA/projections/v5"
SHQ_PYTHON=/usr/bin/python3.12
SHQ_ACTIVE_TODO=docs/architecture/self_hosting_qualification.todo.md
SHQ_V1_HISTORY_TODO=docs/architecture/self_hosting_qualification.v1_history.todo.md
SHQ_V2_HISTORY_TODO=docs/architecture/self_hosting_qualification.v2_history.todo.md
SHQ_V3_HISTORY_TODO=docs/architecture/self_hosting_qualification.v3_history.todo.md
SHQ_V4_HISTORY_TODO=docs/architecture/self_hosting_qualification.v4_history.todo.md
SHQ_RUN=/home/barberb/.local/state/ipfs_accelerate_py/self-hosting-qualification-v5
# Read-only input from the already-live provider monitor. All mutable v5
# coordination, state, worktrees, logs, manifests, metrics, gates and keys use
# SHQ_RUN above; never reopen or alias the retired v1 supervisor namespace.
SHQ_CAPACITY_PATH=/home/barberb/.local/state/ipfs_accelerate_py/self-hosting-qualification-v1/provider-capacity/capacity.json
SHQ_GATE="$SHQ_RUN/operator/objective_completion_gate.json"
SHQ_EXTERNAL_AUTHORITY="$SHQ_RUN/operator/external_completion_authority.json"
SHQ_SIGNING_KEY="$SHQ_RUN/operator/signing.key"
SHQ_IMPLEMENTATION_COMMAND="/usr/local/bin/codex exec --ephemeral --ignore-user-config --strict-config --dangerously-bypass-approvals-and-sandbox --color never -m gpt-5.6-terra -c model_context_window=49152 -c 'model_reasoning_effort=\"high\"' -c agents.max_threads=1 -c agents.max_depth=0 -"
SHQ_G006_RUNTIME_TODO="$SHQ_RUN/state/agent-supervisor-self-hosting-prerequisite-observer-implementation-bounded-v5/state/agent_agent_supervisor_self_hosting_prerequisite_observer_implementation_bounded_v5_runtime.todo.md"
SHQ_G007_RUNTIME_TODO="$SHQ_RUN/state/agent-supervisor-self-hosting-prerequisite-observation-snapshot-bounded-v5/state/agent_agent_supervisor_self_hosting_prerequisite_observation_snapshot_bounded_v5_runtime.todo.md"

# Reviewed migrations: retain SHQ-001 in the v1 history board. Move cancelled
# SHQ-002 and its never-launched dependent SHQ-003 into the v2 history board,
# retain every historical canonical task block byte-for-byte and record outcome
# only in its history-board preamble. SHQ-004/005 are prelaunch v3 projections.
# Archive the current SHQ-006/007 canonical blocks byte-for-byte in the v4
# history board: SHQ-006 was rejected/cancelled retryable after independent
# critical review, and dependent SHQ-007 never launched. Apply that reviewed
# tracked migration before this command, leave SHQ_ACTIVE_TODO title-only, and
# retain all seven discovery files so display IDs SHQ-001 through SHQ-007 stay
# reserved. The v5 generation must allocate SHQ-008 and SHQ-009.
test -f "$SHQ_REPO/$SHQ_V1_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V2_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V3_HISTORY_TODO"
test -f "$SHQ_REPO/$SHQ_V4_HISTORY_TODO"
test -x "$SHQ_PYTHON"
test "$("$SHQ_PYTHON" --version 2>&1)" = 'Python 3.12.3'
! rg -q '^## SHQ-' "$SHQ_REPO/$SHQ_ACTIVE_TODO"
test ! -e "$SHQ_REPO/$SHQ_PROJECTION"
test ! -e "$SHQ_RUN"

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
  --max-findings 2 \
  --scope-goal-id SHQ-G006 \
  --scope-goal-id SHQ-G007 \
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
  --protected-output-path docs/architecture/self_hosting_qualification.todo.md \
  --protected-output-path "$SHQ_ACTIVE_TODO" \
  --protected-output-path "$SHQ_V1_HISTORY_TODO" \
  --protected-output-path "$SHQ_V2_HISTORY_TODO" \
  --protected-output-path "$SHQ_V3_HISTORY_TODO" \
  --protected-output-path "$SHQ_V4_HISTORY_TODO" \
  --protected-output-path docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md \
  --protected-output-path artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json \
  --protected-output-path artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy.json \
  --protected-output-path artifacts/agent_supervisor/self_hosting_qualification/hidden_evaluator_manifest.json \
  --protected-output-path config/self_hosting_qualification_policy.json \
  --protected-output-path config/self_hosting_qualification_trusted_keys.json
)
```

The history boards retain `SHQ-001` as an abandoned combined task, `SHQ-002` as
a cancelled unbounded implementation attempt, and `SHQ-003` as its unlaunched
dependent. SHQ-004/005 record a never-launched v3 projection whose snapshot CID
exposed missing retry semantics before scheduling. SHQ-006 is a rejected and
cancelled/retryable v4 observer task after independent critical review found
fail-closed contract gaps; its dependent SHQ-007 never launched. Every v4
canonical task block is preserved byte-for-byte in the v4 history board, and
only its preamble records disposition. Historical cards are never completion
evidence or scheduler sources. Their discovery records stay
in `$SHQ_DATA/discovery`, reserving `SHQ-001` through `SHQ-007`; the clean active
board therefore receives exactly `SHQ-008` and `SHQ-009`. New graph, dataset and bundle
projections live under `$SHQ_PROJECTION`; no scheduler may read a v1, v2, v3, or v4
bundle index.

Review the portable v5 Markdown and JSON projections and add those exact files
with `git add -f`; do not add DuckDB databases, lock files, runtime state or
provider logs. The archived v1 board and both objective-control documents are
exact protected paths for every subsequent implementation lane.

The `--scan-exclude-path` arguments above are bounded bootstrap-generation
inputs only: they prevent the initial evidence-gap scan from rediscovering the
entire product while it projects `SHQ-G006` and `SHQ-G007`. They must not appear in a goal
completion reconciliation. Completion must compute its tree identity over all
source and recursive gitlinks; carrying these exclusions into reconciliation
would create a different, incomplete completion identity.

### Local `SHQ-G006` implementation and `SHQ-G007` snapshot

G006 owns only `.gitignore`, the observer and its tests; G007 owns only the JSON
snapshot. Their bounded-v5 bundle keys prevent either task from inheriting stale
state from retired projections. They are projected as SHQ-008 and SHQ-009 in
the v5 index while retaining stable goal identities G006 and G007. The G006 task may inspect only the exact rescue commit
named in its content-addressed refinement; all reads and searches remain within
its disposable worktree. The generated task dependency prevents the scheduler
from creating the G007 worktree until G006 has merged, so G007 observes the
clean merged implementation identity. Do not pause for objective reconciliation
between the task commits. Once both runtime todos are terminal, both commits
are merged and the target branch is clean, run the focused suite and no-output
CLI probes.

**Current execution boundary:** stop the bootstrap run after those probes and
the immutable G007 snapshot are complete. Do not execute the reconciliation
commands below in this qualification revision. The scoped reconciler checks
tracked bundle shards and paired successful merge events, but it has not yet
been independently qualified against the authoritative Profile-G TaskReceipt,
coordination lease, fencing token, and state-database lineage. Until that
narrow authority binding is implemented, reviewed, and named here by an exact
commit, G006 and G007 remain implementation evidence rather than formally
completed goals. This limitation is off the bootstrap path because generation
uses `--no-reconcile-goal-completion`, and it cannot open external gate G010.

The following blocks are retained as a **non-executable future protocol**. Once
the missing authority binding is qualified, create current tree-bound local
gates and formally reconcile G006 followed by G007, without external receipts
or scan exclusions:

```bash
test -z "$(git -C "$SHQ_REPO" status --porcelain=v1 --untracked-files=all)"
git -C "$SHQ_REPO" submodule status --recursive
"$SHQ_PYTHON" -m pytest -q test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py
"$SHQ_PYTHON" scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py \
  --repo-root "$SHQ_REPO" --mode observe --quiet
"$SHQ_PYTHON" scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py \
  --repo-root "$SHQ_REPO" --mode require-terminal --quiet && exit 99 || test "$?" -eq 1
test -f "$SHQ_REPO/artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json"
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
  --protected-output-path docs/architecture/self_hosting_qualification.todo.md
  --protected-output-path "$SHQ_ACTIVE_TODO"
  --protected-output-path "$SHQ_V1_HISTORY_TODO"
  --protected-output-path "$SHQ_V2_HISTORY_TODO"
  --protected-output-path "$SHQ_V3_HISTORY_TODO"
  --protected-output-path "$SHQ_V4_HISTORY_TODO"
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
  --protected-output-path docs/architecture/self_hosting_qualification.todo.md
  --protected-output-path "$SHQ_ACTIVE_TODO"
  --protected-output-path "$SHQ_V1_HISTORY_TODO"
  --protected-output-path "$SHQ_V2_HISTORY_TODO"
  --protected-output-path "$SHQ_V3_HISTORY_TODO"
  --protected-output-path "$SHQ_V4_HISTORY_TODO"
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
task. It begins only after the `SHQ-G006` implementation and `SHQ-G007`
snapshot tasks are independently merged, validated and reconciled complete.
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
  --protected-output-path docs/architecture/self_hosting_qualification.todo.md
  --protected-output-path "$SHQ_ACTIVE_TODO"
  --protected-output-path "$SHQ_V1_HISTORY_TODO"
  --protected-output-path "$SHQ_V2_HISTORY_TODO"
  --protected-output-path "$SHQ_V3_HISTORY_TODO"
  --protected-output-path "$SHQ_V4_HISTORY_TODO"
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

"$SHQ_PYTHON" -m ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor \
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
  --max-task-attempts 5 \
  --merge-target-branch agent/self-hosting-qualification-v1 \
  --implementation-protected-path docs/architecture/self_hosting_qualification.objectives.md \
  --implementation-protected-path docs/architecture/self_hosting_qualification.todo.md \
  --implementation-protected-path "$SHQ_ACTIVE_TODO" \
  --implementation-protected-path "$SHQ_V1_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V2_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V3_HISTORY_TODO" \
  --implementation-protected-path "$SHQ_V4_HISTORY_TODO" \
  --implementation-protected-path docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md \
  --implementation-protected-path artifacts/agent_supervisor/self_hosting_qualification/prerequisite_release_admission.json \
  --implementation-protected-path artifacts/agent_supervisor/self_hosting_qualification/preregistered_policy.json \
  --implementation-protected-path artifacts/agent_supervisor/self_hosting_qualification/hidden_evaluator_manifest.json \
  --implementation-protected-path config/self_hosting_qualification_policy.json \
  --implementation-protected-path config/self_hosting_qualification_trusted_keys.json \
  --worktree-submodule-path ipfs_datasets_py \
  --worktree-submodule-path ipfs_kit_py \
  --worktree-submodule-path ipfs_accelerate_py/mcplusplus
```

Protected-path matching is exact, not recursive. The commands therefore name
each governed file instead of relying on a directory such as `trusted_keys` to
protect descendants. `$SHQ_SIGNING_KEY` is deliberately absent from both the
repository and the implementation command; the operator provisions it with
mode `0600` only for the final kit signing port.

`--implement` compiles the exact lane command during this dry plan; the absence
of `--start` guarantees that no process is launched. After inspecting the
manifest, conflicts, exact merge target, protected arguments, finite attempt
limit, provider telemetry and resource claims, start with the same bindings
plus:

```text
--start
--poll-interval 5
--check-interval 30
--daemon-interval 45
--stale-seconds 1200
--watchdog-startup-grace-seconds 300
--implementation-timeout 14400
--max-restarts 8
```

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

The current SHQ-008/009 planning records do not declare a provider route or a
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
G006 task therefore forbids such reads, uses a repo-relative exact rescue-commit
reference, disables subagents, and is actively monitored. Stop the scheduler
wrapper if the implementation event differs from the pinned argv or any child
starts a command whose resolved arguments leave the disposable worktree. Do not
represent this detective boundary as hard provider sandboxing or qualification
evidence.

Do not pass `--allow-missing-provider-telemetry`. Missing or stale telemetry is
valid backpressure. Do not enable objective refinement, codebase refill or a
second one-shot supervisor against a live lane.

The scoped bootstrap scan intentionally produces only `SHQ-G006` and
`SHQ-G007`, in distinct bundle keys with G007 dependent on G006. After a
validated external receipt admits `SHQ-G010`, rerun the daemon without
`--scope-goal-id` and with `--surplus-findings-per-goal 1`; retain targeted
source exclusions only for genuinely unrelated/vendored trees. This preserves
one coherent generated task per leaf goal and avoids overlapping surplus tasks.

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

The scheduler is intentionally persistent after queue drain. An empty queue with
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
