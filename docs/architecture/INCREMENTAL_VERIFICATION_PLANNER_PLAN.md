# Incremental Verification Planner

**Status:** execution-ready implementation plan
**Primary repository:** `endomorphosis/ipfs_accelerate_py`
**Focused module:** `IncrementalVerificationPlanner`
**Board namespace:** `incremental-verification-planner-v1`
**Task prefix:** `IVP-`
**Companion goal heap:** `incremental_verification_planner.objectives.md`
**Companion taskboard:** `incremental_verification_planner.todo.md`

## 1. Outcome

Implement one focused incremental-verification subsystem under
`ipfs_accelerate_py.agent_supervisor.verification`. Given a repository state,
an invalidation plan, a context pack, and a proposed patch, it will select and
execute the minimum defensible verification work, reuse only exact valid
receipts, summarize failures compactly, and recommend a provider-neutral repair
route.

`IncrementalVerificationPlanner` is the primary module. A
`VerificationReceiptCache` and a `ModelRoutePlanner` are narrow collaborators
inside that subsystem, not separate products. The work must not create an
end-user application, generic MCP platform, payment or marketplace machinery,
a model provider, a distributed scheduler, or a zero-knowledge virtual machine.

The production API is equivalent to:

```python
create_verification_plan(
    repository_state,
    invalidation_plan,
    context_pack,
    patch_delta,
    policy,
) -> VerificationPlan

choose_model_route(
    context_pack,
    verification_plan,
    prior_attempts,
    available_models,
    policy,
) -> ModelRouteDecision

build_verification_commitment(
    verification_bundle,
) -> VerificationCommitment
```

## 2. Scope and ownership

### In scope

- Canonical receipt, plan, reuse-decision, route-decision, summary, bundle,
  counterexample, and commitment contracts.
- Exact cache identities binding every authority-relevant input.
- A durable immutable receipt store and compare-and-swap verification index
  through a narrow `ipfs_kit_py` adapter, plus a hermetic local backend.
- Narrow, lazy adapters for `RepositoryState`, `InvalidationPlan`,
  `SemanticCapsule`, and `ContextPack` supplied by `ipfs_datasets_py`.
- Semantic-edge test selection with conservative/full-suite fallbacks.
- Reproducible, bounded execution adapters for pytest, mypy, Z3, and an
  already-operational proof-assistant route when the repository capability
  registry admits one.
- Counterexample minimization and compact ContextPack-ready summaries.
- Provider-neutral deterministic/small/medium/frontier/human routing.
- A deterministic Merkle commitment over admitted receipts. The commitment is
  an extension point and explicitly is not a zero-knowledge proof.
- Controlled-fixture differential evaluation and honest benchmark reporting.

### Out of scope

- Full ZK pytest or Python execution.
- A new theorem prover or proof assistant.
- Mock hardware, mock inference, automatic dependency installation, or
  automatic deployment.
- Provider/vendor selection inside route policy.
- Browser/desktop UI, legal workflow, x402, MCP++ profile, arbitrary remote
  scheduling, or an agent product.

## 3. Reuse before creation

The implementation must compose existing repository primitives rather than
forking them:

- canonical JSON/CID and trust projections from
  `agent_supervisor.proof.formal_verification_contracts` and
  `agent_supervisor.core.multiformats_identity`;
- proof/cache trust doctrine from `formal_verification_cache`,
  `test_proof_cache`, and `proof_cached_test_validation`;
- code-impact and semantic graph queries from `analysis.code_evidence_graph`
  and related program-graph query modules;
- proof capability and backend admission from `code_contract_prover`,
  `multi_prover_router`, `prover_matrix_registry`, and
  `formal_verification_capabilities`;
- resource admission, leases, cancellation tokens, event/artifact storage, and
  process-tree termination from the modern agent-supervisor runtime;
- receipt byte transport from `ipfs_kit_py` behind one optional, lazy adapter;
- semantic input contracts from `ipfs_datasets_py` behind one optional, lazy
  adapter.

No cache presence, provider text, signature, CID string, historical pass, or
structural validation may create verification authority.

## 4. Canonical contracts

### 4.1 Closed terminal statuses

The only receipt terminal statuses are:

```text
passed, failed, proved, disproved, unknown, timeout, unavailable,
not_modeled, stale, invalid, cancelled, simulated
```

They are closed and case-sensitive on the wire. In particular:

- `timeout` and `unavailable` never project to `passed` or `proved`;
- `simulated` never satisfies a production requirement;
- `unknown`, `not_modeled`, `stale`, and `invalid` remain unresolved;
- cancellation fences publication of an otherwise late success;
- test/static success is `passed`; proof success is `proved`.

### 4.2 Required receipt and decision types

The canonical package exports:

- `StaticAnalysisReceipt`
- `TypeCheckReceipt`
- `TestReceipt`
- `ProofReceipt`
- `CounterexampleReceipt`
- `VerificationBundle`
- `VerificationSummary`
- `CacheReuseDecision`
- `ModelRouteDecision`
- `VerificationPlan`
- `VerificationCommitment`

All contracts are bounded, immutable, canonical-JSON compatible, reject
non-finite/open-ended values, and derive their identities from canonical bytes.
Private inputs, secrets, raw environment variables, and proof witnesses never
enter public receipts.

### 4.3 Receipt identity

`VerificationReceiptKey@1` binds at least:

1. repository tree CID;
2. semantic-state root CID;
3. sorted affected symbol-version CIDs;
4. environment CID;
5. dependency-lock CID;
6. check/test selector CID;
7. proof-obligation CID, or an explicit canonical `not_applicable` identity;
8. tool name;
9. tool version;
10. configuration CID;
11. sorted fixture-data CIDs;
12. network policy;
13. receipt-schema version.

The key also binds receipt kind and execution-adapter schema so two adapters
cannot alias. Every component is required and validated before lookup. Any
difference in tree, semantic root, affected symbol versions, environment,
dependency lock, selector, obligation, tool/version, configuration, fixture
data, network policy, or schema produces a different key. A candidate object
whose embedded key or content CID disagrees with the lookup key is corrupt and
is rejected, never repaired in place.

### 4.4 Tree-binding conflict and fail-closed resolution

Two requested conditions are mutually exclusive when "repository tree CID"
means the full executed tree: every edit, including an unrelated edit, changes
that CID, while a receipt from a different tree is explicitly forbidden from
reuse. This program gives the exact tree-binding rule precedence. After an
unrelated edit, the old receipt remains immutable and retrievable under its
original key, but it is not admitted for the new tree and cannot satisfy a
production requirement. A newly required check must execute and issue a new
tree-bound receipt.

The conformance suite therefore interprets "unrelated code change preserves
the receipt" as historical preservation, not cross-tree authority. The final
report must mark the requested cross-tree unaffected-reuse target unmet. A
future policy could introduce an independently verified equivalence receipt,
but silently ignoring or weakening the tree component is forbidden here.

### 4.5 Authoritative identity inventory

CID-shaped caller strings are references, not identity authority. A private
`VerificationIdentityCompiler` derives and cross-checks key components from
the effective execution inputs before lookup and again before publication:

- the repository tree CID identifies the exact patched tree or canonical dirty
  overlay actually mounted in the sandbox, never merely the base tree;
- patch base, RepositoryState tree, InvalidationPlan tree/semantic root, and
  ContextPack tree/semantic root must agree or planning fails closed;
- dependency-lock, configuration, fixture, selector, and obligation CIDs come
  from canonical observed bytes/argv/translation, not unchecked DTO fields;
- environment CID comes from the effective hermetic sandbox, executable,
  interpreter, platform, network policy, and admitted toolchain inventory;
- tool name/version comes from the exact executable selected for execution;
- Z3/proof identity binds the normalized obligation, negation/translation
  scheme, translator version, solver executable, and solver version.

Any pre-execution/post-execution inventory mismatch invalidates the candidate
and fences publication. If an identity cannot be observed reproducibly, the
result is typed unavailable and automatic production acceptance is impossible.

## 5. External adapter boundaries

### 5.1 `ipfs_datasets_py`

`DatasetsVerificationInputAdapter` is a small protocol adapter. It lazily
accepts the installed canonical `RepositoryState`, `InvalidationPlan`,
`SemanticCapsule`, and `ContextPack`, or their canonical mapping
representations. It extracts only documented identity roots, changed symbols,
dependency/test edges, uncertainty/frontier facts, source spans, contracts,
token estimates, and fixture-task references.

If the package or an expected schema is unavailable, the adapter returns a
typed unavailable/unsupported observation. It must not silently manufacture
semantic edges, guess CIDs, inspect arbitrary object internals, or make the
base accelerator import depend eagerly on datasets.

The reviewed gitlink is
`ipfs_datasets_py@6cd037c7738f44904add46391537588e67f6f238`. Workers may
inspect its public
`ipfs_datasets_py.knowledge_graphs.adapters.code_evidence.CodeEvidenceCorpusAdapter`
and `ipfs_datasets_py.logic.backends.process.BoundedToolRunner` seams, but may
not modify the nested repository. Capability checks target the exact leaf
module and symbol: a top-level namespace-package import alone is not evidence
that the dependency is operational.

At the pinned revision, `CodeEvidenceCorpusAdapter` is explicitly
compatibility-only (`OPERATIONAL_AUTHORITY=False`, `COMPATIBILITY_ONLY=True`).
It may supply bounded fixture/mapping evidence but never live semantic-index
authority. The newer DuckDB-backed `CodeEvidenceAuthority` is capability-
probed as the live upstream seam; adopting it as verification authority still
requires a separate narrow adapter and exact authority receipt rather than an
availability check.

### 5.2 `ipfs_kit_py`

`IpfsKitVerificationReceiptStore` adapts immutable byte put/get and a
generation-bound head compare-and-swap operation. It is responsible for:

- immutable receipt and artifact bytes;
- exact CID recomputation on read and write;
- current verification-index root;
- CAS publication of a replacement index root;
- replay of index history;
- corruption rejection;
- stale receipt tombstones;
- reachability/last-access metadata for later GC.

The adapter is optional at import time and performs no dependency installation
or daemon startup. `HermeticVerificationReceiptStore` implements the same
protocol with local files, locks, atomic replace/fsync, and deterministic CIDs
for unit/concurrency tests. An index writer must retry from a freshly read root
after a CAS conflict; it may never overwrite the other writer's entries.

The reviewed gitlink is
`ipfs_kit_py@5a7a2df8181cfdc33bc19be09989df7ff83f2d4e`. Immutable storage
adapts the public
`ipfs_kit_py.mcp_server.mcplusplus.coordination_storage.DurableCoordinationStore`
put/get/recover/CID operations. Current-head CAS is a separate generation-
bound local protocol, with an optional public Iroh manifest bridge only when
that exact capability is operational. Workers may inspect but not modify the
nested repository. Exact leaf-module and symbol probing is required because an
empty namespace-package import must never be treated as availability.

The pinned kit revision also exposes `proof_certificate_store`, but that is a
transport/storage seam only. A stored or signed certificate is not evidence of
test execution unless its issuer and exact execution binding are independently
trusted; it does not replace the receipt cache or production-admission rules.

The supervisor initializes both exact gitlinks for read-only inspection.
Source-tree integration validation explicitly uses
`PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:.` so Python resolves each nested
package root before the empty outer namespace directory. Production adapters
never mutate `sys.path`; they use normal installed-package resolution and an
exact leaf-symbol probe, returning typed unavailable when it fails.

## 6. Incremental planning algorithm

`create_verification_plan` follows a fail-closed, deterministic pipeline:

1. Validate and canonicalize all five inputs. Record typed adapter/capability
   gaps; never continue with an invented identity.
2. Rebind the patch to its declared base, compile the exact executed patched-
   tree/dirty-overlay identity, cross-check every semantic/context root, and
   calculate changed paths, symbols, semantic edges, environment/dependency/
   config/fixture changes, and declared-scope crossings.
3. Traverse the semantic index's test dependency edges from affected symbols.
   Track exact, conservative, opaque, missing, and truncated frontiers.
4. Determine affected static checks and proof obligations from selectors,
   symbol versions, policies, and obligation dependency edges.
5. Build exact required receipt keys for both affected and potentially
   unaffected checks. Query the cache by exact key only.
6. Classify each candidate with `CacheReuseDecision`: reused, stale, missing,
   corrupt, mismatched, simulated, non-authoritative, or policy-rejected.
7. Reuse only terminal, production-admissible receipts that match every key
   component. Planning is side-effect free: it returns stale decisions but
   does not publish tombstones. The cache/executor transaction may publish a
   tombstone only after revalidating the plan inputs and current index root;
   immutable history is retained.
8. Broaden selected tests when the semantic selector is uncertain. Missing,
   opaque, truncated, dynamic, or conflicting critical edges require a
   broader suite; policy can require the full suite.
9. Determine reproducibility from the effective hermetic filesystem/runtime,
   sandbox/toolchain inventory, network policy, and dependency lock. A caller-
   declared CID is insufficient; an unbound effective environment forces human
   review and prevents a production pass.
10. Allocate resource requirements and per-check deadlines using existing
    resource-admission policy. Cap total execution time at the policy maximum.
11. Emit ordered acceptance criteria: every required production check and
    obligation must have a current admissible terminal success, no required
    full-suite fallback may be pending, and human review must be false for
    automatic acceptance. Policy may leave an explicitly advisory, non-
    required obligation unresolved, but can never upgrade unknown, timeout,
    unavailable, or `not_modeled` evidence.

The resulting `VerificationPlan` records:

- reusable and stale receipt decisions;
- affected and fallback tests;
- required static/type checks;
- affected proof obligations;
- full-suite and human-review requirements with reason codes;
- expected CPU, memory, process, proof, and artifact resources;
- per-step and global maximum execution time;
- dependency DAG and deterministic order;
- acceptance criteria and policy identity.

## 7. Check execution

### 7.1 Shared runner

All command adapters use one admitted subprocess runner with:

- argv sequences only; no shell-string interpolation and `shell=False`;
- explicit executable and arguments in the plan and receipt;
- a sanitized allowlist environment and deterministic locale/time/hash inputs;
- an existing hermetic runtime or bounded-tool-runner filesystem boundary,
  read-only source plus private writable artifacts, explicit working tree,
  observed environment/dependency-lock CIDs, and enforced network policy;
- resource admission and a revocable lease before spawn;
- cooperative cancellation plus process-group/tree termination;
- timeout escalation from graceful termination to bounded hard kill;
- bounded stdout/stderr captured as artifacts, with digests in receipts;
- typed `unavailable` for a missing binary/backend;
- no installation, download, network-policy widening, or mock execution;
- publication only after cancellation and lease/fence rechecks.

### 7.2 Required adapters

- `PytestVerificationAdapter`: selected node IDs and full-suite oracle mode;
  setup/call/teardown semantics preserved; a timeout remains timeout. Empty
  collection and usage/malformed output are `invalid`; required skips/xfails
  are `not_modeled` unless policy declared them advisory before execution;
  unexpected xpass or collection/setup/teardown failure cannot pass.
- `MypyVerificationAdapter`: repository-used type checker, explicit file/module
  selectors and configuration; missing mypy is unavailable and usage/malformed
  output is invalid.
- `Z3VerificationAdapter`: use the existing admitted Z3/code-contract path and
  its capability probes; absent Z3 is unavailable, unknown is unknown. A
  proved/disproved projection binds the exact normalized/negated obligation,
  translator version, solver executable/version, and existing authoritative
  proof-assurance verdict; bare sat/unsat text is not authority.
- `ExistingProofAssistantAdapter`: wrap an existing registry-admitted
  Lean/Coq/Isabelle kernel route only when operational. If none is operational,
  return `unavailable`; do not add a prover to satisfy the plan.

## 8. Test selection and differential oracle

Controlled semantic-capsule fixture tasks run two observations:

1. the planner-selected tests;
2. the full suite as the oracle.

Each observation runs in a fresh isolated snapshot with the same tree,
environment, lock, fixture, and policy identities. `TestSelectionEvaluation@1`
binds the controlled fixture's reviewed ground-truth affected-test set as well
as selected/full sets and outcomes. A false negative is either a ground-truth
affected test omitted from selection or a mutation-caused full-suite failure
not observed by selected execution. A false positive is a selected test outside
the ground-truth affected set; a selected test that merely passes is not by
itself a false positive. Flaky, order-dependent, or selected/full outcome
discrepancies are classified separately as inconclusive.

A full-suite timeout/unavailable result, missing canonical semantic-capsule
corpus, or zero evaluated fixtures is `not_measured`, never zero false
negatives. Uncertain selection must set broader/full-suite fallback before
acceptance. Validation IDs from a datasets impact index require an exact
reviewed validation-ID-to-pytest-node-ID mapping; absence forces uncertainty
and a broader suite. The datasets `repository_tree_id` remains opaque selector
evidence and never substitutes for a validated receipt tree CID.

The benchmark corpus includes direct symbol changes, transitive changes,
fixture/config/environment/lock changes, unrelated edits, dynamic/opaque
edges, and deliberately failing tests.

## 9. Counterexample minimization

On failure, minimization operates on already bounded local artifacts. It
extracts the smallest reproduction that retains the failure:

- failed obligation or test selector;
- relevant symbols and source spans;
- minimized traceback and assertion;
- minimized relevant input;
- expected and observed values;
- environment identity and dependency-lock identity;
- argv reproduction command represented as a list;
- artifact CIDs/references.

It may use deterministic traceback slicing, semantic-cone pruning, pytest node
selection, and input shrinking exposed by the failing adapter. Every candidate
reproducer is rerun under a separate bounded admission lease and must retain
the same failure identity before it is called minimized. Sensitive or
inapplicable input/expected/observed values use typed redacted/unavailable
fields. If minimization cannot preserve the failure, the receipt says so and
references bounded logs; the planner never sends complete logs to a model by
default.

## 10. Model routing

`ModelRoutePlanner` selects capability class only. Model/provider resolution is
downstream and out of scope. Supported routes are:

```text
deterministic_only
small_local_model
medium_model
frontier_model
human_review_required
```

Inputs bind context token estimate, analysis classification, opaque dependency
count, risk, dependency-cone size, unresolved obligations, failure kind,
previous failed repairs, counterexample quality, exact-contract availability,
full-suite fallback state, environment reproducibility, and declared scope.

Precedence is fail-closed:

1. unresolved authority/policy, unmodeled high-risk effects, scope crossings,
   proof/test conflict, unsafe context, or non-reproducibility => human review;
2. exact mechanical formatting/import/codemod/rename work => deterministic;
3. localized bounded exact/conservative work with a good minimized
   counterexample and low/moderate risk => small local;
4. several-file nontrivial synthesis without an opaque critical dependency =>
   medium;
5. ambiguity, broad cones, opaque critical behavior, conflicting proof
   requirements, smaller-route failures, or context overflow => frontier.

Every decision includes considered routes, decisive reason codes, policy CID,
required capabilities, context estimate, and no vendor/model identifier.
`available_models` is a provider-neutral inventory of capability tier, context
limit, locality, and current availability, never vendor preference. If the
safely required tier is unavailable, routing does not downgrade: it returns
`human_review_required`. A pending mandatory broader/full suite likewise
prevents model repair and returns human review with a verification-incomplete
reason.

## 11. Bundle, summary, and commitment

`VerificationBundle` admits exact required receipts and explicit unresolved
requirements. `VerificationSummary` is a bounded projection for the next
ContextPack: changed cone, reused/executed checks, failures, compact
counterexamples, unresolved obligations, fallback state, timing/savings, and
model route.

`build_verification_commitment` rejects mixed tree/environment bundles,
canonicalizes required admitted leaf records as canonical DAG-JSON UTF-8,
sorts by receipt key/CID, and uses SHA-256 with schema-versioned domain bytes:
`H("IVP-LEAF@1\\0" || leaf)`,
`H("IVP-NODE@1\\0" || left || right)`, and
`H("IVP-EMPTY@1\\0")`. An unpaired node is promoted unchanged at each level.
It produces:

- receipt Merkle root;
- public statement;
- repository tree CID;
- environment CID;
- required-check-set CID;
- unresolved-obligation count;
- aggregate terminal status.

Aggregate status uses a closed fail-closed precedence: invalid, stale,
simulated, cancelled, timeout, unavailable, unknown, not_modeled, disproved,
failed, then success (proved only for an all-proof bundle; otherwise passed).
It cannot improve any required leaf. Changing, adding, or removing required
membership/content changes the root deterministically; input permutation does
not, because canonical sorting is required.

The contract and documentation must state:

- this commitment is not itself a ZK proof;
- signed receipts do not prove test execution unless the issuer is trusted;
- structural validation is not cryptographic validation.

No ZK backend is added. A future aggregator may prove membership/aggregation
over this stable receipt Merkle tree after the ordinary receipt chain has been
validated in production.

### 11.5 Evidence-envelope source snapshot

Checked benchmark and release-report evidence binds a canonical
`ivp-source-snapshot@1` identity rather than Git HEAD. The manifest represents
the effective set of present tracked and nonignored untracked paths, binding
regular-file bytes and canonical modes, symlink targets, and exact gitlink
object IDs. It excludes exactly the benchmark artifact and release report and
normalizes only exact IVP task-block `- Status: todo` / `- Status: completed`
rows. It never binds tracked/untracked provenance, a ref or commit, timestamps,
observation time, or an absolute checkout path.

The reviewed `ipfs_kit_py` and `ipfs_datasets_py` gitlinks must be initialized,
clean, and have nested HEAD equal to the superproject gitlink at their reviewed
revisions. Benchmark schema v2 and release-report binding schema v2 carry the
same recomputed `source_snapshot_id`; `observed_head` is diagnostic only.
Production receipt tree binding remains unchanged: this fixed-point identity is
only for the self-containing checked evidence envelope.

## 12. Parallel implementation program

### 12.1 Advisory workstreams

`Parallel lane` task metadata names an ownership workstream; it is not dispatch
authority. The configured supervisor dispatches through three strict shards
computed from the trailing decimal task-ID suffix modulo three, plus exact
file/resource claims. `Allow concurrent with` remains empty; readiness comes
only from the dependency DAG and conflicts are fenced by exact predicted-file/
resource claims.

| Workstream | Ownership | Tasks |
| --- | --- | --- |
| contracts | schemas, identities, package boundary | IVP-001 |
| datasets | semantic input adapter | IVP-002 |
| storage | local/ipfs-kit store and cache | IVP-003, IVP-008 |
| runtime | admitted process runner | IVP-004 |
| adapters | pytest/mypy/provers | IVP-005, IVP-006, IVP-007 |
| selection | semantic selection and plan creation | IVP-009, IVP-010 |
| diagnostics | counterexamples and summaries | IVP-011, IVP-013 |
| routing | provider-neutral model route | IVP-012 |
| orchestration | execute plan and bundle results | IVP-014 |
| evaluation | differential fixtures, conformance, benchmark | IVP-015–IVP-017 |
| release | docs/report/public surface, lint repair, and fixed-point evidence | IVP-018, IVP-020, IVP-021, IVP-019 |

Actual strict shard assignment is:

- shard 0: IVP-000, IVP-003, IVP-006, IVP-009, IVP-012, IVP-015,
  IVP-018, IVP-021;
- shard 1: IVP-001, IVP-004, IVP-007, IVP-010, IVP-013, IVP-016,
  IVP-019;
- shard 2: IVP-002, IVP-005, IVP-008, IVP-011, IVP-014, IVP-017,
  IVP-020.

### 12.2 Waves

| Wave | Ready work | Fan-in |
| --- | --- | --- |
| 0 | IVP-000 | sealed planning artifacts |
| 1 | IVP-001 | IVP-000 |
| 2 | IVP-002, IVP-003, IVP-004, IVP-012, IVP-013 | IVP-001 |
| 3 | IVP-005, IVP-006, IVP-007, IVP-008, IVP-009 | declared wave-2 dependencies |
| 4 | IVP-010, IVP-011 | cache/selection and pytest adapter |
| 5 | IVP-014 | all execution collaborators |
| 6 | IVP-015 | integrated executor and selection |
| 7 | IVP-016, IVP-017 | hard conformance and benchmark evidence independently |
| 8 | IVP-018 | benchmark/report evidence even if hard conformance is red |
| 9 | IVP-020 | bounded repair of inherited Ruff debt after retry-budget exhaustion |
| 10 | IVP-021 | canonical fixed-point source-snapshot binding for benchmark/report evidence |
| 11 | IVP-019 | conformance, regenerated benchmark, rebound report, and full release gate |

Tasks have disjoint predicted files wherever possible. IVP-001 owns only the
minimal lazy package stub needed for focused contract imports. IVP-020 is a
sealed, semantics-preserving recovery step for the exact Ruff surface exposed
by terminal validation. IVP-021 replaces the self-referential Git-HEAD evidence
check with the closed source snapshot. IVP-019 alone freezes the complete
public export surface, regenerates the evidence after that export change, and
performs cross-module fan-in.

## 13. Required test matrix

The conformance suite must prove:

1. unchanged receipt is reused;
2. relevant code change invalidates it;
3. unrelated code change preserves the old immutable receipt under its old
   key but rejects it for the changed full-tree key;
4. environment change invalidates it;
5. dependency-lock change invalidates it;
6. tool-version change invalidates it;
7. stale receipt is rejected;
8. simulated receipt is rejected for production acceptance;
9. timeout remains timeout;
10. unavailable prover remains unavailable;
11. selected-test failure produces a minimized counterexample;
12. uncertain test selection triggers a broader suite;
13. concurrent cache writers cannot overwrite one another;
14. cancellation terminates child processes;
15. localized exact work selects the small-model route;
16. broad or opaque work selects the frontier route;
17. unresolved high-risk policy selects human review;
18. commitment changes whenever a required receipt changes.

Additional mutation tests cover every receipt-key component, content
corruption, wrong receipt kind, missing required field, late success after
cancellation, proof/test disagreement, unsafe argv, unbounded output, and
unknown semantic edges.

## 14. Benchmark and success criteria

For each controlled semantic-capsule fixture task, record:

- cache hit rate;
- tests selected and full tests;
- false-negative and false-positive selections;
- static checks and proof obligations executed;
- wall time and reused verification time saved;
- selected model route and frontier escalation rate;
- counterexample context bytes/tokens;
- verification tokens saved compared with bounded raw logs.

The initial release criteria are:

- zero stale receipts accepted;
- zero simulated receipts accepted as production evidence;
- zero selected-test false negatives in the controlled fixture set;
- the requested unaffected cross-tree reuse target is reported unmet because
  exact full-tree binding forbids it; historical receipts remain preserved;
- deterministic verification commitments;
- small-model routing for a meaningful nonzero subset of localized fixtures;
- complete, honest metrics and explicit typed unavailable/not-measured entries
  when a fixture package or real prover capability is absent.

Correctness gates are hard only in IVP-016/IVP-019. Evaluation and benchmark
tasks always emit typed measurements (including red, inconclusive, unavailable,
and `not_measured` observations) so IVP-018 can produce the required honest
report even when release criteria are missed. Performance and routing
distribution are reported without changing status semantics or suppressing
failed targets. Schemas/order are deterministic; wall time is measured with
sample count/tolerance, reused-time savings uses paired cold/hot observations
or is labeled estimated, and token savings binds the tokenizer/estimator
version and compared artifact bounds. "Meaningful subset" means at least one
localized fixture and at least 20% of measured localized fixtures.

## 15. Supervisor execution and monitoring

The board is designed for the repository's implementation supervisor:

- isolated control worktree on `integration/incremental-verification-planner-main-20260811`, based on the merged PR #176 revision;
- 3 strict deterministic shards and isolated worker worktrees;
- one serialized merge queue targeting that branch;
- planning files protected from implementation-agent edits;
- 4 attempts per task, 90-minute ordinary task timeout, 120-minute hard cap;
- 15-minute no-log stall detection, 20-minute stale heartbeat, and bounded
  automatic supervisor restarts;
- no automatic objective/codebase refill for this sealed scope;
- the bounded master drains only after the IVP lifecycle watcher observes a
  fresh terminal task projection for every shard; the operator monitor still
  audits progress, blockers, and exact process-tree cleanup.

Implementation workers use the runtime's reviewed ordered route: Grok 4.5 is
primary and Codex GPT-5.6 Terra at high reasoning is the quota-only fallback.
That sealed implementation-provider route is operational control-plane policy;
it is separate from, and gives no authority to, the provider-neutral
`ModelRouteDecision` implemented and verified by this program.

The configured launcher disables goal-completion reconciliation. Objective
statuses are planning lineage, not runtime terminality; `task_groups` and goal
evidence criteria are validator/report projections. After task drain, operator
closeout maps authoritative task/evidence results to goals without claiming
that master exit itself proved goal completion.

Monitoring must examine the master heartbeat, per-lane supervisor status,
daemon semantic heartbeat, active implementation log growth, retry/validation
receipts, worktree leases, and merge queue. A live PID alone is not progress.

Recovery policy:

1. dependency-idle with a live upstream task is healthy;
2. stale/no-log provider attempts are terminated through the supervisor's
   process-tree fencing and retried within budget;
3. validation failures retain logs/artifacts and reopen the task;
4. merge conflicts use the existing bounded merge-resolution path;
5. repeated identical failures are inspected and repaired as a scoped board or
   infrastructure issue, never hidden by marking a task complete;
6. exhausted tasks remain terminal failed/blocked with evidence and prevent
   the release fan-in.

## 16. Definition of done and final report

The module is done only when all automatic tasks are authoritatively complete,
the full required test matrix passes, differential fixtures have zero false
negatives, the benchmark artifact is generated, and the target branch is clean.

The final report must include:

- modules changed and adapters implemented;
- receipt schemas and exact cache-key definition;
- invalidation behavior;
- selected/full test and proof results;
- cache hit rates and model-route distribution;
- compact counterexample examples;
- commitment format and determinism evidence;
- limitations and unmet targets;
- the exact next step for ZK aggregation: freeze the admitted receipt leaf
  codec and trust policy, publish deterministic cross-implementation Merkle
  vectors, then add an external circuit that proves membership/aggregation over
  that root without changing ordinary verification authority.
