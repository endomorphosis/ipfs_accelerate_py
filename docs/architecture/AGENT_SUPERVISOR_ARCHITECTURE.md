# Agent Supervisor Architecture

`ipfs_accelerate_py.agent_supervisor` is the control plane for objective-driven
agent work. It turns a durable objective into evidence-backed tasks, schedules
those tasks in isolated implementation lanes, validates the resulting changes,
and records enough state for a later run to resume, repair, or audit the work.

The package is intentionally broader than an LLM wrapper. Models propose plans
and edits, but deterministic parsers, policy checks, validation commands, Git
operations, leases, and evidence receipts decide whether work may advance.

**Start here for product vocabulary:**
[Design philosophy](AGENT_SUPERVISOR_PHILOSOPHY.md) ·
[Doc hub](agent_supervisor/README.md) ·
[Package map](agent_supervisor/PACKAGE_MAP.md) ·
[Operator guide](../guides/AGENT_SUPERVISOR_GUIDE.md)

This architecture document is the **implementation map**. Prefer domain package
paths on current `main` (`agent_supervisor.proof.…`). Flat module filenames
still appear where the source tree or historical text uses them; see the
[domain package layout](#domain-package-layout) section.

## Architectural status

The supervisor currently provides executable, resumable formal planning and
Leanstral-assisted goal development. The following capabilities define the
architecture:

| Capability | Implementation (module) | Operational meaning |
| --- | --- | --- |
| Shared prover admission | `multi_prover_resources` | SMT, ATP, kernel, model-checker, protocol, hyperproperty, runtime, validation, model, and artifact-I/O work share one bounded top-level lease. Child processes inherit limits and release capacity on cancellation. |
| Adversarial evidence admission | `formal_planning_adversarial` | Plan identity, provider boundary evidence, cache freshness, conformance, public-output leakage, and property-specific assurance are checked together. Unknown, forged, stale, or insufficient evidence is rejected. |
| Plan conformance and completion | `formal_plan_conformance` | Canonical execution events are compared with the accepted plan; unauthorized, reordered, skipped, failed, overridden, or superseded transitions are retained as findings. Completion requires fresh evidence and can reopen a goal. |
| Counterexample-guided repair | `formal_replanner` | Typed, bounded repair rules produce content-addressed candidates and compact implementation packets. Retry, refinement, candidate, changed-record, and prompt budgets prevent unbounded replanning. |
| Proof-carrying execution | `proof_carrying_planner` | Compile, verify, implement, scope-check, merge, monitor, and repair nodes run as a durable DAG with paired JSON/DuckDB state. The workflow is replayable and only completes when required assurance is present. |
| Rollout measurement and gates | `formal_planning_metrics`, `formal_planning_rollout` | Cold/warm/parallel benchmark samples measure context reduction, defect detection, proof support, counterexample quality, cache reuse, queue latency, CPU, memory, and throughput before promotion. |
| Prompt bootstrap and rescue gate | `prompt_workflow`, `prompt_workflow_benchmark`, `prompt_workflow_rollout` | Frozen prompt/repository fixtures prove Markdown/DuckDB/both and Python/CLI/script/MCP parity; adversarial and chaos populations reject scope/secret/identity/SQL/process/policy/authority/completion escapes; off/shadow/assist/automatic promotion requires a later fresh-root evaluation with immediate rollback. |

Program/board evidence tags that historically labeled these capabilities are
listed in [Appendix: historical program evidence tags](#appendix-historical-program-evidence-tags).

## Stable control surface and operating model

The reviewed public control boundary is intentionally much smaller than the
complete module map in this document. Applications should build on the stable
exports from `ipfs_accelerate_py.agent_supervisor`, especially:

- `SupervisorControlService`, `SupervisorClient`, `SupervisorTarget`, and
  `RepositorySupervisorBackend`;
- `Operation`, `OperationAuthority`, `OperationRequest`, `OperationResult`,
  `OperationStatus`, `OperationError`, `ErrorCode`, and `ControlBounds`;
- `ExpectedEffect`, `EffectClaim`, `DryRunPreview`, `IdempotencyKey`, and
  `AuthorizationDecision`;
- `OperationCapability`, `CapabilityReport`, `ControlDiscoveryManifest`, and
  the request/result schema helpers; and
- `FormalVerificationProbeConfig`,
  `probe_formal_verification_capabilities`, the paired self-improvement rollout
  contracts, and their report store.

Those package exports are convenience names for transport-neutral contracts;
the implementation modules remain the contract owners. Provider-backed
planning and proof adapters are lazy package attributes. Importing the package,
the control service, the contracts, or discovery helpers must not import an
optional provider, resolve an MCP service, inspect a repository, or start a
process. Accessing a provider-specific attribute for the first time may import
that provider adapter, but it still does not establish that the provider is
available or conformant.

The paired-rollout subset has a machine-readable compatibility boundary:
`PAIRED_ROLLOUT_STABLE_EXPORTS` is an immutable, unique, complete manifest
whose members resolve lazily to their identical objects in the provider-free
owner module. Package-owned
`PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID`
(`300500866741873729474343907613893393545`) and
`PAIRED_ROLLOUT_LAZY_EXPORT_GOAL_ID` name the import-isolation
obligation without resolving rollout code. This obligation is validated in a
fresh interpreter across the complete manifest and optional-provider
inventory; a warm-process import or a partial name check is non-authoritative.

Generation 2 has a separate reviewed publication boundary.
`AGENT_SUPERVISOR_V2_STABLE_EXPORTS` (also available as the
`V2_STABLE_EXPORTS` compatibility alias) is the immutable package-root
manifest; `AGENT_SUPERVISOR_V2_EXPORT_MODULES` records the provider-free owner
module for every member. `AGENT_SUPERVISOR_V2_PUBLIC_API_VERSION` is `2`.
Only names in that manifest are public v2 contracts or control entry points.
Accessing a manifest member resolves it once and returns the canonical
owner-module object, so a package import, owner-module import, CLI adapter, and
MCP adapter cannot acquire distinct class, enum, catalog, or operation
identities.

The v2 discovery entry points are
`agent_supervisor_v2_discovery_manifest()` and
`agent_supervisor_v2_control_surface_publication()` for Python,
`agent_cli_v2_discovery_manifest()` and
`v2_cli_control_surface_publication()` for CLI, and
`agent_supervisor_v2_discovery_manifest()` and
`mcp_v2_control_surface_publication()` in the MCP adapter. They all bind the
same `OPERATION_CATALOG_V2`, canonical `Operation` members, schema identities,
behavior identities, direct-service dispatcher identity, and catalog version.
Discovery is static: it must not resolve a backend or service, read
environment allowlists, import an optional dataset/model/prover provider, or
start even a short-lived process. A runtime `capabilities` request is a later,
separate read.

The existing package exports, discovery spellings, request/result records,
operation values, CLI command names, and MCP tool names remain the v1
compatibility surface. Generation 2 extends their metadata and conformance
checks; it does not renumber an operation, replace its enum object, or turn a
legacy import into a provider load. A new name that cannot preserve those
canonical identities is not admitted to the stable manifest.

Availability is therefore an explicit two-step handshake:

1. use the control discovery manifest or capability report to learn the closed
   operation vocabulary, schema identities, authority class, bounds, and
   dry-run/idempotency/authorization requirements; then
2. run the relevant capability probe or provider-matrix self-test before
   admitting work that depends on an optional model, solver, kernel, protocol,
   hyperproperty, or dataset route.

Import success is never a capability signal, and a capability signal is never
proof evidence. A deployment should retain the capability snapshot identity
alongside plans, cache keys, rollout reports, and self-refill epochs so that a
changed provider environment invalidates the appropriate decisions.

### One contract across Python, CLI, and MCP

`control_contracts.py` defines a closed operation vocabulary. The operations
are divided by maximum semantic authority:

| Authority | Operations |
| --- | --- |
| Read | `capabilities`, `status`, `health`, `metrics`, `goals`, `tasks`, `bundles`, `lanes`, `events`, `receipts`, `cache_inspect`, `artifact_query` |
| Proposal | `objective_preview`, `plan` |
| Mutation | `objective_refine`, `objective_reconcile`, `backlog_refill`, `start`, `pause`, `resume`, `drain`, `stop`, `retry`, `cancel`, `quarantine`, `validation_replay` |

Python calls `SupervisorControlService.execute(OperationRequest(...))`. The
unified CLI exposes the same operations under `ipfs-accelerate agent`; the
short CLI spellings `cache`, `artifact`, `preview`, `refine`, `reconcile`,
`refill`, and `validation-replay` decode to their canonical enum values. MCP
registers one tool per canonical operation in the lazily loaded
`agent_supervisor` category. Each MCP tool accepts an `OperationRequest` under
its `request` field and returns the canonical `OperationResult`.

All adapters decode the request before resolving a service and dispatch
directly to `SupervisorControlService`; neither CLI nor MCP shells out to a
standalone supervisor script. The service owns allowlists, target and tree
identity checks, bounds, authorization freshness, lease/fence validation,
idempotency, effect validation, redaction, stable errors, and audit receipts.
An adapter must validate the returned result against the original request
before presenting it as canonical output.

Discovery has a stronger no-side-effect rule than ordinary read execution.
Python, CLI, and MCP discovery manifests cover the same operation population
and canonical request/result schema identities. CLI parser construction is
static. MCP tool registration is static, and listing its category, tools, or
schemas does not resolve the configured service. Repeated discovery must be
byte-deterministic and leave optional-provider imports, process starts, backend
dispatches, and service-resolution counters unchanged.

The generation-2 catalog makes that parity mechanically reviewable. Each
`ControlOperationDescriptor` in `OPERATION_CATALOG_V2` fixes the operation's
authority, target selectors and roots, request/result schemas, bounds,
pagination, backend capability, degradation policy, audit schema, and
dry-run/idempotency/authorization/lease/fence requirements. A
`ControlSurfacePublication` qualifies only when it covers the complete catalog
with the canonical schema and behavior identities and dispatches directly to
the service. Partial CLI or MCP registration fails before publication.

The shared operation contract provides behavioral parity, but it does not make
every backend universally available. `RepositorySupervisorBackend` supplies
bounded adapters for existing status, objective, task-board, event, cache, and
artifact data. A deployment must explicitly register any mutating backend
operation it intends to allow; otherwise that operation returns the stable
`unavailable` result.

### Authorization and mutation boundary

Every request binds absolute repository and state roots plus repository, tree,
objective, objective-revision, policy, policy-revision, and caller identities.
The service accepts only roots in its constructor-supplied allowlists. Read and
proposal operations remain structurally bounded but cannot claim mutation
effects.

A real mutation must satisfy all of these conditions before its backend can be
called:

1. it declares the complete expected-effect set, and every path is contained
   by the selected root;
2. an unexpired permit binds the operation, caller, roots, identities, lease,
   fence, and authorized effect IDs;
3. an idempotency key is scoped to the same operation, caller, repository, and
   objective;
4. the repository/tree identity is current;
5. a live lease validator accepts the lease ID and fencing epoch; and
6. the backend applies only declared effects and a durable, redacted audit
   receipt can be written.

Failure at any preflight step prevents dispatch. Reusing an idempotency key for
a different request is a conflict. Replaying the exact accepted request
returns its original result without applying the backend effect again. A
mutation with `dry_run=True` never invokes the mutating adapter, has proposal
authority, and returns a `DryRunPreview` rather than applied effect claims.

The local CLI service is scoped to the explicit roots in its request, but this
is containment, not authorization: a real mutation still fails closed without
the required permit and a configured live lease/fence validator. MCP is
stricter at configuration time. Its repository and state allowlists must come
from server configuration or the dedicated allowlist environment variables;
it never derives server authority from tool input. Production embedding should
configure a reviewed service or service factory rather than depend on the
environment fallback.

### Objective, task-board, and lifecycle operations

The control API intentionally separates observation, proposal, and mutation:

| Operator need | Read or preview first | Authorized operation |
| --- | --- | --- |
| Inspect objective state | `goals`, `receipts`, `artifact_query` | `objective_reconcile` |
| Develop a goal | `objective_preview`, `plan`, `health` | `objective_refine` |
| Inspect the board | `tasks`, `bundles`, `lanes`, `status` | `backlog_refill`, `retry`, `cancel`, `quarantine` |
| Operate workers | `status`, `health`, `events`, `metrics` | `start`, `pause`, `resume`, `drain`, `stop` |
| Recheck accepted work | `receipts`, `artifact_query` | `validation_replay` |

For example, `ipfs-accelerate agent goals ...`,
`ipfs-accelerate agent tasks ...`, and
`ipfs-accelerate agent preview ...` are bounded reads/proposals.
`ipfs-accelerate agent reconcile ...` and
`ipfs-accelerate agent refill ...` are real mutations unless `--dry-run` is
present, so they need the complete canonical authorization, idempotency,
lease/fence, and expected-effect bindings. CLI output is JSON; a bounded watch
is available only for `status`, `health`, `metrics`, and `events`. MCP clients
perform the identical operation through the same canonical records instead of
translating the CLI example into a command string.

Lifecycle commands are transitions over the service's closed supervisor state
machine. `start`, `pause`, `resume`, `drain`, and `stop` do not mean “run this
shell command”; the backend checks the authoritative current state and returns
`invalid_lifecycle_transition` for an illegal edge. Target, requested state,
reason, dry-run flag, and the containing request must agree.

### Reviewed operating profiles

Profiles are versioned configuration-and-evidence envelopes, not Python
classes, implicit modes, or folklore about worker counts. A profile binds
`ControlBounds`, context/cache limits, `ResourcePolicy`,
`ResourceLeaseBudget`, provider capacity, rollout policy, and daemon settings
to the repository/tree, policy, capability snapshot, host class, and measured
workload. Neither a profile name nor a configured worker maximum grants
capacity or authority.

For each admitted resource class, measure CPU time and saturation, peak RSS and
GPU memory, process count, temporary and durable bytes, model tokens and
provider quota, provider latency, queue delay, validation/merge pressure, and
accepted throughput on the applicable v2 fixtures. The effective ceiling is
the minimum of the contract bound, policy bound, current provider report,
backend bound, and the measured host limit after the reviewed reserve. Persist
the observation window, sample population, percentile/high-watermark method,
reserve, and resulting ceiling in the profile revision. Missing or stale
telemetry produces zero new capacity. Lane count is only the result of this
admission calculation.

The immutable contract ceilings are independent of host sizing:

| Contract quantity | V2 ceiling or gate |
| --- | --- |
| One control request/result envelope | `ControlBounds`: 256 items, 256 KiB serialized, depth 8, 8 KiB text, 128 paths, 64 effects, 30 seconds; callers may only lower these |
| One v2 component receipt / projection | 256 KiB / 1 MiB |
| One refill epoch | At most 8 goals and 24 tasks, at least 6 hours cooldown; policy and measured capacity may lower both counts |
| Drained-board observation | At least 10 minutes, at most 2,000 milli-percent idle CPU, and zero unchanged-state writes |
| Safety and control | Zero authority, escaped-defect, stale-authoritative-hit, idempotency, resource-bound, and control-surface conformance violations |

The reviewed profiles are:

| Profile | Admission and measured ceiling | Required outcome |
| --- | --- | --- |
| Smoke | One serialized lane is chosen for reproducibility, not as a host-capacity claim. Use temporary state, deterministic local fixtures, the smaller 2,048-token/64-KiB context envelope, and disabled optional/network providers. | Complete cold import/discovery, every canonical operation/schema identity, stable errors, restart, exact replay, and a shadow fixture without repository mutation. |
| Production | Start in `shadow`. Derive per-class concurrency and context/cache quotas from a representative cold/warm/broad/parallel/artifact-pressure load run on the deployment host; admit only within the persisted CPU/RSS/GPU/disk/process/token/quota/latency reserve. | Zero hard-gate violations and current capability/conformance receipts. Promotion proceeds through `assist`; `automatic` remains capability- and policy-specific. |
| Distributed | Measure each worker class and provider route separately, then enforce both per-node ceilings and a global lease budget. Conflict-path exclusions, fencing, quota, and persistence throughput can reduce the sum of node capacities. | Independent-lane throughput passes without duplicate compute, conflict regression, stale publication, or resource-bound violations; partitions and stale fences fail closed. |
| Degraded | Recompute capacity after a provider, dataset, prover, network route, or host resource disappears. The missing route has capacity zero. Only catalog-declared `local_read_only` or `proposal_only` fallback is available; mutation and undeclared fallbacks are `fail_closed`. | Capability and reason codes identify the loss. No cached success, import success, or alternate provider increases authority. |
| Recovery | Pause admission, preserve journals/receipts/leases, and measure the restart/replay workload against a dedicated recovery reserve. New-work capacity remains zero until tree, state, fence, and last terminal receipt reconcile. | Restart fixture passes, exact successful mutations replay without another effect, incomplete work is bounded, and recovery produces no false completion. |
| Refill | Run after a drained-board observation and a meaningful binding change or eligible scheduled window. Bound the epoch below both the 8-goal/24-task contract maxima and the measured validation/materialization capacity. | Exact epoch replay is a no-op, duplicate-successor rate is zero, cooldown/healthy-exhaustion guards hold, and proposal/partial evidence cannot create work. |
| Rollback | Trigger on a stale binding, regression, failed current-tree evaluation, or lost capability. Set affected v2 behavior to `shadow`, stop new automatic admission, and drain or quarantine within the measured rollback reserve. | `rollback_applied` and bounded reason codes are persisted; no report, rollback, or shadow execution mutates objectives, policy, code, or completion state. |
| Migration | Run v1 and v2 reads/discovery side by side in smoke/shadow with mutation capacity zero. Measure schema/result parity, state growth, restart time, and operator load before authorizing one mutation family at a time. | V1 imports and operation identities remain valid; v2 uses the canonical catalog objects; any parity or recovery drift returns the migrated family to the v1 path or v2 shadow. |

The smoke limits are deterministic test choices. Production and distributed
limits must never be copied from them or from an example lane count. Increasing
context, cache, process, provider, or concurrency ceilings changes the profile
identity and requires a new measured shadow population.

Context, cache, and resources have independent invariants:

- A `context_contracts.ContextBudgetResolution` computes usable input as the minimum of the
  supervisor ceiling, provider input ceiling, and provider context window
  after output/tool reserves. Zero fails closed. The capsule includes the
  invariant core first, selected evidence second, and on-demand references
  only when requested.
- Cache namespaces are separated by trust and purpose. A cache key binds every
  relevant tree/blob, objective, query, analyzer/provider, configuration,
  policy, scope, and capability dimension. Negative, failed, timed-out,
  truncated, stale, and inconclusive hits are short-lived diagnostics and
  cannot satisfy completion.
- A worker count is only a ceiling. `ResourcePolicy` and
  `ResourceLeaseBudget` bound top-level and child concurrency, process count,
  time, memory, GPU memory, disk, model tokens, quota, context, and provider
  latency. Proof, model, artifact-I/O, validation, merge, and persistence
  classes remain independently accountable.

### Rollout authority

The self-improvement integration uses the public
`SelfImprovementRolloutMode` vocabulary:

```text
shadow -> assist -> automatic
```

`shadow` is the default and may persist bounded metrics, candidate receipts,
and rollout reports, but cannot mutate canonical objectives, task boards,
implementation trees, or completion state. `assist` may present proposals for
operator-authorized execution; it does not inherit mutation authority.
`automatic` means policy-approved automatic use, not unrestricted autonomy. It
still executes each mutation through the normal authorization, identity,
lease/fence, expected-effect, idempotency, validation, and audit gates.

The paired rollout report covers the complete cold, warm, broad-goal,
contradictory, malformed-output, stale-cache, provider-unavailable,
independent-parallel, conflicting-parallel, failed-validation, restart, and
drained-refill fixture population. Promotion requires both the non-negotiable
safety gate and the paired improvement gate. Missing fixtures, false
completion, authority violation, stale authoritative cache reuse, escaped
seeded defects, duplicate execution, unauthorized mutation, unstable restart
state, or a failed quality/performance threshold forces the effective mode
back to `shadow`. The report is rollout evidence, not goal-completion evidence.

The runtime binds two stable objective terms to that report. Term
`109590900757783560279417463762322084165` proves the safety/shadow invariant:
every fixture in the seeded population has zero candidate false completions,
and any seeded false completion makes the non-negotiable gate fail and the
effective mode `shadow`. Term
`146189916032404266364029134505159070240` proves paired efficiency only when
all token, repeated-cache, planning, and independent-lane throughput gates
pass. The planning component is disjunctive but cannot be skipped: it requires
either at least a 1,000-basis-point median evidence-coverage improvement or at
least a 2,000-basis-point aggregate invalid-plan-branch reduction.

`PairedRolloutReport.evidence_for(requirement_id, repository_id=...,
repository_tree=...)` exposes the bounded, criterion-specific typed projection
used for those bindings. It derives the paired-efficiency requirement terms
internally, returns a negative diagnostic for an unmet supported term, and
rejects an unsupported term. `PairedRolloutRequirementEvidence.from_dict`
requires the typed source report and re-derives the complete claim, rejecting
changed, detached, or unknown data. The witness never converts a report into
completion or mutation authority. Its identities, measurements, and stable
reason codes are sufficient for diagnostics; raw prompts, model outputs,
patches, proofs, cache values, and artifact bodies remain outside the report
boundary.

Report version 2 adds the invalid-plan-branch count and explicit token, cache,
planning, and throughput projections. Version-1 reports remain recomputable
audit records, but cannot satisfy the paired-efficiency witness; promotion
requires a fresh version-2 shadow evaluation.

The paired report types, schema versions, requirement identifiers, store, and
evaluator are enumerated by the package-root
`PAIRED_ROLLOUT_STABLE_EXPORTS` compatibility manifest. Cold import, manifest
inspection, and static control discovery do not load optional providers or
start processes; first access to a listed name loads only the provider-free
rollout contracts. The isolated **import-isolation preflight** checks cold
state, exact `__all__`/manifest equality, owner identity for every member, and
absence of all registered and dataset-backed optional providers. It is
public-interface evidence, not a third
`PairedRolloutRequirementEvidence`, because it is independent of paired
fixture measurements.

Smoke operators run that import-isolation preflight before exercising every
seeded forced-shadow path with fixed inputs and bounded state. Production
operators run the same preflight for the deployed package/provider inventory,
retain the two report evidence projections with the current tree, objective,
policy, capability, and profile identities, and return to shadow when any
package, manifest, provider, or report binding changes.

Completion authority remains separate. The **paired-rollout completion
adapter** fixes terminal producers for the reviewed rollout criteria, verified
direct descendants of the paired-efficiency and import-isolation obligations,
the five literal rollout/adoption criteria, and a two-receipt exhaustive
quorum. It recomputes a fresh complete paired report and restores both
current-tree report projections as operational prerequisites, but never treats
them, the import preflight, documentation, or analyzer output as passing
criterion validation. Every criterion has one fresh passing receipt
bound by exact implementation coverage; the analyzer is explicitly healthy,
completion-safe, and fully bound; and two fresh healthy exhaustive members are
independent by member, channel, and receipt identity. Missing or invalid
producers, descendants, validations, coverage, health, or quorum keep the goal
actionable. Even a fully closed pass moves `active` only to
`provisionally_complete`; a later evaluation must revalidate the entire packet
before `verified_complete`.

Other subsystems use related but deliberately distinct vocabularies. Formal
planning has `shadow`, `canary`, and `enforcement`; Leanstral goal development
has `off`, `shadow`, `assist`, `repair_only`, and `auto_safe`. A top-level
automatic decision does not silently translate into either `enforcement` or
`auto_safe`: the downstream subsystem must also pass its own policy,
capability, proof, and freshness gate.

Generation-2 rollout is likewise a distinct contract:
`V2RolloutMode` is `off`, `shadow`, `assist`, or `automatic`. Its default
policy omits `automatic`; explicit policy approval, a qualifying complete v2
evaluation, and a separate later current-tree evaluation are all required.
`evaluate_v2_self_improvement_rollout()` returns the effective mode and
`rollback_applied`. Any stale binding, current regression, or failed
current-tree check returns the affected behavior to `shadow`. Replaying or
verifying a v2 report recomputes it from source evaluations and cannot grant
control mutation or goal-completion authority.

#### Proof-directed decision-runtime rollout

The proof-runtime rollout is a third, deliberately distinct contract.
`DecisionRuntimeRolloutMode` is `off`, `shadow`, `assist`, or `automatic`;
these values do not extend `DecisionRuntimeMode`. Only a completely gated
`automatic` decision maps to runtime `enforce`. `assist` remains runtime
`shadow` plus an operator-reviewed proposal, and neither mode manufactures
mutation or completion authority.

`DecisionRuntimeBenchmark` owns one closed paired population. It sends the
same frozen canonical decisions through the current and proof-directed live
paths, then independently increases each irrelevant legal corpus, codebase,
SkillCenter row set, SkillCenter graph, and conversation history by at least
10x. `ProofDependencyScalingReport` is replayed from the complete producer
receipt population and causal ablations. It recomputes provider tokens,
mandatory proof closure nodes/bytes, total-corpus nodes/bytes, exact warm
cache reuse, invalidation true/false positives and false negatives,
first-valid plans, retries, proof and validation cost, declared/observed
effects, and terminal results. Aggregate averages, estimates, model judgments,
or a narrowed fixture population cannot qualify promotion.

For the proof-directed path, increasing irrelevant corpus changes only bounded
index metadata. The decision, mandatory closure identity and size, effects,
terminal result, and provider input context remain unchanged. Across decisions,
context grows with the mandatory proof closure rather than total corpus size.
Retrieval can nominate context but cannot manufacture authority; mandatory
dependencies never compete for optional budget or truncate.

The non-compensable adversarial population requires zero forged-CID,
canonicalization, schema, stale-root, cross-partition, prompt-injection,
poisoned-embedding, inapplicable-law, legal-conflict, SecurityIR deny/unknown,
intent-authority-confusion, dirty-file, changed-tool-argument, stale-lease,
proof-replay, graph-truncation, recovery, path/effect escape, or
mandatory-omission escapes. Intent, LegalIR, and SecurityIR verdicts remain
independent. Deterministic local degraded operation is fail closed when
optional model, dataset, graph/vector, or prover facilities are unavailable;
static discovery remains side-effect free and resolves none of them.

`evaluate_decision_runtime_rollout()` requires an explicitly approving
`DecisionRuntimeRolloutPolicy` for assist or automatic. Automatic additionally
requires a distinct producer population observed later against the exact
current root; replaying the qualification observation is rejected. Any
binding, safety, population, or configured metric regression returns only the
affected behavior to `shadow` and records both evaluation identities and
bounded reason codes.

`DecisionRuntimePublicAPI` exposes the same canonical request/result identity
for Python, CLI-shaped, and MCP-shaped calls. All three support `off`,
`shadow`, `assist`, policy-approved `automatic`, `status`, `explanation`, and
`rollback`; the aliases dispatch one implementation and cannot add authority.
The benchmark and rollout modules publish
`PROOF_DIRECTED_ROLLOUT_REQUIREMENT_ID` evidence, not a permit, mutation, or
completion receipt.

#### Prompt bootstrap and rescue rollout

`prompt_workflow_benchmark.py` and `prompt_workflow_rollout.py` own the
**prompt bootstrap and rescue rollout gate**. One frozen prompt/repository
population must produce identical admitted task CIDs, ready sets, accepted
effects, and terminal outcomes across deterministic and model planning,
Markdown/DuckDB/both projections, and Python/CLI/script/MCP surfaces. The adversarial population
covers prompt and repository injection, path/symlink escape, secret leakage,
forged CID, schema downgrade, SQL injection, PID reuse, process escape, policy
weakening, authorization and permit forgery, completion forgery, mandatory
evidence omission, stale preview, cross-repository replay, and shell rescue
proposals. Chaos receipts inject crashes before and after every
materialization, lifecycle, and rescue intent/effect/receipt boundary and
require resume, compensate, or quarantine without escape.

`PromptWorkflowRolloutMode` is `off`, `shadow`, `assist`, or `automatic`.
Automatic requires an approving policy for
`behavior:prompt-workflow-bootstrap-rescue@1` and a later distinct current-root
evaluation; binding, safety, population, or metric regression returns only the
affected behavior to `shadow`. `PROMPT_WORKFLOW_ROLLOUT_REQUIREMENT_ID`
identifies the evidence. The report is never a mutation permit or completion
receipt. A separate fresh-root evaluation is required before automatic
promotion.

### Metrics, failure recovery, and self-refill epochs

Operators should correlate canonical identities rather than scrape prose.
`metrics`, `status`, `health`, `events`, and `receipts` expose the control
projection. Supervisor efficiency receipts join stage latency, input/output
and reused tokens, cache outcomes, queue delay, retries, validation cost,
changed scope, and acceptance. Scheduler metrics add admission waits,
utilization, provider pressure, lane throughput, and resource high-watermarks.
Proof and rollout projections report attempt outcomes, assurance, cache
freshness, defect detection, authority violations, restart stability, and
promotion reason codes. Metrics are bounded, low-cardinality projections;
full outputs belong in content-addressed artifacts.

Proof-runtime projections additionally report producer input/output/reused
tokens, mandatory-closure and total-corpus nodes/bytes, exact warm reuse,
invalidation true positives/false positives/false negatives, first-valid plan
count, retries, proof/validation cost, effect parity, and terminal parity.
Invalidation precision means every and only affected transitive dependent, not
a cache hit rate. The report is content-addressed and re-derived from producer
receipts after restart; a stale, corrupt, detached, or replayed report cannot
restore automatic mode.

Failures retain enough identity and evidence for deterministic recovery:

- contract, authorization, bounds, stale-tree, stale-lease, idempotency,
  unavailable-provider, timeout, cancellation, conflict, and lifecycle errors
  have stable codes and redacted audit receipts;
- an exact successful mutation replay returns its persisted result without
  duplicating the backend effect, while conflicting replay fails;
- watchdogs use heartbeat, process liveness, active phase, and retry budgets to
  restart or block work without granting a stale worker publication authority;
- corrupt cache entries are rejected and recomputed, never promoted; torn
  projections recover from append-only journals or authoritative receipts;
- merge conflicts and failed validation become bounded repair work with their
  original commands and evidence; and
- paired rollout reports are content-addressed, append-only, fsynced, bounded,
  symlink-resistant, and recomputed from fixture evidence after restart.

When the actionable task board drains, `run_self_improvement_epoch` evaluates a
bounded epoch rather than looping the refill command. Its
`SelfImprovementEpochBinding` includes repository/tree, objective revision,
task-board revision, self-improvement policy, capability snapshot, observation
window, and operator revision. The content identity of those inputs is the
epoch ID.

The epoch reconciles goals, evaluates the complete benchmark population,
classifies gaps, validates bounded novel successor candidates, and either
transactionally materializes admitted work or records healthy exhaustion.
Replaying an identical persisted epoch produces replay evidence and performs
no second objective/task-board mutation. A failed, partial, blocked, or
inconclusive analyzer cannot create work. Healthy exhaustion requires
independent fresh exhaustive receipts and enters a wait state; another epoch
requires a meaningful trigger such as a changed tree, policy, capability
snapshot, objective revision, regression, stale evidence, cooldown expiry, or
scheduled observation window. A drained board alone is neither permission to
manufacture work nor evidence that the root objective is complete.

### Migration boundary for standalone scripts

Legacy objective, daemon, and supervisor scripts remain project-binding
adapters during migration. New integrations should replace direct script
imports, subprocess calls, ad hoc JSON, and private daemon mutation with:

1. stable package contracts and the `SupervisorControlService` for Python;
2. `ipfs-accelerate agent` for operator and process-boundary use; or
3. the `agent_supervisor` MCP category for policy-configured remote control.

Migrate reads first and compare canonical results across surfaces. Next move
preview/dry-run paths and record schema identities. Finally move mutations one
operation at a time after supplying explicit roots, authorization, idempotency,
lease/fence validators, expected effects, and durable audit storage. Existing
runtime runners may still construct the backend, but they must not maintain a
second authorization or lifecycle policy. A compatibility wrapper should
decode into `OperationRequest`, call the service once, and return
`OperationResult`; it should not translate back into a shell command.

For a v1-to-v2 migration, retain both compatibility manifests while comparing
`agent_supervisor_v2_discovery_manifest()` with the legacy discovery result.
Negotiate catalog version 2 explicitly, require the same canonical `Operation`
objects and request/result schema identities on Python, CLI, and MCP, and keep
v2 in smoke/shadow with mutation admission at zero. Move one operation family
only after its dual-read or dry-run results, measured resource ceiling,
restart/replay behavior, and audit receipts agree. Roll the family back to the
v1 adapter or v2 shadow on any identity, capability, state, or recovery drift.
Do not rewrite persisted v1 reports as v2 records: retain them for audit and
produce fresh v2 evidence on the current binding.

## Architectural synthesis: a constrained feedback controller

The recent implementation completes the first integrated slice of a
proof-aware supervisor. Its central abstraction is a feedback controller:

```text
frozen intent + policy + capacity
              |
              v
  typed plan and proposal refinement
              |
              v
  bounded execution and independent checks
              |
              v
 observations, receipts, counterexamples, and health
              |
              +--> reconcile, repair, reopen, or complete
```

This model gives each subsystem a distinct responsibility:

| Layer | Responsibility | What it cannot do |
| --- | --- | --- |
| Intent | Freeze the root goal, assumptions, scope, vocabulary, and acceptance criteria | Accept a model suggestion as a new requirement |
| Planning | Compile typed goals, subgoals, dependencies, effects, and evidence obligations | Claim that a plan proves the implementation |
| Proposal | Use Leanstral or another model to suggest bounded refinements | Mutate canonical objectives, grant authority, or assert completion |
| Assurance | Route each obligation to semantically appropriate checks and retain provenance | Treat a solver candidate, stale receipt, or unsupported translation as proof |
| Execution | Admit leased work, validate changed scope, merge, and recover from interruption | Publish with a stale fence, missing evidence, or failed policy gate |
| Reconciliation | Compare observed events and fresh evidence with the accepted plan | Infer success from an empty queue or a process exit code alone |

The design follows several computer-science principles. The objective, task,
and evidence graphs are separate projections because planning, scheduling, and
trust answer different questions. Leases and fencing tokens provide the
distributed-systems safety property that stale workers cannot publish after
ownership changes. Canonical content identities make receipts a form of
memoization whose validity is conditional on the exact tree, policy,
translation, toolchain, and bounds. Typed states form a monotone trust lattice:
new observations may strengthen or invalidate a claim, but a proposal cannot
skip directly to authoritative assurance. Bounded retries and explicit
terminal states provide a termination argument for repair loops.

Parallelism is therefore admission control, not simply a worker count.
Independent DAG nodes may run concurrently only when their scopes, leases,
provider capabilities, and host budgets permit it. Cache hits remove work but
do not remove freshness and trust checks. A counterexample cancels redundant
portfolio work, while cancellation preserves a partial receipt and releases
capacity. These rules let throughput increase without weakening causality or
making completion nondeterministic.

Leanstral belongs at the proposal/refinement boundary. It can compress
repository evidence into candidate subgoals, suggest repairs, and expose
alternative decompositions. Deterministic schema checks, typed plan
validation, property-specific provers, code-conformance checks, and merge
policy remain the acceptance boundary. The default rollout is shadow mode;
assist and auto-safe require measured paired improvement plus zero false
completions, zero authority violations, stable restart recovery, and explicit
policy authorization.

These modules provide the execution surface for the design below; they do not
make arbitrary Python formally verified. Provider conformance, reviewed
obligation templates, exact tree and policy identities, and the configured
assurance threshold remain prerequisites for enforcement.

This document is an orientation guide to the implementation. The more detailed
objective scanner description is in
[`agent_supervisor_objective_graph.md`](../agent_supervisor_objective_graph.md),
and the formal-planning and prover design is in
[`AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md`](AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md)
and
[`AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md`](AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md).

## Design goals

The supervisor is designed around six constraints:

1. **Objectives are durable.** A Markdown objective heap is a human-readable
   source of intent; generated todos and derived graphs are projections, not a
   replacement for that intent.
2. **Work is reproducible.** Task identities, plans, commands, artifacts, and
   state transitions use canonical JSON, content identities, and versioned
   schemas where possible.
3. **Lanes are isolated.** Parallel bundles use separate state directories and
   worktrees. A worker should not accidentally share mutable task state or
   uncommitted files with another worker.
4. **Evidence is explicit.** A passing command, proof receipt, scan result, or
   merge event is recorded with provenance and freshness rather than inferred
   from a model's prose.
5. **Failure becomes work.** Repeated implementation, validation, or merge
   failures are bounded by policy and converted into repair or follow-up tasks.
6. **Assurance is scoped.** A bounded model check, runtime trace, solver
   candidate, and kernel-checked theorem are different kinds of evidence; none
   is silently promoted to another.

## Domain package layout

On current `main`, modules are grouped into domain packages with an acyclic
dependency DAG. New code should import from these packages, not from retired
flat paths.

```text
core
  ↑
control, task_sources, context, analysis, proof
  ↑
objectives, planning, validation, prompt
  ↑
merge, rescue, runtime, self_improvement
  ↑
todo_daemon (implementation), integrations
```

| Package | Owns (summary) |
| --- | --- |
| `core/` | Shared identity, conflict graph, wrappers, external completion |
| `control/` | Transport-neutral operations, CLI/MCP contracts, lifecycle, permits |
| `task_sources/` | Markdown/DuckDB boards, queues, task identity, indexes |
| `context/` | Context compiler, decision runtime, capsules |
| `prompt/` | Prompt workflow, scanning, plan admission |
| `analysis/` | Analysis pipeline, AST, cache, retrieval, consensus |
| `proof/` | Formal verification, provers, proof cache, attestation, codebase-proof |
| `objectives/` | Objective heap, daemon, tracker, janitor, goal quality |
| `planning/` | Adaptive/formal plan compile and validate |
| `validation/` | Proposal validation and validation schedulers |
| `merge/` | Merge queue/train/resolver, locks, leases, git hygiene |
| `rescue/` | Rescue/recovery orchestrators and diagnostics |
| `runtime/` | Multi-supervisor runners, event log, CAS, schedulers |
| `self_improvement/` | Epoch contracts, refill, v2 efficiency surfaces |
| `integrations/` | Optional LLM/merge/goose/dataset adapters |
| `todo_daemon/` | Implementation daemon and supervisor process loops |

Human-oriented detail: [Package map](agent_supervisor/PACKAGE_MAP.md) and
[package README index](agent_supervisor/packages/README.md).

## System view

The main flow is:

```text
objective heap / operator request
              |
              v
   objective tracker + graph scanner   (objectives/, analysis/)
              |
              +--> AST, path, text, and vector evidence datasets
              +--> objective graph and bundle index
              v
     todo shards / task-queue payloads   (task_sources/)
              |
              v
 bundle / multi supervisor -> implementation supervisor -> implementation daemon
         (runtime/)                 (todo_daemon/)              (todo_daemon/)
                              |                         |
                              |                         +--> isolated worktree
                              |                         +--> model proposal
                              |                         +--> validation workspace
                              |                         +--> accepted/failed artifacts
                              v
                 proof, policy, lease, and resource gates
                 (proof/, validation/, planning/, merge/)
                              |
                              v
                 merge train / conflict repair / receipts
                              |
                              v
       completion decision, metrics, event log, and next scan
```

The flow is iterative. A completed task can produce new evidence for its goal;
a failed task can produce a bounded repair task; and a reopened goal can return
to the active queue. No component should assume that one pass proves the whole
repository or that an empty current backlog means the ultimate objective is
complete.

## Subsystem boundaries

### 1. Objective and goal lifecycle

`objective_tracker.py` owns the durable objective heap and compatibility with
older Markdown field spellings. `objective_graph.py` parses goals, computes
dependencies and priorities, scans repository evidence, and emits objective
findings and generated work proposals. `objective_daemon.py` is the CLI/runtime
bridge that performs a scan, writes artifacts, and optionally submits bundles
to the existing P2P task queue.

`goal_completion.py` is the lifecycle authority. Its normalized states are:

```text
active -> provisionally_complete -> verified_complete
   |             |                       |
   v             v                       v
blocked       reopened <----------------+
   ^             ^
   +-- analysis_inconclusive
```

The exact transition guards live in the module; the important concept is that
completion is evidence-sensitive and a completed goal may be reopened when
fresh evidence contradicts it or required proof becomes stale. `goal_coverage.py`
and the completion projection in `scheduler_metrics.py` reduce the state into
operator-facing coverage and health summaries.

### 2. Discovery, planning, and task identity

The scanner uses several evidence channels: tracked paths, exact text, parsed
symbols/AST records, and deterministic token-vector similarity. Large AST
payloads are stored as JSONL/dataset artifacts instead of being embedded in
todo prose. `todo_vector_index.py` records related tasks, merge keys, clusters,
and candidate relationships so a worker can select adjacent work without
reloading the entire repository.

`task_identity.py` supplies canonical task and bundle identities. These are
used for duplicate suppression, retry accounting, merge decisions, and
cross-process correlation. `conflict_graph.py` identifies file and semantic
overlap; `plan_evaluator.py` and `task_proposal_router.py` score or route
candidate plans while retaining deterministic fallback behavior.

### 3. Formal plan and policy gate

The formal planning family converts a goal/task into a typed work plan:

- `formal_planning_contracts.py` defines the plan vocabulary and assurance
  requirements.
- `formal_plan_context.py` builds bounded context from repository, task,
  resource, dependency, policy, and evidence records.
- `formal_plan_compiler.py` produces a canonical plan capsule.
- `formal_plan_validator.py` rejects malformed, contradictory, unauthorized,
  unsafe, or dependency-invalid plans.
- `logic_translation_validation.py` records what semantics were preserved or
  approximated while translating a plan into a logic representation.
- `authorization_logic.py`, `lease_coordination.py`, and
  `resource_scheduler.py` enforce authority, fencing, capacity, and admission
  rules at execution time.

The plan gate is not an assertion that arbitrary Python has been formally
verified. It is a bounded, typed check over reviewed obligations and declared
assumptions.

### 4. Prover and assurance layer

The package treats proof providers as capability-scoped adapters. The matrix in
`prover_matrix_registry.py` describes providers, identities, commands, fixtures,
and self-tests. `formal_verification_capabilities.py` probes what is actually
available on the host. `prover_conformance.py` quarantines providers whose
translation or fixture suite is not conformant.

`multi_prover_router.py` maps property kinds to prover lanes and records every
attempt. Supporting components include:

- `supervisor_state_model.py` for deterministic finite transition schemas and
  bounded TLC/Apalache model-check receipts;
- `kernel_verification.py` and `formal_verification_provider.py` for provider
  boundaries and trusted checking;
- `leanstral_proof_provider.py` for bounded, untrusted proof suggestions;
- `hyperproperty_verification.py` for non-interference-style properties;
- `proof_context.py`, `proof_scope_index.py`, and
  `code_proof_obligations.py` for scope and obligation derivation;
- `prover_evidence_store.py`, `formal_verification_cache.py`,
  `proof_attestation.py`, and `proof_metrics.py` for durable receipts,
  freshness, cache identity, and reporting.

The current planning rollout adds two gates around this layer. The adversarial
gate evaluates evidence at the property boundary and derives the authoritative
assurance from typed evidence rather than accepting a provider-declared level.
The rollout gate then compares a benchmark report with the reviewed policy in
one of three modes: `shadow` records diagnostics without changing dispatch,
`canary` allows a limited lane when thresholds pass, and `enforcement` requires
the configured minimum assurance. An expiring, content-addressed override may
waive one exact lane, but it is itself recorded as evidence and cannot weaken
the underlying trust rules.

The trust vocabulary is deliberately non-linear: `solver_candidate`,
`bounded_model_checked`, `runtime_checked`, `protocol_checked`,
`kernel_verified`, and `attested` describe different evidence types. A runtime
trace cannot substitute for a theorem, and a bounded model check must retain its
finite bounds.

### 4a. Leanstral goal-development lifecycle

`leanstral_goal_lifecycle.py` is the reviewed integration boundary for using
Leanstral to develop a frozen goal into candidate formal work. Construct it
with `build_configured_leanstral_goal_lifecycle_supervisor`; the factory
requires an explicit state directory and defaults to **shadow** mode when the
caller does not select a mode. This default is part of the safety contract, not
a deployment convention.

The lifecycle is deliberately staged:

```text
frozen root + bounded context
              |
              v
  one or more untrusted candidate drafts
              |
       schema/type/policy gate
              |
              v
   deterministic proposal receipt
              |
   independent refinement/proof portfolio
              |
              v
 objective preview or gated materialization
              |
   implementation conformance receipts
              |
              v
       goal-completion authority
```

The stages have separate trust boundaries:

- The model provider is an untrusted proposal source. It cannot issue proof,
  admission, implementation-conformance, or completion receipts. A malformed
  response, unavailable provider, wrong return type, or provider exception is
  converted into a stable fallback result rather than being treated as
  acceptance.
- `LeanstralGoalDevelopmentContextBuilder` supplies a bounded, redacted
  context. It includes only the configured number of templates, gap records,
  AST summaries, capability records, counterexamples, and reusable receipt
  summaries. Canonical source, proof bodies, secrets, and unrestricted
  repository data do not cross the model boundary. The goal root and formal
  assumptions are frozen inputs and remain frozen during counterexample repair.
- Contract validators and the deterministic proposal selector decide which
  draft can advance. Multiple candidates are isolated attempts; the stable
  selector prefers the greatest valid proposal coverage and then canonical
  draft identity. Candidate diversity never becomes voting-based proof.
- Refinement evidence is produced outside the drafting provider. A bounded
  counterexample may trigger a bounded repair round, but only an independent,
  authoritative verifier can accept the repaired formal plan. Bounded,
  assumed, unsupported, and inconclusive evidence retain those statuses.
- `objective_daemon.materialize_admitted_objective_work` is the only bridge
  from an admitted proposal to objective work. An admission receipt means that
  the policy gate allowed materialization; it is not a proof of the goal.
- `code_proof_obligations.py` binds implementation evidence to the exact goal
  root, plan, source tree, source paths, tests, and proof receipts.
  `goal_completion.py` remains the sole completion authority. Receipts bound to
  an older tree or root are stale and reopen completion instead of being
  silently reused.

#### Modes and mutation authority

| Mode | Provider and validation behavior | Permitted durable effects |
| --- | --- | --- |
| `off` | The route is disabled; no development invocation is valid. | None. |
| `shadow` (default) | Run bounded candidates, deterministic validation, independent checks supplied by the caller, and an objective materialization preview. | Audit journal, recoverable run state, and proof/operational metrics only. The objective heap, generation ledger, implementation tree, and completion state must remain byte-for-byte unchanged. |
| `assist` | Produce a reviewed proposal and admission decision for an operator. | Review/generation records allowed by objective policy; no automatic objective mutation or completion. |
| `repair_only` | Accept only a bounded repair of an existing plan, preserving the frozen goal root and assumptions. | Repair evidence and review records; no autonomous completion. |
| `auto_safe` | Apply the proposal only after deterministic validation, authoritative proof, freshness, capability, policy, and admission gates all pass. | Objective work may be materialized under the objective daemon's atomic update contract. Completion still requires separately bound implementation evidence. |

`auto_safe` must not be enabled merely because the model generated valid JSON.
It requires a fresh goal/tree binding, no undeclared assumptions or unsupported
semantics, healthy required capabilities, independent authoritative receipts,
and any configured lease/resource checks. Callers should promote through
shadow and assist using observed receipts before enabling it for a narrowly
scoped objective class.

#### Capabilities, resources, and controls

Model execution and proof execution use different capability and resource
classes. Startup should probe the configured model route, legal preprocessing
and codec support, independent verifier route, kernel checker, and any required
solver or model-checker. Effective context is the minimum safe budget after
route, server, and model reserves; a server-advertised maximum is not itself a
safe prompt budget. Provider time, output, context, concurrency, and network
limits remain explicit inputs to the bounded provider and scheduler.

The configured lifecycle bounds candidate count to one through eight and
records every candidate attempt, including fallbacks, in proof metrics. Its
state directory contains:

- an append-only, fsynced JSONL audit journal;
- an atomically replaced latest-run state document; and
- a proof metrics projection with attempt, validation, fallback, acceptance,
  latency, and availability observations.

Run records are schema-validated and content-addressed. Recovery first validates
the latest state and then scans the journal backward for the newest valid
record, so a torn or corrupt projection does not erase the audit trail. In
shadow mode every record also contains before/after digests and explicit
`objective_heap_unchanged` and `completion_state_unchanged` assertions. An
unexpected mutation is a failed run, not an advisory metric.

Receipt stores and caches retain the same separation as the live path:
proposal validation, bounded counterexamples, independent proof, admission,
implementation conformance, and completion are distinct schemas and cache
namespaces. Reuse requires exact schema, provider/version, goal-root, plan,
source-tree, policy, and capability bindings appropriate to that receipt.

### 5. Execution daemons and isolated lanes

The `todo_daemon` package contains the reusable implementation loop. Its
responsibilities are split so that policy and process supervision do not become
one opaque loop:

- `engine.py` parses tasks, materializes proposals, creates validation
  workspaces, runs commands, and promotes accepted files.
- `implementation_daemon.py` owns the task/pass state, selection, retries,
  active phases, heartbeat, LLM invocation, and validation handoff.
- `implementation_supervisor.py` and `supervisor.py` monitor heartbeats,
  process liveness, stuck phases, and restart/repair decisions.
- `lifecycle_wrapper.py`, `core.py`, and `cli.py` provide reusable process
  lifecycle and status interfaces.
- `worktrees.py`, `checkout_lock.py`, and `leased_lane.py` keep Git and lane
  mutations isolated and fenced.

`bundle_supervisor.py` converts bundle indexes into one lane per bundle;
`multi_supervisor_runner.py` manages several configured supervisor tracks. The
runner modules (`implementation_daemon_runner.py` and
`implementation_supervisor_runner.py`) are project-binding adapters: they add
defaults and hooks without duplicating the daemon engine.

### 6. Artifacts, events, and durable state

The supervisor has several intentionally different persistence surfaces:

| Surface | Purpose | Representative modules |
| --- | --- | --- |
| Objective heap | Human-readable intent and goal metadata | `objective_tracker.py` |
| Todo Markdown | Worker-facing executable queue | `taskboard_store.py`, `objective_graph.py` |
| JSON/JSONL artifacts | Bounded scan, proposal, validation, and completion records | `artifact_store.py`, `scan_receipts.py` |
| Event log | Append-only operational history | `event_log.py` |
| Task/daemon state | Resume, heartbeat, retry, active phase | `duckdb_state.py`, `todo_daemon/implementation_daemon.py` |
| DuckDB stores | Queryable task, evidence, and proof projections | `dataset_store.py`, `prover_evidence_store.py` |
| Vector/AST index | Related-work and bundle selection hints | `todo_vector_index.py` |
| Git/worktree state | Candidate changes and merge checkpoints | `merge_checkpoint.py`, `merge_queue.py` |

State files are operational projections and may be repaired or migrated. They
must not be treated as the sole source of truth for acceptance; accepted work
should have validation and provenance sidecars or receipts.

### 7. Merge, recovery, and maintenance

`merge_train.py`, `merge_queue.py`, `merge_resolver.py`,
`merge_conflict_repair.py`, and `merge_checkpoint.py` coordinate promotion of
independent lane results. `llm_merge_resolver_fallback.py` can provide a
bounded external suggestion, but Git checks and validation remain the gate.

`backlog_refinery.py` refills low or drained queues from objective gaps and
codebase findings. `objective_task_janitor.py` handles stale or inconsistent
objective work. `codex_failure_policy.py` classifies repeated failures, while
`submodule_degradation.py` records when nested repositories cannot provide their
normal validation surface. `git_gc.py` and cleanup helpers remove stale
worktrees or branches only under explicit, scoped policy.

`runtime_temporal_monitor.py`, `supervisor_watchdog.py`, `analyzer_health.py`,
and `scheduler_metrics.py` observe the live system. They report freshness,
capacity, deadlines, retries, proof invalidation, and terminal outcomes; they do
not silently mark work complete.

The proof-carrying planner composes these pieces into a durable workflow. Its
terminal result is `completed` only when plan compilation and bounded plan
validation pass, all implementation scopes and merges are accepted, the
runtime trace is accepted, and the required authoritative assurance is present.
Otherwise it returns a rejected, failed, or blocked result with replayable
decisions and evidence. A runtime counterexample is eligible for bounded repair
before finalization, and repaired counterexamples remain linked to the original
finding.

## Typical operating sequence

1. Create or update the objective heap and record acceptance criteria.
2. Run the objective daemon. Inspect `objective_graph.json`, discovery datasets,
   bundle indexes, and generated task identities.
3. Review or validate the proposed formal plan and required assurance. If a
   provider is unavailable, the result should be explicitly `unsupported` or a
   deterministic fallback—not an implicit pass.
4. Start a bundle supervisor in dry-run mode, then launch isolated lanes when
   the plan and resource budget are acceptable.
5. Let the implementation daemon select a ready task, obtain its lease, create
   a worktree, request a proposal, run validation, and persist success/failure
   sidecars.
6. Merge through the merge train. Resolve conflicts with bounded evidence and
   re-run affected validation; do not infer correctness from a clean merge.
7. Reconcile goal completion from fresh evidence. A goal with missing criteria,
   stale receipts, unhealthy analyzers, or unsatisfied exhaustion quorum remains
   open, blocked, or inconclusive.
8. Inspect event logs and metrics before starting another pass. A drained queue
   may trigger bounded objective/codebase refill. Codebase scanning remains a
   general evidence operation; a separate fail-closed admission stage can
   materialize a task only under one specific existing goal/subgoal and records
   its validated ancestor lineage. Goal records with missing status, dangling
   parents, or cycles fail closed, and a broad directory name alone is not
   semantic evidence. Refill must not create unbounded duplicate or out-of-scope
   work.

## The theory behind the design

### The supervisor is a feedback controller

The useful mental model is a feedback-control loop, not an autonomous chatbot:

```text
desired state: acceptance criteria + policy + resource budget
        |
        v
  planner / scheduler ----> worker action ----> repository + runtime
        ^                                          |
        |                                          v
        +--------- observations: tests, receipts, events, health
```

The objective and its acceptance criteria describe the desired state. The
planner selects an action that should reduce the gap. The worker performs the
action in a controlled lane. Scanners, validators, proof providers, and
watchdogs observe the result. The next reconciliation cycle uses those
observations to decide whether to continue, repair, block, or reopen the goal.

This explains several otherwise surprising choices in the code:

- **Scans are repeatable.** A controller needs comparable observations across
  cycles, so scanners use deterministic identities, bounded outputs, and
  explicit analyzer health.
- **The objective is separate from the queue.** The desired state should not be
  rewritten merely because the current actuator queue is empty or a worker
  failed.
- **Completion is a decision, not a counter.** A zero open-task count is only
  one observation. Completion also requires coverage, fresh evidence, healthy
  analyzers, and the configured exhaustion quorum.
- **Repair is feedback.** A failure is useful when it becomes a typed signal
  that changes the next plan; blind retries simply repeat the same control
  input and can create an infinite loop.

### Three graphs, three questions

The package contains several graphs because one graph cannot answer every
question:

| Graph | Question | Main representation |
| --- | --- | --- |
| Objective graph | What outcomes and evidence are still needed? | goals, subgoals, evidence terms, dependencies |
| Task/conflict graph | Which executable items can run together? | task identities, prerequisites, file/symbol conflicts, bundle clusters |
| Evidence/proof graph | Why should a result be trusted? | source/tree identities, receipts, scopes, provider bindings, freshness |

Collapsing these graphs would create two common errors. First, a task that is
easy to schedule could be mistaken for a goal that is complete. Second, a proof
receipt for one source scope could be reused for a different candidate merely
because the task title is similar. The separate projections let each graph use
the invariants appropriate to its job while sharing canonical identities.

### State machines prevent ambiguous progress

The supervisor represents progress as transitions with guards and effects. In
the formal planning vocabulary, a transition is described by:

```text
event + actor + preconditions + effects + evidence requirements
```

For example, “accept implementation” is not equivalent to “the model returned
text.” It requires a current task lease, the expected worktree/fence, a valid
proposal, successful configured validation, and accepted artifacts. The
transition then records progress and releases or advances the relevant lease.

`formal_planning_contracts.py` makes this vocabulary explicit with actors,
goals, subgoals, plan tasks, events, fluents, preconditions, effects, norms,
temporal constraints, evidence requirements, and plan assurance. The
`FormalWorkPlan` validator checks referential integrity and acyclicity before a
plan reaches a provider. `supervisor_state_model.py` translates a reviewed
finite transition schema into TLA+ and records bounded checking results.

The default state model checks safety properties such as:

- `UniqueAcceptance`: one logical task cannot be accepted twice;
- `FencingSafety`: an old worker cannot commit after its lease/fence is stale;
- `DependencyOrder`: prerequisites are accepted before dependents;
- `IdempotentMerge`: replaying a merge decision does not duplicate its effect;
- `CapacitySafety`: admitted work stays within declared capacity; and
- `EvidenceGates`: terminal transitions cannot skip required evidence.

It also checks liveness properties such as bounded progress and terminal
outcomes. These are bounded experiments over a generated finite model, not a
claim that arbitrary Python execution has been proved correct.

### Leases, fencing, and worktrees are a distributed-systems protocol

Multiple daemons may observe the same queue, and processes can pause, crash, or
be restarted. A simple `locked = true` flag is not enough: a delayed worker may
resume after another worker has taken over. The supervisor therefore combines:

1. a task lease with an owner and expiry;
2. a monotonically changing fencing token/generation;
3. heartbeats that prove the owner is still live; and
4. validation of the token at terminal operations and merge boundaries.

The invariant is: **only the current owner with the current fence may perform
state-changing work**. Expiry makes capacity reclaimable; fencing makes stale
work harmless. `lease_coordination.py` persists grants, heartbeats, terminal
receipts, and conflicts; `leased_lane.py` adapts the same idea to a lane;
`checkout_lock.py` protects Git operations; and the daemon state records the
active phase so a watchdog can distinguish slow work from a dead process.

Worktrees provide filesystem isolation, but they are not the authority by
themselves. A worktree can be deleted, reused, or left dirty. The lease/fence
protocol and event/receipt history are what make a promotion decision
auditable.

### Scheduling is admission control, not just priority sorting

Priority answers “which eligible task is preferred?” Admission answers “may
this task run at all right now?” The resource scheduler performs both concepts
separately. It normalizes host capacity, provider concurrency, process slots,
resource classes, and proof-specific pools before issuing a reclaimable
`ResourceAdmissionLease`.

This separation matters when a high-priority proof task needs a scarce kernel
slot, while several ordinary implementation tasks need only CPU. Running the
highest-priority item without admission control can starve the proof lane or
overcommit the model provider. Conversely, refusing to schedule because a
provider is temporarily full should not turn into a permanent task failure;
the scheduler records a wait/admission reason and retries when capacity is
released.

The same principle applies to LLM routing. `resource_scheduler.py` normalizes
provider capacity and `formal_verification_capabilities.py` probes operation-
specific readiness. “The executable exists” is not the same as “this provider
can perform this translation under the requested isolation and timeout.”

### Evidence is a content-addressed claim

An evidence record should answer four questions:

1. **What** was observed or checked?
2. **Where** did it come from (repository/tree/path/provider)?
3. **When** was it valid, and is it still fresh?
4. **How** can another process reproduce or verify it?

That is why receipts contain schema versions, canonical payloads, source
identity, artifact paths or CIDs, and bounded projections. `scan_receipts.py`
computes a deterministic receipt identity from canonical content and rejects a
persisted artifact whose bytes no longer match its identity. `artifact_store.py`
keeps large payloads outside status files; `event_log.py` records compact
operational facts; and proof stores bind a receipt to its scope and provider.

The distinction between a full artifact and a compact projection is deliberate.
Status and heartbeat payloads must remain bounded and low-cardinality, while an
auditor must still be able to follow the content identity to the full scan,
validation output, or proof receipt.

### Why LLM output remains below the acceptance boundary

An LLM is valuable at proposing decomposition, implementation edits, repair
strategies, or proof candidates. It is not a stable authority for repository
state. Prompts can omit context, models can hallucinate files, and a plausible
answer can violate a lease, policy, or hidden dependency.

The supervisor therefore uses a two-stage architecture:

```text
model proposal -> schema/identity checks -> policy/plan checks
               -> isolated materialization -> deterministic validation
               -> evidence and merge gates
```

`todo_daemon.llm` and provider adapters are intentionally replaceable. The
proposal is bounded, normalized, and associated with the current task and
worktree. A deterministic fallback can keep the queue moving, but fallback
usage is recorded as inconclusive or lower-assurance evidence; it is not
silently presented as a model success.

### Retry budgets are a termination argument

Retries are useful for transient network, provider, or test failures. They are
dangerous when the underlying failure is deterministic. The implementation
supervisor tracks failure classes and budgets for implementation, validation,
and merge stages. Once a class crosses its budget, the source task is blocked
and a follow-up repair task carries the diagnostic evidence.

This gives the system a practical termination property: one failing input
cannot consume all worker time forever. It also preserves information that a
blind retry would destroy—the original command, output excerpt, phase, attempt
number, and suggested repair. `backlog_refinery.py` applies the same idea to
drained queues and recurring codebase findings.

### Completion is a quorum over independent observations

The completion gate treats “done” as a conjunction of independent dimensions,
not a single boolean:

```text
complete iff
  acceptance criteria covered
  AND required tasks terminal
  AND evidence is fresh and bound to the current tree
  AND analyzers are healthy
  AND exhaustion quorum is satisfied when required
  AND no contradiction/reopen condition is present
```

The quorum is especially important for negative claims such as “no matching
implementation remains.” One scanner run can miss files because of a parser
failure, ignored submodule, stale checkout, or unavailable analyzer. The
receipt model records distinct channels and rejects duplicate or mismatched
members, so repeated identical scans do not manufacture confidence.

## Reading the implementation as a set of contracts

The most efficient way to understand a new module is to identify which
contract it owns:

| Contract type | What to inspect | Failure if violated |
| --- | --- | --- |
| Identity | `task_identity.py`, canonical JSON helpers | duplicate or cross-scope work |
| Lifecycle | `goal_completion.py`, daemon state, state model | impossible or ambiguous transitions |
| Authority | `authorization_logic.py`, actor/role records | unauthorized work or merge |
| Exclusivity | `lease_coordination.py`, fences, worktrees | stale worker mutation |
| Capacity | `resource_scheduler.py` | overcommitment or starvation |
| Semantics | formal plan contracts and translation validation | proving the wrong statement |
| Evidence | artifacts, receipts, proof stores | unverifiable completion claim |
| Recovery | failure policy, refinery, merge repair | infinite retry or lost diagnostics |
| Observation | event log, watchdog, temporal monitor, metrics | undetected stuck or stale state |

When extending the package, start with the contract and its invariant, then
locate the projection that persists it, the event that records it, and the test
that exercises it. Avoid adding a shortcut directly to a daemon loop: it will
usually bypass identity, fencing, evidence, or migration behavior.

## Worked example: closing one objective gap

Suppose the objective says “serve model discovery through the MCP endpoint.”
The intended lifecycle is:

1. The objective scanner finds that the acceptance evidence is absent and emits
   a task with a goal ID, missing-evidence terms, validation commands, and a
   canonical task identity.
2. The plan compiler adds actors, affected paths, dependencies, resource
   requirements, and an evidence requirement for an endpoint-level test.
3. The plan validator rejects the plan if its dependency graph cycles, its
   actor lacks authority, or its required provider is unavailable.
4. The bundle supervisor admits the task to an isolated lane and the daemon
   obtains a lease/fence before asking the model for an edit.
5. The daemon materializes the proposal, runs the configured tests in a clean
   validation workspace, and persists the output and command receipt.
6. The merge train checks the current fence and target tree, applies the
   accepted change, and records a merge checkpoint.
7. The objective tracker reconciles the new endpoint test and source evidence.
   If all required dimensions are fresh, the goal can become provisionally or
   verified complete; otherwise it remains active, blocked, or inconclusive.

Notice that the model is only one participant in step 4. The evidence that
closes the objective comes from the validated repository and the recorded
receipt, not from the model's assertion that the feature exists.

## What the architecture does not promise

The supervisor is a reliability and assurance control plane, not a universal
program verifier or a guarantee of eventual success. In particular:

- arbitrary Python behavior is not formally verified merely because a plan
  passed a logic check;
- a provider capability report is not proof evidence;
- a bounded model check says nothing beyond its recorded bounds and assumptions;
- a clean Git merge does not imply tests or acceptance criteria passed;
- a healthy heartbeat does not prove useful progress; and
- a drained queue does not prove the objective is complete.

These limits are features of the design. Keeping claims narrower than the
available evidence is what allows the supervisor to combine probabilistic model
proposals with deterministic engineering controls without confusing the two.

## Extension points

New integrations should prefer these boundaries:

- Add an evidence-producing scanner or validator behind a versioned receipt.
- Add a prover through the capability registry and conformance fixtures rather
  than calling a CLI directly from scheduling code.
- Add a task source through `objective_graph`/`backlog_refinery` so identities,
  deduplication, and bundle metadata are preserved.
- Add an LLM through the existing router/provider boundary and keep its output
  in the proposal tier until deterministic checks accept it.
- Add a scheduler policy by extending typed resource/lease contracts and their
  metrics, not by mutating daemon state ad hoc.
- Add persistence through an artifact or projection store with a schema version,
  canonical identity, and migration behavior.

## Operational caveats

The package contains compatibility layers for older todo boards, wrappers, and
provider APIs. A module being importable does not mean every optional backend is
available or conformant. Capability probes and self-tests are therefore part of
normal startup for formal lanes. Similarly, model-generated plans and Leanstral
outputs are proposals; they become actionable only after schema, policy,
validation, and evidence gates pass.

For a compact implementation inventory, start with:

- `objective_tracker.py`, `objective_graph.py`, `objective_daemon.py`;
- `formal_plan_context.py`, `formal_plan_compiler.py`,
  `formal_plan_validator.py`;
- `todo_daemon/engine.py`, `todo_daemon/implementation_daemon.py`,
  `todo_daemon/implementation_supervisor.py`;
- `bundle_supervisor.py`, `resource_scheduler.py`, `lease_coordination.py`;
- `artifact_store.py`, `event_log.py`, `scan_receipts.py`;
- `merge_train.py`, `merge_queue.py`, `merge_resolver.py`; and
- `prover_matrix_registry.py`, `multi_prover_router.py`,
  `prover_evidence_store.py`.

## Appendix: complete module map

The following map is intended to make the package discoverable when reading the
source tree. The names are grouped by the contract they primarily implement;
many modules intentionally participate in more than one group.

**Objective, backlog, and identity:** `objective_tracker.py`,
`objective_graph.py`, `objective_daemon.py`, `objective_task_janitor.py`,
`backlog_refinery.py`, `goal_completion.py`, `goal_coverage.py`,
`task_identity.py`, `taskboard_store.py`, `persistent_task_queue.py`,
`todo_vector_index.py`, `dataset_store.py`, `code_evidence_graph.py`,
`conflict_graph.py`, `task_proposal_router.py`, `plan_evaluator.py`.

**Planning, logic, and policy:** `formal_planning_contracts.py`,
`formal_plan_context.py`, `formal_plan_compiler.py`,
`formal_plan_validator.py`, `formal_logic_vocabulary.py`,
`logic_translation_validation.py`, `authorization_logic.py`,
`interface_contract_codegen.py`, `validation_commands.py`,
`validation_scheduler.py`, `supervisor_state_model.py`.

**Proof-directed decision runtime:** `decision_contracts.py`,
`ir_registry.py`, `semantic_dependency_graph.py`, `decision_context.py`,
`context_compiler.py`, `ir_constraint_compiler.py`, `execution_permit.py`,
`runtime_cas.py`, `decision_runtime.py`, `decision_runtime_benchmark.py`,
`decision_runtime_rollout.py`.

**Prompt bootstrap, task sources, lifecycle, and rescue:**
`prompt_workflow.py`, `prompt_directory_scanner.py`, `prompt_goal_planner.py`,
`prompt_plan_admission.py`, `task_source.py`, `markdown_task_source.py`,
`duckdb_task_source.py`, `lifecycle_orchestrator.py`,
`rescue_orchestrator.py`, `rescue_planner.py`, `recovery_diagnostics.py`,
`prompt_workflow_benchmark.py`, `prompt_workflow_rollout.py`.

**Proof scope, providers, and assurance:** `code_proof_obligations.py`,
`proof_obligation_templates.py`, `proof_context.py`, `proof_scope_index.py`,
`proof_scheduler.py`, `proof_fallbacks.py`, `proof_metrics.py`,
`formal_verification_contracts.py`, `formal_verification_policy.py`,
`formal_verification_provider.py`, `formal_verification_cache.py`,
`formal_verification_capabilities.py`, `prover_matrix_registry.py`,
`prover_conformance.py`, `multi_prover_router.py`,
`prover_evidence_store.py`, `proof_attestation.py`,
`kernel_verification.py`, `leanstral_proof_provider.py`,
`ipfs_datasets_logic_provider.py`, `hyperproperty_verification.py`.

**Execution, scheduling, and coordination:** `bundle_supervisor.py`,
`multi_supervisor_runner.py`, `implementation_daemon_runner.py`,
`implementation_supervisor_runner.py`, `resource_scheduler.py`,
`lease_coordination.py`, `leased_lane.py`, `checkout_lock.py`,
`wrapper_utils.py`, `analyzer_health.py`, `supervisor_watchdog.py`,
`runtime_temporal_monitor.py`, `scheduler_metrics.py`.

**Artifacts, events, and lifecycle state:** `artifact_store.py`,
`scan_receipts.py`, `event_log.py`, `duckdb_state.py`,
`proof_metrics.py`, `submodule_degradation.py`, `git_gc.py`.

**Merge and recovery:** `merge_train.py`, `merge_queue.py`,
`merge_checkpoint.py`, `merge_resolver.py`, `merge_conflict_repair.py`,
`llm_merge_resolver_fallback.py`, `codex_failure_policy.py`.

**The reusable todo-daemon runtime:** `todo_daemon/__init__.py` exposes the
public lifecycle/runtime helpers and `todo_daemon/__main__.py` provides module
execution. `todo_daemon/core.py` provides process
and state primitives; `engine.py` provides task/proposal/validation mechanics;
`implementation_daemon.py` and `implementation_supervisor.py` provide the
worker and watchdog loops; `supervisor.py`, `supervisor_loop.py`,
`supervisor_runtime.py`, `runner.py`, `app.py`, `cli.py`, and `wrapper.py`
provide lifecycle and command-line composition; `worktrees.py`, `git_utils.py`,
`file_replacement.py`, and `auto_commit.py` provide repository operations;
`artifacts.py`, `history.py`, `diagnostics.py`, `status.py`, and
`deterministic_fallback.py` provide observability and recovery; and
`context.py`, `llm.py`, `llm_defaults.py`, `plans.py`, `logic_port.py`,
`registry.py`, `specs.py`, `task_board.py`, `legal_parser.py`,
`legal_parser_daemon.py`, and `typescript.py` provide context, language-specific
adapters, registries, and task formats.

`__init__.py` is a public re-export surface, not the execution coordinator.
Importing a symbol from it should be understood as API convenience; the
invariants remain owned by the implementation module named above.

## Appendix: historical program evidence tags

Older prose and acceptance records labeled capabilities with **program board
IDs**. Those IDs remain valid for audit and taskboards; product documentation
should prefer the semantic names above.

| Historical tag | Semantic topic | Primary modules |
| --- | --- | --- |
| REF-289 | Counterexample-guided repair | `formal_replanner` |
| REF-290 | Plan conformance and completion | `formal_plan_conformance` |
| REF-291 | Shared prover admission | `multi_prover_resources` |
| REF-292 | Adversarial evidence admission | `formal_planning_adversarial` |
| REF-293 | Proof-carrying execution | `proof_carrying_planner` |
| REF-294 | Formal planning rollout metrics | `formal_planning_metrics`, `formal_planning_rollout` |
| ASI-G090 family | Paired-rollout completion adapter | self-improvement / rollout contracts |
| ASI-G112 / ASI-G113 | Paired-efficiency requirement terms | `PairedRolloutRequirementEvidence` |
| ASI-G114 | Import-isolation preflight | `PAIRED_ROLLOUT_STABLE_EXPORTS` cold validation |
| ASI-159 / ASI-G470 | Prompt bootstrap and rescue rollout | `prompt_workflow_*` |

Board prefix glossary: [PROGRAMS.md](agent_supervisor/PROGRAMS.md).

## Appendix: domain package module homes

When reading the source on a domain-layout tree, map flat module names from the
module map above into packages (see [PACKAGE_MAP.md](agent_supervisor/PACKAGE_MAP.md)):

| Concern | Domain package |
| --- | --- |
| Control operations, CLI, permits | `control/` |
| Taskboards, queues, identity | `task_sources/` |
| Objectives, goal lifecycle | `objectives/` |
| Formal plan compile/conformance | `planning/` |
| Proof cache, provers, attestation, codebase-proof | `proof/` |
| Context capsules, decision runtime | `context/` |
| Prompt bootstrap / rescue | `prompt/` |
| Multi-lane runners, event log, CAS | `runtime/` |
| Merge train / resolver | `merge/` |
| Rescue / recovery | `rescue/` |
| Self-improvement epochs | `self_improvement/` |
| Implementation daemon loops | `todo_daemon/` |
| Optional providers / merge LLM fallback | `integrations/` |
| Shared leaf utilities | `core/` |
