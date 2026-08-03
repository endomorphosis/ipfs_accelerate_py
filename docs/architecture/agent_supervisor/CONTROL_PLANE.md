# Supervisor intent, control, and authority

**Status:** Current  
**Audience:** Developers, operators, and implementation agents who must place
code, admit work, or diagnose why a mutation was denied  
**Scope:** Durable objective intent, taskboard projections, the transport-neutral
operation contract, discovery versus capability, and the authority path from
principal and policy through scope-bound, effect-bound, identity-bound mutation  
**Non-goals:** Planning and assurance pipelines (see planned
`PLANNING_AND_ASSURANCE.md` / DOC-012), multi-lane scheduling and rescue
(execution/recovery guides), prompt-first workflow product design, package DAG
placement rules (see [PACKAGE_MAP.md](PACKAGE_MAP.md)), or operator runbooks
(see [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md)). This guide does
not invent new operations, grants, or transports.  
**Last verified:** `d71cc2df31ec89716d30b153c989a8bbb557c0b2` (2026-08-03);
operation vocabulary, `OperationRequest` / `AuthorizationDecision` /
`ExpectedEffect` fields, authority classes, and denial reasons checked against
`control_contracts.py`, `authorization_logic.py`, `execution_permit.py`, and
package READMEs under `agent_supervisor/`.

---

## Source anchors

| Concern | Primary path / symbol | Notes |
| --- | --- | --- |
| Closed operation catalog | `ipfs_accelerate_py/agent_supervisor/control/control_contracts.py` — `Operation`, `OperationAuthority`, `READ_OPERATIONS`, `PROPOSAL_OPERATIONS`, `MUTATION_OPERATIONS` | Shared by Python, CLI, MCP |
| Request / result contracts | same — `OperationRequest`, `OperationResult`, `ExpectedEffect`, `EffectClaim`, `IdempotencyKey`, `AuthorizationDecision` | Canonical content identities |
| Control service | `…/control/control_plane.py` — `SupervisorControlService` | Allowlists, dispatch, audit |
| Authorization policy | `…/control/authorization_logic.py` — `Principal`, `AuthorizationPolicy`, `AuthorizationRequest`, `DenialReason` | Deterministic fail-closed evaluator |
| Mutation permits | `…/control/execution_permit.py` — `ExecutionPermit` | Short-lived, effect-bound |
| Lifecycle mutations | `…/control/lifecycle_orchestrator.py` | Process transitions under control |
| CLI adapter | `…/control/control_cli.py` — `register_agent_cli`, `run_agent_cli` | Builds the same `OperationRequest` |
| Durable intent | `ipfs_accelerate_py/agent_supervisor/objectives/` — `objective_tracker`, `objective_graph`, `goal_completion` | Objective heaps and goal authority |
| Task projections | `ipfs_accelerate_py/agent_supervisor/task_sources/` — `taskboard_store`, `task_identity`, `markdown_task_source` | Boards are schedulable views |
| Package ownership | `docs/architecture/agent_supervisor/packages/control.md`, `objectives.md`, `task_sources.md` | Domain boundaries |
| Design pillars | `docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md` | Authority ladder (product level) |
| Architecture map | `docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md` | Implementation contracts |
| Conformance / parity | `test/api/test_agent_supervisor_control_contracts.py`, `test_agent_supervisor_control_conformance_v2.py`, `test_agent_supervisor_authorization_logic.py`, `test_agent_supervisor_control_plane.py` | Denial and transport evidence |

---

## 1. Why this plane exists

The agent supervisor is a **control plane for objective-driven, evidence-bounded
software work**. Models propose plans and edits; parsers, policy checks,
validation commands, Git isolation, leases, and typed receipts decide whether
work may advance.

Without that separation, agent runs collapse into chat transcripts and ad-hoc
shell scripts: intent is lost when boards are regenerated, a confident model
response looks like completion, and CLI or MCP entrypoints accidentally widen
what Python policy would deny. The control plane makes **intent, authority,
isolation, and audit** first-class and **transport-neutral**.

---

## 2. Component map

```text
  *.objectives.md          *.todo.md / queues
  (durable intent)         (schedulable projection)
         │                        │
         └──────────┬─────────────┘
                    ▼
            objectives/  +  task_sources/
                    │
                    ▼
              control/   ◄── Python | CLI | MCP  (same OperationRequest)
         contracts · authz · permits · service
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      runtime    merge      rescue
    todo_daemon  worktrees  quarantine
```

| Layer | Responsibility | Owning packages |
| --- | --- | --- |
| **Intent** | Goals, parents, evidence expectations, acceptance | `objectives/` |
| **Projection** | Drainable tasks, stable task IDs, queues | `task_sources/` |
| **Control** | Closed ops, policy, leases/fences, effects, audit | `control/` |
| **Actuation** | Lanes, daemons, merge/rescue under admitted effects | `todo_daemon/`, `runtime/`, `merge/`, `rescue/` |

Transports are **adapters**. They do not own policy. They construct
`OperationRequest` records and present `OperationResult` records.

---

## 3. Intent hierarchy: durable goals, projected tasks

### 3.1 Objectives are durable intent

An **objective heap** (`*.objectives.md` and the `objectives/` package) states:

- goal identity and parent/child structure,
- evidence expectations and acceptance criteria,
- completion authority for the goal graph (not for arbitrary board edits).

Intent is **durable** because later scheduling, refill, or taskboard regeneration
must not silently rewrite protected goals. Refinery and projection may *emit*
tasks that serve an objective; they do not *become* the objective. Completing
or advancing a goal is a **policy + evidence** decision (`goal_completion` and
related modules), not a model monologue or a green local test alone.

### 3.2 Taskboards are projections

A **taskboard** (`*.todo.md`, DuckDB sources, queues under `task_sources/`) is a
**schedulable projection**:

- stable machine identity via `## PREFIX-###` headers,
- dependency, shard, and ready filters for daemons,
- status that records work progress for humans and agents.

| Artifact | Role | Must not |
| --- | --- | --- |
| Objective heap | Durable intent and acceptance | Be rewritten by refill without operator policy |
| Taskboard | Drainable tasks projected from intent | Confer mutation or completion authority |
| Model plan / patch | Proposal | Mark goals complete or expand allowlists |
| Receipt / audit | What actually ran | Be invented by transport metadata |

Regenerating or refining todos must preserve protected intent. Board status
alone is **not** authoritative completion. The task_sources layer is forbidden
from re-defining control authority or proof trust
([packages/task_sources.md](packages/task_sources.md)).

### 3.3 Identity bindings

Every control request carries content-bound identities so a replay, retry, or
alternate transport cannot retarget another tree or goal:

| Field on `OperationRequest` | Binds |
| --- | --- |
| `repository_root`, `state_root` | Absolute roots (allowlisted by the service) |
| `repository_id`, `tree_id` | Repository and tree content identity |
| `objective_id`, `objective_revision` | Goal identity and revision |
| `policy_id`, `policy_revision` | Policy identity and revision |
| `caller` | Acting principal / caller string |
| `lease_id`, `fencing_epoch` | Live isolation token (mutations) |
| `expected_effects` | Declared scope of side effects |
| `idempotency` | Replay key scoped to operation/caller/repo/objective |

Stale `tree_id`, expired lease, or mismatched fencing epoch fails closed.

---

## 4. Authority ladder

Work advances only by climbing this ladder. Skipping a rung is a bug, not an
optimization.

```text
1. Intent        objective / task identity (durable goals + projected tasks)
2. Proposal      model plan, patch, or preview operation
3. Validation    deterministic checks, tests, scope policy
4. Isolation     lease, fencing epoch, worktree, protected paths
5. Evidence      receipts, typed proofs, audits (tiered; no silent promotion)
6. Mutation      admitted effect application, merge, state update
```

**Models propose; policies admit.** A fluent explanation, a successful import,
or a capability probe never upgrades trust. Completion is a policy + evidence
decision, not eloquence.

Product-level pillars and non-goals: [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md).

---

## 5. One transport-neutral operation contract

### 5.1 Closed vocabulary and authority classes

`control_contracts.Operation` is a **closed set**. Each operation has a maximum
semantic authority (`OperationAuthority`):

| Authority | Meaning | Example operations |
| --- | --- | --- |
| **read** | Observe state; no proposal and no mutation | `capabilities`, `status`, `health`, `metrics`, `goals`, `tasks`, `bundles`, `lanes`, `events`, `receipts`, `cache_inspect`, `artifact_query` |
| **proposal** | Describe possible work; cannot apply mutation effects | `objective_preview`, `plan`, `workflow_preview`, `rescue_preview` |
| **mutation** | May change state/repo/lifecycle when fully bound | `objective_refine`, `objective_reconcile`, `backlog_refill`, `workflow_materialize`, `start` / `pause` / `resume` / `drain` / `stop` / `restart` / `retry` / `cancel` / `quarantine`, `validation_replay`, `rescue` |

Aliases such as `refill` → `backlog_refill` and `caches` → `cache_inspect`
decode to the same enum members; they do not create new authority.

Effects (`EffectKind`) carry the same authority ranking: `observe` (read),
`propose` (proposal), and write/delete/lifecycle/process/validation/audit kinds
(mutation). An expected effect whose authority exceeds the operation’s authority
is an `AuthorityViolationError` at request construction time.

### 5.2 `OperationRequest` as the unit of control

All surfaces build one record shape:

```text
OperationRequest
  operation + roots + content identities
  bounds + parameters + dry_run
  expected_effects[]          # required for live mutations
  authorization?              # policy decision binding
  lease_id + fencing_epoch    # required for live mutations
  idempotency?                # required for live mutations (scoped key)
       │
       ▼
SupervisorControlService.execute(...)
       │
       ▼
OperationResult
  status + error codes + effect claims + audit receipt identity
```

Contract invariants enforced in code (non-exhaustive):

- Live mutations **must** declare expected effects, including at least one
  mutation-authority effect.
- Paths on effects must stay within the selected root (path escape fails closed).
- Mutation authorization’s effect scope must **exactly match** the request’s
  authorized effect set; deny decisions grant no authority.
- Dry-run mutations never invoke the mutating adapter; results remain
  proposal-class (`DryRunPreview`), even if they *describe* mutation-shaped
  expected effects.
- Idempotency keys bind `operation`, `caller`, `repository_id`, and
  `objective_id`; reuse with a different payload is conflict; exact replay
  returns the original result without re-applying the backend effect.

### 5.3 Transport comparison (parity without authority expansion)

| Surface | How it invokes control | What it must not do |
| --- | --- | --- |
| **Python** | `SupervisorControlService.execute(OperationRequest(...))` | Embed a second policy stack |
| **CLI** | `ipfs-accelerate agent …` via `control_cli` → same service | Shell out to ad-hoc scripts that bypass contracts |
| **MCP** | One tool per canonical operation; `request` field holds the operation request | Derive server authority from tool input or listing tools |

`ControlSurface` (`python` / `cli` / `mcp`) is an independent publication of the
**same** catalog. Transports differ in **how roots and allowlists are
configured**, not in what operations mean. Discovery manifests and generation-2
catalog descriptors make cross-surface parity mechanically reviewable;
partial registration fails before publication.

**Prompt, model, and transport cannot confer authority.** A prompt-control
operation (`workflow_preview`, `workflow_materialize`, `restart`,
`rescue_preview`, `rescue`) still walks the same binding, lease, effect, and
audit path. MCP tool registration is static discovery; listing tools does not
resolve backends or grant mutation.

---

## 6. Discovery versus capability versus proof

| Signal | Means | Does not mean |
| --- | --- | --- |
| **Import / discovery manifest** | Closed vocabulary, schemas, authority class, bounds, dry-run/idempotency/authz requirements exist | Providers work; mutation is allowed |
| **Capability report / probe** | A configured backend can *attempt* an operation | Proof succeeded; future calls are free |
| **Validation / tests** | Deterministic checks passed for a claim set | Kernel proof or attestation |
| **Proof / attestation / receipt** | A specific claim met a stated assurance level | Completes unrelated goals |
| **Model prose** | Proposal only | Admission or completion |

Import success is never a capability signal. Capability is never proof. Cache
hits must **re-derive** assurance; they never invent it.

---

## 7. Principals, policy, effects, leases, and fences

### 7.1 Principals and grants

Authorization is a finite policy graph (`authorization_logic`):

- **`Principal`** — stable `principal_id`, independent of display names.
- **`AuthorizationGrant`** — issuer → subject capabilities with task, worktree,
  path, lease, proof, and override scopes.
- **`AuthorizationPolicy`** — grants, revocations, and **current lease IDs**
  (authoritative lease/fence state for evaluation).
- **`AuthorizationRequest`** — principal, action/capability, task/worktree/paths,
  optional lease and fencing epoch, evaluated against the policy.

Capabilities granted independently include `claim_task`, `execute_task`,
`publish_progress`, `merge`, `promote_proof`, and `override_policy`. Generated
code correctness projections that authorization may carry are intentionally
narrow (`not_established`); fluency does not become correctness.

### 7.2 Authorization decision on the control request

`AuthorizationDecision` is an exact, policy-produced binding for **one**
operation: verdict (`permit` / `deny`), operation, granted authority, roots,
repository/tree/objective/policy identities and revisions, caller, lease,
fencing epoch, authorized effect IDs, reason codes, grant IDs, and expiry.
Deny decisions must not grant authority. The control service requires that a
mutation’s attached decision match the request’s identities, lease, fence, and
effect set.

### 7.3 Expected effects (scope binding)

Every live mutation declares an ordered set of `ExpectedEffect` records:

- `effect_id` (unique within the request),
- `kind` → implied authority,
- `resource` and repository-relative `paths`,
- optional description.

The backend may apply **only** declared effects. Broadening path scope after
admission (e.g. in a candidate graph or permit use) is rejected
(`execution_permit` path-scope checks). Applied effect claims require mutation
authority and an audit `receipt_id`.

### 7.4 Leases and fencing

Isolation defaults:

- A **lease** binds who may act on a task/lane for a window.
- A **fencing epoch** invalidates workers that lost the lease or race after
  reclaim.
- Worktrees and protected paths keep foreign boards, sealed plans, and operator
  inputs outside unauthorized mutation.

Missing lease, lease scope mismatch, missing fencing epoch, or stale fencing
epoch are hard denials. Stale tree identity is a distinct fail-closed code
(`stale_tree`).

### 7.5 Execution permits

For mutation-capable paths that issue short-lived permits, `ExecutionPermit`
binds the decision request, admitted candidate, evidence, validation plan,
caller, policy, lease, fencing, and effect identities, then verifies that
bundle **immediately before** an effect. Changed effects, lost lease/fence,
replayed use sequences, or untrusted permit material fail closed.

---

## 8. Request and mutation flow

### 8.1 Happy path (admitted mutation)

```text
Caller (Python / CLI / MCP)
    │  build OperationRequest (identities, effects, idempotency, dry_run?)
    ▼
Decode / validate contract  ──► invalid_request | path_escape | authority_violation
    ▼
Allowlist roots + current tree  ──► forbidden | stale_tree
    ▼
Authorization freshness + lease/fence  ──► unauthorized | stale_lease | …
    ▼
Idempotency table
    ├─ exact replay ──► return prior OperationResult (no re-apply)
    └─ new key
         ▼
      dry_run? ──yes──► DryRunPreview (proposal authority; no mutate)
         │ no
         ▼
      Backend applies only declared effects
         ▼
      Durable redacted audit receipt + OperationResult
```

### 8.2 Read and proposal paths

Reads and proposals remain structurally bounded (roots, bounds, redaction) but
**cannot claim mutation effects**. Previews exist to plan refill, workflow, or
rescue work without advancing completion authority.

### 8.3 Failure and denial taxonomy

Contract and service errors surface stable codes (`ErrorCode` and related
status values), including:

| Code / reason family | Typical cause |
| --- | --- |
| `invalid_request` / `unknown_operation` | Malformed payload or unknown op |
| `unauthorized` / `forbidden` | Missing or non-matching authorization |
| `authority_violation` | Effect or claim exceeds operation authority |
| `path_escape` | Path leaves selected root |
| `stale_tree` | `tree_id` no longer current |
| `stale_lease` | Lease lost or fence advanced |
| `idempotency_required` / `idempotency_conflict` | Missing key or conflicting reuse |
| `bounds_exceeded` | Effects, paths, depth, or text over bounds |
| `unsupported_capability` / `unavailable` | Backend not registered or probe failed |
| `invalid_lifecycle_transition` | Illegal start/stop/drain/… transition |

Authorization engine denials (`DenialReason`) include no applicable grant,
malformed or over-deep delegation, not-yet-valid / expired / revoked grants,
task / worktree / path scope mismatch, lease required or mismatched, fencing
epoch required or stale, and proof/override authority required or mismatched.

**Fail-closed defaults:** missing lease/fence on mutation, failed validation,
protected-path writes outside declared outputs, undeclared backend mutations,
and any attempt to promote discovery or capability into proof.

**Degradation:** optional providers and backends drop to catalog-declared
`local_read_only` or `proposal_only` where configured; mutation and undeclared
fallbacks remain fail-closed. Import success or an alternate provider never
increases authority.

**Recovery:** retry within lease, rescue/quarantine operations under the same
contract, or operator policy change. Rescue is not a second, weaker control
plane.

---

## 9. Audit and parity

- Successful and denied mutations leave **audit-oriented result material**
  (effect claims with receipt identities when applied; stable error codes when
  not).
- Discovery must be **byte-deterministic** and side-effect free: no optional
  provider import, process start, or service resolution merely from listing
  operations.
- Cross-surface parity cases and mutation-guard evidence types live in
  `control_contracts` (`ControlSurfaceParityEvidence`,
  `ControlMutationGuardEvidence`, and related completion-quorum shapes) and are
  exercised by control conformance tests.
- Redaction applies to operator-visible results; secrets must not ride in argv,
  logs, or tool schemas as a side channel of “helpfulness.”

---

## 10. Rationale

1. **Durable intent** — Multi-lane agents and automated refill will thrash any
   system that treats the latest board rewrite as truth. Objectives hold goals
   and acceptance; boards remain disposable projections with stable IDs.
2. **Proposal ≠ admission** — LLMs are useful proposers and terrible sole
   authority. Binding scope, effects, leases, and evidence to every mutation
   stops trust upgrades by rhetoric.
3. **One contract, three transports** — Operators will use CLI; agents will use
   MCP; tests will use Python. Divergent vocabularies create accidental privilege
   and untestable edges. A single `OperationRequest` keeps parity mechanical.
4. **Identity and fence before write** — Distributed workers need leases and
   content identities so a late retry cannot mutate the wrong tree or goal
   revision.
5. **Fail closed** — Silent success paths are how optional integrations become
   forged completion. Unavailable backends return `unavailable`; they do not
   invent capacity.

---

## 11. Alternatives considered

| Alternative | Breakage |
| --- | --- |
| Treat taskboard status as completion authority | Refill/regeneration rewrites history; agents mark foreign work complete |
| Let model confidence or chat logs admit merges | Trust upgrade without scope/effect binding; unauditable production changes |
| Separate CLI/MCP operation sets “for convenience” | Privilege drift; conformance impossible; docs lie by transport |
| Derive allowlists from prompt text or tool input | Prompt injection becomes root access |
| Skip leases/fences in single-process demos as the default | Races and stale workers on the first multi-lane run |
| Collapse control into inference routing | Inference plane gains mutation authority by accident |

---

## 12. Consequences

**Positive**

- Readers can trace authority from objective → projection → request → deny/admit.
- Transports can be added without rewriting policy if they only adapt the
  shared contract.
- Denials are diagnosable with stable reason codes rather than free-form
  “permission denied.”
- Protected intent survives aggressive backlog automation.

**Negative / operational cost**

- Every live mutation requires more fields (effects, lease, fence, idempotency,
  identities) than a casual shell script.
- Catalog and conformance tests are mandatory gatekeeping when adding
  operations or surfaces.
- Dry-run and proposal paths must be carefully labeled so operators do not
  confuse previews with applied work.
- Dual documentation (philosophy + this guide + architecture) must stay aligned
  with `control_contracts.py` as the source of truth for vocabulary.

---

## 13. Extension and compatibility

1. Add operations only by extending the closed `Operation` enum and catalog
   descriptors with explicit authority, schemas, bounds, and dry-run/authz
   requirements — never by inventing a transport-only verb.
2. Keep the package DAG acyclic; control must not import implementation daemons
   for ordinary reads ([PACKAGE_MAP.md](PACKAGE_MAP.md)).
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Compatibility aliases decode to canonical operations; they must not widen
   authority or skip discovery.
5. New programs (self-improvement, codebase-proof, documentation refresh, …)
   layer **on** this plane; they do not fork a second supervisor
   ([PROGRAMS.md](PROGRAMS.md)).

---

## 14. Operational signals

| Signal | Where | Use |
| --- | --- | --- |
| Capability / discovery snapshot | `capabilities`, discovery manifests | Confirm vocabulary and backend attemptability before work |
| Lease / fence denials | authz + control results | Detect stolen or stale workers |
| Idempotency conflicts | mutation preflight | Detect retried-with-drift clients |
| Effect claims + receipt IDs | `OperationResult` | Audit what was applied |
| Lifecycle state | status/health + lifecycle orchestrator | Diagnose start/drain/quarantine |
| Event / receipt queries | `events`, `receipts` | Reconstruct a run without trusting chat logs |

---

## 15. Verification

Deterministic checks for this guide’s claims:

```bash
# Vocabulary and authority classes exist in source
rg -n "class Operation\b|class OperationAuthority\b|class OperationRequest\b|class AuthorizationDecision\b|class ExpectedEffect\b" \
  ipfs_accelerate_py/agent_supervisor/control/control_contracts.py

# Models propose / policies admit remains the product rule
rg -n "Models propose|policies admit|OperationAuthority" \
  docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md \
  docs/architecture/agent_supervisor/CONTROL_PLANE.md

# Denial reasons and principal policy
rg -n "class DenialReason\b|class Principal\b|class AuthorizationPolicy\b" \
  ipfs_accelerate_py/agent_supervisor/control/authorization_logic.py

# Focused automated suites (when running the full test profile)
# pytest test/api/test_agent_supervisor_control_contracts.py \
#        test/api/test_agent_supervisor_authorization_logic.py \
#        test/api/test_agent_supervisor_control_conformance_v2.py \
#        test/api/test_agent_supervisor_control_plane.py -q
```

Review checklist:

- [ ] Intent vs projection is explicit; boards are not completion authority.
- [ ] Every mutation path cites scope, effect, identity, lease/fence binding.
- [ ] No sentence claims prompt, model, or transport grants authority.
- [ ] Operation lists match `control_contracts.py` (re-verify on catalog edits).
- [ ] Discovery / capability / proof rows stay non-promoting.

---

## Related guides

| Doc | Role |
| --- | --- |
| [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) | Seven pillars and product authority ladder |
| [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md) | Full implementation map and rollout contracts |
| [Developer guide](DEVELOPER_GUIDE.md) | Extend and test inside the package |
| [Package map](PACKAGE_MAP.md) | Domain ownership and DAG |
| [packages/control.md](packages/control.md) | Control package page |
| [packages/objectives.md](packages/objectives.md) | Intent package page |
| [packages/task_sources.md](packages/task_sources.md) | Projection package page |
| [FOR_AGENTS.md](FOR_AGENTS.md) | Fail-closed agent capsule |
| [Operator guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) | Run, authorize, recover |
| [Guide conventions](../GUIDE_CONVENTIONS.md) | Architecture guide contract |
| [Doc hub](README.md) | Entry map for supervisor docs |
