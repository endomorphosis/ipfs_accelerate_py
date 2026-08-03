# Prompt-first runtime, persistence, and steering

**Status:** Current
**Audience:** Developers and agents using or extending prompt-only entrypoints
**Scope:** Current behavior of `agent_supervisor.entrypoints` resolvers, prompt
broker, run registry, plan lint, steering contracts, and verified IPLD access;
durable versus transient data; restart and CAS semantics; exact
`landed|implemented` versus `planned|not yet` product facade matrix
**Non-goals:** Reconciling the ASE taskboard (board statuses may lag source);
implementing missing facades; documenting expert `ipfs-accelerate agent`
control ops (see [CONTROL_PLANE.md](CONTROL_PLANE.md)); low-level prompt
workflow internals (see [packages/prompt.md](packages/prompt.md)); package DAG
placement beyond this composition layer (see [PACKAGE_MAP.md](PACKAGE_MAP.md)).
**Last verified:** `6eb3525aae8143eb56993a6cd96eb9e3fff684e0` (2026-08-03);
modules and matching `test/api/test_agent_supervisor_*` suites under
`ipfs_accelerate_py/agent_supervisor/entrypoints/` are the authority. The ASE
board and plan are intent/history only.

**Status legend (validation keywords):** `landed|implemented` names behavior
present in source + focused tests; `planned|not yet` names product lifecycle
surfaces that remain absent (`Supervisor.open()`, intent saga, CLI/MCP facades,
steering apply).

---

## Source anchors

| Concern | Primary path / symbol | Tests |
| --- | --- | --- |
| Package boundary / cold import | `ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py` — `ENTRYPOINT_CONTRACT_EXPORTS`, empty `ENTRYPOINT_LAZY_FACADE_EXPORTS` | `test/api/test_agent_supervisor_entrypoint_package.py` |
| Closed contracts | `…/entrypoints/contracts.py` — `SupervisorInvocationRequest`, `TargetResolutionReceipt`, `ResolvedSupervisorProfile`, `LaunchPlan`, `RunHandle`, `SecretBearingRecordError` | `test/api/test_agent_supervisor_entrypoint_contracts.py` |
| Repository / checkout / scope | `…/entrypoints/target_resolver.py` — `RepositoryTargetResolver` | `test/api/test_agent_supervisor_target_resolver.py` |
| State root / namespace / run candidates | `…/entrypoints/state_resolver.py` — `StateRootResolver`, `RunCandidateResolver` | `test/api/test_agent_supervisor_state_resolver.py` |
| Objective / plan / task source / output | `…/entrypoints/objective_resolver.py` — `ObjectiveResolver`, `TaskSourceResolver`, `OutputPolicyResolver`, `ObjectivePlanTaskSourceResolver` | `test/api/test_agent_supervisor_objective_resolver.py` |
| Principal / policy / effect ceiling | `…/entrypoints/authority_resolver.py` — `AuthorityResolver` | `test/api/test_agent_supervisor_authority_resolver.py` |
| Provider / resources / lanes / topology | `…/entrypoints/capability_resolver.py` — `CapabilityResolver` | `test/api/test_agent_supervisor_capability_resolver.py` |
| Profile precedence composition | `…/entrypoints/profile_resolver.py` — `SupervisorProfileResolver`, `SOURCE_PRECEDENCE` | `test/api/test_agent_supervisor_profile_resolver.py` |
| Body-free inference explain | `…/entrypoints/inference_explain.py` | `test/api/test_agent_supervisor_inference_explain.py` |
| Plan / goal / profile lint | `…/entrypoints/plan_lint.py` | `test/api/test_agent_supervisor_plan_lint.py` |
| Transient prompt bodies | `…/entrypoints/prompt_broker.py` — `PromptBodyBroker` | `test/api/test_agent_supervisor_prompt_broker.py` |
| Durable run handles + CAS | `…/entrypoints/run_registry.py` — `RunRegistry`, `RunCasConflictError` | `test/api/test_agent_supervisor_run_registry.py` |
| Steering contracts + classify | `…/entrypoints/steering_contracts.py` — `classify_steering_instruction`, `CLOSED_STEERING_INTENT_KINDS` | `test/api/test_agent_supervisor_steering_contracts.py` |
| Verified CID / IPLD admission | `…/entrypoints/verified_ipld_backend.py` — `VerifiedIPLDBackend` | `test/api/test_agent_supervisor_verified_ipld_backend.py` |
| Package narrative | [packages/entrypoints.md](packages/entrypoints.md), code `entrypoints/README.md` | — |
| Intent plan (not landed authority) | [AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md](../AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md) | — |
| Pre-facade friction inventory | [PROMPT_ENTRYPOINT_BASELINE.md](PROMPT_ENTRYPOINT_BASELINE.md) | `test/api/test_agent_supervisor_prompt_entrypoint_baseline.py` |

---

## 1. Purpose

A **prompt-first** supervisor invocation should accept a short instruction,
infer a safe target and profile, create or adopt a durable run, plan and admit
work, and return a resumable handle—without requiring nine target bindings or
hundreds of daemon flags.

That product journey is **planned|not yet** wired end-to-end. What is
**landed|implemented** is the composition layer's **library of typed
primitives**:

1. Leaf and composed **target/profile resolvers** with explicit precedence and
   provenance.
2. A **prompt body broker** that keeps raw prompt text out of durable records.
3. A **run registry** with content-addressed handles, CAS heads, and restart
   reconstruction.
4. **Plan lint** and **inference explain** for body-free review.
5. **Steering contracts** and deterministic classification (application to live
   runs is still planned).
6. A **verified IPLD backend** gate for coordination CIDs.

Operators and embedders can use those modules today. The convenience
`Supervisor.open().run(prompt)` API, product CLI group, MCP tools, runtime
factory, and resolve→materialize→start saga remain **planned|not yet**.

---

## 2. Architecture (landed layer)

```text
  CLI / Python / MCP / MCP++     ← product facades: planned|not yet
              |
              v
     entrypoints (composition)
       contracts (eager, provider-free)     ← landed|implemented
       leaf resolvers                       ← landed|implemented
         target · state · objective · authority · capability
       profile_resolver (precedence merge)  ← landed|implemented
       prompt_broker (transient bodies)     ← landed|implemented
       run_registry (durable handles/CAS)   ← landed|implemented
       plan_lint · inference_explain        ← landed|implemented
       steering_contracts (classify only)   ← landed|implemented
       verified_ipld_backend                ← landed|implemented
       Supervisor facade · intent saga ·
       runtime factory · CLI/MCP            ← planned|not yet
              |
              v
     control · prompt · objectives ·
     planning · runtime · todo_daemon · …
     (existing domain packages; never import entrypoints)
```

**Cold import:** `import ipfs_accelerate_py.agent_supervisor.entrypoints` loads
only package metadata and `contracts`. It must not scan a repository, resolve
providers, open DuckDB/IPLD, or start processes. Facades, when added, must be
listed in `ENTRYPOINT_LAZY_FACADE_EXPORTS` and resolved lazily (currently empty).

**Package direction:** only the product/transport edge imports `entrypoints`.
No lower domain package may import it upward.

---

## 3. Resolver precedence and provenance

Every inferred field is recorded as a `TargetInferenceDecision` with:

- selected value and field name,
- `ResolutionSource` and precedence rank,
- disposition (`unique`, `defaulted`, `ambiguous`, `unavailable`, `denied`),
- alternatives and reason codes,
- whether the field is identity-only, effect-bearing, or authority-bearing.

### 3.1 Precedence ladder

Composition (`profile_resolver.SOURCE_PRECEDENCE` and leaf resolvers) uses one
deterministic source order. Profile composition treats **lower rank as better**
(canonical request = 10 … builtin default = 90). Authority resolution uses an
equivalent **higher-wins** numeric scale with the **same source order**:

```text
complete canonical request   (inference disabled)
  > explicit high-level override
  > existing run binding (resume / steer)
  > authenticated transport / server policy
  > signed local / user supervisor profile
  > reviewed repository hints
  > deterministic repository / runtime discovery
  > conservative built-in defaults
```

**Ceiling rule:** a lower-precedence source may only **narrow** allowlists,
authority, resource ceilings, and effect sets. It never widens them. Prompt
text and untrusted path labels cannot select roots, policies, principals, or
effects.

### 3.2 Leaf resolvers (`landed|implemented`)

| Resolver | Owns | Fail-closed behaviors |
| --- | --- | --- |
| `RepositoryTargetResolver` | Root, repository ID, checkout, scope, HEAD+dirty overlay, submodules/nested | Symlink roots, parent traversal, equal-rank multi-root ambiguity; only allowlisted roots |
| `StateRootResolver` / `RunCandidateResolver` | Platform state root, run namespace, active-run candidate classification | State defaults **outside** the checkout; forks/worktrees get separated namespaces; directory names/PIDs are non-authoritative |
| `ObjectiveResolver` / task-source / output resolvers | Objective binding, task-source kind/paths, output policy | New prompt may create a content-addressed objective; projection paths stay under state root |
| `AuthorityResolver` | Principal, policy, authority source, effect ceiling | Prompt/username/env never mint a principal; local-worktree allows isolated edits + validation; merge/push/deploy denied by default |
| `CapabilityResolver` | Provider route evidence, resources, lane ceiling, validation profile, topology | Prompt-authored shell rejected; lane ceiling from resources/conflicts, not prose |

### 3.3 Profile composition (`landed|implemented`)

`SupervisorProfileResolver` merges leaf outputs and profile layers into:

- one `TargetResolutionReceipt` (evidence about resolution, **not** authorization), and
- one `ResolvedSupervisorProfile` / `LaunchPlan` projection (argv, env **names**,
  paths, ceilings—credentials only as external **handles**).

Material ambiguity or denied authority blocks mutation-oriented profiles while
preserving a safe **preview** receipt (and demoted preview profile when paths
exist).

### 3.4 Explain and lint (`landed|implemented`)

| Module | Role |
| --- | --- |
| `inference_explain` | Human/JSON provenance for each decision; redacts secret-shaped material |
| `plan_lint` | Read-only checks for goal hierarchy, cycles/dependencies, unsafe validation strings, profile completeness |

Neither module re-hydrates prompt bodies into durable explanations.

---

## 4. Durable versus transient data

| Data | Durable? | Where | Must not appear in |
| --- | --- | --- | --- |
| Prompt **CID** + opaque `prompt_ref` | Yes (references only) | Invocation/run contracts, `PromptReference` | Logs as full body |
| Prompt **body** bytes | **No** (transient) | `PromptBodyBroker` memory or encrypted artifact under capability | Receipts, `OperationRequest.parameters`, events, argv, env, ordinary state |
| Capability **token** | Process-local secret | Returned once at deposit | Disk, logs (use `redacted_dict` / digests only) |
| Credentials / bearer UCANs / API keys | External handles only | Profile `credential_handles`; transport auth outside contracts | Canonical DAG-JSON contracts, launch argv |
| Target receipt / profile / launch plan | Yes | Content-addressed contracts | Raw prompt text, secrets |
| Run root + handle snapshots | Yes | `RunRegistry` under registry root | Prompt body, capability tokens |
| Steering instruction body | Transient field only | `SteeringRequest.transient_instruction_body` excluded from CID/JSON | Durable steering event equality surface |
| Instruction prompt CID / refs | Yes | Steering request/event/result | Full instruction text |
| Coordination claims / leases | Mutable shard (DuckDB owner) — composition **planned|not yet** here | Domain coordination packages | IPFS as lease authority |
| Immutable epochs / IPLD CIDs | Yes when admitted | Verified backend + planned replication | Fake/cache-only synthetic CIDs |

Contracts reject secret-shaped and prompt-embedded fields via
`SecretBearingRecordError` and forbidden argv markers (`--prompt`, `--ucan`,
`--token`, `--api-key`, `--authorization`, `--password`, `--secret`, …).
Durable free-form prose that hashes equal to a prompt body is also rejected.

---

## 5. Prompt broker (`landed|implemented`)

`PromptBodyBroker` is the body handoff for prompt-first composition:

1. **Deposit** exact bytes → content-addressed `prompt_cid` + opaque ref + one
   capability token.
2. **Resolve / open_for_planner** only with a matching capability, run ID, and
   unexpired window.
3. **Release / expire / close** zeroizes buffers and marks status.

Storage modes:

| Mode | Restart | Notes |
| --- | --- | --- |
| `memory` (default) | Body **lost**; restart behavior is explicit | Prefer for short local sessions |
| `encrypted_artifact` | Recoverable only with same master secret, artifact root, valid capability, unexpired window | Ciphertext under optional artifact root; not a substitute for capability |

`inspect_durable_surfaces` and `scan_for_secrets` exist so tests and operators
can assert bodies and tokens never landed on durable broker surfaces.

---

## 6. Run registry: CAS, selection, restart (`landed|implemented`)

`RunRegistry` owns high-level durable run records for this layer. Event stores,
process adoption, and task-source mutation stay outside the module.

### 6.1 On-disk layout

```text
<registry_root>/
  .run-registry.lock
  quarantine/
  namespaces/<namespace>/
    current.json              # optional CAS pointer for selected run
    runs/<run_id>/
      root.json               # immutable identity binding
      head.json               # CAS head: revision + handle CID
      handles/<handle_cid>.json
```

### 6.2 Operations

| Method | Behavior |
| --- | --- |
| `create` | Write immutable root + first handle snapshot + head |
| `cas_update` | Advance head only when `expected` revision matches; concurrent same-revision writers cannot both commit (`RunCasConflictError`) |
| `reconstruct` / `get` | Verify root + head + snapshot integrity; never return a canonical-looking handle on corruption |
| `select_current` / `get_current` / `set_current` | Exact unique compatible selection; multiple/incompatible candidates reported, not guessed |
| `list_runs` / `list_candidates` | Bounded listing |
| `repair` | Quarantine corrupt records; fail closed |

Restart semantics (`restart_behavior()`): handles **survive** process restart
via root+head+snapshot reconstruction; CAS survives; corruption is quarantined;
directory names, PID files, timestamps, and prompt text are
**non-authoritative**.

---

## 7. Steering (partial)

### 7.1 `landed|implemented`: contracts and classification

`steering_contracts` defines body-free `SteeringRequest`, `SteeringEvent`,
`SteeringResult`, and `classify_steering_instruction`.

Closed intent vocabulary (`CLOSED_STEERING_INTENT_KINDS`):

```text
append_requirement · answer_question · narrow_scope · reprioritize
request_replan · pause · resume · cancel · request_status
```

Rules:

- Free-form text may **propose** an intent; deterministic code **admits** it.
- Model proposals never grant authority or expand effect ceilings.
- Material multi-intent ambiguity yields a bounded clarification, not a guess.
- Requests bind **expected** `run_revision`, `plan_revision`, and
  `task_source_revision` (generation identity for later CAS).
- Instruction bodies are transient; durable surfaces keep CIDs/refs only.

### 7.2 `planned|not yet`: apply, concurrent CAS, live lifecycle

Still absent from this package:

- applying admitted deltas to live plan/task-source revisions,
- concurrent steering CAS/leases/fencing against a live run,
- wiring classification into `Supervisor.steer` / CLI / MCP.

Until those land, treat classification outputs as **reviewable proposals**, not
mutations.

---

## 8. Verified IPLD access (`landed|implemented`)

`VerifiedIPLDBackend` admits only strict multiformats **CIDv1 / base32 /
sha2-256** objects (`raw` or `dag-json`) into coordination manifests:

- expected CID computed locally before trust;
- put/get rehash verification;
- unsupported codecs and CAR fail closed;
- backend roles (`ipfs_kit_py`, Kubo, memory, cache) reported accurately;
- HuggingFace-style synthetic keys never admitted as IPLD CIDs;
- degradation is explicit on receipts.

IPFS availability still **never** grants a lease, authority, or effect.

---

## 9. Landed versus planned matrix

Status is derived from **source modules and focused tests**, not ASE board
checkboxes (those may still show `todo` after code has landed).

| Capability | Status | Evidence |
| --- | --- | --- |
| Provider-free entrypoint contracts | **landed / implemented** | `contracts.py` + contract tests |
| Highest-layer package + cold import + no upward imports | **landed / implemented** | `__init__.py`, package tests |
| Repository / checkout / dirty-tree resolution | **landed / implemented** | `target_resolver.py` |
| State root / namespace / run-candidate resolution | **landed / implemented** | `state_resolver.py` |
| Objective / plan / task-source / output resolution | **landed / implemented** | `objective_resolver.py` |
| Principal / policy / effect ceiling | **landed / implemented** | `authority_resolver.py` |
| Provider / resource / lane / validation / topology | **landed / implemented** | `capability_resolver.py` |
| Profile precedence + full `TargetResolutionReceipt` | **landed / implemented** | `profile_resolver.py` |
| Inference explain (body-free) | **landed / implemented** | `inference_explain.py` |
| Plan / goal / profile lint | **landed / implemented** | `plan_lint.py` |
| Prompt body broker (transient / capability) | **landed / implemented** | `prompt_broker.py` |
| Run registry create / CAS / reconstruct / select | **landed / implemented** | `run_registry.py` |
| Steering contracts + closed classification | **landed / implemented** | `steering_contracts.py` |
| Verified IPLD/CID admission backend | **landed / implemented** | `verified_ipld_backend.py` |
| `Supervisor.open()` lazy Python facade | **planned / not yet** | `ENTRYPOINT_LAZY_FACADE_EXPORTS == ()`; no `supervisor.py` |
| Intent saga: resolve → preview → authorize → materialize → start/adopt | **planned / not yet** | no intent service module |
| Standard runtime factory (real control/prompt handlers) | **planned / not yet** | no `runtime_factory.py` |
| Profile → lifecycle/daemon argv live launch | **planned / not yet** | compile helpers exist in profile resolver; no process owner here |
| Product CLI `ipfs-accelerate supervisor …` | **planned / not yet** | baseline still documents expert `agent` path |
| MCP / MCP++ prompt-first tools + UCAN binding | **planned / not yet** | transport tools not in package |
| Typed Grok→Codex production provider route module | **planned / not yet** | capability resolver has evidence; no `provider_route.py` facade |
| DuckDB shard ownership / epoch cursor / replication saga | **planned / not yet** | contracts mention shards; composition incomplete |
| Steering apply + concurrent CAS on live runs | **planned / not yet** | contracts only |
| Status / follow / explain / doctor **services** | **planned / not yet** | explain/lint libraries exist; product services do not |
| Installed E2E prompt-to-run acceptance green as product | **planned / not yet** | frozen fixtures/acceptance tests exist as gates |

### Planned product lifecycle (reference only)

When facades land, the intended path remains:

```python
from ipfs_accelerate_py.agent_supervisor.entrypoints import Supervisor  # planned|not yet

supervisor = Supervisor.open()  # planned|not yet
run = supervisor.run("Improve validation-cache correctness")  # planned|not yet
supervisor.steer(run.run_id, "Prioritize concurrent cache tests")  # planned|not yet
for event in supervisor.follow(run.run_id):  # planned|not yet
    print(event.summary)
```

Until then, embedders compose resolvers, broker, and registry explicitly, and
still authorize mutations through the existing control plane.

---

## 10. Ambiguity, restart, and failure semantics

| Situation | Landed behavior |
| --- | --- |
| Multiple equal-rank roots / runs | Typed ambiguity or selection result; no silent pick |
| Denied authority / over-wide effects | Decision disposition `denied`; mutation profile not emitted |
| Corrupt registry records | Quarantine; lookup fails closed |
| CAS revision mismatch | `RunCasConflictError`; neither writer invents a merge |
| Memory broker after restart | Body unavailable; restart_behavior documents loss |
| Secret-shaped durable field | Contract construction fails (`SecretBearingRecordError`) |
| Ambiguous steering intents | Clarification result; no mutation |
| Unverified / synthetic CID | Admission refused; backend role reported honestly |

---

## 11. How to use landed primitives safely

```python
# Contracts only (cold-safe package import)
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    SupervisorInvocationRequest,
    RunHandle,
)

# Leaf / composition modules (import deliberately; still provider-free pure paths
# where documented)
from ipfs_accelerate_py.agent_supervisor.entrypoints.prompt_broker import (
    PromptBodyBroker,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import (
    RunRegistry,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.steering_contracts import (
    classify_steering_instruction,
)
```

Guidelines:

1. Persist **CIDs and handles**, never prompt bodies or capability tokens.
2. Treat `TargetResolutionReceipt` as **evidence**, not a grant.
3. Authorize effects through `control` policy / permits, not resolver success.
4. Do not invent a second mutation path outside `OperationRequest`.
5. Expect `Supervisor.open()` to remain **planned|not yet** until lazy facade
   exports and saga modules land with tests.

---

## 12. Rationale

1. **Prompt-only product without a second control plane.** Embedders need a
   short path from natural language to a resumable run, but mutations must still
   pass through the existing operation/permit ladder.
2. **Inference is evidence, not authority.** Receipts and precedence ranks make
   defaults auditable; they never mint principals or widen effects.
3. **Bodies stay out of durable records.** Prompt text, capability tokens,
   credentials, and bearer UCANs must not reappear in receipts, argv, env, or
   ordinary state—only CIDs, opaque refs, and external handles.
4. **CAS and quarantine over best-effort directories.** Concurrent writers and
   corrupt roots must fail closed so restart reconstruction cannot invent a
   handle.
5. **Facades stay lazy and optional.** Cold import remains provider-free until
   explicit product delivery tasks fill `ENTRYPOINT_LAZY_FACADE_EXPORTS`.

---

## 13. Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Ship `Supervisor.open()` before leaf resolvers | Hides unsafe defaults; no receipt surface for review |
| Persist raw prompt bodies in run roots | Leaks secrets into logs, backups, and replication |
| Let prompt text select principal/policy/root | Authority would be model-shaped and forgeable |
| Silent "best" root/run when multiple candidates | Undebuggable cross-repo damage |
| Treat ASE board `done` as runtime truth | Board lag would document non-existent APIs |
| Put leases on IPFS CIDs | Availability ≠ ownership; CAS belongs to elected shard owner |
| Classify steering and apply in one model step | Model proposals would mutate without deterministic admission |

A simpler chat-loop design would make false admissions cheaper than correct
refusals—the opposite of the supervisor's fail-closed posture.

---

## 14. Consequences

**Positive**

- Operators can use broker, registry, and resolvers today without waiting for
  the product facade.
- Durable records stay free of prompt bodies, UCANs, and credentials.
- Precedence and dispositions make ambiguous targets reviewable.
- Focused tests—not board checkboxes—are the status authority.

**Negative / costs**

- No single `Supervisor.open().run(prompt)` call yet; embedders compose
  modules.
- Memory broker loses bodies across process restart unless encrypted artifacts
  and secrets are configured deliberately.
- Steering classify without apply still requires control-plane mutations for
  live changes.
- Writers must keep `planned|not yet` facades labelled so agents do not invent
  import paths.

---

## 15. Extension and compatibility

| When extending… | Do |
| --- | --- |
| New leaf inference field | Add decision provenance + ceiling tests; never widen from prompt prose |
| New facade symbol | List it in `ENTRYPOINT_LAZY_FACADE_EXPORTS`; keep package import cold |
| New durable field | Forbid secret-shaped and body-equal prose via contract guards |
| New steering intent | Extend closed vocabulary with effect requirements; keep body transient |
| New IPLD backend role | Report role accurately; rehash before trust; never grant leases |
| Docs | Update this guide and [packages/entrypoints.md](packages/entrypoints.md) from **source and tests** when status moves |

Compatibility expert path remains `ipfs-accelerate agent` / control ops—not a
substitute for the planned prompt-first product CLI.

---

## 16. Operational signals

| Signal | Where | Use |
| --- | --- | --- |
| Decision dispositions | `TargetInferenceDecision` / receipts | unique / ambiguous / denied dashboards |
| CAS conflicts | `RunCasConflictError`, head revisions | Concurrent writer detection |
| Quarantine events | `RunRegistry` quarantine dir | Corrupt handle investigation |
| Broker restart behavior | `PromptBodyBroker.restart_behavior()` | Body survival expectations |
| Durable surface scans | `inspect_durable_surfaces`, `scan_for_secrets` | Leak regression gates |
| IPLD admission / degradation | `VerifiedIPLDBackend` receipts | Cache-only vs conformant CID |
| Cold import | package tests | No provider/daemon on import |

Logs should cite **CIDs and run IDs**, never prompt bodies or capability tokens.

---

## 17. Verification

Deterministic checks for this guide's claims:

```bash
# Guide contract + DOC-014 validation keywords (literal landed|implemented and planned|not yet)
test -f docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md
test -f docs/architecture/agent_supervisor/packages/entrypoints.md
rg -qi 'landed\|implemented' docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md
rg -qi 'planned\|not yet' docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md
git diff --check

# Package boundary
test -d ipfs_accelerate_py/agent_supervisor/entrypoints
test -f ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py
test -f ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py
test -f ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py
test -f ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py
test -f ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py
test -f ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py
rg -q 'ENTRYPOINT_LAZY_FACADE_EXPORTS: Final\[tuple\[str, \.\.\.\]\] = ()' \
  ipfs_accelerate_py/agent_supervisor/entrypoints/__init__.py
test ! -f ipfs_accelerate_py/agent_supervisor/entrypoints/supervisor.py
test ! -f ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py

# Landed symbols
rg -q 'class RepositoryTargetResolver' ipfs_accelerate_py/agent_supervisor/entrypoints/target_resolver.py
rg -q 'class SupervisorProfileResolver' ipfs_accelerate_py/agent_supervisor/entrypoints/profile_resolver.py
rg -q 'class PromptBodyBroker' ipfs_accelerate_py/agent_supervisor/entrypoints/prompt_broker.py
rg -q 'class RunRegistry' ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py
rg -q 'class RunCasConflictError' ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py
rg -q 'def classify_steering_instruction' ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py
rg -q 'class VerifiedIPLDBackend' ipfs_accelerate_py/agent_supervisor/entrypoints/verified_ipld_backend.py
rg -q 'class SecretBearingRecordError' ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py
rg -q 'transient_instruction_body' ipfs_accelerate_py/agent_supervisor/entrypoints/steering_contracts.py

# Focused tests (optional when running the full test profile)
# python -m pytest \
#   test/api/test_agent_supervisor_entrypoint_package.py \
#   test/api/test_agent_supervisor_entrypoint_contracts.py \
#   test/api/test_agent_supervisor_prompt_broker.py \
#   test/api/test_agent_supervisor_run_registry.py \
#   test/api/test_agent_supervisor_steering_contracts.py \
#   test/api/test_agent_supervisor_verified_ipld_backend.py -q
```

Review checklist:

- [ ] Source anchors cite live modules; ASE board is not status authority.
- [ ] `Supervisor.open()` / product CLI / MCP remain clearly **planned|not yet**.
- [ ] Credentials, UCANs, and prompt bodies stay out of durable-record claims.
- [ ] Cold import and no-upward-import rules still match `__init__.py`.
- [ ] Relative links resolve; `git diff --check` clean.

---

## 18. Related documents

| Document | Role |
| --- | --- |
| [packages/entrypoints.md](packages/entrypoints.md) | Package semantic page |
| [PROMPT_ENTRYPOINT_BASELINE.md](PROMPT_ENTRYPOINT_BASELINE.md) | Pre-facade friction inventory |
| [PACKAGE_MAP.md](PACKAGE_MAP.md) | DAG placement |
| [CONTROL_PLANE.md](CONTROL_PLANE.md) | Transport-neutral ops and authority |
| [PLANNING_AND_ASSURANCE.md](PLANNING_AND_ASSURANCE.md) | Plan/proof pipeline |
| [EXECUTION_AND_RECOVERY.md](EXECUTION_AND_RECOVERY.md) | Multi-lane execution and rescue |
| [AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md](../AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md) | Product design intent |
| [GUIDE_CONVENTIONS.md](../GUIDE_CONVENTIONS.md) | Architecture guide contract |
| Code `ipfs_accelerate_py/agent_supervisor/entrypoints/README.md` | Cold-import and storage rules |
