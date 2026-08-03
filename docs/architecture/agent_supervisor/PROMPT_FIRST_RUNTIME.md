# Prompt-first runtime, persistence, and steering

**Status:** Partial — leaf primitives **landed** / **implemented**; high-level
product lifecycle still **planned** / **not yet** composed  
**Audience:** Developers and agents using or extending prompt-only entrypoints  
**Scope:** Current behavior of `agent_supervisor.entrypoints` resolvers, prompt
broker, run registry, plan lint, steering contracts, and verified IPLD access;
durable versus transient data; restart and CAS semantics; landed-versus-planned
product facade matrix  
**Non-goals:** Reconciling the ASE taskboard (board statuses may lag source);
implementing missing facades; documenting expert `ipfs-accelerate agent`
control ops (see [CONTROL_PLANE.md](CONTROL_PLANE.md)); low-level prompt
workflow internals (see [packages/prompt.md](packages/prompt.md)); package DAG
placement beyond this composition layer (see [PACKAGE_MAP.md](PACKAGE_MAP.md)).  
**Last verified:** tree `682c6699bf170e37a96439529253c15df77dccf9` (2026-08-03);
modules and matching `test/api/test_agent_supervisor_*` suites under
`ipfs_accelerate_py/agent_supervisor/entrypoints/` are the authority. The ASE
board and plan are intent/history only.

---

## Source anchors

| Concern | Primary path / symbol | Tests |
| --- | --- | --- |
| Package boundary / cold import | `entrypoints/__init__.py`, `ENTRYPOINT_CONTRACT_EXPORTS`, empty `ENTRYPOINT_LAZY_FACADE_EXPORTS` | `test_agent_supervisor_entrypoint_package.py` |
| Closed contracts | `entrypoints/contracts.py` — `SupervisorInvocationRequest`, `TargetResolutionReceipt`, `ResolvedSupervisorProfile`, `LaunchPlan`, `RunHandle` | `test_agent_supervisor_entrypoint_contracts.py` |
| Repository / checkout / scope | `entrypoints/target_resolver.py` — `RepositoryTargetResolver` | `test_agent_supervisor_target_resolver.py` |
| State root / namespace / run candidates | `entrypoints/state_resolver.py` | `test_agent_supervisor_state_resolver.py` |
| Objective / plan / task source / output | `entrypoints/objective_resolver.py` | `test_agent_supervisor_objective_resolver.py` |
| Principal / policy / effect ceiling | `entrypoints/authority_resolver.py` — `AuthorityResolver` | `test_agent_supervisor_authority_resolver.py` |
| Provider / resources / lanes / topology | `entrypoints/capability_resolver.py` — `CapabilityResolver` | `test_agent_supervisor_capability_resolver.py` |
| Profile precedence composition | `entrypoints/profile_resolver.py` — `SupervisorProfileResolver` | `test_agent_supervisor_profile_resolver.py` |
| Body-free inference explain | `entrypoints/inference_explain.py` | `test_agent_supervisor_inference_explain.py` |
| Plan / goal / profile lint | `entrypoints/plan_lint.py` | `test_agent_supervisor_plan_lint.py` |
| Transient prompt bodies | `entrypoints/prompt_broker.py` — `PromptBodyBroker` | `test_agent_supervisor_prompt_broker.py` |
| Durable run handles + CAS | `entrypoints/run_registry.py` — `RunRegistry` | `test_agent_supervisor_run_registry.py` |
| Steering contracts + classify | `entrypoints/steering_contracts.py` — `classify_steering_instruction` | `test_agent_supervisor_steering_contracts.py` |
| Verified CID / IPLD admission | `entrypoints/verified_ipld_backend.py` | `test_agent_supervisor_verified_ipld_backend.py` |
| Package narrative | [packages/entrypoints.md](packages/entrypoints.md), code `entrypoints/README.md` | — |
| Intent plan (not landed authority) | [AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md](../AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md) | — |
| Pre-facade friction inventory | [PROMPT_ENTRYPOINT_BASELINE.md](PROMPT_ENTRYPOINT_BASELINE.md) | `test_agent_supervisor_prompt_entrypoint_baseline.py` |

---

## 1. Purpose

A **prompt-first** supervisor invocation should accept a short instruction,
infer a safe target and profile, create or adopt a durable run, plan and admit
work, and return a resumable handle—without requiring nine target bindings or
hundreds of daemon flags.

That product journey is **not yet** wired end-to-end. What **has landed** is the
composition layer’s **library of typed primitives**:

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
factory, and resolve→materialize→start saga remain **planned**.

---

## 2. Architecture (landed layer)

```text
  CLI / Python / MCP / MCP++     ← product facades: PLANNED (not yet)
              |
              v
     entrypoints (composition)
       contracts (eager, provider-free)     ← implemented
       leaf resolvers                       ← implemented
         target · state · objective · authority · capability
       profile_resolver (precedence merge)  ← implemented
       prompt_broker (transient bodies)     ← implemented
       run_registry (durable handles/CAS)   ← implemented
       plan_lint · inference_explain        ← implemented
       steering_contracts (classify only)   ← implemented
       verified_ipld_backend                ← implemented
       Supervisor facade · intent saga ·
       runtime factory · CLI/MCP            ← planned / not yet
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
deterministic order (lower rank wins when coded that way; authority uses an
equivalent higher-wins scale with the same source order):

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

### 3.2 Leaf resolvers (implemented)

| Resolver | Owns | Fail-closed behaviors |
| --- | --- | --- |
| `RepositoryTargetResolver` | Root, repository ID, checkout, scope, HEAD+dirty overlay, submodules/nested | Symlink roots, parent traversal, equal-rank multi-root ambiguity; only allowlisted roots |
| State resolver | Platform state root, run namespace, active-run candidate classification | State defaults **outside** the checkout; forks/worktrees get separated namespaces; directory names/PIDs are non-authoritative |
| Objective / task-source / output resolvers | Objective binding, task-source kind/paths, output policy | New prompt may create a content-addressed objective; projection paths stay under state root |
| `AuthorityResolver` | Principal, policy, authority source, effect ceiling | Prompt/username/env never mint a principal; local-worktree allows isolated edits + validation; merge/push/deploy denied by default |
| `CapabilityResolver` | Provider route evidence, resources, lane ceiling, validation profile, topology | Prompt-authored shell rejected; lane ceiling from resources/conflicts, not prose |

### 3.3 Profile composition (implemented)

`SupervisorProfileResolver` merges leaf outputs and profile layers into:

- one `TargetResolutionReceipt` (evidence about resolution, **not** authorization), and  
- one `ResolvedSupervisorProfile` / `LaunchPlan` projection (argv, env **names**,
  paths, ceilings—credentials only as external **handles**).

Material ambiguity or denied authority blocks mutation-oriented profiles while
preserving a safe **preview** receipt (and demoted preview profile when paths
exist).

### 3.4 Explain and lint (implemented)

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
| Coordination claims / leases | Mutable shard (DuckDB owner) — **planned** composition here | Domain coordination packages | IPFS as lease authority |
| Immutable epochs / IPLD CIDs | Yes when admitted | Verified backend + planned replication | Fake/cache-only synthetic CIDs |

Contracts reject secret-shaped and prompt-embedded fields via
`SecretBearingRecordError` and forbidden argv markers (`--prompt`, `--ucan`,
`--token`, …). Durable free-form prose that hashes equal to a prompt body is
also rejected.

---

## 5. Prompt broker (implemented)

`PromptBodyBroker` is the body handoff ASE-012 intended:

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

## 6. Run registry: CAS, selection, restart (implemented)

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

### 7.1 Landed: contracts and classification

`steering_contracts` defines body-free `SteeringRequest`, `SteeringEvent`,
`SteeringResult`, and `classify_steering_instruction`.

Closed intent vocabulary:

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

### 7.2 Planned: apply, concurrent CAS, live lifecycle

**Not yet** in this package:

- applying admitted deltas to live plan/task-source revisions (ASE-024),  
- concurrent steering CAS/leases/fencing against a live run (ASE-025),  
- wiring classification into `Supervisor.steer` / CLI / MCP.

Until those land, treat classification outputs as **reviewable proposals**, not
mutations.

---

## 8. Verified IPLD access (implemented)

`verified_ipld_backend` admits only strict multiformats **CIDv1 / base32 /
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
| Provider-free entrypoint contracts | **Landed** | `contracts.py` + contract tests |
| Highest-layer package + cold import + no upward imports | **Landed** | `__init__.py`, package tests |
| Repository / checkout / dirty-tree resolution | **Landed** | `target_resolver.py` |
| State root / namespace / run-candidate resolution | **Landed** | `state_resolver.py` |
| Objective / plan / task-source / output resolution | **Landed** | `objective_resolver.py` |
| Principal / policy / effect ceiling | **Landed** | `authority_resolver.py` |
| Provider / resource / lane / validation / topology | **Landed** | `capability_resolver.py` |
| Profile precedence + full `TargetResolutionReceipt` | **Landed** | `profile_resolver.py` |
| Inference explain (body-free) | **Landed** | `inference_explain.py` |
| Plan / goal / profile lint | **Landed** | `plan_lint.py` |
| Prompt body broker (transient / capability) | **Landed** | `prompt_broker.py` |
| Run registry create / CAS / reconstruct / select | **Landed** | `run_registry.py` |
| Steering contracts + closed classification | **Landed** | `steering_contracts.py` |
| Verified IPLD/CID admission backend | **Landed** | `verified_ipld_backend.py` |
| `Supervisor.open()` lazy Python facade | **Planned** / **not yet** | `ENTRYPOINT_LAZY_FACADE_EXPORTS == ()`; no `supervisor.py` |
| Intent saga: resolve → preview → authorize → materialize → start/adopt | **Planned** / **not yet** | no intent service module |
| Standard runtime factory (real control/prompt handlers) | **Planned** / **not yet** | no `runtime_factory.py` |
| Profile → lifecycle/daemon argv live launch | **Planned** / **not yet** | compile helpers exist in profile resolver; no process owner here |
| Product CLI `ipfs-accelerate supervisor …` | **Planned** / **not yet** | baseline still documents expert `agent` path |
| MCP / MCP++ prompt-first tools + UCAN binding | **Planned** / **not yet** | transport tools not in package |
| Typed Grok→Codex production provider route module | **Planned** / **not yet** | capability resolver has evidence; no `provider_route.py` facade |
| DuckDB shard ownership / epoch cursor / replication saga | **Planned** / **not yet** | contracts mention shards; composition incomplete |
| Steering apply + concurrent CAS on live runs | **Planned** / **not yet** | contracts only |
| Status / follow / explain / doctor **services** | **Planned** / **not yet** | explain/lint libraries exist; product services do not |
| Installed E2E prompt-to-run acceptance green as product | **Planned** / **not yet** | frozen fixtures/acceptance tests exist as gates |

### Planned product lifecycle (reference only)

When facades land, the intended path remains:

```python
from ipfs_accelerate_py.agent_supervisor.entrypoints import Supervisor  # planned

supervisor = Supervisor.open()  # planned
run = supervisor.run("Improve validation-cache correctness")  # planned
supervisor.steer(run.run_id, "Prioritize concurrent cache tests")  # planned
for event in supervisor.follow(run.run_id):  # planned
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
| Secret-shaped durable field | Contract construction fails |
| Ambiguous steering intents | Clarification result; no mutation |

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
5. Expect `Supervisor.open()` to remain **planned** until lazy facade exports
   and saga modules land with tests.

---

## 12. Related documents

| Document | Role |
| --- | --- |
| [packages/entrypoints.md](packages/entrypoints.md) | Package semantic page |
| [PROMPT_ENTRYPOINT_BASELINE.md](PROMPT_ENTRYPOINT_BASELINE.md) | Pre-facade friction inventory |
| [PACKAGE_MAP.md](PACKAGE_MAP.md) | DAG placement |
| [CONTROL_PLANE.md](CONTROL_PLANE.md) | Transport-neutral ops and authority |
| [PLANNING_AND_ASSURANCE.md](PLANNING_AND_ASSURANCE.md) | Plan/proof pipeline |
| [AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md](../AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md) | Product design intent |
| Code `ipfs_accelerate_py/agent_supervisor/entrypoints/README.md` | Cold-import and storage rules |
