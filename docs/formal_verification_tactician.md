# Formal Verification Tactician

**Interface:** `FormalVerificationTacticianDocumentation@1`  
**Public API:** `LogicVerificationAPI@1` + `GoalTacticianAPI@1`  
**Orchestration:** `GoalDirectedProofTactician@1`  
**Lifecycle:** `GoalTacticianSupervisorLifecycle@1`  
**Counterexample loop:** `CounterexampleGuidedProofDevelopment@1`

This document is the product-facing architecture and operator guide for the
goal-directed formal verification tactician. Companion artifacts:

| Path | Role |
| --- | --- |
| `docs/operations/formal_verification_tactician_runbook.md` | Supervisor operations, incident response, failure/rollback matrix |
| `ipfs_datasets_py/docs/logic/proof_tactician_migration.md` | Compatibility aliases and migration from legacy logic surfaces |
| `docs/architecture/formal_verification_readiness_baseline.json` | Machine-specific readiness ledger (implementation ≠ deployment) |
| `ipfs_datasets_py/docs/logic/software_verification_rollout.md` | Property/provider rollout stages (LFV baseline) |
| `ipfs_datasets_py/examples/logic/software_verification/README.md` | Runnable source-bound examples |

---

## 1. Purpose and ownership

The tactician formalizes an explained end state, discovers missing proof
obligations needed to reach it, ranks and validates candidate lemmas /
invariants / contracts, executes a bounded proof plan, and returns safe,
replayable counterexamples when the goal does not hold.

| Layer | Package | Owns |
| --- | --- | --- |
| Canonical semantics | `ipfs_datasets_py.logic` | Goal/proof-hole contracts, goal compiler, counterexample semantics, public verification API, provider registry |
| Orchestration | `ipfs_accelerate_py.agent_supervisor.proof` | Goal-directed tactician, CEGIS/CEGAR loop, lifecycle/leases, resources, replanning, metrics |
| Legal evidence (separate lane) | `agent_supervisor.proof.legal_constraint_adapter` | LegalIR applicability compilation — **not** formal proof planning |

Imports, declarations, inventory, and probes never install packages, download
toolchains, open the network, spawn unbounded processes, or mutate a checkout.

---

## 2. Critical distinctions (fail-closed vocabulary)

Product docs and APIs must keep these axes separate. Collapsing any pair is a
documentation defect and usually a soundness defect.

### 2.1 Legal evidence routing vs formal proof planning

| Axis | Legal evidence routing | Formal proof planning |
| --- | --- | --- |
| Input | LegalIR, provisions, applicability query | End-goal formalization, source/VC obligations, proof graph |
| Output | Constraints, legal proof *obligations* for compliance review | Proof plan steps, candidate lemmas, verifier receipts |
| Authority ceiling | Legal applicability / constraint compilation | Bounded solver, kernel-checked theorem, or explicit non-success |
| Adapter | `legal_constraint_adapter` | `goal_directed_tactician`, `GoalTacticianAPI` |
| May close a software proof? | **No** | Only with a fresh matching verifier receipt |

Legal permission is a constraint result, not SecurityIR authorization and not
an execution grant. When legal evidence is in scope, the goal-directed tactician
records `legal_compatible` without promoting legal constraints to theorem
authority.

### 2.2 Proposals vs proofs

| Observation | Authority | May report theorem success? |
| --- | --- | --- |
| Advisor / Leanstral / SymAI / autoencoder draft | `advisory` / `candidate` | **No** |
| Exact cache hit | Exactly the cached receipt’s authority | Only if the cached item already had that authority |
| Independently validated candidate | Still candidate until re-verified | **No** until a fresh check |
| Fresh verifier / kernel receipt bound to tree + property + assumptions + bounds | Declared authority class | Only for that class |

`advise`, `formalize_goal`, and candidate synthesis produce **proposals**.
`check`, `validate_proof_candidate` (independent check), kernel reconstruction,
and lifecycle completion consume **proofs** (typed receipts). A structural
repair never closes a counterexample; only a fresh matching verifier receipt
can close it.

### 2.3 Bounded checks vs theorem proof

| Kind | Typical sources | Meaning |
| --- | --- | --- |
| Bounded check | Z3/CVC5 under exact resource bounds, bounded model check, finite monitor window | Outcome holds for the exact formula, assumptions, tool identity, and bounds — not an unbounded theorem |
| Theorem / kernel proof | Lean / Rocq / Isabelle after exact check of a proof artifact | Kernel accepted the artifact under the named environment |
| Monitor satisfaction | Runtime MTL / trace monitors | Finite-trace verdict only |
| Authorization policy | Datalog / SecPAL | Policy evaluation, not theorem proof |

A `succeeded` status with `authority: bounded` is **not** a theorem. API
consumers must read `authority` and `assumptions` on every response.

### 2.4 Implementation completeness vs deployment certification

| Claim | Evidence | Does **not** mean |
| --- | --- | --- |
| Implementation complete (code present, fixture-tested) | Source tree, unit/integration tests, baseline `implemented` / `fixture_tested` | Tools installed on this host |
| Live-tested | Bounded live probes on this machine | Production-certified for all hosts |
| Installed / usable | Exact executable identities, offline lock, probes | Deployment certification receipt |
| Production / deployment certified | Hermetic toolchain certificate + completion receipt + rollout gates | That every optional prover is present everywhere |

`docs/architecture/formal_verification_readiness_baseline.json` records the
status ladder. Offline fixture success never upgrades to deployment
certification. Source or PATH presence never implies usability.

### 2.5 Assumptions vs obligations

| Term | Definition | Operator action |
| --- | --- | --- |
| Assumption | A premise treated as given for a check (may be inherited, operator-accepted, or candidate) | Must appear explicitly in the request and receipt; never silent |
| Obligation / proof hole | A missing premise, lemma, invariant, or contract that must still be discharged | Tracked as open work with cost and review state |
| Favorable assumption | An assumption inserted because it makes the goal hold | **Forbidden** as silent success; must become a visible obligation |

New assumptions are visible proof obligations with explicit cost and review
state. A favorable assumption cannot be inserted merely because it entails the
target.

### 2.6 Failure and rollback states

Every non-success is explicit. The runbook enumerates operator responses; the
vocabulary is:

| State / status | Meaning |
| --- | --- |
| `invalid` | Malformed request, forbidden supervisor control, schema failure |
| `unsupported` | Outside the supported language/tool fragment |
| `unavailable` | Required optional tool or runtime not present / not probeable |
| `partial` | Cancelled, timed out, or incomplete under bounds |
| `error` | Unexpected failure with diagnostics |
| `blocked` / `failed` / `invalidated` (lifecycle) | Plan cannot progress; evidence incomplete, control active, or tree changed |
| Quarantine / demotion (rollout) | Pair demoted to shadow/declared; historical receipts immutable |

Tool absence, timeout, disagreement, budget exhaustion, and ambiguity are
**never** silent success.

---

## 3. Architecture

```
                    ┌──────────────────────────────────────┐
                    │  Public surfaces (parity required)   │
                    │  Python API · CLI · MCP tools         │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │  LogicVerificationAPI / GoalTactician │
                    │  (ipfs_datasets_py.logic)             │
                    └──────────────────┬───────────────────┘
           ┌───────────────────────────┼───────────────────────────┐
           ▼                           ▼                           ▼
   End-goal formalizer          Proof graph / holes          Counterexamples
   Ambiguity / interpretations  Candidate synth+validate     Minimize / explain / replay
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  Supervisor orchestration             │
                    │  GoalDirectedProofTactician           │
                    │  CounterexampleGuidedProofDevelopment │
                    │  GoalTacticianSupervisorLifecycle     │
                    └──────────────────┬───────────────────┘
                                       ▼
                    Bounded backends (SMT, kernels, monitors)
                    Exact caches · leases · resource fences
```

### 3.1 End-to-end flow

1. **Author** a prose or structured end goal with source binding and bounds.
2. **Formalize** (`formalize_goal`) into candidate end-goal structures — not
   admitted goals.
3. **Resolve ambiguity** (`compare_interpretations`) when multiple readings exist;
   material interpretation selection is required before planning proceeds.
4. **Discover missing proofs** (`discover_missing_proofs`) as typed proof holes.
5. **Plan** (`plan_proof`) a ranked obligation graph / plan under authority and
   utility ranking.
6. **Validate candidates** (`validate_proof_candidate`) independently of the
   synthesizer.
7. **Execute** (`execute_proof_plan`) under lifecycle fencing and resource
   policy; observe with `proof_status`.
8. On failure: **minimize / explain / replay** counterexamples; feed
   verifier-backed CEGIS repair; close only on fresh matching receipts.
9. **Complete** only when all selected graph leaves and open counterexamples
   carry adequate fresh receipts for the current tree and fencing epoch.

---

## 4. Public API surfaces

### 4.1 Stable verification operations (`LogicVerificationAPI@1`)

| Operation | Role | Authority ceiling |
| --- | --- | --- |
| `list_logic_families` | Declarative family catalog | declarative |
| `list_providers` | Declarative provider catalog | declarative |
| `provider_capabilities` | Declared capabilities | declarative |
| `compile_verification_artifact` | Source/IR compilation | source translation only |
| `check` | Bounded / kernel check | as returned |
| `monitor` | Runtime monitoring | monitor |
| `run_portfolio` | Multi-provider portfolio | per-provider |
| `explain_counterexample` | Public counterexample explanation | none / bounded witness |
| `verify_receipt` | Fail-closed receipt validation | attestation of structure |
| `attest_receipt` | Bind attestation to an existing receipt | does not raise proof authority |
| `advise` | Advisor proposals | advisory only |
| `probe_provider` / `install_provider` | Opt-in discovery / install | capability health only |

Importing `ipfs_datasets_py.logic.verification_api` never probes the environment.

### 4.2 Goal tactician operations (`GoalTacticianAPI@1`)

| Python / API operation | CLI command | MCP tool |
| --- | --- | --- |
| `formalize_goal` | `goal-formalize` | `goal_tactician_formalize_goal` |
| `compare_interpretations` | `goal-compare-interpretations` | `goal_tactician_compare_interpretations` |
| `discover_missing_proofs` | `goal-discover-missing-proofs` | `goal_tactician_discover_missing_proofs` |
| `plan_proof` | `goal-plan-proof` | `goal_tactician_plan_proof` |
| `validate_proof_candidate` | `goal-validate-candidate` | `goal_tactician_validate_proof_candidate` |
| `execute_proof_plan` | `goal-execute-plan` | `goal_tactician_execute_proof_plan` |
| `proof_status` | `goal-proof-status` | `goal_tactician_proof_status` |
| `minimize_counterexample` | `goal-minimize-counterexample` | `goal_tactician_minimize_counterexample` |
| `explain_counterexample_causal` | `goal-explain-counterexample` | `goal_tactician_explain_counterexample_causal` |
| `replay_counterexample` | `goal-replay-counterexample` | `goal_tactician_replay_counterexample` |
| `list_goal_tactician_operations` | `goal-list-operations` | `goal_tactician_list_operations` |

Supervisor-only controls are rejected on public surfaces:

`admit_goal`, `close_plan`, `mutate_supervisor`, `force_complete`, `lease_steal`,
`rewrite_event_log`, `bypass_resource_policy`, `promote_proof_authority`,
`supervisor_mutate`, `supervisor_only`.

---

## 5. Executable examples

Examples use only public APIs. Optional tools may report `unavailable`; that is
a valid non-success outcome.

### 5.1 Python — discover goal-tactician surface

```python
from ipfs_datasets_py.logic.verification_api import (
    LogicVerificationAPI,
    list_goal_tactician_operations,
)

api = LogicVerificationAPI()
surface = list_goal_tactician_operations()
assert surface.status.value == "declarative"
assert "formalize_goal" in surface.result["operations"]

# Declarative catalogs never install or probe tools
families = api.list_logic_families()
providers = api.list_providers()
print(families.status, providers.status)
```

### 5.2 Python — formalize an end goal (proposal only)

```python
from ipfs_datasets_py.logic.verification_api import formalize_goal

response = formalize_goal(
    {
        "prose": "Every acquired lease is released before task completion.",
        "source_binding": {
            "path": "ipfs_accelerate_py/agent_supervisor/runtime/lease.py",
            "language": "python",
        },
        "bounds": {"max_candidates": 4, "timeout_seconds": 5},
    }
)
# Formalization yields candidates — never an admitted proof
assert response.authority.value in {"none", "advisory", "candidate", "declarative"}
assert response.result.get("admitted") is not True
print(response.status, response.authority, response.diagnostics)
```

### 5.3 Python — check with explicit authority inspection

```python
from ipfs_datasets_py.logic.verification_api import check

response = check(
    {
        "property": "contract",
        "source": "def f(x):\n    assert x >= 0\n    return x\n",
        "language": "python",
        "bounds": {"timeout_seconds": 5, "memory_mb": 512},
        "assumptions": [],  # empty list is explicit — not "unspecified"
    }
)
# Bounded success is not theorem proof
if response.status.value == "succeeded":
    assert response.authority.value != "theorem" or "kernel" in str(response.result)
print(
    {
        "status": response.status.value,
        "authority": response.authority.value,
        "assumptions": list(response.assumptions)
        if hasattr(response, "assumptions")
        else response.result.get("assumptions"),
    }
)
```

### 5.4 Python — replay a counterexample envelope

```python
from ipfs_datasets_py.logic.verification_api import replay_counterexample

# Public envelopes never embed raw private witnesses or credentials.
envelope = {
    "schema": "counterexample-envelope/v2",
    "property_id": "contract.nonneg",
    "source_tree_id": "git-tree:example",
    "bounds": {"timeout_seconds": 5},
    "witness_public": {"model": {"x": -1}},
    "origin_provider": "backend.smt.z3",
}
response = replay_counterexample({"envelope": envelope})
print(response.status, response.authority, response.diagnostics)
```

### 5.5 CLI parity (channel-neutral)

```bash
# List goal-tactician operations (declarative)
python -c "from ipfs_datasets_py.logic.verification_api import list_goal_tactician_cli_mcp_surface as s; import json; print(json.dumps(s()['cli_commands'], indent=2))"

# Equivalent CLI command names (when logic CLI is installed):
#   ipfs-datasets logic goal-list-operations
#   ipfs-datasets logic goal-formalize --request goal.json
#   ipfs-datasets logic goal-replay-counterexample --request cex.json
```

### 5.6 MCP tool names (parity)

MCP tools map 1:1 to operations via `GOAL_TACTICIAN_TOOL_TO_OPERATION`. Transport
success is never proof success; inspect `status` and `authority` in the tool
payload.

---

## 6. Proof-authority interpretation

Closed authority vocabulary (see also capability inventory and rollout docs):

| Authority / claim | Established by | Must never authorize |
| --- | --- | --- |
| `none` / empty | Invalid or non-attempt | Anything authoritative |
| `advisory` / `candidate` | Models, advisors, drafts | Proof success, completion |
| `declarative` | Catalogs, frozen schemas | Runtime tool presence |
| `bounded` / `satisfiability` | SMT under exact bounds | Kernel theorem |
| `model_check` | Bounded state exploration | Unbounded liveness theorems |
| `monitor` | Finite-trace monitors | Source translation correctness |
| `authorization` | Policy engines | Ambient trust |
| `protocol` / `hyperproperty` | Protocol / hyper tools | Network deployment success |
| `reconstruction` | Reconstruction request path | Unchecked hammer search |
| `attestation` | ZKP binding to a receipt | Raising that receipt’s proof class |
| `theorem` | Kernel-checked artifact | Advisor transport success |

Adapters, transports, caches, and attestations **preserve** authority; they
never increase it.

---

## 7. End-goal authoring guidance

1. Bind **source** (path, content digest, language) — prose alone is incomplete.
2. State **property kind** and supported fragment; unsupported constructs must
   fail closed as `unsupported`.
3. List **assumptions** explicitly (including the empty list).
4. Set **resource bounds** (timeout, memory, candidate counts, iteration caps).
5. Prefer confirmed formalizations before Leanstral / model goal development.
6. When ambiguity exists, call `compare_interpretations` and select a material
   interpretation before `plan_proof`.
7. Treat every favorable premise as an obligation until independently validated
   and re-checked.

---

## 8. Missing-proof review

`discover_missing_proofs` emits typed proof holes (contracts, invariants,
lemmas, frames, bridge facts). Review checklist:

- [ ] Hole type and dependency edges are explicit.
- [ ] Candidate synthesis sources are labeled (template, SMT core, Houdini,
      legal evidence routing, learned proposal).
- [ ] Learned / legal routes remain non-authoritative until independent
      validation.
- [ ] `validate_proof_candidate` runs on a path independent of the synthesizer.
- [ ] Plan ranking (`plan_proof`) prefers higher authority utility without
      hiding open holes.

---

## 9. Counterexample replay and privacy

Public counterexample envelopes:

- Bind property, source tree, assumptions, bounds, and provider identity.
- Expose only **public** witness material; raw source dumps, credentials,
  tokens, and private witnesses never enter public API responses or model
  contexts.
- State minimization guarantee: `none`, `normalized`, `bounded`,
  `locally_minimal`, or `globally_minimal`.
- `replay_counterexample` re-executes under the same bounds; bound changes keep
  the case open/unknown rather than silently succeeding.
- CEGIS closure requires a **fresh** receipt from the originating verifier
  class; structural admissibility alone does not reduce open-witness count.

---

## 10. Provider and toolchain setup

1. **Declare** providers via the backend registry (import-safe).
2. **Probe** with `probe_provider` (opt-in, bounded).
3. **Install** only through explicit installer surfaces (`install_provider` or
   documented toolchain installers) — never from inventory tests.
4. Pin hermetic identities in `config/formal_verification_toolchains.lock.json`.
5. Certify multi-prover matrices with
   `tools/logic/certify_formal_verification_toolchains.py` when live tools are
   required.
6. Consult `docs/architecture/formal_verification_readiness_baseline.json` for
   this machine’s status ladder; do not treat repository presence as usable.

Unsupported languages and tools must be disclosed as `unsupported` /
`unavailable`. Do not promise tool families that are not in the locked matrix
or capability census.

---

## 11. Supervisor operations (summary)

Full procedures live in
`docs/operations/formal_verification_tactician_runbook.md`.

| Concern | Module | Invariant |
| --- | --- | --- |
| Orchestration | `goal_directed_tactician` | Exact cache keys; no model/cache bypass of validation |
| Lifecycle | `goal_tactician_lifecycle` | Restart replays identical authoritative state; stale workers cannot mutate |
| CEGIS | `counterexample_guided_tactician` | Closure only via verifier-backed receipts |
| Replanning | `formal_replanner` | Verifier-backed repair closure semantics |
| Metrics | `goal_tactician_metrics` | Observability without timing-as-correctness |

Plan statuses: `open`, `running`, `blocked`, `completed`, `failed`,
`invalidated`. Control signals: `none`, `cancelled`, `timed_out`,
`backpressure`. Receipt freshness: `fresh`, `stale_tree`, `stale_epoch`,
`stale_worker`, `inadequate`.

---

## 12. Related tests and evidence

| Validation | Path |
| --- | --- |
| This documentation contract | `test/api/test_formal_verification_tactician_docs.py` |
| Goal tactician public API | `ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py` |
| CLI/MCP parity | `test/api/test_goal_tactician_cli_mcp_parity.py` |
| Lifecycle / restart | `test/api/test_goal_tactician_supervisor_lifecycle.py`, `test_goal_tactician_supervisor_restart.py` |
| CEGIS loop | `test/api/test_counterexample_guided_tactician.py` |
| Receipt authority | `test/api/test_logic_receipt_authority_boundary.py` |
| Readiness baseline | `test/api/test_formal_verification_readiness_baseline.py` |

---

## 13. Non-goals and conflict policy

- Own tactician/readiness documentation and documentation tests.
- Preserve legacy public names through documented compatibility aliases
  (`proof_tactician_migration.md`).
- Do not promise unsupported languages/tools.
- Do not mark implementation complete as deployment certified.
- Do not route legal evidence as formal theorem authority.
- Ticket IDs from objective heaps are tracking metadata, not product vocabulary
  required to operate the system.
