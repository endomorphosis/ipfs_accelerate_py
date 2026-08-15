# LPC-110 SupervisorLogicPlatformClient@1

**Task:** LPC-110 — Implement SupervisorLogicPlatformClient  
**Goal:** LPC-G110  
**Depends on:** LPC-052 (provider responses), LPC-090 (canonical adapter maps), LPC-100 (manifest handshake)  
**Interface:** `SupervisorLogicPlatformClient@1`  
**Evidence module (LPC-110 owned):** `ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py`  
**Tests (LPC-110 owned):** `test/api/test_supervisor_logic_platform_client.py`  
**Acceptance:** Client supports handshake, catalog, formalization, slice/obligation/plan, capability discovery, typed invocation, reconstruction, verification, receipts, counterexamples, and cache freshness.  
**Conflict policy:** Own the client module and tests. Do not create another supervisor.  
**Repair context:** LPC-174 resolves the validation retry-budget blocker filed after
repeated LPC-110 `proposal_gate_failed` attempts (validation never ran; rc 78)
that expanded beyond the task-owned path envelope or omitted this declared
note. This note freezes the client contract so LPC-110 can re-admit against an
exact ownership and non-overclaim surface without weakening production policy.  
**Validation (LPC-110):**  
`python -m pytest test/api/test_supervisor_logic_platform_client.py -q`  
**Validation (LPC-174 repair gate):**  
`test -f data/agent_supervisor/logic_platform_canonicalization/state/discovery/2026-08-15-lpc-174-lpc-110-retry-budget.md`

## Purpose

Supervisors need **one lazy client** for the datasets logic platform. The client
is the single supervisor-side entry that:

1. Handshakes against `LogicPlatformManifest@1` (LPC-100) before any semantic work.
2. Projects residual vocabularies through `SupervisorCanonicalLogicAdapter@1`
   (LPC-090) without reintroducing hand-maintained family/provider lists.
3. Invokes typed platform operations (catalog, formalization, slice/obligation/plan,
   capability, prove/translate/reconstruct/verify/attest) through existing
   facades — never a second supervisor runtime.
4. Surfaces receipts, counterexamples, and cache-freshness signals without
   promoting success into proof authority (LPC-032).
5. Binds every request to task / tree / policy / plan / budget / network /
   cancellation / deadline / correlation / evidence / authority so callers
   cannot overclaim.

This note is the durable LPC-110 / LPC-174 declared output. It freezes the
interface, request-binding rules, authority floors, ownership envelope, and
non-goals. Module and test implementation remain under LPC-110 predicted files
when the source task is re-dispatched after this repair releases it from
strategy `blocked_tasks`.

## Canonical call path

```text
Supervisor caller
  → SupervisorLogicPlatformClient@1
      1. handshake(requirements?)          # LogicPlatformManifest@1 (LPC-100)
      2. project residual maps (lazy)      # SupervisorCanonicalLogicAdapter@1 (LPC-090)
      3. typed platform operation
           catalog | formalize | slice | obligation | plan
           | capability | translate | prove | reconstruct | verify | attest
           | receipts | counterexamples | cache freshness
      4. provider facade conversion        # SupervisorLogicProviderFacade (existing)
      5. response envelope (LPC-052)       # no success ⇒ proof
  → optional receipt admission (LPC-111)   # ten-point gate before merge/completion
```

Importing the client module must remain quiet: no network, install, probe, or
datasets package load until an explicit boundary call (same lazy rule as
`canonical_logic_adapter.py` and `logic_provider_contract.py`).

## Surface

| Symbol | Role |
| --- | --- |
| `SupervisorLogicPlatformClient` | Lazy supervisor-side client; stable declared identity before datasets load |
| `SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE` | Exact interface id `SupervisorLogicPlatformClient@1` |
| `ClientRequestContext` | Immutable binding of task/tree/policy/plan/budget/network/cancellation/deadline/correlation/evidence/authority |
| `ClientOperation` | Closed operation vocabulary matching manifest `operation_versions` plus catalog/formalization/slice/plan helpers |
| `ClientResult` | Typed envelope: ok/error, residual identity, authority ceiling, freshness, non-simulated flag |
| `handshake(...)` | First step; delegates to platform `handshake` / default manifest |
| `get_supervisor_logic_platform_client(...)` | Process-local factory (optional); does not create a second supervisor |

Interface / schema constants:

* `SupervisorLogicPlatformClient@1`
* Listed in `LogicPlatformManifest.compatible_adapter_versions` (LPC-100)
* Task / goal binding: `LPC-110` / `LPC-G110`

## Request binding (fail closed)

Every non-handshake client call binds a `ClientRequestContext`. Missing or
contradictory bindings raise a typed client error; they do not soft-succeed.

| Binding | Required | Rule |
| --- | --- | --- |
| Task identity | yes | `task_id` matches the calling supervisor task authority |
| Repository tree | yes | `repository_tree_id` matches the admitted tree |
| Policy | yes | `policy_id` / policy revision cannot be caller-forged above the bound policy |
| Plan | when planned work | `plan_id` / accepted plan when the operation is plan-scoped |
| Resource budget | yes | Cannot exceed supervisor-granted budget |
| Network policy | yes | Default deny; network only when explicitly allowed |
| Cancellation | yes | Snapshot or live token; cancelled work fails closed |
| Deadline | yes | Unix-ms or equivalent; expired work fails closed |
| Correlation | yes | Correlation / request id for receipt lineage |
| Evidence kind | when claiming results | Kind must support the claimed verdict (LPC-032) |
| Authority ceiling | yes | Caller cannot request a ceiling above the bound policy / evidence |

Authority non-overclaim:

1. Catalog presence ≠ provider availability.
2. Provider `operation_status=succeeded` ≠ `semantic_verdict=proved`.
3. Simulated / advisory / candidate evidence cannot satisfy kernel-required
   admission (LPC-032; LPC-111 owns the full ten-point receipt gate).
4. Handshake compatibility never implies proof authority.
5. Residual map projection never invents canonical identities for unknown
   supervisor values (LPC-090 fail-closed).

## Operation matrix

Operations the client must expose (acceptance surface). Implementations may
forward to platform services / facades; they must not reimplement family provers
or open a parallel supervisor.

| Operation | Purpose | Authority notes |
| --- | --- | --- |
| `handshake` | Manifest compatibility (LPC-100) | Typed incompatibilities; default path needs no Git |
| `catalog` | Read sealed catalog / content root | Declaration only; not executability |
| `formalize` | FormalizationArtifact@3 / domain slice path | Through admitted typed write path (LPC-040+) |
| `slice` / `obligation` / `plan` | Create admitted slices, obligations, plans | No free-form bypass of BackendRequest@2 |
| `capability` | Capability discovery | Non-executable; does not install or probe |
| `translate` / `prove` / `reconstruct` / `verify` / `attest` | Typed provider invocation | Via provider facade + protocol @2 elevation |
| `receipts` | Fetch / project receipt envelopes | Untrusted until LPC-111 admission |
| `counterexamples` | Counterexample projection | Evidence kind constrained |
| `cache_freshness` | Freshness / invalidation signals | Supervisor owns placement/single-flight (LPC-080); datasets owns semantic keys |

## Ten-point receipt floor (client must not short-circuit)

LPC-111 owns `logic_platform_admission` enforcement. The client must **never**
treat a raw provider success as merge/completion authority. The human-plan
ten-point floor (must all hold before influence on completion/merge):

1. Receipt is structurally valid.
2. Content identity is valid.
3. Source / tree / environment / policy bindings match.
4. Translation chain is valid.
5. Evidence kind supports the claimed verdict.
6. Authority ceiling is adequate.
7. Required reconstruction or kernel checks passed.
8. It is not stale.
9. It is not simulated.
10. Supervisor policy admits that authority for the requested operation.

Until LPC-111 lands, the client returns receipts as untrusted envelopes and
labels authority ceilings honestly.

## Dependency surfaces (already landed)

| Task | Surface | Client consumption |
| --- | --- | --- |
| LPC-100 | `ipfs_datasets_py.logic.platform.manifest` | First lazy step: `handshake` / default manifest |
| LPC-090 | `proof/canonical_logic_adapter.py` | Residual ↔ canonical projection; no hand lists |
| LPC-052 | Provider response envelopes | Typed ok/error; no success-implies-proof |
| LPC-062 / LPC-060 | Platform facades / services | Typed ops forward through facades |
| LPC-080 | Cache-key contract | Freshness helpers respect datasets semantic keys |

## Lazy import and process boundaries

| Rule | Requirement |
| --- | --- |
| Quiet import | Importing `logic_platform_client` does not import `ipfs_datasets_py` |
| Explicit boundary | Datasets load only on handshake/catalog/invoke paths |
| No second supervisor | Client does not own scheduling, merge, or daemon loops |
| No sibling/Git requirement | Handshake follows LPC-100 wheel / no-Git rules |
| Facade reuse | Provider wire conversion reuses `SupervisorLogicProviderFacade` |

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_client.md` | This contract note (LPC-110 / LPC-174 declared output) |
| `ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py` | Client module (LPC-110 predicted; implement on source-task re-dispatch) |
| `test/api/test_supervisor_logic_platform_client.py` | Focused regression suite (LPC-110 predicted; create with the module) |

Task-owned proposal envelope for LPC-110 (fail closed):

* **Declared Outputs:** this note only.
* **Predicted files (write-eligible with the note):** client module + client tests + this note.
* Paths outside that envelope (other proof modules, daemon code, protected plan
  files, undeclared companions) are **out of scope** for LPC-110 admission.
* LPC-111 owns `logic_platform_admission.py` and its tests; do not absorb
  admission into the client proposal.

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-090 | Residual maps; client projects through the lazy adapter |
| LPC-091 | Type classification; client does not reintroduce hand maps |
| LPC-100 | Manifest handshake is the first client step |
| LPC-052 | Provider response contract |
| LPC-062 | Outer facades remain the public datasets surface |
| LPC-111 | Receipt admission (ten-point) after client returns envelopes |
| LPC-174 | This repair: emit declared note; release LPC-110 from `blocked_tasks` |

## What this task does **not** do

* Does not create another supervisor, daemon, merge train, or scheduler.
* Does not reimplement family provers, catalog generation, or map tables.
* Does not weaken LPC-032 non-inference (success is not proof).
* Does not require Git, sibling checkouts, or monorepo layout for handshake.
* Does not claim provider availability or production readiness from catalog or
  handshake compatibility alone.
* Does not implement LPC-111 admission helpers (separate predicted files).
* Does not edit protected board/plan/validator files.

## LPC-174 repair notes

| Finding | Resolution |
| --- | --- |
| Failure kind | `proposal_validation_failed` / `proposal_gate_failed` (validation never ran; rc 78) |
| Observed attempts | 4 consecutive LPC-110 failures (retry budget 3) |
| Evidence | `data/agent_supervisor/logic_platform_canonicalization/state/discovery/2026-08-15-lpc-174-lpc-110-retry-budget.md` |
| Root cause | Pre-dispatch proposal gate rejected LPC-110 candidates that left the task-owned envelope (declared note + predicted client/test paths) or omitted the declared note output, so expensive validation never started |
| Repair | Emit this declared note only under `retry_repair_output_exact`; freeze the client contract, ownership envelope, and non-overclaim rules; preserve production policy and tests |
| Release effect | Completing LPC-174 releases LPC-110 from strategy `blocked_tasks` so the supervisor can re-admit the source task against the documented contract and predicted module/test paths |

## Acceptance

- Client interface id is exactly `SupervisorLogicPlatformClient@1` and remains
  listed among manifest `compatible_adapter_versions`.
- Handshake is the first semantic step and follows LPC-100 wheel / no-Git rules.
- Residual vocabulary flows only through `SupervisorCanonicalLogicAdapter@1`.
- Request context binds task/tree/policy/plan/budget/network/cancellation/
  deadline/correlation/evidence/authority; overclaim fails closed.
- Operation surface covers handshake, catalog, formalization, slice/obligation/plan,
  capability, typed invocation, reconstruction, verification, receipts,
  counterexamples, and cache freshness.
- Success never upgrades authority; simulated/stale/advisory evidence stays
  below kernel floors.
- LPC-110 validation (source task):  
  `python -m pytest test/api/test_supervisor_logic_platform_client.py -q`
- LPC-174 validation (repair task): evidence file present at the discovery path
  recorded above.
