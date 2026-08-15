# LPC-042 Security, Software, and Crypto Domain Adapter Conformance

**Task:** LPC-042 — Security, software, and crypto domain adapter conformance  
**Goal:** LPC-G040  
**Depends on:** LPC-040 (typed new-write path: `FormalizationArtifact@3` / `DomainLogicSlice@2`)  
**Acceptance:** Each domain keeps its ontology and lowers through `DomainLogicSlice@2`.  
**Conflict policy:** Keep contracts, STS, authorization, concurrency, separation, hyperproperties, protocols, and monitors distinct. Never collapse security ↔ software_verification ↔ crypto.  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir ipfs_datasets_py/tests/unit/logic/software_verification ipfs_datasets_py/tests/unit/logic/crypto_ir -q`

## Purpose

Security IR, software verification, and crypto IR each own a sealed domain
ontology. New formalization writes lower those ontologies **through**
`DomainLogicSlice@2` (LPC-040) without inventing a universal domain IR and
without silently remapping one domain’s families into another.

This note freezes adapter locations, domain identities, admitted route tables,
namespace axes, assumption axes, preservation/loss declarations, authority
ceilings, and non-collapse rules. It is the durable LPC-042 evidence for the
three domain adapters that feed backend requests via admitted slices only.

## Canonical lowering path

```text
Domain IR view / obligation
  → TypedExpression (family + profile from the domain route)
  → DomainLogicSlice@2   (DomainLogicSliceV2.from_typed_expression)
  → LogicObligation@2
  → BackendRequest@2
  → compiled / parsed / replay / authority lineage
```

Shared contract module: `ipfs_datasets_py.logic.formalization.artifacts_v3`  
(`DomainLogicSlice@2`, admission gates `require_admitted` / `validate_against`).

Every admitted slice binds (LPC-040 inventory):

| Binding | Required on admitted slice |
| --- | --- |
| Source identity | `document_id`, `source_digest` |
| Expression identity | `expression_id`, `expression_digest` |
| Namespace axes | `family`, `profile`, `property`, `view`, `notation` |
| Features / assumptions | `features`, `assumption_ids` |
| Unsupported extensions | empty when `status=admitted` |
| Status / content identity | `status=admitted`, `content_digest` |
| Domain | domain-specific id (see table below) |

Construction pattern used by all three adapters:

```text
DomainLogicSliceV2.from_typed_expression(
    expression,
    slice_id=...,
    domain=<domain_id>,
    document_id=...,
    source_digest=...,
    property=property_id(...),
    view=view_id(...),
    notation=notation_id(...),
    source_range=...,
    features=...,
    assumption_ids=...,
)
domain_slice.require_admitted()
domain_slice.validate_against(document=..., expression=...)
```

## Production adapter modules

Inventory LPC-004 predicted paths named `domain_slice.py`. Live accelerate
implementations live as `*LogicSlice@2` connectors that **emit**
`DomainLogicSlice@2` records. Those connectors are the production domain
adapters for LPC-042.

| Domain | Domain id | Adapter interface | Production module | Emits |
| --- | --- | --- | --- | --- |
| Security | `security_ir` | `SecurityLogicSlice@2` | `ipfs_datasets_py/logic/security_ir/logic_slice_v2.py` | `DomainLogicSlice@2` per admitted security route |
| Software verification | `software_verification` | `SoftwareVerificationLogicSlice@2` | `ipfs_datasets_py/logic/software_verification/logic_slice_v2.py` | `DomainLogicSlice@2` per base obligation |
| Crypto | `crypto_ir` | `CryptoLogicSlice@2` | `ipfs_datasets_py/logic/crypto_ir/formalization/logic_slice_v2.py` | `DomainLogicSlice@2` per crypto-network route |

Supporting ontology / route sources (not alternate DomainLogicSlice generations):

| Domain | Supporting modules | Role |
| --- | --- | --- |
| Security | `security_ir/formalization_adapter_v2.py` | `ADMITTED_SECURITY_VIEW_NAMES`, `resolve_security_route` admission cross-check |
| Software verification | `software_verification/syntax_bridge.py` | IR kind → family/profile route table (`default_ir_routes`) |
| Crypto | `crypto_ir/formalization/typed_adapter.py` | `CryptoNetworkViewKind` network views and formalization adapter |

Out of DomainLogicSlice generation scope (related surfaces, not adapters):

- `security_ir.cvefixes.*`, `security_ir.solidity_cpt_top10.*`, exchange/xaman pipelines
- `software_verification.tactician.*`, `software_verification.counterexamples.*`
- crypto chain adapters (`crypto_ir/adapters/*`) and compliance subpackages

## Shared adapter contract (all three domains)

Each route/obligation descriptor declares the LPC-G040 / LPC-041-class fields:

| Declaration | Where it lives | Rule |
| --- | --- | --- |
| Source domain | `domain` on `DomainLogicSlice@2` | Exact domain id; must match parent formalization artifact |
| View | route `view_name` → `view_id(...)` | Typed view namespace; never free-form |
| Family / profile | expression + slice (`family`, `profile`) | From the domain route table only; no new families |
| Property | route `property_name` → `property_id(...)` | Property is never promoted to a family |
| Notation | route `notation_name` → `notation_id(...)` | Surface notation for the admitted view |
| Preserved semantics | translation edge `preservation` | From reviewed translation catalog edge |
| Lost semantics | `_loss_ids_for(route)` on compiled target | Explicit loss ids; never silent |
| Assumptions | domain-specific assumption axes | Declared even when empty / N/A |
| Unsupported constructs | deferred kind sets | Rejected fail-closed (not admitted) |
| Proof-safety | `authority_ceiling` + `result_authority` | Ceiling never upgrades along lineage |
| Counterexample-safety | sat/model/trace result kinds + replay digests | Counterexamples remain bound to exact request digests |

Lineage stages required on every admitted end-to-end connection:

```text
typed_origin → semantics → translation → request → result → replay → authority_lineage
```

Hermetic fixtures may supply provider execution and replay without live provers.
Tool absence is an availability result, never a mock proof (LPC-032).

---

## 1. Security IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `security_ir` |
| Interface | `SecurityLogicSlice@2` |
| Schema | `security-logic-slice/v2` |
| Connector | `SecurityLogicSlice` in `security_ir/logic_slice_v2.py` |
| Formalization alignment | `ADMITTED_SECURITY_VIEW_NAMES` in `formalization_adapter_v2.py` |

### Ontology kept distinct

Security routes use security-scoped profiles and views. They share **logic
families** from the catalog (authorization, temporal, hyperproperty, …) but
never collapse into the software-verification domain id, IR package, or
software obligation table.

Distinct security surfaces:

| Route kind | Family | Profile | Property | View | Notation | Authority ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `threat` | `transition_system` | `threat_model` | `attacker_reachability` | `threat` | `security_threat_model` | bounded / model_check |
| `authorization` | `authorization` | `secpal` | `authorization` | `policy` | `secpal_surface` | authorization |
| `claim` | `first_order` | `verification_condition` | `validity` | `claim` | `security_claim` | satisfiability |
| `state` | `transition_system` | `default` | `invariant` | `transition` | `security_state` | bounded / model_check |
| `temporal` | `temporal` | `ltl` | `safety` | `transition` | `ltl_surface` | bounded / model_check |
| `protocol` | `cryptographic_protocol` | `default` | `secrecy` | `protocol` | `symbolic_protocol` | protocol |
| `noninterference` | `hyperproperty` | `noninterference` | `noninterference` | `hyperproperty` | `hyperltl_surface` | bounded / hyperproperty |
| `separation` | `separation_logic` | `default` | `frame` | `separation` | `separation_surface` | candidate |
| `concurrency` | `concurrency` | `default` | `rely_guarantee` | `concurrency` | `concurrency_ir` | bounded / model_check |

Namespace discipline notes:

- Verification-condition is a **view role** under `first_order` (claim route), never a family.
- Noninterference is a **property** under `hyperproperty`, never a family.
- Policy authority assumptions stay on authorization routes; mock authorization cannot upgrade authority.

### Assumption axes (security)

Every admitted security route declares all four axes (empty only when N/A):

| Axis | Examples |
| --- | --- |
| `information_flow` | high/low partition, observation equivalence, hypertrace quantifiers |
| `attacker` | attacker identity, Dolev–Yao, perfect cryptography, role/channel identity |
| `bound` | state space, attacker steps, trace length, interleavings, system copies |
| `policy_authority` | policy authority bound, principal identity, delegation scope |

### Preserved / lost semantics (security)

| Route | Explicit losses (`loss_ids`) |
| --- | --- |
| `separation` | `loss.heap_as_array`, `loss.sep_conj_to_and` |
| `protocol` | `loss.proverif_role_to_process`, `loss.proverif_attacker_ceiling`, `loss.proverif_query_renaming` |
| `noninterference` | `loss.bounded_self_composition`, `loss.alternation_restricted` |
| `concurrency` | `loss.bounded_interleaving` |
| `temporal` | `loss.bounded_trace` |
| `state` | `loss.finite_domain` |
| `threat` | `loss.bounded_attacker_steps` |

### Unsupported / deferred (security)

Rejected for executable `SecurityLogicSlice@2`:

`verification_condition` (as a free route kind), `graph_projection`,
`proof_translation`, `free_form`, `boolean_receipt`.

### Proof-safety and counterexample-safety (security)

- Authority ceilings are route-local (`BOUNDED`, `AUTHORIZATION`,
  `SATISFIABILITY`, `PROTOCOL`, `CANDIDATE`); lineage records
  `never_upgrades=true`.
- Separation remains candidate-class until stronger proof authority is
  admitted; protocol secrecy uses protocol authority, not SAT authority.
- Claim/VC unsat and threat reachability results stay digest-bound to the
  exact request; disproof surfaces require counterexample/witness bindings in
  `security_ir.results` (result family discipline).

---

## 2. Software verification adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `software_verification` |
| Interface | `SoftwareVerificationLogicSlice@2` |
| Schema | `software-verification-logic-slice/v2` |
| Connector | `SoftwareVerificationLogicSlice` in `software_verification/logic_slice_v2.py` |
| IR routes | `default_ir_routes()` in `software_verification/syntax_bridge.py` |

### Ontology kept distinct

Software verification keeps contracts, STS (state/transition), authorization,
concurrency, separation, hyperproperties, protocols, and monitors as **distinct
IR kinds and family/profile routes**. They are not flattened into a single
generic program IR.

Base executable obligations (Wave-2):

| Obligation | IR kind | Family (via bridge) | Profile | Property | View | Authority ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `contract` | `CONTRACT` | `program` | `dynamic_hoare` | `postcondition` | `source` | satisfiability |
| `vc` | `VC` | `program` | `wp_vc` | `validity` | `verification_condition` | satisfiability |
| `program` | `PROGRAM` | `program` | `program_ir` | `partial_correctness` | `source` | satisfiability |
| `state` | `STATE` | `transition_system` | `state_schema` | `invariant` | `source` | bounded / model_check |
| `separation` | `SEPARATION` | `separation_logic` | `separation` | `frame` | `separation` | satisfiability |
| `concurrency` | `CONCURRENCY` | `concurrency` | `rely_guarantee` | `rely_guarantee` | `source` | bounded / model_check |
| `refinement` | `REFINEMENT` | `refinement` | `simulation` | `forward_simulation` | `source` | bounded / model_check |
| `temporal` | `TEMPORAL` | `temporal` | `ltl` | `safety` | `source` | finite_trace / monitor |
| `counterexample` | `VC` | `program` | `wp_vc` | `counterexample` | `verification_condition` | satisfiability (sat model) |
| `kernel_target` | `VC` | `program` | `wp_vc` | `theorem` | `verification_condition` | candidate (Lean) |

Additional IR kinds admitted by the syntax bridge (ontology owners even when
not every kind is a base Wave-2 obligation):

| IR kind | Family | Profile | Notes |
| --- | --- | --- | --- |
| `TRANSITION` | `transition_system` | `action_system` | STS action systems stay typed |
| `TRACE` | `temporal` | `finite_trace` | Finite-trace monitors |
| `AUTHORIZATION` | `authorization` | `datalog` | Policy IR, not SecPAL security profile |
| `PROTOCOL` | `cryptographic_protocol` | `dolev_yao` | Protocol IR under SV domain id |
| `HYPERPROPERTY` | `hyperproperty` | `hyperltl` | Hyperproperty IR under SV domain id |
| `HEAP` | `separation_logic` | `heap_model` | Heap model owner |

Conflict-policy surfaces that remain distinct (not collapsed):

- **Contracts** — program `dynamic_hoare` / postcondition discharge  
- **STS** — `transition_system` state schema / action systems  
- **Authorization** — `authorization`/`datalog` IR (SV domain)  
- **Concurrency** — rely/guarantee with interference + fairness assumptions  
- **Separation** — heap/resource frame with explicit heap losses  
- **Hyperproperties** — hyperltl IR (bridge); not security noninterference profile  
- **Protocols** — Dolev–Yao protocol IR under SV domain  
- **Monitors** — temporal finite-trace / runtime MTL authority  

### Assumptions (software verification)

Route-local `assumption_ids` (examples):

| Obligation | Assumptions |
| --- | --- |
| `contract` | `assumption:frame_closed` |
| `vc` | `assumption:wp_fragment` |
| `program` | `assumption:bounded_integers` |
| `state` | `assumption:finite_domain`, `bound:state_space` |
| `separation` | `assumption:heap_as_array`, `assumption:locations_disjoint` |
| `concurrency` | `assumption:interference_declared`, `assumption:weak_fairness`, `bound:interleavings` |
| `refinement` | `assumption:refinement_direction_forward`, `bound:refinement_steps` |
| `temporal` | `assumption:finite_trace`, `bound:trace_length` |
| `counterexample` | `assumption:model_is_counterexample` |
| `kernel_target` | `assumption:official_kernel_sole_authority`, `assumption:candidate_until_acceptance` |

### Preserved / lost semantics (software verification)

| Obligation | Explicit losses |
| --- | --- |
| `separation` | `loss.heap_as_array`, `loss.sep_conj_to_and` |
| `kernel_target` | `loss.candidate_until_kernel_acceptance`, `loss.trust_escape_rejected` |
| `refinement` | `loss.bounded_refinement` |
| `concurrency` | `loss.bounded_interleaving` |
| `temporal` | `loss.finite_trace` |

VC remains a **view role**, never a family (`notes` on route and bridge).

### Unsupported / deferred (software verification)

Deferred to later session/process overlays (not admitted here):

`session`, `process`, `linear_session`, `session_process`.

Free-form typed origins are rejected (`weakens_to_free_form=false` on the
connector wire form).

### Proof-safety and counterexample-safety (software verification)

- Kernel targets are **candidates** until official kernel acceptance; they do
  not mint proof authority.
- Temporal monitors use `FINITE_TRACE` / `MONITOR` authority, not unbounded
  model-check claims.
- Counterexample obligations bind sat models and replay receipts to the exact
  request digest; counterexample minimization/replay packages under
  `software_verification.counterexamples.*` remain outside DomainLogicSlice
  generation scope.

---

## 3. Crypto IR adapter

### Identity

| Field | Value |
| --- | --- |
| Domain id | `crypto_ir` |
| Interface | `CryptoLogicSlice@2` |
| Schema | `crypto-logic-slice/v2` |
| Connector | `CryptoLogicSlice` in `crypto_ir/formalization/logic_slice_v2.py` |
| Network views | `CryptoNetworkViewKind` via `crypto_ir/formalization/typed_adapter.py` |

### Ontology kept distinct

Crypto keeps a **crypto-network** ontology: ledger, balances, consensus,
finality/reorg, bridges, wallets, permissions, symbolic protocols, arithmetic,
and privacy hyperproperties. Profiles are crypto-scoped
(`crypto_network_*`) so routes cannot be mistaken for security_ir or
software_verification obligations even when they reuse catalog families.

| Route | Network view | Family | Profile | Property | View | Authority ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `ledger` | `transactions` | `transition_system` | `crypto_network_transactions` | `safety` | `source` | bounded / model_check |
| `balances` | `balances` | `first_order` | `crypto_network_balances` | `invariant` | `verification_condition` | satisfiability |
| `consensus` | `consensus` | `transition_system` | `crypto_network_consensus` | `safety` | `source` | bounded / model_check |
| `finality` | `reorg_finality` | `transition_system` | `crypto_network_reorg_finality` | `safety` | `source` | bounded / model_check |
| `bridges` | `bridges` | `cryptographic_protocol` | `crypto_network_bridges` | `safety` | `protocol` | protocol |
| `wallets` | `wallets` | `authorization` | `crypto_network_wallets` | `authorization` | `source` | authorization |
| `authorization` | `permissions` | `authorization` | `crypto_network_permissions` | `authorization` | `source` | authorization |
| `protocol` | `symbolic_protocols` | `cryptographic_protocol` | `crypto_network_symbolic_protocol` | `secrecy` | `protocol` | protocol |
| `arithmetic` | `arithmetic` | `first_order` | `crypto_network_arithmetic` | `safety` | `verification_condition` | satisfiability |
| `hyperproperty` | `privacy` | `hyperproperty` | `crypto_network_privacy` | `noninterference` | `hyperproperty` | bounded / hyperproperty |

Evidence subset required on the crypto slice catalog: ledger, consensus,
finality, protocol, arithmetic, hyperproperty.

### Assumption axes (crypto)

Every admitted crypto route declares all six axes (N/A still explicit):

| Axis | Role |
| --- | --- |
| `network_chain` | Declared chain rules, fork choice, per-chain consensus, closed ledger world |
| `arithmetic_domain` | Linear integer / bitvector / N/A |
| `adversary` | None, Byzantine coalition, Dolev–Yao, reorg withholding, relational observer |
| `trace` | Finite trace / hypertrace window / N/A |
| `finality` | Observation bound, protocol finality predicate, source-finality-before-mint / N/A |
| `approximation` | Finite step window, validator set, reorg depth, bitwidth, perfect-crypto overapprox |

### Preserved / lost semantics (crypto)

| Route | Explicit losses |
| --- | --- |
| `protocol` | Proverif role/process/attacker/query losses |
| `bridges` | Proverif losses + `loss.bridge_per_chain_consensus` |
| `hyperproperty` | Bounded self-composition + `loss.not_computational_zk` |
| `ledger` | `loss.finite_step_window`, `loss.observation_bound_finality` |
| `consensus` | `loss.finite_validator_set`, `loss.bounded_rounds` |
| `finality` | `loss.max_reorg_depth`, `loss.finite_chain_growth` |
| `arithmetic` | `loss.finite_domain_or_bitwidth` |
| `balances` | `loss.finite_account_set` |

Wallet authorization does **not** claim ledger consensus or finality
(those axes are declared N/A). Bridge consensus identities remain per-chain
and are never merged silently. Privacy hyperproperties do not claim
computational zero-knowledge.

### Unsupported / deferred (crypto)

Deferred (finite-field / ZK overlays and free-form):

`finite_field`, `finite_field_constraint`, `zk`, `zkp`, `zk_constraint`,
`zero_knowledge`, `probabilistic`, `boolean_receipt`, `free_form`.

### Proof-safety and counterexample-safety (crypto)

- Bounded model-check and SAT results carry explicit finite bounds; deeper
  reorgs, larger validator sets, or unbounded asynchrony are excluded, not
  implied.
- Protocol/bridge routes use protocol authority with Dolev–Yao attacker
  semantics; they do not mint cryptographic break claims.
- Authorization/permission routes use authorization authority only.
- Privacy hyperproperty results are relational/bounded; computational ZK is an
  explicit loss, not an admitted claim.

---

## Non-collapse rules (security ↔ software ↔ crypto)

| Rule | Enforcement |
| --- | --- |
| Distinct domain ids | `security_ir` ≠ `software_verification` ≠ `crypto_ir` on every admitted slice |
| Distinct connector interfaces | `SecurityLogicSlice@2` / `SoftwareVerificationLogicSlice@2` / `CryptoLogicSlice@2` |
| Shared families, domain-local profiles | Catalog families may overlap (e.g. `authorization`, `hyperproperty`); profiles and views remain domain-scoped |
| No universal domain IR | Free-form / universal routes are deferred or rejected |
| Property ≠ family | VC, noninterference, monitors stay roles/properties |
| Security ≠ software | Security threat/policy/claim views do not emit `software_verification` domain slices; SV IR kinds do not emit `security_ir` |
| Crypto ≠ security protocol generic | Crypto routes use `crypto_network_*` profiles and network/finality/adversary axes security routes do not own |
| SV distinct obligation surfaces | Contracts, STS, authorization, concurrency, separation, hyperproperties, protocols, monitors remain separate IR/route owners |
| No new families | Adapters only select existing catalog families (LPC-G040) |

Forbidden silent mappings:

| From | Must not silently become |
| --- | --- |
| Security noninterference | Software hyperproperty IR (or vice versa) without domain rebinding |
| Security protocol | Crypto bridge/network protocol without crypto assumptions |
| Software contract/VC | Security claim FOL without security domain + assumptions |
| Crypto consensus/finality | Generic transition-system model-check without network/finality axes |
| Any domain | Free-form text as typed origin |

## End-to-end admission checklist (per domain)

For each admitted route/obligation the connector must:

1. Build a `SourceDocument` + `TypedExpression` with the route family/profile.
2. Emit `DomainLogicSlice@2` via `from_typed_expression` with the domain id.
3. Call `require_admitted()` and `validate_against(document, expression)`.
4. Lower through `LogicObligationV2.from_slice` → `BackendRequestV2.from_obligation`.
5. Attach translation-edge preservation and explicit `loss_ids`.
6. Record hermetic execution/replay without authority upgrade.
7. Cover all seven lineage stages with digest coherence source → request → execution → replay.

Incomplete slices fail closed before backend request construction (LPC-044
rejects executable requests without an admitted `DomainLogicSlice@2`).

## File ownership (LPC-042)

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/security_ir/logic_slice_v2.py` | Security domain adapter → `DomainLogicSlice@2` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/logic_slice_v2.py` | Software-verification domain adapter → `DomainLogicSlice@2` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/crypto_ir/formalization/logic_slice_v2.py` | Crypto domain adapter → `DomainLogicSlice@2` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Shared `DomainLogicSlice@2` contract (preserve; LPC-040) |
| `data/agent_supervisor/logic_platform_canonicalization/notes/security_software_crypto_adapters.md` | This conformance note |

Inventory aliases (`security_ir.domain_slice`, `software_verification.domain_slice`,
`crypto_ir.domain_slice`) refer to the adapter role satisfied by the
`logic_slice_v2` modules above; those modules are the production write path
for admitted domain lowering under LPC-042.

## Acceptance

- **Security** keeps its threat, authorization, claim, STS, temporal, protocol,
  noninterference, separation, and concurrency ontology and lowers each
  admitted route through `DomainLogicSlice@2` with domain `security_ir`.
- **Software verification** keeps contracts, STS, authorization, concurrency,
  separation, hyperproperties, protocols, and monitors distinct and lowers
  base obligations through `DomainLogicSlice@2` with domain
  `software_verification`.
- **Crypto** keeps ledger, balances, consensus, finality, bridges, wallets,
  permissions, protocol, arithmetic, and privacy ontology and lowers each
  admitted route through `DomainLogicSlice@2` with domain `crypto_ir`.
- No adapter invents a universal domain IR or collapses another domain’s
  ontology.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/security_ir ipfs_datasets_py/tests/unit/logic/software_verification ipfs_datasets_py/tests/unit/logic/crypto_ir -q`
