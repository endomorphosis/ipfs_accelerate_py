# LPC-120 Derive Hammer Adapter Vocabularies from the Catalog

**Task:** LPC-120 — Derive Hammer adapter vocabularies from the catalog  
**Goal:** LPC-G120  
**Depends on:** LPC-090 (`SupervisorCanonicalLogicAdapter@1`), LPC-052 (`LogicProviderResponse@2`)  
**Adapter module:** `ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider`  
**Interface:** `DatasetsLogicProvider@1`  
**Schema:** `ipfs_accelerate_py/agent-supervisor/hammer-adapter-vocab@1`  
**Validation:** `python -m pytest test/api/test_ipfs_datasets_logic_provider.py -q`

## Purpose

Replace hand-maintained Hammer adapter inventories for **logic families**,
**translation / encoding labels**, **solver / provider ids**, and **authority
ceilings** with a **catalog-derived, fail-closed projection**. Residual wire
tokens exist only for lossless dual-read of historical Hammer spellings; they
are not a second authority and must not invent identities outside
`CanonicalLogicCatalogSnapshot@1`.

Hammer remains **candidate-producing**. ATP/SMT portfolio results never become
kernel-verified proof authority. Independent reconstruction stays outside this
adapter (LPC-G120 acceptance).

This note is the durable vocabulary artifact. Machine-readable `hammer-vocab`
fences are exhaustive for the closed Hammer surfaces below. Tests parse every
fence, bind each row to the sealed catalog root, derive the same sets from the
live catalog, and verify the live adapter exports only catalog-projected
identities (plus documented residual wire aliases).

## Catalog root binding

All vocabulary rows bind to the sealed catalog content root from
`CanonicalLogicCatalogSnapshot@1`:

| Field | Authority |
| --- | --- |
| `catalog_root` | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root` (CIDv1) |
| `catalog_digest` | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest` (`sha256:…`) |
| `catalog_interface` | `CanonicalLogicCatalogSnapshot@1` |
| `supervisor_maps` | LPC-090 `SupervisorCanonicalLogicAdapter@1` |
| `provider_responses` | LPC-052 `LogicProviderResponse@2` (untrusted default authority) |

Document-level binding token used in every row:

```hammer-vocab-meta
schema: ipfs_accelerate_py/agent-supervisor/hammer-adapter-vocab@1
task: LPC-120
goal: LPC-G120
adapter_interface: DatasetsLogicProvider@1
adapter_module: ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider
provider_id: hammer
catalog_interface: CanonicalLogicCatalogSnapshot@1
catalog_root_binding: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
catalog_digest_binding: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest
supervisor_adapter_interface: SupervisorCanonicalLogicAdapter@1
response_interface: LogicProviderResponse@2
fail_closed: true
unknown_policy: raise HammerAdapterVocabError
candidate_producing: true
candidate_authoritative: false
authority_ceiling: advisory
```

In each `hammer-vocab` row, `catalog_root: $catalog_root_binding` means the
live sealed content root. Tests expand the binding against the installed
snapshot and reject any row that omits it.

## Row vocabulary

Each residual wire value maps to **all five** required fields:

| Field | Meaning |
| --- | --- |
| `canonical_identity` | Catalog id (family / encoding / notation / provider / evidence / authority) |
| `axis` | Semantic axis (`family`, `encoding`, `notation`, `provider`, `lane`, `evidence_kind`, `evidence_authority`) |
| `disposition` | How the wire value is admitted (`map`, `residual_alias`, `residual_collapse`, `ceiling`) |
| `residual` | Exact Hammer / supervisor wire token required for reverse mapping |
| `catalog_root` | Sealed catalog content root binding (`$catalog_root_binding`) |

### Disposition rules

| Disposition | Behavior |
| --- | --- |
| `map` | Direct projection onto a catalog identity; residual stores wire id |
| `residual_alias` | Alternate spelling of a primary wire value; projects through the primary residual |
| `residual_collapse` | Multiple wire values share one canonical id; residual is mandatory for reverse map |
| `ceiling` | Hard authority / evidence ceiling; never promoted by catalog presence |

### Fail-closed policy

1. **Known label only.** A mapper accepts only labels listed for its domain.
2. **Unknown → error.** Empty, wrong-domain, or unlisted labels raise.
3. **No silent merge of distinct formalisms.** F-logic and frame keep distinct
   wire residuals even when both project to `frame_logic`. DCEC and deontic
   never share a canonical family id.
4. **No axis collapse.** Family ≠ encoding ≠ notation ≠ provider ≠ lane ≠
   evidence kind ≠ evidence authority.
5. **No authority promotion.** Mapping never upgrades evidence authority above
   the Hammer ceiling (`advisory`) and never invents a conclusive verdict.
6. **Catalog presence ≠ availability / proof.** Binding a catalog root never
   claims live prover availability, production admission, or kernel authority.
7. **Candidate-producing only.** Portfolio success remains untrusted until
   independent reconstruction.

## Domain inventory

| Domain | Hammer / adapter surface | Catalog target axis | Source of truth |
| --- | --- | --- | --- |
| `logic_family` | `LogicFamily` / `SUPPORTED_LOGIC_FAMILIES` | family ids | catalog families + LPC-090 analysis_family |
| `translation_family` | `SUPPORTED_TRANSLATION_FAMILIES` | encoding / notation | catalog encodings + notations |
| `translation_alias` | `_FAMILY_ALIASES` residual spellings | encoding / notation | catalog aliases + residual wire |
| `solver_provider` | `KNOWN_HAMMER_SOLVERS` | provider ids | catalog providers + executable matrix |
| `solver_alias` | `_SOLVER_ALIASES` residual spellings | provider ids | matrix aliases (`e` → `eprover`) |
| `target_itp` | `target_itp` / `_FAMILY_ITP` | provider / encoding | catalog providers + encodings |
| `family_target` | `_FAMILY_TARGET` residual target map | encoding / notation | catalog encodings + notations |
| `authority_ceiling` | Hammer portfolio authority | evidence_authority | provider catalog advisory ceiling |
| `evidence_kind` | portfolio evidence default | evidence kind | catalog evidence ids |
| `semantic_separation` | non-collapse invariants | multi-axis | LPC-G120 acceptance |

---

## Mapping tables

Machine-readable blocks use fenced `hammer-vocab` sections. Label lines are:

```text
wire_value: canonical_identity=<id>; axis=<axis>; disposition=<d>; residual=<token>; catalog_root=$catalog_root_binding
```

Residual uses compact `k=v` pairs joined by `|`. Tests parse every block.

### logic_family — analysis LogicFamily (catalog-projected)

Source: LPC-090 `analysis_family` projection onto catalog family ids. Wire
values remain the supervisor `LogicFamily` tokens; canonical ids come from the
catalog (never a free-form Hammer family list).

```hammer-vocab
domain: logic_family
surface: supervisor.LogicFamily
fail_closed: true
catalog_root: $catalog_root_binding
tdfol: canonical_identity=tdfol; axis=family; disposition=map; residual=wire_id=tdfol|supervisor_enum=LogicFamily|supervisor_member=TDFOL; catalog_root=$catalog_root_binding
dcec: canonical_identity=dcec; axis=family; disposition=map; residual=wire_id=dcec|supervisor_enum=LogicFamily|supervisor_member=DCEC; catalog_root=$catalog_root_binding
flogic: canonical_identity=frame_logic; axis=family; disposition=residual_collapse; residual=wire_id=flogic|supervisor_enum=LogicFamily|supervisor_member=FLOGIC|collapse_group=frame_logic; catalog_root=$catalog_root_binding
modal: canonical_identity=modal; axis=family; disposition=map; residual=wire_id=modal|supervisor_enum=LogicFamily|supervisor_member=MODAL; catalog_root=$catalog_root_binding
deontic: canonical_identity=deontic; axis=family; disposition=map; residual=wire_id=deontic|supervisor_enum=LogicFamily|supervisor_member=DEONTIC; catalog_root=$catalog_root_binding
frame: canonical_identity=frame_logic; axis=family; disposition=residual_collapse; residual=wire_id=frame|supervisor_enum=LogicFamily|supervisor_member=FRAME|collapse_group=frame_logic; catalog_root=$catalog_root_binding
kg: canonical_identity=supervisor.kg; axis=family; disposition=map; residual=wire_id=kg|supervisor_enum=LogicFamily|supervisor_member=KNOWLEDGE_GRAPH|namespaced_extension=true; catalog_root=$catalog_root_binding
event_calculus: canonical_identity=event_calculus; axis=family; disposition=map; residual=wire_id=event_calculus|supervisor_enum=LogicFamily|supervisor_member=EVENT_CALCULUS; catalog_root=$catalog_root_binding
```

### translation_family — Hammer translation / encoding wire labels

Translation families are **target encodings / source notations**, never logic
families. Canonical identities are catalog encoding or notation ids. Residual
wire ids preserve historical Hammer portfolio spellings (`smtlib2`, `lean4`,
`tptp`, `coq`, …).

```hammer-vocab
domain: translation_family
surface: hammer.SUPPORTED_TRANSLATION_FAMILIES
fail_closed: true
catalog_root: $catalog_root_binding
coq: canonical_identity=rocq; axis=encoding; disposition=map; residual=wire_id=coq|catalog_provider_alias=coq|encoding=rocq; catalog_root=$catalog_root_binding
first_order: canonical_identity=first_order; axis=family; disposition=map; residual=wire_id=first_order|role=translation_source_family|target_encoding=tptp_fof; catalog_root=$catalog_root_binding
isabelle: canonical_identity=isabelle_hol; axis=encoding; disposition=map; residual=wire_id=isabelle|provider=isabelle|encoding=isabelle_hol; catalog_root=$catalog_root_binding
lean: canonical_identity=lean4; axis=encoding; disposition=map; residual=wire_id=lean|provider=lean|encoding=lean4; catalog_root=$catalog_root_binding
lean4: canonical_identity=lean4; axis=encoding; disposition=map; residual=wire_id=lean4|provider=lean|encoding=lean4; catalog_root=$catalog_root_binding
smtlib: canonical_identity=smt_lib2; axis=encoding; disposition=map; residual=wire_id=smtlib|notation=smt_lib2|encoding=smt_lib2; catalog_root=$catalog_root_binding
smtlib2: canonical_identity=smt_lib2; axis=encoding; disposition=map; residual=wire_id=smtlib2|notation_alias=smtlib2|encoding=smt_lib2; catalog_root=$catalog_root_binding
tptp: canonical_identity=tptp_fof; axis=notation; disposition=map; residual=wire_id=tptp|notation=tptp_fof|encoding=tptp_tff; catalog_root=$catalog_root_binding
```

### translation_alias — residual alternate spellings

Aliases are dual-read only. They project onto the primary translation_family
wire residual and then onto the catalog identity; they are not a second list
of supported formats.

```hammer-vocab
domain: translation_alias
surface: hammer._FAMILY_ALIASES
fail_closed: true
catalog_root: $catalog_root_binding
fol: canonical_identity=first_order; axis=family; disposition=residual_alias; residual=wire_id=fol|alias_of=first_order; catalog_root=$catalog_root_binding
first-order: canonical_identity=first_order; axis=family; disposition=residual_alias; residual=wire_id=first-order|alias_of=first_order; catalog_root=$catalog_root_binding
lean_4: canonical_identity=lean4; axis=encoding; disposition=residual_alias; residual=wire_id=lean_4|alias_of=lean4; catalog_root=$catalog_root_binding
smt-lib: canonical_identity=smt_lib2; axis=encoding; disposition=residual_alias; residual=wire_id=smt-lib|alias_of=smtlib; catalog_root=$catalog_root_binding
smt-lib2: canonical_identity=smt_lib2; axis=encoding; disposition=residual_alias; residual=wire_id=smt-lib2|alias_of=smtlib2; catalog_root=$catalog_root_binding
```

### solver_provider — Hammer portfolio solvers (catalog providers)

Solvers are **provider** ids from the catalog / executable matrix. They are
never families, encodings, or proof-authority claims. `e` is a residual wire
alias of matrix provider `eprover` (see solver_alias).

```hammer-vocab
domain: solver_provider
surface: hammer.KNOWN_HAMMER_SOLVERS
fail_closed: true
catalog_root: $catalog_root_binding
cvc5: canonical_identity=cvc5; axis=provider; disposition=map; residual=wire_id=cvc5|lane=smt|matrix_family=smt; catalog_root=$catalog_root_binding
e: canonical_identity=eprover; axis=provider; disposition=residual_alias; residual=wire_id=e|alias_of=eprover|lane=atp|matrix_family=atp; catalog_root=$catalog_root_binding
vampire: canonical_identity=vampire; axis=provider; disposition=map; residual=wire_id=vampire|lane=atp|matrix_family=atp; catalog_root=$catalog_root_binding
z3: canonical_identity=z3; axis=provider; disposition=map; residual=wire_id=z3|lane=smt|matrix_family=smt; catalog_root=$catalog_root_binding
```

### solver_alias — residual solver spellings

```hammer-vocab
domain: solver_alias
surface: hammer._SOLVER_ALIASES
fail_closed: true
catalog_root: $catalog_root_binding
eprover: canonical_identity=eprover; axis=provider; disposition=residual_alias; residual=wire_id=eprover|alias_of=e|canonical_provider=eprover; catalog_root=$catalog_root_binding
```

### target_itp — ITP targets (providers / encodings, not families)

ITP targets select a kernel provider / encoding lane. They never claim
kernel-verified authority by themselves.

```hammer-vocab
domain: target_itp
surface: hammer.target_itp
fail_closed: true
catalog_root: $catalog_root_binding
lean: canonical_identity=lean; axis=provider; disposition=map; residual=wire_id=lean|encoding=lean4|lane=itp_kernel; catalog_root=$catalog_root_binding
lean4: canonical_identity=lean; axis=provider; disposition=residual_alias; residual=wire_id=lean4|alias_of=lean|encoding=lean4; catalog_root=$catalog_root_binding
coq: canonical_identity=rocq; axis=provider; disposition=map; residual=wire_id=coq|provider_alias=coq|encoding=rocq|lane=itp_kernel; catalog_root=$catalog_root_binding
isabelle: canonical_identity=isabelle; axis=provider; disposition=map; residual=wire_id=isabelle|encoding=isabelle_hol|lane=itp_kernel; catalog_root=$catalog_root_binding
```

### family_target — translation family → target encoding residual

Residual portfolio target map. Canonical targets are catalog encodings or
notations; wire targets remain the historical Hammer portfolio tokens.

```hammer-vocab
domain: family_target
surface: hammer._FAMILY_TARGET
fail_closed: true
catalog_root: $catalog_root_binding
first_order: canonical_identity=tptp_fof; axis=notation; disposition=map; residual=wire_id=first_order|wire_target=tptp|catalog_notation=tptp_fof; catalog_root=$catalog_root_binding
tptp: canonical_identity=tptp_fof; axis=notation; disposition=map; residual=wire_id=tptp|wire_target=tptp|catalog_notation=tptp_fof; catalog_root=$catalog_root_binding
smtlib: canonical_identity=smt_lib2; axis=encoding; disposition=map; residual=wire_id=smtlib|wire_target=smtlib|catalog_encoding=smt_lib2; catalog_root=$catalog_root_binding
smtlib2: canonical_identity=smt_lib2; axis=encoding; disposition=map; residual=wire_id=smtlib2|wire_target=smtlib|catalog_encoding=smt_lib2; catalog_root=$catalog_root_binding
```

### authority_ceiling — Hammer hard ceiling (LPC-052 / provider catalog)

Hammer is an advisory portfolio provider. Catalog presence of the `hammer`
provider id never raises authority. Default response authority remains
untrusted (`advisory`) per LPC-052.

```hammer-vocab
domain: authority_ceiling
surface: hammer.evidence_authority
fail_closed: true
catalog_root: $catalog_root_binding
hammer: canonical_identity=advisory; axis=evidence_authority; disposition=ceiling; residual=wire_id=hammer|provider_id=hammer|ceiling=advisory|candidate_authoritative=false; catalog_root=$catalog_root_binding
atp_candidate: canonical_identity=advisory; axis=evidence_authority; disposition=ceiling; residual=wire_id=atp_candidate|evidence_kind=candidate|ceiling=advisory; catalog_root=$catalog_root_binding
smt_candidate: canonical_identity=advisory; axis=evidence_authority; disposition=ceiling; residual=wire_id=smt_candidate|evidence_kind=candidate|ceiling=advisory; catalog_root=$catalog_root_binding
portfolio_success: canonical_identity=advisory; axis=evidence_authority; disposition=ceiling; residual=wire_id=portfolio_success|operation_success_promotes_authority=false; catalog_root=$catalog_root_binding
```

### evidence_kind — portfolio evidence defaults

```hammer-vocab
domain: evidence_kind
surface: hammer.evidence_kind
fail_closed: true
catalog_root: $catalog_root_binding
candidate: canonical_identity=candidate; axis=evidence_kind; disposition=map; residual=wire_id=candidate|authority_ceiling=advisory; catalog_root=$catalog_root_binding
counterexample: canonical_identity=counterexample; axis=evidence_kind; disposition=map; residual=wire_id=counterexample|authority_ceiling=advisory; catalog_root=$catalog_root_binding
kernel_checked_proof: canonical_identity=kernel_checked_proof; axis=evidence_kind; disposition=map; residual=wire_id=kernel_checked_proof|requires_independent_reconstruction=true; catalog_root=$catalog_root_binding
```

### semantic_separation — non-collapse invariants

These rows are **invariants**, not interchangeable identities. Tests assert
each pair remains distinct on the stated axis even when a bridge or residual
collapse exists elsewhere.

```hammer-vocab
domain: semantic_separation
surface: hammer.semantic_separations
fail_closed: true
catalog_root: $catalog_root_binding
flogic_vs_frame_wire: canonical_identity=frame_logic; axis=family; disposition=residual_collapse; residual=left_wire=flogic|right_wire=frame|shared_canonical=frame_logic|wire_distinct=true; catalog_root=$catalog_root_binding
dcec_vs_deontic: canonical_identity=dcec; axis=family; disposition=map; residual=left_wire=dcec|right_wire=deontic|shared_canonical=false|wire_distinct=true; catalog_root=$catalog_root_binding
family_vs_encoding: canonical_identity=first_order; axis=family; disposition=map; residual=left_wire=first_order|right_wire=smt_lib2|left_axis=family|right_axis=encoding|interchangeable=false; catalog_root=$catalog_root_binding
encoding_vs_provider: canonical_identity=lean4; axis=encoding; disposition=map; residual=left_wire=lean4|right_wire=lean|left_axis=encoding|right_axis=provider|interchangeable=false; catalog_root=$catalog_root_binding
provider_vs_lane: canonical_identity=vampire; axis=provider; disposition=map; residual=left_wire=vampire|right_wire=atp|left_axis=provider|right_axis=lane|interchangeable=false; catalog_root=$catalog_root_binding
atp_candidate_vs_smt_sat: canonical_identity=candidate; axis=evidence_kind; disposition=map; residual=left_wire=atp_candidate|right_wire=smt_candidate|left_lane=atp|right_lane=smt|shared_authority=advisory|proof_authority=false; catalog_root=$catalog_root_binding
lean_source_vs_proof_authority: canonical_identity=lean4; axis=encoding; disposition=map; residual=left_wire=lean4|right_wire=kernel_checked_proof|left_axis=encoding|right_axis=evidence_kind|encoding_implies_proof=false; catalog_root=$catalog_root_binding
portfolio_vs_kernel: canonical_identity=advisory; axis=evidence_authority; disposition=ceiling; residual=left_wire=hammer|right_wire=kernel|ceiling=advisory|kernel_requires_reconstruction=true; catalog_root=$catalog_root_binding
```

---

## Derivation rules (no hand-maintained inventories)

| Adapter surface | Must derive from | Must not |
| --- | --- | --- |
| Logic families | `LogicFamily` enum + LPC-090 catalog projection | Free-form family string lists |
| Translation families | Catalog encodings + notations (+ documented residual wire spellings) | Parallel encoding inventory |
| Solvers | Catalog provider ids + executable matrix (+ residual aliases) | Free-form solver brand list |
| Authority / evidence | Provider catalog advisory ceiling + LPC-052 defaults | Hard-coded promotion to authoritative |
| ITP targets | Catalog providers / encodings | Treating ITP name as proof success |

### Residual wire constants

Module-level residual maps (`_FAMILY_ALIASES`, `_FAMILY_ITP`, `_FAMILY_TARGET`,
`_SOLVER_ALIASES`) and exported closed wire sets
(`SUPPORTED_TRANSLATION_FAMILIES`, `KNOWN_HAMMER_SOLVERS`) may retain exact
historical spellings **only when** every element appears in this note and
projects to a catalog identity under the sealed root. They are dual-read
residuals, not independent semantic authorities. Adding a wire token without a
catalog-bound row is a regression.

### Live adapter exports

| Export | Expected derivation |
| --- | --- |
| `IPFS_DATASETS_LOGIC_PROVIDER_ID` | catalog provider id `hammer` |
| `SUPPORTED_LOGIC_FAMILIES` | `tuple(item.value for item in LogicFamily)` (enum-derived) |
| `SUPPORTED_TRANSLATION_FAMILIES` | residual wire set of `translation_family` domain |
| `KNOWN_HAMMER_SOLVERS` | residual wire set of `solver_provider` domain |
| `to_canonical_registry_logic_family` | LPC-090 `map_analysis_family_to_canonical` |
| capability `candidate_authoritative` | always `False` |
| capability / registry `completion_authority` | always `False` |

## Semantic separations preserved

1. **F-logic vs frame.** Wire ids `flogic` and `frame` stay distinct; both may
   project to catalog `frame_logic` only with residual reverse map.
2. **DCEC vs deontic.** Canonical ids `dcec` and `deontic` remain distinct;
   bridges never collapse them.
3. **Family vs encoding vs solver vs kernel.** Axes are non-interchangeable.
4. **Lean / Rocq / Isabelle source encoding ≠ proof authority.** Selecting an
   ITP target does not mint `kernel_checked_proof` authority.
5. **ATP candidates ≠ SMT sat ≠ kernel proof.** Lane and evidence kind stay
   separate; all Hammer portfolio evidence ceilings remain `advisory`.
6. **Operation success ≠ proof.** Portfolio `succeeded` / `candidate` never
   upgrades evidence authority (LPC-052).

## What this task does **not** do

* Does not make Hammer reconstruction-authoritative.
* Does not collapse F-logic into frame wire identity, or DCEC into deontic.
* Does not treat catalog presence as live solver availability.
* Does not replace LPC-090 supervisor maps or LPC-052 response axes.
* Does not expose supervisor-only mutation controls through datasets logic.

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py` | Live Hammer portfolio adapter (residual wire surfaces) |
| `ipfs_accelerate_py/agent_supervisor/proof/canonical_logic_adapter.py` | LPC-090 family / provider projections |
| `test/api/test_ipfs_datasets_logic_provider.py` | Regression: notes ↔ catalog ↔ live adapter |
| `data/agent_supervisor/logic_platform_canonicalization/notes/hammer_adapter.md` | This durable vocabulary artifact |
