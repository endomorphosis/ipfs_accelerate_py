# LPC-090 Supervisor compatibility maps from the catalog

**Task:** LPC-090  
**Goal:** LPC-G090  
**Depends on:** LPC-023 (generated catalog projection), LPC-031 (legacy axis maps)  
**Interface:** `SupervisorCanonicalLogicAdapter@1`  
**Evidence module:** `ipfs_accelerate_py/agent_supervisor/proof/canonical_logic_adapter.py`  
**Validation:** `python -m pytest test/api/test_canonical_logic_adapter.py -q`

## Summary

LPC-090 replaces hand-maintained supervisor family / property / form /
translation / cache / prover maps with a **generated compatibility projection**
bound to the sealed catalog snapshot. The durable inventory is this note.
Executable regression coverage lives in `test/api/test_canonical_logic_adapter.py`
and must stay exhaustive with these tables.

`SupervisorCanonicalLogicAdapter@1` remains the single lazy boundary. Legacy
supervisor enums exist only through explicit adapters. New supervisor records
write **canonical identities**. Distinct supervisor tokens that share a
canonical id (for example `flogic` / `frame` → `frame_logic`) never silently
collapse: reverse mapping always consults residual compatibility data.

## Catalog root binding

Every generated row is sealed to the same catalog content root:

| Field | Authority |
| --- | --- |
| `catalog_root` | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root` (`CanonicalLogicCatalogSnapshot@1`) |
| `catalog_digest` | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest` |
| Adapter interface | `SupervisorCanonicalLogicAdapter@1` |
| Generated providers/translations | `GeneratedProviderTranslationCatalog@1` (LPC-023) |

The machine-readable meta block below binds the projection. Tests resolve the
live CID/digest from the sealed snapshot and require every map row to declare
the same root authority. Layout paths and Git metadata are never authority.

```supervisor-map-meta
task: LPC-090
goal: LPC-G090
interface: SupervisorCanonicalLogicAdapter@1
catalog_snapshot: CanonicalLogicCatalogSnapshot@1
catalog_root: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
catalog_digest: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest
fail_closed: true
hand_maintained_family_lists: false
unknown_policy: reject
```

## Disposition vocabulary

| Disposition | Meaning |
| --- | --- |
| `map` | One supervisor legacy token projects onto one catalog identity; residual still retained for reverse mapping. |
| `map_with_residual` | Multiple supervisor tokens share one canonical identity; residual is mandatory for lossless restore. |
| `supervisor_extension` | Supervisor-only identity under the reserved `supervisor.*` namespace; not a second catalog authority. |
| `provider_alias` | Supervisor route/prover id aliases a datasets provider id (receipts keep the supervisor id). |

## Deprecation vocabulary

| Deprecation | Meaning |
| --- | --- |
| `active` | Public supervisor surface retained; new durable records should already write the canonical identity. |
| `legacy_via_adapter` | Value remains importable only through the adapter boundary; do not extend hand lists. |
| `none` | Not deprecated (same operational meaning as `active` for non-enum tokens). |

## Fail-closed policy

1. **Known legacy only.** A domain mapper accepts only labels listed for that domain.
2. **Unknown → error.** Any other string, empty label, or cross-domain label raises
   `CanonicalLogicAdapterError` (or the note-level `SupervisorMapError` used by tests).
3. **No silent merge.** Shared canonical ids never erase residual supervisor identity.
4. **No second inventory.** Rows are projections of adapter + catalog identities;
   this note must not invent families, properties, or providers absent from those sources.
5. **Presence ≠ executability / proof.** Mapping success never upgrades availability,
   authority, or production admission.

## Residual shape

Residual fields always retain enough to reverse-map losslessly. Every generated
row residual **must** include `supervisor_id` and `domain` (matching the map
key and block domain) so reverse mapping never depends on table position.

| Key | Role |
| --- | --- |
| `supervisor_id` | Exact supervisor token (also the map key); required on every row |
| `domain` | Vocabulary domain; required on every row |
| `supervisor_enum` | Supervisor enum type name when applicable |
| `supervisor_member` | Supervisor enum member name when applicable |
| `taxonomy_translation_kind` | Datasets taxonomy translation kind (translation_class only) |
| `supervisor_prover_id` | Supervisor route prover id (prover_id domain only) |

Machine-readable blocks use fenced `supervisor-map` sections. Tests parse every
block. Label lines are:

```text
legacy_value: canonical=<id>; disposition=<disp>; residual=<k=v|k=v>; deprecation=<dep>; catalog_root=<authority>
```

---

## Mapping tables

### analysis_family → catalog family identity

Source: supervisor `LogicFamily` via `project_analysis_family`.  
Canonical authority: catalog taxonomy / namespaces family ids.

```supervisor-map
domain: analysis_family
supervisor_enum: LogicFamily
catalog_root: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
fail_closed: true
tdfol: canonical=tdfol; disposition=map; residual=supervisor_id=tdfol|domain=analysis_family|supervisor_enum=LogicFamily|supervisor_member=TDFOL; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
dcec: canonical=dcec; disposition=map; residual=supervisor_id=dcec|domain=analysis_family|supervisor_enum=LogicFamily|supervisor_member=DCEC; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
flogic: canonical=frame_logic; disposition=map_with_residual; residual=supervisor_id=flogic|domain=analysis_family|supervisor_enum=LogicFamily|supervisor_member=FLOGIC; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
modal: canonical=modal; disposition=map; residual=supervisor_id=modal|domain=analysis_family|supervisor_enum=LogicFamily|supervisor_member=MODAL; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
deontic: canonical=deontic; disposition=map; residual=supervisor_id=deontic|domain=analysis_family|supervisor_enum=LogicFamily|supervisor_member=DEONTIC; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
frame: canonical=frame_logic; disposition=map_with_residual; residual=supervisor_id=frame|domain=analysis_family|supervisor_enum=LogicFamily|supervisor_member=FRAME; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
kg: canonical=supervisor.kg; disposition=supervisor_extension; residual=supervisor_id=kg|domain=analysis_family|supervisor_enum=LogicFamily|supervisor_member=KNOWLEDGE_GRAPH; deprecation=legacy_via_adapter; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
event_calculus: canonical=event_calculus; disposition=map; residual=supervisor_id=event_calculus|domain=analysis_family|supervisor_enum=LogicFamily|supervisor_member=EVENT_CALCULUS; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
```

Notes:

* `flogic` and `frame` both project to `frame_logic`; residual restores the exact member.
* `kg` uses the reserved `supervisor.kg` namespace so reverse mapping stays exact
  without inventing a datasets family.

### property_kind → catalog property identity

Source: supervisor `PropertyKind` via `project_property_kind`.  
Canonical authority: software-verification property vocabulary / catalog properties.

```supervisor-map
domain: property_kind
supervisor_enum: PropertyKind
catalog_root: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
fail_closed: true
finite_constraint: canonical=satisfiability; disposition=map; residual=supervisor_id=finite_constraint|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=FINITE_CONSTRAINT; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
state_machine: canonical=reachability; disposition=map; residual=supervisor_id=state_machine|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=STATE_MACHINE; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
authorization: canonical=authorization; disposition=map; residual=supervisor_id=authorization|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=AUTHORIZATION; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
protocol: canonical=trace_conformance; disposition=map_with_residual; residual=supervisor_id=protocol|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=PROTOCOL; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
hyperproperty: canonical=hyperproperty; disposition=map; residual=supervisor_id=hyperproperty|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=HYPERPROPERTY; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
runtime_trace: canonical=trace_conformance; disposition=map_with_residual; residual=supervisor_id=runtime_trace|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=RUNTIME_TRACE; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
kernel_check: canonical=theorem; disposition=map_with_residual; residual=supervisor_id=kernel_check|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=KERNEL_CHECK; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
typed_planning: canonical=invariant; disposition=map; residual=supervisor_id=typed_planning|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=TYPED_PLANNING; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
temporal_deontic: canonical=safety; disposition=map; residual=supervisor_id=temporal_deontic|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=TEMPORAL_DEONTIC; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
first_order_theorem: canonical=theorem; disposition=map_with_residual; residual=supervisor_id=first_order_theorem|domain=property_kind|supervisor_enum=PropertyKind|supervisor_member=FIRST_ORDER_THEOREM; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
```

Notes: `protocol` / `runtime_trace` share `trace_conformance`; `kernel_check` /
`first_order_theorem` share `theorem`. Residual prevents silent collapse.

### logic_form → catalog form / family wire identity

Source: supervisor `LogicForm` via `project_logic_form`.

```supervisor-map
domain: logic_form
supervisor_enum: LogicForm
catalog_root: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
fail_closed: true
ast: canonical=ast; disposition=map; residual=supervisor_id=ast|domain=logic_form|supervisor_enum=LogicForm|supervisor_member=AST; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
dcec: canonical=dcec; disposition=map; residual=supervisor_id=dcec|domain=logic_form|supervisor_enum=LogicForm|supervisor_member=DCEC; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
tdfol: canonical=tdfol; disposition=map; residual=supervisor_id=tdfol|domain=logic_form|supervisor_enum=LogicForm|supervisor_member=TDFOL; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
fol: canonical=first_order; disposition=map; residual=supervisor_id=fol|domain=logic_form|supervisor_enum=LogicForm|supervisor_member=FOL; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
tptp: canonical=tptp; disposition=map; residual=supervisor_id=tptp|domain=logic_form|supervisor_enum=LogicForm|supervisor_member=TPTP; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
smt-lib: canonical=smtlib; disposition=map; residual=supervisor_id=smt-lib|domain=logic_form|supervisor_enum=LogicForm|supervisor_member=SMT_LIB; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
tla+: canonical=transition_system; disposition=map; residual=supervisor_id=tla+|domain=logic_form|supervisor_enum=LogicForm|supervisor_member=TLA_PLUS; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
protocol: canonical=cryptographic_protocol; disposition=map; residual=supervisor_id=protocol|domain=logic_form|supervisor_enum=LogicForm|supervisor_member=PROTOCOL; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
hyperproperty: canonical=hyperproperty; disposition=map; residual=supervisor_id=hyperproperty|domain=logic_form|supervisor_enum=LogicForm|supervisor_member=HYPERPROPERTY; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
```

### translation_class → catalog preservation identity

Source: supervisor `TranslationClass` via `project_translation_class`.  
Canonical id is the datasets preservation kind; residual also carries the
taxonomy translation kind.

```supervisor-map
domain: translation_class
supervisor_enum: TranslationClass
catalog_root: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
fail_closed: true
exact: canonical=exact; disposition=map; residual=supervisor_id=exact|domain=translation_class|supervisor_enum=TranslationClass|supervisor_member=EXACT|taxonomy_translation_kind=lossless; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
equisatisfiable: canonical=equisatisfiable; disposition=map; residual=supervisor_id=equisatisfiable|domain=translation_class|supervisor_enum=TranslationClass|supervisor_member=EQUISATISFIABLE|taxonomy_translation_kind=equisatisfiable; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
bounded_abstraction: canonical=bounded; disposition=map; residual=supervisor_id=bounded_abstraction|domain=translation_class|supervisor_enum=TranslationClass|supervisor_member=BOUNDED_ABSTRACTION|taxonomy_translation_kind=sound_over_approximation; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
conservative_approximation: canonical=conservative; disposition=map; residual=supervisor_id=conservative_approximation|domain=translation_class|supervisor_enum=TranslationClass|supervisor_member=CONSERVATIVE_APPROXIMATION|taxonomy_translation_kind=sound_over_approximation; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
heuristic: canonical=heuristic; disposition=map; residual=supervisor_id=heuristic|domain=translation_class|supervisor_enum=TranslationClass|supervisor_member=HEURISTIC|taxonomy_translation_kind=heuristic; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
```

### cache_scope → catalog cache-scope identity

Source: supervisor `CacheScope` via `project_cache_scope`.

```supervisor-map
domain: cache_scope
supervisor_enum: CacheScope
catalog_root: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
fail_closed: true
exact_tree: canonical=tree; disposition=map; residual=supervisor_id=exact_tree|domain=cache_scope|supervisor_enum=CacheScope|supervisor_member=TREE; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
objective_revision: canonical=policy; disposition=map; residual=supervisor_id=objective_revision|domain=cache_scope|supervisor_enum=CacheScope|supervisor_member=OBJECTIVE; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
request: canonical=request; disposition=map; residual=supervisor_id=request|domain=cache_scope|supervisor_enum=CacheScope|supervisor_member=REQUEST; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
none: canonical=none; disposition=map; residual=supervisor_id=none|domain=cache_scope|supervisor_enum=CacheScope|supervisor_member=NONE; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
```

### prover_id → catalog provider identity

Source: supervisor route / matrix prover ids via
`map_prover_id_to_canonical_provider`. Unlisted prover ids pass through as
identity; only explicit aliases are generated here. Unknown empty tokens fail
closed at the adapter boundary.

```supervisor-map
domain: prover_id
supervisor_enum: none
catalog_root: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
fail_closed: true
coq: canonical=rocq; disposition=provider_alias; residual=supervisor_id=coq|domain=prover_id|supervisor_enum=none|supervisor_member=none|supervisor_prover_id=coq; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
e: canonical=eprover; disposition=provider_alias; residual=supervisor_id=e|domain=prover_id|supervisor_enum=none|supervisor_member=none|supervisor_prover_id=e; deprecation=active; catalog_root=DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
```

---

## Acceptance matrix

| Check | Fail-closed behavior | Primary APIs / artifacts |
| --- | --- | --- |
| Generated coverage | Every inventoried supervisor vocabulary token in the adapter inventory appears in this note | `vocabulary_inventory`, this note |
| Adapter parity | Note canonical ids match live `VocabularyProjection.canonical_id` | `project_*` methods |
| Residual fidelity | Note residual always includes `supervisor_id` + `domain` and is ⊆ live projection residual; restore is lossless | `residual`, `restore_*` |
| Catalog root | Every row binds to sealed snapshot content root | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root` |
| Unknown labels | Unknown domain labels raise | `CanonicalLogicAdapterError`, note lookup |
| No hand family lists | Note domains are projections, not free-form inventories | LPC-023 / LPC-090 conflict policy |
| Shared canonical ids | Distinct supervisor ids keep distinct residuals | `flogic`/`frame`, `protocol`/`runtime_trace` |

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-023 generated catalogs | Supervisor maps project onto generated / snapshot identities; they do not re-list providers by hand |
| LPC-031 legacy axis maps | Status/verdict/evidence axes; LPC-090 owns family/property/form/translation/cache/prover maps |
| LPC-091 type classification | Classifies leftover supervisor semantic types after these maps land |
| LPC-100 package manifest | Catalog root handshake uses the same sealed snapshot content root |
| LPC-110 supervisor client | Consumes adapter projections + manifest handshake |

## What this task does **not** do

* Does not remove public supervisor enum types (LPC-091 owns classification / migration paths).
* Does not hand-edit family lists inside the adapter beyond the generated projection contract.
* Does not claim live prover availability or production admission from a successful map.
* Does not flatten catalog layers or introduce registry v4.

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_map_cutover.md` | This generated compatibility inventory |
| `test/api/test_canonical_logic_adapter.py` | Parse note, adapter parity, fail-closed regressions |
| `ipfs_accelerate_py/agent_supervisor/proof/canonical_logic_adapter.py` | Lazy runtime projection boundary (read by tests; not rewritten by this task's edit budget) |
