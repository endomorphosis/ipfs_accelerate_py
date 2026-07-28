# Semantic-Roundtrip Residual / Structural Bridge (CBP-110)

**Interface:** `CodeProofSrtBridge@1`
**Module:** `ipfs_accelerate_py.agent_supervisor.code_proof_srt_bridge`
**Related:** `CodeEditPacket@1`, `StructuralAdmission@1`, `PlateauCodexPacket@1`,
`PlatResidualCatalog@1`, `Plat2HoldoutRegistry@1`

## Purpose

Bind semantic-roundtrip (SRT) structural constraints and PLAT residual facets
into the shared CBP property catalog, claim, query, context, and edit-packet
path so PLAT / PLAT2 loops reuse CBP infrastructure **without**:

* treating heterogeneous SRT methods as interchangeable
* importing gold IR bodies into proof-cache keys or packet metadata
* relocating experiment selection or semantic promotion authority into CBP
* rewriting sealed PLAT promotion snapshots

This bridge is an **adapter**, not a second promotion authority.

## Heterogeneous method roles (measured, not interchangeable)

PLAT pilots own method-comparison evidence. CBP-110 records the measured role
of each method and refuses to treat them as substitutes.

| Method | Measured role | What it may do | What it must not do |
| --- | --- | --- | --- |
| **autoencoder**, **spaCy** | `bounded_guidance_diagnostics` | Bounded guidance, diagnostics, residual cues | Structural admission, semantic promotion, edit authority |
| **SyMAI** | `orchestration` | Orchestrate residual → gate → packet flow | Admit semantics, replace e2e loss, mint kernel proof |
| **Leanstral** | `proposal_teacher` | Untrusted proposal / repair teaching | Kernel authority, completion authority, structural gate |
| **Hammer**, **cvc5**, **Lean** | `structural_gate` | Declared structural admission over residual tags | Replace SRT e2e loss or PLAT2 holdout promotion |
| **compiler / IR / decompiler** | `edit_target` | Deterministic edit surface for `CodeEditPacket@1` | Semantic scoring or holdout promotion |

API: `resolve_method_role`, `method_role_description`, `method_roles_manifest`.

Unknown method ids **fail closed**. Roles are not interchangeable: a guidance
method cannot be re-labeled as a structural gate, and a structural gate cannot
be treated as the semantic promotion authority.

## Authority doctrine

```text
PLAT residual catalog / PlateauCodexPacket@1
        │  (handles only — no gold IR bodies)
        ▼
CodeProofSrtBridge@1
        ├── typed CodeClaimRecord@1  (family: srt_structural)
        ├── FormalCounterexample     (handles + tags only)
        ├── ContextCapsule           (obligation-first, non-semantic authority)
        ├── CodeEditPacket@1         (edit target = compiler/IR/decompiler)
        └── StructuralAdmission@1 → graph/query  (semantic_authority=false)
                │
                ▼
        TrustAwareProofCache keys  (residual/obligation/tag/tree handles only)

Promotion authority remains OUTSIDE the bridge:
  • semantic_roundtrip_e2e_loss
  • plat2_holdout_promotion_gate
```

Normative constants:

* `STRUCTURAL_SEMANTIC_AUTHORITY = False`
* `PROMOTION_AUTHORITIES = (semantic_roundtrip_e2e_loss, plat2_holdout_promotion_gate)`
* `gold_ir_excluded = True` on every bridge projection and cache-key handle set

Structural receipts, GraphRAG / query projections, and prover bindings projected
by this bridge **never** set `semantic_authority=true` and never mint semantic
promotion.

## Interfaces

### `PlatResidualCatalog@1` / `ResidualCatalogEntry`

Content-addressed catalog of residual **handles** and facets:

* `residual_ref_id` (content-addressed handle; never a gold IR body)
* optional `structural_tags` (closed set shared with catalog seeds:
  `non_vacuous_candidate`, `rule_cardinality_preserved`,
  `untriggered_projection_preserved`)
* optional claim / obligation / property / counterexample ref ids
* `predicted_files` for the deterministic edit target
* status + summary (no proof bodies)

Builder: `build_plat_residual_catalog`.
Aggregate project: `project_plat_residual_catalog`.

### `PlateauCodexPacket@1`

Plateau packet ids project through `project_plateau_codex_packet` /
`project_plateau_packet_bundle` into `CodeEditPacket@1` via the CBP-080
handles-only materializer (`bridge_plateau_codex_packet`). Only residual /
packet / claim / obligation handles are accepted; gold IR keys fail closed.

### `StructuralAdmission@1`

Declared structural gate receipt over residual refs and tags.

* Gate methods must resolve to `structural_gate` (Hammer / cvc5 / Lean).
* `semantic_authority` is **always** forced `false`.
* Disposition vocabulary: admitted | rejected | timeout | unsupported |
  not_measured | error.
* Graph projection: `project_structural_admission_to_graph` emits nodes, edges,
  and query facts with `non_authoritative=true` and explicit
  `promotion_authorities`.
* Claim projection: `structural_admission_to_claims` yields
  `ClaimFamily.SRT_STRUCTURAL` records with unverified derived assurance.

### `Plat2HoldoutRegistry@1`

PLAT2 holdout artifacts are **separately** preregistered and queryable:

* `register` requires `preregistered=true`
* `query` filters by artifact id, holdout split, residual ref, or property id
* Never mixed into the PLAT residual catalog or training residual set
* Holdout promotion remains the PLAT2 gate (`plat2_holdout_promotion_gate`)

## Projection map

| Source | Target | Function |
| --- | --- | --- |
| Residual entry | `CodeClaimRecord@1` | `project_residual_to_claim` |
| Residual entry | `FormalCounterexample` | `project_residual_to_counterexample` |
| Residual entry | `ContextCapsule` | `project_residual_to_context_capsule` |
| Residual entry | `CodeEditPacket@1` | `project_residual_to_code_edit_packet` |
| Plateau packet | `CodeEditPacket@1` | `project_plateau_codex_packet` |
| Residual catalog (+ optional admission) | `SrtBridgeProjection` | `project_plat_residual_catalog` |
| Plateau + residuals + admission | `SrtBridgeProjection` | `project_plateau_packet_bundle` |
| StructuralAdmission | graph / query facts | `project_structural_admission_to_graph` |
| Residual / tag / obligation handles | `ProofCacheKey` | `SrtCacheKeyHandles.build_proof_cache_key` |

Every projection records:

* `semantic_authority: false`
* `gold_ir_excluded: true`
* `promotion_authorities: [semantic_roundtrip_e2e_loss, plat2_holdout_promotion_gate]`

## Cache keys exclude gold IR bodies

`SrtCacheKeyHandles` / `build_srt_cache_key_handles` accept only:

* residual ref handles
* obligation / property ids
* structural tags
* repository tree id
* gate method ids
* catalog version

`reject_gold_ir_bodies` fails closed on nested gold / proof / dump keys
(`gold_ir`, `gold_ir_body`, `proof_body`, `lean_source`, `repository_dump`, …).
Flags such as `gold_ir_excluded` are allowed.

Premises passed to `build_proof_cache_key` are the same handle set — never gold
IR text. This preserves the single trust boundary of
`formal_verification_cache.TrustAwareProofCache`.

## Context capsules

Residual capsules bind:

* **goal** — residual repair identity + property / obligation handles
* **authority** — non-semantic; lists promotion authorities outside the bridge
* **scope** — residual ref, structural tags, predicted files, plateau packet id
* **acceptance** — requires e2e loss and holdout gate; structural admission alone
  is insufficient
* **evidence** — residual (and counterexample) **handles** only

No gold IR, full IR dumps, or sealed promotion report rewrites.

## Conflict policy

* Bridge only: do **not** rewrite sealed PLAT promotion snapshots.
* Do **not** change production SRT arm defaults.
* Do **not** move experiment selection into CBP.
* Unit tests use **fixtures** only — never live gold dumps or sealed-report
  rewrites.

## Tests

`test/api/test_agent_supervisor_code_proof_srt_bridge.py` covers:

* Method-role table and fail-closed unknown methods
* Residual → claim / counterexample / capsule / CodeEditPacket
* PlateauCodexPacket → CodeEditPacket (handles only)
* StructuralAdmission graph/query with `semantic_authority=false`
* PLAT2 holdout separate preregistration and query
* Cache keys exclude gold IR bodies
* Gold-body rejection on residual, plateau, and admission payloads
* Promotion authorities remain e2e loss + holdout gate
* Fixture-only fixtures (no live gold dumps)

Validation:

```bash
python -m pytest test/api/test_agent_supervisor_code_proof_srt_bridge.py -q
```
