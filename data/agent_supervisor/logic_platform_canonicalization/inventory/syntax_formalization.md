# LPC-004 Inventory: syntax, formalization, and domain-slice generations

Machine-readable companion:
`data/agent_supervisor/logic_platform_canonicalization/inventory/syntax_formalization.json`

Interface: `LogicPlatformInventory@1`  
Task: `LPC-004` · Goal: `LPC-G010` · Track: inventory

## Source revisions

| Repository | Reviewed baseline | Implementation authority |
| --- | --- | --- |
| `ipfs_datasets_py` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` (equals baseline) |
| `ipfs_accelerate_py` | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` | `ea11293bb996f052d620eae989f5377a956764b1` |

Rule: current heads are implementation authority; reviewed baselines are comparison anchors only.

### Source availability in this worktree

| Location | Available | Note |
| --- | --- | --- |
| `/home/barberb/lift_coding/external/ipfs_datasets` | no | Plan-named datasets checkout missing |
| `ipfs_datasets_py/` submodule | no | Empty directory; LPC scheduler `worktree_submodule_paths` is `[]` |

Classification method: plan contracts, LPC predicted paths, accelerate import sites, and LFV/FVT evidence paths. Entries with `source_read=false` still list the generation and a classification, but concrete class/schema fields must be re-verified once the datasets tree is mounted.

## Classification vocabulary

`canonical`, `canonical_component`, `compatibility_facade`, `legacy`, `experimental`, `declaration_only`, `generated`, `duplicate`, `obsolete`, `unresolved`

## Canonical typed new-write path

```text
SourceDocument
  → syntax_core parse artifact
  → TypedExpression
  → ElaborationArtifact
  → FormalizationArtifact@3
  → DomainLogicSlice@2
  → LogicObligation@2
  → BackendRequest@2
```

Only **FormalizationArtifact@3** and **DomainLogicSlice@2** are admitted for new domain formalization writes (LPC-G040 / LPC-040).

## Generation summary

### FormalizationArtifact generations

| Generation | Classification | Path / surface | Role |
| --- | --- | --- | --- |
| **FormalizationArtifact@3** | **canonical** | `ipfs_datasets_py/logic/formalization/artifacts_v3.py` | Only new-write formalization generation |
| FormalizationArtifact (compiler) | legacy | `ipfs_datasets_py.logic.formalization.compiler.FormalizationArtifact` | Live import used by admissibility tests/bridge |
| FormalizationArtifact@2 | unresolved | (none observed) | Implied by `artifacts_v3` naming |
| FormalizationArtifact@1 | unresolved | (none observed) | Implied by `artifacts_v3` naming |

### DomainLogicSlice generations

| Generation | Classification | Path / surface | Role |
| --- | --- | --- | --- |
| **DomainLogicSlice@2** | **canonical** | domain adapters under `*_ir` / `software_verification` | Only admitted domain lowering generation |
| DomainLogicSlice@1 | unresolved | (none observed) | Implied by `@2` naming |

### AST / typed-expression stages

These stages are not labeled `@N` in plan text; they are the typed front of the new-write path.

| Stage | Classification | Package |
| --- | --- | --- |
| SourceDocument (logic) | canonical_component | `syntax_core` |
| syntax_core package | canonical | `ipfs_datasets_py.logic.syntax_core` |
| syntax_core parse artifact | canonical_component | `syntax_core` |
| TypedExpression | canonical | `syntax_core` |
| ElaborationArtifact | canonical_component | `syntax_core` (placement unconfirmed without source) |
| supervisor `program_ast_adapters.SourceDocument` | **duplicate** (name collision) | `ipfs_accelerate_py.agent_supervisor.analysis` |

## Item inventory

### AST / typed expression (`ast_or_typed_expression`)

| ID | Name | Classification | Notes |
| --- | --- | --- | --- |
| `ast:source-document` | SourceDocument | canonical_component | Logic pipeline entry; not supervisor program AST |
| `ast:syntax-core-package` | syntax_core | canonical | LPC-004 read-only production package |
| `ast:syntax-core-parse-artifact` | syntax_core parse artifact | canonical_component | Parse output before typing |
| `ast:typed-expression` | TypedExpression | canonical | Required expression identity on new writes |
| `ast:elaboration-artifact` | ElaborationArtifact | canonical_component | Between TypedExpression and FormalizationArtifact@3 |
| `ast:supervisor-program-source-document` | program_ast_adapters.SourceDocument | duplicate | Analysis AST only; not formalization |

### FormalizationArtifact (`formalization_artifact`)

| ID | Name | Generation | Classification | Notes |
| --- | --- | --- | --- | --- |
| `fa:formalization-artifact-v3` | FormalizationArtifact@3 | @3 | canonical | New-write authority |
| `fa:formalization-artifacts-v3-module` | artifacts_v3 module | @3 | canonical | Module owning @3 types |
| `fa:formalization-artifact-compiler` | compiler.FormalizationArtifact | @compiler | legacy | Live accelerate import path |
| `fa:formalization-artifact-v1` | FormalizationArtifact@1 | @1 | unresolved | No path observed |
| `fa:formalization-artifact-v2` | FormalizationArtifact@2 | @2 | unresolved | No path observed |
| `fa:formalization-admission` | formalization.admission | @3 gate | declaration_only | LPC-040 predicted helper |
| `fa:proposal-advisors` | proposal_advisors | n/a | experimental | Candidate/advisor only |
| `fa:autoencoder-advisor` | autoencoder_advisor | n/a | experimental | Candidate/advisor only |
| `fa:artifact-envelope-from-intent` | ArtifactEnvelope | n/a | compatibility_facade | Wraps formalization for corpus |

### DomainLogicSlice (`domain_logic_slice`)

| ID | Name | Generation | Classification | Notes |
| --- | --- | --- | --- | --- |
| `dls:domain-logic-slice-v2` | DomainLogicSlice@2 | @2 | canonical | Canonical lowering contract |
| `dls:domain-logic-slice-v1` | DomainLogicSlice@1 | @1 | unresolved | No path observed |
| `dls:legal-domain-slice` | legal_ir.domain_slice | @2 | canonical_component | Keep TDFOL/DCEC/frame distinct (LPC-041) |
| `dls:security-domain-slice` | security_ir.domain_slice | @2 | canonical_component | LPC-042 |
| `dls:software-verification-domain-slice` | software_verification.domain_slice | @2 | canonical_component | LPC-042; keep SV families distinct |
| `dls:crypto-domain-slice` | crypto_ir.domain_slice | @2 | canonical_component | LPC-042 |
| `dls:intent-domain-slice` | intent_ir.domain_slice | @2 | canonical_component | LPC-043 |
| `dls:ui-ux-domain-slice` | ui_ux_ir.domain_slice | @2 | canonical_component | LPC-043 |
| `dls:unadmitted-slice-gate` | unadmitted-slice rejection | @2 gate | declaration_only | LPC-044 predicted test/gate |

## Domain adapters and non-collapse rules

| Domain adapter | Must keep distinct | Forbidden silent mapping |
| --- | --- | --- |
| legal | TDFOL, DCEC, frame logic | FOL / generic deontic / object framing |
| security | security families | software-verification collapse |
| software_verification | contracts, STS, authorization, concurrency, separation, hyperproperties, protocols, monitors | single generic program IR |
| crypto | crypto ontology | security/software collapse |
| intent | intent ontology | universal domain IR |
| ui_ux | UI/UX ontology | universal domain IR |

Executable requests without an admitted **DomainLogicSlice@2** are rejected (LPC-044).

## Related surfaces explicitly out of generation scope

These appear near formalization/domain code but are **not** FormalizationArtifact or DomainLogicSlice generations:

- `software_verification.tactician.*` proof-plan / goal surfaces
- `software_verification.counterexamples.*`
- `security_ir.cvefixes.*` CVEFix evaluation pipelines
- `intent_ir.graphrag.retrieval`
- `formalization.proposal_advisors` / `autoencoder_advisor` (advisors only)
- Supervisor program AST adapters

## Counts

| Metric | Value |
| --- | --- |
| Total items | 24 |
| ast_or_typed_expression | 6 |
| formalization_artifact | 9 |
| domain_logic_slice | 9 |
| canonical | 5 |
| canonical_component | 9 |
| compatibility_facade | 1 |
| legacy | 1 |
| experimental | 2 |
| declaration_only | 2 |
| duplicate | 1 |
| unresolved | 3 |
| source_read true | 1 |
| source_read false | 23 |

## Follow-up

1. Mount datasets at `ac82107e246b30e35a2bbdcf75e01370d22350c6` and re-walk `syntax_core`, `formalization`, and each `domain_slice` module.
2. Confirm whether `formalization.compiler.FormalizationArtifact` aliases `artifacts_v3`, is a facade, or is a distinct pre-@3 generation.
3. Confirm whether `@1` / pre-@3 historical generations exist as modules, aliases, or only as wire labels.
4. After LPC-040..044 land, flip `declaration_only` admission/gate entries to their post-implementation classification.
