# ProofGroundedIRLearningFabric final evidence report

This report is `RESULT(PGIR-111)`. Every metric cites a content-addressed
receipt or an explicit `not_run` / `no_go` status. Missing evidence is never
inferred as a pass. The exact qualified-claim text is withheld because the
admitted gates did not pass.

- Decision: `no_go`
- Decision CID: `baguqeeraejs56hwzs3bqtgzoayrc2fxwgfnhcsxjthi4dh7gh64wptlkfhwa`
- Freeze root: `baguqeerarkgpz4xl663tlpfpiajjtxlya3b576lqzg5yd7nrthqgs2rm6v2q`
- Freeze result: `baguqeerai2ipwhyywztjob62ju5pokmm4o6unqqee3poyrabj37aby6fuoca`
- Promotion receipt: `baguqeerazlsqaghk6m2c2ru3qbrsfprv7b5nbc5oa5lvfxz6d4b6qqgxpvra`
- Publication receipt: `baguqeeramcfkbj3ri7efgljybtde5dwhxfpelp25qdjfxt2aiwdt56sdg2zq`
- Qualified claim emitted: `false`

## Final acceptance criteria

### F01 All PGIR-G010 through PGIR-G110 children have fresh current-input evidence.

- Status: `resolved_with_evidence`
- Evidence: Inventory, freeze, contract, campaign, and security children have sealed current-input receipts. RESULT(PGIR-110) is a documented execution no-go, not a missing child.

### F02 No source-lineage leakage across related derivatives.

- Status: `satisfied`
- Evidence: IRSplitManifest@1 leakage audit passed with zero violations. Related derivatives remain in one leakage group.

### F03 One canonical typed bridge is bound and used.

- Status: `satisfied_with_limitations`
- Evidence: COMPILER-CURRENT-1 and DECOMPILER-CURRENT-1 are bound. The gap matrix records remaining unsupported directions; no second canonical bridge was admitted.

### F04 A current-input deterministic baseline is measured and qualified.

- Status: `no_go`
- Evidence: Historical RESULT(PGIR-023) fixture metrics exist by CID only and remain not currently qualified for campaign heldouts.

### F05 Proof-aware pair, loss, and evaluation contracts are sealed.

- Status: `satisfied`
- Evidence: Closed pair, loss, and evaluation contracts are sealed. Model output cannot silently become canonical or proof-grounded.

### F06 A resumable resource-aware campaign exists and may execute only from sealed inputs.

- Status: `satisfied_execution_denied`
- Evidence: IRLearningCampaign@1 and R1-R6 leases exist. Descendant execution remains unauthorized under RESULT(PGIR-014).

### F07 Promotion is deterministic policy admission or a documented no-go.

- Status: `satisfied`
- Evidence: Independent promotion comparison produced a documented no-go. No candidate was compared and no pointer mutated.

### F08 Append-only qualified publication occurs only when independently authorized.

- Status: `satisfied`
- Evidence: Remote publication is denied. The PGIR-090 local package is not a qualified upload.

### F09 Actual token metrics are reported with paired uncertainty.

- Status: `no_go`
- Evidence: Token metrics were reported as not_run with zero denominators and unavailable paired uncertainty.

### F10 Actual latent and retrieval metrics are reported with paired uncertainty.

- Status: `no_go`
- Evidence: Latent and retrieval metrics were reported as not_run with zero denominators and unavailable paired uncertainty.

### F11 Actual structural metrics are reported with paired uncertainty.

- Status: `no_go`
- Evidence: Structural metrics were reported as not_run with zero denominators and unavailable paired uncertainty.

### F12 Actual semantic metrics are reported with paired uncertainty.

- Status: `no_go`
- Evidence: Current-input semantic campaign metrics were reported as not_run. Historical R1 scores stay unqualified.

### F13 Actual proof metrics are reported with paired uncertainty.

- Status: `no_go`
- Evidence: Proof metrics were reported as not_run. No independently checked campaign proof entered curriculum.

### F14 Actual calibration and OOD metrics are reported with paired uncertainty.

- Status: `no_go`
- Evidence: Calibration and OOD metrics were reported as not_run with unavailable paired uncertainty.

### F15 Actual latency and resource results are reported with paired uncertainty.

- Status: `no_go`
- Evidence: Latency and resource campaign measures were reported as not_run. No GPU or prover lease was granted.

### F16 Hidden tests are never used for tuning; only an admitted candidate may be published; the next content-addressed board is produced.

- Status: `satisfied`
- Evidence: Hidden tests stayed sealed, no candidate was published, and the next content-addressed board is issued.

## 1. Exact source revisions

Campaign authority remains SRCSET-1. Reviewed revisions are comparison inputs only.

- Section status: `resolved`
- `accelerate_authority_commit`: `"8d46a6d25dd006c8cab3c9d9612707d2a014e79c"`
- `accelerate_authority_tree`: `"697ee660025fbf14a1cbe6c24fd8da5365df84d5"`
- `accelerate_checkout`: `"0cc04ebb640c4c981cf4650016e096a73ab0e8c0"`
- `accelerate_live_delta_sha256`: `"0d13706bbdd5f50118999dc928172c8f0df29aea8f86613b0f5664e60435c87c"`
- `accelerate_range_log_sha256`: `"0a70de8c18be990e59660a0a4cbaf00cf81cf31b3321ad9b03bab0a666eaf61e"`
- `accelerate_reviewed`: `"c821d0b43877591bbb0fa3f328fbccff187b56e7"`
- `accelerate_reviewed_commits_ahead`: `3616`
- `datasets_authority_commit`: `"df93e91e6338c84a17c3208ef68b88de8566f78c"`
- `datasets_authority_tree`: `"37b9cb40644831c85c6fdf07d0228e45061e239a"`
- `datasets_checkout`: `"d144be65ffe4c6423e4e1c30cd692812607343eb"`
- `datasets_live_delta_sha256`: `"2f93a232612d1b8d1da6b52abfa1639621a86ac82eef2180f163eaa9d6b547f4"`
- `datasets_range_log_sha256`: `"aaeff6d8976787159e8ec747fc60a5d27b6515773068c06e968cfb3a107dd21e"`
- `datasets_reviewed`: `"7f0fe2bbad3c70928234c6e2312ee3182fd7681f"`
- `datasets_reviewed_commits_ahead`: `1717`
- `selected_datasets_commit`: `"b20bd9e3cfae79e8888929daf64f52b2f8a5689a"`
- `source_manifest_cid`: `"baguqeerasownoxqyrppw3ft3us3yvd26ghvqnjl74nr2rw5o7sm3sjehip7a"`
- `source_result_identity`: `"sha256:86f7c07cf7b62847b81315b6529942da7647acf9922f32cf5ccef9d8bb221e9c"`
- `source_set_id`: `"SRCSET-1"`
- `source_tree_id`: `"04fbb09b4a8b34e77d11bd8da6642e0978baa02c"`

## 2. Exact JusticeDAO repository revisions and configurations

JDAO-PINSET-1 pins 21 Hub revisions and admits zero repositories for proof-grounded training.

- Section status: `resolved`
- `pinset_id`: `"JDAO-PINSET-1"`
- `pinset_sha256`: `"8e3a4b1bd81639393ddda35e5dfb3b95f9e7320afa898bde0b3eb9a0317a6b76"`
- `public_dataset_repository_count`: `21`
- `source_releases`: `[{"disposition": "quarantined", "repository_id": "justicedao/patent-legal-ir-graphrag", "revision": "845669408081f1334c54519d2bb7df6bf780ccd5", "source_records": 2174}, {"disposition": "quarantined", "repository_id": "justicedao/wetwijzer_netherlands_legal_corpus", "revision": "827e9412f55cbe332f18824ff669bdbbae39005d", "source_records": 4999}]`
- `training_repositories_admitted`: `0`

## 3. Current-state inventory

Datasets, supervisor, release, baseline, and gap inventories are sealed. Baseline tests remain mixed and non-qualifying.

- Section status: `resolved`
- `accelerator_baseline`: `"360 passed"`
- `baseline_summary_cid`: `"baguqeerasr4hxpxwkhe64btsxu56moh7v7ww2a6covaavjd7qnzb4xgl5cpq"`
- `datasets_baseline`: `"801 passed, 2 failed, 2 skipped, 13 errors"`
- `gap_matrix_cid`: `"baguqeeraspldjlypaoamdclucjsbktrmkramzhe3kra7tip5h3e5s5zkfnia"`
- `modules_identity`: `"sha256:532e1c8b60fcf77515e772368f80475d2f62ba1545490826c3baeea1b402920b"`
- `release_inventory_identity`: `"sha256:f58457d29289d5140bfc63596f28a4648e2bad1ea6222e7b62d22e8cf9b95bb6"`
- `supervisor_inventory_cid`: `"baguqeerablvf72zunpjvbievbspxqnc4eqgxneqjwrg5v6imr7edavovmwca"`

## 4. Source versus derived record counts

Source and derived counts stay distinct. Derivatives do not inflate source rows.

- Section status: `resolved`
- `derived_count`: `38690`
- `patent_source_groups`: `2174`
- `source_count`: `7173`
- `training_admitted_rows`: `0`

## 5. Lineage and split design

Lineage-safe multidimensional splits exist. Thirteen required holdouts remain insufficient.

- Section status: `resolved_with_no_go`
- `hidden_test_commitment`: `"sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded"`
- `insufficient_holdouts`: `["compiler", "cross_reference", "domain", "exception", "length", "lineage", "notation", "premise", "proof_library", "publication", "rare_operator", "time", "type"]`
- `populated_holdouts`: `[{"count": 4999, "name": "family", "split": "statute_family"}, {"count": 74, "name": "jurisdiction", "split": "jurisdiction"}]`
- `split_manifest_digest`: `"sha256:047b263b85067aa3dad6760f623c2855fbaf776d565ec9c273c49425fcc14eb4"`
- `split_manifest_sha256`: `"sha256:9e552a46d1f850fd0455d2c5b1d87810077fd35eb88ea849e64de24090bc167f"`
- `split_root_sha256`: `"sha256:b522f15f2597ed4902f1af9b7f3aac5b855193d289369df70ccfda5ce8798f9d"`

## 6. Leakage-audit results

The leakage audit passed with an empty violation set.

- Section status: `satisfied`
- `leakage_passed`: `true`
- `violations`: `0`

## 7. Canonical bridge-IR design

One canonical bridge is bound. Unsupported constructs remain explicit in the gap matrix.

- Section status: `satisfied_with_limitations`
- `compiler_identity`: `"RESULT(PGIR-021)"`
- `decompiler_identity`: `"RESULT(PGIR-022)"`
- `gap_matrix_cid`: `"baguqeeraspldjlypaoamdclucjsbktrmkramzhe3kra7tip5h3e5s5zkfnia"`

## 8. Compiler architecture

TypedDeonticCanonicalCompiler is COMPILER-CURRENT-1. No learned compiler stage is admitted.

- Section status: `resolved`
- `entrypoint`: `"ipfs_datasets_py.logic.legal_ir.canonical_compiler.TypedDeonticCanonicalCompiler"`
- `interface`: `"CanonicalStructuredTextCompiler@1"`
- `learned_stages`: `[]`
- `symbolic_alias`: `"COMPILER-CURRENT-1"`

## 9. Decompiler architecture

SourceWithheldCanonicalDecompiler is DECOMPILER-CURRENT-1. It does not use a model.

- Section status: `resolved`
- `entrypoint`: `"ipfs_datasets_py.logic.legal_ir.canonical_decompiler.SourceWithheldCanonicalDecompiler"`
- `interface`: `"SourceWithheldCanonicalParaphraser@1"`
- `symbolic_alias`: `"DECOMPILER-CURRENT-1"`
- `uses_model`: `false`

## 10. Deterministic baseline

Historical R1 fixture metrics are referenced by CID and are not currently qualified.

- Section status: `no_go`
- `hidden_test_selection`: `false`
- `historical_manifest_cid`: `"baguqeeraf3mevd4zrpkcy6hmsamfyszkq5zeisq2ipu6bvupquprtfqi53ta"`
- `historical_recipe_cid`: `"baguqeerazuhonzzynznbhtlfgmsbmlrl4fzs73ogo3ogek4e5xbb577unuea"`
- `historical_report_cid`: `"baguqeerau73uowpiy22d7rohi7gvbtfkwivyldlivxnxqw4zzc7zeyzavasq"`
- `measured_populations`: `["pilot", "repair_development"]`
- `qualification`: `"not_currently_qualified"`

## 11. Learned-model architecture

Shared-latent and shared-encoder/typed-head arms were declared. No weights were written.

- Section status: `not_run`
- `architectures_instantiated`: `[]`
- `architectures_intended`: `["shared_latent", "shared_encoder_typed_head"]`
- `candidate_checkpoint`: `null`

## 12. Tokenizer and vocabulary

No learned tokenizer or vocabulary is admitted. Unknown tokens fail closed.

- Section status: `no_go`
- `learned_vocabulary_identity`: `"none"`
- `policy_cid`: `"baguqeerahoedy5eyabjpcixpxwcjlbh54femnpb3krvirv3scto4nqfkplua"`
- `status`: `"no_learned_tokenizer_admitted"`

## 13. Loss configuration

IRLossConfiguration@1 is the fixed-point identity. No training step consumed it.

- Section status: `resolved_unused`
- `identity`: `"IRLossConfiguration@1"`
- `proof_in_gradient_path`: `false`

## 14. Training curriculum

R1-R6 arms and seeds were bound. Every arm/seed lease remained ungranted.

- Section status: `not_run`
- `arms`: `["R1", "R2", "R3", "R4", "R5", "R6"]`
- `learned_seeds`: `[32, 33, 34]`
- `pgir_110_revised_task_cid`: `"baguqeeranjhprx27smrkacgi7t5wuwu5kogvlishd77wwddi3dgzsxcijeba"`
- `pgir_110_task_cid`: `"baguqeeragjtn4knjvdexk4ya373ljydixx5r6c4moxr3sj6apuf2slfutexq"`

## 15. Hard-negative generation

Hard-negative recipes exist. Timeout, unknown, and model-only labels cannot become negatives.

- Section status: `resolved_unused`
- `negative_recipe_cid`: `"bafkreifzdty5u3uf34z7id3e2mjdidy3fkoucloe2qnudywspggygv4cvy"`
- `positive_recipe_cid`: `"bafkreigxfl76kqkwaydsbbgagirxw6i2fnoymzxcd3sernzkeey3qwa76u"`

## 16. Lean-capable model results

Lean-capable providers remain candidate producers. No campaign proof authority was conferred.

- Section status: `not_run`
- `attempts_admitted_to_curriculum`: `0`
- `role`: `"proposal_only"`

## 17. Tactician results

Tactician surfaces exist in inventory. No campaign tactician lease was granted.

- Section status: `not_run`
- `attempts_admitted_to_curriculum`: `0`
- `role`: `"proposal_only"`

## 18. Hammer results

Hammer/ATP/SMT routing exists. No campaign hammer result became proof authority.

- Section status: `not_run`
- `attempts_admitted_to_curriculum`: `0`
- `role`: `"proposal_only"`

## 19. Kernel-verification results

Independent kernel verification remains the only proof authority. No campaign kernel receipt exists.

- Section status: `not_run`
- `proof_root`: `"bafkreiedk7zooeftd4qnhysbuazs6ulntis3ixn5vye6q7bgtxgrdlrfna"`
- `verified_campaign_proofs`: `0`

## 20. Cross-entropy metrics

Cross-entropy was not measured on admitted heldouts.

- Section status: `not_run`
- `confidence_interval`: `"unavailable"`
- `denominator`: `0`
- `family`: `"token"`
- `hidden_test_used`: `false`
- `missing_as_zero`: `false`
- `numerator`: `0`
- `paired_uncertainty`: `"unavailable"`
- `reason`: `"admission_denied;no_candidate_checkpoint;freeze_descendant_execution_unauthorized"`
- `status`: `"not_run"`
- `value`: `null`

## 21. Cosine and contrastive metrics

Cosine and contrastive metrics were not measured on admitted heldouts.

- Section status: `not_run`
- `confidence_interval`: `"unavailable"`
- `denominator`: `0`
- `family`: `"latent_retrieval"`
- `hidden_test_used`: `false`
- `missing_as_zero`: `false`
- `numerator`: `0`
- `paired_uncertainty`: `"unavailable"`
- `reason`: `"admission_denied;no_candidate_checkpoint;freeze_descendant_execution_unauthorized"`
- `status`: `"not_run"`
- `value`: `null`

## 22. Retrieval metrics

Retrieval metrics were not measured on admitted heldouts.

- Section status: `not_run`
- `confidence_interval`: `"unavailable"`
- `denominator`: `0`
- `family`: `"latent_retrieval"`
- `hidden_test_used`: `false`
- `missing_as_zero`: `false`
- `numerator`: `0`
- `paired_uncertainty`: `"unavailable"`
- `reason`: `"admission_denied;no_candidate_checkpoint;freeze_descendant_execution_unauthorized"`
- `status`: `"not_run"`
- `value`: `null`

## 23. Structural metrics

Structural metrics were not measured on admitted heldouts.

- Section status: `not_run`
- `confidence_interval`: `"unavailable"`
- `denominator`: `0`
- `family`: `"structural"`
- `hidden_test_used`: `false`
- `missing_as_zero`: `false`
- `numerator`: `0`
- `paired_uncertainty`: `"unavailable"`
- `reason`: `"admission_denied;no_candidate_checkpoint;freeze_descendant_execution_unauthorized"`
- `status`: `"not_run"`
- `value`: `null`

## 24. Semantic metrics

Current-input semantic campaign metrics were not measured. Historical R1 scores stay unqualified.

- Section status: `not_run`
- `confidence_interval`: `"unavailable"`
- `denominator`: `0`
- `family`: `"semantic"`
- `hidden_test_used`: `false`
- `missing_as_zero`: `false`
- `numerator`: `0`
- `paired_uncertainty`: `"unavailable"`
- `reason`: `"admission_denied;no_candidate_checkpoint;freeze_descendant_execution_unauthorized"`
- `status`: `"not_run"`
- `value`: `null`

## 25. Proof metrics

Proof replay was not measured on admitted heldouts.

- Section status: `not_run`
- `confidence_interval`: `"unavailable"`
- `denominator`: `0`
- `family`: `"proof"`
- `hidden_test_used`: `false`
- `missing_as_zero`: `false`
- `numerator`: `0`
- `paired_uncertainty`: `"unavailable"`
- `reason`: `"admission_denied;no_candidate_checkpoint;freeze_descendant_execution_unauthorized"`
- `status`: `"not_run"`
- `value`: `null`

## 26. Calibration and OOD metrics

Calibration and OOD metrics were not measured on admitted heldouts.

- Section status: `not_run`
- `confidence_interval`: `"unavailable"`
- `denominator`: `0`
- `family`: `"calibration_ood"`
- `hidden_test_used`: `false`
- `missing_as_zero`: `false`
- `numerator`: `0`
- `paired_uncertainty`: `"unavailable"`
- `reason`: `"admission_denied;no_candidate_checkpoint;freeze_descendant_execution_unauthorized"`
- `status`: `"not_run"`
- `value`: `null`

## 27. Resource utilization

No GPU, prover, or training resource lease was granted.

- Section status: `not_run`
- `confidence_interval`: `"unavailable"`
- `denominator`: `0`
- `family`: `"latency_resource"`
- `hidden_test_used`: `false`
- `missing_as_zero`: `false`
- `numerator`: `0`
- `paired_uncertainty`: `"unavailable"`
- `reason`: `"admission_denied;no_candidate_checkpoint;freeze_descendant_execution_unauthorized"`
- `status`: `"not_run"`
- `value`: `null`

## 28. Multi-supervisor scheduling results

Multi-supervisor inventory and campaign control exist. Learning stages did not overlap because no training lease issued.

- Section status: `resolved`
- `descendant_execution_authorized`: `false`
- `resource_profile`: `"RP-CPU-M"`
- `supervisor_inventory_cid`: `"baguqeerablvf72zunpjvbievbspxqnc4eqgxneqjwrg5v6imr7edavovmwca"`

## 29. Checkpoint promotion or rejection decision

Every non-compensable M2 gate is represented. No candidate existed, so promotion is no_go.

- Section status: `no_go`
- `candidate`: `null`
- `decision`: `"no_go"`
- `human_approval`: `false`
- `m2_gates`: `["lineage", "syntax", "type", "semantic", "proof", "calibration", "family", "jurisdiction", "source_span", "latency", "resource"]`
- `pointer_mutated`: `false`

## 30. Published artifacts

No remote artifact was published. The local PGIR-090 package remains unqualified packaging evidence.

- Section status: `denied`
- `local_release_cid`: `"bafkreigwdei25h3eg2k2l6gp6ak5tbkbcfabi6vsoqzrzv6k6mpm73lhge"`
- `local_release_id`: `"sha256-b8b062360926fa1fb09c22f44740982f9401f435b524cc41db08e093a206c425"`
- `p4_evidence_cid`: `"bafkreia35e5pexkbnq7x2lqtoomcwx34hceroyzsltkb4rqirjjvlqkdle"`
- `remote_revision`: `null`
- `upload_authorized`: `false`

## 31. Known limitations

Zero rights-admitted rows, unmaterialized corpus, incomplete holdouts, unqualified historical baseline, and no learned tokenizer block qualification.

- Section status: `resolved`
- `reason_codes`: `["corpus_not_materialized", "historical_semantic_baseline_not_currently_qualified", "no_candidate_checkpoint", "no_learned_tokenizer_admitted", "no_rights_admitted_training_rows", "publication_not_authorized", "required_holdouts_insufficient"]`

## 32. Exact recommendation for the next training and data-improvement board

The next board is docs/architecture/proof_grounded_ir_learning/next.todo.md. It starts with rights, corpus materialization, holdouts, tokenizer, baseline requalification, a superseding freeze, then R1-R6 and requalification.

- Section status: `resolved`
- `next_board_path`: `"docs/architecture/proof_grounded_ir_learning/next.todo.md"`
- `next_task_ids`: `["PGIR-200", "PGIR-201", "PGIR-202", "PGIR-203", "PGIR-204", "PGIR-205", "PGIR-206", "PGIR-207"]`

## Authorized closing claim

The qualification decision is `no_go`. The exact required qualified-claim
text is withheld because the admitted gates did not pass. No candidate
checkpoint exists. Promotion is `no_go`. Remote publication is `denied`.
This is an integrity success and an execution denial, not a fabricated
training result.

Reason codes:

- `corpus_not_materialized`
- `historical_semantic_baseline_not_currently_qualified`
- `no_candidate_checkpoint`
- `no_learned_tokenizer_admitted`
- `no_rights_admitted_training_rows`
- `publication_not_authorized`
- `required_holdouts_insufficient`

Never claim universal legal-semantic understanding.
