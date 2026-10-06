# Original contextual IR checkpoint recovery

The October 6 recovery registered the two retained LegalIR contextual selected
states in the explicitly selected ModelManager store. The original checkpoint,
paragraph/clause embedding caches, normalization and codec bytes were reused.
No encoder, decoder inference or training ran in this increment.

## Registration and source recovery

The store is `/home/barberb/lift_coding/external/ipfs_accelerate/model_manager.duckdb`.
It contains **670 records**, including the two new states below. Genuine
`ModelManager.add_model` calls, independently reopened native DuckDB reads,
both genuine closes and a cold genuine reload verified persistence. All 668
pre-existing records remain exactly equal, including their activity timestamps;
the schema and indexes remain unchanged. Only the owned registration instance's
in-memory save population was narrowed to the two new IDs. No other live
manager process was refreshed. The active instance's catalog and the full cold
catalog both expose the new records.

Appending records changes the persisted catalog generation. Previously sealed
task contexts that bind the older generation must be regenerated through their
existing reviewed preparation path. Preserve the old bundles and their pins;
do not rewrite a stored generation field to make a stale context load.

| LegalIR input lane | Original complete state SHA256 | ModelManager record suffix |
| --- | --- | --- |
| 384D | `0ac5c21656187d1d8040a02bcc0b4716a8093db17f05adc9a22e99d5b6b2cc00` | `10485c538f0f4a64e9ec0b6fa1c62e06af1f368446cd2c0e8e342e127e9055d9` |
| 768D | `8892a3261c0750a6247ba5400069ed18cad3354426e2095ed1d295287e3559b4` | `c992440d87ae57cffb689aa37fe2f3b63e9ae4c251a9ad65697148e576f840e2` |

Both record IDs use the `ir-model-asset-binding/v1:` prefix. The dimension role
is `input_embedding`, the role is `selected_contextual_semantic_decoder_state`,
and the retained task label is `semantic_IR_reconstruction`. Native output
schema/version, profile ID and format ID remain **null**. The known
`private-native-dimension-source-state/v1` checkpoint serialization schema does
not establish an IR output schema. Each record retains its original pin and
immutable publication location; no pre-existing unknown-profile declaration
was replaced. Complete runtime IO qualification, runtime readiness, teacher
qualification and proof authority remain false.

The [registration receipt](evidence/contextual-state-registration-20261006/model-manager-registration.json)
and [independent driver review](evidence/contextual-state-registration-20261006/review/registration-driver-review.json)
bind the actual API run and its original files. The
[independent postwrite audit](evidence/contextual-state-registration-20261006/review/registration-postwrite-review.json)
reopens all 670 persisted records and confirms the unchanged baseline and schema.
The registration driver disables
optional provenance/storage/GraphRAG owners, network access and unreviewed
extra-process execution. Exact read-only Git source checks remain permitted.
It seeds a standard logger before executing the unchanged genuine module,
because its optional import handlers otherwise reference that name before its
normal assignment. This is an isolated metadata registration scope, not an
inference or service launch qualification.

The supervisor's existing exact nomination and byte-authentication APIs also
[resolve both new records](evidence/contextual-state-registration-20261006/supervisor-registered-state-observation.json)
from the persisted store and authenticate the original checkpoint bytes. This
does not supply a contextual decoder runtime or give a candidate proof status.

Both complete states are also published in their separate existing public
dimension repositories: [LegalIR 384D release](https://huggingface.co/Publicus/legal-ir-autoencoder-384d/blob/f2c3714f7bf8e30177a799270f0cead51baf19dc/releases/20261006-contextual-selected-v1/README.md)
and [LegalIR 768D release](https://huggingface.co/Publicus/legal-ir-autoencoder-768d/blob/2fc77ccd037e7276c8bdd56a2c79cada0334fd79/releases/20261006-contextual-selected-v1/README.md).
Each release adds exactly three files: the unchanged complete selected state,
its contextual contract manifest and a scoped README. The
[publication and preservation receipts](evidence/contextual-state-registration-20261006/dimension-mirrors/publication-summary.json)
verify all 502 prior 384D paths and 142 prior 768D paths unchanged, including
root cards/defaults, public visibility and gated settings. Original cached
inputs are referenced by exact pins, not copied into these mirrors. ModelManager
retains its first verified immutable aggregate release binding; equivalent-byte
mirror receipts are append-only observations rather than record replacements.
Separate [fresh mirror downloads](evidence/contextual-state-registration-20261006/dimension-mirror-fresh-downloads.json)
also match both original complete state hashes.

Datasets `origin/main` at `3b3b994407b2fcfef955ce5153afc2eea01e4eff`
already restored the six original metadata owners and their six tests. Their
bytes match the retained `73db2c8f` source. The shared checkpoint loader preserves
its newer Legal optimizations and the four additive replay entry points.
No source restoration or hybrid import was needed in this increment.

The [current native qualification](evidence/contextual-state-registration-20261006/native-owner/qualification.json)
rebuilds all twelve original family/dimension declarations through the public
owner and resolves both original IntentIR/SecurityIR 384D fragment routes.
Their format, profile and checkpoint-record identities remain unchanged. A new
detached inventory binds current source locations explicitly; the original two
historical inventories remain unchanged and still refuse their old source
custody in the current checkout. Those fragment contracts do not admit the
contextual Legal states or fill other families' missing format identities.

## Preserve the actual contextual inputs

The [custody survey](evidence/contextual-state-registration-20261006/asset-survey/contextual-state-custody-survey.json)
freshly binds 44 files and checks the original source rows, text spans, offsets,
vector hashes, masks and frozen transformations. Each width has 48 TRAIN and
48 exposed validation paragraphs, with 180 clause occurrences in each split.
Both source-only validation copies match the original cached subsets.

The contextual model requires the native paragraph vector **and** ordered
literal-source-clause vectors. Its batching producer applies the saved TRAIN
input transform to raw clause vectors, then zero-pads to eight slots and makes
a boolean `[batch, 8]` mask from the original clause counts. The model's frozen
paragraph/clause feature normalizations are separate decoder stages. Preserve
this order; do not apply feature normalization again when building the padded
packet. The stored broad phrase “padding after normalization” is interpreted
by this precise recipe and its pinned original producer, not as a new pipeline.
The mask is constructed from original counts; it is not a separate cached mask
file or information taken from evaluation targets.
The [separate semantic plan review](evidence/contextual-state-registration-20261006/asset-survey/independent-plan-review.json)
checks both exact state/cache joins and retains this recipe precision note
without rewriting the registered metadata.

The paragraph normalization was fitted on 48 TRAIN vectors. The clause
normalization was fitted on 113 unique TRAIN clauses. Neither used validation
rows for fitting. Preserve the ordered `typed-json-lexical/v1` codec, its
`cba3e5384e2bee12e709ccc8e9ee430e51cf09a8fc2e8abdd57143fdaa38dba2`
digest and the inherited 512-token output budget. Its vocabulary size of 32 is
not a 32-token output limit.

The [producer joins](evidence/contextual-state-registration-20261006/asset-survey/source-producer-joins.json)
bind each original vector to its actual producer receipts. The 384D paragraph
and clause paths preserve their distinct CPU/CUDA GTE-small provenance. The
768D cache mixes retained complete-native receipts and later bounded 512-token
receipts. A native 8192-token profile declaration does not establish that these
cached experiments encoded or decoded 8192-token paragraphs.

Historical replay reports **48/48 exact canonical IRs, 180/180 rules and 720/720
actor/action/modality/object fields** at both widths. This increment did not
rerun that numerical comparison. Those scores describe the specified exposed
authored panel with clause context. They do not measure originating legal prose,
fresh held-out generalization, a single paragraph vector or an 8192-token span.
The earlier fragment checkpoints and these complete contextual states retain
different contracts and evaluation populations.

## Next implementation gates

The later October 6
[cached contextual runtime and reconstruction measurements](https://github.com/endomorphosis/ipfs_datasets_py/blob/main/docs/autoencoders/contextual_legal_reconstruction_runtime.md)
implement the explicit original-asset replay portion of gates 1 and 2 below.
Frozen datasets source `37c2d63f0bf9490e5b019788afcafbc3b806d8c7`
restores every selected tensor from the existing 384D/768D states, original
raw384 donor and saved preprocessing. The public metadata path remains
Torch-free; opening and inference explicitly use the source-only cached
paragraph/clause packets. Fitting helpers, optimizers, encoders, databases,
network operations and historical reference/evaluation-file reads are blocked
in the actual qualification run. The original 44 surveyed files remain
unchanged. The only existing numerical-owner edit moves an unused training
helper import into its training-only function.

Fresh replay reproduces **48/48 ordered exact semantic IRs and 180/180 rules at
each width**, with exact token/status/EOS parity against the original retained
outputs. The separate Torch-free evaluator also observes **0/48 original-text
UTF-8 matches and 0/48 NFC/whitespace matches at each width** when the existing
source-withheld canonical decompiler renders the generated IR. All 48 text
outputs are present; this is not a missing-output denominator effect. That
owner receives no source text or reference IR and is a deterministic baseline,
not a trained prose decoder. Every reference qualifier is empty; the exposed
authored panel does not establish fresh-holdout or broad legal-meaning quality.
The source change passed 244 contextual/source-value and 210 historical-owner
regression controls, with separate independent source and execution reviews.

This runtime does not update the registered record IDs, immutable Hugging Face
bindings, database generations or existing native fragment routes. There are
no new weights or embeddings to register or upload. Its generated documents
remain candidates; native schema/profile/format identities and teacher,
production runtime, holdout and proof authority remain unqualified. The native
canonical output inspector's contract conformance does not mint a checkpoint
profile identity. The linked plan now separates original-text decoder training,
fresh semantic holdout/qualifier ablations, warm-started 8D/384D-to-768D task
distillation and independent source/output token-budget increases.

For the supervisor, use these retained-state observations when choosing
candidate assets. Admit an execution route only through its complete native
task contract and source-currentness gates. Keep family/width/schema/task
namespaces separate, and require current repository evidence and an independent
checker before any generated CodebaseIR or logic projection becomes a checked
proof-index entry. On-the-fly training records a new run/catalog generation;
it does not rewrite a sealed planner context or grant its output proof status.

1. Add an explicit contextual Legal decoder runtime that authenticates the
   complete state, source/context generations, ordered codec, TRAIN transforms,
   mask producer and frozen split identities before numerical loading. Infer
   only from source inputs; keep gold targets in a separate evaluator. Recover
   and independently validate the actual generated IR dialect and version
   before assigning a native format/profile identity. Preserve the current
   asset registrations as historical observations.
2. Replay the original numerical panel as a regression, then reserve fresh
   source groups before any new fitting or selection. Report generated semantic
   exactness, per-field/rule coverage, syntax validity, refusal coverage and
   held-out source fidelity independently. Compare paragraph-only and
   paragraph-plus-clause conditioning on matched examples and decoding budgets.
3. Give `legal text -> legal_ir -> legal text` its own decoder task and checkpoint
   namespace. Measure verbatim bytes, normalized text and reviewed legal meaning
   separately. Define which metric the experiment optimizes. Preserve lexical
   anchors or an explicit residual for information discarded by semantic IR;
   count those inputs and include an ablation without them. Canonical formula
   decompilation does not establish original prose recovery.
4. Warm-start future 768D training from the authenticated 8D/384D teachers and
   compatible decoder tensors, with explicit projection/adaptor contracts.
   Preserve donor hashes, original cached embeddings and labels; compare the
   reused initialization against a documented control. Neither today's
   registration nor historical contextual selection establishes that cross-width
   distillation has executed or that a teacher is qualified for every output.
5. Increase source-span and output budgets independently after the matching
   shorter-span path passes. Use retained 768D native producer assets where
   compatible, and authenticate any additional long-context cache separately.
   Validate ordering, references, qualifier scope, negation and multiple rules
   as spans grow. Keep unsupported cases explicit rather than truncating silently.
6. Retain separate CodebaseIR, SecurityIR, LegalIR and IntentIR inventories,
   physical DuckDB/DuckLake lanes and Hugging Face repositories at 8D/384D/768D.
   Within a lane, separate schema/version, decoder task and ablation/run records.
   Bind repository scans and IntentIR matching to the current commit/content
   generation before planning; autoformalized candidates enter the proof cache
   with their producer/profile identities. Only the applicable independent
   validation backend can upgrade them to checked proof-index entries.

The current increment passed 158 existing importer/profile controls and the
actual original-asset, publication-byte, registration, cold-reload and supervisor
nomination checks above. These operations establish availability and custody.
They do not grant model quality, source semantics, runtime execution, proof or
completion authority.
