# Security training sources and code logic projections

Reusable model/corpus code is owned by `ipfs_datasets_py`; see the
[ownership guide](../../../ipfs_datasets/docs/autoencoder_ownership.md). The supervisor
only consumes model APIs and adapts the learning declaration into its task graph.

The security training path selects these datasets independently:

- Corpus: [Publicus/cvefixes-security-ir-graphrag](https://huggingface.co/datasets/Publicus/cvefixes-security-ir-graphrag).
- Initializer: [justicedao/legal-ir-autoencoder-checkpoints](https://huggingface.co/datasets/justicedao/legal-ir-autoencoder-checkpoints).

Every selection binds a full Hub commit, manifest hashes, and the relevant
source identities. A repository URL identifies the source; it is insufficient
to reproduce a run without these pins.

The published LegalIR state at revision
`94ca549d102e3e31781370aec1247f91365440eb` has SHA256
`7236de26bd3d7f8414ffa04805f1b6e8a8849f9e0103cec6edb4985b911658be`.
It matches the retained parent used by the earlier security fork. The published
manifest adds provenance for ten contributing runs. It does not retroactively
establish code-domain quality or complete corpus history.

## Published initialization

[published_legal_initializer.py](../../../ipfs_datasets/ipfs_datasets_py/logic/formalization/autoencoder/security/published_legal_initializer.py)
resolves the pinned manifest and card, checks the selected state digest, and
binds a separately owned security initializer through native replay. A retained
immutable snapshot with the same digest avoids another large download.
`published_legal_source_pin(source)` returns a validated host-path-free pin for
the training profile. `validate_published_legal_initializer(...)` rechecks the
new lineage evidence and the original initializer without importing historical
training code by default.

Only compatible lexical embedding rows transfer, with their values and
dimensions unchanged. The code feature projection and security heads have
separate identities. LegalIR heads and checkpoint files are not overwritten.
Neither this binding operation nor frozen inference randomly initializes a
replacement backbone.

## Corpus admission and splits

[security_cve_corpus.py](../../../ipfs_datasets/ipfs_datasets_py/logic/formalization/autoencoder/security/security_cve_corpus.py)
adds explicit train/validation/test repository-family assignments and bounded
original-row access around the existing native canonical exporter. The public
GraphRAG, BM25 and vector tables select and verify source identities; their
labels and graph counts are not returned as code input features.

The reader verifies original-row CIDs, native source/code/policy records and
code-body hashes. It rejects benchmark families before source access and
rejects source/body overlap between splits. Returned model inputs contain only
source-derived lexical observations and supported AST features, with targets
kept separate. Unsupported AST cases remain explicit. Creating a test split
does not mean a model has been evaluated on it. Current ingestion is bounded
qualification tooling, not an assertion of dataset-scale training.

The first three-family qualification admitted six before/after examples. Their
exact recorded bodies are fragments or combined changes; all six are
unsupported by the current complete-module AST adapter. Exact body identity
does not establish complete-file context. The learning plan requires source
context resolution and typed model construction before logic projection.

## Code-specific logic

The native [code logic guide](../../../ipfs_datasets/docs/security_code_logic_projection.md)
describes seven typed-owner projections in `ipfs_datasets_py`:

| Owner | Native family/profile |
| --- | --- |
| Program | `program/program_ir` |
| Contract | `program/dynamic_hoare` |
| Transition system | `transition_system/action_system`, with bounded TLA+ encoding |
| Temporal formula | `temporal/ltl` |
| Heap | `separation_logic/heap_model` |
| Separation formula | `separation_logic/separation` |
| Information-flow model | `hyperproperty/hyperltl` |

Each target requires complete matching source bytes and explicit native typed
evidence. Native syntax round trips and contract/program references are
checked. TLA+ compilation preserves bounds and translation losses and does not
execute a model checker. Missing evidence produces an unsupported frontier;
CWE labels cannot create formulas. Source identity and structural preservation
alone do not prove that an authored model faithfully represents code.

## Native planning

[security_code_training_profile.py](../../ipfs_accelerate_py/agent_supervisor/runtime/security_code_training_profile.py)
consumes the datasets-owned declaration combining corpus, published initializer,
guarded source modeling and installed projection profiles. It distinguishes classification targets, modeling declarations and
independently proved targets. The profile does not advertise trained formula
heads merely because those projections are available.

```python
profile = build_security_code_training_profile(
    corpus_profile=corpus_profile,
    legal_parent=published_legal_source_pin(source_descriptor),
)
planned = compile_security_code_learning_plan(
    profile=profile,
    repository_tree_id=admitted_repository_tree_identity,
)
```

This uses the native `IRLearningCampaign` and formal-plan compiler to produce
15 tasks, including source-context recovery and native model construction.
Independent projection tasks have separate output paths. Contract
projection depends on the exact program-projection result. Fitting depends on
all projection results and the initializer; evaluation precedes release.

The generated campaign is a **draft declaration**. Its source admission and
dependency outputs are unresolved, no leases are granted, and stage-specific
execution/evidence validators are not configured. Its validation command fails
closed instead of substituting profile unit tests for actual training or
evaluation evidence. It does not start a training run, publish weights or
change the currently selected supervisor checkpoint. The reusable inference
path is documented in [the checkpoint guide](security_autoencoder_checkpoint.md).
