# Optional 384D Intent action advice

`intent_384_advisor` consumes the datasets-owned action-contract runtime through
an explicit local checkpoint selection. It is separate from the existing
paired-text Intent advisor. The historical published atom-only Intent weights
do not gain effect statements through this adapter.

The Terminal-Bench supervisor startup now selects this adapter with
`--intent-action-384-config /absolute/config.json`. The option is available in
`terminal_indexed_preparation prepare` and `terminal_container_supervisor`.
Inference precedes goal declarations. Preparation pins the configuration bytes
and saves the advice; the model planner reloads it, repeats numerical inference,
and receives a bounded summary of the decoded contract alongside the original
instruction. Changed configuration, source, or saved advice fails open without
supplying a candidate summary. `--disable-intent-autoencoder` disables this route.
Legacy Intent and source-unit selections cannot be combined with this option.

The deterministic requirement-contract v2 planner continues to use independently
reviewed operations. Its report says `not_used_for_symbolic_selection`; merely
running Intent preprocessing does not make the learned candidate select tasks.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.intent_384_advisor import (
    prepare_intent_384_advice,
    validate_intent_384_advice,
)

config = {
    "schema": "supervisor-intent-action-384-config/v1",
    "checkpoint_path": "/models/intent-action-structured384.json",
    "checkpoint_sha256": exact_checkpoint_sha256,
    "embedding_snapshot_path": "/models/gte-small",
}
advice = prepare_intent_384_advice(instruction=original_instruction, config=config)
```

Select a checkpoint trained for the shared action-contract profile. The shared
runtime uses the pinned GTE-small embedding assets and the exact structured
checkpoint. A missing checkpoint, unsupported instruction, wrong prediction,
or unavailable optional dependency keeps planning available and returns an
explicit failure disposition. Nothing trains or downloads weights here.
`embedding_snapshot_path=None` permits the existing pinned local cache lookup.
Omitting the configuration disables this adapter without importing its model.

The advice retains `raw_candidate_ir` exactly as produced by the learned
decoder. `candidate_intent_ir` separately contains the native document whose
source provenance was bound by the shared owner after its source audit. Both
have independent digests. Source binding cannot insert an effect, replace a
semantic prediction, or turn a permission-only instruction into a postcondition.

Preparation verifies the shared numerical replay. Calling
`validate_intent_384_advice(advice, instruction=original_instruction)` replays the
same source, embedding selection, checkpoint, candidate, and provenance binding
before accepting saved advice. Editing and rehashing a saved document does not
make it valid. The original instruction remains a separate input.

To use the existing explicit Intent/code effect check, pass this advice as
`intent_code_effect_intent_advice` to `prepare_supervised_task_context`, together
with `intent_code_effect_instruction`, `intent_code_effect_config`, and the
existing `security_source_program_config`. The effect consumer recognizes the
new advice schema and verifies it through this adapter. Its older paired-text
route remains available unchanged. Supply the startup advice, or call
`prepare_intent_384_advice` explicitly before creating that task context. The
context hook does not select or invoke the Intent model automatically. This
option does not promote the checkpoint to the default.

The caller supplies the selected source/action, finite input domains, and input
parameter mapping described in [explicit Intent/code effect advice](intent_code_effects.md).
Configuration v2 generates typed formulas from the source-audited decoded fields;
v1 retains caller-authored formulas. This inference adapter does not associate a
task or code file automatically. Learned contracts
remain candidate interpretations; source meaning, normative compliance, proof,
execution, and completion authority remain false.

For a container, pass the same `--intent-action-384-config` option to
`terminal_deployment build`. Packaging requires an explicit
`embedding_snapshot_path` pointing to the pinned GTE-small snapshot. It checks
the Intent checkpoint/domain and every embedding asset, copies regular bytes,
and rewrites paths beneath `/opt/ipfs-supervisor/models/intent-action-384/`.
No weights are downloaded at startup. The archive manifest carries the selection;
`FullSupervisorAgent` forwards it to both full and no-index arms unless Intent
preprocessing is disabled. `full_supervisor_benchmark prepare` can additionally
receive the host config to check that it matches the packaged selection.
Packaging Intent embeddings alone does not enable code-retrieval embeddings.

The published experimental action checkpoint remains limited to controlled
scalar contracts. The original `fix-code-vulnerability` prompt is outside that
grammar and returns `fail_open_input_out_of_scope`. Startup wiring does not
extend the checkpoint's training coverage or establish a new benchmark score.
