# Optional 384D Intent action advice

`intent_384_advisor` consumes the datasets-owned action-contract runtime through
an explicit local checkpoint selection. It is separate from the existing
paired-text Intent advisor. The historical published atom-only Intent weights
do not gain effect statements through this adapter.

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
route remains available unchanged. Call `prepare_intent_384_advice` explicitly
before creating that task context; the context hook does not invoke this model
automatically or install a default preplanning route. This opt-in adapter does
not start the full supervisor daemon or promote the checkpoint to the default.

The caller still supplies the action/source association, finite input domains,
typed state-variable mapping, and effect formulas described in
[explicit Intent/code effect advice](intent_code_effects.md). This inference
adapter does not associate a task or code file automatically. Learned contracts
remain candidate interpretations; source meaning, normative compliance, proof,
execution, and completion authority remain false.
