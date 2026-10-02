# Optional finite source-state advice

The supervisor can ask `ipfs_datasets_py` to derive a finite operational state
model from an unchanged 384D SecurityIR prediction and its original Python
source. Select the v2 configuration explicitly; the existing v1 configuration
keeps its previous behavior.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.security_source_program_advisor_384 import (
    prepare_repository_source_program_advice,
)

config = {
    "schema": "supervisor-security-source-program-384-config/v2",
    "checkpoint_path": "/models/security-structured384.json",
    "checkpoint_sha256": checkpoint_sha256,  # Independently verified exact file SHA256.
    "decoder": "structured",
    "embedding_snapshot_path": "/models/gte-small",
    "lake": None,  # Or {"executable": "/tools/lake", "timeout_seconds": 60}.
    "finite_state_domains": {
        "example.py": {
            "capacity": {"lower": -1, "upper": 1},
            "threshold": {"lower": 0, "upper": 1},
        },
    },
}

advice = prepare_repository_source_program_advice(
    repository=repository_path,
    paths=["example.py"],
    config=config,
)
state_advice = advice["source_state"]
```

Use an existing local, hash-verified Security checkpoint and the pinned local
GTE-small snapshot. This path performs inference, without training, downloading
weights, executing the analyzed Python, or changing checkpoint selection. The
same configuration can be passed as `security_source_program_config` to
`prepare_supervised_task_context`.

For `decoder="structured"`, v2 uses the shared Security-only compatibility
loader. It admits only the recorded old-to-new UI decoder implementation-pin
change; numerical code, Security code, and every other implementation field
must still match. The loader constructs a detached metadata view and preserves
the checkpoint file and its weights. The supervisor independently replays the
`checkpoint_compatibility` receipt against the exact artifact at loading and
inference, and requires the receipts to agree. Raw and explicitly normalized
embedding views retain that receipt. V1 and `decoder="sequence_v2"` keep their
strict existing loaders and receive no implementation-pin exception.

`finite_state_domains` must cover exactly the captured Python source IDs. Each
source needs two parameter names with inclusive integer `lower`/`upper` bounds;
Boolean bounds are rejected. The product of the two ranges is limited to 64
input cases. The datasets owner checks names, exact source/candidate agreement,
and its supported scalar Python fragment. These ranges are caller assumptions;
the supervisor never infers them from examples, predictions, or the task prompt.

With `lake=None`, `source_state.native` contains prepared declarations and the
per-input state model. Selecting Lake also checks the generated finite
ProgramIR/state correspondence and verifies its live execution handle before
serializing the receipt. This minimal supervisor configuration does not request
SANY; each `sany_status` remains `not_run`. Direct datasets APIs can additionally
select Java and the TLA+ SANY JAR. SANY parses the generated specification; it is
not a model-checking result.

`source_state.rows` maps native inference IDs back to the original source IDs.
Unsupported predictions and abstentions stay visible. Wrong predictions are not
repaired. Invalid ranges, unavailable dependencies, failed live verification,
or oversized optional state output leave the independently valid base inference
available and `continue_planning=True`. A failure before inference, such as an
invalid checkpoint, retains the existing fail-open behavior.

A successful finite correspondence check applies only to the declared input
ranges and the supported ProgramIR interpretation, including its integer
assumptions. It does not prove Python equivalence, a security requirement, or
compliance with an Intent effect. Code hashes and original source bytes remain
independent of prompt hashes; this adapter does not attach code effects to an
Intent world model. Source, proof, execution, and completion authority remain
false, and saved JSON receipts do not become live authority.

The lower-level consumer is
`security_source_state_advisor_384.consume_source_state_advice`. It accepts the
existing inference report, exact `id/source_text/source_sha256` rows,
`source_id/inference_id/source_sha256` bindings, explicit domains, and the
optional Lake selection. Model derivation, formal projections, and prover calls
remain owned by `ipfs_datasets_py`.
