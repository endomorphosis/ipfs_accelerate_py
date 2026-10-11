# Experimental numerical checkpoint in the supervisor

The explicit `numpy-unary-selection-384@1` profile consumes the published
development checkpoint through the shared datasets owner
`logic.software_contracts.codebase_numerical_selection_384`. Training, embedding
execution, decoding and the immutable model operation remain in datasets. The
supervisor binds the retained result to its signed source population, native
world and metadata DuckLakes, planning summary, task bundle and coding worker.

This checkpoint supports a narrow source grammar: one plain function in its
original module, with imports, returning a direct unshadowed `numpy.argmax` or
`numpy.argmin` call on one of one to four parameters. It is not a general NumPy
decoder. It uses the frozen Legal384 projection and a learned ridge head. A
source-aligned candidate may reference independently generated program, FOL and
Lean artifacts. Mismatches retain the actual prediction and withhold those
references; no deterministic repair replaces a prediction. These are advisory
CALL fragments with pending obligations, not proofs of NumPy behavior or task
correctness. No source obligations are discharged.

## Select cached assets explicitly

Use the existing local release checkpoint and complete pinned GTE-small snapshot.
This descriptor helper performs no download:

```python
import json
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.runtime.numerical_selection_config import (
    published_numerical_selection_config,
    validate_numerical_selection_assets,
)

config = published_numerical_selection_config(
    checkpoint_path=Path("/absolute/local/checkpoint.json"),
    embedding_snapshot=Path("/absolute/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd"),
)
validate_numerical_selection_assets(config)
Path("numerical-selection.json").write_text(json.dumps(config, sort_keys=True))
```

The selected release is `Publicus/security-ir-autoencoder` revision
`38666f43e998d55b5f4259e354286f5929027ba1`, file
`releases/20261011-numpy-unary-selection-384-development-v1/checkpoint.json`,
with raw file SHA-256
`26ccc7cf8420e0ed6cbbe22603e09be0e3572fb32a1e489e877fc5081f506e03`.
The native inference key separately records its canonical checkpoint, frozen
parent, projection and head digests. The adapter does not modify LegalIR weights.

Pass the absolute descriptor path as `numerical_selection_config` to
`terminal_indexed_preparation.initial_context`, or use
`--numerical-selection-config` with `terminal_native_preflight`,
`terminal_container_supervisor`, or `terminal_deployment bundle`. For a portable
archive, the builder validates and copies checkpoint/config/GTE files under
`models/numerical-selection-384/` and verifies the selected producer bytes in the
tar archive. It rejects setup-cache reuse for this profile. The Harbor full arm
requires an explicit common resource profile; the no-index arm records the
selection as disabled and does not invoke its inference. Use the same resource
profile across comparison arms.

Selection is opt-in and learned-only container execution does not start legacy
training. Explicit frozen Security checkpoint advice can coexist. Incompatible
training declarations and empty program populations are refused explicitly.
All source and model assets must remain available for validated replay.

## Bounds and lifecycle

The owner accounts for up to 256 declared paths and selects at most 128 eligible
modules. Unsupported modules and inputs above 512 tokenizer tokens are recorded
as deferred or unsupported; they are not silently truncated. Identical source
bodies can share one embedding, while distinct path/source identities remain
separate. The current adapter uses CPU inference with cached GTE assets.
`model_loads` records embedding-model initialization in the original execution;
it is not a count of checkpoint parsing during replay.

Initial preparation and current-source validation bind exact source, checkpoint,
producer and embedding runtime identities. Current validation reconstructs the
native observation and DuckLake links without repeating neural inference.
The normal inference deadline is 180 seconds. Selecting the existing extended
formalization resource profile permits its 3,600-second bound and retains one
overall deadline, cancellation and memory admission.

The planner and required worker evidence receive at most two candidate samples
within an 8,192-byte summary, including explicit omission counts. Full candidates
and projection bodies remain in the retained inference artifact. Task bundle
`@5` can carry learned numerical advice alongside Source384 and deterministic
numerical evidence. The router verifies current bindings before provider
dispatch. The final historical audit checks delivery against the original
captured source even after an expected code edit.

After completion and native STOP, learned advice remains historical. No admitted
learned-successor refresh policy exists yet: refresh reports
`successor_unavailable`, retains every selected family, and does not regenerate
only part of a mixed selection. Reindexing changed source requires an independent
fresh preparation and admission.

Qualification and remaining benchmark limitations are recorded in the
[current supervisor status](../../../docs/agent_supervisor/durable_cleanup_status.md).
