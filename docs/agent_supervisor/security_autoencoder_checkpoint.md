# Frozen Security-CVE checkpoint operations

This guide describes the implemented local export, frozen inference, catalog,
Hub transport and isolated supervisor interfaces. It does **not** announce an
uploaded model or activate a default checkpoint. As of 2026-09-29,
`Publicus/security-ir-autoencoder` exists as a public model repository containing
only its bootstrap `.gitattributes`; the checkpoint and model card are not yet
uploaded. Exact release approval is pending. A usable remote descriptor must come
from successful publication and pinned readback.

## Dependencies and model scope

Standalone training and inference require `ipfs_datasets_py`; supervisor catalog
and world-state integration additionally use `ipfs_accelerate_py`. See the
[ownership and checkpoint migration guide](../../../ipfs_datasets/docs/autoencoder_ownership.md). Frozen
loading uses the native `CheckpointManifest` contract and CPU PyTorch float64
inference. It does not import the modal trainer, initialize weights, open the
LegalIR teacher or require the historical task repository/CVE export. The package
pins installed inference and feature-projection source hashes; install the
qualified runtime revision rather than downloading executable model code.

Export has additional dependencies: the original validated seven-tensor training
receipt, unchanged training inputs, its isolated lexical initializer, canonical
CVE training artifacts and native training validators. Export copies existing
trained parameters and verifies numerical parity; it performs no training.

The current development model reconstructs 44 AST/control-flow features and
predicts audit/classification, vulnerability polarity and observed CWE labels.
Its training lineage is benchmark-informed and transductive: 358 task functions
and two external CVE body/target pairs in the qualified example. Scores are
uncalibrated; body-to-function transfer and held-out performance are unvalidated.
There is no formula decoder or TLA+ proof head. Predictions grant no omission,
execution, proof or completion authority. LegalIR heads and mutable state stay
outside the security package.

## Export and load

The reusable APIs live in [datasets-owned security_autoencoder_checkpoint.py](../../../ipfs_datasets/ipfs_datasets_py/logic/formalization/autoencoder/security/security_autoencoder_checkpoint.py).
All output directories must be fresh, canonical absolute paths outside the source
repository and existing model artifacts.

```python
import json
from pathlib import Path
from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_checkpoint import (
    export_security_checkpoint, load_security_checkpoint,
)

trained = json.loads(Path("/absolute/training-descriptor.json").read_text())
package = export_security_checkpoint(
    repository=Path("/absolute/original-training-repository"),
    expected_receipt=trained,
    output=Path("/absolute/new-security-package"),
    provenance_review=json.loads(Path("/absolute/provenance-review.json").read_text()),
    model_card=Path("/absolute/MODEL_CARD.md").read_text(),
)
loaded = load_security_checkpoint(
    Path(package["output"]), expected_manifest_sha256=package["manifest_sha256"]
)
```

Omitting both review/card arguments creates explicitly unreviewed local development
metadata. A public release requires both, with the reviewed schema and exact
parent, training, vocabulary and CVE lineage bindings. A provenance review records
known rights and limits; it does not manufacture a blanket inherited-weight license
or authorize publication.

The closed package has exactly nine files:

| File | Purpose |
| --- | --- |
| `checkpoint.json` | Seven inert named tensors; no pickle or optimizer |
| `config.json` | Architecture, feature rules and installed inference-code pins |
| `vocabularies.json` | Ordered features, lexical keys and candidate targets |
| `lineage.json` | Recorded ancestry, training configuration and explicit unknowns |
| `checkpoint-manifest.json` | Native security-domain checkpoint identity |
| `inference-fixture.json` | Authored numeric probe with replayable outputs |
| `README.md` | Version-specific model card |
| `provenance-review.json` | Review bound to this model's lineage |
| `release-manifest.json` | Hash/size ledger for the other eight files |

The release-manifest SHA is pinned externally. Raw code, training bodies, original
teacher weights, authentication material and remote Python code are excluded.

## Frozen observations and ModelManager

For numeric probes, `score_security_observations(checkpoint=package,
observations=...)` accepts the closed normalized feature/lexical-index format
illustrated in `inference-fixture.json`. Those vectors alone establish no source
identity. Use `infer_security_checkpoint(repository=..., paths=...,
source_hashes=..., checkpoint=package, output=...)` for observations bound to exact
current permitted Python source hashes.

For indexed supervisor advice, use
[prepare_security_advice](../../ipfs_accelerate_py/agent_supervisor/runtime/security_autoencoder_advisor.py)
with the same repository, paths, source hashes, checkpoint and fresh output. It
registers the real installed adapter in an isolated ModelManager, runs inference,
and projects/hydrates the model, observations and world record through native
ProgramWorldDatabase and SupervisorMetaIndex/DuckLake. Validation binds an exact
persisted-file inventory and reads the actual projected DuckLake catalog/parquet
rows; missing or corrupt lake files invalidate advice. This path additionally
requires the configured native DuckDB/DuckLake runtime. It returns a receipt;
`validate_security_advice(repository=..., expected_receipt=...)` returns the bounded
planner summary only after checking current source and artifact bindings.

The [Hub/catalog module](../../ipfs_accelerate_py/agent_supervisor/runtime/security_autoencoder_hub.py)
also exposes `register_security_checkpoint(manager=..., package=...,
expected_manifest_sha256=..., hub_descriptor=None)` and
`score_registered_security_observations(manager=..., model_id=...,
observations=...)`. Registration runs a real numeric probe. The only advertised
operation is **`security.advise`**, with JSON input/output; this is not a generative
`llm_router` provider or general text embedding endpoint. Scoring rejects models
or providers whose lifecycle/readiness or operational state no longer permits
execution, then revalidates the package. Local packages need no invented Hub URI.

After an edit, old observations are historical and fail current-source validation.
`refresh_security_advice(repository=..., previous=..., output=...)` recomputes
observations using the same frozen checkpoint. It does not retrain or download.
Planner/admission replay checks these bindings; symbolic extractors, contract
checks and provers independently establish any admitted formal claims.

## Pinned Hub resolution and cache

An actual Hub descriptor has the following shape. The placeholders below are not
a published revision:

```json
{
  "schema": "security-autoencoder-hub@1",
  "repository_id": "Publicus/security-ir-autoencoder",
  "repository_type": "model",
  "revision": "<full 40-character lowercase commit SHA>",
  "release_prefix": "releases/sha256-<64-character release-manifest SHA>",
  "manifest_sha256": "<64-character release-manifest SHA>"
}
```

`download_security_checkpoint(descriptor=..., cache_root=...)` resolves only the
closed file list, verifies native pinned readback and strict model inference, and
returns `package`, `checkpoint` and `hub`. Cache files are immutable and addressed
by manifest hash. `local_files_only=True` permits a verified warm cache and rejects
a miss. Warm reuse rechecks content against the supplied pin; it does not contact
the Hub or independently verify that a different supplied repository/revision
also serves those bytes. Preserve the original publication/readback receipt when
making remote provenance claims. Invalid pins, corrupt files and missing assets
fail rather than triggering training or provider fallback.

## Full supervisor runtime archive

The [runtime archive builder](../../benchmarks/agent_supervisor/container_coding/terminal_deployment.py)
accepts either a local package or an explicit Hub descriptor. The controller
downloads Hub assets before packaging; task containers use only bundled files.

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_deployment bundle \
  --output /absolute/new-runtime-archive \
  --source /absolute/ipfs_accelerate \
  --datasets /absolute/ipfs_datasets \
  --kit /absolute/kit \
  --extension-dir /absolute/duckdb-extensions \
  --security-checkpoint /absolute/security-package \
  --security-checkpoint-manifest-sha256 "$SECURITY_MANIFEST_SHA"
```

For Hub resolution, replace the final two options with
`--security-checkpoint-hub-descriptor /absolute/hub-descriptor.json` and
`--security-checkpoint-cache /absolute/security-cache`. Keep the existing required
Lean, embedding-model and other profile dependencies configured for the chosen
supervisor task. Frozen selection is mutually exclusive with the task-training
options `--security-initializer`, `--canonical-cve-export` and
`--canonical-cve-manifest-sha256`.

The archive binds the package under `models/security-autoencoder` and optional
Hub provenance under `models/security-autoencoder-hub.json`. Its manifest records
the relocated `/opt/ipfs-supervisor/models/security-autoencoder` descriptor. The
full Harbor arm forwards local `--security-checkpoint`,
`--security-checkpoint-manifest-sha256` and optional
`--security-checkpoint-hub-descriptor` to
[terminal_container_supervisor.py](../../benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py).
The no-index arm receives no checkpoint. Workers have no Hub-cache/download option.

Explicit frozen selection sets `train_autoencoder=False`; there is no implicit
conversion from training mode or default production activation.
[prepare_initial_context](../../benchmarks/agent_supervisor/container_coding/terminal_initial_context.py)
accepts `security_checkpoint` and optional `security_checkpoint_hub`, freezes the
selection, and supplies validated advice to the actual planner and admission
checks. Post-publication refresh keeps fixed weights and updates source-bound
observations; exhausted budget is recorded as a deferred refresh with historical
observations rather than falsely current scores.

## Qualification and publication boundary

Run focused package/Hub/advisor tests in the qualified dependency environment.
The provider-free native integration requires installed Quack, Z3 and Lean, plus
an explicit portable package:

```bash
IPFS_SECURITY_CHECKPOINT_PACKAGE=/absolute/security-package \
IPFS_SECURITY_CHECKPOINT_MANIFEST_SHA256="$SECURITY_MANIFEST_SHA" \
python -m pytest -q test/integration/test_native_frozen_security_checkpoint.py
```

Missing prerequisites skip this integration test. A passing run exercises native
START, materialization, public validation, publication, STOP and frozen observation
refresh on an authored task. It does not measure a held-out benchmark, neural
repair benefit or learned goal decomposition. The reviewed Doctor contract remains
independent of candidate scores. Qualification timing records frozen preparation
separately from its main `seconds` field.

Create a reviewable offline proposal with
`plan_security_checkpoint_publication(package=..., expected_manifest_sha256=...,
repository_id="Publicus/security-ir-autoencoder", private=False,
audited_parent_commit=...)`, then save `publication.to_dict()`. It includes the
native plan digest, destination, visibility, exact file/byte inventory and cost
bound. Public provenance validation and a successful local probe do not upload.

The native [PublicationApproval](../../../ipfs_datasets/ipfs_datasets_py/huggingface/publisher.py)
contract requires **"Explicit human approval of one exact dry-run plan digest and
cost bound."** Supply that approval to `publish_security_checkpoint` only after
review of the concrete final plan. Broad intent to publish does not supply this
exact approval. The repository must already exist with the selected visibility
and a real audited parent commit; creation/bootstrap is separate. Parent drift
requires a fresh plan and approval. Publication appends an immutable release,
performs pinned redownload and real inference, and returns the actual full-commit
Hub descriptor. It does not promote a runtime pointer or update the repository's
root model card. Manifest hash/approval binding is not a cryptographic signature.

Further formal-candidate heads, clean dataset splits, calibration and production
promotion criteria remain in the
[checkpoint and formal-planning roadmap](../../benchmarks/agent_supervisor/container_coding/SECURITY_CHECKPOINT_PLAN.md).
