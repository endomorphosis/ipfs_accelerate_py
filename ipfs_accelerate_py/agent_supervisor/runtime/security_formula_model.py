"""ModelManager binding for the datasets-owned, frozen production decoder."""
from __future__ import annotations

from dataclasses import dataclass

INFERENCE_CODE = "ipfs_accelerate_py.agent_supervisor.runtime.security_formula_model:decode_registered_security_formula"


def _probe(checkpoint):
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder import decode_security_formula
    result = decode_security_formula(source_bytes=b"def probe(value):\n    return value + 1\n",
        source_path="model_registration_probe.py", checkpoint=checkpoint)
    return {"executed": True, "status": result["status"],
        "learned_formula_count": result["learned_formula_count"],
        "candidate_validation": result["validation"], "proof_authority": False}


@dataclass(frozen=True)
class _FormulaSource:
    model: object
    provider: object
    checkpoint: dict
    source: str
    precedence: int = 100
    side_effecting: bool = False

    def load(self):
        from ipfs_accelerate_py.model_catalog.schema import CatalogSnapshot
        from ipfs_accelerate_py.model_catalog.sources.static import CatalogSourceResult, SourceMetadata
        _probe(self.checkpoint)
        return CatalogSourceResult(CatalogSnapshot(providers=(self.provider,), models=(self.model,)),
            SourceMetadata(source=self.source, precedence=self.precedence, revision=self.checkpoint["manifest_sha256"]))


def register_security_formula_decoder(*, manager, checkpoint):
    from ipfs_accelerate_py.model_catalog.schema import (
        CapabilityDescriptor, LifecycleState, ModelDescriptor, Modality, Operation,
        OperationalState, ProviderDescriptor,
    )
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder import load_security_formula_decoder
    load_security_formula_decoder(checkpoint)
    probe = _probe(checkpoint)
    capability = CapabilityDescriptor(operations=(Operation.SECURITY_ADVISE,),
        input_modalities=(Modality.JSON,), output_modalities=(Modality.JSON,), media_types=("application/json",))
    state = OperationalState(known=True, configured=True, authorized=True, reachable=True, healthy=True, routable=True)
    provider = ProviderDescriptor(name="local-security-formula-decoder", capabilities=(capability,),
        lifecycle=LifecycleState.READY, state=state,
        description="Bounded learned source productions; independent checking required for formal candidates.")
    model = ModelDescriptor(provider_id=provider.provider_id,
        name="security-formula-" + checkpoint["manifest_sha256"], capabilities=(capability,),
        architecture="source-and-inherited-lexical-production-head", lifecycle=LifecycleState.READY, state=state,
        labels=tuple(sorted({"package": checkpoint["output"],
            "manifest-sha256": checkpoint["manifest_sha256"], "weights-sha256": checkpoint["weights_sha256"],
            "inference-code": INFERENCE_CODE, "authority": "independently_checked_candidate_only"}.items())))
    source = _FormulaSource(model, provider, checkpoint, "security.formula." + checkpoint["manifest_sha256"])
    manager.catalog.register_source(source.source, source, load=True, strict=True, side_effecting=False)
    if manager.get_model_descriptor(model.model_id).record != model:
        raise ValueError("ModelManager formula checkpoint binding differs")
    return {"schema": "security-formula-model-registration@1", "model_id": model.model_id,
        "provider_id": provider.provider_id, "checkpoint": checkpoint, "inference_code": INFERENCE_CODE,
        "inference_probe": probe, "operation": Operation.SECURITY_ADVISE.value,
        "text_generation": False, "proof_authority": False}


def decode_registered_security_formula(*, manager, model_id, source_bytes, source_path):
    from ipfs_accelerate_py.model_catalog.schema import LifecycleState, Operation
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder import (
        decode_security_formula, SCHEMA, _AUTHORITY,
    )
    snapshot = manager.catalog.snapshot()
    model = manager.get_model_descriptor(model_id, snapshot=snapshot).record
    if model is None or {op for capability in model.capabilities for op in capability.operations} != {Operation.SECURITY_ADVISE}:
        raise ValueError("registered security formula model required")
    provider = manager.get_service(model.provider_id, snapshot=snapshot).record
    if any(record is None or record.lifecycle is not LifecycleState.READY
           or any(value is not True for value in record.state.to_dict().values()) for record in (model, provider)):
        raise ValueError("formula model and provider must both be operational")
    labels = dict(model.labels)
    if labels.get("inference-code") != INFERENCE_CODE or labels.get("authority") != "independently_checked_candidate_only":
        raise ValueError("registered formula adapter differs")
    checkpoint = {"schema": SCHEMA, "output": labels["package"],
        "manifest_sha256": labels["manifest-sha256"], "weights_sha256": labels["weights-sha256"],
        "mode": "frozen_production_inference", "authority": "independently_checked_candidate_only", **_AUTHORITY}
    return decode_security_formula(checkpoint=checkpoint, source_bytes=source_bytes, source_path=source_path)
