"""ModelManager consumer of the datasets-owned security checkpoint API."""
from dataclasses import dataclass
from pathlib import Path
from ipfs_datasets_py.logic.formalization.autoencoder.security import security_autoencoder_hub as distribution
from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_hub import (
    HUB_SCHEMA, HUB_FIELDS, MANIFEST, SecurityCheckpointPublication,
    validate_hub_descriptor, download_security_checkpoint,
    plan_security_checkpoint_publication, publish_security_checkpoint,
    _loaded, _probe, _directory,
)
INFERENCE_CODE = "ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_hub:score_registered_security_observations"

@dataclass(frozen=True)
class _RegisteredSecuritySource:
    source: str
    model: object
    provider: object
    manifest_sha256: str
    precedence: int = 100
    side_effecting: bool = False

    def load(self):
        from ipfs_accelerate_py.model_catalog.schema import CatalogSnapshot
        from ipfs_accelerate_py.model_catalog.sources.static import CatalogSourceResult, SourceMetadata
        labels = dict(self.model.labels)
        _probe(_loaded(labels["package"], self.manifest_sha256))
        return CatalogSourceResult(CatalogSnapshot(providers=(self.provider,), models=(self.model,)),
            SourceMetadata(source=self.source, precedence=self.precedence, revision=self.manifest_sha256))


def register_security_checkpoint(*, manager, package, expected_manifest_sha256, hub_descriptor=None):
    """Register only the executed JSON security-advice capability in ModelManager."""
    from ipfs_accelerate_py.model_catalog.schema import (
        CapabilityDescriptor, LifecycleState, ModelDescriptor, Modality, Operation,
        OperationalState, ProviderDescriptor,
    )
    hub = validate_hub_descriptor(hub_descriptor) if hub_descriptor is not None else None
    if hub is not None and hub["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("Hub and local checkpoint manifest differ")
    loaded = _loaded(_directory(package), expected_manifest_sha256)
    probe = _probe(loaded)
    capability = CapabilityDescriptor(operations=(Operation.SECURITY_ADVISE,),
        input_modalities=(Modality.JSON,), output_modalities=(Modality.JSON,),
        media_types=("application/json",))
    state = OperationalState(known=True, configured=True, authorized=True, reachable=True, healthy=True, routable=True)
    provider = ProviderDescriptor(name="local-security-autoencoder", capabilities=(capability,),
        lifecycle=LifecycleState.READY, state=state,
        description="Installed source-free security candidate inference; no text generation or proof authority.")
    model = ModelDescriptor(provider_id=provider.provider_id,
        name="security-autoencoder-" + expected_manifest_sha256, capabilities=(capability,),
        architecture=loaded["config"]["architecture"], lifecycle=LifecycleState.READY, state=state,
        labels=tuple(sorted({"package": str(Path(package).absolute()), "manifest-sha256": expected_manifest_sha256,
            **({"hub-repository": hub["repository_id"], "hub-revision": hub["revision"],
                "hub-release-prefix": hub["release_prefix"]} if hub is not None else {}), "inference-code": INFERENCE_CODE,
            "authority": "unverified_candidate_only", "inference-mode": "frozen_inference"}.items())))
    source = _RegisteredSecuritySource("security.autoencoder." + expected_manifest_sha256,
        model, provider, expected_manifest_sha256)
    manager.catalog.register_source(source.source, source, load=True, strict=True, side_effecting=False)
    selected = manager.get_model_descriptor(model.model_id)
    if selected.record != model:
        raise ValueError("ModelManager did not retain the exact security capability")
    return {"schema": "security-autoencoder-model-registration@1", "model_id": model.model_id,
        "provider_id": provider.provider_id, "catalog_revision": manager.catalog_revision,
        "hub": hub, "checkpoint": loaded["descriptor"], "inference_probe": probe,
        "operation": Operation.SECURITY_ADVISE.value, "inference_code": INFERENCE_CODE,
        "text_generation": False, "general_text_embedding": False, "proof_authority": False}


def score_registered_security_observations(*, manager, model_id, observations):
    """Resolve the registered installed adapter and recheck bytes before inference."""
    from ipfs_accelerate_py.model_catalog.schema import LifecycleState, Operation
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_checkpoint import score_security_observations
    snapshot = manager.catalog.snapshot()
    found = manager.get_model_descriptor(model_id, snapshot=snapshot)
    if found.record is None or {op for c in found.record.capabilities for op in c.operations} != {Operation.SECURITY_ADVISE}:
        raise ValueError("registered security-only model required")
    provider = manager.get_service(found.record.provider_id, snapshot=snapshot).record
    # A model-level positive observation cannot override an explicitly disabled
    # provider. Both records must retain the readiness established by probing;
    # unknown operational facts are not permission to run this installed adapter.
    if any(record is None or record.lifecycle is not LifecycleState.READY
           or any(value is not True for value in record.state.to_dict().values())
           for record in (found.record, provider)):
        raise ValueError("registered security model and provider must be ready and operational")
    labels = dict(found.record.labels)
    if labels.get("inference-code") != INFERENCE_CODE or labels.get("authority") != "unverified_candidate_only":
        raise ValueError("registered local security adapter differs")
    loaded = _loaded(_directory(labels["package"]), labels["manifest-sha256"])
    return score_security_observations(checkpoint=loaded["descriptor"], observations=observations)
