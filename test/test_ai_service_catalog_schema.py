"""Contract tests for the v1 AI service catalog schemas and identities."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.model_catalog import (
    CanonicalizationError,
    CapabilityDescriptor,
    CatalogSnapshot,
    DeploymentDescriptor,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
    SCHEMA_VERSION,
    SchemaValidationError,
    canonical_json,
    canonical_json_bytes,
    content_cid,
    redact_secrets,
    stable_id,
)
from ipfs_accelerate_py.model_catalog.identity import MAX_CANONICAL_BYTES
from ipfs_accelerate_py.model_catalog.schema import (
    MAX_ALIASES,
    MAX_DESCRIPTION_LENGTH,
    MAX_SNAPSHOT_RECORDS,
)


def text_capability(*extra):
    return CapabilityDescriptor(
        operations=(Operation.TEXT_CHAT,) + tuple(extra),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.TEXT,),
        max_context_tokens=8192,
    )


def records():
    capability = text_capability(Operation.STREAM, Operation.BATCH, Operation.TOOL_CALL)
    provider = ProviderDescriptor(
        name="Example-AI",
        display_name="Example AI",
        aliases=("example", "ex-ai"),
        description="A provider descriptor",
        website_uri="HTTPS://EXAMPLE.COM:443/",
        documentation_uri="https://docs.example.com/catalog",
        capabilities=(capability,),
        lifecycle=LifecycleState.READY,
        state=OperationalState(
            known=True,
            configured=False,
            authorized=None,
            reachable=True,
            healthy=False,
            routable=True,
        ),
        provenance=(
            Provenance(
                source="router.static",
                source_record_id="example-ai",
                observed_at="2026-07-26T02:00:00+02:00",
                expires_at="2026-07-27T00:00:00Z",
            ),
        ),
        labels={"locality": "remote", "owner": "router"},
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="Example/Chat-8B",
        aliases=("chat-8b",),
        architecture="transformer",
        capabilities=(capability,),
        lifecycle="ready",
        state=OperationalState(known=True),
    )
    deployment = DeploymentDescriptor(
        provider_id=provider.provider_id,
        model_id=model.model_id,
        name="Production",
        endpoint_uri="HTTPS://API.EXAMPLE.COM:443/v1",
        capabilities=(capability,),
        lifecycle=LifecycleState.READY,
        state=OperationalState(configured=True, authorized=True, reachable=True),
        created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        updated_at="2026-07-26T00:00:00Z",
    )
    binding = RouterBinding(
        router="llm_router",
        provider_id=provider.provider_id,
        model_id=model.model_id,
        deployment_id=deployment.deployment_id,
        operations=(Operation.TEXT_CHAT, Operation.STREAM, Operation.BATCH, Operation.TOOL_CALL),
        priority=10,
        state=OperationalState(routable=True),
    )
    return capability, provider, model, deployment, binding


def test_operation_taxonomy_is_exact_and_versioned():
    assert SCHEMA_VERSION == "1.0"
    assert {operation.value for operation in Operation} == {
        "text.generate",
        "text.chat",
        "embedding.generate",
        "vision.generate",
        "audio.transcribe",
        "audio.synthesize",
        "batch",
        "stream",
        "tool.call",
    }
    with pytest.raises(ValueError, match="unknown operations"):
        CapabilityDescriptor(operations=("text.complete",))


def test_operational_facts_are_orthogonal_and_tri_state():
    state = OperationalState(
        known=False,
        configured=True,
        authorized=None,
        reachable=True,
        healthy=False,
        routable=True,
    )
    assert state.to_dict() == {
        "known": False,
        "configured": True,
        "authorized": None,
        "reachable": True,
        "healthy": False,
        "routable": True,
    }
    assert OperationalState.from_dict(state.to_dict()) == state
    with pytest.raises(SchemaValidationError, match="boolean or null"):
        OperationalState(known=1)


def test_all_records_are_frozen_normalized_and_round_trip():
    capability, provider, model, deployment, binding = records()
    snapshot = CatalogSnapshot(
        providers=(provider,),
        models=(model,),
        deployments=(deployment,),
        bindings=(binding,),
        created_at="2026-07-26T00:00:00Z",
    )
    for record in (capability, provider, model, deployment, binding, snapshot):
        assert type(record).from_dict(record.to_dict()) == record
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.schema_version = "changed"

    assert provider.name == "example-ai"
    assert provider.aliases == ("ex-ai", "example")
    assert provider.website_uri == "https://example.com/"
    assert deployment.endpoint_uri == "https://api.example.com/v1"
    assert provider.provenance[0].observed_at == "2026-07-26T00:00:00.000000Z"
    assert snapshot.cid == snapshot.revision


def test_canonical_serialization_is_deterministic_bounded_and_order_aware():
    left = {
        "operations": {Operation.STREAM, Operation.TEXT_CHAT},
        "nested": {"b": 2, "a": 1},
    }
    right = {
        "nested": {"a": 1, "b": 2},
        "operations": {Operation.TEXT_CHAT, Operation.STREAM},
    }
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert content_cid(left) == content_cid(right)
    assert canonical_json({"ordered": ["a", "b"]}) != canonical_json({"ordered": ["b", "a"]})
    assert content_cid({"x": 1}).startswith("bafkrei")
    with pytest.raises(CanonicalizationError, match="byte bound"):
        canonical_json({"large": ["x" * 65_536] * 17})
    with pytest.raises(CanonicalizationError, match="string.*size bound"):
        canonical_json({"large": "x" * MAX_CANONICAL_BYTES})
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_json({"value": float("nan")})
    with pytest.raises(CanonicalizationError, match="64-bit"):
        canonical_json({"value": 1 << 64})


def test_stable_ids_use_framed_collision_resistant_inputs():
    assert stable_id("model", "ab", "c") != stable_id("model", "a", "bc")
    assert stable_id("model", {"a": 1, "b": 2}) == stable_id("model", {"b": 2, "a": 1})
    assert stable_id("model", ["a", "b"]) != stable_id("model", ["b", "a"])
    with pytest.raises(ValueError, match="identity kind"):
        stable_id("../model", "unsafe")


def test_descriptor_identities_ignore_presentation_and_reject_spoofing():
    _, provider, model, deployment, binding = records()
    renamed_display = dataclasses.replace(provider, display_name="A new display name")
    changed_state = dataclasses.replace(
        provider, state=OperationalState(healthy=True, routable=False)
    )
    assert renamed_display.provider_id == provider.provider_id
    assert changed_state.provider_id == provider.provider_id
    assert renamed_display.cid != provider.cid
    assert changed_state.cid != provider.cid

    for record, field_name in (
        (provider, "provider_id"),
        (model, "model_id"),
        (deployment, "deployment_id"),
        (binding, "binding_id"),
    ):
        value = record.to_dict()
        value[field_name] = stable_id(field_name.split("_")[0], "spoofed")
        with pytest.raises(SchemaValidationError, match="canonical identity"):
            type(record).from_dict(value)


@pytest.mark.parametrize(
    "bad_alias",
    ["", " leading", "../escape", "contains space", "A" * 129, "a//b"],
)
def test_alias_validation_rejects_malformed_values(bad_alias):
    with pytest.raises(SchemaValidationError):
        ProviderDescriptor(name="provider", aliases=(bad_alias,))


def test_alias_validation_is_unique_canonical_and_bounded():
    with pytest.raises(SchemaValidationError, match="canonical name"):
        ProviderDescriptor(name="provider", aliases=("provider",))
    provider = ProviderDescriptor(name="provider", aliases=("B", "a", "b"))
    assert provider.aliases == ("a", "b")
    with pytest.raises(SchemaValidationError, match="maximum"):
        ProviderDescriptor(
            name="provider",
            aliases=tuple("alias-%d" % index for index in range(MAX_ALIASES + 1)),
        )


@pytest.mark.parametrize(
    "uri",
    [
        "ftp://example.com/model",
        "https://user:password@example.com/v1",
        "https://example.com/v1?api_key=secret",
        "https://example.com/v1#token",
        "https:///missing-host",
        "not a uri",
    ],
)
def test_uri_validation_fails_closed(uri):
    provider = ProviderDescriptor(name="provider")
    with pytest.raises(SchemaValidationError):
        DeploymentDescriptor(
            provider_id=provider.provider_id,
            name="deployment",
            endpoint_uri=uri,
        )


def test_timestamp_validation_and_ordering():
    provider = ProviderDescriptor(name="provider")
    with pytest.raises(SchemaValidationError, match="RFC 3339"):
        DeploymentDescriptor(
            provider_id=provider.provider_id,
            name="deployment",
            endpoint_uri="https://example.com/v1",
            created_at="2026-01-01T00:00:00",
        )
    with pytest.raises(SchemaValidationError, match="RFC 3339"):
        DeploymentDescriptor(
            provider_id=provider.provider_id,
            name="deployment",
            endpoint_uri="https://example.com/v1",
            created_at="2026-01-01 00:00:00Z",
        )
    with pytest.raises(SchemaValidationError, match="precede"):
        DeploymentDescriptor(
            provider_id=provider.provider_id,
            name="deployment",
            endpoint_uri="https://example.com/v1",
            created_at="2026-01-02T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
    with pytest.raises(SchemaValidationError, match="later"):
        Provenance(
            source="test",
            observed_at="2026-01-02T00:00:00Z",
            expires_at="2026-01-01T00:00:00Z",
        )


def test_size_and_capability_combination_bounds():
    with pytest.raises(SchemaValidationError, match="batch and stream"):
        CapabilityDescriptor(operations=(Operation.STREAM, Operation.BATCH))
    with pytest.raises(SchemaValidationError, match="embedding_dimensions"):
        CapabilityDescriptor(operations=(Operation.TEXT_GENERATE,), embedding_dimensions=1536)
    with pytest.raises(SchemaValidationError, match="max_batch_size"):
        CapabilityDescriptor(operations=(Operation.TEXT_GENERATE,), max_batch_size=4)
    with pytest.raises(SchemaValidationError, match="between"):
        CapabilityDescriptor(operations=(Operation.TEXT_GENERATE,), max_input_bytes=0)
    with pytest.raises(SchemaValidationError, match="description"):
        ProviderDescriptor(name="provider", description="x" * (MAX_DESCRIPTION_LENGTH + 1))


@pytest.mark.parametrize(
    "cls,minimal",
    [
        (
            CapabilityDescriptor,
            {"schema_version": SCHEMA_VERSION, "operations": ["text.chat"]},
        ),
        (Provenance, {"schema_version": SCHEMA_VERSION, "source": "test"}),
        (
            ProviderDescriptor,
            {"schema_version": SCHEMA_VERSION, "name": "provider"},
        ),
    ],
)
def test_schema_versions_and_unknown_fields_are_strict(cls, minimal):
    with pytest.raises(SchemaValidationError, match="unsupported schema_version"):
        cls.from_dict(dict(minimal, schema_version="2.0"))
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        cls.from_dict(dict(minimal, future_field=True))
    without_version = dict(minimal)
    without_version.pop("schema_version")
    with pytest.raises(SchemaValidationError, match="missing required"):
        cls.from_dict(without_version)


def test_unknown_enums_and_malformed_records_are_rejected():
    provider = ProviderDescriptor(name="provider")
    with pytest.raises(SchemaValidationError, match="unknown lifecycle"):
        ProviderDescriptor(name="provider", lifecycle="running")
    with pytest.raises(SchemaValidationError, match="must be an object"):
        ProviderDescriptor.from_dict(["not", "an", "object"])
    with pytest.raises(SchemaValidationError, match="missing required"):
        ModelDescriptor.from_dict(
            {"schema_version": SCHEMA_VERSION, "provider_id": provider.provider_id}
        )
    with pytest.raises(SchemaValidationError, match="must not be null"):
        DeploymentDescriptor(
            provider_id=provider.provider_id,
            name="deployment",
            endpoint_uri=None,
        )
    with pytest.raises(SchemaValidationError, match="requires model_id"):
        RouterBinding(
            router="llm",
            provider_id=provider.provider_id,
            operations=(Operation.TEXT_CHAT,),
        )


def test_credentials_are_rejected_or_explicitly_redacted():
    secret_record = {
        "provider": "example",
        "api_key": "sk-0123456789abcdef0123456789abcdef",
        "nested": {
            "authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
            "safe": "visible",
        },
    }
    with pytest.raises(CanonicalizationError, match="credential-bearing"):
        canonical_json(secret_record)
    assert redact_secrets(secret_record) == {
        "provider": "example",
        "api_key": "[REDACTED]",
        "nested": {"authorization": "[REDACTED]", "safe": "visible"},
    }
    assert "[REDACTED]" in canonical_json(redact_secrets(secret_record))
    with pytest.raises(CanonicalizationError, match="credential-shaped"):
        canonical_json({"value": "sk-0123456789abcdef0123456789abcdef"})
    assert "secret" not in canonical_json(redact_secrets({"password": "secret"}))
    with pytest.raises(SchemaValidationError, match="credential-bearing label"):
        ProviderDescriptor(name="provider", labels={"api_key": "not-even-a-key"})
    with pytest.raises(SchemaValidationError, match="credential-shaped"):
        ProviderDescriptor(name="provider", labels={"note": "sk-0123456789abcdef0123456789abcdef"})


def test_snapshot_order_is_identity_independent_and_content_sensitive():
    capability, provider, model, deployment, binding = records()
    provider_two = ProviderDescriptor(name="another", capabilities=(capability,))
    left = CatalogSnapshot(providers=(provider, provider_two), models=(model,))
    right = CatalogSnapshot(providers=(provider_two, provider), models=(model,))
    assert left.revision == right.revision
    assert left.to_dict() == right.to_dict()

    changed = CatalogSnapshot(
        providers=(dataclasses.replace(provider, description="changed"), provider_two),
        models=(model,),
    )
    assert changed.revision != left.revision
    # Collection time is intentionally snapshot metadata, not catalog content.
    later = dataclasses.replace(left, created_at="2030-01-01T00:00:00Z")
    assert later.revision == left.revision


def test_snapshot_rejects_duplicate_and_excessive_records():
    provider = ProviderDescriptor(name="provider")
    with pytest.raises(SchemaValidationError, match="duplicate"):
        CatalogSnapshot(providers=(provider, provider))
    with pytest.raises(SchemaValidationError, match="maximum record"):
        # The bound is checked before individual values are parsed.
        CatalogSnapshot(providers=(None,) * (MAX_SNAPSHOT_RECORDS + 1))


def test_cold_import_does_not_start_provider_process_network_install_or_model_load(tmp_path):
    """An isolated interpreter makes import-time side effects observable."""

    script = r"""
import json
import os
import socket
import subprocess
import urllib.request

events = []
def blocked(kind):
    def call(*args, **kwargs):
        events.append(kind)
        raise AssertionError("forbidden import side effect: " + kind)
    return call

subprocess.Popen = blocked("process")
subprocess.run = blocked("process")
subprocess.call = blocked("process")
subprocess.check_call = blocked("install")
subprocess.check_output = blocked("process")
os.system = blocked("process")
socket.create_connection = blocked("network")
urllib.request.urlopen = blocked("network")
os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"

import ipfs_accelerate_py.model_catalog as catalog

assert catalog.ProviderDescriptor(name="offline").name == "offline"
for forbidden in (
    "ipfs_accelerate_py.llm_router",
    "ipfs_accelerate_py.embeddings_router",
    "ipfs_accelerate_py.multimodal_router",
    "ipfs_accelerate_py.voice_router",
    "torch",
    "transformers",
):
    assert forbidden not in __import__("sys").modules, forbidden
print(json.dumps(events))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
