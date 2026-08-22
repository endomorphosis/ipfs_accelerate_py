"""Deterministic tests for EAAEF-031 source-disclosure policy.

Fixtures live under pytest tmp_path.  No network is used.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.authority.external_principal import (
    AutonomyCeiling,
    EffectName,
    ExternalPrincipal,
    ResourceCeilings,
)
from ipfs_accelerate_py.agent_supervisor.authority.source_disclosure import (
    CONTRACT_VERSION,
    DEFAULT_MAX_BYTES,
    DISCLOSURE_DECISION_INTERFACE,
    SCHEMA_VERSION,
    SOURCE_DISCLOSURE_POLICY_INTERFACE,
    ConfidentialityClass,
    ContextPackIdentityError,
    DisclosureDecision,
    DisclosurePolicyError,
    DisclosureVerdict,
    PolicyMissingError,
    ProviderLocality,
    SecretKind,
    SecretMaterialError,
    SourceDisclosurePolicy,
    admit_disclosure,
    bind_context_pack_identity,
    classify_provider_locality,
    evaluate_disclosure,
    load_policy,
    scan_secret_material,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

DID = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
NOW_MS = 1_700_000_000_000
CANARY_API_VALUE = "test" + "value1234"
CANARY_BEARER = "aaaa" + "bbbbccccdddd"
CANARY_PEM_BODY = "MIIB" + "ogIBAAJBAK8="


def _policy(**changes: object) -> SourceDisclosurePolicy:
    values: dict[str, object] = {
        "policy_id": "policy:disclosure@1",
        "confidentiality": ConfidentialityClass.INTERNAL,
        "exclusions": ("secrets/", ".env"),
        "allowed_providers": ("local:ollama", "hermetic:echo"),
        "local_only": True,
        "max_bytes": 4096,
        "require_secret_scan": True,
    }
    values.update(changes)
    return SourceDisclosurePolicy(**values)  # type: ignore[arg-type]


def _pack(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "interface": "ContextPack@1",
        "objective": "implement-EAAEF-031",
        "target_source_cid": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
        "exclusions": ["secrets/"],
        "route": "local",
    }
    values.update(changes)
    return values


def _pack_id(pack: dict[str, object] | None = None) -> str:
    payload = pack if pack is not None else _pack()
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"content_id", "cid", "identity", "canonical_id"}
    }
    return content_identity(body)


def _principal(**changes: object) -> ExternalPrincipal:
    values: dict[str, object] = {
        "principal_id": DID,
        "repository_id": "repo:example",
        "run_id": "run:example-1",
        "exact_effects": (EffectName.INSPECT_REPOSITORY, EffectName.RUN_VALIDATION),
        "expires_at_ms": NOW_MS + 60_000,
        "autonomy_ceiling": AutonomyCeiling.SUPERVISED,
        "resource_ceilings": ResourceCeilings(cpu=4000, ram=8192, disk=16384, timeout=7200000),
        "disclosure_policy_id": "policy:disclosure@1",
        "provider_policy_id": "policy:provider@1",
        "nonce": "nonce-bound-001",
    }
    values.update(changes)
    return ExternalPrincipal(**values)  # type: ignore[arg-type]


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def test_policy_is_frozen_source_disclosure_policy_v1() -> None:
    policy = _policy()
    assert SOURCE_DISCLOSURE_POLICY_INTERFACE == "SourceDisclosurePolicy@1"
    assert DISCLOSURE_DECISION_INTERFACE == "DisclosureDecision@1"
    assert CONTRACT_VERSION == 1
    assert SCHEMA_VERSION == 1
    assert policy.schema.endswith("@1")
    assert policy.INTERFACE == "SourceDisclosurePolicy@1"
    assert policy.confidentiality is ConfidentialityClass.INTERNAL
    assert policy.local_only is True
    assert policy.require_secret_scan is True
    assert policy.max_bytes == 4096
    assert policy.content_id
    assert policy.content_id == policy.cid
    with pytest.raises(FrozenInstanceError):
        policy.local_only = False  # type: ignore[misc]
    round_trip = SourceDisclosurePolicy.from_dict(policy.to_dict())
    assert round_trip.content_id == policy.content_id
    assert DEFAULT_MAX_BYTES == 65_536


def test_tmp_fixture_roundtrip_and_permit(tmp_path: Path) -> None:
    policy = _policy()
    policy_path = tmp_path / "source-disclosure-policy.json"
    policy_path.write_text(policy.to_json(), encoding="utf-8")
    loaded = load_policy(policy_path)
    assert loaded.content_id == policy.content_id

    pack = _pack()
    pack_path = _write_json(tmp_path / "context-pack.json", pack)
    pack_payload = json.loads(pack_path.read_text(encoding="utf-8"))
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")

    decision = admit_disclosure(
        policy_path,
        payload=source,
        provider_id="local:ollama",
        context_pack=pack_payload,
    )
    assert decision.verdict is DisclosureVerdict.PERMIT
    assert decision.permitted is True
    assert decision.reason_code == "bound"
    assert decision.context_pack_content_id == _pack_id(pack)
    assert decision.policy_content_id == policy.content_id
    assert decision.provider_locality is ProviderLocality.LOCAL
    assert decision.schema.endswith("@1")
    assert DisclosureDecision.from_dict(decision.to_dict()).content_id == decision.content_id


def test_scan_api_key_bearer_and_pem(tmp_path: Path) -> None:
    api_file = tmp_path / "leaked.env"
    api_file.write_text("api_key=" + CANARY_API_VALUE, encoding="utf-8")
    bearer_file = tmp_path / "auth.txt"
    bearer_file.write_bytes(b"Authorization: Bearer " + CANARY_BEARER.encode("ascii"))
    pem_file = tmp_path / "id.pem"
    pem_file.write_text(
        "-----BEGIN PRIVATE KEY-----\n"
        + CANARY_PEM_BODY
        + "\n-----END PRIVATE KEY-----\n",
        encoding="utf-8",
    )

    assert scan_secret_material(api_file) == (SecretKind.API_KEY.value,)
    assert scan_secret_material(bearer_file) == (SecretKind.BEARER.value,)
    assert scan_secret_material(pem_file) == (SecretKind.PRIVATE_KEY_PEM.value,)
    assert scan_secret_material("print('ok')") == ()
    assert scan_secret_material({"api_key": CANARY_API_VALUE}) == (SecretKind.API_KEY.value,)
    assert scan_secret_material(b"bearer " + CANARY_BEARER.encode("ascii")) == (
        SecretKind.BEARER.value,
    )

    pack_id = _pack_id()
    for payload, kind in (
        (api_file, SecretKind.API_KEY.value),
        (bearer_file, SecretKind.BEARER.value),
        (pem_file, SecretKind.PRIVATE_KEY_PEM.value),
    ):
        decision = evaluate_disclosure(
            _policy(),
            payload=payload,
            provider_id="local:ollama",
            context_pack_content_id=pack_id,
        )
        assert decision.verdict is DisclosureVerdict.DENY
        assert decision.reason_code == "secret_material"
        assert kind in decision.secret_kinds
        with pytest.raises(SecretMaterialError, match="secret-shaped") as denied:
            admit_disclosure(
                _policy(),
                payload=payload,
                provider_id="local:ollama",
                context_pack_content_id=pack_id,
            )
        assert denied.value.reason_code == "secret_material"


def test_provider_allowlist_and_local_only(tmp_path: Path) -> None:
    source = tmp_path / "ok.py"
    source.write_text("x = 1\n", encoding="utf-8")
    pack_id = _pack_id()
    policy = _policy()

    assert classify_provider_locality("local:ollama") is ProviderLocality.LOCAL
    assert classify_provider_locality("openai:gpt") is ProviderLocality.EXTERNAL

    missing = evaluate_disclosure(
        policy,
        payload=source,
        provider_id="hermetic:echo",
        context_pack_content_id=pack_id,
    )
    assert missing.permitted is True

    unknown = evaluate_disclosure(
        policy,
        payload=source,
        provider_id="openai:gpt",
        context_pack_content_id=pack_id,
    )
    assert unknown.verdict is DisclosureVerdict.DENY
    assert unknown.reason_code == "provider_not_allowlisted"

    remote_policy = _policy(
        allowed_providers=("openai:gpt", "local:ollama"),
        local_only=True,
        confidentiality=ConfidentialityClass.INTERNAL,
    )
    local_only = evaluate_disclosure(
        remote_policy,
        payload=source,
        provider_id="openai:gpt",
        context_pack_content_id=pack_id,
    )
    assert local_only.reason_code == "local_only"
    with pytest.raises(DisclosurePolicyError, match="local_only") as denied:
        admit_disclosure(
            remote_policy,
            payload=source,
            provider_id="openai:gpt",
            context_pack_content_id=pack_id,
        )
    assert denied.value.reason_code == "local_only"

    with pytest.raises(DisclosurePolicyError, match="local_only"):
        _policy(
            confidentiality=ConfidentialityClass.SECRET,
            local_only=False,
            allowed_providers=("local:ollama",),
        )


def test_byte_limits_and_exclusions(tmp_path: Path) -> None:
    pack_id = _pack_id()
    policy = _policy(max_bytes=32)
    oversized = tmp_path / "big.py"
    oversized.write_bytes(b"x = " + (b"a" * 64))
    decision = evaluate_disclosure(
        policy,
        payload=oversized,
        provider_id="local:ollama",
        context_pack_content_id=pack_id,
    )
    assert decision.reason_code == "byte_limit"
    assert decision.payload_bytes > 32

    secret_dir = tmp_path / "secrets" / "token.txt"
    secret_dir.parent.mkdir()
    secret_dir.write_text("not a credential assignment", encoding="utf-8")
    excluded = evaluate_disclosure(
        _policy(),
        payload=secret_dir,
        provider_id="local:ollama",
        context_pack_content_id=pack_id,
        paths=("secrets/token.txt",),
    )
    assert excluded.reason_code == "excluded"
    assert "secrets/" in excluded.excluded_matches

    env = tmp_path / ".env"
    env.write_text("DEBUG=1\n", encoding="utf-8")
    env_decision = evaluate_disclosure(
        _policy(),
        payload=env,
        provider_id="local:ollama",
        context_pack_content_id=pack_id,
    )
    assert env_decision.reason_code == "excluded"

    with pytest.raises(DisclosurePolicyError, match="excluded") as denied:
        admit_disclosure(
            _policy(),
            payload={"secrets/key.txt": "plain"},
            provider_id="local:ollama",
            context_pack_content_id=pack_id,
        )
    assert denied.value.reason_code == "excluded"


def test_confidentiality_ceiling(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("public notes\n", encoding="utf-8")
    pack_id = _pack_id()
    policy = _policy(confidentiality=ConfidentialityClass.INTERNAL)
    permitted = evaluate_disclosure(
        policy,
        payload=source,
        provider_id="local:ollama",
        context_pack_content_id=pack_id,
        confidentiality=ConfidentialityClass.PUBLIC,
    )
    assert permitted.permitted is True
    denied = evaluate_disclosure(
        policy,
        payload=source,
        provider_id="local:ollama",
        context_pack_content_id=pack_id,
        confidentiality=ConfidentialityClass.SECRET,
    )
    assert denied.reason_code == "confidentiality"


def test_context_pack_identity_binding(tmp_path: Path) -> None:
    pack = _pack()
    actual = _pack_id(pack)
    source = tmp_path / "ok.py"
    source.write_text("ok\n", encoding="utf-8")

    bound = bind_context_pack_identity(pack, actual)
    assert bound == actual

    with pytest.raises(ContextPackIdentityError) as missing:
        bind_context_pack_identity()
    assert missing.value.reason_code == "identity_mismatch"

    mismatched = {**pack, "content_id": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"}
    with pytest.raises(ContextPackIdentityError, match="does not match") as mismatch:
        admit_disclosure(
            _policy(),
            payload=source,
            provider_id="local:ollama",
            context_pack=mismatched,
        )
    assert mismatch.value.reason_code == "identity_mismatch"

    with pytest.raises(ContextPackIdentityError):
        admit_disclosure(
            _policy(),
            payload=source,
            provider_id="local:ollama",
            context_pack=pack,
            context_pack_content_id="bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
        )

    decision = admit_disclosure(
        _policy(),
        payload=source,
        provider_id="local:ollama",
        context_pack=pack,
        context_pack_content_id=actual,
    )
    assert decision.context_pack_content_id == actual


def test_policy_miss_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "ok.py"
    source.write_text("ok\n", encoding="utf-8")
    pack_id = _pack_id()
    with pytest.raises(PolicyMissingError, match="required") as missing:
        evaluate_disclosure(
            None,
            payload=source,
            provider_id="local:ollama",
            context_pack_content_id=pack_id,
        )
    assert missing.value.reason_code == "policy_missing"

    with pytest.raises(PolicyMissingError) as absent:
        load_policy(tmp_path / "missing-policy.json")
    assert absent.value.reason_code == "policy_missing"

    principal = _principal(disclosure_policy_id="policy:other@1")
    with pytest.raises(DisclosurePolicyError, match="disclosure_policy_id") as mismatched:
        evaluate_disclosure(
            _policy(),
            payload=source,
            provider_id="local:ollama",
            context_pack_content_id=pack_id,
            principal=principal,
        )
    assert mismatched.value.reason_code == "policy_mismatch"

    bound = admit_disclosure(
        _policy(),
        payload=source,
        provider_id="local:ollama",
        context_pack_content_id=pack_id,
        principal=_principal(),
    )
    assert bound.principal_content_id == _principal().content_id
    assert bound.permitted is True


def test_secret_scan_cannot_be_disabled_and_unknown_fields_fail() -> None:
    with pytest.raises(DisclosurePolicyError, match="require_secret_scan"):
        _policy(require_secret_scan=False)
    with pytest.raises(DisclosurePolicyError, match="unsupported fields"):
        SourceDisclosurePolicy.from_dict({**_policy().to_dict(), "ambient_trust": True})
    with pytest.raises(DisclosurePolicyError, match="unsupported") as version:
        SourceDisclosurePolicy.from_dict(
            {**_policy().to_dict(), "interface": "SourceDisclosurePolicy@2"}
        )
    assert version.value.reason_code == "unsupported_version"


def test_context_pack_exclusions_union(tmp_path: Path) -> None:
    source = tmp_path / "private" / "notes.py"
    source.parent.mkdir()
    source.write_text("notes = 1\n", encoding="utf-8")
    pack = _pack(exclusions=["private/"])
    decision = evaluate_disclosure(
        _policy(exclusions=(".env",)),
        payload=source,
        provider_id="local:ollama",
        context_pack=pack,
        paths=("private/notes.py",),
    )
    assert decision.reason_code == "excluded"
    assert "private/" in decision.excluded_matches
