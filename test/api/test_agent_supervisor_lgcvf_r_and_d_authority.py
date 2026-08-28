from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from jsonschema import Draft202012Validator, ValidationError

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    lgcvf_r_and_d_authority as authority,
)
from ipfs_accelerate_py.agent_supervisor.validation.lgcvf_r_and_d_authority import (
    LGCVF_BASE64URL_ENCODING,
    LGCVF_ED25519_ALGORITHM,
    LGCVF_EXTERNAL_R_AND_D_DISPOSITION,
    LGCVF_EXTERNAL_RECEIPT_SCHEMA_V2,
    LGCVF_PRODUCTION_DECLINED_DISPOSITION,
    LGCVF_PRODUCTION_RECEIPT_SCHEMA_V2,
    LGCVF_R_AND_D_AUTHORITY_SCOPE,
    LGCVF_R_AND_D_SIGNATURE_DOMAIN,
    LGCVF_R_AND_D_TRUST_MODEL,
    LgcvfAuthorityBindings,
    LgcvfRAndDAuthorityError,
    LgcvfRAndDTrustPolicy,
    LgcvfSourceRevisions,
    ed25519_public_key_id,
    load_lgcvf_r_and_d_trust_policy,
    validate_lgcvf_external_r_and_d_receipt,
    validate_lgcvf_production_declined_r_and_d_receipt,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "docs/architecture/lgcvf_external_qualification_receipt.v2.schema.json"
)
PRODUCTION_SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "docs/architecture/lgcvf_production_authorization_receipt.v2.schema.json"
)
NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)

# Fixed test material only.  Production code intentionally has no private-key API.
PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
PUBLIC_KEY = PRIVATE_KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _cid(label: str) -> str:
    return content_identity({"fixture": label})


SOURCES = LgcvfSourceRevisions(
    accelerator_head="1" * 40,
    accelerator_tree="2" * 40,
    datasets_head="3" * 40,
    datasets_tree="4" * 40,
    datasets_gitlink="5" * 40,
)
BINDINGS = LgcvfAuthorityBindings(
    plan_cid=_cid("plan"),
    qualification_result_cid=_cid("qualification-result"),
    qualification_checkout_fingerprint_cid=_cid("checkout-fingerprint"),
    benchmark_report_cid=_cid("benchmark"),
    release_report_sha256="sha256:" + "a" * 64,
    source_revisions=SOURCES,
)
TRUST = LgcvfRAndDTrustPolicy(
    identity="Sole User Test Operator",
    role="sole R&D verifier and operator",
    key_id=ed25519_public_key_id(PUBLIC_KEY),
    public_key=PUBLIC_KEY,
)


def _signer() -> dict[str, str]:
    return {
        "identity": TRUST.identity,
        "role": TRUST.role,
        "key_id": TRUST.key_id,
        "public_key_base64url": TRUST.public_key_base64url,
    }


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    receipt = copy.deepcopy(payload)
    receipt["payload_cid"] = content_identity(payload)
    receipt["signature"] = {
        "algorithm": LGCVF_ED25519_ALGORITHM,
        "encoding": LGCVF_BASE64URL_ENCODING,
        "value": _b64url(
            PRIVATE_KEY.sign(
                LGCVF_R_AND_D_SIGNATURE_DOMAIN + canonical_json_bytes(payload)
            )
        ),
    }
    receipt["receipt_cid"] = content_identity(receipt)
    return receipt


def _payload(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key not in {"payload_cid", "signature", "receipt_cid"}
    }


def _external_receipt() -> dict[str, Any]:
    return _seal(
        {
            "schema": LGCVF_EXTERNAL_RECEIPT_SCHEMA_V2,
            "receipt_kind": "external_qualification_r_and_d",
            "trust_model": LGCVF_R_AND_D_TRUST_MODEL,
            "authority_scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
            "issuer": _signer(),
            "third_party_independence_claimed": False,
            "issued_at": "2026-08-28T10:00:00Z",
            "expires_at": "2026-08-28T14:00:00Z",
            "plan_cid": BINDINGS.plan_cid,
            "qualification_result_cid": BINDINGS.qualification_result_cid,
            "qualification_checkout_fingerprint_cid": (
                BINDINGS.qualification_checkout_fingerprint_cid
            ),
            "benchmark_report_cid": BINDINGS.benchmark_report_cid,
            "source_revisions": SOURCES.to_dict(),
            "cohorts": {
                "live_local_model_execution": "passed",
                "live_remote_model_execution": "unavailable",
                "production_authoritative_evidence": "unavailable",
            },
            "provider_disclosure_policy": "providers disclosed in retained evidence",
            "multi_writer": {
                "quack_qualified": False,
                "disposition": "unavailable",
                "notes": "not qualified by sole-user R&D evidence",
            },
            "disposition": LGCVF_EXTERNAL_R_AND_D_DISPOSITION,
            "release_qualified": False,
            "production_authorized": False,
            "limitations": [
                "self-signed single-user R&D evidence; no third-party independence"
            ],
        }
    )


def _production_receipt(external: dict[str, Any]) -> dict[str, Any]:
    return _seal(
        {
            "schema": LGCVF_PRODUCTION_RECEIPT_SCHEMA_V2,
            "receipt_kind": "production_authorization_r_and_d",
            "trust_model": LGCVF_R_AND_D_TRUST_MODEL,
            "authority_scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
            "operator": _signer(),
            "issued_at": "2026-08-28T11:00:00Z",
            "expires_at": "2026-08-28T13:00:00Z",
            "plan_cid": BINDINGS.plan_cid,
            "qualification_result_cid": BINDINGS.qualification_result_cid,
            "qualification_checkout_fingerprint_cid": (
                BINDINGS.qualification_checkout_fingerprint_cid
            ),
            "benchmark_report_cid": BINDINGS.benchmark_report_cid,
            "external_qualification_receipt_cid": external["receipt_cid"],
            "external_qualification_payload_cid": external["payload_cid"],
            "release_report_sha256": BINDINGS.release_report_sha256,
            "source_revisions": SOURCES.to_dict(),
            "scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
            "lgswf_006_reused": False,
            "depends_on_lgcvf_121": True,
            "depends_on_lgcvf_122": True,
            "disposition": LGCVF_PRODUCTION_DECLINED_DISPOSITION,
            "release_qualified": False,
            "production_authorized": False,
            "limitations": [
                "production explicitly declined for self-signed single-user R&D"
            ],
        }
    )


def _schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_schemas_are_closed_and_accept_authenticated_r_and_d_receipts() -> None:
    external = _external_receipt()
    production = _production_receipt(external)
    external_schema = _schema(EXTERNAL_SCHEMA_PATH)
    production_schema = _schema(PRODUCTION_SCHEMA_PATH)
    Draft202012Validator.check_schema(external_schema)
    Draft202012Validator.check_schema(production_schema)
    Draft202012Validator(external_schema).validate(external)
    Draft202012Validator(production_schema).validate(production)

    external["unexpected"] = True
    production["production_authorized"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(external_schema).validate(external)
    with pytest.raises(ValidationError):
        Draft202012Validator(production_schema).validate(production)


def test_semantic_validators_verify_chain_but_never_project_authority() -> None:
    external_receipt = _external_receipt()
    external = validate_lgcvf_external_r_and_d_receipt(
        external_receipt,
        trust=TRUST,
        expected=BINDINGS,
        now=NOW,
    )
    production = validate_lgcvf_production_declined_r_and_d_receipt(
        _production_receipt(external_receipt),
        external_receipt=external_receipt,
        trust=TRUST,
        expected=BINDINGS,
        now=NOW,
    )

    assert external.disposition == "self_verified_r_and_d"
    assert production.disposition == "production_declined_r_and_d"
    assert external.release_qualified is False
    assert external.production_authorized is False
    assert production.release_qualified is False
    assert production.production_authorized is False
    with pytest.raises(TypeError):
        authority.ValidatedLgcvfRAndDReceipt(
            receipt_kind="forged",
            disposition="forged",
            signer_identity="forged",
            issued_at=NOW,
            expires_at=NOW,
            payload_cid=_cid("forged-payload"),
            receipt_cid=_cid("forged-receipt"),
            production_authorized=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("third_party_independence_claimed", True),
        ("release_qualified", True),
        ("production_authorized", True),
        ("disposition", "externally_qualified"),
    ],
)
def test_external_receipt_cannot_raise_its_authority(field: str, value: Any) -> None:
    payload = _payload(_external_receipt())
    payload[field] = value
    with pytest.raises(LgcvfRAndDAuthorityError, match=field):
        validate_lgcvf_external_r_and_d_receipt(
            _seal(payload),
            trust=TRUST,
            expected=BINDINGS,
            now=NOW,
        )


def test_checkout_fingerprint_and_sources_are_caller_pinned() -> None:
    payload = _payload(_external_receipt())
    payload["qualification_checkout_fingerprint_cid"] = _cid("foreign-checkout")
    with pytest.raises(LgcvfRAndDAuthorityError, match="checkout_fingerprint"):
        validate_lgcvf_external_r_and_d_receipt(
            _seal(payload), trust=TRUST, expected=BINDINGS, now=NOW
        )

    payload = _payload(_external_receipt())
    payload["source_revisions"]["ipfs_datasets_py"]["gitlink"] = "6" * 40
    with pytest.raises(LgcvfRAndDAuthorityError, match="source revisions"):
        validate_lgcvf_external_r_and_d_receipt(
            _seal(payload), trust=TRUST, expected=BINDINGS, now=NOW
        )


def test_signature_payload_cid_and_receipt_cid_are_each_verified() -> None:
    receipt = _external_receipt()
    payload = _payload(receipt)
    payload["limitations"] = ["content changed without the private test key"]
    forged = copy.deepcopy(payload)
    forged["payload_cid"] = content_identity(payload)
    forged["signature"] = receipt["signature"]
    forged["receipt_cid"] = content_identity(forged)
    with pytest.raises(LgcvfRAndDAuthorityError, match="signature is invalid"):
        validate_lgcvf_external_r_and_d_receipt(
            forged, trust=TRUST, expected=BINDINGS, now=NOW
        )

    receipt = _external_receipt()
    receipt["payload_cid"] = _cid("wrong-payload")
    with pytest.raises(LgcvfRAndDAuthorityError, match="payload_cid"):
        validate_lgcvf_external_r_and_d_receipt(
            receipt, trust=TRUST, expected=BINDINGS, now=NOW
        )

    receipt = _external_receipt()
    receipt["receipt_cid"] = _cid("wrong-receipt")
    with pytest.raises(LgcvfRAndDAuthorityError, match="receipt_cid"):
        validate_lgcvf_external_r_and_d_receipt(
            receipt, trust=TRUST, expected=BINDINGS, now=NOW
        )


@pytest.mark.parametrize(
    ("issued_at", "expires_at", "message"),
    [
        ("2026-08-28 10:00:00Z", "2026-08-28T14:00:00Z", "RFC3339"),
        ("2026-08-28T13:00:00Z", "2026-08-28T14:00:00Z", "not yet valid"),
        ("2026-08-28T08:00:00Z", "2026-08-28T12:00:00Z", "expired"),
        ("2026-08-28T14:00:00Z", "2026-08-28T10:00:00Z", "inverted"),
    ],
)
def test_receipt_validity_is_strict_and_bounded(
    issued_at: str,
    expires_at: str,
    message: str,
) -> None:
    payload = _payload(_external_receipt())
    payload["issued_at"] = issued_at
    payload["expires_at"] = expires_at
    with pytest.raises(LgcvfRAndDAuthorityError, match=message):
        validate_lgcvf_external_r_and_d_receipt(
            _seal(payload), trust=TRUST, expected=BINDINGS, now=NOW
        )


def test_embedded_key_cannot_nominate_a_different_trust_root() -> None:
    other_private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(33, 65)))
    other_public_key = other_private_key.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    payload = _payload(_external_receipt())
    payload["issuer"]["public_key_base64url"] = _b64url(other_public_key)
    payload["issuer"]["key_id"] = ed25519_public_key_id(other_public_key)
    with pytest.raises(LgcvfRAndDAuthorityError, match="locally pinned"):
        validate_lgcvf_external_r_and_d_receipt(
            _seal(payload), trust=TRUST, expected=BINDINGS, now=NOW
        )
    with pytest.raises(LgcvfRAndDAuthorityError, match="signer role"):
        LgcvfRAndDTrustPolicy(
            identity=TRUST.identity,
            role="independent third-party auditor",
            key_id=TRUST.key_id,
            public_key=TRUST.public_key,
        )


def test_operator_receipt_is_bound_to_valid_external_receipt_and_declines_production() -> (
    None
):
    external = _external_receipt()
    payload = _payload(_production_receipt(external))
    payload["external_qualification_receipt_cid"] = _cid("foreign-external")
    with pytest.raises(
        LgcvfRAndDAuthorityError, match="external_qualification_receipt_cid"
    ):
        validate_lgcvf_production_declined_r_and_d_receipt(
            _seal(payload),
            external_receipt=external,
            trust=TRUST,
            expected=BINDINGS,
            now=NOW,
        )

    payload = _payload(_production_receipt(external))
    payload["production_authorized"] = True
    with pytest.raises(LgcvfRAndDAuthorityError, match="production_authorized"):
        validate_lgcvf_production_declined_r_and_d_receipt(
            _seal(payload),
            external_receipt=external,
            trust=TRUST,
            expected=BINDINGS,
            now=NOW,
        )


def test_operator_validity_must_be_contained_by_external_receipt() -> None:
    external = _external_receipt()
    payload = _payload(_production_receipt(external))
    payload["expires_at"] = "2026-08-28T15:00:00Z"
    with pytest.raises(LgcvfRAndDAuthorityError, match="contained"):
        validate_lgcvf_production_declined_r_and_d_receipt(
            _seal(payload),
            external_receipt=external,
            trust=TRUST,
            expected=BINDINGS,
            now=NOW,
        )


def test_repository_trust_manifest_loads_only_the_pinned_public_key() -> None:
    trust = load_lgcvf_r_and_d_trust_policy(REPOSITORY_ROOT)
    assert trust.identity == "Benjamin Barber"
    assert trust.role == "sole R&D verifier and operator"
    assert trust.key_id == (
        "baguqeeraof5lqknosljjp2d26xqynxi2um53vtfq74dx6apttc3xxsapvslq"
    )
    assert "sha256:" + hashlib.sha256(trust.public_key).hexdigest() == (
        "sha256:8c3b0a628ca26fde650090269ab3653bc3fdb920536e9585e383a7e47041d0ce"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("third_party_independence_claimed", True, "third_party"),
        ("private_key_committed", True, "private_key_committed"),
        ("trust_model", "independent_external", "trust_model"),
        (
            "public_key_raw_sha256",
            "sha256:" + "0" * 64,
            "public_key_raw_sha256",
        ),
    ],
)
def test_trust_manifest_cannot_raise_authority_or_replace_key(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    manifest = json.loads(
        (REPOSITORY_ROOT / "config/lgcvf_r_and_d_authority_trust.json").read_text(
            encoding="utf-8"
        )
    )
    manifest[field] = value
    (config / "lgcvf_r_and_d_authority_trust.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (config / "lgcvf_r_and_d_authority_public_key.pem").write_bytes(
        (REPOSITORY_ROOT / "config/lgcvf_r_and_d_authority_public_key.pem").read_bytes()
    )
    with pytest.raises(LgcvfRAndDAuthorityError, match=message):
        load_lgcvf_r_and_d_trust_policy(tmp_path)


def test_manifest_and_pem_cannot_jointly_nominate_a_new_trust_root(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    other_private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(33, 65)))
    other_public = other_private_key.public_key()
    other_raw = other_public.public_bytes(Encoding.Raw, PublicFormat.Raw)
    manifest = json.loads(
        (REPOSITORY_ROOT / "config/lgcvf_r_and_d_authority_trust.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["key_id"] = ed25519_public_key_id(other_raw)
    manifest["public_key_base64url"] = _b64url(other_raw)
    manifest["public_key_raw_sha256"] = (
        "sha256:" + hashlib.sha256(other_raw).hexdigest()
    )
    (config / "lgcvf_r_and_d_authority_trust.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (config / "lgcvf_r_and_d_authority_public_key.pem").write_bytes(
        other_public.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    with pytest.raises(LgcvfRAndDAuthorityError, match="key_id differs"):
        load_lgcvf_r_and_d_trust_policy(tmp_path)


def test_production_module_exposes_no_private_key_or_issuance_primitive() -> None:
    source = Path(authority.__file__).read_text(encoding="utf-8")
    assert "Ed25519PrivateKey" not in source
    assert "load_pem_private_key" not in source
    assert all(
        "issue" not in name and "sign_receipt" not in name for name in authority.__all__
    )
