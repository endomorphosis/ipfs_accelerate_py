"""Focused tests for the authorized, external-key LGCVF R&D issuer CLI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.lgcvf_r_and_d_authority import (
    LgcvfAuthorityBindings,
    LgcvfRAndDTrustPolicy,
    LgcvfSourceRevisions,
    ed25519_public_key_id,
)
from scripts import resolve_lgcvf_r_and_d_successors as issuer


def _cid(label: str) -> str:
    return content_identity({"fixture": label})


def _key_material() -> tuple[Ed25519PrivateKey, LgcvfRAndDTrustPolicy]:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    trust = LgcvfRAndDTrustPolicy(
        identity="Authorized R&D User",
        role="sole R&D verifier and operator",
        key_id=ed25519_public_key_id(public),
        public_key=public,
    )
    return private, trust


def _bindings() -> LgcvfAuthorityBindings:
    return LgcvfAuthorityBindings(
        plan_cid=_cid("plan"),
        qualification_result_cid=_cid("qualification"),
        qualification_checkout_fingerprint_cid=_cid("checkout"),
        benchmark_report_cid=_cid("benchmark"),
        release_report_sha256="sha256:" + "1" * 64,
        source_revisions=LgcvfSourceRevisions(
            accelerator_head="1" * 40,
            accelerator_tree="2" * 40,
            datasets_head="3" * 40,
            datasets_tree="4" * 40,
            datasets_gitlink="3" * 40,
        ),
    )


def test_issuer_emits_valid_false_only_signed_receipt_chain() -> None:
    private, trust = _key_material()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    external, production = issuer._issue_receipts(
        trust=trust,
        expected=_bindings(),
        private_key=private,
        issued_at=(now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    )

    assert external["disposition"] == "self_verified_r_and_d"
    assert production["disposition"] == "production_declined_r_and_d"
    assert production["external_qualification_receipt_cid"] == external["receipt_cid"]
    assert external["release_qualified"] is False
    assert production["production_authorized"] is False


def test_bindings_hash_the_selfless_formal_plan_and_require_qualification_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"schema": "formal-plan@1", "tasks": []}
    plan_cid = content_identity(plan)
    qualification = {
        "plan_cid": plan_cid,
        "result_cid": _cid("qualification"),
        "checkout_fingerprint_cid": _cid("checkout"),
    }
    benchmark = {"report_cid": _cid("benchmark")}
    monkeypatch.setattr(issuer, "_load_object", lambda *args, **kwargs: plan)
    monkeypatch.setattr(issuer, "_sha256_file", lambda path: "sha256:" + "a" * 64)

    observed = issuer._bindings(qualification, benchmark, _bindings().source_revisions)

    assert observed.plan_cid == plan_cid
    with pytest.raises(issuer.ResolutionCommandError, match="current formal plan"):
        issuer._bindings(
            {**qualification, "plan_cid": _cid("stale-plan")},
            benchmark,
            _bindings().source_revisions,
        )


def test_outputs_are_strictly_append_only(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    first = {"schema": "receipt@2", "value": "first"}
    admitted = issuer._admit_output(path, first)
    issuer._write_guarded(path, first, admitted_previous=admitted)
    original = path.read_bytes()

    assert issuer._admit_output(path, first) == original
    issuer._write_guarded(path, first, admitted_previous=original)
    assert path.read_bytes() == original
    with pytest.raises(issuer.ResolutionCommandError, match="append-only"):
        issuer._admit_output(path, {"schema": "receipt@2", "value": "second"})

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text('{"schema":"receipt@1"}\n', encoding="utf-8")
    with pytest.raises(issuer.ResolutionCommandError, match="append-only"):
        issuer._admit_output(
            legacy_path,
            {"schema": "receipt@2", "value": "replacement"},
        )


def test_private_key_requires_owner_only_regular_bounded_file(tmp_path: Path) -> None:
    private, trust = _key_material()
    path = tmp_path / "private.pem"
    path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o644)
    with pytest.raises(issuer.ResolutionCommandError, match="owner-only"):
        issuer._load_private_key(path, trust)

    path.chmod(0o600)
    assert isinstance(issuer._load_private_key(path, trust), Ed25519PrivateKey)


def test_dirty_source_check_allows_only_declared_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = iter(
        (
            {
                "data/agent_supervisor/logic_governed_compositional_verification_fabric/benchmark_result.json"
            },
            set(),
            set(),
            set(),
        )
    )
    monkeypatch.setattr(issuer, "_git_path_set", lambda *args: next(calls))
    issuer._assert_no_uncommitted_source()

    calls = iter(({"ipfs_accelerate_py/unsafe.py"}, set(), set(), set()))
    monkeypatch.setattr(issuer, "_git_path_set", lambda *args: next(calls))
    with pytest.raises(issuer.ResolutionCommandError, match="unsafe.py"):
        issuer._assert_no_uncommitted_source()


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(issuer.ResolutionCommandError, match="duplicate JSON key"):
        issuer._decode_object(b'{"schema":"one","schema":"two"}', label="fixture")
    with pytest.raises(issuer.ResolutionCommandError, match="non-finite"):
        issuer._decode_object(b'{"value":NaN}', label="fixture")
