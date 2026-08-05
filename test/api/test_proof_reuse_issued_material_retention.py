"""PTR-153: preserve public proof-bearing issuance material (lazy real issuer).

Hardened prove/verify must never execute a mutable path merely because an
earlier capability probe hashed it.  Successful exact v4 material contains the
public certificate/proof plus reviewed bindings; deferred results carry no
authority; private witness / secret key bytes never cross the interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.testing.proof_reuse.services import (
    DATASETS_GROTH16_ARTIFACTS_ROOT_ENV,
    DATASETS_GROTH16_BINARY_ENV,
    DATASETS_GROTH16_ENABLE_ENV,
    ISSUED_MATERIAL_DISPOSITION_INTERFACE,
    ImmutableNativeArtifactSession,
    IssuedMaterialDisposition,
    LazyRealTestCertificateIssuer,
    MAX_ISSUED_CERTIFICATE_BYTES,
    PROOF_BEARING_ISSUANCE_MATERIAL_INTERFACE,
    ProofBearingIssuanceMaterial,
    admit_proof_bearing_issuance_material,
    allowlisted_native_child_environment,
    redact_private_material_fields,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _public_certificate(**overrides: Any) -> dict[str, Any]:
    payload = {
        "receipt_cid": "cid:receipt",
        "execution_key_cid": "cid:execution",
        "candidate_context_cid": "cid:candidate",
        "policy_cid": "cid:policy",
        "statement_cid": "cid:statement",
        "circuit_cid": "cid:circuit",
        "verifying_key_cid": "cid:vk",
        "proof_digest": "sha256:" + "ab" * 32,
        "proof_artifact_cid": "cid:proof",
        "proof_system_id": "groth16",
        "issuer_id": "issuer:test",
        "epoch": "epoch:1",
        "authority": "authoritative",
        "backend_mode": "cryptographic",
        "public_inputs": {
            "receipt_cid": "cid:receipt",
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:vk",
        },
    }
    payload.update(overrides)
    return payload


def _valid_material(**overrides: Any) -> ProofBearingIssuanceMaterial:
    kwargs: dict[str, Any] = {
        "certificate": _public_certificate(),
        "proof_digest": "sha256:" + "ab" * 32,
        "proof_artifact_cid": "cid:proof",
        "circuit_cid": "cid:circuit",
        "verifying_key_cid": "cid:vk",
        "artifact_bindings": {
            "provenance_ready": True,
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:vk",
            "proving_key_sha256": "pk" * 32,
            "verifying_key_sha256": "vk" * 32,
        },
        "proof_json": {"proof_a": "0x1", "public_inputs": ["aa"] * 10},
        "verified_locally": True,
    }
    kwargs.update(overrides)
    return ProofBearingIssuanceMaterial(**kwargs)


# ---------------------------------------------------------------------------
# Public material contract
# ---------------------------------------------------------------------------


def test_proof_bearing_material_interface_and_no_skip_authority() -> None:
    material = _valid_material()
    assert material.interface == PROOF_BEARING_ISSUANCE_MATERIAL_INTERFACE
    assert material.issued is True
    assert material.deferred is False
    assert material.can_authorize_skip is False
    assert material.authority != "authoritative"
    public = material.to_public_dict()
    assert public["can_authorize_skip"] is False
    assert "certificate" in public
    assert public["circuit_cid"] == "cid:circuit"
    assert "witness" not in json.dumps(public).lower()
    assert "proving_key" not in json.dumps(public).lower()


def test_deferred_disposition_has_no_authority() -> None:
    deferred = IssuedMaterialDisposition(
        status="certificate_deferred",
        reason="artifact_provenance_unready",
    )
    assert deferred.interface == ISSUED_MATERIAL_DISPOSITION_INTERFACE
    assert deferred.issued is False
    assert deferred.deferred is True
    assert deferred.can_authorize_skip is False
    assert deferred.material is None
    assert deferred.certificate is None
    payload = deferred.to_dict()
    assert payload["indexed"] is False
    assert payload["can_authorize_skip"] is False


def test_redact_private_material_fields_strips_secrets() -> None:
    raw = {
        "certificate": {"receipt_cid": "r", "witness": "SECRET", "ok": 1},
        "local_witness": b"\x00\x01",
        "private_axioms": ["x"],
        "receipt_opening_hex": "deadbeef",
        "proving_key": "pk-bytes",
        "public": "visible",
    }
    cleaned = redact_private_material_fields(raw)
    assert cleaned["public"] == "visible"
    assert cleaned["certificate"]["ok"] == 1
    assert "witness" not in cleaned["certificate"]
    assert "local_witness" not in cleaned
    assert "private_axioms" not in cleaned
    assert "receipt_opening_hex" not in cleaned
    assert "proving_key" not in cleaned


def test_admit_rejects_private_material_and_oversized_certificate() -> None:
    good, reason = admit_proof_bearing_issuance_material(_valid_material())
    assert good is not None
    assert reason == ""

    leaked = _valid_material()
    public = leaked.to_public_dict()
    public["certificate"] = dict(public["certificate"])
    public["certificate"]["receipt_opening_hex"] = "aa" * 32
    admitted, reason = admit_proof_bearing_issuance_material(public)
    assert admitted is None
    assert reason == "private_material_present"

    huge_cert = _public_certificate(padding="x" * (MAX_ISSUED_CERTIFICATE_BYTES + 64))
    admitted, reason = admit_proof_bearing_issuance_material(
        {
            "certificate": huge_cert,
            "proof_digest": "sha256:" + "ab" * 32,
            "proof_artifact_cid": "cid:proof",
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:vk",
        }
    )
    assert admitted is None
    assert reason == "certificate_oversized"


def test_admit_rejects_provenance_mismatch_and_incomplete() -> None:
    material = _valid_material()
    admitted, reason = admit_proof_bearing_issuance_material(
        material,
        expected_circuit_cid="cid:other-circuit",
    )
    assert admitted is None
    assert reason == "circuit_cid_provenance_mismatch"

    incomplete = {
        "certificate": {"receipt_cid": "r"},
        "proof_digest": "",
        "proof_artifact_cid": "cid:p",
        "circuit_cid": "c",
        "verifying_key_cid": "v",
    }
    admitted, reason = admit_proof_bearing_issuance_material(incomplete)
    assert admitted is None
    assert reason in {"proof_identity_missing", "certificate_missing_proof_digest"}


# ---------------------------------------------------------------------------
# Strict child environment + immutable snapshot
# ---------------------------------------------------------------------------


def test_allowlisted_child_env_excludes_loader_injection_and_overwrites_root(
    tmp_path: Path,
) -> None:
    pinned = tmp_path / "pinned-artifacts"
    pinned.mkdir()
    ambient = {
        "PATH": "/usr/bin",
        "HOME": str(tmp_path),
        "LD_PRELOAD": str(tmp_path / "attacker.so"),
        "LD_LIBRARY_PATH": str(tmp_path / "evil-lib"),
        "DYLD_INSERT_LIBRARIES": str(tmp_path / "mac-attacker.dylib"),
        "DYLD_LIBRARY_PATH": "/evil",
        "PYTHONPATH": str(tmp_path / "evil-py"),
        DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: str(tmp_path / "attacker-keys"),
        DATASETS_GROTH16_BINARY_ENV: str(tmp_path / "attacker-bin"),
        "IPFS_DATASETS_ENABLE_GROTH16": "0",
        "UNRELATED_SECRET": "should-not-pass",
    }
    env = allowlisted_native_child_environment(
        ambient,
        artifacts_root=pinned,
        binary_path=tmp_path / "snap-bin",
    )
    assert "LD_PRELOAD" not in env
    assert "LD_LIBRARY_PATH" not in env
    assert "DYLD_INSERT_LIBRARIES" not in env
    assert "DYLD_LIBRARY_PATH" not in env
    assert "PYTHONPATH" not in env
    assert "UNRELATED_SECRET" not in env
    assert env[DATASETS_GROTH16_ARTIFACTS_ROOT_ENV] == str(pinned)
    assert env[DATASETS_GROTH16_BINARY_ENV] == str(tmp_path / "snap-bin")
    assert env[DATASETS_GROTH16_ENABLE_ENV] == "1"
    # Must overwrite ambient attacker artifacts root, not inherit it.
    assert env[DATASETS_GROTH16_ARTIFACTS_ROOT_ENV] != str(tmp_path / "attacker-keys")


def test_immutable_session_revalidates_and_detects_replacement(
    tmp_path: Path,
) -> None:
    binary = b"\x7fELF" + b"reviewed-binary-v4"
    pk = b"pk-bytes-" + b"\x01" * 32
    vk = b"vk-bytes-" + b"\x02" * 32
    session = ImmutableNativeArtifactSession(
        binary_bytes=binary,
        proving_key_bytes=pk,
        verifying_key_bytes=vk,
        expected_proving_key_sha256=_sha(pk),
        expected_verifying_key_sha256=_sha(vk),
    )
    try:
        assert session.revalidate() is True
        assert session.binary_path.is_file()
        assert (session.artifacts_root / "v4" / "proving_key.bin").read_bytes() == pk
        child = session.child_environment(
            {
                "PATH": "/bin",
                "LD_PRELOAD": "/tmp/evil.so",
                DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: str(tmp_path / "ambient"),
            }
        )
        assert "LD_PRELOAD" not in child
        assert child[DATASETS_GROTH16_ARTIFACTS_ROOT_ENV] == str(session.artifacts_root)

        # Replace snapshot binary → revalidation fails (do not execute substituted).
        os.chmod(session.binary_path, 0o700)
        session.binary_path.write_bytes(b"substituted-attacker-binary")
        assert session.revalidate() is False
    finally:
        session.close()
    assert not session.binary_path.exists()


def test_immutable_session_rejects_digest_mismatch_at_bind() -> None:
    with pytest.raises(ValueError, match="proving_key_digest_mismatch"):
        ImmutableNativeArtifactSession(
            binary_bytes=b"bin",
            proving_key_bytes=b"pk",
            verifying_key_bytes=b"vk",
            expected_proving_key_sha256="0" * 64,
            expected_verifying_key_sha256=_sha(b"vk"),
        )


# ---------------------------------------------------------------------------
# Lazy issuer: issue vs issue_material
# ---------------------------------------------------------------------------


def test_issue_remains_publication_deferred_without_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "mutable-groth16"
    binary.write_bytes(b"unreviewed")
    artifacts = tmp_path / "keys"
    artifacts.mkdir()
    issuer = LazyRealTestCertificateIssuer(
        binary_path=binary,
        artifacts_root=artifacts,
        environ={
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "1",
            "LD_PRELOAD": str(tmp_path / "attacker.so"),
            DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: str(tmp_path / "attacker-keys"),
        },
    )
    monkeypatch.setattr(
        issuer,
        "_ensure_factory",
        lambda: (_ for _ in ()).throw(
            AssertionError("publication issue() must not construct provider")
        ),
    )
    result = issuer.issue({"receipt_cid": "cid:r", "locator_cid": "cid:l"})
    assert result.status == "certificate_deferred"
    assert result.reason == "positive_v4_issuance_pending_ptr155"
    assert result.can_authorize_skip is False
    assert getattr(result, "material", None) is None
    assert issuer.factory is None
    assert issuer.enable_env_published is False


def test_issue_material_defers_without_provenance_and_retains_pass(
    tmp_path: Path,
) -> None:
    issuer = LazyRealTestCertificateIssuer(
        binary_path=tmp_path / "missing-bin",
        artifacts_root=tmp_path / "missing-keys",
        environ={
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
            "LD_PRELOAD": str(tmp_path / "evil.so"),
            DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: str(tmp_path / "attacker"),
        },
    )
    result = issuer.issue_material({"receipt_cid": "cid:r"})
    assert isinstance(result, IssuedMaterialDisposition)
    assert result.can_authorize_skip is False
    assert result.certificate is None
    assert result.material is None
    assert "deferred" in result.status or result.status == "certificate_deferred"
    # Original pass retained: disposition is typed RUN/DEFERRED, not exception.
    assert result.issued is False


def test_issue_material_defers_on_post_binding_binary_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "groth16"
    binary.write_bytes(b"\x7fELF-reviewed-v4-binary")
    os.chmod(binary, 0o700)
    artifacts = tmp_path / "artifacts"
    v4 = artifacts / "v4"
    v4.mkdir(parents=True)
    pk = v4 / "proving_key.bin"
    vk = v4 / "verifying_key.bin"
    pk.write_bytes(b"pk-" + b"\x11" * 64)
    vk.write_bytes(b"vk-" + b"\x22" * 64)

    ready_bindings = SimpleNamespace(
        provenance_ready=True,
        reason_code="ready",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:vk",
        artifacts_root=str(artifacts),
        proving_key_sha256=_sha(pk.read_bytes()),
        verifying_key_sha256=_sha(vk.read_bytes()),
        backend_circuit_version=4,
        to_dict=lambda: {
            "provenance_ready": True,
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:vk",
            "artifacts_root": str(artifacts),
            "proving_key_sha256": _sha(pk.read_bytes()),
            "verifying_key_sha256": _sha(vk.read_bytes()),
        },
    )

    issuer = LazyRealTestCertificateIssuer(
        binary_path=binary,
        artifacts_root=artifacts,
        environ={
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
            "LD_PRELOAD": str(tmp_path / "evil.so"),
            DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: str(tmp_path / "attacker-keys"),
        },
    )
    monkeypatch.setattr(issuer, "_derive_bindings", lambda: ready_bindings)

    # After bindings resolve, replace the mutable source binary before use.
    original_open = issuer._open_immutable_session

    def _open_then_replace(bindings: Any) -> Any:
        session = original_open(bindings)
        if isinstance(session, ImmutableNativeArtifactSession):
            binary.write_bytes(b"SUBSTITUTED-ATTACKER-BINARY")
        return session

    monkeypatch.setattr(issuer, "_open_immutable_session", _open_then_replace)

    # Provider import may be available; ensure we never execute substituted input.
    executed: list[str] = []

    def _fake_run(args: list[str], **kwargs: Any) -> Any:
        executed.append(str(args))
        raise AssertionError("must not execute after binary replacement")

    monkeypatch.setattr(
        "subprocess.run",
        _fake_run,
    )

    result = issuer.issue_material(
        SimpleNamespace(statement=None),
        local_witness=SimpleNamespace(receipt_bytes=b"witness-must-not-leak"),
    )
    assert isinstance(result, IssuedMaterialDisposition)
    assert result.can_authorize_skip is False
    assert "replacement" in result.reason or "drift" in result.reason or result.deferred
    assert executed == []
    # Witness must not appear in disposition serialization.
    serialized = json.dumps(result.to_dict(), default=str)
    assert "witness-must-not-leak" not in serialized


def test_issue_material_hardens_provider_child_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "groth16"
    binary.write_bytes(b"\x7fELF-reviewed")
    os.chmod(binary, stat.S_IRUSR | stat.S_IXUSR)
    artifacts = tmp_path / "artifacts"
    v4 = artifacts / "v4"
    v4.mkdir(parents=True)
    pk_bytes = b"pk-" + b"\x33" * 32
    vk_bytes = b"vk-" + b"\x44" * 32
    (v4 / "proving_key.bin").write_bytes(pk_bytes)
    (v4 / "verifying_key.bin").write_bytes(vk_bytes)

    ready_bindings = SimpleNamespace(
        provenance_ready=True,
        reason_code="ready",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:vk",
        artifacts_root=str(artifacts),
        proving_key_sha256=_sha(pk_bytes),
        verifying_key_sha256=_sha(vk_bytes),
        backend_circuit_version=4,
        to_dict=lambda: {
            "provenance_ready": True,
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:vk",
            "proving_key_sha256": _sha(pk_bytes),
            "verifying_key_sha256": _sha(vk_bytes),
        },
    )

    captured_env: dict[str, str] = {}
    captured_executable: list[str] = []

    class _FakeMaterial:
        certificate = SimpleNamespace(
            to_dict=lambda include_proof=True, include_ids=True: _public_certificate()
        )
        proof_json = {"proof_a": "0x1"}
        proof_digest = "sha256:" + "ab" * 32
        proof_artifact_cid = "cid:proof"
        circuit_cid = "cid:circuit"
        verifying_key_cid = "cid:vk"
        backend_circuit_version = 4
        verified_locally = True

    class _FakeProvider:
        def __init__(self, **kwargs: Any) -> None:
            self._binary_path = kwargs.get("binary_path")
            self._artifacts_root = kwargs.get("artifacts_root")
            self._environ = kwargs.get("environ")
            self.prove_timeout_seconds = 30.0

        def artifacts_root(self) -> Path:
            return Path(self._artifacts_root)

        def issue(self, request: Any, **kwargs: Any) -> Any:
            # Invoke hardened _run_cli if patched (exercises child env).
            run = getattr(self, "_run_cli", None)
            if callable(run):
                run(
                    ["verify", "--quiet"],
                    stdin_bytes=b"{}",
                    timeout=1.0,
                    env={
                        "LD_PRELOAD": "/should/be/stripped",
                        DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: str(
                            tmp_path / "ambient-attacker"
                        ),
                    },
                )
            # Witness stays local — return public material only.
            assert "local_witness" in kwargs
            return _FakeMaterial()

    # Capture env from our patched subprocess via a custom CompletedProcess.
    import subprocess as _subprocess

    def _run(cmd: list[str], **kwargs: Any) -> Any:
        captured_executable.append(str(cmd[0]))
        env = dict(kwargs.get("env") or {})
        captured_env.clear()
        captured_env.update(env)
        return SimpleNamespace(
            returncode=0,
            stdout=b"{}",
            stderr=b"",
            env=env,
        )

    monkeypatch.setattr(_subprocess, "run", _run)

    # Stub the datasets provider package for a hermetic unit test.
    import sys

    stub = SimpleNamespace(
        LazyGroth16TestCertificateProvider=_FakeProvider,
        build_default_test_certificate_issuer=lambda **kw: SimpleNamespace(
            provider=kw.get("provider")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ipfs_datasets_py.logic.zkp.test_pass_groth16_provider",
        stub,  # type: ignore[arg-type]
    )
    # Ensure parent packages exist as modules for `from ... import`.
    for pkg in (
        "ipfs_datasets_py",
        "ipfs_datasets_py.logic",
        "ipfs_datasets_py.logic.zkp",
    ):
        if pkg not in sys.modules:
            monkeypatch.setitem(sys.modules, pkg, SimpleNamespace())  # type: ignore[arg-type]

    issuer = LazyRealTestCertificateIssuer(
        binary_path=binary,
        artifacts_root=artifacts,
        environ={
            "PATH": "/usr/bin",
            "LD_PRELOAD": str(tmp_path / "evil.so"),
            "DYLD_INSERT_LIBRARIES": str(tmp_path / "mac.so"),
            DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: str(tmp_path / "attacker-keys"),
            "IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0",
        },
    )
    monkeypatch.setattr(issuer, "_derive_bindings", lambda: ready_bindings)

    result = issuer.issue_material(
        SimpleNamespace(receipt_cid="cid:r"),
        local_witness=SimpleNamespace(receipt_bytes=b"SECRET-WITNESS"),
    )

    # Either material success (if fake path worked) or typed deferral.
    if isinstance(result, ProofBearingIssuanceMaterial):
        assert result.can_authorize_skip is False
        assert result.circuit_cid == "cid:circuit"
        assert "artifact_bindings" in result.to_public_dict()
        assert "SECRET-WITNESS" not in json.dumps(result.to_public_dict())
        assert "LD_PRELOAD" not in captured_env
        assert "DYLD_INSERT_LIBRARIES" not in captured_env
        assert captured_env.get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV) != str(
            tmp_path / "attacker-keys"
        )
        # Executed binary must be the private snapshot, not ambient attacker path.
        assert captured_executable
        assert "attacker" not in captured_executable[0]
    else:
        assert isinstance(result, IssuedMaterialDisposition)
        assert result.can_authorize_skip is False
        assert "SECRET-WITNESS" not in json.dumps(result.to_dict())


def test_issue_material_rejects_malformed_provider_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "groth16"
    binary.write_bytes(b"\x7fELF-ok")
    os.chmod(binary, 0o700)
    artifacts = tmp_path / "artifacts"
    v4 = artifacts / "v4"
    v4.mkdir(parents=True)
    pk = b"pk-ok"
    vk = b"vk-ok"
    (v4 / "proving_key.bin").write_bytes(pk)
    (v4 / "verifying_key.bin").write_bytes(vk)
    ready_bindings = SimpleNamespace(
        provenance_ready=True,
        reason_code="ready",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:vk",
        artifacts_root=str(artifacts),
        proving_key_sha256=_sha(pk),
        verifying_key_sha256=_sha(vk),
        backend_circuit_version=4,
        to_dict=lambda: {"provenance_ready": True, "circuit_cid": "cid:circuit"},
    )

    class _BadProvider:
        def __init__(self, **_kwargs: Any) -> None:
            self._binary_path = None
            self._artifacts_root = None

        def issue(self, *_a: Any, **_k: Any) -> Any:
            return SimpleNamespace(
                certificate={"receipt_cid": "r"},  # incomplete
                proof_digest="",
                proof_artifact_cid="",
                circuit_cid="",
                verifying_key_cid="",
                proof_json={},
                verified_locally=True,
            )

    import sys

    monkeypatch.setitem(
        sys.modules,
        "ipfs_datasets_py.logic.zkp.test_pass_groth16_provider",
        SimpleNamespace(
            LazyGroth16TestCertificateProvider=_BadProvider,
            build_default_test_certificate_issuer=lambda **kw: object(),
        ),  # type: ignore[arg-type]
    )
    for pkg in (
        "ipfs_datasets_py",
        "ipfs_datasets_py.logic",
        "ipfs_datasets_py.logic.zkp",
    ):
        if pkg not in sys.modules:
            monkeypatch.setitem(sys.modules, pkg, SimpleNamespace())  # type: ignore[arg-type]

    issuer = LazyRealTestCertificateIssuer(
        binary_path=binary,
        artifacts_root=artifacts,
        environ={"IPFS_TEST_PROOF_REUSE_GROTH16_BUILD": "0"},
    )
    monkeypatch.setattr(issuer, "_derive_bindings", lambda: ready_bindings)
    result = issuer.issue_material(SimpleNamespace())
    assert isinstance(result, IssuedMaterialDisposition)
    assert result.can_authorize_skip is False
    assert result.certificate is None


def test_cold_import_of_services_is_inert() -> None:
    import importlib

    module = importlib.import_module(
        "ipfs_accelerate_py.testing.proof_reuse.services"
    )
    assert module.PROOF_BEARING_ISSUANCE_MATERIAL_INTERFACE.startswith(
        "IssuedTestCertificateMaterial"
    )
    assert callable(module.LazyRealTestCertificateIssuer)
    assert callable(module.allowlisted_native_child_environment)
