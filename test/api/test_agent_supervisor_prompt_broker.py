"""ASE-012 transient and capability-protected prompt-body brokering tests."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest


def _ensure_hermetic_cid_utils() -> None:
    """Install a multiformats-backed cid_utils when editable deps are missing.

    Hermetic validation sets ``PYTHONNOUSERSITE=1`` and a neutral ``HOME``, so
    editable installs of ``ipfs_datasets_py`` are invisible.  Empty worktree
    stubs then resolve as a namespace package without ``utils``, and importing
    entrypoint contracts (via ``entrypoints/__init__.py``) fails collection.
    """

    try:
        from ipfs_datasets_py.utils import cid_utils as _cid_utils  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    def _canonical_json_bytes(obj: object) -> bytes:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=repr,
        ).encode("utf-8")

    def _canonical_dag_json_bytes(obj: object) -> bytes:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def _cid_for_bytes(
        data: bytes,
        *,
        base: str = "base32",
        codec: str = "raw",
        mh_type: str = "sha2-256",
        version: int = 1,
    ) -> str:
        from multiformats import CID, multihash

        digest = multihash.digest(bytes(data), mh_type)
        return str(CID(base, version, codec, digest))

    def _cid_for_dag_json(
        obj: object,
        *,
        base: str = "base32",
        mh_type: str = "sha2-256",
        version: int = 1,
    ) -> str:
        return _cid_for_bytes(
            _canonical_dag_json_bytes(obj),
            base=base,
            codec="dag-json",
            mh_type=mh_type,
            version=version,
        )

    def _cid_for_obj(
        obj: object,
        *,
        base: str = "base32",
        codec: str = "raw",
        mh_type: str = "sha2-256",
        version: int = 1,
    ) -> str:
        return _cid_for_bytes(
            _canonical_json_bytes(obj),
            base=base,
            codec=codec,
            mh_type=mh_type,
            version=version,
        )

    def _validate_cid(
        value: object,
        *,
        codecs: object = ("raw", "dag-json"),
        mh_type: str = "sha2-256",
        version: int = 1,
        base: str = "base32",
    ) -> str:
        if not isinstance(value, str) or not value or value != value.lower():
            raise ValueError("CID must be a nonempty lowercase string")
        from multiformats import CID, multihash

        try:
            parsed = CID.decode(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("CID is not decodable") from exc
        allowed = frozenset(codecs)  # type: ignore[arg-type]
        expected_size = multihash.get(mh_type).max_digest_size
        if (
            parsed.version != version
            or parsed.codec.name not in allowed
            or parsed.hashfun.name != mh_type
            or (
                expected_size is not None
                and len(parsed.raw_digest) != expected_size
            )
            or parsed.base.name != base
            or str(parsed) != value
        ):
            raise ValueError(
                "CID must use the requested canonical version/base/codec/multihash"
            )
        return value

    datasets = sys.modules.get("ipfs_datasets_py")
    if datasets is None:
        datasets = types.ModuleType("ipfs_datasets_py")
        datasets.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ipfs_datasets_py"] = datasets

    utils = sys.modules.get("ipfs_datasets_py.utils")
    if utils is None:
        utils = types.ModuleType("ipfs_datasets_py.utils")
        utils.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ipfs_datasets_py.utils"] = utils
        datasets.utils = utils  # type: ignore[attr-defined]

    cid_mod = types.ModuleType("ipfs_datasets_py.utils.cid_utils")
    cid_mod.canonical_json_bytes = _canonical_json_bytes  # type: ignore[attr-defined]
    cid_mod.canonical_dag_json_bytes = _canonical_dag_json_bytes  # type: ignore[attr-defined]
    cid_mod.cid_for_bytes = _cid_for_bytes  # type: ignore[attr-defined]
    cid_mod.cid_for_dag_json = _cid_for_dag_json  # type: ignore[attr-defined]
    cid_mod.cid_for_obj = _cid_for_obj  # type: ignore[attr-defined]
    cid_mod.validate_cid = _validate_cid  # type: ignore[attr-defined]
    cid_mod.__all__ = [  # type: ignore[attr-defined]
        "canonical_dag_json_bytes",
        "canonical_json_bytes",
        "cid_for_bytes",
        "cid_for_dag_json",
        "cid_for_obj",
        "validate_cid",
    ]
    sys.modules["ipfs_datasets_py.utils.cid_utils"] = cid_mod
    utils.cid_utils = cid_mod  # type: ignore[attr-defined]


_ensure_hermetic_cid_utils()

from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (  # noqa: E402
    SupervisorInvocationRequest,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.prompt_broker import (  # noqa: E402
    DEFAULT_TTL_MS,
    PROMPT_BROKER_REQUIREMENT_ID,
    PromptBodyBroker,
    PromptBodyStatus,
    PromptBrokerBoundsError,
    PromptCapability,
    PromptCapabilityError,
    PromptCrossRunError,
    PromptExpiredError,
    PromptNotFoundError,
    PromptReference,
    PromptStorageKind,
    cid_for_bytes,
)

PROMPT_CANARY = "ASE_PROMPT_CANARY_DO_NOT_PERSIST_8d76d6d9"
CREDENTIAL_CANARY = "ASE_CREDENTIAL_CANARY_DO_NOT_PERSIST_4f8a5c11"
PROMPT = (
    f"Improve the validation cache. {PROMPT_CANARY} "
    f"credential={CREDENTIAL_CANARY}"
)
RUN_A = "run:ase-012-alpha"
RUN_B = "run:ase-012-beta"


class _Clock:
    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        self.now = start_ms

    def __call__(self) -> int:
        return self.now

    def advance(self, delta_ms: int) -> int:
        self.now += delta_ms
        return self.now


def _body() -> bytes:
    return PROMPT.encode("utf-8")


def test_planner_receives_exact_bytes_during_authorized_window(tmp_path: Path) -> None:
    clock = _Clock()
    with PromptBodyBroker(clock_ms=clock, artifact_dir=tmp_path / "broker") as broker:
        reference, capability = broker.deposit(_body(), run_id=RUN_A, max_uses=2)
        assert reference.prompt_cid == cid_for_bytes(_body(), codec="raw")
        assert reference.storage is PromptStorageKind.MEMORY
        assert reference.status is PromptBodyStatus.ACTIVE

        exact = broker.resolve(reference, capability, run_id=RUN_A)
        assert exact == _body()
        assert type(exact) is bytes

        with broker.open_for_planner(reference, capability, run_id=RUN_A) as body:
            assert body == _body()


def test_routine_surfaces_contain_only_cid_and_reference(tmp_path: Path) -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock, artifact_dir=tmp_path / "broker")
    reference, capability = broker.deposit(
        _body(),
        run_id=RUN_A,
        enable_encrypted_artifact=True,
    )

    durable_ref = reference.to_dict()
    durable_json = reference.to_json()
    redacted_cap = capability.redacted_dict()
    cap_repr = repr(capability)
    invocation = SupervisorInvocationRequest.from_prompt(
        PROMPT,
        prompt_ref=reference.prompt_ref,
    )
    invocation_json = invocation.to_json()
    argv = [
        "ipfs-accelerate",
        "agent",
        "run",
        f"--prompt-ref={reference.prompt_ref}",
        f"--prompt-cid={reference.prompt_cid}",
    ]
    environment = {
        "PROMPT_REF": reference.prompt_ref,
        "PROMPT_CID": reference.prompt_cid,
        "RUN_ID": RUN_A,
    }
    event = {
        "type": "prompt.deposited",
        "prompt_ref": reference.prompt_ref,
        "prompt_cid": reference.prompt_cid,
        "capability": redacted_cap,
    }
    log_line = (
        f"deposited prompt_ref={reference.prompt_ref} "
        f"prompt_cid={reference.prompt_cid}"
    )

    for surface in (
        durable_json,
        json.dumps(durable_ref),
        json.dumps(redacted_cap),
        cap_repr,
        invocation_json,
        " ".join(argv),
        json.dumps(environment),
        json.dumps(event),
        log_line,
    ):
        assert PROMPT_CANARY not in surface
        assert CREDENTIAL_CANARY not in surface
        assert capability.token not in surface
        assert "transient_prompt_body" not in surface or PROMPT not in surface

    assert "token" not in redacted_cap or redacted_cap.get("token_redacted") is True
    assert redacted_cap["capability_digest"].startswith("sha256:")
    assert reference.prompt_ref in durable_json
    assert reference.prompt_cid in durable_json
    assert PROMPT not in durable_json
    broker.close()


def test_cross_run_access_fails(tmp_path: Path) -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock)
    reference, capability = broker.deposit(_body(), run_id=RUN_A)

    with pytest.raises(PromptCrossRunError, match="not authorized"):
        broker.resolve(reference, capability, run_id=RUN_B)

    other_ref, other_cap = broker.deposit(
        "other prompt body for a different run",
        run_id=RUN_B,
    )
    # Capability issued for run A cannot open run B's body (and vice versa).
    with pytest.raises(PromptCrossRunError):
        broker.resolve(other_ref, capability, run_id=RUN_B)
    with pytest.raises(PromptCrossRunError):
        broker.resolve(reference, other_cap, run_id=RUN_A)
    with pytest.raises(PromptCrossRunError):
        broker.resolve(other_ref, other_cap, run_id=RUN_A)
    broker.close()


def test_forged_or_missing_capability_is_denied() -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock)
    reference, capability = broker.deposit(_body(), run_id=RUN_A)

    with pytest.raises(PromptCapabilityError):
        broker.resolve(reference, "A" * 43, run_id=RUN_A)
    with pytest.raises(PromptNotFoundError):
        broker.resolve("prompt-broker:" + "0" * 32, capability, run_id=RUN_A)

    forged = PromptCapability(
        token="B" * 43,
        prompt_ref=reference.prompt_ref,
        run_id=RUN_A,
        prompt_cid=reference.prompt_cid,
        issued_at_ms=reference.issued_at_ms,
        expires_at_ms=reference.expires_at_ms,
    )
    with pytest.raises(PromptCapabilityError):
        broker.resolve(reference, forged, run_id=RUN_A)
    broker.close()


def test_expiry_is_explicit_and_blocks_retrieval() -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock, default_ttl_ms=1_000)
    reference, capability = broker.deposit(_body(), run_id=RUN_A, ttl_ms=500)
    assert broker.resolve(reference, capability, run_id=RUN_A, consume=False) == _body()

    clock.advance(500)
    with pytest.raises(PromptExpiredError):
        broker.resolve(reference, capability, run_id=RUN_A)

    described = broker.describe(reference)
    assert described.status is PromptBodyStatus.EXPIRED
    broker.close()


def test_expire_due_zeroizes_and_reports_refs() -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock)
    reference, capability = broker.deposit(_body(), run_id=RUN_A, ttl_ms=100)
    clock.advance(100)
    expired = broker.expire_due()
    assert reference.prompt_ref in expired
    with pytest.raises(PromptExpiredError):
        broker.resolve(reference, capability, run_id=RUN_A)
    surfaces = broker.inspect_durable_surfaces()
    # Body no longer resident after expiry.
    for surface in surfaces:
        if surface.get("kind") == "broker_entry":
            if surface["reference"]["prompt_ref"] == reference.prompt_ref:
                assert surface["body_resident"] is False
                assert surface["status"] == PromptBodyStatus.EXPIRED.value
    broker.close()


def test_single_use_capability_exhausts() -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock)
    reference, capability = broker.deposit(_body(), run_id=RUN_A, max_uses=1)
    assert broker.resolve(reference, capability, run_id=RUN_A) == _body()
    with pytest.raises(PromptNotFoundError):
        broker.resolve(reference, capability, run_id=RUN_A)
    assert broker.describe(reference).status is PromptBodyStatus.EXHAUSTED
    broker.close()


def test_release_zeroizes_before_expiry(tmp_path: Path) -> None:
    clock = _Clock()
    broker = PromptBodyBroker(
        clock_ms=clock,
        artifact_dir=tmp_path / "broker",
        master_secret=b"test-master-secret-32-bytes-long!!",
    )
    reference, capability = broker.deposit(
        _body(),
        run_id=RUN_A,
        enable_encrypted_artifact=True,
        max_uses=3,
    )
    artifact_path = broker.artifact_dir / reference.artifact_ref
    assert artifact_path.is_file()
    released = broker.release(reference, run_id=RUN_A, capability=capability)
    assert released.status is PromptBodyStatus.RELEASED
    assert not artifact_path.exists()
    with pytest.raises(PromptNotFoundError):
        broker.resolve(reference, capability, run_id=RUN_A)
    broker.close()


def test_encrypted_artifact_survives_restart_with_master_secret(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    artifact_dir = tmp_path / "broker"
    master = b"stable-master-secret-for-restart!!"
    broker = PromptBodyBroker(
        artifact_dir=artifact_dir,
        master_secret=master,
        clock_ms=clock,
    )
    reference, capability = broker.deposit(
        _body(),
        run_id=RUN_A,
        enable_encrypted_artifact=True,
        max_uses=2,
    )
    token = capability.token
    prompt_ref = reference.prompt_ref
    assert broker.restart_behavior()["encrypted_artifacts_recoverable_after_restart"]
    broker.close()

    # New process/broker with the same secret and artifact root.
    restored = PromptBodyBroker(
        artifact_dir=artifact_dir,
        master_secret=master,
        clock_ms=clock,
    )
    behavior = restored.restart_behavior()
    assert behavior["memory_bodies_survive_restart"] is False
    assert behavior["encrypted_artifacts_recoverable_after_restart"] is True
    assert behavior["capability_required_after_restart"] is True
    assert behavior["requirement_id"] == PROMPT_BROKER_REQUIREMENT_ID

    exact = restored.resolve(prompt_ref, token, run_id=RUN_A)
    assert exact == _body()
    restored.close()


def test_memory_only_body_missing_after_restart(tmp_path: Path) -> None:
    clock = _Clock()
    artifact_dir = tmp_path / "broker"
    master = b"stable-master-secret-for-restart!!"
    broker = PromptBodyBroker(
        artifact_dir=artifact_dir,
        master_secret=master,
        clock_ms=clock,
    )
    reference, capability = broker.deposit(
        _body(),
        run_id=RUN_A,
        enable_encrypted_artifact=False,
        max_uses=2,
    )
    token = capability.token
    prompt_ref = reference.prompt_ref
    broker.close()

    restored = PromptBodyBroker(
        artifact_dir=artifact_dir,
        master_secret=master,
        clock_ms=clock,
    )
    described = restored.describe(prompt_ref)
    assert described.status is PromptBodyStatus.MISSING_AFTER_RESTART
    with pytest.raises(PromptNotFoundError, match="not recoverable after restart"):
        restored.resolve(prompt_ref, token, run_id=RUN_A)
    restored.close()


def test_ephemeral_master_secret_cannot_decrypt_after_restart(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    artifact_dir = tmp_path / "broker"
    broker = PromptBodyBroker(artifact_dir=artifact_dir, clock_ms=clock)
    assert (
        broker.restart_behavior()["encrypted_artifacts_recoverable_after_restart"]
        is False
    )
    reference, capability = broker.deposit(
        _body(),
        run_id=RUN_A,
        enable_encrypted_artifact=True,
        max_uses=2,
    )
    token = capability.token
    prompt_ref = reference.prompt_ref
    broker.close()

    # Fresh ephemeral master cannot decrypt prior ciphertext.
    restored = PromptBodyBroker(artifact_dir=artifact_dir, clock_ms=clock)
    with pytest.raises(Exception):
        restored.resolve(prompt_ref, token, run_id=RUN_A)
    restored.close()


def test_secrets_absent_from_inspected_durable_surfaces(tmp_path: Path) -> None:
    clock = _Clock()
    broker = PromptBodyBroker(
        artifact_dir=tmp_path / "broker",
        master_secret=b"leak-scan-master-secret-32b!!!!!",
        clock_ms=clock,
    )
    reference, capability = broker.deposit(
        _body(),
        run_id=RUN_A,
        enable_encrypted_artifact=True,
    )
    leaks = broker.scan_for_secrets(
        (PROMPT_CANARY, CREDENTIAL_CANARY, capability.token, PROMPT),
        extra_surfaces=(
            reference.to_dict(),
            capability.redacted_dict(),
            {"argv": ["--prompt-ref", reference.prompt_ref]},
            {"env": {"PROMPT_CID": reference.prompt_cid}},
        ),
    )
    assert leaks == ()
    # Plaintext canary is present only in the in-memory body path, not durable.
    surfaces = broker.inspect_durable_surfaces()
    blob_text = ""
    for surface in surfaces:
        if surface.get("kind") == "encrypted_blob":
            blob_text = Path(str(surface["path"])).read_bytes().decode(
                "latin-1", errors="ignore"
            )
    assert PROMPT_CANARY not in blob_text
    assert CREDENTIAL_CANARY not in blob_text
    index_path = broker.artifact_dir / "prompt_broker_index.json"
    index_text = index_path.read_text(encoding="utf-8")
    assert PROMPT_CANARY not in index_text
    assert CREDENTIAL_CANARY not in index_text
    assert capability.token not in index_text
    broker.close()


def test_bounds_and_validation_errors() -> None:
    with pytest.raises(PromptBrokerBoundsError):
        PromptBodyBroker(default_ttl_ms=0)
    with pytest.raises(PromptBrokerBoundsError):
        PromptBodyBroker(max_prompt_bytes=0)

    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock, max_prompt_bytes=32)
    with pytest.raises(PromptBrokerBoundsError):
        broker.deposit("x" * 64, run_id=RUN_A)
    with pytest.raises(PromptBrokerBoundsError):
        broker.deposit("ok", run_id=RUN_A, ttl_ms=DEFAULT_TTL_MS * 1000)
    with pytest.raises(PromptBrokerBoundsError):
        broker.deposit("", run_id=RUN_A)
    with pytest.raises(Exception):
        broker.deposit("ok", run_id="bad run id with spaces")
    broker.close()


def test_prompt_reference_round_trip_and_closed_schema() -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock)
    reference, _capability = broker.deposit(_body(), run_id=RUN_A)
    restored = PromptReference.from_dict(reference.to_dict())
    assert restored == reference
    with pytest.raises(Exception):
        PromptReference.from_dict({**reference.to_dict(), "extra": True})
    broker.close()


def test_invocation_request_uses_broker_reference_without_body_in_json() -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock)
    reference, capability = broker.deposit(_body(), run_id=RUN_A, max_uses=2)
    request = SupervisorInvocationRequest.from_prompt(
        PROMPT,
        prompt_ref=reference.prompt_ref,
    )
    assert request.prompt_cid == reference.prompt_cid
    assert request.prompt_ref == reference.prompt_ref
    assert request.transient_prompt_body == _body()
    encoded = request.to_json()
    assert PROMPT not in encoded
    assert reference.prompt_ref in encoded
    # Broker still hands exact bytes to the planner via capability.
    assert broker.resolve(reference, capability, run_id=RUN_A) == _body()
    broker.close()


def test_close_prevents_further_use() -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock)
    reference, capability = broker.deposit(_body(), run_id=RUN_A, max_uses=2)
    broker.close()
    with pytest.raises(Exception, match="closed"):
        broker.resolve(reference, capability, run_id=RUN_A)


def test_encrypted_artifact_requires_artifact_dir() -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock)
    with pytest.raises(Exception, match="artifact_dir"):
        broker.deposit(_body(), run_id=RUN_A, enable_encrypted_artifact=True)
    broker.close()


def test_master_secret_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    secret = "env-master-secret-value-32-bytes!!"
    monkeypatch.setenv("IPFS_ACCELERATE_PROMPT_BROKER_MASTER_SECRET", secret)
    artifact_dir = tmp_path / "broker"
    broker = PromptBodyBroker(artifact_dir=artifact_dir, clock_ms=clock)
    assert broker.restart_behavior()["master_secret_source"] == "caller_or_environment"
    reference, capability = broker.deposit(
        _body(),
        run_id=RUN_A,
        enable_encrypted_artifact=True,
        max_uses=2,
    )
    token = capability.token
    prompt_ref = reference.prompt_ref
    broker.close()

    restored = PromptBodyBroker(artifact_dir=artifact_dir, clock_ms=clock)
    assert restored.resolve(prompt_ref, token, run_id=RUN_A) == _body()
    restored.close()
    monkeypatch.delenv("IPFS_ACCELERATE_PROMPT_BROKER_MASTER_SECRET", raising=False)


def test_cid_mismatch_is_rejected() -> None:
    clock = _Clock()
    broker = PromptBodyBroker(clock_ms=clock)
    reference, capability = broker.deposit(_body(), run_id=RUN_A, max_uses=2)
    wrong = PromptReference(
        prompt_cid=cid_for_bytes(b"different-bytes", codec="raw"),
        prompt_ref=reference.prompt_ref,
        run_id=reference.run_id,
        byte_count=reference.byte_count,
        issued_at_ms=reference.issued_at_ms,
        expires_at_ms=reference.expires_at_ms,
        storage=reference.storage,
        purpose=reference.purpose,
    )
    with pytest.raises(PromptCapabilityError, match="prompt_cid"):
        broker.resolve(wrong, capability, run_id=RUN_A)
    broker.close()
