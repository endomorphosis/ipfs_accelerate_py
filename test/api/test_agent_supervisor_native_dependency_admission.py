from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import sysconfig
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import ed25519_did_key
from ipfs_accelerate_py.agent_supervisor.validation import (
    agent_native_dependency_admission as native,
)

from ipfs_accelerate_py import llm_router

NOW_MS = 1_777_777_777_000


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _pin(*, payload_character: str = "a") -> llm_router.AgentSupervisorNativeDependencyPin:
    machine = {"x86_64": 62, "aarch64": 183}.get(os.uname().machine)
    if machine is None:
        pytest.skip("native pin fixture supports x86_64 and aarch64")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    soabi = sysconfig.get_config_var("SOABI")
    assert isinstance(suffix, str) and isinstance(soabi, str)
    values: dict[str, object] = {
        "schema": "ipfs_accelerate_py.agent_supervisor.native-dependency-pin@1",
        "dependency_id": "",
        "module_name": "_duckdb",
        "public_alias": "duckdb",
        "distribution_name": "duckdb",
        "distribution_version": "1.5.5",
        "engine_version": "v1.5.5",
        "extension_filename": "_duckdb" + suffix,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_soabi": soabi,
        "platform_name": sys.platform,
        "platform_machine": os.uname().machine,
        "python_executable_sha256": (
            llm_router._agent_native_python_executable_sha256()
        ),
        "payload_sha256": _sha(payload_character),
        "size_bytes": 4096,
        "elf_class_bits": 64,
        "elf_endianness": sys.byteorder,
        "elf_ident_version": 1,
        "elf_osabi": 3,
        "elf_abi_version": 0,
        "elf_object_type": 3,
        "elf_machine": machine,
        "elf_object_version": 1,
        "elf_flags": 0,
        "elf_dt_needed": ["libc.so.6"],
    }
    values["dependency_id"] = llm_router._content_addressed_mapping(
        values,
        identity_field="dependency_id",
    )
    values["elf_dt_needed"] = tuple(values["elf_dt_needed"])
    pin = llm_router.AgentSupervisorNativeDependencyPin(**values)  # type: ignore[arg-type]
    return llm_router.parse_agent_supervisor_native_dependency_pin(pin.as_dict())


def _key() -> tuple[Ed25519PrivateKey, str]:
    key = Ed25519PrivateKey.generate()
    return key, ed25519_did_key(key.public_key())


def _signature(key: Ed25519PrivateKey, value: dict[str, object]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return base64.b64encode(key.sign(raw)).decode("ascii")


def _bindings(pin: llm_router.AgentSupervisorNativeDependencyPin) -> dict[str, object]:
    return {
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "source_head": "1" * 40,
        "source_tree": "2" * 40,
        "configuration_root": _sha("3"),
        "accepted_control_plane_capsule_id": _sha("4"),
        "accepted_control_plane_pin_cid": _sha("5"),
        "active_plan_root_cid": _sha("6"),
        "active_plan_revision": 7,
        "active_plan_revision_cid": _sha("7"),
        "slice_manifest_cid": _sha("8"),
        "slice_id": "slice:eaaef:0",
        "lane_id": "lane:eaaef:0",
        "lane_session_id": "session:eaaef:birth:7",
        "lane_generation": 7,
        "process_instance_id": "process:eaaef:birth:7",
        "process_birth_nonce": "birth:eaaef:7",
        "expected_process_uid": os.geteuid(),
        "expected_parent_pid": os.getpid(),
        "expected_parent_process_start_time_ticks": 12345,
        "expected_executable_sha256": pin.python_executable_sha256,
        "launch_argv_cid": _sha("9"),
    }


def _statement(
    pin: llm_router.AgentSupervisorNativeDependencyPin,
    reviewer_did: str,
    *,
    updates: dict[str, object] | None = None,
) -> dict[str, object]:
    value = {
        "schema": native.AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_SCHEMA,
        "interface": native.AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_INTERFACE,
        **_bindings(pin),
        "native_dependency_pin": pin.as_dict(),
        "native_dependency_pin_cid": pin.dependency_id,
        "sealed_descriptor_required": True,
        "ambient_loader_environment_allowed": False,
        "raw_path_authority": False,
        "launch_authority_granted": False,
        "admission_outcome": "admitted",
        "issued_at_ms": NOW_MS - 1_000,
        "expires_at_ms": NOW_MS + 60_000,
        "issuance_nonce": "nonce:native-admission:7",
        "reviewer_did": reviewer_did,
        "reviewer_role": (
            native.AGENT_SUPERVISOR_NATIVE_DEPENDENCY_ADMISSION_REVIEWER_ROLE
        ),
    }
    value.update(updates or {})
    return value


def _sealed(
    key: Ed25519PrivateKey,
    pin: llm_router.AgentSupervisorNativeDependencyPin,
    reviewer_did: str,
    *,
    updates: dict[str, object] | None = None,
    signature: str | None = None,
) -> dict[str, object]:
    statement = _statement(pin, reviewer_did, updates=updates)
    return dict(
        native.seal_agent_supervisor_native_dependency_admission(
            statement,
            reviewer_signature=signature or _signature(key, statement),
        )
    )


def _verify(
    value: dict[str, object],
    *,
    pin: llm_router.AgentSupervisorNativeDependencyPin,
    reviewer_did: str,
    bindings: dict[str, object] | None = None,
    now_ms: int = NOW_MS,
    forbidden: tuple[str, ...] = (),
) -> native.VerifiedAgentSupervisorNativeDependencyAdmission:
    return native.verify_agent_supervisor_native_dependency_admission(
        value,
        trusted_reviewer_dids=(reviewer_did,),
        expected_native_dependency_pin=pin,
        expected_bindings=bindings or _bindings(pin),
        now_ms=now_ms,
        forbidden_reviewer_dids=forbidden,
    )


def test_signed_native_admission_binds_exact_pin_and_birth_without_launch() -> None:
    pin = _pin()
    key, reviewer = _key()
    verified = _verify(_sealed(key, pin, reviewer), pin=pin, reviewer_did=reviewer)

    assert type(verified) is native.VerifiedAgentSupervisorNativeDependencyAdmission
    assert verified.native_dependency_pin == pin
    assert verified.admission_cid == verified["admission_cid"]
    assert verified["native_dependency_pin_cid"] == pin.dependency_id
    assert verified["launch_authority_granted"] is False
    assert verified["sealed_descriptor_required"] is True
    assert not any(
        name in verified
        for name in ("source_path", "descriptor", "raw_token", "callback")
    )
    with pytest.raises(TypeError, match="exact verifier"):
        native.VerifiedAgentSupervisorNativeDependencyAdmission(
            object(),
            verified,
            pin,
        )


def test_native_admission_rejects_binding_pin_and_reviewer_collisions() -> None:
    pin = _pin()
    other_pin = _pin(payload_character="b")
    key, reviewer = _key()
    value = _sealed(key, pin, reviewer)

    altered = _bindings(pin)
    altered["lane_id"] = "lane:eaaef:other"
    with pytest.raises(native.AgentSupervisorNativeDependencyAdmissionError, match="birth binding"):
        _verify(value, pin=pin, reviewer_did=reviewer, bindings=altered)
    with pytest.raises(native.AgentSupervisorNativeDependencyAdmissionError, match="exact pin"):
        _verify(value, pin=other_pin, reviewer_did=reviewer)
    with pytest.raises(native.AgentSupervisorNativeDependencyAdmissionError, match="unauthorized"):
        _verify(value, pin=pin, reviewer_did=reviewer, forbidden=(reviewer,))


def test_native_admission_rejects_invalid_signature_policy_and_lifetime() -> None:
    pin = _pin()
    key, reviewer = _key()
    invalid_signature = _sealed(key, pin, reviewer, signature=base64.b64encode(b"x" * 64).decode())
    with pytest.raises(native.AgentSupervisorNativeDependencyAdmissionError, match="signature"):
        _verify(invalid_signature, pin=pin, reviewer_did=reviewer)

    ambient = _sealed(
        key,
        pin,
        reviewer,
        updates={"ambient_loader_environment_allowed": True},
    )
    with pytest.raises(native.AgentSupervisorNativeDependencyAdmissionError, match="policy"):
        _verify(ambient, pin=pin, reviewer_did=reviewer)

    expired = _sealed(
        key,
        pin,
        reviewer,
        updates={"issued_at_ms": NOW_MS - 2_000, "expires_at_ms": NOW_MS - 1},
    )
    with pytest.raises(native.AgentSupervisorNativeDependencyAdmissionError, match="lifetime"):
        _verify(expired, pin=pin, reviewer_did=reviewer)


def test_native_admission_source_is_hash_pinned_nofollow_and_reverified(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    pin = _pin()
    key, reviewer = _key()
    value = _sealed(key, pin, reviewer)
    relative = native.agent_supervisor_native_dependency_admission_relative_path(
        str(value["source_head"]),
        str(value["active_plan_root_cid"]),
        str(value["lane_session_id"]),
        int(value["lane_generation"]),
        registry_prefix="authority/eaaef",
    )
    path = tmp_path / relative
    path.parent.mkdir(parents=True, mode=0o700)
    (tmp_path / "authority").chmod(0o700)
    path.parent.chmod(0o700)
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    path.write_bytes(raw)
    path.chmod(0o600)
    file_sha = "sha256:" + hashlib.sha256(raw).hexdigest()

    verified = native.load_and_verify_agent_supervisor_native_dependency_admission(
        tmp_path,
        source_head=str(value["source_head"]),
        active_plan_root_cid=str(value["active_plan_root_cid"]),
        lane_session_id=str(value["lane_session_id"]),
        lane_generation=int(value["lane_generation"]),
        registry_prefix="authority/eaaef",
        expected_file_sha256=file_sha,
        trusted_reviewer_dids=(reviewer,),
        expected_native_dependency_pin=pin,
        expected_bindings=_bindings(pin),
        now_ms=NOW_MS,
    )
    assert verified.reverify(now_ms=NOW_MS + 1)["admission_cid"] == value["admission_cid"]

    original = path.read_bytes()
    path.unlink()
    attacker = tmp_path / "attacker.json"
    attacker.write_bytes(original)
    attacker.chmod(0o600)
    path.symlink_to(attacker)
    with pytest.raises(native.AgentSupervisorNativeDependencyAdmissionError, match="unavailable"):
        verified.reverify(now_ms=NOW_MS + 2)


def test_native_admission_rejects_noncanonical_or_duplicate_source_json(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    pin = _pin()
    key, reviewer = _key()
    value = _sealed(key, pin, reviewer)
    relative = native.agent_supervisor_native_dependency_admission_relative_path(
        str(value["source_head"]),
        str(value["active_plan_root_cid"]),
        str(value["lane_session_id"]),
        int(value["lane_generation"]),
        registry_prefix="authority/eaaef",
    )
    path = tmp_path / relative
    path.parent.mkdir(parents=True, mode=0o700)
    (tmp_path / "authority").chmod(0o700)
    path.parent.chmod(0o700)
    raw = json.dumps(value, indent=2, sort_keys=True).encode("ascii")
    path.write_bytes(raw)
    path.chmod(0o600)
    with pytest.raises(native.AgentSupervisorNativeDependencyAdmissionError, match="canonical"):
        native.load_and_verify_agent_supervisor_native_dependency_admission(
            tmp_path,
            source_head=str(value["source_head"]),
            active_plan_root_cid=str(value["active_plan_root_cid"]),
            lane_session_id=str(value["lane_session_id"]),
            lane_generation=int(value["lane_generation"]),
            registry_prefix="authority/eaaef",
            expected_file_sha256=("sha256:" + hashlib.sha256(raw).hexdigest()),
            trusted_reviewer_dids=(reviewer,),
            expected_native_dependency_pin=pin,
            expected_bindings=_bindings(pin),
            now_ms=NOW_MS,
        )

    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    duplicate = b'{"schema":"forged",' + canonical[1:]
    path.write_bytes(duplicate)
    path.chmod(0o600)
    with pytest.raises(native.AgentSupervisorNativeDependencyAdmissionError, match="duplicate"):
        native.load_and_verify_agent_supervisor_native_dependency_admission(
            tmp_path,
            source_head=str(value["source_head"]),
            active_plan_root_cid=str(value["active_plan_root_cid"]),
            lane_session_id=str(value["lane_session_id"]),
            lane_generation=int(value["lane_generation"]),
            registry_prefix="authority/eaaef",
            expected_file_sha256=(
                "sha256:" + hashlib.sha256(duplicate).hexdigest()
            ),
            trusted_reviewer_dids=(reviewer,),
            expected_native_dependency_pin=pin,
            expected_bindings=_bindings(pin),
            now_ms=NOW_MS,
        )
