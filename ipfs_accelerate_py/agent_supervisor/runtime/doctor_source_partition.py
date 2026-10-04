"""Replay the exact signed harness support population without dropping inputs.

Only the closed public task profile is supported. Arbitrary Markdown, JSON,
validators, or code are never classified by extension or caller nomination.
All input hashes remain in the Doctor ledger. This classification grants no
proof, execution, or completion authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat

from ..proof.formal_verification_contracts import canonical_json, content_identity
from . import local_planning_admission as local
from .terminal_task_profile import (
    INSTRUCTION, PROFILE, SMOKE, task_profile_bytes, task_profile_smoke,
    task_profile_spec, task_profile_worker_inputs, validate_task_profile, normalized_instruction,
)


class DoctorSourcePartitionError(ValueError):
    """A harness role cannot be independently replayed from signed inputs."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DoctorSourcePartitionError("duplicate task profile key")
        result[key] = value
    return result


def _read(root: Path, name: str, sources: dict, limit: int) -> bytes:
    path = root / name
    if (path.is_symlink() or path.resolve(strict=True) != path or not path.is_file()
            or path.stat().st_size > limit):
        raise DoctorSourcePartitionError("bounded regular signed support file required")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise DoctorSourcePartitionError("bounded regular signed support file required")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise DoctorSourcePartitionError("harness support changed during verification")
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != sources[name]["sha256"]:
        raise DoctorSourcePartitionError("harness support differs from signed source")
    return raw


@dataclass(frozen=True)
class DoctorSourcePartition:
    """Retain signed admission so every consumer can rederive the partition."""

    admission_json: str = field(repr=False)
    task_cid: str
    manifest_cid: str
    profile_sha256: str
    program_paths: tuple[str, ...]
    support_hashes: tuple[tuple[str, str, str], ...]

    def observation(self) -> dict:
        payload = {
            "schema": "doctor-terminal-source-partition@1",
            "manifest_cid": self.manifest_cid, "task_cid": self.task_cid,
            "profile_sha256": self.profile_sha256,
            "program_paths": list(self.program_paths),
            "harness_support": [{"path": path, "role": role, "sha256": digest}
                                for path, role, digest in self.support_hashes],
            "execution_authority": False, "proof_authority": False,
            "completion_authority": False,
        }
        return {**payload, "partition_cid": content_identity(payload)}

    def assert_current(self, repository: Path) -> None:
        if type(self.admission_json) is not str or len(self.admission_json.encode()) > 4_194_304:
            raise DoctorSourcePartitionError("bounded signed partition admission required")
        expected = terminal_doctor_source_partition(
            repository=repository, admission=json.loads(self.admission_json), task_cid=self.task_cid)
        if expected is None or self != expected:
            raise DoctorSourcePartitionError("Doctor source partition does not replay")


def terminal_doctor_source_partition(
    *, repository: Path, admission: dict, task_cid: str,
) -> DoctorSourcePartition | None:
    """Recognize exactly generated support, preserving all actual task inputs.

    Without the selected profile, callers retain the old complete AST gate.
    A malformed selected profile is a hard refusal, never an ignore list.
    """
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    manifest = verified["manifest"]
    root = Path(repository).absolute()
    if root.resolve(strict=True) != root or str(root) != manifest["repository"]:
        raise DoctorSourcePartitionError("partition repository differs from admission")
    sources = manifest["sources"]
    if PROFILE not in sources:
        return None
    selected = [task for task in verified["graph"].tasks if task.task_cid == task_cid]
    if len(selected) != 1 or len(verified["graph"].tasks) != 1 or len(manifest["tasks"]) != 1:
        raise DoctorSourcePartitionError("partition requires the single signed public task")
    if not {INSTRUCTION, SMOKE} <= set(sources):
        raise DoctorSourcePartitionError("signed harness support is incomplete")
    raw = _read(root, PROFILE, sources, 65536)
    instruction = _read(root, INSTRUCTION, sources, 32768).decode("utf-8")
    if not instruction.strip() or normalized_instruction(instruction) != instruction:
        raise DoctorSourcePartitionError("canonical nonempty public instruction required")
    try:
        profile = validate_task_profile(json.loads(raw, object_pairs_hook=_unique_pairs), instruction=instruction)
    except (ValueError, TypeError) as error:
        raise DoctorSourcePartitionError("signed public task profile is invalid") from error
    if raw != task_profile_bytes(profile):
        raise DoctorSourcePartitionError("task profile is not the canonical producer output")
    if set(sources) != set(task_profile_worker_inputs(profile)):
        raise DoctorSourcePartitionError("partition omits or adds signed source inputs")
    expected_spec = task_profile_spec(profile, policy_cid=content_identity(local.LOCAL_POLICY))
    if manifest["tasks"][0] != expected_spec or selected[0].task_key != expected_spec["task_key"]:
        raise DoctorSourcePartitionError("profile differs from independently signed task effects or checks")
    expected_smoke = task_profile_smoke(profile).encode("utf-8")
    if _read(root, SMOKE, sources, len(expected_smoke)) != expected_smoke:
        raise DoctorSourcePartitionError("structural smoke differs from the fixed producer")
    return DoctorSourcePartition(
        admission_json=canonical_json(admission), task_cid=task_cid,
        manifest_cid=verified["receipt"]["manifest_cid"],
        profile_sha256=hashlib.sha256(raw).hexdigest(),
        program_paths=tuple(profile["input_paths"]),
        support_hashes=tuple((name, role, sources[name]["sha256"]) for name, role in (
            (INSTRUCTION, "instruction"), (PROFILE, "task_profile"), (SMOKE, "structural_smoke"))),
    )
