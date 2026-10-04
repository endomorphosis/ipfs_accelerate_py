"""Replay the exact signed harness support population without dropping inputs.

Only the closed public task profile is supported. Arbitrary Markdown, JSON,
validators, or code are never classified by extension or caller nomination.
All input hashes remain in the Doctor ledger. This classification grants no
proof, execution, or completion authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from ..proof.formal_verification_contracts import canonical_json, content_identity
from . import local_planning_admission as local
from .terminal_task_profile import PROFILE
from .terminal_source_partition import (terminal_profile_partition,
    TerminalSourcePartitionError as DoctorSourcePartitionError)


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
    partition = terminal_profile_partition(repository=root, manifest=manifest)
    if partition is None or selected[0].task_key != manifest["tasks"][0]["task_key"]:
        raise DoctorSourcePartitionError("profile differs from signed graph task")
    return DoctorSourcePartition(
        admission_json=canonical_json(admission), task_cid=task_cid,
        manifest_cid=verified["receipt"]["manifest_cid"],
        profile_sha256=partition.profile_sha256,
        program_paths=partition.program_paths, support_hashes=partition.support_hashes,
    )
