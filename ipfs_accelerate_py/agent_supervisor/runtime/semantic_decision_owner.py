"""Materialize a task/parent-bound policy at the owner-to-worker handoff.

The signed owner command independently nominates a complete capsule partition
and task key. This module authenticates the actual compiled stdin before a
worker starts, then publishes only a read-only launch descriptor. It performs
no provider call and grants no completion or source mutation authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

from . import semantic_router_translation as codec
from .doctor_candidate_runner import _directory
from .local_prompt_tokenizer import _bounded_local_bytes, _json_object, _local_path, _sha
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES
from .semantic_decision_packet import SemanticDecisionPolicy
from .semantic_decision_worker import build_decision_launch_policy

NOMINATION_SCHEMA = "supervisor-semantic-decision-nomination@1"
NOMINATION_SCHEMAS = frozenset({NOMINATION_SCHEMA, "supervisor-semantic-decision-nomination@2"})
SCHEMA = "supervisor-semantic-decision-owner-binding@1"
MAX_NOMINATION_BYTES = 256_000
_TYPED_MODE_FLAG = "--semantic-decision-typed-nomination"
_OWNER_FLAGS = frozenset({
    "--semantic-decision-nomination-artifact", "--semantic-decision-nomination-sha256",
    "--semantic-decision-task-id",
})
_RETRIEVAL_FLAGS = frozenset({
    "--semantic-decision-context-bundle-artifact", "--semantic-decision-context-bundle-sha256",
    "--semantic-decision-task-cid",
})
_BLOCKED_FLAGS = frozenset({
    "--semantic-decision-policy-artifact", "--semantic-decision-policy-sha256",
    "--preflight", "--preflight-sleep", "--cleanup", "--validate",
    "--semantic-slot-policy-revision", "--semantic-context-projection",
    "--doctor-candidate-artifact", "--doctor-candidate-sha256", "--doctor-task-cid",
    "--doctor-contract-artifact", "--doctor-contract-sha256", "--doctor-contract-task-cid",
    "--finite-repository-artifact", "--finite-repository-sha256", "--finite-repository-task-cid",
    "--finite-proof-query-context", "--finite-proof-query-sha256", "--finite-proof-query-context-cid",
})


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_directory(path: Path, name: str) -> Path:
    selected = _local_path(path)
    if selected.resolve(strict=True) != selected or not selected.is_dir():
        raise ValueError(f"canonical existing {name} directory required")
    return selected


def build_decision_nomination(policy: SemanticDecisionPolicy) -> dict:
    """Serialize explicit owner declarations without inferring optionality."""
    if type(policy) is not SemanticDecisionPolicy:
        raise ValueError("independent typed semantic decision policy required")
    value = {"schema": "supervisor-semantic-decision-nomination@" + policy.schema_revision,
             "policy": policy.to_dict()}
    if len(_json(value)) > MAX_NOMINATION_BYTES:
        raise ValueError("semantic decision nomination exceeds its byte bound")
    return value


def load_decision_nomination(
    *, artifact: Path, expected_sha256: str, workspace: Path,
) -> SemanticDecisionPolicy:
    expected = _sha(expected_sha256, "semantic decision nomination")
    work = _canonical_directory(workspace, "allocated workspace")
    path = _local_path(artifact)
    if path.resolve(strict=True) != path or path.is_relative_to(work):
        raise ValueError("semantic decision nomination must be canonical and outside the candidate")
    raw = _bounded_local_bytes(path, maximum=MAX_NOMINATION_BYTES, expected_sha256=expected)
    value = _json_object(raw, "semantic decision nomination")
    if (set(value) != {"schema", "policy"} or type(value.get("schema")) is not str
            or value["schema"] not in NOMINATION_SCHEMAS):
        raise ValueError("exact semantic decision nomination schema and fields required")
    policy = SemanticDecisionPolicy.from_dict(value["policy"])
    if value["schema"] != "supervisor-semantic-decision-nomination@" + policy.schema_revision:
        raise ValueError("semantic decision nomination schema and policy revision differ")
    return policy


@dataclass(frozen=True)
class SemanticDecisionOwnerDispatch:
    worker_arguments: tuple[str, ...]
    prompt_bytes: bytes
    receipt_json: str

    @property
    def receipt(self) -> dict:
        return json.loads(self.receipt_json)


def _literal_arguments(arguments):
    if type(arguments) not in {tuple, list} or not 1 <= len(arguments) <= 128:
        raise ValueError("bounded literal owner worker arguments required")
    for part in arguments:
        if (type(part) is not str or not part or len(part.encode()) > 8192
                or any(char in part for char in ("\x00", "\n", "\r"))):
            raise ValueError("bounded literal owner worker argument required")
    owner, remaining, index = {}, [], 0
    while index < len(arguments):
        part = arguments[index]
        flag = part.split("=", 1)[0]
        if flag in _BLOCKED_FLAGS:
            raise ValueError("semantic decision nomination requires exclusive model coding")
        if flag in _OWNER_FLAGS | _RETRIEVAL_FLAGS | {_TYPED_MODE_FLAG}:
            if part != flag or flag in owner or index + 1 >= len(arguments):
                raise ValueError("owner nomination flags require one literal two-token binding each")
            value = arguments[index + 1]
            if value.startswith("--"):
                raise ValueError("owner nomination flag requires a literal value")
            owner[flag] = value
            index += 2
        else:
            remaining.append(part)
            index += 1
    if not _OWNER_FLAGS <= set(owner) or set(owner) - _OWNER_FLAGS - _RETRIEVAL_FLAGS - {_TYPED_MODE_FLAG}:
        raise ValueError("complete independent owner nomination and task binding required")
    if owner.get(_TYPED_MODE_FLAG, "disabled") not in {"enabled", "disabled"}:
        raise ValueError("literal enabled or disabled typed nomination mode required")
    retrieval = set(owner) & _RETRIEVAL_FLAGS
    if retrieval and retrieval != _RETRIEVAL_FLAGS:
        raise ValueError("complete signed retrieval context bundle and admitted task binding required")
    return owner, remaining


def _argument(arguments: list[str], flag: str, *, default=None):
    matches = [index for index, part in enumerate(arguments) if part.split("=", 1)[0] == flag]
    if not matches and default is not None:
        return default
    if len(matches) != 1:
        raise ValueError(f"one literal {flag} binding required")
    index = matches[0]
    if (arguments[index] != flag or index + 1 == len(arguments)
            or arguments[index + 1].startswith("--")):
        raise ValueError(f"literal two-token {flag} binding required")
    return arguments[index + 1]


def _owned_directory(descriptor: int, *, exact_mode: int | None = None):
    info = os.fstat(descriptor)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o022
            or (exact_mode is not None and stat.S_IMODE(info.st_mode) != exact_mode)):
        raise ValueError("owner-controlled non-writable policy directory required")


def _child_directory(parent: int, name: str) -> int:
    created = False
    try:
        os.mkdir(name, 0o755, dir_fd=parent)
        created = True
    except FileExistsError:
        pass
    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
    try:
        if created:
            os.fchmod(child, 0o755)
            os.fsync(child)
            os.fsync(parent)
        _owned_directory(child, exact_mode=0o755)
        return child
    except BaseException:
        os.close(child)
        raise


def _immutable_file(parent: int, name: str, expected: bytes, *, maximum: int = MAX_NOMINATION_BYTES):
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_size != len(expected)):
            raise ValueError("exact owner-controlled immutable policy file required")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink", "st_uid")
        if raw != expected or any(getattr(before, field) != getattr(after, field) for field in fields):
            raise ValueError("immutable semantic decision policy differs or changed")
    finally:
        os.close(descriptor)


def _publish(repository: Path, workspace: Path, payload: dict, *,
             directory: str = "semantic-decision-policies",
             maximum: int = MAX_NOMINATION_BYTES) -> tuple[Path, str]:
    limits = {"semantic-decision-policies": MAX_NOMINATION_BYTES,
              "semantic-decision-nominations": MAX_NOMINATION_BYTES,
              "semantic-typed-nominations": 2_000_000}
    if directory not in limits or type(maximum) is not int or not 1 <= maximum <= limits[directory]:
        raise ValueError("closed bounded semantic owner publication namespace required")
    raw = _json(payload)
    if not 1 <= len(raw) <= maximum:
        raise ValueError("bounded semantic decision launch descriptor required")
    digest = _digest(raw)
    path = repository / ".runtime" / directory / (digest + ".json")
    if path.is_relative_to(workspace):
        raise ValueError("published semantic decision policy must remain outside candidate")
    root_fd = _directory(repository)
    runtime_fd = policy_fd = lock_fd = None
    temporary = ".pending-" + uuid.uuid4().hex
    temporary_created = False
    try:
        _owned_directory(root_fd)
        runtime_fd = _child_directory(root_fd, ".runtime")
        lock_fd = os.open(
            ".semantic-decision-publish.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            0o600, dir_fd=runtime_fd,
        )
        lock_info = os.fstat(lock_fd)
        if (not stat.S_ISREG(lock_info.st_mode) or lock_info.st_uid != os.geteuid()
                or stat.S_IMODE(lock_info.st_mode) != 0o600 or lock_info.st_nlink != 1
                or lock_info.st_size != 0):
            raise ValueError("owner-only regular single-link empty publication lock required")
        # The final hardlink briefly shares its inode with the pending name.
        # Serialize publishers on an owner-only file; a worker must not be
        # able to acquire this advisory lock through a readable directory fd.
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        policy_fd = _child_directory(runtime_fd, directory)
        try:
            _immutable_file(policy_fd, path.name, raw, maximum=maximum)
        except FileNotFoundError:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600, dir_fd=policy_fd,
            )
            temporary_created = True
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(raw)
                    stream.flush()
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.link(temporary, path.name, src_dir_fd=policy_fd, dst_dir_fd=policy_fd,
                        follow_symlinks=False)
            except FileExistsError:
                pass
            os.unlink(temporary, dir_fd=policy_fd)
            temporary_created = False
            os.fsync(policy_fd)
            _immutable_file(policy_fd, path.name, raw, maximum=maximum)
        # Bind the returned public path too; parent directory replacement may
        # never make the authenticated fd name a different launch artifact.
        if _bounded_local_bytes(path, maximum=maximum, expected_sha256=digest) != raw:
            raise ValueError("published semantic decision policy path differs")
        return path, digest
    finally:
        try:
            if temporary_created and policy_fd is not None:
                os.unlink(temporary, dir_fd=policy_fd)
        finally:
            for descriptor in (policy_fd, lock_fd, runtime_fd, root_fd):
                if descriptor is not None:
                    os.close(descriptor)


def prepare_decision_owner_dispatch(
    *, arguments: tuple[str, ...] | list[str], prompt_bytes: bytes, workspace: Path,
) -> SemanticDecisionOwnerDispatch:
    """Authenticate final stdin and publish its independently nominated policy."""
    owner, worker = _literal_arguments(arguments)
    task = owner["--semantic-decision-task-id"]
    if (not task or task != task.strip() or len(task.encode()) > 512
            or any(char in task for char in ("\x00", "\n", "\r"))):
        raise ValueError("bounded explicit signed task key required")
    work = _canonical_directory(workspace, "allocated workspace")
    repository = _canonical_directory(Path(_argument(worker, "--semantic-repository")), "semantic repository")
    if repository == work or repository.is_relative_to(work):
        raise ValueError("semantic source repository must remain outside allocated candidate")
    if _argument(worker, "--purpose", default="coding") != "coding":
        raise ValueError("semantic decision nomination requires coding purpose")
    if type(prompt_bytes) is not bytes or not 1 <= len(prompt_bytes) <= MAX_MODEL_PROMPT_BYTES:
        raise ValueError("bounded nonempty native worker stdin bytes required")
    prompt = prompt_bytes.decode("utf-8", errors="strict")
    if not prompt.strip():
        raise ValueError("nonempty native compiled worker prompt required")
    wire, _, _, _, parent_text, parent_artifact = codec._prompt_parts(prompt)
    if wire.get("objective_id") != task:
        raise ValueError("native compiled prompt differs from signed owner task key")
    nomination_artifact = _local_path(owner["--semantic-decision-nomination-artifact"])
    nomination_sha = _sha(owner["--semantic-decision-nomination-sha256"], "semantic decision nomination")
    policy = load_decision_nomination(
        artifact=nomination_artifact, expected_sha256=nomination_sha, workspace=work,
    )
    retrieval_binding = None
    if set(owner) & _RETRIEVAL_FLAGS:
        from .task_context_bundle import load_task_context_nomination
        from .semantic_retrieval_query_reference import SemanticRetrievalQueryBinding
        if policy.task_text_source_path is None:
            raise ValueError("retrieval query reference requires a nominated required task text source")
        task_cid = owner["--semantic-decision-task-cid"]
        if (not task_cid.strip() or task_cid != task_cid.strip() or len(task_cid.encode()) > 512):
            raise ValueError("bounded explicit admitted task CID required")
        bundle_artifact = owner["--semantic-decision-context-bundle-artifact"]
        bundle_sha = _sha(owner["--semantic-decision-context-bundle-sha256"], "task context bundle")
        bundle_path = repository / bundle_artifact
        if bundle_path.resolve(strict=True).is_relative_to(work):
            raise ValueError("owner context bundle must remain outside allocated candidate")
        metadata = load_task_context_nomination(repository=repository, artifact=bundle_artifact,
            expected_sha256=bundle_sha, task_cid=task_cid, task_id=task)
        if (metadata.get("semantic context artifact") != parent_artifact
                or metadata.get("semantic context sha256") != _digest(parent_text.encode())):
            raise ValueError("signed context bundle differs from native semantic parent")
        retrieval_binding = SemanticRetrievalQueryBinding(
            artifact=metadata.get("code retrieval artifact"),
            expected_sha256=metadata.get("code retrieval sha256"), task_id=task)
        if (repository / retrieval_binding.artifact).resolve(strict=True).is_relative_to(work):
            raise ValueError("owner retrieval parent must remain outside allocated candidate")
    typed_commit = None
    if owner.get(_TYPED_MODE_FLAG, "disabled") == "enabled":
        from .semantic_typed_owner_policy import build_typed_owner_decision_nomination
        from .semantic_typed_nomination_lineage import (
            BINDING_SCHEMA, LINEAGE_SCHEMA, RETRIEVAL_LINEAGE_SCHEMA, MAX_LINEAGE_BYTES, owner_typed_graph_root,
        )
        origin_raw = _bounded_local_bytes(nomination_artifact, maximum=MAX_NOMINATION_BYTES,
                                          expected_sha256=nomination_sha)
        parent = codec._parse(parent_text)
        typed = build_typed_owner_decision_nomination(
            native_prompt=prompt, repository=repository, workspace=work,
            nomination_artifact=nomination_artifact, nomination_sha256=nomination_sha,
            task_id=task, expected_parent_sha256=codec._sha(parent_text),
            expected_semantic_root_cid=parent["semantic_root_cid"],
            graph_root_id=owner_typed_graph_root(task_id=task, parent_sha256=codec._sha(parent_text)),
            **({"retrieval_query_binding": retrieval_binding} if retrieval_binding is not None else {}))
        lineage = {"schema": RETRIEVAL_LINEAGE_SCHEMA if retrieval_binding is not None else LINEAGE_SCHEMA,
                   "origin_nomination_text": origin_raw.decode("utf-8"),
                   "typed_nomination": typed.to_dict()}
        # Check the complete sidecar before the first immutable commit.
        if len(_json(lineage)) > MAX_LINEAGE_BYTES:
            raise ValueError("complete typed nomination lineage exceeds its byte bound")
        derived_artifact, derived_sha = _publish(repository, work, typed.nomination,
            directory="semantic-decision-nominations")
        lineage_artifact, lineage_sha = _publish(repository, work, lineage,
            directory="semantic-typed-nominations", maximum=MAX_LINEAGE_BYTES)
        descriptor, policy = typed.launch_policy, typed.policy
        typed_commit = {"schema": BINDING_SCHEMA,
            "derived_nomination_artifact": str(derived_artifact), "derived_nomination_sha256": derived_sha,
            "lineage_artifact": str(lineage_artifact), "lineage_sha256": lineage_sha}
    else:
        descriptor = build_decision_launch_policy(native_prompt=prompt, policy=policy, repository=repository,
            **({"retrieval_query_binding": retrieval_binding} if retrieval_binding is not None else {}))
    artifact, digest = _publish(repository, work, descriptor)
    worker.extend(["--semantic-decision-policy-artifact", str(artifact),
                   "--semantic-decision-policy-sha256", digest])
    receipt = {
        "schema": SCHEMA, "binding_id": uuid.uuid4().hex, "task_id": task,
        "native_prompt_sha256": _digest(prompt_bytes), "native_prompt_bytes": len(prompt_bytes),
        "parent_artifact": descriptor["parent_artifact"], "parent_sha256": descriptor["parent_sha256"],
        "launch_policy_artifact": str(artifact), "launch_policy_sha256": digest,
        "nomination_artifact": str(nomination_artifact), "nomination_sha256": nomination_sha,
        "policy_sha256": _digest(_json(policy.to_dict())),
        "provider_called": False, "completion_authority": False,
    }
    if retrieval_binding is not None:
        # Publication must not outlive the task nomination or either native
        # parent. Rebuild with the independently held pair, never current-path
        # discovery, and keep the extra custody in the pinned descriptor.
        final_metadata = load_task_context_nomination(repository=repository, artifact=bundle_artifact,
            expected_sha256=bundle_sha, task_cid=task_cid, task_id=task)
        final = build_decision_launch_policy(native_prompt=prompt, policy=policy, repository=repository,
            retrieval_query_binding=retrieval_binding)
        load_decision_nomination(artifact=nomination_artifact, expected_sha256=nomination_sha, workspace=work)
        if final_metadata != metadata or final != descriptor:
            raise ValueError("signed retrieval bundle or native parents changed during publication")
    if typed_commit is not None:
        receipt["typed_nomination"] = typed_commit
        from .semantic_typed_nomination_lineage import load_and_verify_typed_nomination_lineage
        load_and_verify_typed_nomination_lineage(
            artifact=Path(typed_commit["lineage_artifact"]), expected_sha256=typed_commit["lineage_sha256"],
            native_prompt=prompt, repository=repository, workspace=work,
            expected_origin_nomination_sha256=nomination_sha, expected_task_id=task,
            owner_binding=receipt, require_current_source=True)
        # Publication can interleave with owner/source edits. Never return a
        # dispatch until the original independent file and actual stdin close.
        load_decision_nomination(artifact=nomination_artifact, expected_sha256=nomination_sha, workspace=work)
        final, _ = codec._verified_semantic(repository, artifact=parent_artifact,
            text=parent_text, task_id=task, current=True)
        if codec._json(final) != codec._json(codec._parse(parent_text)) or codec._read(repository, parent_artifact) != parent_text.encode():
            raise ValueError("typed owner source/parent changed during publication")
    if len(_json(receipt)) > 65_536:
        raise ValueError("compact semantic owner binding exceeds its byte bound")
    return SemanticDecisionOwnerDispatch(tuple(worker), prompt_bytes, _json(receipt).decode())
