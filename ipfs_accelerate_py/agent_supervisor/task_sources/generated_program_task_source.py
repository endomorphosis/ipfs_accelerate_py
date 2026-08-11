"""Generated-program authority and scheduler observation (ASE3-025).

DuckDB owns the authoritative program revision before Markdown/IPLD
projections. The configured runtime observes that revision through a
revision-fenced :class:`GeneratedProgramSourceObserver` without requiring a
Git-tracked board.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .duckdb_state import connect_duckdb_with_policy
from .duckdb_task_source import DuckDBTaskSource, duckdb_available

GENERATED_PROGRAM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authoritative-program-revision@1"
)
GENERATED_BOARD_PROFILE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/generated-board-runtime-profile@1"
)
GENERATED_BOARD_EXECUTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/generated-board-execution-receipt@1"
)
_SHA_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{16,}|^b[a-z2-7]{20,}$")


class GeneratedProgramError(RuntimeError):
    """Raised when generated-program authority is missing or inconsistent."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class EmbeddedTaskIdentity:
    """Task identity that must survive DuckDB/write/observe/claim/receipt."""

    task_cid: str
    task_key: str
    goal_cid: str
    subgoal_owner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_cid": self.task_cid,
            "task_key": self.task_key,
            "goal_cid": self.goal_cid,
            "subgoal_owner": self.subgoal_owner,
        }


@dataclass(frozen=True)
class CanonicalSupervisorProgram:
    """Complete canonical program record for generated boards."""

    plan_root_cid: str
    root_goal_cid: str
    goal_cids: tuple[str, ...]
    task_identities: tuple[EmbeddedTaskIdentity, ...]
    repository_tree_id: str
    namespace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_root_cid": self.plan_root_cid,
            "root_goal_cid": self.root_goal_cid,
            "goal_cids": list(self.goal_cids),
            "task_identities": [item.to_dict() for item in self.task_identities],
            "repository_tree_id": self.repository_tree_id,
            "namespace": self.namespace,
        }

    @property
    def content_id(self) -> str:
        return _cid(self.to_dict())


@dataclass(frozen=True)
class AuthoritativeProgramRevision:
    """DuckDB-first authoritative program revision."""

    schema: str
    plan_root_cid: str
    revision: int
    projection_cid: str
    program_cid: str
    root_goal_cid: str
    goal_cids: tuple[str, ...]
    task_identities: tuple[EmbeddedTaskIdentity, ...]
    repository_tree_id: str
    namespace: str
    committed_at_ms: int
    fence_token: str

    @property
    def task_count(self) -> int:
        return len(self.task_identities)

    @property
    def goal_count(self) -> int:
        return len(self.goal_cids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_root_cid": self.plan_root_cid,
            "revision": self.revision,
            "projection_cid": self.projection_cid,
            "program_cid": self.program_cid,
            "root_goal_cid": self.root_goal_cid,
            "goal_cids": list(self.goal_cids),
            "task_identities": [item.to_dict() for item in self.task_identities],
            "repository_tree_id": self.repository_tree_id,
            "namespace": self.namespace,
            "committed_at_ms": self.committed_at_ms,
            "fence_token": self.fence_token,
            "task_count": self.task_count,
            "goal_count": self.goal_count,
        }

    @property
    def content_id(self) -> str:
        return _cid(self.to_dict())


@dataclass(frozen=True)
class GeneratedBoardRuntimeProfile:
    """Namespace-independent per-run runtime profile from observed revision."""

    schema: str
    namespace: str
    plan_root_cid: str
    revision: int
    projection_cid: str
    repository_tree_id: str
    task_cids: tuple[str, ...]
    signed_bounds_cid: str
    profile_cid: str = ""

    def __post_init__(self) -> None:
        if self.schema != GENERATED_BOARD_PROFILE_SCHEMA:
            raise GeneratedProgramError("unsupported runtime profile schema")
        ns = str(self.namespace or "").strip()
        # ASE3 seed identifiers are not required; arbitrary valid namespaces
        # must work. Reject only the empty namespace.
        if not ns:
            raise GeneratedProgramError("namespace is required")
        payload = {
            "schema": self.schema,
            "namespace": self.namespace,
            "plan_root_cid": self.plan_root_cid,
            "revision": self.revision,
            "projection_cid": self.projection_cid,
            "repository_tree_id": self.repository_tree_id,
            "task_cids": list(self.task_cids),
            "signed_bounds_cid": self.signed_bounds_cid,
        }
        expected = _cid(payload)
        if self.profile_cid and self.profile_cid != expected:
            raise GeneratedProgramError("runtime profile content drifted")
        object.__setattr__(self, "profile_cid", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "namespace": self.namespace,
            "plan_root_cid": self.plan_root_cid,
            "revision": self.revision,
            "projection_cid": self.projection_cid,
            "repository_tree_id": self.repository_tree_id,
            "task_cids": list(self.task_cids),
            "signed_bounds_cid": self.signed_bounds_cid,
            "profile_cid": self.profile_cid,
        }


@dataclass(frozen=True)
class GeneratedBoardExecutionReceipt:
    """Receipt for genuine generated-board runtime observation/execution."""

    schema: str
    plan_root_cid: str
    revision: int
    profile_cid: str
    scheduler_argv: tuple[str, ...]
    supervisor_argv: tuple[str, ...]
    daemon_argv: tuple[str, ...]
    observed_task_cids: tuple[str, ...]
    planner_invocations: int
    terminal: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_root_cid": self.plan_root_cid,
            "revision": self.revision,
            "profile_cid": self.profile_cid,
            "scheduler_argv": list(self.scheduler_argv),
            "supervisor_argv": list(self.supervisor_argv),
            "daemon_argv": list(self.daemon_argv),
            "observed_task_cids": list(self.observed_task_cids),
            "planner_invocations": self.planner_invocations,
            "terminal": self.terminal,
            "reason_codes": list(self.reason_codes),
        }

    @property
    def content_id(self) -> str:
        return _cid(self.to_dict())


def _duckdb_module() -> Any:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise GeneratedProgramError("duckdb is required") from exc
    return duckdb



def _authority_sidecar_path(database_path: Path) -> Path:
    return Path(database_path).with_name(Path(database_path).name + ".authority.json")


def commit_authoritative_program_revision(
    database_path: str | Path,
    *,
    plan_root_cid: str,
    revision: int,
    projection_cid: str,
    task_cids: Sequence[str],
    goal_cids: Sequence[str],
    root_goal_cid: str,
    goal_by_task: Mapping[str, str],
    repository_tree_id: str,
    namespace: str,
    now_ms: int | None = None,
) -> AuthoritativeProgramRevision:
    """CAS the authoritative program revision next to the DuckDB source.

    Authority is deliberately *not* stored as an extra DuckDB table so the
    closed task-source schema inventory remains exact. The sidecar is a
    regular private file bound to the DuckDB path and revision fence.
    """

    path = Path(database_path)
    if not path.exists():
        raise GeneratedProgramError(
            "DuckDB task source must exist before authority commit"
        )
    if path.is_symlink() or path.is_dir():
        raise GeneratedProgramError("DuckDB task source path is invalid")

    identities = tuple(
        EmbeddedTaskIdentity(
            task_cid=str(task_cid),
            task_key=str(task_cid),
            goal_cid=str(goal_by_task.get(task_cid) or root_goal_cid),
            subgoal_owner=str(goal_by_task.get(task_cid) or root_goal_cid),
        )
        for task_cid in task_cids
    )
    program = CanonicalSupervisorProgram(
        plan_root_cid=str(plan_root_cid),
        root_goal_cid=str(root_goal_cid),
        goal_cids=tuple(str(item) for item in goal_cids),
        task_identities=identities,
        repository_tree_id=str(repository_tree_id),
        namespace=str(namespace or "generated"),
    )
    committed_at = int(now_ms if now_ms is not None else _now_ms())
    fence = hashlib.sha256(
        f"{plan_root_cid}:{revision}:{projection_cid}:{committed_at}".encode("utf-8")
    ).hexdigest()
    revision_record = AuthoritativeProgramRevision(
        schema=GENERATED_PROGRAM_SCHEMA,
        plan_root_cid=program.plan_root_cid,
        revision=int(revision),
        projection_cid=str(projection_cid),
        program_cid=program.content_id,
        root_goal_cid=program.root_goal_cid,
        goal_cids=program.goal_cids,
        task_identities=program.task_identities,
        repository_tree_id=program.repository_tree_id,
        namespace=program.namespace,
        committed_at_ms=committed_at,
        fence_token=fence,
    )

    sidecar = _authority_sidecar_path(path)
    if sidecar.exists():
        if sidecar.is_symlink():
            raise GeneratedProgramError("authority sidecar must not be a symlink")
        try:
            existing_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GeneratedProgramError("authority sidecar is torn") from exc
        existing_rev = int(existing_payload.get("revision") or 0)
        existing_root = str(existing_payload.get("plan_root_cid") or "")
        if existing_root == revision_record.plan_root_cid:
            if existing_rev == revision_record.revision:
                return AuthoritativeProgramRevision(
                    schema=str(
                        existing_payload.get("schema") or GENERATED_PROGRAM_SCHEMA
                    ),
                    plan_root_cid=str(existing_payload["plan_root_cid"]),
                    revision=int(existing_payload["revision"]),
                    projection_cid=str(existing_payload["projection_cid"]),
                    program_cid=str(existing_payload["program_cid"]),
                    root_goal_cid=str(existing_payload.get("root_goal_cid") or ""),
                    goal_cids=tuple(existing_payload.get("goal_cids") or ()),
                    task_identities=tuple(
                        EmbeddedTaskIdentity(**item)
                        for item in existing_payload.get("task_identities") or ()
                    ),
                    repository_tree_id=str(
                        existing_payload.get("repository_tree_id") or ""
                    ),
                    namespace=str(existing_payload.get("namespace") or "generated"),
                    committed_at_ms=int(existing_payload.get("committed_at_ms") or 0),
                    fence_token=str(existing_payload.get("fence_token") or ""),
                )
            if existing_rev > revision_record.revision:
                raise GeneratedProgramError("authoritative revision is stale")

    payload = json.dumps(revision_record.to_dict(), sort_keys=True, indent=2)
    tmp = sidecar.with_suffix(".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(tmp, flags, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, sidecar)
    try:
        os.chmod(sidecar, 0o600)
    except OSError:
        pass
    return revision_record


class GeneratedProgramSourceObserver:
    """Revision-fenced observer over the authoritative DuckDB program source."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def observe(self) -> AuthoritativeProgramRevision:
        if not self.database_path.exists():
            raise GeneratedProgramError("generated program database is absent")
        if self.database_path.is_symlink():
            raise GeneratedProgramError(
                "generated program database must not be a symlink"
            )
        sidecar = _authority_sidecar_path(self.database_path)
        if sidecar.exists():
            if sidecar.is_symlink():
                raise GeneratedProgramError(
                    "authority sidecar must not be a symlink"
                )
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise GeneratedProgramError("authority sidecar is torn") from exc
            identities = tuple(
                EmbeddedTaskIdentity(**item)
                for item in payload.get("task_identities") or ()
            )
            # Cross-check DuckDB projection still opens under policy.
            duckdb = _duckdb_module()
            connection = connect_duckdb_with_policy(
                duckdb,
                self.database_path,
                read_only=True,
                configuration={"threads": 1, "memory_limit": "256MB"},
            )
            connection.close()
            return AuthoritativeProgramRevision(
                schema=str(payload.get("schema") or GENERATED_PROGRAM_SCHEMA),
                plan_root_cid=str(payload["plan_root_cid"]),
                revision=int(payload["revision"]),
                projection_cid=str(payload["projection_cid"]),
                program_cid=str(payload["program_cid"]),
                root_goal_cid=str(payload.get("root_goal_cid") or ""),
                goal_cids=tuple(payload.get("goal_cids") or ()),
                task_identities=identities,
                repository_tree_id=str(payload.get("repository_tree_id") or ""),
                namespace=str(payload.get("namespace") or "generated"),
                committed_at_ms=int(payload.get("committed_at_ms") or 0),
                fence_token=str(payload.get("fence_token") or ""),
            )

        # Fall back to DuckDBTaskSource snapshot metadata when no sidecar.
        source = DuckDBTaskSource(self.database_path)
        snapshot = source.snapshot()
        return AuthoritativeProgramRevision(
            schema=GENERATED_PROGRAM_SCHEMA,
            plan_root_cid=str(snapshot.plan_root_cid),
            revision=int(snapshot.revision),
            projection_cid=str(snapshot.projection_cid),
            program_cid=str(snapshot.projection_cid),
            root_goal_cid="",
            goal_cids=(),
            task_identities=(),
            repository_tree_id=str(snapshot.repository_tree_id or ""),
            namespace="generated",
            committed_at_ms=_now_ms(),
            fence_token=hashlib.sha256(
                f"{snapshot.plan_root_cid}:{snapshot.revision}".encode()
            ).hexdigest(),
        )

    def build_runtime_profile(
        self,
        *,
        namespace: str,
        signed_bounds_cid: str = "sha256:" + ("0" * 64),
    ) -> GeneratedBoardRuntimeProfile:
        revision = self.observe()
        return GeneratedBoardRuntimeProfile(
            schema=GENERATED_BOARD_PROFILE_SCHEMA,
            namespace=str(namespace),
            plan_root_cid=revision.plan_root_cid,
            revision=revision.revision,
            projection_cid=revision.projection_cid,
            repository_tree_id=revision.repository_tree_id,
            task_cids=tuple(item.task_cid for item in revision.task_identities),
            signed_bounds_cid=str(signed_bounds_cid),
        )


def build_generated_board_execution_receipt(
    *,
    revision: AuthoritativeProgramRevision,
    profile: GeneratedBoardRuntimeProfile,
    scheduler_argv: Sequence[str],
    supervisor_argv: Sequence[str],
    daemon_argv: Sequence[str],
    planner_invocations: int,
    terminal: bool = True,
    reason_codes: Sequence[str] = (),
) -> GeneratedBoardExecutionReceipt:
    if planner_invocations < 0:
        raise GeneratedProgramError("planner_invocations must be non-negative")
    return GeneratedBoardExecutionReceipt(
        schema=GENERATED_BOARD_EXECUTION_SCHEMA,
        plan_root_cid=revision.plan_root_cid,
        revision=revision.revision,
        profile_cid=profile.profile_cid,
        scheduler_argv=tuple(str(item) for item in scheduler_argv),
        supervisor_argv=tuple(str(item) for item in supervisor_argv),
        daemon_argv=tuple(str(item) for item in daemon_argv),
        observed_task_cids=tuple(item.task_cid for item in revision.task_identities),
        planner_invocations=int(planner_invocations),
        terminal=bool(terminal),
        reason_codes=tuple(str(item) for item in reason_codes),
    )


def inventory_generated_board_duckdb_connects(package_root: str | Path) -> dict[str, Any]:
    """AST inventory: generated-board/planning paths must use policy helper."""

    import ast

    root = Path(package_root)
    targets = [
        root / "planning" / "formal_plan_compiler.py",
        root / "task_sources" / "generated_program_task_source.py",
        root / "task_sources" / "duckdb_task_source.py",
        root / "entrypoints" / "plan_materializer.py",
    ]
    raw_connects: list[str] = []
    policy_uses: list[str] = []
    for path in targets:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        text = path.read_text(encoding="utf-8")
        if "connect_duckdb_with_policy" in text:
            policy_uses.append(str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Attribute) and func.attr == "connect":
                    if isinstance(func.value, ast.Name) and func.value.id == "duckdb":
                        name = "duckdb.connect"
                if name == "duckdb.connect":
                    # Allowed only inside duckdb_state policy helper.
                    if path.name != "duckdb_state.py":
                        raw_connects.append(f"{path}:{node.lineno}")
    return {
        "policy_helper_users": policy_uses,
        "raw_connects": raw_connects,
        "ok": not raw_connects and bool(policy_uses),
    }


__all__ = [
    "AuthoritativeProgramRevision",
    "CanonicalSupervisorProgram",
    "EmbeddedTaskIdentity",
    "GENERATED_BOARD_EXECUTION_SCHEMA",
    "GENERATED_BOARD_PROFILE_SCHEMA",
    "GENERATED_PROGRAM_SCHEMA",
    "GeneratedBoardExecutionReceipt",
    "GeneratedBoardRuntimeProfile",
    "GeneratedProgramError",
    "GeneratedProgramSourceObserver",
    "build_generated_board_execution_receipt",
    "commit_authoritative_program_revision",
    "inventory_generated_board_duckdb_connects",
]
