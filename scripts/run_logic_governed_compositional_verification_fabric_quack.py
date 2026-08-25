#!/usr/bin/env python3
"""Run the additive LGCVF DuckDB + Quack successor controller.

The canonical run-v16 database is forensic input and the sealed run-v17/run-v23
generations remain preserved recovery history.  The active operator has two
explicit stages:

* ``bootstrap`` materializes the exact tracked candidate projection and
  atomically publishes one no-overwrite run-v39 database with provenance;
* ``bootstrap-sealed-continuity`` admits a separately preserved run-v17 only
  into the legacy run-v23 boundary through six explicit raw-byte pins;
* ``launch`` owns run-v39 in-process, starts exactly one foreground
  configured-board scheduler child, and services the closed mutation inbox.

The Quack attach credential exists only in the controller's memory and in the
trusted scheduler process environment.  It is never placed in argv, status,
logs, or a token-vault file.  Implementation-provider environments are still
scrubbed by the existing multi-supervisor boundary.

DuckLake is deliberately a separate, stopped-checkpoint observation.  The
``projection-once`` command writes a physically distinct BoardControlPlane
catalog and marks it non-authoritative; neither ``launch`` nor the configured
scheduler reads that projection for scheduling, leasing, or completion.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import errno
import fcntl
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
_AMBIENT_PYTHONPATH: Final = frozenset(
    item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item
)
_NESTED_DATASETS_ROOT: Final = ROOT / "ipfs_datasets_py"
sys.path[:] = [
    str(ROOT),
    str(_NESTED_DATASETS_ROOT),
    *(
        item
        for item in sys.path
        if item
        and item not in _AMBIENT_PYTHONPATH
        and item not in {str(ROOT), str(_NESTED_DATASETS_ROOT)}
        and not item.startswith("__editable__.")
    ),
]
_RUNTIME_PYCACHE: Final = tempfile.TemporaryDirectory(
    prefix=f"lgcvf-quack-pycache-{os.geteuid()}-"
)
os.chmod(_RUNTIME_PYCACHE.name, 0o700)
sys.pycache_prefix = _RUNTIME_PYCACHE.name
PROGRAM_ROOT_RELATIVE: Final = Path(
    "data/agent_supervisor/logic_governed_compositional_verification_fabric"
)
SOURCE_RUN_RELATIVE: Final = PROGRAM_ROOT_RELATIVE / "run-v17"
LEGACY_SUCCESSOR_RUN_RELATIVE: Final = PROGRAM_ROOT_RELATIVE / "run-v23"
SUCCESSOR_RUN_RELATIVE: Final = PROGRAM_ROOT_RELATIVE / "run-v39"
SOURCE_DATABASE_RELATIVE: Final = SOURCE_RUN_RELATIVE / "control.duckdb"
SUCCESSOR_DATABASE_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "control.duckdb"
OWNER_STATE_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "quack-owner"
PROVENANCE_RELATIVE: Final = (
    SUCCESSOR_RUN_RELATIVE / "evidence" / "quack-successor-provenance.json"
)
CONTROLLER_STATUS_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "controller.status.json"
CONTROLLER_LOCK_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "controller.lock"
CONTROLLER_LOG_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "logs" / "scheduler.log"
PROJECTION_ROOT_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "ducklake-board-projection"
PROJECTION_RECEIPT_RELATIVE: Final = (
    SUCCESSOR_RUN_RELATIVE / "evidence" / "ducklake-board-projection.json"
)
MATERIALIZER_RELATIVE: Final = Path(
    "scripts/materialize_logic_governed_compositional_verification_fabric_control_plane.py"
)
DEFAULT_SUCCESSOR_CONFIG_RELATIVE: Final = Path(
    "config/agent_supervisor_logic_governed_compositional_verification_fabric_quack_candidate_scheduler.json"
)

PROVENANCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-quack-successor-provenance@2"
)
NATIVE_RESUME_ADMISSION_MODE: Final = "tracked_candidate_initial_projection_reset"
NATIVE_RESUME_SOURCE_GENERATION: Final = "lgcvf-tracked-candidate-projection"
SUCCESSOR_STORE_GENERATION: Final = "lgcvf-run-v39"
INTERNAL_CLIENT_GRANT_TTL_SECONDS: Final = 86_400.0
INTERNAL_CLIENT_GRANT_RENEWAL_SECONDS: Final = 43_200.0
STATE_OWNER_BOOTSTRAP_CLIENT_TIMEOUT_SECONDS: Final = 1.0
STATE_OWNER_BOOTSTRAP_PROCESS_STOP_GRACE_SECONDS: Final = 35.0
# Four lanes cold-start serially before the controller can observe the
# required stable all-ready interval.  Keep that finite boundary comfortably
# above the measured four-lane cold-start plus stability budget.
STATE_OWNER_BOOTSTRAP_READY_TIMEOUT_SECONDS: Final = 180.0
STATE_OWNER_BOOTSTRAP_STABILITY_SECONDS: Final = 12.0
LGCVF_SCHEDULER_TREE_STOP_GRACE_SECONDS: Final = 300.0
LGCVF_DATABASE_OWNER_SESSIONS: Final = tuple(
    f"lgcvf-quack-lane-{index}" for index in range(4)
)
LGCVF_TASK_ALIASES: Final = (
    "LGCVF-001",
    "LGCVF-002",
    "LGCVF-010",
    "LGCVF-020",
    "LGCVF-030",
    "LGCVF-040",
    "LGCVF-050",
    "LGCVF-051",
    "LGCVF-060",
    "LGCVF-061",
    "LGCVF-070",
    "LGCVF-071",
    "LGCVF-080",
    "LGCVF-081",
    "LGCVF-090",
    "LGCVF-091",
    "LGCVF-100",
    "LGCVF-101",
    "LGCVF-102",
    "LGCVF-110",
    "LGCVF-111",
    "LGCVF-112",
    "LGCVF-113",
    "LGCVF-120",
    "LGCVF-121",
    "LGCVF-122",
    "LGCVF-123",
    "LGCVF-124",
)
SEALED_CONTINUITY_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-target-only-initial-continuity-verification@1"
)
SEALED_CONTINUITY_MODE: Final = "target_only_initial_continuity"
SEALED_CONTINUITY_AUTHORITY_CEILING: Final = "operational_continuity_only"
FRESH_RECOVERY_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-receipt@1"
)
FRESH_RECOVERY_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-manifest@1"
)
BOOTSTRAP_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-duckdb-materialization@1"
)
NATIVE_RESUME_STAGE_DIRECTORIES: Final = frozenset(
    {
        "evidence",
        "evidence/bootstrap",
    }
)
NATIVE_RESUME_STAGE_LOCK_FILES: Final = frozenset(
    {
        ".control.coordination.duckdb.lock",
        ".control.coordination.duckdb.writer.lock",
        ".control.duckdb.intent.lock",
        ".control.duckdb.lock",
        ".control.duckdb.migration.lock",
        ".control.execution.duckdb.lock",
        ".control.execution.duckdb.writer.lock",
    }
)
NATIVE_RESUME_STAGE_DATA_FILES: Final = frozenset(
    {
        "control.coordination.duckdb",
        "control.duckdb",
        "control.execution.duckdb",
        "evidence/bootstrap/materialization.json",
        "evidence/quack-successor-provenance.json",
    }
)
CONTROLLER_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-quack-successor-status@1"
)
PROJECTION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-ducklake-board-projection@1"
)
TOKEN_ENV: Final = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
TOKEN_FILE_ENV: Final = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE"
DATABASE_PROGRAM_JSON_ENV: Final = "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON"
STORE_GENERATION_ENV: Final = "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"
BOARD_EXTENSION_INSTALL_POLICY_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY"
)
BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY: Final = "load_only"
LGCVF_LIVE_NATIVE_AUTHORIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-configured-board-native-launch-authorization@1"
)
LGCVF_LIVE_SCHEDULER_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler"
)
LGCVF_LIVE_CAPSULE_MANIFEST_MEMBER: Final = (
    ".lgcvf-configured-board-live-capsule-manifest.json"
)
LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES: Final = (
    "ipfs_accelerate_py.agent_implementation_route",
    "ipfs_accelerate_py.llm_router",
    "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts",
    "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema",
    "ipfs_accelerate_py.agent_supervisor.merge.database_coordination",
    "ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry",
    "ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle",
    "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler",
    "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
    "ipfs_accelerate_py.agent_supervisor.runtime.process_security",
    "ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server",
    "ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane",
    "ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client",
    "ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap",
    "ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source",
)
LGCVF_LIVE_REPOSITORY_MODULE_PREFIXES: Final = (
    "ipfs_accelerate_py",
    "ipfs_datasets_py",
    "scripts",
)
LGCVF_LIVE_QUALIFICATION_HOMES_RELATIVE: Final = Path(
    SUCCESSOR_RUN_RELATIVE / "qualification-homes"
)
LGCVF_LIVE_RENDERED_ENV_NAMES: Final = frozenset(
    {
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL",
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT",
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON",
        "IPFS_ACCELERATE_AGENT_EVENT_STORE_PATH",
        "IPFS_ACCELERATE_AGENT_EXPORT_PROFILE",
        "IPFS_ACCELERATE_AGENT_GROK_MODEL",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER",
        "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT",
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH",
        "IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE",
        "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE",
        "IPFS_ACCELERATE_AGENT_STATE_FAILOVER_POLICY",
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID",
        "IPFS_ACCELERATE_AGENT_TASK_SOURCE_KIND",
    }
)
SECRET_HANDLE: Final = f"env://{TOKEN_ENV}"
APPROVED_BOARD_BRANCH: Final = (
    "agent/logic-governed-compositional-verification-fabric-v1"
)
APPROVED_REMOTE_BRANCH_REF: Final = "refs/remotes/github/" + APPROVED_BOARD_BRANCH
MAX_DATABASE_BYTES: Final = 8 * 1024 * 1024 * 1024
MAX_JSON_BYTES: Final = 4 * 1024 * 1024
MAX_SECRET_SURFACE_BYTES: Final = 1024 * 1024 * 1024
MAX_STOP_SECONDS: Final = 360.0
UNIX_SOCKET_PATH_CEILING: Final = 100
COMPLETED_TASK_IDS: Final = (
    "LGCVF-001",
    "LGCVF-002",
    "LGCVF-010",
    "LGCVF-020",
    "LGCVF-030",
    "LGCVF-040",
    "LGCVF-050",
    "LGCVF-051",
    "LGCVF-060",
    "LGCVF-061",
    "LGCVF-070",
    "LGCVF-071",
    "LGCVF-080",
)
TODO_TASK_IDS: Final = (
    "LGCVF-081",
    "LGCVF-090",
    "LGCVF-091",
    "LGCVF-100",
    "LGCVF-101",
    "LGCVF-102",
    "LGCVF-110",
    "LGCVF-111",
    "LGCVF-112",
    "LGCVF-113",
    "LGCVF-120",
    "LGCVF-122",
    "LGCVF-124",
)
BLOCKED_TASK_IDS: Final = ("LGCVF-121", "LGCVF-123")
CONSTRUCTION_COMPLETED_TASK_IDS: Final = COMPLETED_TASK_IDS[:7]
RECOVERED_COMPLETED_TASK_IDS: Final = COMPLETED_TASK_IDS[7:]
SEALED_CONTINUITY_EXPECTED_PINS: Final = {
    "control_sha256": (
        "sha256:c931eb71c8ef861c0b4823341989298311a11414b5a7e69ec13f74db62c09238"
    ),
    "coordination_sha256": (
        "sha256:1882695aba63a3d872cbbb6bb737eb173ea81fd9e0b8b6a5131f11f10f7fa2c4"
    ),
    "execution_sha256": (
        "sha256:ca13093d54c55461eea9250b36a06f16764b51e70f0e25965efb207bafd7e9a5"
    ),
    "bootstrap_sha256": (
        "sha256:dd8baaeaf285a23a4e848f03e4a1fd0532c4127e67210d63896e557219b126ab"
    ),
    "manifest_sha256": (
        "sha256:ba418511fec39660765763b012781b8109d437dc02008c01aa1374f843727c71"
    ),
    "recovery_receipt_sha256": (
        "sha256:24fcad13eb74537b1cd0f7531e27282833a77782323aba4a9e2b98c787b013f2"
    ),
}
SEALED_CONTINUITY_EXPECTED_IDENTITIES: Final = {
    "bootstrap_receipt_cid": (
        "baguqeeraujtyr6ywjlmjagd5ijtvkvcxkag5hrtdhyonb66cyhfq55zpfvaa"
    ),
    "manifest_cid": ("baguqeeravix5cxsnflvjmvniwzpqtkrstappy3z5vgjehgk2xlwdn3yhq62a"),
    "receipt_cid": ("baguqeeramzbpvvpb262jwlqa627d4zbqip6tlg6q5gxycdvr4gaoqonpt5ca"),
    "population_root": (
        "baguqeerar2vrvf44pbumffg65zh5etmged3va3ocumu75v3fdgzqbzlk4nja"
    ),
    "source_evidence_cid": (
        "baguqeera4aybmwbobzlojc4u2cdqxxznmd4bgjkwv2kqka5cukhywnmhy4uq"
    ),
    "sealed_operational_verification_root": (
        "baguqeeraqdjtxgx6wjxkb6u3635s633xy7ymqjby4xxo7xq6wrstfnzym4pa"
    ),
    "target_source_head": "092c95725b9642daa479162d631eff3983e67af6",
    "target_source_tree": "83488b19d20f06da44762a2dfecb4a2666c3b192",
}
GIT_EXECUTABLE: Final = Path("/usr/bin/git")
GIT_TIMEOUT_SECONDS: Final = 120.0


class SuccessorOperatorError(RuntimeError):
    """The successor cannot be admitted without weakening a boundary."""


def _closed_option_values(argv: Sequence[str], option: str) -> tuple[str, ...]:
    """Read one closed CLI option without accepting a missing value."""

    values: list[str] = []
    tokens = tuple(str(item) for item in argv)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == option:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise SuccessorOperatorError(f"{option} has no value")
            values.append(tokens[index + 1])
            index += 2
            continue
        prefix = option + "="
        if token.startswith(prefix):
            value = token[len(prefix) :]
            if not value:
                raise SuccessorOperatorError(f"{option} has no value")
            values.append(value)
        index += 1
    return tuple(values)


def _seal_lgcvf_execution_route_policy(
    *,
    server: Any,
    program: Any,
    identity: Any,
    controller_birth: Any,
    owner_socket: Path,
) -> Any:
    """Seal the exact 28-task Grok/Codex route through a temporary grant."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
        process_birth_id,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
        QuackStateClient,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
        TypedDatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_SOCKET_ENV,
        TYPED_STATE_OWNER_TOKEN_ENV,
    )

    birth_id = process_birth_id(controller_birth)
    if str(getattr(identity, "process_birth_id", "") or "") != birth_id:
        raise SuccessorOperatorError(
            "route sealer process birth differs from the state owner"
        )
    token, grant = server.issue_typed_client_grant_record(
        client_id="lgcvf-route-sealer",
        process_birth_id=birth_id,
        allowed_operations=(
            "whoami_metadata",
            "load_store_generation",
            "executor_control_snapshot",
            "executor_task_projection_page",
        ),
        allowed_command_operations=(),
        peer_pid=os.getpid(),
        ttl_seconds=60.0,
    )
    client: Any | None = None
    projection: Any | None = None
    previous_token = os.environ.get(TYPED_STATE_OWNER_TOKEN_ENV)
    previous_socket = os.environ.get(TYPED_STATE_OWNER_SOCKET_ENV)
    try:
        os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = token
        os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = str(owner_socket)
        client = QuackStateClient(
            owner_id="lgcvf-route-sealer",
            store_id=str(program.store_id),
            process_birth_id=birth_id,
        )
        client.attach(
            str(program.quack_endpoint),
            server_id=str(identity.server_id),
        )
        if previous_token is None:
            os.environ.pop(TYPED_STATE_OWNER_TOKEN_ENV, None)
        else:
            os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = previous_token
        if previous_socket is None:
            os.environ.pop(TYPED_STATE_OWNER_SOCKET_ENV, None)
        else:
            os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = previous_socket
        projection = TypedDatabaseTaskSource(client, owns_client=False)
        execution_modes = {
            alias: GROK_CODEX_EXECUTION_MODE for alias in LGCVF_TASK_ALIASES
        }
        policy = projection.seal_execution_route_policy(execution_modes)
        entries = tuple(policy.entries_by_cid.values())
        if (
            len(entries) != len(LGCVF_TASK_ALIASES)
            or {entry.task_alias for entry in entries} != set(LGCVF_TASK_ALIASES)
            or any(
                entry.execution_mode != GROK_CODEX_EXECUTION_MODE
                for entry in entries
            )
        ):
            raise SuccessorOperatorError(
                "sealed execution route differs from the admitted LGCVF population"
            )
        return policy
    finally:
        if previous_token is None:
            os.environ.pop(TYPED_STATE_OWNER_TOKEN_ENV, None)
        else:
            os.environ[TYPED_STATE_OWNER_TOKEN_ENV] = previous_token
        if previous_socket is None:
            os.environ.pop(TYPED_STATE_OWNER_SOCKET_ENV, None)
        else:
            os.environ[TYPED_STATE_OWNER_SOCKET_ENV] = previous_socket
        if projection is not None:
            projection.close()
        if client is not None:
            client.close()
        server.revoke_typed_client_grant(grant.grant_id)


class _LgcvfStateOwnerBootstrapBroker:
    """Mint one exact-birth typed grant per live LGCVF lane daemon."""

    def __init__(
        self,
        *,
        channel: socket.socket,
        descriptor: int,
        server: Any,
        scheduler_birth: Any,
        endpoint: str,
        socket_path: Path,
        store_id: str,
        execution_route_policy: Any,
        process_stop_grace_seconds: float = (
            STATE_OWNER_BOOTSTRAP_PROCESS_STOP_GRACE_SECONDS
        ),
    ) -> None:
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            validate_state_owner_bootstrap_listener,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
            TaskExecutionRoutePolicy,
        )

        if not isinstance(execution_route_policy, TaskExecutionRoutePolicy):
            raise SuccessorOperatorError(
                "bootstrap broker requires an immutable execution route policy"
            )
        validate_state_owner_bootstrap_listener(descriptor)
        self.channel = channel
        self.descriptor = int(descriptor)
        self.server = server
        self.scheduler_birth = scheduler_birth
        self.endpoint = str(endpoint)
        self.socket_path = Path(socket_path)
        self.store_id = str(store_id)
        self.execution_route_policy = execution_route_policy
        self.process_stop_grace_seconds = float(process_stop_grace_seconds)
        if (
            not math.isfinite(self.process_stop_grace_seconds)
            or self.process_stop_grace_seconds < 0.05
            or self.process_stop_grace_seconds > 300.0
        ):
            raise SuccessorOperatorError(
                "bootstrap broker process-stop grace is invalid"
            )
        self.stopping = threading.Event()
        self.failure = ""
        self.last_rejection = ""
        self.rejection_count = 0
        self._lock = threading.RLock()
        self._accepted: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="lgcvf-state-owner-bootstrap",
            daemon=True,
        )
        self._started = False
        self.current_by_session: dict[str, dict[str, Any]] = {}
        self.active_grants: dict[str, str] = {}

    @property
    def ready_sessions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                session
                for session in LGCVF_DATABASE_OWNER_SESSIONS
                if session in self.current_by_session
                and session in self.active_grants
            )

    @property
    def live_ready_signature(self) -> tuple[str, ...]:
        """Return all four exact daemon births only while each remains alive."""

        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            ProcessBirthIdentity,
            owner_liveness,
        )

        with self._lock:
            records = {
                session: dict(self.current_by_session.get(session) or {})
                for session in LGCVF_DATABASE_OWNER_SESSIONS
            }
        signature: list[str] = []
        for session in LGCVF_DATABASE_OWNER_SESSIONS:
            record = records[session]
            raw_birth = record.get("daemon_process_birth")
            birth_id = str(record.get("daemon_process_birth_id") or "")
            if (
                not isinstance(raw_birth, Mapping)
                or not birth_id
                or owner_liveness(ProcessBirthIdentity.from_dict(raw_birth))
                is not OwnerLiveness.ALIVE
            ):
                return ()
            signature.append(birth_id)
        return tuple(signature)

    def start(self) -> None:
        if self._started:
            raise SuccessorOperatorError("bootstrap broker was already started")
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self.stopping.set()
        with self._lock:
            accepted = self._accepted
        if accepted is not None:
            try:
                accepted.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                accepted.close()
            except OSError:
                pass
        try:
            self.channel.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.channel.close()
        except OSError:
            pass
        if self._started:
            self._thread.join(timeout=5.0)
        if self._started and self._thread.is_alive():
            raise SuccessorOperatorError(
                "state-owner bootstrap broker did not stop"
            )
        self._fence_admitted_births()
        with self._lock:
            grant_ids = tuple(self.active_grants.values())
        revoke_failure = ""
        for grant_id in grant_ids:
            try:
                self.server.revoke_typed_client_grant(grant_id)
            except Exception as exc:  # noqa: BLE001 - revoke every lane.
                revoke_failure = revoke_failure or type(exc).__name__
        if revoke_failure:
            raise SuccessorOperatorError(
                "state-owner bootstrap grant revocation failed: "
                + revoke_failure
            )
        with self._lock:
            for session, grant_id in tuple(self.active_grants.items()):
                if grant_id in grant_ids:
                    self.active_grants.pop(session, None)

    def _admitted_births(self) -> tuple[Any, ...]:
        """Return each exact current credential-holder birth once."""

        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            ProcessBirthIdentity,
        )

        with self._lock:
            records = tuple(
                dict(record) for record in self.current_by_session.values()
            )
        result: list[Any] = []
        seen: set[tuple[int, int, str]] = set()
        for field in (
            "supervisor_process_birth",
            "daemon_process_birth",
        ):
            for record in records:
                raw = record.get(field)
                if not isinstance(raw, Mapping):
                    raise SuccessorOperatorError(
                        "state-owner admitted process birth is unavailable"
                    )
                try:
                    birth = ProcessBirthIdentity.from_dict(raw)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise SuccessorOperatorError(
                        "state-owner admitted process birth is malformed"
                    ) from exc
                if birth.pid <= 1 or birth.start_time_ticks <= 0:
                    raise SuccessorOperatorError(
                        "state-owner admitted process birth is unsafe"
                    )
                key = (birth.pid, birth.start_time_ticks, birth.boot_id)
                if key not in seen:
                    seen.add(key)
                    result.append(birth)
        return tuple(result)

    @staticmethod
    def _signal_admitted_birth(birth: Any, signum: int) -> None:
        """Signal one PID-reuse-resistant admitted birth, or prove it dead."""

        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            owner_liveness,
        )

        if not hasattr(os, "pidfd_open") or not hasattr(
            signal,
            "pidfd_send_signal",
        ):
            raise SuccessorOperatorError(
                "state-owner admitted process fencing requires Linux pidfds"
            )
        pidfd = -1
        try:
            pidfd = os.pidfd_open(birth.pid, 0)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise SuccessorOperatorError(
                "state-owner admitted process pidfd is unavailable"
            ) from exc
        try:
            # Opening the pidfd first makes the subsequent signal immune to a
            # PID disappearing and being reused after this identity check.
            state = owner_liveness(birth)
            if state is OwnerLiveness.DEAD:
                return
            if state is not OwnerLiveness.ALIVE:
                raise SuccessorOperatorError(
                    "state-owner admitted process birth is uninspectable"
                )
            try:
                signal.pidfd_send_signal(pidfd, signum)
            except ProcessLookupError:
                return
            except OSError as exc:
                raise SuccessorOperatorError(
                    "state-owner admitted process could not be signalled"
                ) from exc
        finally:
            os.close(pidfd)

    @staticmethod
    def _live_admitted_births(births: Sequence[Any]) -> tuple[Any, ...]:
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            owner_liveness,
        )

        live: list[Any] = []
        for birth in births:
            state = owner_liveness(birth)
            if state is OwnerLiveness.UNKNOWN:
                raise SuccessorOperatorError(
                    "state-owner admitted process became uninspectable"
                )
            if state is OwnerLiveness.ALIVE:
                live.append(birth)
        return tuple(live)

    def _fence_admitted_births(self) -> None:
        """Prove every credential-holding lane birth dead before revocation."""

        births = self._admitted_births()
        live = self._live_admitted_births(births)
        for birth in live:
            self._signal_admitted_birth(birth, signal.SIGTERM)
        deadline = time.monotonic() + self.process_stop_grace_seconds
        while live and time.monotonic() < deadline:
            time.sleep(0.02)
            live = self._live_admitted_births(live)
        for birth in live:
            self._signal_admitted_birth(birth, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while live and time.monotonic() < deadline:
            time.sleep(0.02)
            live = self._live_admitted_births(live)
        if live:
            raise SuccessorOperatorError(
                "state-owner admitted process births survived bounded stop"
            )

    @staticmethod
    def _require_dead(birth_payload: Mapping[str, Any], *, noun: str) -> None:
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            ProcessBirthIdentity,
            owner_liveness,
        )

        birth = ProcessBirthIdentity.from_dict(birth_payload)
        if owner_liveness(birth) is not OwnerLiveness.DEAD:
            raise SuccessorOperatorError(f"prior {noun} birth remains live")

    def _supervisor_for_daemon(self, daemon_birth: Any, *, session: str) -> Any:
        from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
            process_birth_id,
        )
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            read_process_birth,
        )

        observed_scheduler = read_process_birth(self.scheduler_birth.pid)
        if observed_scheduler != self.scheduler_birth:
            raise SuccessorOperatorError(
                "bootstrap scheduler process birth is no longer exact"
            )
        supervisor = read_process_birth(int(daemon_birth.parent_pid))
        if (
            supervisor is None
            or supervisor.pid <= 1
            or supervisor.parent_pid != self.scheduler_birth.pid
        ):
            raise SuccessorOperatorError(
                "bootstrap daemon is not a child of an admitted lane supervisor"
            )
        before = supervisor
        try:
            raw = Path(f"/proc/{supervisor.pid}/cmdline").read_bytes()
        except OSError as exc:
            raise SuccessorOperatorError(
                "bootstrap lane supervisor argv is unavailable"
            ) from exc
        after = read_process_birth(supervisor.pid)
        if before != after or len(raw) > 1_048_576:
            raise SuccessorOperatorError(
                "bootstrap lane supervisor identity changed during inspection"
            )
        try:
            argv = tuple(
                item.decode("utf-8") for item in raw.split(b"\0") if item
            )
        except UnicodeError as exc:
            raise SuccessorOperatorError(
                "bootstrap lane supervisor argv is malformed"
            ) from exc
        lane_index = LGCVF_DATABASE_OWNER_SESSIONS.index(session)
        exact = {
            "--board-namespace": (
                "logic-governed-compositional-verification-fabric-v1"
            ),
            "--task-shard-count": "4",
            "--task-shard-index": str(lane_index),
            "--state-prefix": f"lgcvf_lane_{lane_index}",
            "--database-owner-session-id": session,
            "--state-owner-bootstrap-fd": str(self.descriptor),
            "--state-owner-bootstrap-store-id": self.store_id,
        }
        if any(
            _closed_option_values(argv, option) != (expected,)
            for option, expected in exact.items()
        ):
            raise SuccessorOperatorError(
                "bootstrap lane supervisor argv differs from its sealed lane"
            )
        supervisor_id = process_birth_id(supervisor)
        for other_session, record in self.current_by_session.items():
            if (
                other_session != session
                and record.get("supervisor_process_birth_id") == supervisor_id
            ):
                raise SuccessorOperatorError(
                    "one lane supervisor requested multiple owner sessions"
                )
        return supervisor

    def _admit(
        self,
        request: Mapping[str, Any],
        *,
        peer_pid: int,
        peer_uid: int,
    ) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
            process_birth_id,
        )
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            ProcessBirthIdentity,
            read_process_birth,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA,
            STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
            daemon_required_owner_command_operations,
            daemon_required_owner_operations,
        )

        required = {
            "schema",
            "pid",
            "process_birth",
            "process_birth_id",
            "client_id",
            "store_id",
        }
        if (
            set(request) != required
            or request.get("schema") != STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA
            or self.stopping.is_set()
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap request differs from its closed schema"
            )
        request_birth = request.get("process_birth")
        raw_pid = request.get("pid")
        if (
            isinstance(raw_pid, bool)
            or not isinstance(raw_pid, int)
            or raw_pid <= 1
            or not isinstance(request_birth, Mapping)
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap request identity is malformed"
            )
        pid = raw_pid
        if pid != peer_pid or peer_uid != os.geteuid():
            raise SuccessorOperatorError(
                "state-owner bootstrap SO_PEERCRED identity differs"
            )
        birth_integer_fields = ("pid", "start_time_ticks", "parent_pid")
        if (
            set(request_birth)
            != {"pid", "start_time_ticks", "boot_id", "parent_pid"}
            or any(
                isinstance(request_birth.get(name), bool)
                or not isinstance(request_birth.get(name), int)
                for name in birth_integer_fields
            )
            or request_birth.get("pid") != pid
            or request_birth.get("start_time_ticks", 0) <= 0
            or request_birth.get("parent_pid", -1) < 0
            or not isinstance(request_birth.get("boot_id"), str)
            or len(request_birth.get("boot_id", "")) > 128
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap process birth is malformed"
            )
        try:
            supplied = ProcessBirthIdentity.from_dict(request_birth)
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            raise SuccessorOperatorError(
                "state-owner bootstrap process birth is malformed"
            ) from exc
        observed = read_process_birth(pid)
        supplied_birth_id = request.get("process_birth_id")
        if (
            not isinstance(supplied_birth_id, str)
            or not supplied_birth_id
            or observed is None
            or observed != supplied
            or process_birth_id(observed) != supplied_birth_id
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap process birth is stale"
            )
        client_id = request.get("client_id")
        requested_store = request.get("store_id")
        if not isinstance(client_id, str) or not isinstance(
            requested_store,
            str,
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap lane scope is malformed"
            )
        matching_sessions = tuple(
            session
            for session in LGCVF_DATABASE_OWNER_SESSIONS
            if client_id == f"database-implementation-daemon:{session}"
        )
        if (
            len(matching_sessions) != 1
            or requested_store != self.store_id
        ):
            raise SuccessorOperatorError(
                "state-owner bootstrap lane scope differs from admission"
            )
        session = matching_sessions[0]
        with self._lock:
            supervisor = self._supervisor_for_daemon(observed, session=session)
            prior = self.current_by_session.get(session)
            if prior is not None:
                prior_daemon = prior.get("daemon_process_birth")
                if not isinstance(prior_daemon, Mapping):
                    raise SuccessorOperatorError(
                        "prior daemon bootstrap record is malformed"
                    )
                self._require_dead(prior_daemon, noun="lane daemon")
                prior_supervisor = prior.get("supervisor_process_birth")
                if (
                    isinstance(prior_supervisor, Mapping)
                    and dict(prior_supervisor) != supervisor.to_dict()
                ):
                    self._require_dead(
                        prior_supervisor,
                        noun="lane supervisor",
                    )
            prior_grant = self.active_grants.pop(session, "")
            if prior_grant:
                self.server.revoke_typed_client_grant(prior_grant)
            token, grant = self.server.issue_typed_client_grant_record(
                client_id=client_id,
                process_birth_id=supplied_birth_id,
                allowed_operations=daemon_required_owner_operations(),
                allowed_command_operations=(
                    daemon_required_owner_command_operations()
                ),
                peer_pid=pid,
                ttl_seconds=INTERNAL_CLIENT_GRANT_TTL_SECONDS,
            )
            if self.stopping.is_set():
                self.server.revoke_typed_client_grant(grant.grant_id)
                raise SuccessorOperatorError(
                    "state-owner bootstrap admission closed during grant issue"
                )
            owner_identity = self.server.identity
            if owner_identity is None:
                self.server.revoke_typed_client_grant(grant.grant_id)
                raise SuccessorOperatorError(
                    "state owner lost identity during bootstrap"
                )
            self.current_by_session[session] = {
                "session": session,
                "client_id": client_id,
                "daemon_process_birth": supplied.to_dict(),
                "daemon_process_birth_id": supplied_birth_id,
                "supervisor_process_birth": supervisor.to_dict(),
                "supervisor_process_birth_id": process_birth_id(supervisor),
                "execution_route_policy": (
                    self.execution_route_policy.public_summary()
                ),
                "grant_expires_at_ms": int(grant.expires_at),
                "grant_renew_after": (
                    time.monotonic() + INTERNAL_CLIENT_GRANT_RENEWAL_SECONDS
                ),
            }
            self.active_grants[session] = grant.grant_id
            return {
                "schema": STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA,
                "ok": True,
                "endpoint": self.endpoint,
                "socket_path": str(self.socket_path),
                "store_id": self.store_id,
                "server_id": str(owner_identity.server_id),
                "client_id": client_id,
                "process_birth_id": supplied_birth_id,
                "token": token,
                "execution_route_policy": self.execution_route_policy.to_dict(),
            }

    def _renew_due_grants(self) -> None:
        """Keep exact live-birth grants bounded and usable indefinitely."""

        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            OwnerLiveness,
            ProcessBirthIdentity,
            owner_liveness,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
            TypedStateOwnerAuthorizationError,
        )

        now = time.monotonic()
        with self._lock:
            due = tuple(
                (
                    session,
                    grant_id,
                    float(
                        (self.current_by_session.get(session) or {}).get(
                            "grant_renew_after",
                            0.0,
                        )
                    ),
                    dict(self.current_by_session.get(session) or {}),
                )
                for session, grant_id in self.active_grants.items()
            )
        for session, grant_id, renew_after, record in due:
            if now < renew_after:
                continue
            raw_birth = record.get("daemon_process_birth")
            if not isinstance(raw_birth, Mapping):
                raise SuccessorOperatorError(
                    "state-owner renewal daemon birth is unavailable"
                )
            daemon_birth = ProcessBirthIdentity.from_dict(raw_birth)
            liveness = owner_liveness(daemon_birth)
            if liveness is OwnerLiveness.DEAD:
                # The supervisor may already be creating this lane's next
                # exact birth.  Leave the old grant for `_admit` to revoke so
                # this serial broker can accept the replacement immediately.
                continue
            if liveness is not OwnerLiveness.ALIVE:
                raise SuccessorOperatorError(
                    "state-owner renewal daemon birth is uninspectable"
                )
            try:
                renewed = self.server.renew_typed_client_grant(
                    grant_id,
                    ttl_seconds=INTERNAL_CLIENT_GRANT_TTL_SECONDS,
                )
            except TypedStateOwnerAuthorizationError:
                if owner_liveness(daemon_birth) is OwnerLiveness.DEAD:
                    continue
                raise
            with self._lock:
                if self.active_grants.get(session) != grant_id:
                    raise SuccessorOperatorError(
                        "state-owner grant rotated during renewal"
                    )
                record = self.current_by_session.get(session)
                if not isinstance(record, dict):
                    raise SuccessorOperatorError(
                        "state-owner renewal record is unavailable"
                    )
                record["grant_expires_at_ms"] = int(renewed.expires_at)
                record["grant_renew_after"] = (
                    time.monotonic() + INTERNAL_CLIENT_GRANT_RENEWAL_SECONDS
                )

    def _run(self) -> None:
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            StateOwnerBootstrapError,
            _receive_frame,
            _send_frame,
        )

        self.channel.settimeout(1.0)
        while not self.stopping.is_set():
            accepted: socket.socket | None = None
            try:
                self._renew_due_grants()
                accepted, _address = self.channel.accept()
                with self._lock:
                    if self.stopping.is_set():
                        accepted.close()
                        return
                    self._accepted = accepted
                accepted.settimeout(
                    STATE_OWNER_BOOTSTRAP_CLIENT_TIMEOUT_SECONDS
                )
                peer = accepted.getsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_PEERCRED,
                    struct.calcsize("3i"),
                )
                peer_pid, peer_uid, _peer_gid = struct.unpack("3i", peer)
                response = self._admit(
                    _receive_frame(accepted),
                    peer_pid=int(peer_pid),
                    peer_uid=int(peer_uid),
                )
                _send_frame(accepted, response)
            except TimeoutError:
                continue
            except (EOFError, StateOwnerBootstrapError, SuccessorOperatorError) as exc:
                if not self.stopping.is_set():
                    self.last_rejection = type(exc).__name__
                    self.rejection_count += 1
                continue
            except OSError:
                if not self.stopping.is_set() and accepted is None:
                    self.failure = "state_owner_bootstrap_channel_closed"
                    return
                continue
            except BaseException as exc:
                self.failure = type(exc).__name__
                try:
                    self.channel.close()
                except OSError:
                    pass
                return
            finally:
                if accepted is not None:
                    with self._lock:
                        if self._accepted is accepted:
                            self._accepted = None
                    try:
                        accepted.close()
                    except OSError:
                        pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    return content_identity(value)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _contained(root: Path, relative: Path | str) -> Path:
    base = root.resolve()
    candidate = (base / Path(relative)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SuccessorOperatorError(
            f"runtime path escapes repository: {relative}"
        ) from exc
    return candidate


def _paths(root: Path = ROOT) -> dict[str, Path]:
    paths = {
        "source_database": _contained(root, SOURCE_DATABASE_RELATIVE),
        "successor_database": _contained(root, SUCCESSOR_DATABASE_RELATIVE),
        "owner_state": _contained(root, OWNER_STATE_RELATIVE),
        "provenance": _contained(root, PROVENANCE_RELATIVE),
        "controller_status": _contained(root, CONTROLLER_STATUS_RELATIVE),
        "controller_lock": _contained(root, CONTROLLER_LOCK_RELATIVE),
        "controller_log": _contained(root, CONTROLLER_LOG_RELATIVE),
        "projection_root": _contained(root, PROJECTION_ROOT_RELATIVE),
        "projection_receipt": _contained(root, PROJECTION_RECEIPT_RELATIVE),
    }
    socket_identity = hashlib.sha256(
        _canonical_bytes(
            {
                "program": "lgcvf-quack-successor-v1",
                "repository_root": str(root.resolve()),
                "runtime_root": str(_contained(root, SUCCESSOR_RUN_RELATIVE)),
                "database": str(paths["successor_database"]),
            }
        )
    ).hexdigest()[:20]
    owner_socket = (
        Path(tempfile.gettempdir())
        / f"ipfs-accelerate-lgcvf-{os.geteuid()}"
        / f"owner-{socket_identity}.sock"
    )
    if len(os.fsencode(owner_socket)) > UNIX_SOCKET_PATH_CEILING:
        raise SuccessorOperatorError(
            "derived state-owner socket path exceeds its bound"
        )
    paths["owner_socket"] = owner_socket
    return paths


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    noun: str,
    require_private_owner: bool = False,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unreadable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > max_bytes
            or (
                require_private_owner
                and (
                    before.st_uid != os.geteuid()
                    or before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode) & 0o077
                )
            )
        ):
            raise SuccessorOperatorError(
                f"{noun} is not a bounded private regular file: {path}"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) > max_bytes or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise SuccessorOperatorError(f"{noun} changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _strict_json(
    path: Path,
    *,
    expected_schema: str = "",
    require_private_owner: bool = False,
    verify_content_identity: bool = True,
) -> dict[str, Any]:
    raw = _read_bounded_regular_file(
        path,
        max_bytes=MAX_JSON_BYTES,
        noun="required receipt",
        require_private_owner=require_private_owner,
    )
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(f"receipt is malformed: {path}") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise SuccessorOperatorError(f"receipt is not a canonical object: {path}")
    if expected_schema and value.get("schema") != expected_schema:
        raise SuccessorOperatorError(f"receipt schema differs: {path}")
    claimed = str(value.get("receipt_cid") or value.get("status_cid") or "")
    if claimed and verify_content_identity:
        unsigned = dict(value)
        unsigned.pop("receipt_cid", None)
        unsigned.pop("status_cid", None)
        if claimed != _content_id(unsigned):
            raise SuccessorOperatorError(f"receipt content identity differs: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise SuccessorOperatorError(f"refusing to overwrite {path}") from exc
            temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _rename_directory_noreplace(
    parent_descriptor: int, source_name: str, target_name: str
) -> None:
    """Atomically publish one same-parent directory without an overwrite fallback."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise SuccessorOperatorError(
            "atomic no-replace directory publication is unavailable"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(target_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    observed_errno = ctypes.get_errno()
    if observed_errno in (errno.EEXIST, errno.ENOTEMPTY):
        raise SuccessorOperatorError("refusing to overwrite an existing successor")
    raise SuccessorOperatorError(
        "atomic no-replace successor publication failed: " + os.strerror(observed_errno)
    )


def _cleanup_successor_stage(
    stage: Path, *, staged_database: Path, staged_provenance: Path
) -> None:
    """Remove only the exact unpublished objects this process created."""

    lock_paths = tuple(
        stage / name
        for name in (
            f".{staged_database.name}.intent.lock",
            f".{staged_database.name}.lock",
            f".{staged_database.name}.migration.lock",
        )
    )
    for path in (staged_provenance, staged_database, *lock_paths):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    cursor = staged_provenance.parent
    while cursor != stage:
        try:
            cursor.rmdir()
        except (FileNotFoundError, OSError):
            break
        cursor = cursor.parent
    try:
        stage.rmdir()
    except (FileNotFoundError, OSError):
        pass


def _remove_staged_database_locks(stage: Path, database_name: str) -> None:
    """Remove only empty, owner-held lock artifacts created by read verification."""

    for name in (
        f".{database_name}.intent.lock",
        f".{database_name}.lock",
        f".{database_name}.migration.lock",
    ):
        path = stage / name
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != 0
        ):
            raise SuccessorOperatorError("staged database lock custody differs")
        path.unlink()


def _open_private_lock(path: Path) -> Any:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise SuccessorOperatorError("controller lock custody is unsafe")
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "a+b")
    except BaseException:
        os.close(descriptor)
        raise


def _sha256_regular_file(
    path: Path,
    *,
    max_bytes: int = MAX_DATABASE_BYTES,
    noun: str = "database",
    require_private_owner: bool = False,
) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unreadable: {path}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > max_bytes
            or (
                require_private_owner
                and (
                    before.st_uid != os.geteuid()
                    or before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode) & 0o077
                )
            )
        ):
            raise SuccessorOperatorError(
                f"{noun} is not a bounded private regular file: {path}"
            )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise SuccessorOperatorError(f"{noun} changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return "sha256:" + digest.hexdigest()


def _regular_file_contains(path: Path, needle: bytes) -> bool:
    if not needle:
        return False
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SuccessorOperatorError(
            f"could not inspect credential surface: {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 0
            or metadata.st_size > MAX_SECRET_SURFACE_BYTES
        ):
            raise SuccessorOperatorError(
                f"credential surface is not a bounded regular file: {path}"
            )
        carry = b""
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return False
            observed = carry + block
            if needle in observed:
                return True
            overlap = max(0, len(needle) - 1)
            carry = observed[-overlap:] if overlap else b""
    finally:
        os.close(descriptor)


def _database_identity(path: Path) -> dict[str, str]:
    import duckdb

    try:
        connection = duckdb.connect(str(path), read_only=True)
        try:
            rows = connection.execute(
                "SELECT key, value FROM control_plane_metadata "
                "WHERE key IN ('database_uuid','schema_version',"
                "'schema_fingerprint','migration_catalog_fingerprint')"
            ).fetchall()
        finally:
            connection.close()
    except Exception as exc:
        raise SuccessorOperatorError(
            f"could not read control-plane identity from {path}: {type(exc).__name__}"
        ) from exc
    return {str(key): str(value or "") for key, value in rows}


def datasets_profile_migration(path: Path) -> Any:
    """Idempotently admit the datasets-authoritative migration catalog."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
        load_datasets_authoritative_operational_catalog,
        verify_datasets_authoritative_operational_schema,
    )

    report = install_datasets_authoritative_operational_schema(
        path,
        application_version="lgcvf-quack-successor-v1",
        tool_version="lgcvf-quack-controller-v1",
        owner_id=f"lgcvf-quack-controller:{os.getpid()}",
    )
    verification = verify_datasets_authoritative_operational_schema(path)
    expected_catalog = load_datasets_authoritative_operational_catalog().fingerprint()
    if (
        verification.get("valid") is not True
        or report.schema_fingerprint != verification.get("schema_fingerprint")
        or report.catalog_fingerprint != expected_catalog
        or verification.get("catalog_fingerprint") != expected_catalog
    ):
        raise SuccessorOperatorError(
            "datasets-authoritative migration report and verification differ"
        )
    return report


def _verify_profile(path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        load_datasets_authoritative_operational_catalog,
        verify_datasets_authoritative_operational_schema,
    )

    verification = verify_datasets_authoritative_operational_schema(path)
    expected = load_datasets_authoritative_operational_catalog().fingerprint()
    if (
        verification.get("valid") is not True
        or verification.get("catalog_fingerprint") != expected
    ):
        raise SuccessorOperatorError(
            f"datasets-authoritative schema verification failed: {path}"
        )
    return verification


def _strict_addressed_mapping(
    value: Mapping[str, Any],
    *,
    identity_field: str,
    noun: str,
) -> dict[str, Any]:
    normalized = dict(value)
    claimed = str(normalized.get(identity_field) or "")
    unsigned = dict(normalized)
    unsigned.pop(identity_field, None)
    if not claimed or claimed != _content_id(unsigned):
        raise SuccessorOperatorError(f"{noun} content identity differs")
    return normalized


def _strict_addressed_json(
    path: Path,
    *,
    expected_schema: str,
    identity_field: str,
    noun: str,
) -> dict[str, Any]:
    value = _strict_json(
        path,
        expected_schema=expected_schema,
        require_private_owner=True,
    )
    return _strict_addressed_mapping(
        value,
        identity_field=identity_field,
        noun=noun,
    )


def _plain_json_object(path: Path, *, noun: str) -> dict[str, Any]:
    raw = _read_bounded_regular_file(path, max_bytes=MAX_JSON_BYTES, noun=noun)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(f"{noun} is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise SuccessorOperatorError(f"{noun} is not an object: {path}")
    return value


def _require_sha256_pin(value: str, *, noun: str) -> str:
    normalized = str(value or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", normalized) is None:
        raise SuccessorOperatorError(f"{noun} SHA-256 pin is malformed")
    return normalized


def _require_private_directory(path: Path, *, noun: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unavailable: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise SuccessorOperatorError(f"{noun} custody is not private: {path}")


def _privatize_owned_directory(path: Path, *, noun: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unavailable: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise SuccessorOperatorError(f"{noun} is not an owned directory: {path}")
    os.chmod(path, 0o700, follow_symlinks=False)
    _require_private_directory(path, noun=noun)


def _sealed_source_paths(source_root: Path) -> dict[str, Path]:
    lexical = Path(os.path.abspath(os.fspath(source_root)))
    if lexical.name != "run-v17":
        raise SuccessorOperatorError("sealed continuity source must be named run-v17")
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor = cursor / component
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise SuccessorOperatorError(
                "sealed continuity source path cannot be inspected"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise SuccessorOperatorError(
                "sealed continuity source path contains a link or non-directory"
            )
    _require_private_directory(lexical, noun="sealed continuity source")
    evidence = lexical / "evidence"
    bootstrap_root = evidence / "bootstrap"
    recovery_root = evidence / "fresh-generation-recovery"
    for directory, noun in (
        (evidence, "sealed evidence directory"),
        (bootstrap_root, "sealed bootstrap directory"),
        (recovery_root, "sealed recovery directory"),
    ):
        _require_private_directory(directory, noun=noun)
    paths = {
        "root": lexical,
        "control": lexical / "control.duckdb",
        "coordination": lexical / "control.coordination.duckdb",
        "execution": lexical / "control.execution.duckdb",
        "bootstrap": bootstrap_root / "materialization.json",
        "recovery_root": recovery_root,
        "recovery_receipt": recovery_root / "recovery-receipt.json",
    }
    for key in ("control", "coordination", "execution"):
        if paths[key].with_name(paths[key].name + ".wal").exists():
            raise SuccessorOperatorError(f"sealed {key} database has a live WAL")
    return paths


def _git_text(root: Path, arguments: Sequence[str], *, noun: str) -> str:
    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    completed = subprocess.run(
        [
            str(GIT_EXECUTABLE),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *arguments,
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise SuccessorOperatorError(
            f"{noun} failed: {(completed.stderr or completed.stdout)[-1000:].strip()}"
        )
    return completed.stdout.strip()


def _git_quiet(root: Path, arguments: Sequence[str], *, noun: str) -> None:
    _git_text(root, arguments, noun=noun)


def _regular_git_blob_oid(path: Path, *, noun: str) -> str:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unreadable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > MAX_DATABASE_BYTES
        ):
            raise SuccessorOperatorError(f"{noun} is not a bounded regular file")
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {before.st_size}\0".encode("ascii"))
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise SuccessorOperatorError(f"{noun} changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _tracked_runtime_inventory(
    repository: Path,
    *,
    head: str,
    pathspecs: Sequence[str],
    noun: str,
) -> dict[str, Any]:
    if (
        _git_text(
            repository,
            ("rev-parse", "--show-object-format"),
            noun=f"{noun} object format",
        )
        != "sha1"
    ):
        raise SuccessorOperatorError(f"{noun} object format is unsupported")
    special_index = _git_text(
        repository,
        ("ls-files", "-v", "-z", "--", *pathspecs),
        noun=f"{noun} index flags",
    )
    if any(
        record and not record.startswith("H ") for record in special_index.split("\0")
    ):
        raise SuccessorOperatorError(f"{noun} has special index flags")
    raw_records = _git_text(
        repository,
        ("ls-tree", "-r", "-z", head, "--", *pathspecs),
        noun=f"{noun} tracked inventory",
    )
    observed: list[tuple[str, str, str]] = []
    for raw in raw_records.split("\0"):
        if not raw:
            continue
        try:
            metadata, relative = raw.split("\t", 1)
            mode, object_type, expected_oid = metadata.split(" ", 2)
        except ValueError as exc:
            raise SuccessorOperatorError(f"{noun} inventory is malformed") from exc
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SuccessorOperatorError(f"{noun} contains an unsafe tracked object")
        if object_type == "commit" and mode == "160000":
            gitlink_path = repository / relative_path
            metadata_status = os.lstat(gitlink_path)
            with os.scandir(gitlink_path) as entries:
                gitlink_is_empty = next(entries, None) is None
            if (
                not stat.S_ISDIR(metadata_status.st_mode)
                or stat.S_ISLNK(metadata_status.st_mode)
                or metadata_status.st_uid != os.geteuid()
                or not gitlink_is_empty
            ):
                raise SuccessorOperatorError(
                    f"{noun} uninitialized gitlink custody differs"
                )
            observed.append((relative_path.as_posix(), mode, expected_oid))
            continue
        if object_type == "blob" and mode == "120000":
            link_path = repository / relative_path
            metadata_status = os.lstat(link_path)
            target_text = os.readlink(link_path)
            target_bytes = os.fsencode(target_text)
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {len(target_bytes)}\0".encode("ascii"))
            digest.update(target_bytes)
            try:
                (link_path.parent / target_text).resolve(strict=True).relative_to(
                    repository.resolve(strict=True)
                )
            except (OSError, ValueError) as exc:
                raise SuccessorOperatorError(
                    f"{noun} tracked link escapes its repository"
                ) from exc
            if (
                not stat.S_ISLNK(metadata_status.st_mode)
                or metadata_status.st_uid != os.geteuid()
                or digest.hexdigest() != expected_oid
            ):
                raise SuccessorOperatorError(f"{noun} tracked link differs from HEAD")
            observed.append((relative_path.as_posix(), mode, expected_oid))
            continue
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise SuccessorOperatorError(f"{noun} contains an unsafe tracked object")
        observed_oid = _regular_git_blob_oid(
            repository / relative_path, noun=f"{noun} tracked object"
        )
        if observed_oid != expected_oid:
            raise SuccessorOperatorError(f"{noun} tracked bytes differ from HEAD")
        observed.append((relative_path.as_posix(), mode, observed_oid))
    ignored = _git_text(
        repository,
        (
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            *pathspecs,
        ),
        noun=f"{noun} ignored inventory",
    )
    for raw in ignored.split("\0"):
        if not raw:
            continue
        relative_path = Path(raw)
        path = repository / relative_path
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise SuccessorOperatorError(
                f"{noun} ignored object cannot be inspected"
            ) from exc
        if (
            relative_path.suffix != ".pyc"
            or "__pycache__" not in relative_path.parts
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise SuccessorOperatorError(
                f"{noun} contains an ignored executable or data object"
            )
    inventory_root = "sha256:" + hashlib.sha256(_canonical_bytes(observed)).hexdigest()
    return {
        "tracked_object_count": len(observed),
        "tracked_inventory_root": inventory_root,
    }


def _candidate_runtime_continuity(root: Path) -> dict[str, Any]:
    if (
        _AMBIENT_PYTHONPATH
        or sys.path[:2] != [str(root), str(root / "ipfs_datasets_py")]
        or sys.pycache_prefix != _RUNTIME_PYCACHE.name
    ):
        raise SuccessorOperatorError("candidate Python import boundary differs")
    quarantine_path = Path(sys.pycache_prefix)
    _require_private_directory(
        quarantine_path, noun="candidate Python bytecode quarantine"
    )
    try:
        quarantine = quarantine_path.resolve(strict=True)
        candidate_root = root.resolve(strict=True)
    except OSError as exc:
        raise SuccessorOperatorError(
            "candidate Python bytecode quarantine cannot be resolved"
        ) from exc
    try:
        quarantine.relative_to(candidate_root)
    except ValueError:
        pass
    else:
        raise SuccessorOperatorError(
            "candidate Python bytecode quarantine is inside the worktree"
        )
    branch = _git_text(root, ("symbolic-ref", "--short", "HEAD"), noun="board branch")
    if branch != APPROVED_BOARD_BRANCH:
        raise SuccessorOperatorError(
            "continuity verification is not on the approved board branch"
        )
    current_head = _git_text(root, ("rev-parse", "HEAD"), noun="current HEAD")
    current_tree = _git_text(root, ("rev-parse", "HEAD^{tree}"), noun="current tree")
    dirty = _git_text(
        root,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        noun="candidate source inventory",
    )
    if dirty:
        raise SuccessorOperatorError(
            "continuity verification requires a completely clean candidate worktree"
        )
    datasets_relative = "ipfs_datasets_py"
    datasets = _contained(root, datasets_relative)
    datasets_metadata = os.lstat(datasets)
    if (
        not stat.S_ISDIR(datasets_metadata.st_mode)
        or stat.S_ISLNK(datasets_metadata.st_mode)
        or datasets_metadata.st_uid != os.geteuid()
    ):
        raise SuccessorOperatorError("nested runtime source custody differs")
    datasets_head = _git_text(
        datasets, ("rev-parse", "HEAD"), noun="nested runtime HEAD"
    )
    datasets_tree = _git_text(
        datasets, ("rev-parse", "HEAD^{tree}"), noun="nested runtime tree"
    )
    datasets_dirty = _git_text(
        datasets,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        noun="nested runtime source inventory",
    )
    gitlink = _git_text(
        root,
        ("ls-tree", current_head, "--", datasets_relative),
        noun="nested runtime gitlink",
    ).split()
    if (
        datasets_dirty
        or len(gitlink) < 3
        or gitlink[0] != "160000"
        or gitlink[1] != "commit"
        or gitlink[2] != datasets_head
    ):
        raise SuccessorOperatorError(
            "continuity verification requires the exact clean nested runtime gitlink"
        )
    remote_head = _git_text(
        root,
        ("rev-parse", APPROVED_REMOTE_BRANCH_REF),
        noun="resolved remote board branch",
    )
    if current_head != remote_head:
        raise SuccessorOperatorError(
            "current board candidate is not the resolved remote branch"
        )
    superproject_inventory = _tracked_runtime_inventory(
        root,
        head=current_head,
        pathspecs=(
            "ipfs_accelerate_py",
            "scripts/ops",
            "scripts/run_logic_governed_compositional_verification_fabric_quack.py",
            "scripts/validate_logic_governed_compositional_verification_fabric_plan.py",
            (
                "config/agent_supervisor_logic_governed_compositional_verification_"
                "fabric_scheduler.json"
            ),
            str(DEFAULT_SUCCESSOR_CONFIG_RELATIVE),
            (
                "docs/architecture/logic_governed_compositional_verification_"
                "fabric.todo.md"
            ),
            (
                "docs/architecture/logic_governed_compositional_verification_"
                "fabric.objectives.md"
            ),
            (
                "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_"
                "FABRIC_PLAN.md"
            ),
            (
                "data/agent_supervisor/logic_governed_compositional_verification_"
                "fabric/formal_work_plan.json"
            ),
        ),
        noun="candidate runtime",
    )
    datasets_inventory = _tracked_runtime_inventory(
        datasets,
        head=datasets_head,
        pathspecs=("__init__.py", "ipfs_datasets_py"),
        noun="nested runtime",
    )
    return {
        "approved_branch": branch,
        "resolved_remote_head": remote_head,
        "current_head": current_head,
        "current_tree": current_tree,
        "candidate_worktree_clean": True,
        "datasets_head": datasets_head,
        "datasets_tree": datasets_tree,
        "datasets_worktree_clean": True,
        "python_bytecode_quarantine": {
            "enabled": True,
            "ephemeral": True,
            "ignored_worktree_pycache": "quarantined_not_imported",
            "outside_candidate_root": True,
            "private": True,
        },
        "superproject_runtime_inventory": superproject_inventory,
        "datasets_runtime_inventory": datasets_inventory,
    }


def _target_source_continuity(
    root: Path,
    *,
    source_head: str,
    source_tree: str,
    config: Mapping[str, Any],
) -> dict[str, str]:
    if (
        re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
    ):
        raise SuccessorOperatorError("sealed source Git identity is malformed")
    candidate = _candidate_runtime_continuity(root)
    branch = str(candidate["approved_branch"])
    if config.get("merge_target_branch") != branch:
        raise SuccessorOperatorError(
            "continuity verification is not on the approved board branch"
        )
    current_head = str(candidate["current_head"])
    observed_source_tree = _git_text(
        root,
        ("show", "-s", "--format=%T", source_head),
        noun="sealed source commit",
    )
    if observed_source_tree != source_tree:
        raise SuccessorOperatorError("sealed source commit/tree binding differs")
    _git_quiet(
        root,
        ("merge-base", "--is-ancestor", source_head, current_head),
        noun="sealed source ancestry",
    )
    authority_paths = []
    for field in (
        "taskboard_path",
        "objectives_path",
        "plan_path",
        "formal_plan_path",
        "validator_path",
    ):
        value = str(config.get(field) or "")
        if not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise SuccessorOperatorError(f"scheduler {field} is unsafe")
        authority_paths.append(value)
    config_relative = (
        "config/agent_supervisor_logic_governed_compositional_verification_fabric_"
        "scheduler.json"
    )
    _git_quiet(
        root,
        (
            "diff",
            "--no-ext-diff",
            "--quiet",
            "HEAD",
            "--",
            config_relative,
            *authority_paths,
        ),
        noun="current authority source worktree",
    )
    _git_quiet(
        root,
        (
            "diff",
            "--no-ext-diff",
            "--quiet",
            source_head,
            current_head,
            "--",
            *authority_paths,
        ),
        noun="sealed/current authority source",
    )
    return {
        **candidate,
        "target_source_head": source_head,
        "target_source_tree": source_tree,
    }


def _require_false_authority(value: Mapping[str, Any], *, noun: str) -> None:
    false_fields = (
        "validation_self_authority",
        "validation_completion_authoritative",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
    )
    if any(value.get(field) is not False for field in false_fields):
        raise SuccessorOperatorError(f"{noun} exceeds the continuity authority ceiling")
    if (
        value.get("candidate_authored_validation") is not True
        or value.get("network_isolation_enforced") is not True
        or value.get("model_provider_route") != "none"
        or value.get("source_database_statuses_read") is not False
        or value.get("source_database_completion_records_imported") is not False
        or value.get("synthetic_source_disposition") != "quarantined_not_imported"
    ):
        raise SuccessorOperatorError(f"{noun} recovery limitations differ")


def _validate_recovery_policy_projection(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    policy = config.get("fresh_generation_recovery")
    plan_binding = config.get("plan_binding")
    if not isinstance(policy, Mapping) or not isinstance(plan_binding, Mapping):
        raise SuccessorOperatorError("tracked fresh-recovery policy is unavailable")
    expected_partition = {
        "construction_completed_task_ids": list(CONSTRUCTION_COMPLETED_TASK_IDS),
        "recovered_completed_task_ids": list(RECOVERED_COMPLETED_TASK_IDS),
        "rejected_synthetic_task_ids": list(TODO_TASK_IDS),
        "preserved_blocked_task_ids": list(BLOCKED_TASK_IDS),
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
    }
    if manifest.get("completion_partition") != expected_partition:
        raise SuccessorOperatorError("sealed completion partition differs")
    retained = manifest.get("retained_completion_binding")
    expected_retained = {
        "binding_cid": policy.get("retained_completion_binding_cid"),
        "construction_completion_count": 7,
        "delta_cid": policy.get("retained_delta_cid"),
        "dynamic_completion_receipt_count": 5,
        "logical_completion_count": 12,
        "path": policy.get("retained_revision_receipt_path"),
        "protected_blocker_binding_cid": policy.get(
            "retained_protected_blocker_binding_cid"
        ),
        "receipt_cid": policy.get("retained_revision_receipt_cid"),
        "sha256": policy.get("retained_revision_receipt_sha256"),
        "successor_revision_cid": policy.get("retained_successor_revision_cid"),
    }
    if retained != expected_retained:
        raise SuccessorOperatorError("sealed retained-completion projection differs")
    quarantine = manifest.get("wrong_default_quarantine")
    if not isinstance(quarantine, Mapping):
        raise SuccessorOperatorError("sealed wrong-default quarantine is unavailable")
    quarantine_projection = {
        "incident_manifest_path": policy.get("wrong_default_incident_manifest_path"),
        "incident_manifest_sha256": policy.get(
            "wrong_default_incident_manifest_sha256"
        ),
        "incident_manifest_cid": policy.get("wrong_default_incident_manifest_cid"),
        "contaminated_coordination_manifest_path": policy.get(
            "contaminated_coordination_projection_path"
        ),
        "contaminated_coordination_manifest_sha256": policy.get(
            "contaminated_coordination_projection_sha256"
        ),
        "contaminated_coordination_manifest_cid": policy.get(
            "contaminated_coordination_projection_manifest_cid"
        ),
        "rejected_record_set_cid": policy.get(
            "contaminated_coordination_rejected_record_set_cid"
        ),
        "rejected_contaminated_coordination_projection_root": policy.get(
            "rejected_contaminated_coordination_projection_root"
        ),
        "rejected_synthetic_task_ids": list(TODO_TASK_IDS),
        "disposition": "preserved_forensic_quarantine_not_imported",
        "source_database_opened": False,
    }
    if any(
        quarantine.get(key) != value for key, value in quarantine_projection.items()
    ):
        raise SuccessorOperatorError(
            "sealed wrong-default quarantine projection differs"
        )
    policy_merges = policy.get("merge_completions")
    manifest_merges = manifest.get("merge_completion_evidence")
    if (
        not isinstance(policy_merges, list)
        or not isinstance(manifest_merges, list)
        or len(policy_merges) != len(RECOVERED_COMPLETED_TASK_IDS)
        or len(manifest_merges) != len(policy_merges)
    ):
        raise SuccessorOperatorError("sealed merge-completion population differs")
    for expected, observed in zip(policy_merges, manifest_merges, strict=True):
        if (
            not isinstance(expected, Mapping)
            or not isinstance(observed, Mapping)
            or any(observed.get(key) != value for key, value in expected.items())
        ):
            raise SuccessorOperatorError("sealed merge-completion projection differs")
    common_fields = (
        "source_generation",
        "target_generation",
        "source_head",
        "source_tree",
        "source_evidence_cid",
        "plan_root_cid",
        "population_root",
        "validation_qualification_cid",
        "candidate_authored_validation",
        "validation_self_authority",
        "validation_completion_authoritative",
        "source_database_statuses_read",
        "source_database_completion_records_imported",
        "synthetic_source_disposition",
        "network_isolation_enforced",
        "model_provider_route",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
    )
    if any(manifest.get(field) != receipt.get(field) for field in common_fields):
        raise SuccessorOperatorError(
            "sealed recovery receipt/manifest projection differs"
        )
    if (
        manifest.get("source_generation") != policy.get("source_generation")
        or manifest.get("target_generation") != policy.get("target_generation")
        or manifest.get("source_runtime_root") != policy.get("source_runtime_root")
        or manifest.get("target_runtime_root") != policy.get("target_runtime_root")
        or manifest.get("plan_root_cid") != plan_binding.get("formal_plan_content_id")
        or receipt.get("completed_task_ids") != list(COMPLETED_TASK_IDS)
        or receipt.get("todo_task_ids") != list(TODO_TASK_IDS)
        or receipt.get("blocked_task_ids") != list(BLOCKED_TASK_IDS)
        or receipt.get("completed_count") != 13
        or receipt.get("todo_count") != 13
        or receipt.get("blocked_count") != 2
        or receipt.get("atomic_publish") is not True
    ):
        raise SuccessorOperatorError("sealed recovery policy binding differs")
    _require_false_authority(manifest, noun="sealed recovery manifest")
    _require_false_authority(receipt, noun="sealed recovery receipt")


def _validate_historical_qualification(manifest: Mapping[str, Any]) -> None:
    qualification = manifest.get("validation_qualification")
    if not isinstance(qualification, Mapping):
        raise SuccessorOperatorError("sealed historical qualification is unavailable")
    normalized = _strict_addressed_mapping(
        qualification,
        identity_field="receipt_cid",
        noun="sealed historical qualification",
    )
    if (
        normalized.get("receipt_cid") != manifest.get("validation_qualification_cid")
        or normalized.get("passed") is not True
        or normalized.get("disposition") != "passed"
        or normalized.get("candidate_authored_replay") is not True
        or normalized.get("completion_authoritative") is not False
        or normalized.get("production_authoritative") is not False
        or normalized.get("production_authorized") is not False
        or normalized.get("objective_complete") is not False
        or normalized.get("provider_route") != "none"
        or normalized.get("network_permitted") is not False
        or normalized.get("cache_reused") is not False
    ):
        raise SuccessorOperatorError(
            "sealed historical qualification limitations differ"
        )
    recovery_manifest = normalized.get("recovery_manifest")
    if not isinstance(recovery_manifest, Mapping):
        raise SuccessorOperatorError("historical qualification manifest is unavailable")
    _strict_addressed_mapping(
        recovery_manifest,
        identity_field="manifest_cid",
        noun="historical qualification manifest",
    )


def _verify_sealed_control_state(
    database: Path,
    *,
    expected_sha256: str,
    manifest: Mapping[str, Any],
    formal_plan: Mapping[str, Any],
) -> dict[str, Any]:
    import duckdb

    before = _sha256_regular_file(
        database,
        noun="sealed control database",
        require_private_owner=True,
    )
    if before != expected_sha256:
        raise SuccessorOperatorError("sealed control database SHA-256 differs")
    profile = _verify_profile(database)
    formal_tasks = formal_plan.get("tasks")
    if not isinstance(formal_tasks, list):
        raise SuccessorOperatorError("tracked formal task population is unavailable")
    formal_by_alias = {
        str(item.get("task_id") or ""): dict(item)
        for item in formal_tasks
        if isinstance(item, Mapping)
    }
    all_aliases = set(COMPLETED_TASK_IDS + TODO_TASK_IDS + BLOCKED_TASK_IDS)
    if set(formal_by_alias) != all_aliases:
        raise SuccessorOperatorError("tracked formal task population differs")
    try:
        connection = duckdb.connect(
            str(database),
            read_only=True,
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
            },
        )
        try:
            task_rows = connection.execute(
                "SELECT task_cid, task_alias, status, revision, plan_cid, "
                "identity_json, body_json FROM tasks ORDER BY task_alias"
            ).fetchall()
            plan_rows = connection.execute(
                "SELECT plan_cid, plan_alias, status, revision, body_json "
                "FROM plans ORDER BY plan_cid"
            ).fetchall()
            dependency_rows = connection.execute(
                "SELECT task_cid, dependency_task_cid, kind "
                "FROM task_dependencies ORDER BY task_cid, dependency_task_cid, kind"
            ).fetchall()
            completion_rows = connection.execute(
                "SELECT task_cid FROM completion_receipts ORDER BY task_cid"
            ).fetchall()
            zero_counts = {
                table: int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                for table in (
                    "task_claims",
                    "task_attempts",
                    "task_assignments",
                    "task_blocks",
                    "resource_claims",
                    "maintenance_leases",
                    "leases",
                    "lease_events",
                    "token_history",
                    "client_sessions",
                )
            }
        finally:
            connection.close()
    except Exception as exc:
        if isinstance(exc, SuccessorOperatorError):
            raise
        raise SuccessorOperatorError(
            f"sealed control state cannot be reconstructed: {type(exc).__name__}"
        ) from exc
    if len(task_rows) != 28 or any(zero_counts.values()):
        raise SuccessorOperatorError(
            "sealed control database has unexpected live state"
        )
    expected_status_revision = {
        **{alias: ("completed", 1) for alias in CONSTRUCTION_COMPLETED_TASK_IDS},
        **{alias: ("completed", 2) for alias in RECOVERED_COMPLETED_TASK_IDS},
        **{alias: ("todo", 1) for alias in TODO_TASK_IDS},
        **{alias: ("blocked", 1) for alias in BLOCKED_TASK_IDS},
    }
    tasks_by_cid: dict[str, str] = {}
    rows_by_alias: dict[str, dict[str, Any]] = {}
    for (
        task_cid,
        alias,
        status,
        revision,
        plan_cid,
        identity_raw,
        body_raw,
    ) in task_rows:
        task_cid = str(task_cid)
        alias = str(alias)
        try:
            identity = json.loads(str(identity_raw))
            body = json.loads(str(body_raw))
        except json.JSONDecodeError as exc:
            raise SuccessorOperatorError("sealed task JSON is malformed") from exc
        if (
            alias not in expected_status_revision
            or (str(status), int(revision)) != expected_status_revision[alias]
            or str(plan_cid) != manifest.get("plan_root_cid")
            or not isinstance(identity, Mapping)
            or identity.get("task_alias") != alias
            or identity.get("task_cid") != task_cid
            or identity.get("repository_tree_id")
            != "git-tree:" + str(manifest.get("source_tree") or "")
            or not isinstance(body, Mapping)
            or body.get("formal_record") != formal_by_alias[alias]
            or body.get("formal_task_content_id") != task_cid
            or body.get("board_namespace")
            != "logic-governed-compositional-verification-fabric-v1"
        ):
            raise SuccessorOperatorError(f"{alias}: sealed task authority differs")
        if task_cid in tasks_by_cid or alias in rows_by_alias:
            raise SuccessorOperatorError("sealed task identity is duplicated")
        tasks_by_cid[task_cid] = alias
        rows_by_alias[alias] = {
            "task_cid": task_cid,
            "status": str(status),
            "body": dict(body),
        }
    if set(rows_by_alias) != all_aliases:
        raise SuccessorOperatorError("sealed task alias population differs")
    if len(plan_rows) != 1:
        raise SuccessorOperatorError("sealed plan population differs")
    plan_cid, plan_alias, plan_status, plan_revision, plan_body_raw = plan_rows[0]
    try:
        plan_body = json.loads(str(plan_body_raw))
    except json.JSONDecodeError as exc:
        raise SuccessorOperatorError("sealed plan JSON is malformed") from exc
    if (
        str(plan_cid) != manifest.get("plan_root_cid")
        or str(plan_alias) != "logic-governed-compositional-verification-fabric-v1"
        or str(plan_status) != "active"
        or int(plan_revision) != 1
        or not isinstance(plan_body, Mapping)
        or plan_body.get("source_head") != manifest.get("source_head")
        or plan_body.get("repository_tree_id")
        != "git-tree:" + str(manifest.get("source_tree") or "")
    ):
        raise SuccessorOperatorError("sealed active plan differs")
    observed_dependencies: set[tuple[str, str]] = set()
    for task_cid, dependency_cid, kind in dependency_rows:
        task_alias = tasks_by_cid.get(str(task_cid), "")
        dependency_alias = tasks_by_cid.get(str(dependency_cid), "")
        if not task_alias or not dependency_alias or str(kind) != "depends_on":
            raise SuccessorOperatorError("sealed dependency identity differs")
        observed_dependencies.add((task_alias, dependency_alias))
    expected_dependencies = {
        (alias, str(dependency))
        for alias, task in formal_by_alias.items()
        for dependency in task.get("depends_on") or ()
    }
    if (
        len(dependency_rows) != 46
        or len(observed_dependencies) != 46
        or observed_dependencies != expected_dependencies
    ):
        raise SuccessorOperatorError("sealed dependency graph differs")
    completed_cids = {rows_by_alias[alias]["task_cid"] for alias in COMPLETED_TASK_IDS}
    ready = []
    dependencies_by_alias: dict[str, set[str]] = {alias: set() for alias in all_aliases}
    for alias, dependency in observed_dependencies:
        dependencies_by_alias[alias].add(rows_by_alias[dependency]["task_cid"])
    for alias in TODO_TASK_IDS:
        row = rows_by_alias[alias]
        if (
            row["body"].get("is_schedulable") is True
            and dependencies_by_alias[alias] <= completed_cids
        ):
            ready.append(alias)
    if ready != ["LGCVF-081"]:
        raise SuccessorOperatorError("sealed ready frontier differs")
    completion_aliases = [tasks_by_cid.get(str(row[0]), "") for row in completion_rows]
    if sorted(completion_aliases) != sorted(RECOVERED_COMPLETED_TASK_IDS):
        raise SuccessorOperatorError("sealed reconstructed completion receipts differ")
    after = _sha256_regular_file(
        database,
        noun="sealed control database",
        require_private_owner=True,
    )
    if before != after:
        raise SuccessorOperatorError(
            "sealed control database changed during verification"
        )
    identity = _database_identity(database)
    return {
        "sha256": before,
        "database_uuid": identity.get("database_uuid", ""),
        "schema_fingerprint": profile.get("schema_fingerprint", ""),
        "catalog_fingerprint": profile.get("catalog_fingerprint", ""),
        "task_count": 28,
        "dependency_count": 46,
        "completion_receipt_count": 6,
        "ready_task_ids": ready,
        "zero_state_counts": zero_counts,
        "task_cids_by_alias": {
            alias: rows_by_alias[alias]["task_cid"] for alias in sorted(rows_by_alias)
        },
    }


def _verify_sealed_coordination_state(
    database: Path,
    *,
    expected_sha256: str,
    control_tasks: Mapping[str, str],
    formal_plan: Mapping[str, Any],
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )

    before = _sha256_regular_file(
        database,
        noun="sealed coordination database",
        require_private_owner=True,
    )
    if before != expected_sha256:
        raise SuccessorOperatorError("sealed coordination database SHA-256 differs")
    try:
        projection = read_coordination_registry_projection(database)
    except Exception as exc:
        raise SuccessorOperatorError(
            f"sealed coordination projection is unreadable: {type(exc).__name__}"
        ) from exc
    expected_counts = {
        "registered_tasks": 28,
        "dependency_edges": 46,
        "logical_completions": 13,
        "task_claims": 0,
        "active_task_claims": 0,
        "resource_claims": 0,
        "active_resource_claims": 0,
        "task_attempts": 0,
        "active_task_attempts": 0,
        "fenced_leases": 0,
        "active_fenced_leases": 0,
        "maintenance_leases": 0,
        "active_maintenance_leases": 0,
    }
    if projection.get("counts") != expected_counts or any(
        projection.get(field) != []
        for field in (
            "task_claims",
            "task_attempts",
            "fenced_leases",
            "resource_claims",
            "maintenance_leases",
        )
    ):
        raise SuccessorOperatorError("sealed coordination database has live state")
    registered = {
        str(item.get("task_id") or ""): str(item.get("task_cid") or "")
        for item in projection.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    if registered != dict(control_tasks):
        raise SuccessorOperatorError("sealed coordination task registry differs")
    cid_to_alias = {cid: alias for alias, cid in control_tasks.items()}
    observed_dependencies = {
        (
            cid_to_alias.get(str(item.get("task_cid") or ""), ""),
            cid_to_alias.get(str(item.get("dependency_task_cid") or ""), ""),
        )
        for item in projection.get("dependency_edges") or ()
        if isinstance(item, Mapping)
    }
    formal_tasks = formal_plan.get("tasks") or ()
    expected_dependencies = {
        (str(task.get("task_id") or ""), str(dependency))
        for task in formal_tasks
        if isinstance(task, Mapping)
        for dependency in task.get("depends_on") or ()
    }
    completion_aliases = {
        cid_to_alias.get(str(item.get("task_cid") or ""), "")
        for item in projection.get("logical_completions") or ()
        if isinstance(item, Mapping) and item.get("status") == "succeeded"
    }
    if observed_dependencies != expected_dependencies or completion_aliases != set(
        COMPLETED_TASK_IDS
    ):
        raise SuccessorOperatorError("sealed coordination authority differs")
    after = _sha256_regular_file(
        database,
        noun="sealed coordination database",
        require_private_owner=True,
    )
    if before != after:
        raise SuccessorOperatorError(
            "sealed coordination database changed during verification"
        )
    return {"sha256": before, "counts": expected_counts}


def _verify_sealed_execution_state(
    database: Path,
    *,
    expected_sha256: str,
    control_schema_fingerprint: str,
) -> dict[str, Any]:
    import duckdb

    before = _sha256_regular_file(
        database,
        noun="sealed execution database",
        require_private_owner=True,
    )
    if before != expected_sha256:
        raise SuccessorOperatorError("sealed execution database SHA-256 differs")
    try:
        connection = duckdb.connect(
            str(database),
            read_only=True,
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
            },
        )
        try:
            counts = {
                table: int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                for table in (
                    "attempt_phases",
                    "daemon_execution_events",
                    "database_task_attempts",
                    "effect_claims",
                    "provider_invocations",
                )
            }
            metadata = {
                str(key): str(value)
                for key, value in connection.execute(
                    "SELECT key, value FROM daemon_execution_metadata ORDER BY key"
                ).fetchall()
            }
        finally:
            connection.close()
    except Exception as exc:
        raise SuccessorOperatorError(
            f"sealed execution state is unreadable: {type(exc).__name__}"
        ) from exc
    if any(counts.values()) or (
        metadata.get("authority_mode") != "embedded"
        or metadata.get("control_schema_fingerprint") != control_schema_fingerprint
        or metadata.get("control_schema_profile_id")
        != "datasets-authoritative-operational-control-plane@1"
        or metadata.get("interface") != "DatabaseImplementationDaemon@1"
        or metadata.get("process_instance_id") != "fresh-recovery-bootstrap"
        or metadata.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1"
        or metadata.get("state_schema_revision")
        != "datasets-authoritative-operational-v1"
        or not str(metadata.get("logical_owner_session_id") or "").startswith(
            "embedded-store:"
        )
    ):
        raise SuccessorOperatorError("sealed execution database has unexpected state")
    after = _sha256_regular_file(
        database,
        noun="sealed execution database",
        require_private_owner=True,
    )
    if before != after:
        raise SuccessorOperatorError(
            "sealed execution database changed during verification"
        )
    return {"sha256": before, "row_counts": counts, "metadata": metadata}


def _verify_sealed_layout(paths: Mapping[str, Path], *, manifest_name: str) -> None:
    expected_root = {
        ".control.coordination.duckdb.lock",
        ".control.duckdb.intent.lock",
        ".control.duckdb.lock",
        ".control.duckdb.migration.lock",
        ".control.execution.duckdb.lock",
        "control.coordination.duckdb",
        "control.duckdb",
        "control.execution.duckdb",
        "evidence",
    }
    expected_evidence = {"bootstrap", "fresh-generation-recovery"}
    expected_bootstrap = {"materialization.json"}
    expected_recovery = {"recovery-receipt.json", manifest_name}
    observed = {
        "root": {item.name for item in os.scandir(paths["root"])},
        "evidence": {item.name for item in os.scandir(paths["root"] / "evidence")},
        "bootstrap": {
            item.name for item in os.scandir(paths["root"] / "evidence" / "bootstrap")
        },
        "recovery": {item.name for item in os.scandir(paths["recovery_root"])},
    }
    if observed != {
        "root": expected_root,
        "evidence": expected_evidence,
        "bootstrap": expected_bootstrap,
        "recovery": expected_recovery,
    }:
        raise SuccessorOperatorError("sealed run-v17 layout differs")
    for name in sorted(expected_root):
        if not name.startswith("."):
            continue
        lock_path = paths["root"] / name
        metadata = os.lstat(lock_path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != 0
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise SuccessorOperatorError("sealed empty lock-file custody differs")


def _assert_sealed_report_snapshot(
    paths: Mapping[str, Path],
    report: Mapping[str, Any],
) -> None:
    pins = report.get("pins")
    manifest_cid = str(report.get("manifest_cid") or "")
    if (
        not isinstance(pins, Mapping)
        or re.fullmatch(r"bagu[a-z2-7]{20,}", manifest_cid) is None
    ):
        raise SuccessorOperatorError("sealed continuity report pins are unavailable")
    manifest = paths["recovery_root"] / f"{manifest_cid}.manifest.json"
    observed = {
        "control_sha256": _sha256_regular_file(
            paths["control"], noun="sealed control database", require_private_owner=True
        ),
        "coordination_sha256": _sha256_regular_file(
            paths["coordination"],
            noun="sealed coordination database",
            require_private_owner=True,
        ),
        "execution_sha256": _sha256_regular_file(
            paths["execution"],
            noun="sealed execution database",
            require_private_owner=True,
        ),
        "bootstrap_sha256": _sha256_regular_file(
            paths["bootstrap"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed bootstrap receipt",
            require_private_owner=True,
        ),
        "manifest_sha256": _sha256_regular_file(
            manifest,
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery manifest",
            require_private_owner=True,
        ),
        "recovery_receipt_sha256": _sha256_regular_file(
            paths["recovery_receipt"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery receipt",
            require_private_owner=True,
        ),
    }
    if any(pins.get(key) != value for key, value in observed.items()):
        raise SuccessorOperatorError(
            "sealed continuity snapshot changed after verification"
        )
    _verify_sealed_layout(paths, manifest_name=manifest.name)


def _validate_bootstrap_receipt(
    bootstrap: Mapping[str, Any],
    *,
    recovery_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    verification = bootstrap.get("verification")
    if not isinstance(verification, Mapping):
        raise SuccessorOperatorError("sealed bootstrap verification is unavailable")
    _strict_addressed_mapping(
        verification,
        identity_field="verification_root",
        noun="sealed bootstrap verification",
    )
    expected_paths = {
        "control": (
            "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
            "run-v17/control.duckdb"
        ),
        "coordination": (
            "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
            "run-v17/control.coordination.duckdb"
        ),
        "execution": (
            "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
            "run-v17/control.execution.duckdb"
        ),
    }
    if (
        bootstrap.get("receipt_cid") != recovery_receipt.get("bootstrap_receipt_cid")
        or bootstrap.get("population_root") != manifest.get("population_root")
        or bootstrap.get("plan_root_cid") != manifest.get("plan_root_cid")
        or bootstrap.get("source_head") != manifest.get("source_head")
        or bootstrap.get("repository_tree_id")
        != "git-tree:" + str(manifest.get("source_tree") or "")
        or bootstrap.get("authority_mode") != "embedded"
        or bootstrap.get("task_source_kind") != "duckdb"
        or bootstrap.get("maximum_writer_processes") != 1
        or bootstrap.get("quack_qualified") is not False
        or bootstrap.get("schema_revision") != "datasets-authoritative-operational-v1"
        or bootstrap.get("schema_profile") != "datasets-authoritative-operational"
        or bootstrap.get("database_paths") != expected_paths
        or verification.get("valid") is not True
        or verification.get("stores_unchanged") is not True
    ):
        raise SuccessorOperatorError("sealed bootstrap receipt binding differs")


def verify_sealed_target_continuity(
    *,
    root: Path,
    source_root: Path,
    control_sha256: str,
    coordination_sha256: str,
    execution_sha256: str,
    bootstrap_sha256: str,
    manifest_sha256: str,
    recovery_receipt_sha256: str,
) -> dict[str, Any]:
    """Admit one reviewed hash-pinned snapshot with bounded semantic checks."""

    root = root.resolve(strict=True)
    _candidate_runtime_continuity(root)
    paths = _sealed_source_paths(source_root)
    pins = {
        "control_sha256": _require_sha256_pin(
            control_sha256, noun="sealed control database"
        ),
        "coordination_sha256": _require_sha256_pin(
            coordination_sha256, noun="sealed coordination database"
        ),
        "execution_sha256": _require_sha256_pin(
            execution_sha256, noun="sealed execution database"
        ),
        "bootstrap_sha256": _require_sha256_pin(
            bootstrap_sha256, noun="sealed bootstrap receipt"
        ),
        "manifest_sha256": _require_sha256_pin(
            manifest_sha256, noun="sealed recovery manifest"
        ),
        "recovery_receipt_sha256": _require_sha256_pin(
            recovery_receipt_sha256, noun="sealed recovery receipt"
        ),
    }
    if pins != SEALED_CONTINUITY_EXPECTED_PINS:
        raise SuccessorOperatorError(
            "sealed continuity pins differ from the reviewed board candidate"
        )
    recovery_receipt = _strict_addressed_json(
        paths["recovery_receipt"],
        expected_schema=FRESH_RECOVERY_RECEIPT_SCHEMA,
        identity_field="receipt_cid",
        noun="sealed recovery receipt",
    )
    manifest_cid = str(recovery_receipt.get("manifest_cid") or "")
    if re.fullmatch(r"bagu[a-z2-7]{20,}", manifest_cid) is None:
        raise SuccessorOperatorError("sealed recovery manifest CID is unsafe")
    manifest_path = paths["recovery_root"] / f"{manifest_cid}.manifest.json"
    manifest = _strict_addressed_json(
        manifest_path,
        expected_schema=FRESH_RECOVERY_MANIFEST_SCHEMA,
        identity_field="manifest_cid",
        noun="sealed recovery manifest",
    )
    bootstrap = _strict_addressed_json(
        paths["bootstrap"],
        expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
        identity_field="receipt_cid",
        noun="sealed bootstrap receipt",
    )
    observed_identities = {
        "bootstrap_receipt_cid": bootstrap.get("receipt_cid"),
        "manifest_cid": manifest.get("manifest_cid"),
        "receipt_cid": recovery_receipt.get("receipt_cid"),
        "population_root": recovery_receipt.get("population_root"),
        "source_evidence_cid": recovery_receipt.get("source_evidence_cid"),
        "sealed_operational_verification_root": recovery_receipt.get(
            "operational_verification_root"
        ),
        "target_source_head": manifest.get("source_head"),
        "target_source_tree": manifest.get("source_tree"),
    }
    if observed_identities != SEALED_CONTINUITY_EXPECTED_IDENTITIES:
        raise SuccessorOperatorError(
            "sealed continuity identities differ from the reviewed board candidate"
        )
    observed_artifact_hashes = {
        "bootstrap_sha256": _sha256_regular_file(
            paths["bootstrap"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed bootstrap receipt",
            require_private_owner=True,
        ),
        "manifest_sha256": _sha256_regular_file(
            manifest_path,
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery manifest",
            require_private_owner=True,
        ),
        "recovery_receipt_sha256": _sha256_regular_file(
            paths["recovery_receipt"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery receipt",
            require_private_owner=True,
        ),
    }
    if any(
        observed_artifact_hashes[key] != pins[key] for key in observed_artifact_hashes
    ):
        raise SuccessorOperatorError("sealed recovery artifact SHA-256 differs")
    if (
        manifest.get("manifest_cid") != manifest_cid
        or recovery_receipt.get("bootstrap_receipt_sha256") != pins["bootstrap_sha256"]
    ):
        raise SuccessorOperatorError("sealed recovery artifact cross-binding differs")
    _verify_sealed_layout(paths, manifest_name=manifest_path.name)
    _validate_bootstrap_receipt(
        bootstrap,
        recovery_receipt=recovery_receipt,
        manifest=manifest,
    )
    config = _plain_json_object(
        _contained(
            root,
            "config/agent_supervisor_logic_governed_compositional_verification_fabric_"
            "scheduler.json",
        ),
        noun="tracked scheduler config",
    )
    formal_plan = _plain_json_object(
        _contained(root, str(config.get("formal_plan_path") or "")),
        noun="tracked formal plan",
    )
    _validate_recovery_policy_projection(
        config=config,
        manifest=manifest,
        receipt=recovery_receipt,
    )
    _validate_historical_qualification(manifest)
    source_binding = _target_source_continuity(
        root,
        source_head=str(manifest.get("source_head") or ""),
        source_tree=str(manifest.get("source_tree") or ""),
        config=config,
    )
    control = _verify_sealed_control_state(
        paths["control"],
        expected_sha256=pins["control_sha256"],
        manifest=manifest,
        formal_plan=formal_plan,
    )
    coordination = _verify_sealed_coordination_state(
        paths["coordination"],
        expected_sha256=pins["coordination_sha256"],
        control_tasks=control["task_cids_by_alias"],
        formal_plan=formal_plan,
    )
    execution = _verify_sealed_execution_state(
        paths["execution"],
        expected_sha256=pins["execution_sha256"],
        control_schema_fingerprint=str(control["schema_fingerprint"]),
    )
    after_artifact_hashes = {
        "bootstrap_sha256": _sha256_regular_file(
            paths["bootstrap"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed bootstrap receipt",
            require_private_owner=True,
        ),
        "manifest_sha256": _sha256_regular_file(
            manifest_path,
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery manifest",
            require_private_owner=True,
        ),
        "recovery_receipt_sha256": _sha256_regular_file(
            paths["recovery_receipt"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery receipt",
            require_private_owner=True,
        ),
    }
    if after_artifact_hashes != observed_artifact_hashes:
        raise SuccessorOperatorError(
            "sealed recovery artifacts changed during verification"
        )
    report: dict[str, Any] = {
        "schema": SEALED_CONTINUITY_VERIFICATION_SCHEMA,
        "valid": True,
        "verification_mode": "read_only_hash_pinned_target_snapshot",
        "admission_mode": SEALED_CONTINUITY_MODE,
        "authority_ceiling": SEALED_CONTINUITY_AUTHORITY_CEILING,
        "source_root": str(paths["root"]),
        "candidate_root": str(root),
        "source_generation": "lgcvf-run-v17",
        "target_generation": "lgcvf-run-v17",
        "manifest_cid": manifest_cid,
        "receipt_cid": recovery_receipt["receipt_cid"],
        "bootstrap_receipt_cid": bootstrap["receipt_cid"],
        "source_evidence_cid": recovery_receipt["source_evidence_cid"],
        "population_root": recovery_receipt["population_root"],
        "plan_root_cid": recovery_receipt["plan_root_cid"],
        "sealed_operational_verification_root": recovery_receipt[
            "operational_verification_root"
        ],
        "pins": pins,
        "source_binding": source_binding,
        "control": {
            key: value for key, value in control.items() if key != "task_cids_by_alias"
        },
        "coordination": coordination,
        "execution": execution,
        "completed_task_ids": list(COMPLETED_TASK_IDS),
        "todo_task_ids": list(TODO_TASK_IDS),
        "blocked_task_ids": list(BLOCKED_TASK_IDS),
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
        "ready_task_ids": ["LGCVF-081"],
        "stores_unchanged": True,
        "target_database_statuses_read": True,
        "source_database_statuses_read": False,
        "fresh_source_evidence_revalidated": False,
        "historical_source_bytes_revalidated": False,
        "source_provenance_authoritative": False,
        "target_snapshot_hash_pinned": True,
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
        "source_database_completion_records_imported": False,
        "synthetic_source_disposition": "quarantined_not_imported",
        "network_isolation_enforced": True,
        "model_provider_route": "none",
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "authoritative_for_release": False,
        "production_authorized": False,
    }
    report["verification_root"] = _content_id(report)
    return report


def _canonical_recovery_verification(root: Path = ROOT) -> dict[str, Any]:
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(_contained(root, MATERIALIZER_RELATIVE)),
        "recovery-verify",
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300.0,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SuccessorOperatorError(
            "canonical run-v17 recovery verifier returned malformed output"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(report, dict)
        or report.get("valid") is not True
        or report.get("target_generation") != "lgcvf-run-v17"
        or report.get("stores_unchanged") is not True
        or report.get("source_database_statuses_read") is not False
        or report.get("completed_count") != 13
        or report.get("todo_count") != 13
        or report.get("blocked_count") != 2
        or report.get("ready_task_ids") != ["LGCVF-081"]
    ):
        raise SuccessorOperatorError(
            "canonical run-v17 recovery is not a verified 13/13/2 recovery: "
            + str(report.get("error") or completed.stderr[-1000:])
        )
    return report


def clone_verified_successor(
    source_database: Path,
    target_database: Path,
    provenance_path: Path,
    *,
    recovery_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one complete, verified, no-overwrite successor run."""

    source = Path(source_database).resolve(strict=True)
    target = Path(os.path.abspath(os.fspath(target_database)))
    provenance = Path(os.path.abspath(os.fspath(provenance_path)))
    final_run = target.parent
    try:
        provenance_relative = provenance.relative_to(final_run)
    except ValueError as exc:
        raise SuccessorOperatorError(
            "successor provenance must be inside the target generation"
        ) from exc
    if (
        source.parent.name != "run-v17"
        or final_run.name != "run-v23"
        or target.name != "control.duckdb"
        or len(provenance_relative.parts) != 2
        or provenance_relative.parts[0] != "evidence"
    ):
        raise SuccessorOperatorError("successor clone must be run-v17 -> run-v23")
    if source == target:
        raise SuccessorOperatorError("successor source and target are identical")
    try:
        os.lstat(final_run)
    except FileNotFoundError:
        pass
    else:
        raise SuccessorOperatorError("refusing to overwrite an existing successor")
    if os.path.lexists(source.with_name(source.name + ".wal")):
        raise SuccessorOperatorError("run-v17 control database has a live WAL")
    admission_mode = str(
        recovery_verification.get("admission_mode")
        or "canonical_fresh_generation_recovery"
    )
    sealed_source_paths: dict[str, Path] | None = None
    if admission_mode == SEALED_CONTINUITY_MODE:
        sealed_source_paths = _sealed_source_paths(
            Path(str(recovery_verification.get("source_root") or ""))
        )
        if (
            source != sealed_source_paths["control"]
            or recovery_verification.get("authority_ceiling")
            != SEALED_CONTINUITY_AUTHORITY_CEILING
            or recovery_verification.get("target_snapshot_hash_pinned") is not True
            or recovery_verification.get("historical_source_bytes_revalidated")
            is not False
            or recovery_verification.get("source_provenance_authoritative") is not False
            or recovery_verification.get("authoritative_for_release") is not False
            or recovery_verification.get("production_authorized") is not False
        ):
            raise SuccessorOperatorError(
                "sealed target continuity report is not admissible"
            )
        _require_false_authority(
            recovery_verification, noun="sealed target continuity report"
        )
        _assert_sealed_report_snapshot(sealed_source_paths, recovery_verification)
    elif admission_mode != "canonical_fresh_generation_recovery":
        raise SuccessorOperatorError("successor admission mode is unsupported")
    if (
        recovery_verification.get("valid") is not True
        or recovery_verification.get("target_generation") != "lgcvf-run-v17"
        or recovery_verification.get("stores_unchanged") is not True
        or recovery_verification.get("source_database_statuses_read") is not False
    ):
        raise SuccessorOperatorError("run-v17 recovery verification is not admissible")

    source_verification = _verify_profile(source)
    source_identity = _database_identity(source)
    source_digest = _sha256_regular_file(source)
    if sealed_source_paths is not None and source_digest != (
        recovery_verification.get("pins") or {}
    ).get("control_sha256"):
        raise SuccessorOperatorError(
            "sealed control source differs from its admitted pin"
        )

    publish_parent = final_run.parent
    publish_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _privatize_owned_directory(publish_parent, noun="successor publication parent")
    # Keep the unpublished generation under the same reviewed run-v* ignore
    # boundary as the final generation.  Sealed admission is repeated after
    # cloning; a hidden .run-v23.* stage would otherwise appear as untracked
    # worktree dirt and make every real bootstrap fail closed on itself.
    stage = publish_parent / f"{final_run.name}.stage-{uuid.uuid4().hex}"
    os.mkdir(stage, mode=0o700)
    staged_database = stage / target.name
    staged_provenance = stage / provenance_relative
    parent_descriptor = os.open(
        publish_parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    parent_before = os.fstat(parent_descriptor)
    stage_before = os.lstat(stage)
    source_descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    target_descriptor: int | None = None
    published = False
    try:
        source_before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(source_before.st_mode)
            or source_before.st_size <= 0
            or source_before.st_size > MAX_DATABASE_BYTES
        ):
            raise SuccessorOperatorError(
                "run-v17 source is not a bounded regular database"
            )
        target_descriptor = os.open(
            staged_database,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(target_descriptor, view)
                if written <= 0:
                    raise SuccessorOperatorError("run-v23 clone write made no progress")
                view = view[written:]
        os.fsync(target_descriptor)
        os.close(target_descriptor)
        target_descriptor = None

        target_verification = _verify_profile(staged_database)
        target_identity = _database_identity(staged_database)
        target_digest = _sha256_regular_file(
            staged_database,
            noun="staged successor database",
            require_private_owner=True,
        )
        if (
            _sha256_regular_file(source) != source_digest
            or target_digest != source_digest
            or target_identity != source_identity
            or target_verification.get("schema_fingerprint")
            != source_verification.get("schema_fingerprint")
        ):
            raise SuccessorOperatorError("run-v23 clone differs from verified run-v17")
        if sealed_source_paths is not None:
            pins = recovery_verification.get("pins") or {}
            refreshed = verify_sealed_target_continuity(
                root=Path(str(recovery_verification.get("candidate_root") or "")),
                source_root=sealed_source_paths["root"],
                control_sha256=str(pins.get("control_sha256") or ""),
                coordination_sha256=str(pins.get("coordination_sha256") or ""),
                execution_sha256=str(pins.get("execution_sha256") or ""),
                bootstrap_sha256=str(pins.get("bootstrap_sha256") or ""),
                manifest_sha256=str(pins.get("manifest_sha256") or ""),
                recovery_receipt_sha256=str(pins.get("recovery_receipt_sha256") or ""),
            )
            if refreshed != dict(recovery_verification):
                raise SuccessorOperatorError(
                    "sealed continuity report changed before successor publication"
                )

        receipt = {
            "schema": PROVENANCE_SCHEMA,
            "issued_at": _utc_now(),
            "source_generation": "lgcvf-run-v17",
            "target_generation": "lgcvf-run-v23",
            "source_database": str(source),
            "target_database": str(target),
            "source_sha256": source_digest,
            "target_initial_sha256": target_digest,
            "database_uuid": source_identity.get("database_uuid", ""),
            "schema_fingerprint": source_verification["schema_fingerprint"],
            "catalog_fingerprint": source_verification["catalog_fingerprint"],
            "recovery_verification_root": str(
                recovery_verification.get("verification_root") or ""
            ),
            "recovery_receipt_cid": str(recovery_verification.get("receipt_cid") or ""),
            "recovery_manifest_cid": str(
                recovery_verification.get("manifest_cid") or ""
            ),
            "bootstrap_receipt_cid": str(
                recovery_verification.get("bootstrap_receipt_cid") or ""
            ),
            "source_evidence_cid": str(
                recovery_verification.get("source_evidence_cid") or ""
            ),
            "population_root": str(recovery_verification.get("population_root") or ""),
            "plan_root_cid": str(recovery_verification.get("plan_root_cid") or ""),
            "admission_mode": admission_mode,
            "authority_ceiling": str(
                recovery_verification.get("authority_ceiling")
                or "operational_recovery_only"
            ),
            "source_root": str(
                sealed_source_paths["root"]
                if sealed_source_paths is not None
                else source.parent
            ),
            "source_coordination_database": str(
                sealed_source_paths["coordination"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_execution_database": str(
                sealed_source_paths["execution"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_bootstrap_receipt": str(
                sealed_source_paths["bootstrap"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_manifest": str(
                (
                    sealed_source_paths["recovery_root"]
                    / f"{recovery_verification.get('manifest_cid')}.manifest.json"
                )
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_receipt": str(
                sealed_source_paths["recovery_receipt"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_coordination_sha256": str(
                (recovery_verification.get("pins") or {}).get("coordination_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_execution_sha256": str(
                (recovery_verification.get("pins") or {}).get("execution_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_bootstrap_sha256": str(
                (recovery_verification.get("pins") or {}).get("bootstrap_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_manifest_sha256": str(
                (recovery_verification.get("pins") or {}).get("manifest_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_receipt_sha256": str(
                (recovery_verification.get("pins") or {}).get("recovery_receipt_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "target_source_head": str(
                (recovery_verification.get("source_binding") or {}).get(
                    "target_source_head"
                )
                if sealed_source_paths is not None
                else ""
            ),
            "target_source_tree": str(
                (recovery_verification.get("source_binding") or {}).get(
                    "target_source_tree"
                )
                if sealed_source_paths is not None
                else ""
            ),
            "sealed_operational_verification_root": str(
                recovery_verification.get("sealed_operational_verification_root") or ""
            ),
            "fresh_source_evidence_revalidated": admission_mode
            != SEALED_CONTINUITY_MODE,
            "historical_source_bytes_revalidated": admission_mode
            != SEALED_CONTINUITY_MODE,
            "source_provenance_authoritative": admission_mode != SEALED_CONTINUITY_MODE,
            "target_snapshot_hash_pinned": admission_mode == SEALED_CONTINUITY_MODE,
            "target_database_statuses_read": admission_mode == SEALED_CONTINUITY_MODE,
            "source_database_statuses_read_scope": (
                "lost_fresh_recovery_source_generation_lgcvf-run-v16"
            ),
            "restart_requires_live_continuity_receipt": admission_mode
            == SEALED_CONTINUITY_MODE,
            "live_continuity_receipt_implemented": False,
            "clone_preserves_database_uuid": True,
            "owner_generation_rotates_on_start": True,
            "source_database_statuses_read": False,
            "source_database_completion_records_imported": False,
            "candidate_authored_validation": True,
            "validation_self_authority": False,
            "validation_completion_authoritative": False,
            "synthetic_source_disposition": "quarantined_not_imported",
            "network_isolation_enforced": True,
            "model_provider_route": "none",
            "task_implementation_complete": False,
            "test_qualification_complete": False,
            "objective_complete": False,
            "release_qualified": False,
            "authoritative_for_release": False,
            "production_authorized": False,
        }
        receipt["receipt_cid"] = _content_id(receipt)
        _atomic_json(staged_provenance, receipt, replace=False)
        if (
            _strict_json(
                staged_provenance,
                expected_schema=PROVENANCE_SCHEMA,
                require_private_owner=True,
            )
            != receipt
        ):
            raise SuccessorOperatorError("staged successor provenance differs")
        _remove_staged_database_locks(stage, staged_database.name)
        if (
            {item.name for item in os.scandir(stage)}
            != {target.name, provenance_relative.parts[0]}
            or {item.name for item in os.scandir(staged_provenance.parent)}
            != {staged_provenance.name}
            or os.path.lexists(staged_database.with_name(staged_database.name + ".wal"))
        ):
            raise SuccessorOperatorError("staged successor inventory differs")
        _require_private_directory(stage, noun="staged successor generation")
        _require_private_directory(
            staged_provenance.parent, noun="staged successor evidence"
        )
        source_after = os.fstat(source_descriptor)
        stage_after = os.lstat(stage)
        parent_after = os.fstat(parent_descriptor)
        if (
            (
                source_before.st_dev,
                source_before.st_ino,
                source_before.st_size,
                source_before.st_mtime_ns,
                source_before.st_ctime_ns,
            )
            != (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_size,
                source_after.st_mtime_ns,
                source_after.st_ctime_ns,
            )
            or (stage_before.st_dev, stage_before.st_ino)
            != (stage_after.st_dev, stage_after.st_ino)
            or (parent_before.st_dev, parent_before.st_ino)
            != (parent_after.st_dev, parent_after.st_ino)
            or _sha256_regular_file(
                staged_database,
                noun="staged successor database",
                require_private_owner=True,
            )
            != target_digest
            or _sha256_regular_file(source) != source_digest
        ):
            raise SuccessorOperatorError(
                "source or staged successor changed before publication"
            )
        stage_descriptor = os.open(
            stage,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(stage_descriptor)
        finally:
            os.close(stage_descriptor)
        _rename_directory_noreplace(parent_descriptor, stage.name, final_run.name)
        published = True
        try:
            os.fsync(parent_descriptor)
        except OSError as exc:
            raise SuccessorOperatorError(
                "successor published completely but parent durability is uncertain"
            ) from exc
        return receipt
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(source_descriptor)
        os.close(parent_descriptor)
        if not published:
            _cleanup_successor_stage(
                stage,
                staged_database=staged_database,
                staged_provenance=staged_provenance,
            )


def _require_ignored_successor(
    root: Path,
    *,
    run_relative: Path = SUCCESSOR_RUN_RELATIVE,
) -> None:
    stage_lock = (
        run_relative.with_name(run_relative.name + ".stage-probe")
        / ".control.duckdb.lock"
    )
    for relative, noun in (
        (run_relative / "control.duckdb", "successor Git-ignore policy"),
        (stage_lock, "successor staging Git-ignore policy"),
    ):
        _git_quiet(
            root,
            ("check-ignore", "-q", "--no-index", str(relative)),
            noun=noun,
        )


def _load_native_resume_config(root: Path) -> tuple[dict[str, Any], bytes]:
    """Load the exact tracked run-v39 profile with duplicate-key rejection."""

    path = _contained(root, DEFAULT_SUCCESSOR_CONFIG_RELATIVE)
    raw = _read_bounded_regular_file(
        path,
        max_bytes=MAX_JSON_BYTES,
        noun="LGCVF native-resume candidate config",
    )

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate candidate config key: {key}")
            value[key] = item
        return value

    try:
        config = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(
            "LGCVF native-resume candidate config is invalid"
        ) from exc
    if not isinstance(config, dict):
        raise SuccessorOperatorError(
            "LGCVF native-resume candidate config must be an object"
        )
    program = config.get("database_program")
    runtime = config.get("runtime_paths")
    projection = config.get("initial_projection")
    bootstrap = config.get("bootstrap_writer_policy")
    expected_projection = {
        "task_count": 28,
        "completed_task_ids": list(CONSTRUCTION_COMPLETED_TASK_IDS),
        "ready_task_ids": ["LGCVF-051", "LGCVF-060", "LGCVF-070", "LGCVF-080"],
        "blocked_task_ids": list(BLOCKED_TASK_IDS),
        "terminal_task_id": "LGCVF-124",
        "goal_count": 14,
        "root_goal_id": "LGCVF-G000",
    }
    if (
        config.get("schema")
        != (
            "ipfs_accelerate_py.agent_supervisor."
            "logic_governed_compositional_verification_fabric.scheduler_config@1"
        )
        or config.get("board_namespace")
        != "logic-governed-compositional-verification-fabric-v1"
        or not isinstance(program, dict)
        or not isinstance(runtime, dict)
        or program.get("store_id") != SUCCESSOR_DATABASE_RELATIVE.as_posix()
        or program.get("store_generation") != SUCCESSOR_STORE_GENERATION
        or program.get("authority_mode") != "quack"
        or runtime.get("root") != SUCCESSOR_RUN_RELATIVE.as_posix()
        or projection != expected_projection
        or bootstrap
        != {
            "maximum_processes": 1,
            "quack_required": False,
            "offline_single_writer_materialization_permitted": True,
            "quack_required_after_publish": True,
            "direct_multi_process_duckdb_permitted": False,
            "automatic_installation_permitted": False,
        }
    ):
        raise SuccessorOperatorError(
            "LGCVF native-resume candidate projection or generation differs"
        )
    return config, raw


def _native_resume_stage_config(
    config: Mapping[str, Any],
    *,
    root: Path,
    stage: Path,
) -> dict[str, Any]:
    """Retarget only unpublished materializer paths into the private stage."""

    staged = copy.deepcopy(dict(config))
    try:
        relative = stage.relative_to(root.resolve(strict=True)).as_posix()
        program = staged["database_program"]
        runtime = staged["runtime_paths"]
        program["store_id"] = f"{relative}/control.duckdb"
        runtime["evidence"] = f"{relative}/evidence"
    except (KeyError, TypeError, ValueError) as exc:
        raise SuccessorOperatorError(
            "LGCVF native-resume staging paths are unavailable"
        ) from exc
    return staged


def _verify_native_resume_projection(
    database: Path,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact initial task frontier from the unpublished DB."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    projection = config.get("initial_projection")
    if not isinstance(projection, Mapping):
        raise SuccessorOperatorError("native-resume initial projection is unavailable")
    with DatabaseTaskSource(database, install_schema=False) as source:
        records = list(source.list_tasks(limit=100).tasks)
        ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]
    completed = [item.task_alias for item in records if item.status == "completed"]
    todo = [item.task_alias for item in records if item.status == "todo"]
    blocked = [item.task_alias for item in records if item.status == "blocked"]
    expected_completed = list(projection.get("completed_task_ids") or ())
    expected_ready = list(projection.get("ready_task_ids") or ())
    expected_blocked = list(projection.get("blocked_task_ids") or ())
    expected_todo_count = (
        int(projection.get("task_count") or 0)
        - len(expected_completed)
        - len(expected_blocked)
    )
    if (
        len(records) != projection.get("task_count")
        or completed != expected_completed
        or ready != expected_ready
        or blocked != expected_blocked
        or len(todo) != expected_todo_count
    ):
        raise SuccessorOperatorError(
            "materialized native-resume authority differs from initial_projection"
        )
    result = {
        "task_count": len(records),
        "completed_count": len(completed),
        "todo_count": len(todo),
        "blocked_count": len(blocked),
        "completed_task_ids": completed,
        "ready_task_ids": ready,
        "blocked_task_ids": blocked,
    }
    result["projection_root"] = _content_id(result)
    return result


def _validate_native_bootstrap_receipt(
    receipt: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    database_paths: Mapping[str, str],
    source_head: str,
    repository_tree_id: str,
    population_root: str,
    plan_root_cid: str,
    schema_fingerprint: str,
    catalog_fingerprint: str,
) -> None:
    """Replay the exact initial materializer receipt semantics."""

    projection = config.get("initial_projection")
    materialization = receipt.get("materialization")
    verification = receipt.get("verification")
    schema_install = receipt.get("schema_install")
    if not all(
        isinstance(item, Mapping)
        for item in (projection, materialization, verification, schema_install)
    ):
        raise SuccessorOperatorError(
            "native-resume bootstrap receipt structure differs"
        )
    assert isinstance(projection, Mapping)
    assert isinstance(materialization, Mapping)
    assert isinstance(verification, Mapping)
    assert isinstance(schema_install, Mapping)
    receipt_body = dict(receipt)
    claimed_receipt_cid = str(receipt_body.pop("receipt_cid", ""))
    verification_body = dict(verification)
    claimed_verification_root = str(
        verification_body.pop("verification_root", "")
    )
    task_source = materialization.get("task_source")
    control = verification.get("control")
    coordination = verification.get("coordination")
    execution = verification.get("execution")
    if not all(
        isinstance(item, Mapping)
        for item in (task_source, control, coordination, execution)
    ):
        raise SuccessorOperatorError(
            "native-resume bootstrap receipt projection differs"
        )
    assert isinstance(task_source, Mapping)
    assert isinstance(control, Mapping)
    assert isinstance(coordination, Mapping)
    assert isinstance(execution, Mapping)
    registered = materialization.get("registered_task_cids")
    completed = materialization.get("bootstrap_completed_task_cids")
    task_cids = task_source.get("task_cids")
    statuses = control.get("statuses")
    ready = list(projection.get("ready_task_ids") or ())
    completed_aliases = list(projection.get("completed_task_ids") or ())
    blocked_aliases = list(projection.get("blocked_task_ids") or ())
    task_count = int(projection.get("task_count") or 0)
    goal_count = int(projection.get("goal_count") or 0)
    expected_top_level_fields = {
        "schema",
        "authority_mode",
        "task_source_kind",
        "maximum_writer_processes",
        "quack_qualified",
        "schema_revision",
        "schema_profile",
        "semantic_truth_authority",
        "operational_coordination_authority",
        "population_root",
        "plan_root_cid",
        "repository_tree_id",
        "source_head",
        "database_paths",
        "schema_install",
        "materialization",
        "verification",
        "receipt_cid",
    }
    expected_verification_fields = {
        "schema",
        "valid",
        "verification_mode",
        "expected_stage",
        "population_root",
        "plan_root_cid",
        "repository_tree_id",
        "control",
        "coordination",
        "execution",
        "stores_unchanged",
        "maximum_writer_processes",
        "quack_qualified",
        "verification_root",
    }
    expected_coordination_counts = {
        "active_fenced_leases": 0,
        "active_maintenance_leases": 0,
        "active_resource_claims": 0,
        "active_task_attempts": 0,
        "active_task_claims": 0,
        "dependency_edges": 46,
        "fenced_leases": 0,
        "logical_completions": len(completed_aliases),
        "maintenance_leases": 0,
        "registered_tasks": task_count,
        "resource_claims": 0,
        "task_attempts": 0,
        "task_claims": 0,
    }
    expected_execution_counts = {
        "attempt_phases": 0,
        "daemon_execution_events": 0,
        "database_task_attempts": 0,
        "effect_claims": 0,
        "provider_invocations": 0,
    }
    expected_schema_install_fields = {
        "catalog_fingerprint",
        "changed",
        "from_version",
        "receipts",
        "schema",
        "schema_fingerprint",
        "to_version",
    }
    expected_migration_receipt_fields = {
        "application_version",
        "checksum",
        "error_text",
        "finished_at",
        "migration_id",
        "outcome",
        "receipt_cid",
        "schema",
        "schema_fingerprint",
        "started_at",
        "tool_version",
        "version",
    }
    expected_task_source_fields = {
        "event_watermark",
        "goal_count",
        "goal_edge_count",
        "plan_count",
        "plan_root_cid",
        "projection_cid",
        "repository_tree_id",
        "schema",
        "task_cids",
        "task_count",
    }
    expected_control_fields = {
        "catalog_projection",
        "completion_receipts",
        "dependency_count",
        "event_stream_root",
        "evidence",
        "goal_count",
        "objective_revision_history",
        "plan_projection",
        "plan_revision_history",
        "ready_task_aliases",
        "relation_count",
        "relation_inventory",
        "residual_content_projection",
        "runtime_progress_observed",
        "schema_verification",
        "semantic_event_stream_root",
        "semantic_events",
        "statuses",
        "table_counts",
        "task_count",
        "task_revision_histories",
        "tasks",
    }
    expected_coordination_fields = {
        "catalog_projection",
        "counts",
        "projection_root",
    }
    expected_execution_fields = {
        "catalog_projection",
        "metadata",
        "row_counts",
        "runtime_progress_observed",
        "schema_inventory",
    }

    def exact_integer(value: Any, expected: int) -> bool:
        return type(value) is int and value == expected

    def exact_integer_mapping(value: Any, expected: Mapping[str, int]) -> bool:
        return (
            isinstance(value, Mapping)
            and set(value) == set(expected)
            and all(
                exact_integer(value.get(key), item)
                for key, item in expected.items()
            )
        )

    def canonical_cid(value: Any) -> bool:
        return (
            isinstance(value, str)
            and re.fullmatch(r"b[a-z2-7]{60}", value) is not None
        )

    migration_receipts = schema_install.get("receipts")
    migration_receipt: Mapping[str, Any] = (
        migration_receipts[0]
        if isinstance(migration_receipts, list)
        and len(migration_receipts) == 1
        and isinstance(migration_receipts[0], Mapping)
        else {}
    )
    semantic_difference = (
        set(receipt) != expected_top_level_fields
        or claimed_receipt_cid != _content_id(receipt_body)
        or not exact_integer(projection.get("task_count"), 28)
        or not exact_integer(projection.get("goal_count"), 14)
        or receipt.get("schema") != BOOTSTRAP_RECEIPT_SCHEMA
        or receipt.get("authority_mode") != "embedded"
        or receipt.get("task_source_kind") != "duckdb"
        or not exact_integer(receipt.get("maximum_writer_processes"), 1)
        or receipt.get("quack_qualified") is not False
        or receipt.get("schema_revision")
        != "datasets-authoritative-operational-v1"
        or receipt.get("schema_profile")
        != "datasets-authoritative-operational"
        or receipt.get("semantic_truth_authority") != "ipfs_datasets_py"
        or receipt.get("operational_coordination_authority")
        != "ipfs_accelerate_py"
        or receipt.get("population_root") != population_root
        or receipt.get("plan_root_cid") != plan_root_cid
        or receipt.get("repository_tree_id") != repository_tree_id
        or receipt.get("source_head") != source_head
        or receipt.get("database_paths") != dict(database_paths)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in database_paths.items()
        )
        or set(schema_install) != expected_schema_install_fields
        or schema_install.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/control-plane-migration-run@1"
        or schema_install.get("changed") is not True
        or not exact_integer(schema_install.get("from_version"), 0)
        or not exact_integer(schema_install.get("to_version"), 1)
        or schema_install.get("schema_fingerprint") != schema_fingerprint
        or schema_install.get("catalog_fingerprint") != catalog_fingerprint
        or set(migration_receipt) != expected_migration_receipt_fields
        or migration_receipt.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "control-plane-migration-receipt@1"
        )
        or migration_receipt.get("schema_fingerprint") != schema_fingerprint
        or migration_receipt.get("outcome") != "applied"
        or migration_receipt.get("application_version") != "lgcvf-v1"
        or not exact_integer(migration_receipt.get("version"), 1)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(migration_receipt.get("checksum") or ""),
        )
        is None
        or not canonical_cid(migration_receipt.get("receipt_cid"))
        or set(materialization)
        != {
            "bootstrap_completed_task_cids",
            "registered_task_cids",
            "task_source",
        }
        or not isinstance(registered, list)
        or len(registered) != task_count
        or not all(canonical_cid(item) for item in registered)
        or len(set(registered)) != task_count
        or task_cids != registered
        or not isinstance(completed, list)
        or completed != registered[: len(completed_aliases)]
        or set(task_source) != expected_task_source_fields
        or task_source.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/database-task-source@1"
        or not exact_integer(task_source.get("task_count"), task_count)
        or not exact_integer(task_source.get("goal_count"), goal_count)
        or not exact_integer(task_source.get("goal_edge_count"), 38)
        or not exact_integer(task_source.get("plan_count"), 1)
        or not exact_integer(task_source.get("event_watermark"), 82)
        or not canonical_cid(task_source.get("projection_cid"))
        or task_source.get("plan_root_cid") != plan_root_cid
        or task_source.get("repository_tree_id") != repository_tree_id
        or set(verification) != expected_verification_fields
        or claimed_verification_root != _content_id(verification_body)
        or verification.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "lgcvf-duckdb-read-only-verification@1"
        )
        or verification.get("valid") is not True
        or verification.get("verification_mode") != "read_only"
        or verification.get("expected_stage") != "initial"
        or verification.get("population_root") != population_root
        or verification.get("plan_root_cid") != plan_root_cid
        or verification.get("repository_tree_id") != repository_tree_id
        or verification.get("stores_unchanged") is not True
        or not exact_integer(verification.get("maximum_writer_processes"), 1)
        or verification.get("quack_qualified") is not False
        or set(control) != expected_control_fields
        or not exact_integer(control.get("task_count"), task_count)
        or not exact_integer(control.get("goal_count"), goal_count)
        or not exact_integer(control.get("dependency_count"), 46)
        or control.get("ready_task_aliases") != ready
        or control.get("runtime_progress_observed") is not False
        or not isinstance(statuses, Mapping)
        or not all(
            isinstance(alias, str) and isinstance(status, str)
            for alias, status in statuses.items()
        )
        or len(statuses) != task_count
        or [alias for alias, status in statuses.items() if status == "completed"]
        != completed_aliases
        or [alias for alias, status in statuses.items() if status == "blocked"]
        != blocked_aliases
        or sum(status == "todo" for status in statuses.values())
        != task_count - len(completed_aliases) - len(blocked_aliases)
        or set(coordination) != expected_coordination_fields
        or not exact_integer_mapping(
            coordination.get("counts"), expected_coordination_counts
        )
        or set(execution) != expected_execution_fields
        or not exact_integer_mapping(
            execution.get("row_counts"), expected_execution_counts
        )
        or execution.get("runtime_progress_observed") is not False
    )
    if semantic_difference:
        raise SuccessorOperatorError(
            "native-resume bootstrap receipt semantics differ"
        )


def _verify_native_resume_stage_allowlist(
    stage: Path,
    *,
    include_provenance: bool,
) -> None:
    """Require the materializer to leave only the declared initial objects."""

    expected_files = set(NATIVE_RESUME_STAGE_DATA_FILES)
    if not include_provenance:
        expected_files.remove("evidence/quack-successor-provenance.json")
    observed_directories: set[str] = set()
    observed_files: set[str] = set()
    for path in stage.rglob("*"):
        relative = path.relative_to(stage).as_posix()
        metadata = os.lstat(path)
        if stat.S_ISDIR(metadata.st_mode):
            if (
                metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise SuccessorOperatorError(
                    "native-resume stage directory custody differs"
                )
            observed_directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            if (
                metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
            ):
                raise SuccessorOperatorError(
                    "native-resume stage file custody differs"
                )
            observed_files.add(relative)
        else:
            raise SuccessorOperatorError(
                "native-resume stage contains an undeclared object"
            )
    if (
        observed_directories != set(NATIVE_RESUME_STAGE_DIRECTORIES)
        or observed_files
        != expected_files | set(NATIVE_RESUME_STAGE_LOCK_FILES)
        or any(
            os.lstat(stage / relative).st_size != 0
            for relative in NATIVE_RESUME_STAGE_LOCK_FILES
        )
    ):
        raise SuccessorOperatorError(
            "native-resume stage inventory differs from the exact allowlist"
        )


def _privatize_and_sync_native_resume_stage(stage: Path) -> None:
    """Reject special/aliased stage members, privatize them, and fsync all."""

    _privatize_owned_directory(stage, noun="native-resume stage root")
    entries = sorted(stage.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    if len(entries) > 128:
        raise SuccessorOperatorError("native-resume stage inventory exceeds its bound")
    for path in entries:
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise SuccessorOperatorError("native-resume stage custody differs")
        if stat.S_ISDIR(metadata.st_mode):
            os.chmod(path, 0o700)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            os.chmod(path, 0o600)
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            raise SuccessorOperatorError(
                "native-resume stage contains a special or aliased object"
            )
    for directory in [
        *[item for item in entries if item.is_dir()],
        stage,
    ]:
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _native_resume_stage_inventory(stage: Path) -> tuple[tuple[Any, ...], ...]:
    """Return an inode-bound inventory for final publication race checks."""

    inventory: list[tuple[Any, ...]] = []
    for path in sorted(stage.rglob("*")):
        metadata = os.lstat(path)
        kind = (
            "directory"
            if stat.S_ISDIR(metadata.st_mode)
            else "file" if stat.S_ISREG(metadata.st_mode) else "special"
        )
        inventory.append(
            (
                path.relative_to(stage).as_posix(),
                kind,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        )
    return tuple(inventory)


def _cleanup_native_resume_stage(stage: Path, *, publish_parent: Path) -> None:
    """Remove only an unpublished, owner-private stage created by this call."""

    try:
        metadata = os.lstat(stage)
    except FileNotFoundError:
        return
    try:
        valid_parent = stage.parent.resolve(strict=True) == publish_parent.resolve(
            strict=True
        )
    except OSError:
        valid_parent = False
    if (
        valid_parent
        and stage.name.startswith(SUCCESSOR_RUN_RELATIVE.name + ".stage-")
        and stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
    ):
        shutil.rmtree(stage)


def bootstrap_native_resume(root: Path = ROOT) -> dict[str, Any]:
    """Atomically publish run-v39 from the tracked candidate projection."""

    root = root.resolve(strict=True)
    paths = _paths(root)
    _require_ignored_successor(root)
    config, config_raw = _load_native_resume_config(root)
    continuity_before = _candidate_runtime_continuity(root)
    final_run = paths["successor_database"].parent
    try:
        os.lstat(final_run)
    except FileNotFoundError:
        pass
    else:
        raise SuccessorOperatorError("refusing to overwrite an existing successor")
    publish_parent = final_run.parent
    publish_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _privatize_owned_directory(
        publish_parent,
        noun="native-resume publication parent",
    )
    parent_descriptor = os.open(
        publish_parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    parent_before = os.fstat(parent_descriptor)
    stage = publish_parent / (
        f"{SUCCESSOR_RUN_RELATIVE.name}.stage-{uuid.uuid4().hex}"
    )
    os.mkdir(stage, mode=0o700)
    published = False
    try:
        materializer = importlib.import_module(
            "scripts."
            "materialize_logic_governed_compositional_verification_fabric_control_plane"
        )
        population = materializer.build_population(config, root=root)
        staged_config = _native_resume_stage_config(
            config,
            root=root,
            stage=stage,
        )
        bootstrap = materializer._materialize_canonical(
            staged_config,
            population,
            root=root,
            recheck_source=True,
        )
        staged_database = stage / "control.duckdb"
        staged_coordination = stage / "control.coordination.duckdb"
        staged_execution = stage / "control.execution.duckdb"
        staged_bootstrap = stage / "evidence" / "bootstrap" / "materialization.json"
        staged_provenance = stage / "evidence" / paths["provenance"].name
        for database in (staged_database, staged_coordination, staged_execution):
            if os.path.lexists(database.with_name(database.name + ".wal")):
                raise SuccessorOperatorError(
                    "native-resume materialization retained a live WAL"
                )
        projection = _verify_native_resume_projection(
            staged_database,
            config=config,
        )
        final_database_paths = {
            "control": SUCCESSOR_DATABASE_RELATIVE.as_posix(),
            "coordination": SUCCESSOR_DATABASE_RELATIVE.with_name(
                "control.coordination.duckdb"
            ).as_posix(),
            "execution": SUCCESSOR_DATABASE_RELATIVE.with_name(
                "control.execution.duckdb"
            ).as_posix(),
        }
        bootstrap = dict(bootstrap)
        bootstrap["database_paths"] = final_database_paths
        bootstrap.pop("receipt_cid", None)
        bootstrap["receipt_cid"] = _content_id(bootstrap)
        profile = _verify_profile(staged_database)
        _validate_native_bootstrap_receipt(
            bootstrap,
            config=config,
            database_paths=final_database_paths,
            source_head=str(population["source_head"]),
            repository_tree_id=str(population["repository_tree_id"]),
            population_root=str(population["population_root"]),
            plan_root_cid=str(population["plan_root_cid"]),
            schema_fingerprint=str(profile.get("schema_fingerprint") or ""),
            catalog_fingerprint=str(profile.get("catalog_fingerprint") or ""),
        )
        _atomic_json(staged_bootstrap, bootstrap, replace=True)

        identity = _database_identity(staged_database)
        config_after, config_raw_after = _load_native_resume_config(root)
        continuity_after = _candidate_runtime_continuity(root)
        if (
            config_after != config
            or config_raw_after != config_raw
            or continuity_after != continuity_before
            or population.get("source_head") != continuity_before.get("current_head")
            or population.get("repository_tree_id")
            != "git-tree:" + str(continuity_before.get("current_tree") or "")
        ):
            raise SuccessorOperatorError(
                "candidate source changed during native-resume materialization"
            )
        target_digest = _sha256_regular_file(staged_database)
        coordination_digest = _sha256_regular_file(staged_coordination)
        execution_digest = _sha256_regular_file(staged_execution)
        receipt = {
            "schema": PROVENANCE_SCHEMA,
            "issued_at": _utc_now(),
            "admission_mode": NATIVE_RESUME_ADMISSION_MODE,
            "source_generation": NATIVE_RESUME_SOURCE_GENERATION,
            "target_generation": SUCCESSOR_STORE_GENERATION,
            "source_database": "",
            "target_database": str(paths["successor_database"]),
            "source_head": str(population["source_head"]),
            "source_tree": str(continuity_before["current_tree"]),
            "source_forest_root": str(population["source_forest_root"]),
            "datasets_head": str(continuity_before["datasets_head"]),
            "datasets_tree": str(continuity_before["datasets_tree"]),
            "candidate_config_path": DEFAULT_SUCCESSOR_CONFIG_RELATIVE.as_posix(),
            "candidate_config_sha256": (
                "sha256:" + hashlib.sha256(config_raw).hexdigest()
            ),
            "population_root": str(population["population_root"]),
            "plan_root_cid": str(population["plan_root_cid"]),
            "initial_projection": copy.deepcopy(config["initial_projection"]),
            "materialized_projection": projection,
            "bootstrap_receipt_cid": str(bootstrap["receipt_cid"]),
            "bootstrap_verification_root": str(
                (bootstrap.get("verification") or {}).get("verification_root") or ""
            ),
            "target_initial_sha256": target_digest,
            "target_coordination_initial_sha256": coordination_digest,
            "target_execution_initial_sha256": execution_digest,
            "database_uuid": str(identity.get("database_uuid") or ""),
            "schema_fingerprint": str(profile.get("schema_fingerprint") or ""),
            "catalog_fingerprint": str(profile.get("catalog_fingerprint") or ""),
            "initial_projection_reset": True,
            "continuity_completion_records_imported": False,
            "source_database_statuses_read": False,
            "source_database_completion_records_imported": False,
            "quack_required_after_publish": True,
            "direct_multi_process_duckdb_permitted": False,
            "ducklake_projection_authoritative": False,
            "restart_requires_live_continuity_receipt": True,
            "live_continuity_receipt_implemented": False,
            "candidate_authored_validation": True,
            "validation_self_authority": False,
            "validation_completion_authoritative": False,
            "network_isolation_enforced": True,
            "model_provider_route": "none",
            "task_implementation_complete": False,
            "test_qualification_complete": False,
            "objective_complete": False,
            "release_qualified": False,
            "authoritative_for_release": False,
            "production_authorized": False,
        }
        receipt["receipt_cid"] = _content_id(receipt)
        _atomic_json(staged_provenance, receipt, replace=False)
        _privatize_and_sync_native_resume_stage(stage)
        _verify_native_resume_stage_allowlist(stage, include_provenance=True)
        if _strict_json(
            staged_bootstrap,
            expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
            require_private_owner=True,
        ) != bootstrap:
            raise SuccessorOperatorError(
                "native-resume bootstrap receipt replay differs"
            )
        stage_sealed = os.lstat(stage)
        sealed_inventory = _native_resume_stage_inventory(stage)
        if _strict_json(
            staged_provenance,
            expected_schema=PROVENANCE_SCHEMA,
            require_private_owner=True,
        ) != receipt:
            raise SuccessorOperatorError("native-resume provenance replay differs")
        _require_private_directory(stage, noun="native-resume stage root")
        _verify_native_resume_stage_allowlist(stage, include_provenance=True)
        parent_after = os.fstat(parent_descriptor)
        stage_after = os.lstat(stage)
        if (
            (parent_before.st_dev, parent_before.st_ino)
            != (parent_after.st_dev, parent_after.st_ino)
            or (
                stage_sealed.st_dev,
                stage_sealed.st_ino,
                stage_sealed.st_uid,
                stage_sealed.st_mode,
                stage_sealed.st_nlink,
            )
            != (
                stage_after.st_dev,
                stage_after.st_ino,
                stage_after.st_uid,
                stage_after.st_mode,
                stage_after.st_nlink,
            )
            or os.path.lexists(final_run)
            or _candidate_runtime_continuity(root) != continuity_before
            or _native_resume_stage_inventory(stage) != sealed_inventory
            or _sha256_regular_file(staged_database) != target_digest
            or _sha256_regular_file(staged_coordination) != coordination_digest
            or _sha256_regular_file(staged_execution) != execution_digest
            or _strict_json(
                staged_bootstrap,
                expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
                require_private_owner=True,
            )
            != bootstrap
            or _strict_json(
                staged_provenance,
                expected_schema=PROVENANCE_SCHEMA,
                require_private_owner=True,
            )
            != receipt
        ):
            raise SuccessorOperatorError(
                "native-resume publication boundary changed before rename"
            )
        _rename_directory_noreplace(
            parent_descriptor,
            stage.name,
            final_run.name,
        )
        published = True
        try:
            os.fsync(parent_descriptor)
        except OSError as exc:
            raise SuccessorOperatorError(
                "native resume published completely but parent durability is uncertain"
            ) from exc
        return receipt
    finally:
        os.close(parent_descriptor)
        if not published:
            _cleanup_native_resume_stage(stage, publish_parent=publish_parent)


def bootstrap_successor(root: Path = ROOT) -> dict[str, Any]:
    return bootstrap_native_resume(root)


def bootstrap_sealed_successor(
    *,
    root: Path,
    source_root: Path,
    control_sha256: str,
    coordination_sha256: str,
    execution_sha256: str,
    bootstrap_sha256: str,
    manifest_sha256: str,
    recovery_receipt_sha256: str,
) -> dict[str, Any]:
    _require_ignored_successor(root, run_relative=LEGACY_SUCCESSOR_RUN_RELATIVE)
    verification = verify_sealed_target_continuity(
        root=root,
        source_root=source_root,
        control_sha256=control_sha256,
        coordination_sha256=coordination_sha256,
        execution_sha256=execution_sha256,
        bootstrap_sha256=bootstrap_sha256,
        manifest_sha256=manifest_sha256,
        recovery_receipt_sha256=recovery_receipt_sha256,
    )
    source_paths = _sealed_source_paths(source_root)
    return clone_verified_successor(
        source_paths["control"],
        _contained(root, LEGACY_SUCCESSOR_RUN_RELATIVE / "control.duckdb"),
        _contained(
            root,
            LEGACY_SUCCESSOR_RUN_RELATIVE
            / "evidence"
            / "quack-successor-provenance.json",
        ),
        recovery_verification=verification,
    )


def _load_provenance(
    paths: Mapping[str, Path],
    *,
    root: Path = ROOT,
    expected_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    database = paths["successor_database"]
    coordination = database.with_name("control.coordination.duckdb")
    execution = database.with_name("control.execution.duckdb")
    _require_private_directory(database.parent, noun="successor generation")
    _require_private_directory(
        paths["provenance"].parent, noun="successor evidence directory"
    )
    for noun, store in (
        ("control", database),
        ("coordination", coordination),
        ("execution", execution),
    ):
        if os.path.lexists(store.with_name(store.name + ".wal")):
            raise SuccessorOperatorError(
                f"successor {noun} database has a live WAL"
            )
    initial_receipt = _strict_json(
        paths["provenance"],
        expected_schema=PROVENANCE_SCHEMA,
        require_private_owner=True,
        verify_content_identity=False,
    )
    if expected_receipt is not None and initial_receipt != dict(expected_receipt):
        raise SuccessorOperatorError(
            "verified successor provenance differs from native authorization"
        )
    if initial_receipt.get("admission_mode") == SEALED_CONTINUITY_MODE:
        _candidate_runtime_continuity(root)
    receipt = _strict_json(
        paths["provenance"],
        expected_schema=PROVENANCE_SCHEMA,
        require_private_owner=True,
    )
    if receipt != initial_receipt:
        raise SuccessorOperatorError("successor provenance changed during admission")
    target_digest = _sha256_regular_file(
        database,
        noun="successor control database",
        require_private_owner=True,
    )
    admission_mode = str(receipt.get("admission_mode") or "")
    if receipt.get("target_database") != str(database):
        raise SuccessorOperatorError("successor provenance target differs")
    source_database: Path | None = None
    if admission_mode == NATIVE_RESUME_ADMISSION_MODE:
        config, config_raw = _load_native_resume_config(root)
        continuity = _candidate_runtime_continuity(root)
        source_head_value = receipt.get("source_head")
        source_tree_value = receipt.get("source_tree")
        source_head = source_head_value if type(source_head_value) is str else ""
        source_tree = source_tree_value if type(source_tree_value) is str else ""
        native_fields = {
            "schema",
            "issued_at",
            "admission_mode",
            "source_generation",
            "target_generation",
            "source_database",
            "target_database",
            "source_head",
            "source_tree",
            "source_forest_root",
            "datasets_head",
            "datasets_tree",
            "candidate_config_path",
            "candidate_config_sha256",
            "population_root",
            "plan_root_cid",
            "initial_projection",
            "materialized_projection",
            "bootstrap_receipt_cid",
            "bootstrap_verification_root",
            "target_initial_sha256",
            "target_coordination_initial_sha256",
            "target_execution_initial_sha256",
            "database_uuid",
            "schema_fingerprint",
            "catalog_fingerprint",
            "initial_projection_reset",
            "continuity_completion_records_imported",
            "source_database_statuses_read",
            "source_database_completion_records_imported",
            "quack_required_after_publish",
            "direct_multi_process_duckdb_permitted",
            "ducklake_projection_authoritative",
            "restart_requires_live_continuity_receipt",
            "live_continuity_receipt_implemented",
            "candidate_authored_validation",
            "validation_self_authority",
            "validation_completion_authoritative",
            "network_isolation_enforced",
            "model_provider_route",
            "task_implementation_complete",
            "test_qualification_complete",
            "objective_complete",
            "release_qualified",
            "authoritative_for_release",
            "production_authorized",
            "receipt_cid",
        }

        def native_content_cid(field: str) -> bool:
            value = receipt.get(field)
            return (
                type(value) is str
                and re.fullmatch(r"b[a-z2-7]{60}", value) is not None
            )

        def native_sha256(field: str) -> bool:
            value = receipt.get(field)
            return (
                type(value) is str
                and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
            )

        if (
            set(receipt) != native_fields
            or receipt.get("source_generation") != NATIVE_RESUME_SOURCE_GENERATION
            or receipt.get("target_generation") != SUCCESSOR_STORE_GENERATION
            or receipt.get("source_database") != ""
            or type(receipt.get("issued_at")) is not str
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
                receipt["issued_at"],
            )
            is None
            or source_head != continuity.get("current_head")
            or source_tree != continuity.get("current_tree")
            or receipt.get("datasets_head") != continuity.get("datasets_head")
            or receipt.get("datasets_tree") != continuity.get("datasets_tree")
            or receipt.get("candidate_config_path")
            != DEFAULT_SUCCESSOR_CONFIG_RELATIVE.as_posix()
            or receipt.get("candidate_config_sha256")
            != "sha256:" + hashlib.sha256(config_raw).hexdigest()
            or receipt.get("initial_projection") != config.get("initial_projection")
            or not native_content_cid("source_forest_root")
            or not native_content_cid("population_root")
            or not native_content_cid("plan_root_cid")
            or not native_content_cid("bootstrap_receipt_cid")
            or not native_content_cid("bootstrap_verification_root")
            or not native_content_cid("schema_fingerprint")
            or not native_content_cid("catalog_fingerprint")
            or not native_content_cid("receipt_cid")
            or not native_sha256("target_initial_sha256")
            or not native_sha256("target_coordination_initial_sha256")
            or not native_sha256("target_execution_initial_sha256")
            or type(receipt.get("database_uuid")) is not str
            or not str(receipt.get("database_uuid") or "")
            or receipt.get("initial_projection_reset") is not True
            or receipt.get("continuity_completion_records_imported") is not False
            or receipt.get("source_database_statuses_read") is not False
            or receipt.get("source_database_completion_records_imported") is not False
            or receipt.get("quack_required_after_publish") is not True
            or receipt.get("direct_multi_process_duckdb_permitted") is not False
            or receipt.get("ducklake_projection_authoritative") is not False
            or receipt.get("restart_requires_live_continuity_receipt") is not True
            or receipt.get("live_continuity_receipt_implemented") is not False
            or receipt.get("candidate_authored_validation") is not True
            or receipt.get("validation_self_authority") is not False
            or receipt.get("validation_completion_authoritative") is not False
            or receipt.get("network_isolation_enforced") is not True
            or receipt.get("model_provider_route") != "none"
            or receipt.get("task_implementation_complete") is not False
            or receipt.get("test_qualification_complete") is not False
            or receipt.get("objective_complete") is not False
            or receipt.get("release_qualified") is not False
            or receipt.get("authoritative_for_release") is not False
            or receipt.get("production_authorized") is not False
            or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
            or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
            or _git_text(
                root,
                ("show", "-s", "--format=%T", source_head),
                noun="native-resume source commit",
            )
            != source_tree
        ):
            raise SuccessorOperatorError("native-resume provenance binding differs")
        _git_quiet(
            root,
            (
                "merge-base",
                "--is-ancestor",
                source_head,
                str(continuity.get("current_head") or ""),
            ),
            noun="native-resume source ancestry",
        )
        bootstrap_path = (
            database.parent / "evidence" / "bootstrap" / "materialization.json"
        )
        bootstrap = _strict_json(
            bootstrap_path,
            expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
            require_private_owner=True,
        )
        _validate_native_bootstrap_receipt(
            bootstrap,
            config=config,
            database_paths={
                "control": SUCCESSOR_DATABASE_RELATIVE.as_posix(),
                "coordination": SUCCESSOR_DATABASE_RELATIVE.with_name(
                    "control.coordination.duckdb"
                ).as_posix(),
                "execution": SUCCESSOR_DATABASE_RELATIVE.with_name(
                    "control.execution.duckdb"
                ).as_posix(),
            },
            source_head=source_head,
            repository_tree_id="git-tree:" + source_tree,
            population_root=receipt["population_root"],
            plan_root_cid=receipt["plan_root_cid"],
            schema_fingerprint=receipt["schema_fingerprint"],
            catalog_fingerprint=receipt["catalog_fingerprint"],
        )
        bootstrap_verification = bootstrap["verification"]
        assert isinstance(bootstrap_verification, Mapping)
        if (
            bootstrap.get("receipt_cid") != receipt.get("bootstrap_receipt_cid")
            or bootstrap_verification.get("verification_root")
            != receipt.get("bootstrap_verification_root")
            or bootstrap.get("population_root") != receipt.get("population_root")
            or bootstrap.get("plan_root_cid") != receipt.get("plan_root_cid")
        ):
            raise SuccessorOperatorError(
                "native-resume bootstrap/provenance cross-binding differs"
            )
        if (
            target_digest != receipt.get("target_initial_sha256")
            or _sha256_regular_file(
                coordination,
                noun="native-resume coordination database",
                require_private_owner=True,
            )
            != receipt.get("target_coordination_initial_sha256")
            or _sha256_regular_file(
                execution,
                noun="native-resume execution database",
                require_private_owner=True,
            )
            != receipt.get("target_execution_initial_sha256")
        ):
            raise SuccessorOperatorError(
                "native-resume state changed after initial admission; restart "
                "requires an unimplemented live-continuity receipt"
            )
        projection = _verify_native_resume_projection(database, config=config)
        if projection != receipt.get("materialized_projection"):
            raise SuccessorOperatorError(
                "native-resume initial projection replay differs"
            )
    elif admission_mode == "canonical_fresh_generation_recovery":
        source_database = paths["source_database"]
        if receipt.get("source_database") != str(source_database):
            raise SuccessorOperatorError("successor provenance no longer binds run-v17")
    elif admission_mode == SEALED_CONTINUITY_MODE:
        if (
            receipt.get("authority_ceiling") != SEALED_CONTINUITY_AUTHORITY_CEILING
            or receipt.get("fresh_source_evidence_revalidated") is not False
            or receipt.get("historical_source_bytes_revalidated") is not False
            or receipt.get("source_provenance_authoritative") is not False
            or receipt.get("target_snapshot_hash_pinned") is not True
            or receipt.get("target_database_statuses_read") is not True
            or receipt.get("source_database_statuses_read_scope")
            != "lost_fresh_recovery_source_generation_lgcvf-run-v16"
            or receipt.get("restart_requires_live_continuity_receipt") is not True
            or receipt.get("live_continuity_receipt_implemented") is not False
            or receipt.get("authoritative_for_release") is not False
            or receipt.get("production_authorized") is not False
            or receipt.get("source_generation") != "lgcvf-run-v17"
            or receipt.get("target_generation") != "lgcvf-run-v23"
            or receipt.get("clone_preserves_database_uuid") is not True
            or receipt.get("owner_generation_rotates_on_start") is not True
        ):
            raise SuccessorOperatorError("sealed successor authority ceiling differs")
        _require_false_authority(receipt, noun="sealed successor provenance")
        sealed = _sealed_source_paths(Path(str(receipt.get("source_root") or "")))
        source_database = sealed["control"]
        expected_manifest = (
            sealed["recovery_root"]
            / f"{receipt.get('recovery_manifest_cid')}.manifest.json"
        )
        expected_paths = {
            "source_database": sealed["control"],
            "source_coordination_database": sealed["coordination"],
            "source_execution_database": sealed["execution"],
            "source_bootstrap_receipt": sealed["bootstrap"],
            "source_recovery_receipt": sealed["recovery_receipt"],
            "source_recovery_manifest": expected_manifest,
        }
        if any(
            receipt.get(field) != str(path) for field, path in expected_paths.items()
        ):
            raise SuccessorOperatorError("sealed successor source path binding differs")
        sealed_hashes = {
            "source_sha256": _sha256_regular_file(
                sealed["control"],
                noun="sealed control database",
                require_private_owner=True,
            ),
            "source_coordination_sha256": _sha256_regular_file(
                sealed["coordination"],
                noun="sealed coordination database",
                require_private_owner=True,
            ),
            "source_execution_sha256": _sha256_regular_file(
                sealed["execution"],
                noun="sealed execution database",
                require_private_owner=True,
            ),
            "source_bootstrap_sha256": _sha256_regular_file(
                sealed["bootstrap"],
                max_bytes=MAX_JSON_BYTES,
                noun="sealed bootstrap receipt",
                require_private_owner=True,
            ),
            "source_recovery_manifest_sha256": _sha256_regular_file(
                expected_manifest,
                max_bytes=MAX_JSON_BYTES,
                noun="sealed recovery manifest",
                require_private_owner=True,
            ),
            "source_recovery_receipt_sha256": _sha256_regular_file(
                sealed["recovery_receipt"],
                max_bytes=MAX_JSON_BYTES,
                noun="sealed recovery receipt",
                require_private_owner=True,
            ),
        }
        if any(receipt.get(field) != digest for field, digest in sealed_hashes.items()):
            raise SuccessorOperatorError("sealed successor source hash binding differs")
        if sealed_hashes != {
            "source_sha256": SEALED_CONTINUITY_EXPECTED_PINS["control_sha256"],
            "source_coordination_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "coordination_sha256"
            ],
            "source_execution_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "execution_sha256"
            ],
            "source_bootstrap_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "bootstrap_sha256"
            ],
            "source_recovery_manifest_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "manifest_sha256"
            ],
            "source_recovery_receipt_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "recovery_receipt_sha256"
            ],
        }:
            raise SuccessorOperatorError("sealed successor reviewed pins differ")
        refreshed = verify_sealed_target_continuity(
            root=root,
            source_root=sealed["root"],
            **SEALED_CONTINUITY_EXPECTED_PINS,
        )
        source_binding = refreshed.get("source_binding") or {}
        semantic_bindings = {
            "recovery_verification_root": refreshed.get("verification_root"),
            "recovery_receipt_cid": refreshed.get("receipt_cid"),
            "recovery_manifest_cid": refreshed.get("manifest_cid"),
            "bootstrap_receipt_cid": refreshed.get("bootstrap_receipt_cid"),
            "source_evidence_cid": refreshed.get("source_evidence_cid"),
            "population_root": refreshed.get("population_root"),
            "plan_root_cid": refreshed.get("plan_root_cid"),
            "target_source_head": source_binding.get("target_source_head"),
            "target_source_tree": source_binding.get("target_source_tree"),
            "sealed_operational_verification_root": refreshed.get(
                "sealed_operational_verification_root"
            ),
        }
        if any(
            receipt.get(field) != expected
            for field, expected in semantic_bindings.items()
        ):
            raise SuccessorOperatorError(
                "sealed successor provenance cross-binding differs"
            )
        if (
            target_digest != receipt.get("target_initial_sha256")
            or target_digest != SEALED_CONTINUITY_EXPECTED_PINS["control_sha256"]
        ):
            raise SuccessorOperatorError(
                "sealed successor changed after its initial admission; restart "
                "requires an unimplemented live-continuity receipt"
            )
    else:
        raise SuccessorOperatorError("successor provenance admission mode differs")
    if source_database is not None and _sha256_regular_file(
        source_database,
        noun="successor provenance source database",
        require_private_owner=admission_mode == SEALED_CONTINUITY_MODE,
    ) != receipt.get("source_sha256"):
        raise SuccessorOperatorError("successor provenance no longer binds run-v17")
    verification = _verify_profile(database)
    identity = _database_identity(database)
    if (
        verification.get("schema_fingerprint") != receipt.get("schema_fingerprint")
        or verification.get("catalog_fingerprint") != receipt.get("catalog_fingerprint")
        or identity.get("database_uuid") != receipt.get("database_uuid")
    ):
        raise SuccessorOperatorError(
            "successor database identity differs from provenance"
        )
    return receipt


def _load_lgcvf_live_raw_provenance_receipt(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Read the content-addressed receipt without importing the database stack."""

    _require_private_directory(
        paths["provenance"].parent,
        noun="successor evidence directory",
    )
    receipt = _strict_json(
        paths["provenance"],
        expected_schema=PROVENANCE_SCHEMA,
        require_private_owner=True,
    )
    receipt_cid = receipt.get("receipt_cid")
    if (
        type(receipt_cid) is not str
        or re.fullmatch(r"[a-z2-7]{32,256}", receipt_cid) is None
        or receipt.get("target_database") != str(paths["successor_database"])
        or (
            receipt.get("source_generation"),
            receipt.get("target_generation"),
            receipt.get("admission_mode"),
        )
        not in {
            (
                NATIVE_RESUME_SOURCE_GENERATION,
                SUCCESSOR_STORE_GENERATION,
                NATIVE_RESUME_ADMISSION_MODE,
            ),
        }
    ):
        raise SuccessorOperatorError(
            "raw successor provenance is not the exact live generation receipt"
        )
    return receipt


def _parse_quack_endpoint(endpoint: str) -> tuple[str, int]:
    match = re.fullmatch(r"quack:(?://)?(127\.0\.0\.1|localhost):(\d{1,5})", endpoint)
    if match is None or not 1 <= int(match.group(2)) <= 65535:
        raise SuccessorOperatorError("successor Quack endpoint must be fixed loopback")
    return match.group(1), int(match.group(2))


def _validate_successor_board(
    config_path: Path,
    root: Path = ROOT,
    *,
    config_bytes: bytes | None = None,
    admitted_live_validator_sha256: str = "",
) -> tuple[Any, Any, str, int]:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        load_configured_board,
        preflight_configured_board,
    )

    board = load_configured_board(
        config_path,
        repo_root=root,
        config_bytes=config_bytes,
    )
    program = board.resolved_database_program()
    raw_program = board.payload.get("database_program")
    expected_store = SUCCESSOR_DATABASE_RELATIVE.as_posix()
    expected_registry = OWNER_STATE_RELATIVE.as_posix()
    provider = board.payload.get("provider")
    bootstrap = board.payload.get("bootstrap_writer_policy")
    projection = board.payload.get("ducklake_projection_program")
    if (
        board.max_lanes != 4
        or board.board_namespace
        != "logic-governed-compositional-verification-fabric-v1"
        or program.authority_mode != "quack"
        or program.task_source_kind != "duckdb"
        or program.failover_policy != "fail_closed"
        or program.endpoint_secret_handle != SECRET_HANDLE
        or program.store_id != expected_store
        or program.runtime_registry_path != expected_registry
        or program.store_generation != SUCCESSOR_STORE_GENERATION
        or program.schema_revision != "datasets-authoritative-operational-v1"
        or not isinstance(raw_program, Mapping)
        or raw_program.get("schema_profile") != "datasets-authoritative-operational"
        or board.runtime_paths.get("root") != SUCCESSOR_RUN_RELATIVE.as_posix()
        or not isinstance(provider, Mapping)
        or provider.get("primary_provider_id") != "grok_cli"
        or provider.get("primary_model_id") != "grok-4.6"
        or provider.get("fallback_provider_id") != "codex"
        or provider.get("fallback_model_id") != "gpt-5.6-terra"
        or provider.get("fallback_trigger") != "primary_quota_exhausted"
        or provider.get("fallback_reasoning_effort") != "high"
        or provider.get("max_concurrency") != 4
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("maximum_processes") != 1
        or bootstrap.get("quack_required") is not False
        or bootstrap.get("offline_single_writer_materialization_permitted") is not True
        or bootstrap.get("quack_required_after_publish") is not True
        or bootstrap.get("direct_multi_process_duckdb_permitted") is not False
        or not isinstance(projection, Mapping)
        or projection.get("root") != PROJECTION_ROOT_RELATIVE.as_posix()
        or projection.get("catalog_path")
        != (PROJECTION_ROOT_RELATIVE / "lake.ducklake").as_posix()
        or projection.get("data_path")
        != (PROJECTION_ROOT_RELATIVE / "lake-data").as_posix()
        or projection.get("authority") is not False
        or projection.get("scheduling_prerequisite") is not False
        or projection.get("completion_prerequisite") is not False
        or "fresh_generation_recovery" in board.payload
    ):
        raise SuccessorOperatorError(
            "scheduler config is not the exact four-lane successor"
        )
    host, port = _parse_quack_endpoint(program.quack_endpoint)
    preflight = preflight_configured_board(
        board,
        admitted_live_validator_sha256=admitted_live_validator_sha256,
    )
    if preflight.get("valid") is not True:
        raise SuccessorOperatorError(
            "configured-board preflight failed: "
            + ", ".join(preflight.get("errors") or ())
        )
    return board, program, host, port


def _status_payload(
    *,
    lifecycle: str,
    controller_birth: Mapping[str, Any],
    provenance_cid: str,
    owner_identity: Mapping[str, Any] | None = None,
    scheduler_birth: Mapping[str, Any] | None = None,
    scheduler_returncode: int | None = None,
    error: str = "",
    projection_root: Path | None = None,
) -> dict[str, Any]:
    observed_projection_root = (
        _paths()["projection_root"]
        if projection_root is None
        else Path(projection_root).resolve()
    )
    payload: dict[str, Any] = {
        "schema": CONTROLLER_STATUS_SCHEMA,
        "lifecycle": lifecycle,
        "updated_at": _utc_now(),
        "controller_birth": dict(controller_birth),
        "provenance_cid": provenance_cid,
        "owner_identity": dict(owner_identity or {}),
        "scheduler_birth": dict(scheduler_birth or {}),
        "scheduler_returncode": scheduler_returncode,
        "error": error,
        "ducklake_projection": {
            "path": str(observed_projection_root),
            "control_catalog_path": str(observed_projection_root / "control.duckdb"),
            "ducklake_catalog_path": str(observed_projection_root / "lake.ducklake"),
            "ducklake_data_path": str(observed_projection_root / "lake-data"),
            "authoritative": False,
            "read_by_scheduler": False,
            "scheduling_authority": False,
            "completion_authority": False,
            "live_quack_endpoint": False,
            "mode": "separate_stopped_checkpoint",
        },
    }
    payload["status_cid"] = _content_id(payload)
    return payload


def _write_status(path: Path, payload: Mapping[str, Any], *, token: str = "") -> None:
    encoded = _canonical_bytes(payload)
    if token and token.encode("ascii") in encoded:
        raise SuccessorOperatorError("Quack token would enter controller status")
    _atomic_json(path, payload, replace=True)


def _token_sink(owner_state: Path) -> Path:
    """Return an impossible child path so legacy helpers cannot persist the token."""

    marker = owner_state / ".ephemeral-token-persistence-disabled"
    payload = b"trusted controller keeps the Quack attach credential in memory\n"
    if marker.exists():
        observed = os.lstat(marker)
        if not stat.S_ISREG(observed.st_mode) or marker.read_bytes() != payload:
            raise SuccessorOperatorError("ephemeral token sink marker is unsafe")
    else:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(marker, 0o400)
    # The parent component is a regular file, so mkdir/open in the legacy
    # persistence helper fails without ever creating credential material.
    return marker / "unavailable"


def _prepare_private_owner_socket(socket_path: Path) -> None:
    """Admit one short same-UID directory without following a symlink."""

    path = Path(socket_path)
    temporary_root = Path(tempfile.gettempdir()).resolve()
    parent = path.parent
    if (
        not path.is_absolute()
        or parent.parent.resolve() != temporary_root
        or parent.name != f"ipfs-accelerate-lgcvf-{os.geteuid()}"
        or not path.name.startswith("owner-")
        or not path.name.endswith(".sock")
        or len(os.fsencode(path)) > UNIX_SOCKET_PATH_CEILING
    ):
        raise SuccessorOperatorError("state-owner socket identity is unsafe")
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    try:
        metadata = os.lstat(parent)
    except OSError as exc:
        raise SuccessorOperatorError(
            "state-owner socket directory is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SuccessorOperatorError("state-owner socket directory custody is unsafe")
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SuccessorOperatorError("state-owner socket cannot be inspected") from exc
    if (
        not stat.S_ISSOCK(existing.st_mode)
        or stat.S_ISLNK(existing.st_mode)
        or existing.st_uid != os.geteuid()
        or existing.st_nlink != 1
        or stat.S_IMODE(existing.st_mode) & 0o077
    ):
        raise SuccessorOperatorError("existing state-owner socket custody is unsafe")


def _installed_extension_version(info_path: Path, *, name: str) -> str:
    """Read the sole short hexadecimal build identity from DuckDB metadata."""

    raw = _read_bounded_regular_file(
        info_path,
        max_bytes=64 * 1024,
        noun=f"installed {name} extension metadata",
    )
    versions = tuple(
        match.decode("ascii")
        for match in re.findall(
            rb"(?<![0-9a-f])([0-9a-f]{7,8})(?![0-9a-f])",
            raw,
        )
    )
    if len(versions) != 1:
        raise SuccessorOperatorError(
            f"installed {name} extension build identity is ambiguous"
        )
    return versions[0]


def _resolve_installed_duckdb_live_runtime() -> dict[str, Any]:
    """Resolve the exact installed DuckDB facade, native ELF, and extensions.

    Resolution deliberately uses import metadata and module specs without
    importing DuckDB.  Native module creation is permitted only after the
    capsule/native/admission join has been verified.
    """

    try:
        distribution = importlib.metadata.distribution("duckdb")
        version = str(distribution.version)
        site_root = Path(distribution.locate_file("")).resolve(strict=True)
        metadata_value = getattr(distribution, "_path", None)
        if metadata_value is None:
            raise SuccessorOperatorError(
                "installed DuckDB distribution metadata root is unavailable"
            )
        metadata_root = Path(metadata_value).resolve(strict=True)
        package_root = (site_root / "duckdb").resolve(strict=True)
        native_spec = importlib.util.find_spec("_duckdb")
        native_origin = getattr(native_spec, "origin", None)
        if not isinstance(native_origin, str) or not native_origin:
            raise SuccessorOperatorError("installed DuckDB native module is absent")
        native_path = Path(native_origin).resolve(strict=True)
    except (ImportError, importlib.metadata.PackageNotFoundError, OSError) as exc:
        raise SuccessorOperatorError(
            "installed DuckDB runtime cannot be resolved"
        ) from exc
    if (
        re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,63}", version) is None
        or metadata_root.parent != site_root
        or metadata_root.name != f"duckdb-{version}.dist-info"
        or package_root.parent != site_root
        or package_root.name != "duckdb"
        or native_path.parent != site_root
        or not native_path.name.startswith("_duckdb.")
        or not native_path.name.endswith(".so")
    ):
        raise SuccessorOperatorError("installed DuckDB runtime layout differs")
    machine = os.uname().machine if hasattr(os, "uname") else ""
    platform_name = {
        "aarch64": "linux_arm64",
        "x86_64": "linux_amd64",
    }.get(machine, "")
    ambient_home = str(os.environ.get("HOME", "") or "")
    if (
        not platform_name
        or not ambient_home
        or "\x00" in ambient_home
        or not Path(ambient_home).is_absolute()
    ):
        raise SuccessorOperatorError(
            "installed DuckDB extension platform is unavailable"
        )
    extension_root = (
        Path(ambient_home)
        / ".duckdb"
        / "extensions"
        / f"v{version}"
        / platform_name
    )
    try:
        extension_root = extension_root.resolve(strict=True)
    except OSError as exc:
        raise SuccessorOperatorError(
            "installed DuckDB extension directory is unavailable"
        ) from exc
    extensions: dict[str, dict[str, str | Path]] = {}
    for name in ("quack", "ducklake", "httpfs"):
        path = extension_root / f"{name}.duckdb_extension"
        info_path = Path(str(path) + ".info")
        extensions[name] = {
            "path": path,
            "version": _installed_extension_version(info_path, name=name),
        }
    return {
        "version": version,
        "engine_version": f"v{version}",
        "package_root": package_root,
        "metadata_root": metadata_root,
        "native_path": native_path,
        "extension_platform": platform_name,
        "quack_path": extensions["quack"]["path"],
        "quack_version": extensions["quack"]["version"],
        "ducklake_path": extensions["ducklake"]["path"],
        "ducklake_version": extensions["ducklake"]["version"],
        "httpfs_path": extensions["httpfs"]["path"],
        "httpfs_version": extensions["httpfs"]["version"],
    }


def _lgcvf_live_native_authorization_id(
    *,
    native_pin: Any,
    provenance: Mapping[str, Any],
    source_head: str,
    source_tree: str,
    candidate_config_sha256: str,
) -> str:
    """Bind native evidence to this exact source/config/provenance admission."""

    pin_payload = getattr(native_pin, "as_dict", lambda: None)()
    receipt_cid = provenance.get("receipt_cid")
    if (
        not isinstance(pin_payload, Mapping)
        or not isinstance(receipt_cid, str)
        or not receipt_cid
        or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", candidate_config_sha256)
        is None
    ):
        raise SuccessorOperatorError(
            "LGCVF native launch authorization inputs are incomplete"
        )
    body = {
        "schema": LGCVF_LIVE_NATIVE_AUTHORIZATION_SCHEMA,
        "board_namespace": (
            "logic-governed-compositional-verification-fabric-v1"
        ),
        "target_generation": SUCCESSOR_STORE_GENERATION,
        "source_head": source_head,
        "source_tree": source_tree,
        "candidate_config_path": DEFAULT_SUCCESSOR_CONFIG_RELATIVE.as_posix(),
        "candidate_config_sha256": candidate_config_sha256,
        "successor_provenance_cid": receipt_cid,
        "native_pin": dict(pin_payload),
        "claims": {
            "capsule_exact_match_required": True,
            "parent_loader_environment_sanitized_before_exec": True,
            "quack_extension_install_policy": "load_only",
            "ducklake_authority": False,
        },
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _remove_private_live_capsule_parent(path: Path | None) -> None:
    """Remove only this launch's owner-private temporary capsule directory."""

    if path is None:
        return
    parent = Path(path)
    try:
        temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
        observed_parent = os.lstat(parent)
        if (
            parent.parent.resolve(strict=True) != temporary_root
            or not parent.name.startswith(
                f"lgcvf-live-capsule-{os.geteuid()}-"
            )
            or not stat.S_ISDIR(observed_parent.st_mode)
            or stat.S_ISLNK(observed_parent.st_mode)
            or observed_parent.st_uid != os.geteuid()
        ):
            return
        entries = tuple(parent.rglob("*"))
        if any(
            stat.S_ISLNK(os.lstat(entry).st_mode)
            or os.lstat(entry).st_uid != os.geteuid()
            for entry in entries
        ):
            return
        for directory in sorted(
            (entry for entry in entries if entry.is_dir()),
            key=lambda entry: len(entry.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o700)
        os.chmod(parent, 0o700)
        shutil.rmtree(parent)
    except OSError:
        return


def _child_environment(
    *,
    token: str,
    identity: Any,
    owner_state: Path,
    root: Path,
    rendered_environment: Mapping[str, Any] | None = None,
    launch_home: Path | None = None,
) -> dict[str, str]:
    rendered = dict(rendered_environment or {})
    if (
        TOKEN_ENV in rendered
        or TOKEN_FILE_ENV in rendered
        or not set(rendered).issubset(LGCVF_LIVE_RENDERED_ENV_NAMES)
        or any(
            not isinstance(name, str)
            or not isinstance(value, (str, int, float))
            or "\x00" in str(value)
            for name, value in rendered.items()
        )
    ):
        raise SuccessorOperatorError(
            "configured scheduler rendered a foreign environment field"
        )
    environment = {
        name: str(os.environ[name])
        for name in ("LANG", "LC_ALL", "LC_CTYPE", "TZ")
        if name in os.environ and "\x00" not in str(os.environ[name])
    }
    environment["PATH"] = "/usr/bin:/bin"
    environment.update({str(name): str(value) for name, value in rendered.items()})
    environment[TOKEN_ENV] = token
    environment[TOKEN_FILE_ENV] = str(_token_sink(owner_state))
    environment["IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION"] = str(
        identity.generation
    )
    environment["IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION"] = str(
        identity.schema_revision
    )
    environment["IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT"] = str(root)
    environment[BOARD_EXTENSION_INSTALL_POLICY_ENV] = (
        BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
    )
    home = Path(launch_home) if launch_home is not None else owner_state
    environment["HOME"] = str(home)
    if launch_home is not None:
        environment["IPFS_ACCELERATE_AGENT_TRUSTED_DUCKDB_HOME"] = str(home)
        environment["XDG_CACHE_HOME"] = str(home / ".cache" / "xdg")
        environment["CUDA_CACHE_PATH"] = str(home / ".cache" / "cuda")
        environment["CUDA_CACHE_DISABLE"] = "1"
    if any(
        name.startswith(("LD_", "PYTHON")) or name == "GLIBC_TUNABLES"
        for name in environment
    ):
        raise SuccessorOperatorError(
            "scheduler environment retained ambient loader or Python authority"
        )
    return environment


def _exact_birth(pid: int) -> Any:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        read_process_birth,
    )

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        birth = read_process_birth(pid)
        if birth is not None:
            return birth
        time.sleep(0.01)
    raise SuccessorOperatorError("could not capture scheduler process birth")


def _terminate_exact(
    birth: Any,
    *,
    grace_seconds: float = 10.0,
    child_process: subprocess.Popen[Any] | None = None,
) -> str:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        owner_liveness,
    )

    def send(signum: int) -> None:
        if child_process is not None and child_process.poll() is not None:
            return
        state = owner_liveness(birth)
        if state is OwnerLiveness.DEAD:
            return
        if state is not OwnerLiveness.ALIVE:
            raise SuccessorOperatorError("scheduler birth is uninspectable")
        if birth.pid <= 1:
            raise SuccessorOperatorError("refusing to signal an unsafe PID")
        try:
            group = os.getpgid(birth.pid)
            if group == birth.pid:
                os.killpg(group, signum)
            else:
                os.kill(birth.pid, signum)
        except ProcessLookupError:
            return

    if child_process is not None:
        if child_process.pid != birth.pid:
            raise SuccessorOperatorError("scheduler child differs from its birth")
        if child_process.poll() is not None:
            return "already_dead"
    if owner_liveness(birth) is OwnerLiveness.DEAD:
        return "already_dead"
    send(signal.SIGTERM)
    deadline = time.monotonic() + max(0.1, grace_seconds)
    while time.monotonic() < deadline:
        if child_process is not None and child_process.poll() is not None:
            return "terminated"
        state = owner_liveness(birth)
        if state is OwnerLiveness.DEAD:
            return "terminated"
        if state is OwnerLiveness.UNKNOWN:
            raise SuccessorOperatorError("scheduler became uninspectable during stop")
        time.sleep(0.05)
    send(signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if child_process is not None and child_process.poll() is not None:
            return "killed"
        state = owner_liveness(birth)
        if state is OwnerLiveness.DEAD:
            return "killed"
        if state is OwnerLiveness.UNKNOWN:
            raise SuccessorOperatorError("scheduler became uninspectable after kill")
        time.sleep(0.05)
    raise SuccessorOperatorError("exact scheduler birth survived bounded stop")


def run_successor(
    config_path: Path,
    *,
    root: Path = ROOT,
    implement: bool,
    duration_seconds: float,
) -> int:
    paths = _paths(root)
    lock_handle = _open_private_lock(paths["controller_lock"])
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuccessorOperatorError(
                "another successor controller owns the lock"
            ) from exc
        return _run_locked_successor(
            config_path,
            root=root,
            implement=implement,
            duration_seconds=duration_seconds,
        )
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def _preload_lgcvf_live_controller_dependency_closure() -> tuple[str, ...]:
    """Import every repository module the live controller can call later."""

    loaded: list[str] = []
    for module_name in LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise SuccessorOperatorError(
                "LGCVF live controller dependency closure is unavailable: "
                f"{module_name}: {type(exc).__name__}"
            ) from exc
        if getattr(module, "__name__", None) != module_name:
            raise SuccessorOperatorError(
                "LGCVF live controller dependency identity differs: "
                f"{module_name}"
            )
        loaded.append(module_name)
    return tuple(loaded)


def _lgcvf_live_module_expected_members(module_name: str) -> frozenset[str]:
    """Return the only capsule members allowed to implement one module name."""

    if module_name == "ipfs_datasets_py":
        return frozenset(
            {
                "ipfs_datasets_py/__init__.py",
                "ipfs_datasets_py/ipfs_datasets_py/__init__.py",
            }
        )
    if module_name.startswith("ipfs_datasets_py."):
        stem = "ipfs_datasets_py/" + module_name.replace(".", "/")
    elif any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in ("ipfs_accelerate_py", "scripts")
    ):
        stem = module_name.replace(".", "/")
    else:
        return frozenset()
    return frozenset({stem + ".py", stem + "/__init__.py"})


def _lgcvf_live_manifest_member(
    relative: str,
    *,
    manifest_files: Mapping[str, str],
    read_member: Any,
    noun: str,
) -> bytes:
    """Read one member and bind it to the authenticated manifest digest."""

    digest = manifest_files.get(relative)
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
    ):
        raise SuccessorOperatorError(f"{noun} is absent from the sealed capsule")
    try:
        raw = read_member(relative)
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SuccessorOperatorError(
            f"{noun} cannot be read from the sealed capsule"
        ) from exc
    if (
        type(raw) is not bytes
        or len(raw) > 64 * 1024 * 1024
        or "sha256:" + hashlib.sha256(raw).hexdigest() != digest
    ):
        raise SuccessorOperatorError(f"{noun} sealed capsule bytes differ")
    return raw


def _lgcvf_live_sealed_manifest_inventory(
    capsule_pin: Any,
    capsule_descriptor: int,
) -> tuple[str, dict[str, str]]:
    """Read the already-verified manifest from the immutable archive itself."""

    from ipfs_accelerate_py.agent_implementation_route import (
        verify_lgcvf_configured_board_live_sealed_capsule,
    )

    try:
        archive_path = verify_lgcvf_configured_board_live_sealed_capsule(
            capsule_pin,
            capsule_descriptor,
        )
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            manifest_raw = archive.read(LGCVF_LIVE_CAPSULE_MANIFEST_MEMBER)
    except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise SuccessorOperatorError(
            "LGCVF live sealed capsule manifest is unavailable"
        ) from exc
    if not 0 < len(manifest_raw) <= 8 * 1024 * 1024:
        raise SuccessorOperatorError(
            "LGCVF live sealed capsule manifest is out of bounds"
        )

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate manifest key")
            result[key] = value
        return result

    try:
        manifest = json.loads(
            manifest_raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
        canonical = (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(
            "LGCVF live sealed capsule manifest is invalid"
        ) from exc
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if (
        manifest_raw != canonical
        or not isinstance(files, dict)
        or manifest.get("capsule_id") != getattr(capsule_pin, "capsule_id", None)
        or manifest.get("operator_path")
        != getattr(capsule_pin, "operator_path", None)
        or manifest.get("operator_sha256")
        != getattr(capsule_pin, "operator_sha256", None)
        or manifest.get("candidate_config_path")
        != getattr(capsule_pin, "candidate_config_path", None)
        or manifest.get("candidate_config_sha256")
        != getattr(capsule_pin, "candidate_config_sha256", None)
    ):
        raise SuccessorOperatorError(
            "LGCVF live sealed capsule manifest identity differs"
        )
    normalized: dict[str, str] = {}
    for relative, digest in files.items():
        path = Path(str(relative))
        if (
            not isinstance(relative, str)
            or not relative
            or path.is_absolute()
            or ".." in path.parts
            or relative != path.as_posix()
            or not isinstance(digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise SuccessorOperatorError(
                "LGCVF live sealed capsule manifest inventory differs"
            )
        normalized[relative] = digest
    return archive_path, normalized


def _audit_lgcvf_live_loaded_repository_modules(
    *,
    root: Path,
    operator_path: Path,
    manifest_files: Mapping[str, str],
    read_member: Any,
    modules: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Bind loaded repository source origins and current bytes to the capsule."""

    try:
        exact_root = root.resolve(strict=True)
        exact_datasets_root = (exact_root / "ipfs_datasets_py").resolve(
            strict=True
        )
        exact_operator = operator_path.resolve(strict=True)
    except OSError as exc:
        raise SuccessorOperatorError(
            "LGCVF live loaded-source roots are unavailable"
        ) from exc
    expected_operator = exact_root / (
        "scripts/run_logic_governed_compositional_verification_fabric_quack.py"
    )
    if exact_operator != expected_operator:
        raise SuccessorOperatorError(
            "LGCVF live outer operator origin differs"
        )
    operator_member = expected_operator.relative_to(exact_root).as_posix()
    sealed_operator = _lgcvf_live_manifest_member(
        operator_member,
        manifest_files=manifest_files,
        read_member=read_member,
        noun="LGCVF live outer operator",
    )
    current_operator = _read_bounded_regular_file(
        exact_operator,
        max_bytes=64 * 1024 * 1024,
        noun="LGCVF live outer operator",
    )
    if current_operator != sealed_operator:
        raise SuccessorOperatorError(
            "LGCVF live outer operator bytes differ from the sealed capsule"
        )

    module_table = sys.modules if modules is None else modules
    audited: list[str] = []
    member_owners: dict[str, tuple[str, Any]] = {}
    for module_name, module in sorted(module_table.items()):
        namespace_member = any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in LGCVF_LIVE_REPOSITORY_MODULE_PREFIXES
        )
        module_file = getattr(module, "__file__", None) if module is not None else None
        if isinstance(module_file, str):
            lexical_file = Path(module_file)
            under_repository = False
            if lexical_file.is_absolute():
                try:
                    lexical_file.relative_to(exact_root)
                    under_repository = True
                except ValueError:
                    pass
        else:
            lexical_file = None
            under_repository = False
        if not namespace_member and not under_repository:
            continue
        if module is None or lexical_file is None:
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin is invalid: {module_name}"
            )
        if module_name == "__main__":
            try:
                main_origin = lexical_file.resolve(strict=True)
            except OSError as exc:
                raise SuccessorOperatorError(
                    "LGCVF live outer operator module origin is unavailable"
                ) from exc
            if main_origin == exact_operator:
                continue
        if not lexical_file.is_absolute():
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin is invalid: {module_name}"
            )
        spec = getattr(module, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        if not isinstance(spec_origin, str) or Path(spec_origin) != lexical_file:
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin differs: {module_name}"
            )
        try:
            exact_file = lexical_file.resolve(strict=True)
        except OSError as exc:
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin is unavailable: {module_name}"
            ) from exc
        if exact_file != lexical_file:
            raise SuccessorOperatorError(
                f"LGCVF live loaded repository module origin contains a link: {module_name}"
            )
        if exact_file == exact_operator:
            continue
        try:
            nested_relative = exact_file.relative_to(exact_datasets_root)
        except ValueError:
            try:
                relative = exact_file.relative_to(exact_root).as_posix()
            except ValueError as exc:
                raise SuccessorOperatorError(
                    f"LGCVF live loaded repository module escaped the source root: {module_name}"
                ) from exc
        else:
            relative = (
                Path("ipfs_datasets_py") / nested_relative
            ).as_posix()
        previous_owner = member_owners.setdefault(relative, (module_name, module))
        if previous_owner[1] is not module:
            raise SuccessorOperatorError(
                "LGCVF live loaded repository module origin is aliased"
            )
        sealed_source = _lgcvf_live_manifest_member(
            relative,
            manifest_files=manifest_files,
            read_member=read_member,
            noun=f"LGCVF live loaded repository module {module_name}",
        )
        current_source = _read_bounded_regular_file(
            exact_file,
            max_bytes=64 * 1024 * 1024,
            noun=f"LGCVF live loaded repository module {module_name}",
        )
        if current_source != sealed_source:
            raise SuccessorOperatorError(
                "LGCVF live loaded repository module bytes differ from the "
                f"sealed capsule: {module_name}"
            )
        audited.append(module_name)
    if not set(LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES).issubset(audited):
        missing = sorted(set(LGCVF_LIVE_CONTROLLER_PRELOAD_MODULES) - set(audited))
        raise SuccessorOperatorError(
            "LGCVF live controller dependency closure is incomplete: "
            + ", ".join(missing[:3])
        )
    return tuple(audited)


def _retarget_lgcvf_live_repository_imports(
    *,
    root: Path,
    archive_path: str,
    modules: Mapping[str, Any] | None = None,
    path_entries: list[str] | None = None,
    meta_path: list[Any] | None = None,
) -> tuple[str, ...]:
    """Remove mutable repository import roots and project package paths to ZIP."""

    exact_root = root.resolve(strict=True)
    if (
        not archive_path.startswith("/proc/self/fd/")
        or not Path(archive_path).is_absolute()
    ):
        raise SuccessorOperatorError("LGCVF live sealed import root is invalid")
    capsule_roots = (
        archive_path + "/ipfs_datasets_py",
        archive_path,
    )
    target_path = sys.path if path_entries is None else path_entries
    retained: list[str] = []
    for entry in target_path:
        if entry in capsule_roots or not isinstance(entry, str) or not entry:
            continue
        if entry.startswith("__editable__.") or not Path(entry).is_absolute():
            continue
        try:
            resolved = Path(entry).resolve(strict=False)
            resolved.relative_to(exact_root)
        except ValueError:
            retained.append(entry)
        else:
            continue
    target_path[:] = list(dict.fromkeys((*capsule_roots, *retained)))

    allowed_meta = (
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    )
    target_meta = sys.meta_path if meta_path is None else meta_path
    if any(finder not in target_meta for finder in allowed_meta):
        raise SuccessorOperatorError(
            "LGCVF live standard import machinery is unavailable"
        )
    target_meta[:] = list(allowed_meta)

    module_table = sys.modules if modules is None else modules
    retargeted: list[str] = []
    for module_name, module in sorted(module_table.items()):
        if module is None:
            continue
        package_path = getattr(module, "__path__", None)
        if package_path is None:
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not Path(module_file).is_absolute():
            continue
        try:
            exact_file = Path(module_file).resolve(strict=True)
            nested_relative = exact_file.relative_to(
                exact_root / "ipfs_datasets_py"
            )
        except ValueError:
            try:
                relative = exact_file.relative_to(exact_root)
            except (OSError, ValueError):
                continue
        except OSError:
            continue
        else:
            relative = Path("ipfs_datasets_py") / nested_relative
        if module_name == "ipfs_datasets_py":
            sealed_package = archive_path + "/ipfs_datasets_py/ipfs_datasets_py"
        else:
            sealed_package = archive_path + "/" + relative.parent.as_posix()
        projected = [sealed_package]
        module.__path__ = projected
        spec = getattr(module, "__spec__", None)
        if spec is None or getattr(spec, "submodule_search_locations", None) is None:
            raise SuccessorOperatorError(
                f"LGCVF live loaded package spec differs: {module_name}"
            )
        spec.submodule_search_locations = projected
        retargeted.append(module_name)

    if path_entries is None and modules is None:
        sys.path_importer_cache.clear()
        importlib.invalidate_caches()
    if tuple(target_path[:2]) != capsule_roots:
        raise SuccessorOperatorError("LGCVF live sealed import roots drifted")
    return tuple(retargeted)


def _prepare_lgcvf_configured_board_live_launch(
    *,
    root: Path,
    config_path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize and authenticate every byte needed by the live child.

    This function has no database or Quack-owner effect.  It may create only a
    private content-addressed capsule, sealed anonymous descriptors, and the
    verified load-only extension HOME.  It deliberately leaves the candidate
    import boundary intact for provenance verification.  Its caller must close
    both descriptors on every exit path.
    """

    expected_config = _contained(root, DEFAULT_SUCCESSOR_CONFIG_RELATIVE)
    try:
        exact_config = config_path.resolve(strict=True)
    except OSError as exc:
        raise SuccessorOperatorError(
            "LGCVF live candidate config is unavailable"
        ) from exc
    if exact_config != expected_config:
        raise SuccessorOperatorError(
            "LGCVF live capsule requires the exact candidate config"
        )
    continuity = _candidate_runtime_continuity(root)
    source_head = str(continuity.get("current_head") or "")
    source_tree = str(continuity.get("current_tree") or "")
    config_raw = _read_bounded_regular_file(
        exact_config,
        max_bytes=4 * 1024 * 1024,
        noun="LGCVF live candidate config",
    )
    candidate_config_sha256 = (
        "sha256:" + hashlib.sha256(config_raw).hexdigest()
    )
    runtime = _resolve_installed_duckdb_live_runtime()

    from ipfs_accelerate_py.agent_implementation_route import (
        materialize_lgcvf_configured_board_live_capsule,
        project_lgcvf_configured_board_live_extensions,
        seal_lgcvf_configured_board_live_capsule,
        verify_lgcvf_configured_board_live_sealed_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        build_lgcvf_configured_board_live_admission,
        verify_lgcvf_configured_board_live_context,
    )
    from ipfs_accelerate_py.llm_router import (
        inspect_agent_supervisor_native_dependency_source,
        seal_agent_supervisor_native_dependency,
        verify_agent_supervisor_native_dependency_sealed_fd,
    )

    native_pin = inspect_agent_supervisor_native_dependency_source(
        runtime["native_path"],
        distribution_version=str(runtime["version"]),
        engine_version=str(runtime["engine_version"]),
    )
    native_authorization_id = _lgcvf_live_native_authorization_id(
        native_pin=native_pin,
        provenance=provenance,
        source_head=source_head,
        source_tree=source_tree,
        candidate_config_sha256=candidate_config_sha256,
    )
    capsule_parent = Path(
        tempfile.mkdtemp(
            prefix=f"lgcvf-live-capsule-{os.geteuid()}-",
            dir=tempfile.gettempdir(),
        )
    )
    os.chmod(capsule_parent, 0o700)
    capsule = None
    native_launch = None
    try:
        capsule_pin = materialize_lgcvf_configured_board_live_capsule(
            source_root=root,
            capsule_parent=capsule_parent,
            source_head=source_head,
            source_tree=source_tree,
            python_executable=sys.executable,
            duckdb_package_root=runtime["package_root"],
            duckdb_distribution_metadata_root=runtime["metadata_root"],
            duckdb_distribution_version=str(runtime["version"]),
            quack_extension_path=runtime["quack_path"],
            quack_extension_version=str(runtime["quack_version"]),
            ducklake_extension_path=runtime["ducklake_path"],
            ducklake_extension_version=str(runtime["ducklake_version"]),
            httpfs_extension_path=runtime["httpfs_path"],
            httpfs_extension_version=str(runtime["httpfs_version"]),
            native_authorization_id=native_authorization_id,
            native_dependency_id=native_pin.dependency_id,
        )
        if (
            capsule_pin.source_head != source_head
            or capsule_pin.source_tree != source_tree
            or capsule_pin.candidate_config_sha256
            != candidate_config_sha256
            or capsule_pin.native_authorization_id
            != native_authorization_id
            or capsule_pin.native_dependency_id != native_pin.dependency_id
            or capsule_pin.duckdb_distribution_version
            != runtime["version"]
            or capsule_pin.quack_extension.version
            != runtime["quack_version"]
            or capsule_pin.ducklake_extension.version
            != runtime["ducklake_version"]
            or capsule_pin.httpfs_extension.version
            != runtime["httpfs_version"]
        ):
            raise SuccessorOperatorError(
                "LGCVF live capsule differs from its controller admission"
            )
        capsule = seal_lgcvf_configured_board_live_capsule(capsule_pin)
        if (
            verify_lgcvf_configured_board_live_sealed_capsule(
                capsule_pin,
                capsule.descriptor,
            )
            != capsule.executable_path
        ):
            raise SuccessorOperatorError(
                "LGCVF live capsule sealed descriptor drifted"
            )
        native_launch = seal_agent_supervisor_native_dependency(
            runtime["native_path"],
            expected_pin=native_pin,
            accepted_authorization_id=native_authorization_id,
        )
        if (
            native_launch.pin != native_pin
            or native_launch.accepted_authorization_id
            != native_authorization_id
            or verify_agent_supervisor_native_dependency_sealed_fd(
                native_launch
            )
            != f"/proc/self/fd/{native_launch.descriptor.descriptor}"
        ):
            raise SuccessorOperatorError(
                "LGCVF native dependency sealed descriptor drifted"
            )
        admission = build_lgcvf_configured_board_live_admission(
            capsule_pin,
            native_launch,
        )
        capsule_pin_json = capsule_pin.to_json()
        admission_json = admission.to_json()
        native_launch_json = native_launch.to_json()
        context = verify_lgcvf_configured_board_live_context(
            capsule_pin_json=capsule_pin_json,
            capsule_descriptor=capsule.descriptor,
            admission_json=admission_json,
            native_launch_json=native_launch_json,
            native_descriptor=native_launch.descriptor.descriptor,
        )
        if context.admission != admission:
            raise SuccessorOperatorError(
                "LGCVF live capsule/native admission join drifted"
            )
        qualification_parent = _contained(
            root,
            LGCVF_LIVE_QUALIFICATION_HOMES_RELATIVE,
        )
        launch_home = project_lgcvf_configured_board_live_extensions(
            capsule_pin,
            capsule.descriptor,
            qualification_parent,
        )
        expected_home = qualification_parent / capsule_pin.capsule_id.removeprefix(
            "sha256:"
        )
        if launch_home != expected_home:
            raise SuccessorOperatorError(
                "LGCVF live extension HOME identity drifted"
            )
        preloaded_modules = _preload_lgcvf_live_controller_dependency_closure()
        archive_path, manifest_files = _lgcvf_live_sealed_manifest_inventory(
            capsule_pin,
            capsule.descriptor,
        )
        try:
            with zipfile.ZipFile(archive_path, mode="r") as archive:
                sealed_config_raw = _lgcvf_live_manifest_member(
                    capsule_pin.candidate_config_path,
                    manifest_files=manifest_files,
                    read_member=archive.read,
                    noun="LGCVF live candidate config",
                )
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise SuccessorOperatorError(
                "LGCVF live sealed controller closure is unreadable"
            ) from exc
        if sealed_config_raw != config_raw:
            raise SuccessorOperatorError(
                "LGCVF live candidate config differs from the sealed capsule"
            )
        board, program, host, port = _validate_successor_board(
            exact_config,
            root,
            config_bytes=sealed_config_raw,
            admitted_live_validator_sha256=capsule_pin.validator_sha256,
        )
        try:
            with zipfile.ZipFile(archive_path, mode="r") as archive:
                audited_modules = _audit_lgcvf_live_loaded_repository_modules(
                    root=root,
                    operator_path=Path(__file__),
                    manifest_files=manifest_files,
                    read_member=archive.read,
                )
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise SuccessorOperatorError(
                "LGCVF live sealed controller closure is unreadable"
            ) from exc
        final_continuity = _candidate_runtime_continuity(root)
        if final_continuity != continuity:
            raise SuccessorOperatorError(
                "LGCVF live source changed after controller closure admission"
            )
        return {
            "capsule_parent": capsule_parent,
            "capsule_pin": capsule_pin,
            "capsule": capsule,
            "capsule_pin_json": capsule_pin_json,
            "admission": admission,
            "admission_json": admission_json,
            "native_launch": native_launch,
            "native_launch_json": native_launch_json,
            "launch_home": launch_home,
            "pass_fds": context.pass_fds,
            "board": board,
            "program": program,
            "host": host,
            "port": port,
            "sealed_config_raw": sealed_config_raw,
            "archive_path": archive_path,
            "continuity": dict(continuity),
            "preloaded_modules": preloaded_modules,
            "audited_modules": audited_modules,
        }
    except BaseException:
        if native_launch is not None:
            try:
                os.close(native_launch.descriptor.descriptor)
            except OSError:
                pass
        if capsule is not None:
            try:
                os.close(capsule.descriptor)
            except OSError:
                pass
        _remove_private_live_capsule_parent(capsule_parent)
        raise


def _verify_lgcvf_live_provenance_before_import_retarget(
    *,
    paths: Mapping[str, Path],
    root: Path,
    raw_provenance: Mapping[str, Any],
    live_launch: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify state, re-audit newly loaded source, then seal import routing."""

    provenance = _load_provenance(
        paths,
        root=root,
        expected_receipt=raw_provenance,
    )
    if provenance != raw_provenance:
        raise SuccessorOperatorError(
            "verified successor provenance differs from native authorization"
        )
    archive_path = live_launch.get("archive_path")
    admitted_continuity = live_launch.get("continuity")
    capsule_pin = live_launch.get("capsule_pin")
    capsule_descriptor = getattr(live_launch.get("capsule"), "descriptor", None)
    if (
        not isinstance(archive_path, str)
        or not isinstance(admitted_continuity, Mapping)
        or type(capsule_descriptor) is not int
        or capsule_descriptor < 3
    ):
        raise SuccessorOperatorError(
            "LGCVF live controller admission is incomplete before import sealing"
        )
    post_provenance_preloaded_modules = (
        _preload_lgcvf_live_controller_dependency_closure()
    )
    refreshed_archive_path, manifest_files = (
        _lgcvf_live_sealed_manifest_inventory(
            capsule_pin,
            capsule_descriptor,
        )
    )
    if refreshed_archive_path != archive_path:
        raise SuccessorOperatorError(
            "LGCVF live sealed import root changed during provenance verification"
        )
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            audited_modules = _audit_lgcvf_live_loaded_repository_modules(
                root=root,
                operator_path=Path(__file__),
                manifest_files=manifest_files,
                read_member=archive.read,
            )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise SuccessorOperatorError(
            "LGCVF live sealed controller closure is unreadable"
        ) from exc
    final_continuity = _candidate_runtime_continuity(root)
    if final_continuity != admitted_continuity:
        raise SuccessorOperatorError(
            "LGCVF live source changed during provenance verification"
        )
    retargeted_packages = _retarget_lgcvf_live_repository_imports(
        root=root,
        archive_path=archive_path,
    )
    return {
        "provenance": provenance,
        "preloaded_modules": post_provenance_preloaded_modules,
        "audited_modules": audited_modules,
        "retargeted_packages": retargeted_packages,
    }


def _close_lgcvf_configured_board_live_launch(
    launch: Mapping[str, Any] | None,
) -> None:
    """Close only the two controller-owned sealed descriptors and capsule."""

    if not launch:
        return
    descriptors = {
        int(getattr(launch.get("capsule"), "descriptor", -1)),
        int(
            getattr(
                getattr(launch.get("native_launch"), "descriptor", None),
                "descriptor",
                -1,
            )
        ),
    }
    for descriptor in descriptors:
        if descriptor >= 3:
            try:
                os.close(descriptor)
            except OSError:
                pass
    _remove_private_live_capsule_parent(launch.get("capsule_parent"))


def _run_locked_successor(
    config_path: Path,
    *,
    root: Path,
    implement: bool,
    duration_seconds: float,
) -> int:
    paths = _paths(root)
    raw_provenance = _load_lgcvf_live_raw_provenance_receipt(paths)
    live_launch = _prepare_lgcvf_configured_board_live_launch(
        root=root,
        config_path=config_path,
        provenance=raw_provenance,
    )
    server: Any | None = None
    bootstrap_channel: socket.socket | None = None
    bootstrap_broker: _LgcvfStateOwnerBootstrapBroker | None = None
    previous_extension_environment: dict[str, str | None] = {}

    def stop_owner() -> Mapping[str, Any]:
        nonlocal server
        if server is None:
            return {"stopped": True, "already_stopped": True}
        owned_server = server
        server = None
        return owned_server.stop()

    def stop_bootstrap_broker() -> None:
        nonlocal bootstrap_broker, bootstrap_channel
        if bootstrap_broker is not None:
            owned_broker = bootstrap_broker
            owned_broker.stop()
            bootstrap_broker = None
            bootstrap_channel = None
            return
        if bootstrap_channel is not None:
            owned_channel = bootstrap_channel
            bootstrap_channel = None
            try:
                owned_channel.close()
            except OSError:
                pass

    try:
        launch_home = Path(live_launch["launch_home"])
        extension_environment = {
            "HOME": str(launch_home),
            "IPFS_ACCELERATE_AGENT_TRUSTED_DUCKDB_HOME": str(launch_home),
            "XDG_CACHE_HOME": str(launch_home / ".cache" / "xdg"),
            "CUDA_CACHE_PATH": str(launch_home / ".cache" / "cuda"),
            "CUDA_CACHE_DISABLE": "1",
            BOARD_EXTENSION_INSTALL_POLICY_ENV: (
                BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
            ),
            STORE_GENERATION_ENV: SUCCESSOR_STORE_GENERATION,
        }
        previous_extension_environment.update(
            {name: os.environ.get(name) for name in extension_environment}
        )
        forbidden_loader_environment = {
            name
            for name in os.environ
            if name.startswith("LD_") or name == "GLIBC_TUNABLES"
        }
        if forbidden_loader_environment:
            raise SuccessorOperatorError(
                "LGCVF native owner inherited ambient loader authority"
            )
        os.environ.update(extension_environment)
        from ipfs_accelerate_py.llm_router import (
            preload_agent_supervisor_native_dependency,
        )

        preload_agent_supervisor_native_dependency(live_launch["native_launch"])
        sealed_admission = _verify_lgcvf_live_provenance_before_import_retarget(
            paths=paths,
            root=root,
            raw_provenance=raw_provenance,
            live_launch=live_launch,
        )
        provenance = sealed_admission["provenance"]
        from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
            current_process_birth,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
            configured_board_launch_plan,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
            establish_state_authority_process_boundary,
            harden_state_authority_process,
        )

        board = live_launch["board"]
        program = live_launch["program"]
        host = str(live_launch["host"])
        port = int(live_launch["port"])
        if program.store_generation != extension_environment[STORE_GENERATION_ENV]:
            raise SuccessorOperatorError(
                "configured board differs from the admitted live generation"
            )
        rendered_plan = configured_board_launch_plan(
            board,
            implement=implement,
            detach=False,
            duration_seconds=duration_seconds,
        )
        rendered_environment = rendered_plan.get("environment")
        if not isinstance(rendered_environment, Mapping):
            raise SuccessorOperatorError(
                "configured scheduler environment is unavailable"
            )
        expected_route_environment = {
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
            "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.6",
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
                "primary_quota_exhausted"
            ),
            "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
            "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "high",
        }
        if any(
            rendered_environment.get(name) != value
            for name, value in expected_route_environment.items()
        ):
            raise SuccessorOperatorError(
                "configured scheduler did not render the reviewed ordered "
                "provider route"
            )
        owner_program_json = str(
            rendered_environment.get(DATABASE_PROGRAM_JSON_ENV) or ""
        ).strip()
        if not owner_program_json:
            raise SuccessorOperatorError(
                "configured scheduler did not render the database program"
            )
        previous_extension_environment.setdefault(
            DATABASE_PROGRAM_JSON_ENV,
            os.environ.get(DATABASE_PROGRAM_JSON_ENV),
        )
        os.environ[DATABASE_PROGRAM_JSON_ENV] = owner_program_json
        bootstrap_channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bootstrap_channel.bind(
            "\0ipfs-lgcvf-bootstrap-" + uuid.uuid4().hex
        )
        bootstrap_channel.listen(8)
        bootstrap_descriptor = bootstrap_channel.fileno()
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import (
            validate_state_owner_bootstrap_listener,
        )

        validate_state_owner_bootstrap_listener(bootstrap_descriptor)
        scheduler_argv = [
            "--repo-root",
            str(root),
            "--config",
            str(config_path),
            "--configured-board-live-capsule-pin-json",
            str(live_launch["capsule_pin_json"]),
            "--configured-board-live-capsule-fd",
            str(live_launch["capsule"].descriptor),
            "--configured-board-live-admission-json",
            str(live_launch["admission_json"]),
            "--configured-board-live-native-launch-json",
            str(live_launch["native_launch_json"]),
            "--configured-board-live-native-fd",
            str(live_launch["native_launch"].descriptor.descriptor),
            "--state-owner-bootstrap-fd",
            str(bootstrap_descriptor),
            "--state-owner-bootstrap-store-id",
            str(program.store_id),
            "launch",
            "--foreground",
            "--duration-seconds",
            str(duration_seconds),
        ]
        if implement:
            scheduler_argv.append("--implement")
        from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
            build_lgcvf_configured_board_live_module_command,
        )

        command = build_lgcvf_configured_board_live_module_command(
            python_executable=sys.executable,
            capsule_pin_json=str(live_launch["capsule_pin_json"]),
            capsule_descriptor=live_launch["capsule"].descriptor,
            admission_json=str(live_launch["admission_json"]),
            native_launch_json=str(live_launch["native_launch_json"]),
            native_descriptor=live_launch["native_launch"].descriptor.descriptor,
            module_name=LGCVF_LIVE_SCHEDULER_MODULE,
            argv=scheduler_argv,
        )
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
            build_server,
        )

        # The owner-command dispatcher validates the logical generation
        # independently of the live integer server generation.
        # Establish the kernel boundary before Quack owner construction can
        # mint or stage an attach credential.  The post-mint call below also
        # verifies that the recognized credential-bearing path remains hard.
        establish_state_authority_process_boundary()
        paths["owner_state"].mkdir(mode=0o700, parents=True, exist_ok=True)
        _prepare_private_owner_socket(paths["owner_socket"])
        controller_birth = current_process_birth()
        server = build_server(
            database_path=paths["successor_database"],
            state_dir=paths["owner_state"],
            host=host,
            port=port,
            repository_id="repository:lgcvf-quack-successor",
            store_id=program.store_id,
            secret_handle=program.endpoint_secret_handle,
            migrate=datasets_profile_migration,
            typed_command_socket_path=paths["owner_socket"],
        )
        if server.typed_command_socket_path() != paths["owner_socket"]:
            raise SuccessorOperatorError("owner did not retain its short socket path")
        identity = server.start()
        if identity.listen_uri != program.quack_endpoint:
            stop_owner()
            raise SuccessorOperatorError(
                "owner endpoint differs from scheduler program"
            )
        if server._vault is None:
            stop_owner()
            raise SuccessorOperatorError("owner token vault is unavailable")
        token = server._vault.resolve(identity.secret_handle)
        # Harden without copying the credential into the controller environment.
        harden_state_authority_process({TOKEN_ENV: token})
        token_path = paths["owner_state"] / (
            identity.secret_handle.replace(":", "_").replace("/", "_") + ".quack-token"
        )
        if token_path.exists() or token.encode("ascii") in _canonical_bytes(
            server.status()
        ):
            stop_owner()
            raise SuccessorOperatorError("owner published its Quack attach token")
        execution_route_policy = _seal_lgcvf_execution_route_policy(
            server=server,
            program=program,
            identity=identity,
            controller_birth=controller_birth,
            owner_socket=paths["owner_socket"],
        )

        if any(token in item for item in command):
            stop_owner()
            raise SuccessorOperatorError("scheduler argv would contain the Quack token")
        environment = _child_environment(
            token=token,
            identity=identity,
            owner_state=paths["owner_state"],
            root=root,
            rendered_environment=rendered_environment,
            launch_home=launch_home,
        )
        paths["controller_log"].parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_handle = paths["controller_log"].open("ab")
        os.chmod(paths["controller_log"], 0o600)
        scheduler: subprocess.Popen[Any] | None = None
        scheduler_birth: Any | None = None
        stop_requested = False
        prior_handlers: dict[int, Any] = {}

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True

        try:
            scheduler = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                pass_fds=tuple(
                    dict.fromkeys(
                        (*live_launch["pass_fds"], bootstrap_descriptor)
                    )
                ),
                start_new_session=True,
            )
            # Popen returns only after the child has crossed exec.  The child
            # owns inherited references; close the controller copies and drop
            # the disk capsule now so descriptor numbers cannot later be reused
            # and accidentally re-closed by the outer cleanup.
            _close_lgcvf_configured_board_live_launch(live_launch)
            live_launch = None
            scheduler_birth = _exact_birth(scheduler.pid)
            assert bootstrap_channel is not None
            bootstrap_broker = _LgcvfStateOwnerBootstrapBroker(
                channel=bootstrap_channel,
                descriptor=bootstrap_descriptor,
                server=server,
                scheduler_birth=scheduler_birth,
                endpoint=str(program.quack_endpoint),
                socket_path=paths["owner_socket"],
                store_id=str(program.store_id),
                execution_route_policy=execution_route_policy,
            )
            bootstrap_broker.start()
            bootstrap_deadline = (
                time.monotonic()
                + STATE_OWNER_BOOTSTRAP_READY_TIMEOUT_SECONDS
            )
            stable_signature: tuple[str, ...] = ()
            stable_since = 0.0
            while True:
                if scheduler.poll() is not None:
                    raise SuccessorOperatorError(
                        "scheduler exited before all lane daemons attached"
                    )
                if bootstrap_broker.failure:
                    raise SuccessorOperatorError(
                        "lane state-owner bootstrap failed closed: "
                        + bootstrap_broker.failure
                    )
                if time.monotonic() >= bootstrap_deadline:
                    raise SuccessorOperatorError(
                        "lane state-owner bootstrap readiness timed out"
                    )
                observed_signature = bootstrap_broker.live_ready_signature
                if observed_signature != stable_signature:
                    stable_signature = observed_signature
                    stable_since = time.monotonic()
                if (
                    len(stable_signature)
                    == len(LGCVF_DATABASE_OWNER_SESSIONS)
                    and time.monotonic() - stable_since
                    >= STATE_OWNER_BOOTSTRAP_STABILITY_SECONDS
                ):
                    break
                server.service_mutation_inbox(max_requests=32)
                time.sleep(0.01)
            for signum in (signal.SIGINT, signal.SIGTERM):
                prior_handlers[signum] = signal.signal(signum, request_stop)
            ready_status = _status_payload(
                lifecycle="ready",
                controller_birth=controller_birth.to_dict(),
                provenance_cid=str(provenance["receipt_cid"]),
                owner_identity=identity.to_dict(),
                scheduler_birth=scheduler_birth.to_dict(),
                projection_root=paths["projection_root"],
            )
            _write_status(paths["controller_status"], ready_status, token=token)
            started = time.monotonic()
            pump_error = ""
            while scheduler.poll() is None and not stop_requested:
                if (
                    bootstrap_broker is None
                    or bootstrap_broker.failure
                ):
                    pump_error = (
                        "state-owner bootstrap broker failed: "
                        + (
                            "missing"
                            if bootstrap_broker is None
                            else bootstrap_broker.failure
                        )
                    )
                    stop_requested = True
                    break
                if duration_seconds != float("inf") and (
                    time.monotonic() - started >= duration_seconds
                ):
                    stop_requested = True
                    break
                try:
                    server.service_mutation_inbox(max_requests=32)
                except Exception as exc:  # noqa: BLE001 - owner pump fails closed.
                    pump_error = f"{type(exc).__name__}: {exc}"
                    stop_requested = True
                    break
                time.sleep(0.01)
            if stop_requested and scheduler.poll() is None:
                _terminate_exact(
                    scheduler_birth,
                    grace_seconds=LGCVF_SCHEDULER_TREE_STOP_GRACE_SECONDS,
                    child_process=scheduler,
                )
            returncode = scheduler.wait(timeout=5.0)
            stop_bootstrap_broker()
            if pump_error:
                raise SuccessorOperatorError(
                    "mutation inbox pump failed: " + pump_error
                )
        finally:
            for signum, handler in prior_handlers.items():
                signal.signal(signum, handler)
            if (
                scheduler is not None
                and scheduler.poll() is None
                and scheduler_birth is not None
            ):
                _terminate_exact(
                    scheduler_birth,
                    grace_seconds=LGCVF_SCHEDULER_TREE_STOP_GRACE_SECONDS,
                    child_process=scheduler,
                )
                scheduler.wait(timeout=5.0)
            stop_bootstrap_broker()
            log_handle.close()
            stop_receipt = stop_owner()
            credential_leak = bool(tuple(paths["owner_state"].glob("*.quack-token")))
            for surface in (
                paths["controller_log"],
                paths["controller_status"],
                paths["owner_state"] / "quack-state-server.status.json",
            ):
                credential_leak = credential_leak or _regular_file_contains(
                    surface,
                    token.encode("ascii"),
                )
            stopped = _status_payload(
                lifecycle="stopped",
                controller_birth=controller_birth.to_dict(),
                provenance_cid=str(provenance["receipt_cid"]),
                owner_identity=identity.to_dict(),
                scheduler_birth=(
                    scheduler_birth.to_dict() if scheduler_birth is not None else {}
                ),
                scheduler_returncode=(
                    scheduler.returncode if scheduler is not None else None
                ),
                error=(
                    "attach_credential_persisted"
                    if credential_leak
                    else "" if stop_receipt.get("stopped") else "owner_stop_failed"
                ),
                projection_root=paths["projection_root"],
            )
            _write_status(paths["controller_status"], stopped, token=token)
            token = ""
            if credential_leak:
                raise SuccessorOperatorError(
                    "raw Quack attach credential reached a persistent surface"
                )
        return int(returncode)
    finally:
        try:
            stop_bootstrap_broker()
        except Exception as cleanup_exc:  # noqa: BLE001
            sys.stderr.write(
                "LGCVF bootstrap broker emergency stop failed: "
                f"{type(cleanup_exc).__name__}\n"
            )
        if server is not None:
            try:
                stop_owner()
            except Exception as cleanup_exc:  # noqa: BLE001
                sys.stderr.write(
                    "LGCVF owner emergency stop failed: "
                    f"{type(cleanup_exc).__name__}\n"
                )
        for name, previous in previous_extension_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        _close_lgcvf_configured_board_live_launch(live_launch)


def controller_status(root: Path = ROOT) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )

    paths = _paths(root)
    status = _strict_json(
        paths["controller_status"],
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    birth = ProcessBirthIdentity.from_dict(status.get("controller_birth"))
    observed = owner_liveness(birth)
    projection = dict(status.get("ducklake_projection") or {})
    projection["receipt_present"] = paths["projection_receipt"].is_file()
    return {
        **status,
        "observed_controller_liveness": observed.value,
        "running": observed is OwnerLiveness.ALIVE
        and status.get("lifecycle") == "ready",
        "ducklake_projection": projection,
    }


def stop_controller(
    root: Path = ROOT, *, timeout_seconds: float = MAX_STOP_SECONDS
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        ProcessBirthIdentity,
    )

    status = controller_status(root)
    birth = ProcessBirthIdentity.from_dict(status.get("controller_birth"))
    selected_timeout = float(timeout_seconds)
    if (
        not math.isfinite(selected_timeout)
        or selected_timeout < 1.0
        or selected_timeout > MAX_STOP_SECONDS
    ):
        raise SuccessorOperatorError(
            "controller stop timeout is outside the closed bound"
        )
    disposition = _terminate_exact(
        birth,
        grace_seconds=selected_timeout,
    )
    return {
        "stopped": True,
        "disposition": disposition,
        "controller_birth": birth.to_dict(),
    }


def _extension_preflight() -> dict[str, Any]:
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET autoinstall_known_extensions = false")
            connection.execute("SET autoload_known_extensions = false")
            loaded: dict[str, str] = {}
            for extension in ("quack", "ducklake", "httpfs"):
                connection.execute(f"LOAD {extension}")
                row = connection.execute(
                    "SELECT installed, loaded, extension_version FROM duckdb_extensions() "
                    "WHERE extension_name = ?",
                    [extension],
                ).fetchone()
                if row is None or row[0] is not True or row[1] is not True:
                    raise SuccessorOperatorError(f"{extension} is not preinstalled")
                loaded[extension] = str(row[2] or "")
        finally:
            connection.close()
    except Exception as exc:  # noqa: BLE001 - capability is typed unavailable.
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "automatic_installation_permitted": False,
        }
    return {
        "available": True,
        "extensions": loaded,
        "automatic_installation_permitted": False,
    }


def _controller_lock_is_held(path: Path) -> bool:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SuccessorOperatorError("controller lock cannot be inspected") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise SuccessorOperatorError("controller lock custody is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def projection_preflight(
    root: Path = ROOT,
    *,
    _checkpoint_lock_held: bool = False,
) -> dict[str, Any]:
    paths = _paths(root)
    lock_held = (
        True
        if _checkpoint_lock_held
        else _controller_lock_is_held(paths["controller_lock"])
    )
    running = lock_held and not _checkpoint_lock_held
    try:
        running = running or bool(controller_status(root).get("running"))
    except SuccessorOperatorError:
        pass
    capability = _extension_preflight()
    source_admitted = False
    source_error = ""
    if not running:
        try:
            _load_provenance(paths, root=root)
            source_admitted = True
        except (OSError, RuntimeError, ValueError) as exc:
            source_error = f"{type(exc).__name__}: {exc}"
    return {
        "schema": PROJECTION_RECEIPT_SCHEMA,
        "valid": (
            capability.get("available") is True and not running and source_admitted
        ),
        "projection_root": str(paths["projection_root"]),
        "control_catalog_path": str(paths["projection_root"] / "control.duckdb"),
        "ducklake_catalog_path": str(paths["projection_root"] / "lake.ducklake"),
        "ducklake_data_path": str(paths["projection_root"] / "lake-data"),
        "source_database": str(paths["successor_database"]),
        "controller_running": running,
        "controller_lock_held": lock_held,
        "source_database_present": paths["successor_database"].is_file(),
        "provenance_receipt_present": paths["provenance"].is_file(),
        "source_admitted": source_admitted,
        "source_error": source_error,
        "requires_stopped_checkpoint": True,
        "capability": capability,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "read_by_scheduler": False,
        "quack_endpoint_served": False,
        "separate_projection_reason": (
            "BoardControlPlane owns a distinct DuckLake catalog but does not expose "
            "a qualified Quack state-owner endpoint; direct source-file reads are "
            "admitted only after the LGCVF owner stops"
        ),
    }


@contextlib.contextmanager
def _exclusive_projection_checkpoint(paths: Mapping[str, Path]) -> Any:
    """Hold the controller lock so an owner cannot race a direct checkpoint."""

    lock_path = paths["controller_lock"]
    handle = _open_private_lock(lock_path)
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuccessorOperatorError(
                "LGCVF owner is active; refusing direct DuckLake checkpoint"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def project_ducklake_once(root: Path = ROOT) -> dict[str, Any]:
    paths = _paths(root)
    with _exclusive_projection_checkpoint(paths):
        return _project_ducklake_once_locked(root)


def _open_projection_plane(root: Path, projection_root: Path) -> Any:
    """Open the stopped projection with a strict local LOAD-only policy."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        open_board_control_plane,
    )

    return open_board_control_plane(
        root,
        root=projection_root,
        allow_extension_install=False,
    )


def _project_ducklake_once_locked(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    preflight = projection_preflight(root, _checkpoint_lock_held=True)
    if preflight.get("valid") is not True:
        raise SuccessorOperatorError("DuckLake projection preflight is not valid")
    if paths["projection_receipt"].exists():
        raise SuccessorOperatorError(
            "refusing to overwrite DuckLake projection receipt"
        )
    provenance = _load_provenance(paths, root=root)
    source_digest = _sha256_regular_file(
        paths["successor_database"],
        noun="successor control database",
        require_private_owner=True,
    )
    import duckdb

    source = duckdb.connect(str(paths["successor_database"]), read_only=True)
    try:
        columns = tuple(
            str(item[0])
            for item in source.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'tasks' ORDER BY ordinal_position"
            ).fetchall()
        )
        rows = source.execute(
            "SELECT * FROM tasks ORDER BY ordinal, task_cid"
        ).fetchall()
    finally:
        source.close()
    tasks: list[dict[str, Any]] = []
    for row in rows:
        record = {columns[index]: row[index] for index in range(len(columns))}
        body: dict[str, Any] = {}
        try:
            parsed = json.loads(str(record.get("body_json") or "{}"))
            if isinstance(parsed, dict):
                body = parsed
        except json.JSONDecodeError:
            pass
        tasks.append(
            {
                "task_id": str(
                    record.get("task_alias") or record.get("task_cid") or ""
                ),
                "status": str(record.get("status") or ""),
                "title": str(body.get("title") or ""),
                "depends_on": body.get("depends_on") or [],
                "body": body,
            }
        )
    with _open_projection_plane(root, paths["projection_root"]) as plane:
        registration = plane.register_board(
            "logic-governed-compositional-verification-fabric-history-shadow-v1",
            source_path=str(paths["successor_database"]),
            source_kind="duckdb-stopped-checkpoint-observation",
            merge_target_branch="agent/logic-governed-compositional-verification-fabric-v1",
            extra={
                "authoritative": False,
                "scheduling_authority": False,
                "completion_authority": False,
                "source_provenance_cid": provenance["receipt_cid"],
            },
            tasks=tasks,
        )
        aggregate = plane.aggregate_boards()
        if plane.backend != "ducklake+quack" or not plane.ducklake_attached:
            raise SuccessorOperatorError(
                "physical BoardControlPlane did not admit DuckLake + Quack"
            )
        backend = plane.backend
        extensions = {
            "quack_loaded": plane.quack_loaded,
            "ducklake_loaded": plane.ducklake_loaded,
            "ducklake_attached": plane.ducklake_attached,
        }
    if _sha256_regular_file(paths["successor_database"]) != source_digest:
        raise SuccessorOperatorError("projection source changed during checkpoint")
    receipt = {
        "schema": PROJECTION_RECEIPT_SCHEMA,
        "issued_at": _utc_now(),
        "projection_root": str(paths["projection_root"]),
        "control_catalog_path": str(paths["projection_root"] / "control.duckdb"),
        "ducklake_catalog_path": str(paths["projection_root"] / "lake.ducklake"),
        "ducklake_data_path": str(paths["projection_root"] / "lake-data"),
        "source_database": str(paths["successor_database"]),
        "source_sha256": source_digest,
        "source_provenance_cid": provenance["receipt_cid"],
        "board_namespace": registration["board_namespace"],
        "task_count": len(tasks),
        "backend": backend,
        "extensions": extensions,
        "aggregate": aggregate,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "read_by_scheduler": False,
        "quack_endpoint_served": False,
        "requires_stopped_checkpoint": True,
        "production_authorized": False,
    }
    receipt["receipt_cid"] = _content_id(receipt)
    _atomic_json(paths["projection_receipt"], receipt, replace=False)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap")
    sealed = subparsers.add_parser("bootstrap-sealed-continuity")
    sealed.add_argument("--source-root", type=Path, required=True)
    sealed.add_argument("--control-sha256", required=True)
    sealed.add_argument("--coordination-sha256", required=True)
    sealed.add_argument("--execution-sha256", required=True)
    sealed.add_argument("--bootstrap-sha256", required=True)
    sealed.add_argument("--manifest-sha256", required=True)
    sealed.add_argument("--recovery-receipt-sha256", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument(
        "--config", type=Path, default=DEFAULT_SUCCESSOR_CONFIG_RELATIVE
    )
    launch.add_argument("--implement", action="store_true")
    launch.add_argument("--duration-seconds", type=float, default=float("inf"))
    subparsers.add_parser("status")
    stop = subparsers.add_parser("stop")
    stop.add_argument("--timeout-seconds", type=float, default=MAX_STOP_SECONDS)
    subparsers.add_parser("projection-preflight")
    subparsers.add_parser("projection-once")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.repo_root).resolve()
    try:
        if args.command == "bootstrap":
            result: Any = bootstrap_successor(root)
        elif args.command == "bootstrap-sealed-continuity":
            result = bootstrap_sealed_successor(
                root=root,
                source_root=Path(args.source_root),
                control_sha256=str(args.control_sha256),
                coordination_sha256=str(args.coordination_sha256),
                execution_sha256=str(args.execution_sha256),
                bootstrap_sha256=str(args.bootstrap_sha256),
                manifest_sha256=str(args.manifest_sha256),
                recovery_receipt_sha256=str(args.recovery_receipt_sha256),
            )
        elif args.command == "launch":
            config = Path(args.config)
            if not config.is_absolute():
                config = _contained(root, config)
            return run_successor(
                config,
                root=root,
                implement=bool(args.implement),
                duration_seconds=float(args.duration_seconds),
            )
        elif args.command == "status":
            result = controller_status(root)
        elif args.command == "stop":
            result = stop_controller(root, timeout_seconds=float(args.timeout_seconds))
        elif args.command == "projection-preflight":
            result = projection_preflight(root)
        elif args.command == "projection-once":
            result = project_ducklake_once(root)
        else:  # pragma: no cover - argparse closes this branch.
            parser.error("unsupported command")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": CONTROLLER_STATUS_SCHEMA,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
