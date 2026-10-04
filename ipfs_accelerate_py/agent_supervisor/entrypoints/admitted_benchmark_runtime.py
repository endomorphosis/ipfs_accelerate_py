"""Signed local supervisor launch for an actually admitted native Quack plan.

This is a separate grant from empty process qualification. The existing native
owner issues each managed child's credential only after a kernel-authenticated
rendezvous proves the exact marked child birth of this lifecycle run.
"""
from __future__ import annotations

import json
import os
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import uuid
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path

from ..control.authorization_logic import ControlMutationAuthorizer
from ..control.control_contracts import Operation, get_operation_catalog
from ..control.control_plane import SupervisorControlService
from ..control.lifecycle_orchestrator import (
    LifecycleOrchestrator,
    LifecycleProfile,
    ProcessTreeSnapshot,
)
from ..control.profile_authority import load_local_profile, sign_profile_binding, verify_did_key_signature
from ..merge.database_coordination import open_database_coordinator
from ..merge.database_worktree_registry import process_birth_id
from ..merge.worktree_lifecycle import read_process_birth
from ..runtime.local_planning_admission import (
    CONTRACT_KEY,
    _pending_contract_payload,
    _verify_signature,
    content_identity,
    verify_local_benchmark_admission,
)
from ..runtime.quack_state_server import QuackStateServer, ServerLifecycle
from ..task_sources.state_owner_bootstrap import (
    STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA,
    STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA,
    _receive_frame,
    _send_frame,
)
from ..task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
    daemon_required_owner_command_operations,
    daemon_required_owner_operations,
)
from ..task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
)
from .isolated_benchmark_runtime import (
    IsolatedBenchmarkRuntime,
    NativeSupervisorHealthAdapter,
    _digest,
    _git,
)


def _candidate_git_environment():
    """Bind owner commit identity and closed Git configuration to this launch.

    The worker cannot choose this exception: it is this imported module's
    installed source root. A root-owned installation needs an exact exception
    because isolated candidate execution deliberately disables system config.
    The fixed author/committer lets the owner publish merges without borrowing
    a user's Git configuration. The worker entry clears this environment.
    """
    from ..runtime.candidate_execution import GIT_OWNER_ENV
    environment = dict(GIT_OWNER_ENV)
    environment.update({
        'GIT_AUTHOR_NAME': 'Isolated Supervisor',
        'GIT_AUTHOR_EMAIL': 'supervisor@example.invalid',
        'GIT_COMMITTER_NAME': 'Isolated Supervisor',
        'GIT_COMMITTER_EMAIL': 'supervisor@example.invalid',
    })
    module = Path(__file__).absolute()
    source = module.parents[3]
    if source.resolve(strict=True) != source:
        raise ValueError('installed runtime source must have an exact non-symlink path')
    if source.stat().st_uid == os.getuid():
        return environment
    for path in (module, *module.parents):
        info = path.lstat()
        expected = stat.S_ISREG if path == module else stat.S_ISDIR
        if info.st_uid != 0 or info.st_mode & 0o022 or not expected(info.st_mode):
            raise ValueError('foreign-owned runtime source must have immutable root-owned ancestors')
    index = int(environment['GIT_CONFIG_COUNT'])
    environment.update({
        'GIT_CONFIG_COUNT': str(index + 1),
        f'GIT_CONFIG_KEY_{index}': 'safe.directory',
        f'GIT_CONFIG_VALUE_{index}': str(source),
    })
    return environment


def _bounded_git_environment(*, candidate_runner: bool) -> dict[str, str]:
    """Keep Git's optional background workers inside the signed run budget.

    Auto-maintenance can detach from its invoking process while inheriting the
    exact lifecycle markers. That creates a second process root and correctly
    fails the lifecycle fence. Disable that optional work at launch rather than
    ignoring its processes or weakening ownership checks.
    """
    environment = _candidate_git_environment() if candidate_runner else {}
    # Git gives inherited command-line parameters precedence over indexed
    # environment settings. Bind the empty value so a parent's ``git -c``
    # configuration cannot re-enable detached maintenance for this run.
    environment["GIT_CONFIG_PARAMETERS"] = ""
    index = int(environment.get("GIT_CONFIG_COUNT", "0"))
    for key, value in (("gc.auto", "0"), ("gc.autoDetach", "false"),
                       ("maintenance.auto", "false")):
        environment[f"GIT_CONFIG_KEY_{index}"] = key
        environment[f"GIT_CONFIG_VALUE_{index}"] = value
        index += 1
    environment["GIT_CONFIG_COUNT"] = str(index)
    return environment


def _bounded_proof_resource_environment() -> dict[str, str]:
    """Bind an explicitly selected benchmark policy to the same shared ledger."""
    profile_key = "IPFS_DATASETS_PROOF_RESOURCE_PROFILE"
    ledger_key = "IPFS_DATASETS_RESOURCE_SCHEDULER_PATH"
    profile = os.environ.get(profile_key, "")
    if not profile:
        return {}
    if profile != "local-benchmark@1":
        raise ValueError("unsupported proof resource profile")
    value = os.environ.get(ledger_key, "")
    if not value or len(value) > 4096 or any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError("benchmark proof resource profile requires an explicit bounded scheduler path")
    path = Path(value)
    if (not path.is_absolute() or path.resolve(strict=False) != path
            or (path.exists() and not path.is_file())):
        raise ValueError("benchmark scheduler path must be canonical and identify a regular file")
    return {profile_key: profile, ledger_key: value}


class AdmittedSupervisorHealthAdapter(NativeSupervisorHealthAdapter):
    def remember_bootstrapped_child(self, identity):
        # Called only after kernel peer authentication and exact argv/marker/
        # parent validation, immediately before the native credential response.
        if not hasattr(self, "_authenticated_children"):
            self._authenticated_children = {}
        self._authenticated_children[identity.pid] = identity

    def snapshot(self, profile):
        observed = super().snapshot(profile)
        members = {member.pid: member for member in observed.members}
        for pid, identity in tuple(getattr(self, "_authenticated_children", {}).items()):
            if (pid in members or identity.profile_id != profile.profile_id
                    or identity.run_id != profile.run_id or not self.identity_alive(identity)):
                continue
            try:
                self._environ(pid)
            except PermissionError:
                # PR_SET_DUMPABLE=0 intentionally hides credentials from the
                # launching parent. Retain a witnessed identity, never derive
                # ownership from heartbeat bytes or a supplied PID.
                try:
                    parent, group, session, started = self._stat(pid)
                    if (parent == identity.parent_pid and group == identity.process_group_id
                            and session == identity.session_id and started == identity.start_time_ticks):
                        members[pid] = identity
                except (OSError, ValueError, ProcessLookupError):
                    pass
            except (OSError, ValueError):
                pass
        return ProcessTreeSnapshot(profile_id=profile.profile_id, run_id=profile.run_id,
                                   members=tuple(members.values()), captured_at_ms=observed.captured_at_ms)

    def child_scope_matches(self, profile, child) -> bool:
        self.last_scope_mismatch = ""
        for option in ("--todo-path", "--quack-endpoint", "--state-store-id",
                       "--state-owner-bootstrap-fd", "--state-owner-client-id"):
            if option not in child.argv or option not in profile.argv:
                self.last_scope_mismatch = option + " absent"
                return False
            if child.argv[child.argv.index(option) + 1] != profile.argv[profile.argv.index(option) + 1]:
                self.last_scope_mismatch = option + " differs"
                return False
        return ("--implement" in child.argv) == ("--implement" in profile.argv)

    def healthy(self, profile, tree, *, fencing_epoch, now_ms):
        if not super().healthy(profile, tree, fencing_epoch=fencing_epoch, now_ms=now_ms):
            return False
        child_pid = self.last_heartbeat_evidence["daemon_pid"]
        try:
            path = Path(profile.run_root) / "admitted_native_owner_heartbeat.json"
            if path.is_symlink() or path.stat().st_size > 65_536:
                return False
            evidence = json.loads(path.read_text())
            birth = read_process_birth(child_pid)
            if (birth is None or evidence["process_birth"] != birth.to_dict()
                    or evidence["schema"] != "native-typed-owner-live-observation@1"
                    or evidence["process_birth_id"] != process_birth_id(birth)
                    or evidence["owner_read_succeeded"] is not True
                    or evidence["completion_authority"] is not False
                    or evidence["task_progress_authority"] is not False
                    or evidence["dispatch_authority"] is not False
                    or evidence["store_id"] != self.owner_identity.store_id
                    or evidence["server_id"] != self.owner_identity.server_id
                    or type(evidence["generation"]) is not int
                    or evidence["generation"] != self.owner_identity.generation
                    or type(evidence["fence_epoch"]) is not int
                    or evidence["fence_epoch"] != self.owner_identity.fence_epoch
                    or evidence["route_policy_id"] != self.route_policy_id
                    or evidence["client_id"] != self.client_id
                    or type(evidence["sequence"]) is not int or evidence["sequence"] < 1):
                return False
            if type(evidence["observed_at_ms"]) is not int or not 0 <= now_ms - evidence["observed_at_ms"] <= profile.health_stale_ms:
                return False
            self.last_heartbeat_evidence.update({
                "owner_read_sequence": evidence["sequence"],
                "process_birth_id": process_birth_id(birth),
                "completion_authority": False, "task_progress_authority": False,
            })
            return True
        except (OSError, KeyError, TypeError, ValueError):
            return False


class AdmittedBenchmarkRuntime(IsolatedBenchmarkRuntime):
    @classmethod
    def create(cls, directory: Path, *, admission, server, source,
               implement: bool = False, implementation_command: str = "",
               timeout_ms: int = 30_000, start_timeout_ms: int | None = None,
               implementation_timeout_seconds: int | None = None,
               max_task_attempts: int = 1, context_bundle: dict | None = None,
               lifetime_seconds: int = 300, worker_worktree_root: Path | None = None,
               candidate_runner_argv=(), refresh_context_on_completion: bool = False,
               refresh_source384_on_completion: bool = False,
               published_retrieval_policy: str | None = None,
               published_learned_artifacts: dict | None = None,
               finite_execution_scope=None, inventory_execution_scope=None):
        if finite_execution_scope is not None and inventory_execution_scope is not None:
            raise ValueError("finite and inventory execution scopes are mutually exclusive")
        arguments = dict(admission=admission, server=server, source=source, implement=implement,
            implementation_command=implementation_command, timeout_ms=timeout_ms,
            start_timeout_ms=start_timeout_ms,
            implementation_timeout_seconds=implementation_timeout_seconds,
            max_task_attempts=max_task_attempts, context_bundle=context_bundle,
            lifetime_seconds=lifetime_seconds, worker_worktree_root=worker_worktree_root,
            candidate_runner_argv=candidate_runner_argv,
            refresh_context_on_completion=refresh_context_on_completion,
            refresh_source384_on_completion=refresh_source384_on_completion,
            published_retrieval_policy=published_retrieval_policy,
            published_learned_artifacts=published_learned_artifacts,
            finite_execution_scope=finite_execution_scope, inventory_execution_scope=inventory_execution_scope)
        if finite_execution_scope is not None:
            from ..runtime.finite_repository_execution import FrozenFiniteRepositoryExecutionScope
            if (type(finite_execution_scope) is FrozenFiniteRepositoryExecutionScope
                    and finite_execution_scope._uses_paired_receiving()):
                with finite_execution_scope._receiving_operation(purpose="create"):
                    runtime = cls._create(directory, **arguments)
                    try:
                        finite_execution_scope._finish_creation_receiving(runtime)
                    except BaseException:
                        try:
                            stopped = runtime.stop()
                            if not stopped.succeeded or not runtime._context_refresh_stopped():
                                raise RuntimeError("refused proof-query construction lacks native STOP cleanup")
                            runtime.close()
                        except BaseException as cleanup_error:
                            runtime._construction_cleanup_failed = True
                            raise RuntimeError("proof-query receiving cleanup unproven; resource custody retained") from cleanup_error
                        raise
                return runtime
        if inventory_execution_scope is not None:
            from ..runtime.codebase_inventory_execution import FrozenInventoryExecutionScope
            if (type(inventory_execution_scope) is FrozenInventoryExecutionScope
                    and inventory_execution_scope._uses_paired_receiving()):
                with inventory_execution_scope._receiving_operation(purpose="create"):
                    runtime = cls._create(directory, **arguments)
                    try:
                        inventory_execution_scope._finish_creation_receiving(runtime)
                    except BaseException:
                        # No process was launched, but the constructor has
                        # installed an owner completion handler. Its existing
                        # native STOP/UID cleanup contract still governs close.
                        inventory_execution_scope._retain_constructor_cleanup(runtime)
                        try:
                            stopped = runtime.stop()
                            if not stopped.succeeded or not runtime._context_refresh_stopped():
                                raise RuntimeError("refused inventory construction lacks native STOP cleanup")
                            runtime.close()
                        except BaseException as cleanup_error:
                            runtime._construction_cleanup_failed = True
                            raise RuntimeError("inventory receiving cleanup unproven; resource custody retained") from cleanup_error
                        raise
                return runtime
        return cls._create(directory, **arguments)

    @classmethod
    def _create(cls, directory: Path, *, admission, server, source,
               implement: bool = False, implementation_command: str = "",
               timeout_ms: int = 30_000, start_timeout_ms: int | None = None,
               implementation_timeout_seconds: int | None = None,
               max_task_attempts: int = 1, context_bundle: dict | None = None,
               lifetime_seconds: int = 300, worker_worktree_root: Path | None = None,
               candidate_runner_argv=(), refresh_context_on_completion: bool = False,
               refresh_source384_on_completion: bool = False,
               published_retrieval_policy: str | None = None,
               published_learned_artifacts: dict | None = None,
               finite_execution_scope=None, inventory_execution_scope=None):
        from ..runtime.header_intent_applicability import capture_applicability_budget
        # Capture the enclosing work scope, not a transient constructor replay
        # ceiling: later START and its bootstrap thread receive fresh ceilings
        # bounded by this same original deadline and the native run lifetime.
        replay_work_scope = capture_applicability_budget()
        creation_started = time.monotonic()
        if finite_execution_scope is not None and inventory_execution_scope is not None:
            raise ValueError("finite and inventory execution scopes are mutually exclusive")
        if type(refresh_context_on_completion) is not bool or (refresh_context_on_completion and context_bundle is None):
            raise ValueError("automatic context refresh requires an explicit context bundle")
        if (type(refresh_source384_on_completion) is not bool
                or (refresh_source384_on_completion and (not refresh_context_on_completion
                    or replay_work_scope is None or finite_execution_scope is not None
                    or inventory_execution_scope is not None))):
            raise ValueError("Source384 refresh requires an explicit local work scope and ordinary context refresh")
        if published_retrieval_policy is not None and (
                published_retrieval_policy not in {"lexical-tfidf-symbols@1", "local-safetensors-symbols@1"}
                or not refresh_context_on_completion):
            raise ValueError("published retrieval requires an explicit admitted refresh policy")
        if (published_learned_artifacts is not None) != (published_retrieval_policy == "local-safetensors-symbols@1"):
            raise ValueError("learned refresh requires exact initial model/policy artifacts")
        if type(implement) is not bool or (implement and not implementation_command.strip()):
            raise ValueError("live task launch requires an explicit implementation command")
        if not implement and implementation_command:
            raise ValueError("observation launch cannot carry an implementation command")
        maximum_timeout = min(get_operation_catalog().by_name[operation.value].bounds.timeout_ms
                              for operation in (Operation.START, Operation.STOP))
        if type(timeout_ms) is not int or not 2_000 <= timeout_ms <= maximum_timeout:
            raise ValueError(f"bounded native launch timeout must fit START/STOP catalog (2000..{maximum_timeout})")
        if type(max_task_attempts) is not int or not 1 <= max_task_attempts <= 10:
            raise ValueError("coordination attempt cap must be in 1..10")
        if type(lifetime_seconds) is not int or not 120 <= lifetime_seconds <= 900:
            raise ValueError("signed launch lifetime_seconds must be in 120..900")
        if implementation_timeout_seconds is not None and (
                type(implementation_timeout_seconds) is not int
                or not 1 <= implementation_timeout_seconds <= lifetime_seconds
                or not implement):
            raise ValueError("explicit implementation timeout requires live execution and an integer within the signed lifetime")
        proof_resource_environment = _bounded_proof_resource_environment()
        if start_timeout_ms is not None and (
                type(start_timeout_ms) is not int or not 2_000 <= start_timeout_ms <= 120_000
                or proof_resource_environment.get("IPFS_DATASETS_PROOF_RESOURCE_PROFILE") != "local-benchmark@1"
                or start_timeout_ms > lifetime_seconds * 1000):
            raise ValueError("explicit START budget requires the local benchmark profile and 2000..120000 milliseconds within its lifetime")
        from ..runtime.candidate_execution import (
            CANDIDATE_RUNNER_ENV,
            bind_candidate_runner,
        )
        candidate_runner = bind_candidate_runner(candidate_runner_argv) if candidate_runner_argv else None
        if worker_worktree_root is not None:
            worker_worktree_root = Path(worker_worktree_root)
            if (not worker_worktree_root.is_absolute() or worker_worktree_root.is_symlink()
                    or worker_worktree_root.resolve() != worker_worktree_root
                    or not worker_worktree_root.is_dir()
                    or worker_worktree_root.stat().st_uid != os.getuid()
                    or worker_worktree_root.stat().st_mode & 0o022):
                raise ValueError("worker_worktree_root must be an existing exact owner-controlled directory")
        verified = verify_local_benchmark_admission(admission, initial=True)
        from ..runtime.local_planning_admission import INVENTORY_MANIFEST_SCHEMAS
        if verified["manifest"]["schema"] in INVENTORY_MANIFEST_SCHEMAS and inventory_execution_scope is None:
            raise ValueError("inventory admission requires an exact native inventory execution scope")
        if refresh_context_on_completion and len(verified["graph"].tasks) != 1:
            raise ValueError("published context refresh currently requires one admitted task")
        if finite_execution_scope is not None:
            from ..runtime.finite_repository_execution import FrozenFiniteRepositoryExecutionScope
            if type(finite_execution_scope) is not FrozenFiniteRepositoryExecutionScope:
                raise ValueError("an exact native finite execution scope is required")
            finite_execution_scope.require_launch(admission=admission, server=server, source=source,
                implement=implement, candidate_runner=candidate_runner, context_bundle=context_bundle,
                refresh_context_on_completion=refresh_context_on_completion,
                implementation_command=implementation_command)
        if inventory_execution_scope is not None:
            from ..runtime.codebase_inventory_execution import FrozenInventoryExecutionScope
            if type(inventory_execution_scope) is not FrozenInventoryExecutionScope:
                raise ValueError("an exact native inventory execution scope is required")
            inventory_execution_scope.require_launch(admission=admission, server=server, source=source,
                implement=implement, candidate_runner=candidate_runner, context_bundle=context_bundle,
                refresh_context_on_completion=refresh_context_on_completion,
                implementation_command=implementation_command)
        if (candidate_runner is not None and len(verified["graph"].tasks) != 1
                and finite_execution_scope is None and inventory_execution_scope is None):
            raise ValueError("isolated candidate runner requires one admitted task and one worker")
        if type(server) is not QuackStateServer or server.lifecycle is not ServerLifecycle.READY:
            raise ValueError("a live native Quack owner is required")
        if type(source) is not TypedDatabaseTaskSource or source.execution_route_policy is None:
            raise ValueError("a native route-sealed task source is required")
        declared, profile = verified["manifest"], verified["profile"]
        if server.identity.repository_id != declared["repository_cid"]:
            raise ValueError("native owner repository differs from signed admission")
        directory = Path(directory).absolute()
        if directory.exists() or directory.resolve() != directory:
            raise ValueError("admitted launch requires a fresh state directory")
        repository = Path(declared["repository"])
        if directory.is_relative_to(repository):
            raise ValueError("owner launch state must remain outside the task repository")
        if worker_worktree_root is not None and (
                worker_worktree_root.is_relative_to(directory)
                or directory.is_relative_to(worker_worktree_root)
                or worker_worktree_root.is_relative_to(repository)
                or repository.is_relative_to(worker_worktree_root)):
            raise ValueError("worker_worktree_root must be separate from canonical repository and owner launch state")
        directory.mkdir(parents=True, mode=0o700)
        runtime = cls()
        try:
            runtime._replay_work_scope = replay_work_scope
            runtime._replay_lifetime_deadline = creation_started + lifetime_seconds
            runtime._replay_start_deadline = None
            runtime.directory, runtime.state, runtime.repository = directory, directory / "state", repository
            runtime.state.mkdir(mode=0o700)
            (runtime.state / "run").mkdir(mode=0o700)
            runtime.admission = json.loads(json.dumps(admission))
            runtime.finite_execution_scope = finite_execution_scope
            runtime.inventory_execution_scope = inventory_execution_scope
            runtime.server, runtime.source, runtime.owner_identity = server, source, server.identity
            if inventory_execution_scope is not None:
                inventory_execution_scope._remember_constructor_cleanup(runtime)
            runtime.route_policy = source.execution_route_policy
            runtime.baseline = declared["baseline_commit"]
            runtime.tree_id = _git(repository, "rev-parse", "HEAD^{tree}")
            runtime.repository_id = declared["repository_cid"]
            runtime.profile_dir, runtime.lifecycle_dir = Path(declared["profile_dir"]), Path(declared["lifecycle_dir"])
            runtime.local_profile, runtime.timeout_ms = profile, timeout_ms
            runtime.start_timeout_ms = start_timeout_ms
            runtime.implementation_timeout_seconds = implementation_timeout_seconds
            runtime._startup_trace_lock = threading.Lock()
            runtime._startup_trace = []
            runtime._startup_trace_truncated = False
            runtime.context_bundle = dict(context_bundle) if context_bundle is not None else None
            runtime._verify_context(verified)
            if refresh_source384_on_completion:
                from ..runtime.task_context_bundle import read_task_context_historical_selection
                task = verified["graph"].tasks[0]
                selected = read_task_context_historical_selection(repository=repository,
                    artifact=runtime.context_bundle["artifact"],
                    expected_sha256=runtime.context_bundle["sha256"],
                    task_cid=task.task_cid, task_id=task.task_key)
                if "source384_context" not in selected:
                    raise ValueError("Source384 refresh requires the admitted Source384 predecessor")
                (runtime.state / "published-source384").mkdir(mode=0o700)
            retrieval_binding = None
            if published_retrieval_policy is not None:
                task = verified["graph"].tasks[0]
                if published_retrieval_policy == "local-safetensors-symbols@1":
                    from ..runtime.published_learned_retrieval import bind_published_learned_retrieval_policy
                    retrieval_binding = bind_published_learned_retrieval_policy(repository=repository,
                        bundle=runtime.context_bundle, task_cid=task.task_cid, task_id=task.task_key,
                        artifacts=published_learned_artifacts)
                else:
                    from ..runtime.published_retrieval import bind_published_retrieval_policy
                    retrieval_binding = bind_published_retrieval_policy(repository=repository,
                        bundle=runtime.context_bundle, task_cid=task.task_cid, task_id=task.task_key)
            runtime.run_id = "admitted-" + uuid.uuid4().hex
            runtime._published_context = {}
            runtime._context_refresh_attempts = {}
            runtime._context_refresh_stop_receipt = None
            runtime.client_id = "database-implementation-daemon:" + runtime.run_id
            runtime._verify_tasks(verified)
            runtime._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            if inventory_execution_scope is not None:
                inventory_execution_scope._remember_constructor_cleanup(runtime)
            runtime._listener.bind("\0ipfs-local-" + uuid.uuid4().hex)
            runtime._listener.listen(4)
            runtime._listener.settimeout(0.25)
            runtime._bootstrap_stop = threading.Event()
            if inventory_execution_scope is not None:
                inventory_execution_scope._remember_constructor_cleanup(runtime)
            runtime.bootstrap_receipts = []
            runtime.bootstrap_errors = []
            options = [
                "--todo-path", str(server.config.database_path), "--state-dir", str(runtime.state / "run"),
                "--state-prefix", "admitted", "--task-prefix", "## ",
                "--task-source-kind", "duckdb", "--authority-mode", "quack",
                "--quack-endpoint", server.identity.listen_uri,
                "--endpoint-secret-handle", server.identity.secret_handle,
                "--state-store-id", server.identity.store_id,
                "--state-store-generation", str(server.identity.generation),
                "--state-schema-revision", str(server.identity.schema_revision),
                "--state-owner-bootstrap-fd", str(runtime._listener.fileno()),
                "--state-owner-client-id", runtime.client_id,
                "--check-interval", "0.25", "--daemon-interval", "0.25", "--max-restarts", "1",
                "--max-task-attempts", str(max_task_attempts), "--strict-task-sharding",
                "--no-worktree-reconciliation", "--no-retry-budget-guardrail",
                "--no-dependency-guardrail", "--no-reconciliation-guardrail",
                "--no-objective-task-janitor", "--no-objective-goal-refinement",
                "--no-objective-goal-completion-reconcile", "--no-objective-goal-migration",
                "--no-objective-ast-dataset", "--no-objective-todo-vector-index",
                "--merge-target-branch", _git(repository, "branch", "--show-current"),
                "--merge-queue-dir", str(runtime.state / "merge_queue"),
                "--worktree-root", str(worker_worktree_root or runtime.state / "worktrees"),
            ]
            for task in verified["graph"].tasks:
                options += ["--execution-slice-task-cid", task.task_cid,
                            "--execution-slice-task-id", task.task_key]
            if runtime.context_bundle is not None:
                options += ["--task-context-bundle-artifact", runtime.context_bundle["artifact"],
                            "--task-context-bundle-sha256", runtime.context_bundle["sha256"]]
            options += ["--implement", "--implementation-command", implementation_command] if implement else ["--no-implement"]
            if implementation_timeout_seconds is not None:
                options += ["--implementation-timeout", str(implementation_timeout_seconds)]
            from ..todo_daemon.implementation_supervisor import parse_args
            parse_args(options)
            argv = (sys.executable, "-P", "-m",
                    "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor", *options)
            pythonpath = os.pathsep.join(filter(None, (str(Path(__file__).resolve().parents[3]), os.environ.get("PYTHONPATH", ""))))
            from ..task_sources.board_control_plane import ORCHESTRATION_DIR_ENV
            # The signed isolated run owns its derived catalogs. An unconfigured
            # lookup migrates account-wide legacy databases before startup.
            orchestration = runtime.state / "orchestration"
            orchestration.mkdir(mode=0o700)
            environment = (("PYTHONPATH", pythonpath), ("PYTHONUNBUFFERED", "1"),
                           (ORCHESTRATION_DIR_ENV, str(orchestration)))
            environment += tuple(proof_resource_environment.items())
            if replay_work_scope is not None:
                # Receive allowance only; the owner enforces the shorter
                # original work/lifetime/grant and whole-callback deadlines.
                from ..task_sources.typed_state_owner import LOCAL_COMPLETION_RPC_TIMEOUT_ENV
                environment += ((LOCAL_COMPLETION_RPC_TIMEOUT_ENV, "125"),)
            from ..todo_daemon.native_owner_bootstrap import STARTUP_WAIT_ENV
            # Bind the default too: ambient state must not enlarge this wait.
            environment += ((STARTUP_WAIT_ENV, str(start_timeout_ms or 30_000)),)
            environment += tuple(_bounded_git_environment(candidate_runner=candidate_runner is not None).items())
            if candidate_runner is not None:
                environment += ((CANDIDATE_RUNNER_ENV, json.dumps(candidate_runner, sort_keys=True)),)
            runtime.manifest = {
                "schema": "isolated-admitted-supervisor-launch@1", "run_id": runtime.run_id,
                "repository_root": str(repository), "repository_id": runtime.repository_id,
                "baseline_commit": runtime.baseline, "baseline_tree_id": runtime.tree_id,
                "state_root": str(runtime.state),
                "local_admission": runtime.admission, "owner_identity": server.identity.to_dict(),
                "execution_route_policy_id": runtime.route_policy.policy_id,
                "argv": list(argv), "argv_digest": _digest(argv),
                "environment": [list(item) for item in environment],
                "bootstrap_listener_inode": os.fstat(runtime._listener.fileno()).st_ino,
                "bootstrap_client_id": runtime.client_id, "provider_dispatch_allowed": implement,
                "coordination_attempt_safety_cap": max_task_attempts,
                "lifetime_seconds": lifetime_seconds,
                "worker_worktree_root": str(worker_worktree_root) if worker_worktree_root else None,
                "candidate_runner": candidate_runner,
                "task_context_bundle": runtime.context_bundle,
                "published_retrieval_policy": retrieval_binding,
                "context_refresh_policy": ({
                    "schema": "admitted-context-refresh-policy@1",
                    "output_root": ".runtime/published-context/" + runtime.run_id,
                    "trigger": "after_native_stop",
                    "max_attempts_per_task": 2,
                    "completion_authority": False,
                } if refresh_context_on_completion else None),
                "production_activation": False, "completion_authority": False,
            }
            if start_timeout_ms is not None:
                runtime.manifest["start_timeout_ms"] = start_timeout_ms
            if implementation_timeout_seconds is not None:
                runtime.manifest["implementation_timeout_seconds"] = implementation_timeout_seconds
            if refresh_source384_on_completion:
                runtime.manifest["context_refresh_policy"].update(
                    schema="admitted-context-refresh-policy@2",
                    source384={"schema": "admitted-source384-successor-policy@1",
                        "output_root": str(runtime.state / "published-source384"),
                        "preparation_timeout_seconds": 180.,
                        "requires_independent_manifest": True,
                        "proof_authority": False, "completion_authority": False})
            if finite_execution_scope is not None:
                runtime.manifest["finite_execution_scope"] = finite_execution_scope.material_binding
                runtime.manifest["finite_worker_launcher"] = finite_execution_scope.worker_launcher_binding
                finite_execution_scope.bind_runtime(runtime)
                if finite_execution_scope._uses_paired_receiving():
                    from ..runtime.finite_proof_query_worker_dispatch import start_owner_finite_proof_query_dispatch_broker
                    runtime.proof_query_dispatch_broker = start_owner_finite_proof_query_dispatch_broker(
                        scope=finite_execution_scope, runtime=runtime)
                    environment += tuple(runtime.proof_query_dispatch_broker.daemon_environment().items())
                    runtime.manifest["environment"] = [list(item) for item in environment]
            if inventory_execution_scope is not None:
                runtime.manifest["inventory_execution_scope"] = inventory_execution_scope.material_binding
                runtime.manifest["inventory_worker_launcher"] = inventory_execution_scope.worker_launcher_binding
                inventory_execution_scope.bind_runtime(runtime)
            runtime.manifest_id = _digest(runtime.manifest)
            runtime.signature = sign_profile_binding(profile_dir=runtime.profile_dir,
                                                    lifecycle_dir=runtime.lifecycle_dir, payload=runtime.manifest)
            if finite_execution_scope is not None:
                finite_execution_scope.require_runtime(runtime, before_spawn=True)
            if inventory_execution_scope is not None:
                inventory_execution_scope.require_runtime(runtime, before_spawn=True)
            (runtime.state / "local-process-grant.json").write_text(json.dumps({
                "manifest": runtime.manifest, "signature": runtime.signature,
            }, sort_keys=True, indent=2) + "\n")
            runtime.profile = LifecycleProfile(
                target_id=runtime.repository_id, run_id=runtime.run_id, configuration_root=runtime.manifest_id,
                repository_root=str(repository), state_root=str(runtime.state), run_root=str(runtime.state / "run"),
                argv=argv, cwd=str(repository), environment=environment,
                health_path=str(runtime.state / "run" / "admitted_supervisor_status.json"), health_stale_ms=5_000,
            )
            runtime.coordinator = open_database_coordinator(runtime.state / "coordination.duckdb")
            if inventory_execution_scope is not None:
                inventory_execution_scope._remember_constructor_cleanup(runtime)
            runtime.lease = runtime.coordinator.acquire(
                lease_kind="resource", scope=runtime.run_id, owner_session_id=profile.identity_did,
                lease_ms=lifetime_seconds * 1000, resource_kind="supervisor_run", resource_id=runtime.run_id,
                repository_id=runtime.repository_id, idempotency_key=runtime.manifest_id,
                body={"local_profile_id": profile.profile_id, "launch_grant": runtime.manifest_id},
            )
            runtime._children = []
            if inventory_execution_scope is not None:
                inventory_execution_scope._remember_constructor_cleanup(runtime)

            def popen(*args, **kwargs):
                from ..task_sources.duckdb_state import QUACK_TOKEN_ENV
                runtime._startup_validate("launch_validation")
                # Resolve through this live native owner, never ambient credentials
                # or a possibly stale filesystem handoff for another generation.
                token = server._vault.resolve(runtime.owner_identity.secret_handle)
                if not token:
                    raise ValueError("native owner's read-only Quack transport credential is unavailable")
                environment = dict(kwargs["env"])
                environment.pop(TYPED_STATE_OWNER_SOCKET_ENV, None)
                environment.pop(TYPED_STATE_OWNER_TOKEN_ENV, None)
                environment[QUACK_TOKEN_ENV] = token
                kwargs["env"] = environment
                kwargs["pass_fds"] = (runtime._listener.fileno(),)
                with (runtime.state / "supervisor-process.log").open("ab") as output:
                    kwargs["stdout"], kwargs["stderr"] = output, subprocess.STDOUT
                    if runtime.finite_execution_scope is not None:
                        runtime.finite_execution_scope.require_runtime(runtime, before_spawn=True)
                    if runtime.inventory_execution_scope is not None:
                        runtime.inventory_execution_scope.require_runtime(runtime, before_spawn=True)
                        runtime.inventory_execution_scope.prepare_spawn_fence(runtime)
                    with server._lock if (runtime.finite_execution_scope is not None
                                          or runtime.inventory_execution_scope is not None) else nullcontext():
                        # The index lock stays held across literal supervisor birth.
                        # Only failures while entering this guard can mint a no-effect refusal.
                        with ExitStack() as dispatch_fence:
                            if runtime.finite_execution_scope is not None:
                                try:
                                    runtime.finite_execution_scope.require_spawn_fence(runtime)
                                    dispatch_fence.enter_context(runtime.finite_execution_scope._proof_query_spawn_guard(runtime))
                                except Exception as error:
                                    from ..runtime.finite_repository_execution import ADVISORY_PROFILE
                                    from ..runtime.finite_proof_query_execution import PROFILE as PROOF_QUERY_PROFILE
                                    selected = runtime.finite_execution_scope.to_dict()["payload"]["profile"]
                                    if selected == ADVISORY_PROFILE:
                                        from ..control.before_popen_refusal import refuse_finite_advisory_before_popen
                                        raise refuse_finite_advisory_before_popen(runtime, error) from error
                                    if selected == PROOF_QUERY_PROFILE:
                                        from ..control.before_popen_refusal import refuse_finite_proof_query_before_popen
                                        raise refuse_finite_proof_query_before_popen(runtime, error) from error
                                    raise
                            if runtime.inventory_execution_scope is not None:
                                runtime.inventory_execution_scope.require_spawn_fence(runtime)
                            child = subprocess.Popen(*args, **kwargs)
                            runtime._children.append(child)
                            if runtime.finite_execution_scope is not None:
                                runtime.finite_execution_scope.note_spawned(runtime)
                            if runtime.inventory_execution_scope is not None:
                                runtime.inventory_execution_scope.note_spawned(runtime)
                return child

            runtime.process = AdmittedSupervisorHealthAdapter(popen=popen)
            runtime.process.owner_identity = runtime.owner_identity
            runtime.process.route_policy_id = runtime.route_policy.policy_id
            runtime.process.client_id = runtime.client_id
            orchestrator_type = LifecycleOrchestrator
            if replay_work_scope is not None:
                class ReplayBoundLifecycleOrchestrator(LifecycleOrchestrator):
                    def _start_new(self, state, profile, deadline):
                        # The control policy performs fresh checks after issuing
                        # its request. Charge that time before native health
                        # polling without rewriting the signed request or saga.
                        fixed = runtime._replay_start_deadline
                        if fixed is None:
                            raise TimeoutError("admitted START lacks its fixed replay deadline")
                        bounded = min(deadline, fixed, runtime._replay_lifetime_deadline,
                                      runtime._replay_work_scope.deadline_monotonic)
                        return super()._start_new(state, profile, bounded)

                orchestrator_type = ReplayBoundLifecycleOrchestrator
            runtime.orchestrator = orchestrator_type(
                state_root=runtime.state, profiles=(runtime.profile,), process_adapter=runtime.process,
                poll_interval_ms=50, stop_grace_ms=1_000,
            )
            runtime._permits, runtime._requests = {}, {}
            runtime.service = SupervisorControlService(
                repository_allowlist=(repository,), state_allowlist=(runtime.state,),
                handlers={Operation.START: runtime._bounded_lifecycle_response,
                          Operation.STOP: runtime._bounded_lifecycle_response},
                authorization_validator=ControlMutationAuthorizer(runtime._policy),
                identity_validator=runtime._validate_identity, lease_validator=runtime._validate_lease,
                local_start_timeout_ms=start_timeout_ms,
            )
            runtime._bootstrap_thread = threading.Thread(target=runtime._serve_bootstrap, daemon=True,
                                                         name="admitted-supervisor-bootstrap")
            if inventory_execution_scope is not None:
                inventory_execution_scope._remember_constructor_cleanup(runtime)
            runtime._bootstrap_thread.start()
            if implement:
                from ..runtime.local_completion_bridge import bind_owner_local_completion_service
                from ..runtime.finite_repository_execution import ADVISORY_PROFILE
                from ..runtime.finite_proof_query_execution import PROFILE as PROOF_QUERY_PROFILE
                from ..task_sources.board_control_plane import infer_board_namespace
                target_branch = _git(repository, "branch", "--show-current")
                if finite_execution_scope is None and inventory_execution_scope is None:
                    gateway = server._command_gateway
                    with gateway._grants_lock:
                        runtime._ordinary_completion_binding_before = (
                            gateway, gateway._local_task_validation_handler)
                runtime.completion_service = bind_owner_local_completion_service(
                    server=server, portal_attempt_root=runtime.state / "run" / "admitted_database_portal_attempts",
                    repo_root=repository, merge_queue_dir=runtime.state / "merge_queue",
                    board_namespace=infer_board_namespace(merge_target_branch=target_branch,
                                                           todo_path=Path(server.config.database_path), state_prefix="admitted"),
                    target_branch=target_branch,
                    candidate_runner=candidate_runner,
                    retirable=(inventory_execution_scope is not None or (finite_execution_scope is not None
                        and finite_execution_scope.to_dict()["payload"]["profile"] in {ADVISORY_PROFILE, PROOF_QUERY_PROFILE})),
                    **({"replay_scope": runtime._replay_work_scope,
                        "deadline_monotonic": runtime._replay_lifetime_deadline}
                       if runtime._replay_work_scope is not None else {}),
                )

            return runtime
        except BaseException as construction_error:
            if inventory_execution_scope is None and (
                    finite_execution_scope is None or not finite_execution_scope._uses_paired_receiving()):
                if finite_execution_scope is None:
                    try:
                        runtime._close_unlaunched_construction()
                    except BaseException as cleanup_error:
                        runtime._construction_cleanup_failed = True
                        raise RuntimeError("ordinary constructor cleanup unproven; resource custody retained") from cleanup_error
                raise
            cleanup_kind = "inventory" if inventory_execution_scope is not None else "proof-query"
            from ..runtime.local_completion_bridge import _CompletionBindingCleanupError
            if type(construction_error) is _CompletionBindingCleanupError:
                runtime._pending_completion_binding = construction_error._custody
            if inventory_execution_scope is not None:
                inventory_execution_scope._retain_constructor_cleanup(runtime)
            try:
                if all(hasattr(runtime, name) for name in
                       ("service", "process", "profile", "lease", "_bootstrap_thread")):
                    stopped = runtime.stop()
                    if not stopped.succeeded or not runtime._context_refresh_stopped():
                        raise RuntimeError(f"failed {cleanup_kind} construction lacks native STOP cleanup")
                    runtime.close()
                else:
                    # No lifecycle or completion service exists at this stage,
                    # and no process has been launched. This is transport/run
                    # lease disposal, never an invented STOP receipt.
                    if getattr(runtime, "_children", ()) or getattr(runtime, "completion_service", None):
                        raise RuntimeError(f"partial {cleanup_kind} construction has undisposed process custody")
                    broker = getattr(runtime, "proof_query_dispatch_broker", None)
                    if broker is not None:
                        broker.close_unlaunched_construction()
                    listener = getattr(runtime, "_listener", None)
                    if listener is not None:
                        listener.close()
                    coordinator = getattr(runtime, "coordinator", None)
                    if coordinator is not None:
                        if inventory_execution_scope is not None:
                            inventory_execution_scope._close_constructor_run_coordinator(runtime)
                        else:
                            lease = getattr(runtime, "lease", None)
                            if lease is not None:
                                coordinator.release(lease, expected_fencing_token=lease.fencing_token,
                                                    expected_fence_epoch=lease.fence_epoch)
                            coordinator.close()
            except BaseException as cleanup_error:
                runtime._construction_cleanup_failed = True
                if hasattr(runtime, "state"):
                    (runtime.state / "construction-cleanup-failure.json").write_text(json.dumps({
                        "construction_error_type": type(construction_error).__name__,
                        "cleanup_error_type": type(cleanup_error).__name__,
                        "native_STOP_proved": False, "resources_retained": True,
                    }, sort_keys=True) + "\n")
                raise RuntimeError(f"{cleanup_kind} constructor cleanup unproven; resource custody retained") from cleanup_error
            raise

    def _close_unlaunched_construction(self):
        """Dispose an ordinary failed constructor without inventing a STOP."""
        if getattr(self, "_children", ()):
            raise RuntimeError("ordinary construction has undisposed process custody")
        process = getattr(self, "process", None)
        if process is not None:
            observed = process.snapshot(self.profile)
            if observed.members or observed.roots:
                raise RuntimeError("ordinary construction has a live native process tree")
        stop = getattr(self, "_bootstrap_stop", None)
        if stop is not None:
            stop.set()
        listener = getattr(self, "_listener", None)
        if listener is not None:
            listener.close()
        thread = getattr(self, "_bootstrap_thread", None)
        if thread is not None and thread.ident is not None:
            # The bootstrap channel itself has a five-second receive bound.
            thread.join(timeout=6)
            if thread.is_alive():
                raise RuntimeError("ordinary constructor bootstrap remains active")
        before = getattr(self, "_ordinary_completion_binding_before", None)
        if before is not None:
            gateway, previous_handler = before
            with gateway._grants_lock:
                if gateway._local_task_validation_handler is not previous_handler:
                    raise RuntimeError("ordinary construction has unresolved completion binding custody")
        if getattr(self, "completion_service", None):
            raise RuntimeError("ordinary construction has undisposed completion custody")
        coordinator = getattr(self, "coordinator", None)
        released = False
        if coordinator is not None:
            lease = getattr(self, "lease", None)
            if lease is not None:
                coordinator.release(lease, expected_fencing_token=lease.fencing_token,
                                    expected_fence_epoch=lease.fence_epoch)
                released = True
            coordinator.close()
        if getattr(self, "state", None) is not None and self.state.is_dir():
            (self.state / "construction-cleanup.json").write_text(json.dumps({
                "schema": "ordinary-unlaunched-constructor-cleanup@1",
                "native_STOP_proved": False,
                "process_observation": "empty_native_tree" if process is not None else "lifecycle_not_constructed",
                "bootstrap_stopped": thread is None or not thread.is_alive(),
                "run_lease_released": released,
                "completion_authority": False,
            }, sort_keys=True) + "\n")

    def _bounded_lifecycle_response(self, request):
        if request.operation is Operation.START and self._replay_work_scope is not None:
            if self._replay_start_deadline is None:
                raise TimeoutError("admitted START lacks its fixed replay deadline")
            # Refuse exhausted validation time before creating a lifecycle
            # reservation; a new rejection must not strand a PREPARED saga.
            self._operation_timeout_ms(Operation.START)
        response = self.orchestrator(request)
        worker_cleanup = None
        if request.operation is Operation.STOP and self.manifest.get('candidate_runner') is not None:
            if response.data.get('old_tree_fenced') is not True:
                raise RuntimeError('worker cleanup requires the native STOP fencing receipt')
            # A privilege-separated provider can create a new process session.
            # Native owner-tree fencing cannot itself signal that other UID.
            # The signed single-worker binding delegates this exact cleanup
            # to its fixed worker entry, which checks its namespace and reaps
            # every remaining process of the isolated worker identity.
            from ..runtime.candidate_execution import verify_candidate_runner
            binding = verify_candidate_runner(self.manifest['candidate_runner'])
            entry = str(Path(binding['argv'][0]).with_name('worker-entry'))
            completed = subprocess.run(
                ['/usr/bin/sudo', '-n', '-u', 'benchmarkworker', '--', entry, '--cleanup'],
                cwd='/', stdin=subprocess.DEVNULL, capture_output=True, timeout=15,
            )
            worker_cleanup = {'schema': 'isolated-worker-stop-observation@1',
                              'worker_uid': binding['worker_uid'],
                              'namespaces': binding['namespaces'],
                              'returncode': completed.returncode,
                              'single_worker': True, 'completion_authority': False}
            self._record('worker-stop', worker_cleanup)
            if completed.returncode != 0:
                raise RuntimeError('signed isolated worker cleanup did not complete')
        # The native journal retains the complete receipt and process trees.
        # Return its identity and independently observed root within the
        # canonical control catalog's fixed result bound.
        self._record("native-lifecycle", dict(response.data))
        data = {key: value for key, value in response.data.items() if key != "transition"}
        data["transition_artifact_sha256"] = _digest(response.data["transition"])
        if worker_cleanup is not None:
            data['isolated_worker_cleanup'] = worker_cleanup
        return replace(response, data=data)

    def _verify_tasks(self, verified):
        from ..task_sources.intent_repository import IntentRepository

        page = self.source.list_tasks(limit=17)
        expected = {task.task_cid: task for task in verified["graph"].tasks}
        if page.next_cursor or {task.task_cid for task in page.tasks} != set(expected):
            raise ValueError("native owner task population differs from signed admission")
        with self.server._lock:
            intent = IntentRepository(bound_connection=self.server._connection, install_schema=False)
            for task in page.tasks:
                envelope = task.body.get(CONTRACT_KEY, {})
                contract = _verify_signature(envelope, verified["profile"])
                native = intent.get_task(task.task_cid)
                if (native is None or native["body"].get(CONTRACT_KEY) != envelope
                        or native["identity"].get("local_contract_cid") != content_identity(envelope)):
                    raise ValueError("native task contract differs from its persisted identity")
                owner_id = contract.get("intent_owner_id")
                if not isinstance(owner_id, str) or not owner_id.strip():
                    raise ValueError("native task lacks the exact signed local contract owner")
                expected_contract = _pending_contract_payload(
                    admission=self.admission, verified=verified,
                    task=expected[task.task_cid], intent_owner_id=owner_id,
                )
                if (contract != expected_contract
                        or task.task_alias != expected_contract["task_key"]
                        or list(task.dependencies) != expected_contract["dependencies"]
                        or native["task_alias"] != expected_contract["task_key"]
                        or list(native["dependencies"]) != expected_contract["dependencies"]
                        or native["goal_cid"] != expected[task.task_cid].goal_cid
                        or native["plan_cid"] != expected_contract["plan_id"]):
                    raise ValueError("native task lacks the exact signed local contract")

    def _verify_context(self, verified, *, completed_observation=False):
        observations = []
        if self.context_bundle is None:
            return observations
        from ..runtime.task_context_bundle import load_task_context_nomination, read_task_context_historical_selection
        if set(self.context_bundle) != {"artifact", "sha256"}:
            raise ValueError("context bundle requires exact artifact and digest")
        published_observation = (completed_observation
            and _git(self.repository, "rev-parse", "HEAD") != verified["profile"].baseline_commit)
        for task in verified["graph"].tasks:
            if published_observation:
                selected = read_task_context_historical_selection(repository=self.repository,
                    artifact=self.context_bundle["artifact"], expected_sha256=self.context_bundle["sha256"],
                    task_cid=task.task_cid, task_id=task.task_key)
                if "source384_context" in selected:
                    from ..runtime.published_task_context import observe_source384_successor_unavailable
                    observations.append(observe_source384_successor_unavailable(server=self.server,
                        admission=self.admission, predecessor_bundle=self.context_bundle, task_cid=task.task_cid))
                    continue
            load_task_context_nomination(repository=self.repository,
                                         artifact=self.context_bundle["artifact"],
                                         expected_sha256=self.context_bundle["sha256"],
                                         task_cid=task.task_cid, task_id=task.task_key)
        return observations

    def _verify(self, *, completed_observation=False):
        from ..runtime.local_completion_bridge import verify_owner_local_benchmark_observation
        if self.manifest.get("start_timeout_ms") != self.start_timeout_ms:
            raise ValueError("admitted START budget differs from its signed launch")
        if self.manifest.get("implementation_timeout_seconds") != getattr(self, "implementation_timeout_seconds", None):
            raise ValueError("implementation timeout differs from its signed launch")
        if self.service._local_start_timeout_ms != self.start_timeout_ms:
            raise ValueError("local START service budget differs from its signed launch")
        if self.manifest.get("candidate_runner") is not None:
            from ..runtime.candidate_execution import verify_candidate_runner
            verify_candidate_runner(self.manifest["candidate_runner"])
        verified = verify_owner_local_benchmark_observation(server=self.server, admission=self.admission)
        if verified["profile"] != self.local_profile or self.server.identity != self.owner_identity:
            raise ValueError("local profile or native owner generation changed")
        if self.server.lifecycle is not ServerLifecycle.READY:
            raise ValueError("native owner is no longer ready")
        if self.source.execution_route_policy != self.route_policy:
            raise ValueError("native execution route changed")
        self._verify_tasks(verified)
        observations = self._verify_context(verified, completed_observation=completed_observation)
        if dict(self.profile.environment) != dict(self.manifest["environment"]):
            raise ValueError("admitted launch environment changed")
        if (_digest(self.manifest) != self.manifest_id
                or list(self.profile.argv) != self.manifest["argv"]
                or os.fstat(self._listener.fileno()).st_ino != self.manifest["bootstrap_listener_inode"]):
            raise ValueError("admitted launch grant changed")
        persisted = json.loads((self.state / "local-process-grant.json").read_text())
        if persisted != {"manifest": self.manifest, "signature": self.signature}:
            raise ValueError("persisted admitted launch grant changed")
        verify_did_key_signature(identity_did=self.local_profile.identity_did,
                                 payload=self.manifest, signature=self.signature["signature"])
        if self.finite_execution_scope is not None:
            self.finite_execution_scope.require_runtime(self)
        if self.inventory_execution_scope is not None:
            self.inventory_execution_scope.require_runtime(self)
        return observations

    def _verify_observation(self):
        from ..runtime.header_intent_applicability import resume_applicability_budget, require_applicability_budget
        with resume_applicability_budget(self._replay_work_scope,
                deadline_monotonic=self._replay_lifetime_deadline):
            observations = self._verify(completed_observation=True)
            require_applicability_budget()
        return {"source384": observations} if observations else None

    def _operation_timeout_ms(self, operation):
        configured = self._configured_operation_timeout_ms(operation)
        if operation is Operation.START and self._replay_work_scope is not None:
            deadline = min(self._replay_work_scope.deadline_monotonic, self._replay_lifetime_deadline)
            if self._replay_start_deadline is not None:
                deadline = min(deadline, self._replay_start_deadline)
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms < 2_000:
                raise TimeoutError("admitted START has insufficient remaining replay budget")
            return min(configured, remaining_ms)
        return configured

    def _configured_operation_timeout_ms(self, operation):
        if operation is Operation.START and self.start_timeout_ms is not None:
            if (type(self.start_timeout_ms) is not int
                    or not 2_000 <= self.start_timeout_ms <= 120_000
                    or self.manifest.get("start_timeout_ms") != self.start_timeout_ms):
                raise ValueError("admitted START budget differs from its signed launch")
            return self.start_timeout_ms
        return super()._operation_timeout_ms(operation)

    def _startup_validate(self, phase):
        """Time each fresh check without reusing any validation result."""
        if phase not in {"control_validation", "launch_validation", "bootstrap_validation"}:
            raise ValueError("unknown native startup observation phase")
        observation = {"phase": phase, "status": "running", "started": time.monotonic(), "finished": None}
        with self._startup_trace_lock:
            if len(self._startup_trace) < 16:
                self._startup_trace.append(observation)
            else:
                self._startup_trace_truncated = True
        status = "failed"
        try:
            from ..runtime.header_intent_applicability import resume_applicability_budget, require_applicability_budget
            deadline = self._replay_lifetime_deadline
            if self._replay_start_deadline is not None:
                deadline = min(deadline, self._replay_start_deadline)
            # Explicit handoff is required in the bootstrap thread. ContextVars
            # from the driver or control thread are not inherited there.
            with resume_applicability_budget(self._replay_work_scope, deadline_monotonic=deadline):
                result = self._verify()
                require_applicability_budget()
            status = "completed"
            return result
        finally:
            with self._startup_trace_lock:
                observation.update(status=status, finished=time.monotonic())

    def startup_diagnostics(self):
        """Closed observations only: no credentials, errors, or source bodies."""
        now = time.monotonic()
        with self._startup_trace_lock:
            observations = [{"phase": item["phase"], "status": item["status"],
                "seconds": min(900.0, max(0.0, (item["finished"] or now) - item["started"]))}
                for item in self._startup_trace]
            truncated = self._startup_trace_truncated
        return {"schema": "admitted-native-startup-observation@1",
            "start_timeout_ms": self.start_timeout_ms or self.timeout_ms,
            "stop_timeout_ms": self.timeout_ms,
            "bootstrap_wait_seconds": (self.start_timeout_ms or 30_000) / 1000,
            "observations": observations, "observations_truncated": truncated,
            "bootstrap_receipt_count": min(65_535, len(self.bootstrap_receipts)),
            "bootstrap_error_count": min(65_535, len(self.bootstrap_errors))}

    def _verify_operation(self, operation):
        if operation is not Operation.STOP:
            return self._startup_validate("control_validation")
        # Stopping this already admitted process tree must remain possible
        # between publication and its owner validation. Source acceptance and
        # task completion grant no shutdown authority; the original immutable
        # launch, lifecycle policy, exact process births and live lease do.
        for path in (self.directory, self.repository, self.state,
                     self.profile_dir, self.lifecycle_dir):
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("admitted shutdown root changed")
        profile = load_local_profile(
            repository_cid=self.repository_id, profile_dir=self.profile_dir,
            lifecycle_dir=self.lifecycle_dir,
        )
        if profile != self.local_profile or profile.baseline_commit != self.baseline:
            raise ValueError("admitted shutdown profile generation changed")
        manifest = self.manifest
        if (_digest(manifest) != self.manifest_id
                or manifest["repository_root"] != str(self.repository)
                or manifest["state_root"] != str(self.state)
                or manifest["repository_id"] != self.repository_id
                or manifest["run_id"] != self.run_id
                or manifest["baseline_commit"] != self.baseline
                or manifest["baseline_tree_id"] != self.tree_id
                or manifest["local_admission"] != self.admission
                or manifest["owner_identity"] != self.owner_identity.to_dict()
                or self.state != self.directory / "state"):
            raise ValueError("admitted shutdown launch grant changed")
        expected_profile = LifecycleProfile(
            target_id=manifest["repository_id"], run_id=manifest["run_id"],
            configuration_root=self.manifest_id,
            repository_root=manifest["repository_root"], state_root=manifest["state_root"],
            run_root=str(self.state / "run"), argv=tuple(manifest["argv"]),
            cwd=manifest["repository_root"],
            environment=tuple(tuple(item) for item in manifest["environment"]),
            health_path=str(self.state / "run/admitted_supervisor_status.json"),
            health_stale_ms=5_000,
        )
        if self.profile != expected_profile:
            raise ValueError("shutdown lifecycle profile differs from signed launch")
        grant = self.state / "local-process-grant.json"
        if grant.is_symlink() or not grant.is_file() or json.loads(grant.read_text()) != {
            "manifest": manifest, "signature": self.signature,
        }:
            raise ValueError("persisted admitted shutdown grant changed")
        if (self.signature["identity"] != profile.identity_did
                or self.signature["profile_id"] != profile.profile_id):
            raise ValueError("shutdown signer differs from the installed profile")
        verify_did_key_signature(identity_did=profile.identity_did, payload=manifest,
                                 signature=self.signature["signature"])
        if manifest.get("candidate_runner") is not None:
            from ..runtime.candidate_execution import verify_candidate_runner
            verify_candidate_runner(manifest["candidate_runner"])
        if self.finite_execution_scope is not None:
            self.finite_execution_scope.require_runtime(self, stopping=True)
        if self.inventory_execution_scope is not None:
            self.inventory_execution_scope.require_runtime(self, stopping=True)

    def _serve_bootstrap(self):
        while not self._bootstrap_stop.is_set():
            try:
                channel, _address = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with channel:
                channel.settimeout(5)
                try:
                    self._startup_validate("bootstrap_validation")
                    pid, uid, _gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.geteuid():
                        raise ValueError("bootstrap peer belongs to another owner")
                    request = _receive_frame(channel)
                    birth = read_process_birth(pid)
                    if (set(request) != {"schema", "pid", "process_birth", "process_birth_id", "client_id", "store_id"}
                            or request["schema"] != STATE_OWNER_BOOTSTRAP_REQUEST_SCHEMA
                            or birth is None or request["pid"] != pid or request["process_birth"] != birth.to_dict()
                            or request["process_birth_id"] != process_birth_id(birth)
                            or request["client_id"] != self.client_id or request["store_id"] != self.owner_identity.store_id):
                        raise ValueError("bootstrap request differs from kernel peer and exact grant")
                    tree = self.process.snapshot(self.profile)
                    matches = [item for item in tree.members if item.pid == pid]
                    if (len(tree.roots) != 1 or len(matches) != 1 or pid == tree.roots[0].pid
                            or matches[0].parent_pid != tree.roots[0].pid
                            or not self.process.child_scope_matches(self.profile, matches[0])):
                        raise ValueError(
                            "bootstrap peer is not the exact native supervisor child "
                            f"(roots={len(tree.roots)}, matches={len(matches)}, "
                            f"parent={matches[0].parent_pid if matches else 0}, "
                            f"root={tree.roots[0].pid if tree.roots else 0}, "
                            f"scope={bool(matches and self.process.child_scope_matches(self.profile, matches[0]))}, "
                            f"option={self.process.last_scope_mismatch})"
                        )
                    if any(item["process_birth_id"] == process_birth_id(birth) for item in self.bootstrap_receipts):
                        raise ValueError("bootstrap for this child birth was already issued")
                    protected = self.coordinator.protect_write(
                        self.lease, expected_fencing_token=self.lease.fencing_token,
                        expected_fence_epoch=self.lease.fence_epoch,
                    )
                    if protected.resource_id != self.run_id or protected.owner_session_id != self.local_profile.identity_did:
                        raise ValueError("bootstrap run lease changed owner")
                    remaining_seconds = (protected.expires_at_ms - time.time_ns() // 1_000_000) // 1000 - 1
                    if remaining_seconds < 1:
                        raise ValueError("native launch lease has no remaining child grant lifetime")
                    token, _grant = self.server.issue_typed_client_grant_record(
                        client_id=self.client_id, process_birth_id=process_birth_id(birth), peer_pid=pid,
                        allowed_operations=daemon_required_owner_operations(),
                        allowed_command_operations=daemon_required_owner_command_operations(),
                        ttl_seconds=min(self.manifest["lifetime_seconds"], remaining_seconds),
                    )
                    if _grant.expires_at > protected.expires_at_ms:
                        raise ValueError("child grant issuance exceeded its native run lease")
                    self.process.remember_bootstrapped_child(matches[0])
                    receipt = {"schema": "admitted-child-bootstrap-observation@1", "pid": pid,
                               "process_birth_id": process_birth_id(birth), "client_id": self.client_id,
                               "store_id": self.owner_identity.store_id, "server_id": self.owner_identity.server_id,
                               "configuration_root": self.manifest_id, "route_policy_id": self.route_policy.policy_id,
                               "grant_expires_at_ms": _grant.expires_at,
                               "run_lease_expires_at_ms": protected.expires_at_ms,
                               "credential_in_argv": False, "completion_authority": False}
                    self.bootstrap_receipts.append(receipt)
                    self._record("child-bootstrap", receipt)
                    _send_frame(channel, {"schema": STATE_OWNER_BOOTSTRAP_RESPONSE_SCHEMA, "ok": True,
                                          "endpoint": self.owner_identity.listen_uri,
                                          "socket_path": str(self.server.typed_command_socket_path()),
                                          "store_id": self.owner_identity.store_id, "server_id": self.owner_identity.server_id,
                                          "client_id": self.client_id, "process_birth_id": process_birth_id(birth),
                                          "token": token, "execution_route_policy": self.route_policy.to_dict()})
                except Exception as exc:
                    self.bootstrap_errors.append({"type": type(exc).__name__, "message": str(exc)[:512]})

    def start(self):
        from ..runtime.header_intent_applicability import resume_applicability_budget, require_applicability_budget
        if self._replay_work_scope is None:
            return self._start_in_replay_scope()
        deadline = min(self._replay_lifetime_deadline, self._replay_work_scope.deadline_monotonic,
                       time.monotonic() + self._configured_operation_timeout_ms(Operation.START) / 1000)
        # Keep the fixed deadline available to the independent bootstrap
        # thread even if the control call times out while it is unwinding.
        self._replay_start_deadline = deadline
        with resume_applicability_budget(self._replay_work_scope, deadline_monotonic=deadline):
            result = self._start_in_replay_scope()
            if result.succeeded:
                require_applicability_budget()
                # Later replacement daemon births use the remaining work/run
                # lifetime. In-flight or failed START retains its fixed cap.
                self._replay_start_deadline = None
            return result

    def _start_in_replay_scope(self):
        if self.finite_execution_scope is not None and self.finite_execution_scope._uses_paired_receiving():
            from ..runtime.finite_proof_query_execution import _ProofQueryStartRefused, FiniteProofQueryExecutionError
            try:
                with self.finite_execution_scope._receiving_operation(purpose="start", runtime=self) as operation:
                    verify_local_benchmark_admission(self.admission, initial=True)
                    self.finite_execution_scope.require_runtime(self, before_spawn=True)
                    self._context_refresh_stop_receipt = None
                    result = super().start()
                    if not operation.closed:
                        if result.succeeded:
                            raise FiniteProofQueryExecutionError("native START succeeded without its closing proof-query fence")
                        raise _ProofQueryStartRefused(result)
                return result
            except _ProofQueryStartRefused as refused:
                return refused.result
        if self.inventory_execution_scope is not None and self.inventory_execution_scope._uses_paired_receiving():
            from ..runtime.codebase_inventory_execution import _InventoryStartRefused, InventoryExecutionError
            try:
                with self.inventory_execution_scope._receiving_operation(purpose="start", runtime=self) as operation:
                    verify_local_benchmark_admission(self.admission, initial=True)
                    self.inventory_execution_scope.require_runtime(self, before_spawn=True)
                    self._context_refresh_stop_receipt = None
                    result = super().start()
                    if not operation.closed:
                        if result.succeeded:
                            raise InventoryExecutionError("native START succeeded without its closing receiving fence")
                        raise _InventoryStartRefused(result)
                return result
            except _InventoryStartRefused as refused:
                return refused.result
        verify_local_benchmark_admission(self.admission, initial=True)
        if self.finite_execution_scope is not None:
            self.finite_execution_scope.require_runtime(self, before_spawn=True)
        if self.inventory_execution_scope is not None:
            self.inventory_execution_scope.require_runtime(self, before_spawn=True)
        self._context_refresh_stop_receipt = None
        return super().start()

    def stop(self):
        self._context_refresh_stop_receipt = None
        result = super().stop()
        if result.succeeded:
            self._context_refresh_stop_receipt = result
            if self.finite_execution_scope is not None:
                self.finite_execution_scope.finish_runtime(self)
            if self.inventory_execution_scope is not None:
                self.inventory_execution_scope.finish_runtime(self)
        return result

    def _context_refresh_stopped(self):
        receipt = self._context_refresh_stop_receipt
        if (receipt is None or not receipt.succeeded
                or receipt.data.get("old_tree_fenced") is not True
                or self.process.snapshot(self.profile).members):
            return False
        if self.manifest.get("candidate_runner") is not None:
            cleanup = receipt.data.get("isolated_worker_cleanup", {})
            if cleanup.get("returncode") != 0 or cleanup.get("single_worker") is not True:
                return False
        return True

    def refresh_after_stop(self):
        """Observe completed context using a separate post-shutdown budget.

        Cold source reconstruction never runs within mandatory STOP or close.
        An opted-in ordinary observation after STOP calls the same path.
        """
        if self.manifest.get("context_refresh_policy") is None:
            raise ValueError("published context refresh was not admitted by this launch")
        if not self._context_refresh_stopped():
            raise ValueError("published context refresh requires successful native STOP")
        return self.observe()["published_context"]

    def observe(self):
        from ..runtime.header_intent_applicability import resume_applicability_budget, require_applicability_budget
        # Requirement and successor observations share the same immutable
        # work scope as their initial owner/source verification.
        with resume_applicability_budget(self._replay_work_scope,
                deadline_monotonic=self._replay_lifetime_deadline):
            result = self._observe_in_replay_scope()
            require_applicability_budget()
            return result

    def _observe_in_replay_scope(self):
        result = super().observe()
        if self.admission["manifest"]["payload"]["schema"] == "supervisor-local-benchmark-manifest@4":
            from ..runtime.intent_requirement_observation import observe_owner_intent_requirements
            from ..planning.intent_requirement_repair import build_intent_requirement_repair_proposal

            try:
                requirements = observe_owner_intent_requirements(server=self.server, admission=self.admission)
                result["intent_requirements"] = requirements
            except (ValueError, TypeError, KeyError, OSError) as exc:
                result["intent_requirements"] = {
                    "schema": "intent-requirement-observation@1", "status": "unavailable",
                    "reason": type(exc).__name__, "message": str(exc)[:1024],
                    "source_semantics_verified": False, "official_reward": None,
                    "semantic_alignment_verified": False, "proof_authority": False,
                    "execution_authority": False, "completion_authority": False,
                    "canonical_state_mutated": False, "provider_calls": 0,
                }
            else:
                try:
                    result["intent_requirement_repair"] = build_intent_requirement_repair_proposal(requirements)
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    result["intent_requirement_repair"] = {
                        "schema": "intent-requirement-repair-proposal@1", "status": "unavailable",
                        "reason": type(exc).__name__, "message": str(exc)[:1024],
                        "observation_cid": requirements["observation_cid"],
                        "source_semantics_verified": False, "semantic_alignment_verified": False,
                        "official_reward": None, "nomination_only": True, "provider_calls": 0,
                        "write_authority": False, "proof_authority": False,
                        "execution_authority": False, "completion_authority": False,
                        "canonical_state_mutated": False,
                    }
        policy = self.manifest.get("context_refresh_policy")
        if policy is None:
            return result
        if policy.get("trigger") != "after_native_stop":
            raise ValueError("published context refresh trigger differs from signed policy")
        # super().observe verifies the signed manifest, exact owner and current
        # publication before any derived state is built. The launch nomination
        # remains immutable; successors are returned for the next planning pass.
        from ..runtime.published_task_context import (
            refresh_published_task_context, load_published_task_context,
        )
        from ..prompt.prompt_goal_planner import PromptGoalGraph
        refresh_started = time.monotonic()
        refreshed = []
        for task in PromptGoalGraph.from_dict(self.admission["graph"]).tasks:
            current = self.source.get_task(task.task_cid)
            if current.status != "completed":
                refreshed.append({"task_cid": task.task_cid, "status": "pending_completion"})
                continue
            if not self._context_refresh_stopped():
                refreshed.append({"task_cid": task.task_cid, "status": "pending_native_stop",
                                  "completion_authority": False})
                continue
            refresh_phase = "rebuilder_selection"
            try:
                historical = next((item for item in result.get("context_observation", {}).get("source384", [])
                    if item["task_cid"] == task.task_cid), None)
                source384_policy = policy.get("source384")
                if historical is not None and source384_policy is None:
                    refreshed.append(historical)
                    continue
                cached_refresh = task.task_cid in self._published_context
                if cached_refresh:
                    refresh_phase = "cache_reload"
                    cached = self._published_context[task.task_cid]
                    context = load_published_task_context(server=self.server, admission=self.admission,
                        artifact=cached["refresh_artifact"], expected_sha256=cached["refresh_sha256"],
                        **({"deadline_monotonic": min(self._replay_work_scope.deadline_monotonic,
                                                      self._replay_lifetime_deadline),
                            "source384_timeout_seconds": source384_policy["preparation_timeout_seconds"]}
                           if source384_policy is not None else {}))
                else:
                    count = self._context_refresh_attempts.get(task.task_cid, 0)
                    if count >= policy["max_attempts_per_task"]:
                        refreshed.append({"task_cid": task.task_cid, "status": "retry_budget_exhausted"})
                        continue
                    self._context_refresh_attempts[task.task_cid] = count + 1
                    output = self.repository / policy["output_root"] / (str(count + 1) + "-" + _digest(task.task_cid))
                    rebuilder = None
                    if self.manifest.get("published_retrieval_policy") is not None:
                        binding = self.manifest["published_retrieval_policy"]
                        if binding.get("policy") == "local-safetensors-symbols@1":
                            from ..runtime.published_learned_retrieval import published_learned_retrieval_rebuilder
                            rebuilder = published_learned_retrieval_rebuilder(repository=self.repository,
                                bundle=self.context_bundle, binding=binding)
                        else:
                            from ..runtime.published_retrieval import published_retrieval_rebuilder
                            rebuilder = published_retrieval_rebuilder(repository=self.repository,
                                bundle=self.context_bundle, binding=binding)
                    refresh_phase = "successor_refresh"
                    context = refresh_published_task_context(server=self.server, admission=self.admission,
                        predecessor_bundle=self.context_bundle, task_cid=task.task_cid, output=output,
                        retrieval_rebuilder=rebuilder,
                        **({"source384_output": Path(source384_policy["output_root"]) / (str(count + 1) + "-" + _digest(task.task_cid)),
                            "source384_timeout_seconds": source384_policy["preparation_timeout_seconds"],
                            "deadline_monotonic": min(self._replay_work_scope.deadline_monotonic,
                                                      self._replay_lifetime_deadline)}
                           if source384_policy is not None else {}))
                    self._published_context[task.task_cid] = context
                refreshed.append({"task_cid": task.task_cid, "status": "refreshed",
                    "refresh_artifact": context["refresh_artifact"], "refresh_sha256": context["refresh_sha256"],
                    "context_bundle": context["context_bundle"], "semantic_root_cid": context["semantic_root_cid"],
                    "world_snapshot_cid": context["world_snapshot_cid"],
                    "retrieval_status": context["retrieval"]["status"],
                    "source_scope": context["source_scope"],
                    "task_revision": context["task_revision"], "completion_authority": False})
                embedding_receipt = (context["retrieval"].get("policy_lineage") or {}).get("embedding_receipt")
                if embedding_receipt is None:
                    embedding_receipt = context["retrieval"].get("embedding_receipt")
                if embedding_receipt is not None:
                    refreshed[-1]["embedding_receipt"] = json.loads(json.dumps(embedding_receipt))
                if "goal_progress" in context:
                    refreshed[-1]["goal_progress"] = json.loads(json.dumps(context["goal_progress"]))
                if "source384_refresh" in context:
                    refreshed[-1]["source384_refresh"] = json.loads(json.dumps(context["source384_refresh"]))
                    refreshed[-1]["source384_refresh_reused"] = cached_refresh
                    refreshed[-1]["requires_independent_manifest"] = True
            except Exception as error:
                # Retain task success and shutdown availability. A failed or
                # stale derivative is unusable; it never authorizes dispatch.
                refreshed.append({"task_cid": task.task_cid, "status": "unavailable",
                    "error_type": type(error).__name__, "error_phase": refresh_phase,
                    "completion_authority": False})
        result["published_context"] = refreshed
        result["published_context_observation_seconds"] = time.monotonic() - refresh_started
        self._record("published-context", {"results": refreshed,
            "observation_seconds": result["published_context_observation_seconds"],
            "completion_authority": False})
        return result

    def close(self):
        self._require_no_live_launched_children()
        inventory_scope = getattr(self, "inventory_execution_scope", None)
        if self.finite_execution_scope is not None:
            self.finite_execution_scope.require_close(self)
        if inventory_scope is not None:
            inventory_scope.require_close(self)
        from ..runtime.local_completion_bridge import OwnerLocalCompletionService, _OwnerCompletionBindingCustody
        pending = getattr(self, "_pending_completion_binding", None)
        if pending is not None:
            if type(pending) is not _OwnerCompletionBindingCustody:
                raise RuntimeError("completion rollback requires its exact native binding custody")
            self._record("completion-binding-rollback-close", pending.close(self))
        completion = getattr(self, "completion_service", None)
        if type(completion) is OwnerLocalCompletionService:
            self._record("completion-service-close", completion.close(self))
        broker = getattr(self, "proof_query_dispatch_broker", None)
        if broker is not None:
            self._record("proof-query-dispatch-broker-close", broker.close_after_native_stop())
        if inventory_scope is not None:
            if self.process.snapshot(self.profile).members:
                raise RuntimeError("stop the exact supervisor tree before closing its coordinator")
            inventory_scope._close_constructor_run_coordinator(self)
        else:
            super().close()
        self._bootstrap_stop.set()
        self._listener.close()
        self._bootstrap_thread.join(timeout=2)
