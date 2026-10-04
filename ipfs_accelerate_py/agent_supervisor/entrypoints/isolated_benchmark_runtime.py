"""Bounded local process qualification, separate from production activation.

This adapter launches the real supervisor with an empty, explicitly legacy
queue and implementation disabled.  Its signed local policy authorizes only
that exact process configuration in a freshly created disposable repository.
It does not issue task admission, proof, completion, or production authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ..control.authorization_logic import ControlMutationAuthorizer, ControlMutationPolicy
from ..control.control_contracts import (
    AuthorizationDecision, AuthorizationVerdict, ControlBounds, EffectKind,
    ExpectedEffect, IdempotencyKey, Operation, OperationAuthority, OperationRequest,
    get_operation_catalog,
)
from ..control.control_plane import SupervisorControlService
from ..control.lifecycle_orchestrator import (
    LifecycleOrchestrator, LifecycleProfile, LinuxProcessAdapter, ProcessTreeSnapshot,
)
from ..control.profile_authority import (
    assert_capability_allowed, initialize_local_profile, load_local_profile,
    sign_profile_binding, verify_did_key_signature,
)
from ..merge.database_coordination import open_database_coordinator


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True, timeout=15,
    ).stdout.strip()


class NativeSupervisorHealthAdapter(LinuxProcessAdapter):
    """Read native status and independently verify both process identities.

    The supervisor's run_id identifies its managed child, rather than the
    lifecycle run. Lifecycle identity comes from the native /proc adapter;
    the status file never supplies process ownership or fence authority.
    """

    def launch(self, profile, *, fencing_epoch):
        identity = super().launch(profile, fencing_epoch=fencing_epoch)
        # Preserve only the identity independently verified at our own launch.
        # Hardening can subsequently hide /proc/environ from this same parent.
        if not hasattr(self, "_launch_witnesses"):
            self._launch_witnesses = {}
        self._launch_witnesses[identity.pid] = identity
        return identity

    def snapshot(self, profile):
        observed = super().snapshot(profile)
        members = {member.pid: member for member in observed.members}
        witnesses = getattr(self, "_launch_witnesses", {})
        for pid, identity in tuple(witnesses.items()):
            if not self.identity_alive(identity):
                witnesses.pop(pid, None)
                continue
            if (pid in members or identity.profile_id != profile.profile_id
                    or identity.run_id != profile.run_id):
                continue
            try:
                self._environ(pid)
            except PermissionError:
                try:
                    actual = self._stat(pid)
                    expected = (identity.parent_pid, identity.process_group_id,
                                identity.session_id, identity.start_time_ticks)
                    if actual == expected and self._argv(pid) == identity.argv:
                        members[pid] = identity
                except (OSError, ValueError, ProcessLookupError):
                    pass
            except (OSError, ValueError):
                pass
        return ProcessTreeSnapshot(profile_id=profile.profile_id, run_id=profile.run_id,
                                   members=tuple(members.values()), captured_at_ms=observed.captured_at_ms)

    def child_scope_matches(self, profile, child) -> bool:
        if "--implement" in child.argv or "--todo-path" not in child.argv:
            return False
        return child.argv[child.argv.index("--todo-path") + 1] == str(
            Path(profile.repository_root) / "tasks.todo.md"
        )

    def healthy(self, profile, tree, *, fencing_epoch: int, now_ms: int) -> bool:
        self.last_heartbeat_evidence = None
        if len(tree.roots) != 1 or len(tree.members) < 2:
            return False
        if any(not self.identity_alive(item) for item in tree.members):
            return False
        if any(item.fencing_epoch != fencing_epoch for item in tree.members):
            return False
        path = Path(profile.health_path)
        try:
            if path.is_symlink() or path.stat().st_size > 262_144:
                return False
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict) or payload.get("status") != "running":
                return False
            root = tree.roots[0]
            if type(payload.get("supervisor_pid")) is not int or payload["supervisor_pid"] != root.pid:
                return False
            if payload.get("repo_root") != profile.repository_root:
                return False
            child_pid = payload.get("daemon_pid")
            children = [item for item in tree.members if item.pid == child_pid and item.pid != root.pid]
            if type(child_pid) is not int or len(children) != 1:
                return False
            child = children[0]
            if child.parent_pid != root.pid:
                return False
            if "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon" not in child.argv:
                return False
            # The child command must itself retain the empty source and have
            # no implementation flag; root argv alone is insufficient.
            if not self.child_scope_matches(profile, child):
                return False
            updated = datetime.fromisoformat(str(payload["updated_at"]).replace("Z", "+00:00"))
            if updated.tzinfo is None:
                return False
            updated_ms = int(updated.timestamp() * 1000)
            if not 0 <= now_ms - updated_ms <= profile.health_stale_ms:
                return False
            self.last_heartbeat_evidence = {
                "updated_at_ms": updated_ms, "supervisor_pid": root.pid,
                "daemon_pid": child.pid, "status": "running",
                "authority": False,
            }
            return True
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            return False


def empty_supervisor_argv(repository: Path, state: Path) -> tuple[str, ...]:
    """Compile only options accepted by the actual native supervisor parser."""
    from ..todo_daemon.implementation_supervisor import parse_args

    options = [
        "--todo-path", str(repository / "tasks.todo.md"),
        "--state-dir", str(state / "run"), "--state-prefix", "isolated",
        "--task-prefix", "## ISOLATED-", "--task-source-kind", "legacy-markdown",
        "--authority-mode", "legacy_markdown", "--explicit-legacy-task-source",
        "--no-implement", "--check-interval", "0.25", "--daemon-interval", "0.25",
        "--max-restarts", "1", "--max-task-attempts", "1",
        "--merge-target-branch", "isolated-benchmark",
        "--no-worktree-reconciliation", "--no-retry-budget-guardrail",
        "--no-dependency-guardrail", "--no-reconciliation-guardrail",
        "--no-objective-task-janitor", "--no-objective-goal-refinement",
        "--no-objective-goal-completion-reconcile", "--no-objective-goal-migration",
        "--no-objective-ast-dataset", "--no-objective-todo-vector-index",
    ]
    parsed = parse_args(options)
    if parsed.implement or parsed.task_source_kind != "legacy-markdown":
        raise ValueError("isolated qualification parser changed its authority contract")
    return (sys.executable, "-P", "-m",
            "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor", *options)


class IsolatedBenchmarkRuntime:
    """One fresh repository, one signed configuration, one fenced process tree."""

    @classmethod
    def create(cls, directory: Path, *, timeout_ms: int = 30_000) -> "IsolatedBenchmarkRuntime":
        maximum_timeout = min(get_operation_catalog().by_name[operation.value].bounds.timeout_ms
                              for operation in (Operation.START, Operation.STOP))
        if type(timeout_ms) is not int or not 2_000 <= timeout_ms <= maximum_timeout:
            raise ValueError(f"timeout_ms must fit START/STOP catalog (2000..{maximum_timeout})")
        directory = Path(directory).absolute()
        if directory.resolve() != directory or directory.exists():
            raise ValueError("qualification requires a new directory with no symlink ancestors")
        directory.mkdir(parents=True, mode=0o700)
        repository, state = directory / "repository", directory / "state"
        repository.mkdir(mode=0o700)
        state.mkdir(mode=0o700)
        (state / "run").mkdir(mode=0o700)
        (repository / "tasks.todo.md").write_text("# Isolated supervisor qualification\n\nNo coding tasks admitted.\n")
        _git(repository, "init", "-q", "--initial-branch=isolated-benchmark")
        _git(repository, "add", "tasks.todo.md")
        _git(repository, "-c", "user.name=Isolated Benchmark", "-c", "user.email=isolated@localhost",
             "commit", "-qm", "Initialize empty isolated benchmark queue")
        runtime = cls()
        runtime.directory, runtime.repository, runtime.state = directory, repository, state
        runtime.timeout_ms = timeout_ms
        runtime.baseline = _git(repository, "rev-parse", "HEAD")
        runtime.tree_id = _git(repository, "rev-parse", "HEAD^{tree}")
        from ..core.multiformats_identity import cid_for_dag_json
        runtime.repository_id = cid_for_dag_json({
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-repository-root@1",
            "root": str(repository), "head_tree": runtime.tree_id,
        })
        runtime.run_id = "isolated-" + uuid.uuid4().hex
        runtime.profile_dir, runtime.lifecycle_dir = directory / "profile", directory / "lifecycle"
        runtime.local_profile = initialize_local_profile(
            repository_cid=runtime.repository_id, baseline_commit=runtime.baseline,
            capabilities=("read", "test", "isolated_worktree", "write_worktree"),
            effect_bounds=("read", "test", "isolated_worktree", "write_worktree"),
            profile_dir=runtime.profile_dir, lifecycle_dir=runtime.lifecycle_dir,
        )
        argv = empty_supervisor_argv(repository, state)
        package_root = str(Path(__file__).resolve().parents[3])
        pythonpath = os.pathsep.join(filter(None, (package_root, os.environ.get("PYTHONPATH", ""))))
        environment = (("PYTHONPATH", pythonpath), ("PYTHONUNBUFFERED", "1"))
        runtime.manifest = {
            "schema": "isolated-supervisor-process-qualification@1",
            "repository_id": runtime.repository_id, "repository_root": str(repository),
            "state_root": str(state), "baseline_commit": runtime.baseline,
            "tree_id": runtime.tree_id, "run_id": runtime.run_id,
            "argv": list(argv), "argv_digest": _digest(argv),
            "environment": [list(item) for item in environment],
            "task_source_digest": hashlib.sha256((repository / "tasks.todo.md").read_bytes()).hexdigest(),
            "profile_id": runtime.local_profile.profile_id,
            "lifecycle_generation": runtime.local_profile.lifecycle_generation,
            "operations": ["start", "stop"], "timeout_ms": timeout_ms,
            "provider_dispatch_allowed": False, "production_activation": False,
            "task_admission_authority": False, "completion_authority": False,
        }
        runtime.manifest_id = _digest(runtime.manifest)
        runtime.signature = sign_profile_binding(
            profile_dir=runtime.profile_dir, lifecycle_dir=runtime.lifecycle_dir, payload=runtime.manifest,
        )
        (state / "local-process-grant.json").write_text(json.dumps({
            "manifest": runtime.manifest, "signature": runtime.signature,
        }, sort_keys=True, indent=2) + "\n")
        runtime.profile = LifecycleProfile(
            target_id=runtime.repository_id, run_id=runtime.run_id,
            configuration_root=runtime.manifest_id, repository_root=str(repository),
            state_root=str(state), run_root=str(state / "run"), argv=argv, cwd=str(repository),
            environment=environment,
            health_path=str(state / "run" / "isolated_supervisor_status.json"), health_stale_ms=5_000,
        )
        runtime.coordinator = open_database_coordinator(state / "coordination.duckdb")
        runtime.lease = runtime.coordinator.acquire(
            lease_kind="resource", scope=runtime.run_id,
            owner_session_id=runtime.local_profile.identity_did, lease_ms=max(120_000, timeout_ms * 3),
            resource_kind="supervisor_run", resource_id=runtime.run_id, repository_id=runtime.repository_id,
            idempotency_key=runtime.manifest_id,
            body={"local_profile_id": runtime.local_profile.profile_id, "launch_grant": runtime.manifest_id},
        )
        runtime._children = []

        def logged_popen(*args, **kwargs):
            # Capture actual startup diagnostics without modifying native launch
            # identity, environment, process group, or signal behavior.
            with (state / "supervisor-process.log").open("ab") as output:
                kwargs["stdout"], kwargs["stderr"] = output, subprocess.STDOUT
                child = subprocess.Popen(*args, **kwargs)
            runtime._children.append(child)
            return child

        runtime.process = NativeSupervisorHealthAdapter(popen=logged_popen)
        runtime.orchestrator = LifecycleOrchestrator(
            state_root=state, profiles=(runtime.profile,), process_adapter=runtime.process,
            poll_interval_ms=50, stop_grace_ms=1_000,
        )
        runtime._permits = {}
        runtime._requests = {}
        runtime.service = SupervisorControlService(
            repository_allowlist=(repository,), state_allowlist=(state,),
            handlers={Operation.START: runtime.orchestrator, Operation.STOP: runtime.orchestrator},
            authorization_validator=ControlMutationAuthorizer(runtime._policy),
            identity_validator=runtime._validate_identity, lease_validator=runtime._validate_lease,
        )
        return runtime

    def _verify(self) -> None:
        for path in (self.directory, self.repository, self.state, self.profile_dir, self.lifecycle_dir):
            if path.resolve() != path or path.is_symlink():
                raise ValueError("isolated root changed")
        profile = load_local_profile(repository_cid=self.repository_id,
                                     profile_dir=self.profile_dir, lifecycle_dir=self.lifecycle_dir)
        if profile != self.local_profile:
            raise ValueError("local profile generation changed")
        for capability in ("read", "test", "isolated_worktree", "write_worktree"):
            assert_capability_allowed(profile, capability)
        if profile.baseline_commit != self.baseline:
            raise ValueError("profile baseline changed")
        if _digest(self.manifest) != self.manifest_id or list(self.profile.argv) != self.manifest["argv"]:
            raise ValueError("local launch grant changed")
        if ([list(item) for item in self.profile.environment] != self.manifest["environment"]
                or self.profile.configuration_root != self.manifest_id
                or self.profile.repository_root != str(self.repository)
                or self.profile.state_root != str(self.state)
                or self.profile.run_id != self.run_id):
            raise ValueError("lifecycle profile differs from the signed local grant")
        persisted = json.loads((self.state / "local-process-grant.json").read_text())
        if persisted != {"manifest": self.manifest, "signature": self.signature}:
            raise ValueError("persisted launch grant changed")
        if self.signature["identity"] != profile.identity_did or self.signature["profile_id"] != profile.profile_id:
            raise ValueError("launch signer does not own the installed profile")
        verify_did_key_signature(identity_did=profile.identity_did, payload=self.manifest,
                                 signature=self.signature["signature"])
        if _git(self.repository, "rev-parse", "HEAD") != self.baseline:
            raise ValueError("isolated baseline changed")
        if _git(self.repository, "rev-parse", "HEAD^{tree}") != self.tree_id:
            raise ValueError("isolated tree changed")
        source = self.repository / "tasks.todo.md"
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != self.manifest["task_source_digest"]:
            raise ValueError("empty qualification queue changed")

    def _verify_operation(self, operation: Operation) -> None:
        """Keep strict qualification unless a runtime binds a narrower effect."""
        self._verify()

    def _validate_identity(self, request: OperationRequest) -> bool:
        self._verify_operation(request.operation)
        return all(getattr(request, field) == expected for field, expected in (
            ("repository_root", str(self.repository)), ("state_root", str(self.state)),
            ("repository_id", self.repository_id), ("tree_id", self.tree_id),
            ("objective_id", self.manifest_id), ("objective_revision", self.manifest_id),
        ))

    def _validate_lease(self, request: OperationRequest) -> bool:
        if request.lease_id != self.lease.lease_id or request.fencing_epoch != self.lease.fence_epoch:
            return False
        lease = self.coordinator.protect_write(
            self.lease, expected_fencing_token=self.lease.fencing_token,
            expected_fence_epoch=request.fencing_epoch,
        )
        return (lease.owner_session_id == self.local_profile.identity_did
                and lease.resource_id == self.run_id and lease.repository_id == self.repository_id)

    def _policy(self, request: OperationRequest) -> ControlMutationPolicy:
        self._verify_operation(request.operation)
        if self._requests.get(request.request_id) != request:
            raise ValueError("request is not the exact locally issued operation")
        return ControlMutationPolicy(
            policy_id=self.manifest_id, policy_revision=self.manifest_id,
            permits=tuple(self._permits.values()),
            current_tree_ids={self.repository_id: self.tree_id},
            current_objective_revisions={self.manifest_id: self.manifest_id},
            active_lease_fences={self.lease.lease_id: self.lease.fence_epoch},
        )

    def _operation_timeout_ms(self, operation: Operation) -> int:
        return self.timeout_ms

    def request(self, operation: Operation) -> OperationRequest:
        if operation not in (Operation.START, Operation.STOP):
            raise ValueError("isolated local grant permits only START and STOP")
        self._verify_operation(operation)
        timeout_ms = self._operation_timeout_ms(operation)
        latest = self.orchestrator.store.latest().get(self.profile.target_id)
        revision = latest.receipt.revision if latest and latest.receipt else 0
        binding = dict(
            operation=operation, repository_root=str(self.repository), state_root=str(self.state),
            repository_id=self.repository_id, tree_id=self.tree_id, objective_id=self.manifest_id,
            objective_revision=self.manifest_id, policy_id=self.manifest_id, policy_revision=self.manifest_id,
            caller=self.local_profile.identity_did, lease_id=self.lease.lease_id,
            fencing_epoch=self.lease.fence_epoch,
        )
        effect = ExpectedEffect(effect_id=f"{operation.value}:isolated-process-tree",
                                kind=EffectKind.LIFECYCLE_TRANSITION, resource=self.profile.target_id,
                                paths=("lifecycle-transitions.jsonl", "run"))
        now = _now_ms()
        permit = AuthorizationDecision(
            **binding, verdict=AuthorizationVerdict.PERMIT, granted_authority=OperationAuthority.MUTATION,
            authorized_effect_ids=(effect.effect_id,), grant_ids=(self.manifest_id,),
            evaluated_at_ms=now, expires_at_ms=min(now + timeout_ms + 5_000, self.lease.expires_at_ms),
        )
        self._permits[permit.decision_id] = permit
        request = OperationRequest(
            **binding, bounds=ControlBounds(timeout_ms=timeout_ms), authorization=permit,
            expected_effects=(effect,), parameters={
                "target_id": self.profile.target_id, "run_id": self.run_id,
                "configuration_root": self.manifest_id, "expected_revision": revision,
                "deadline_ms": timeout_ms, "health_window_ms": 500,
                "reason": "isolated signed supervisor process qualification",
            }, idempotency=IdempotencyKey(
                key=f"{self.run_id}:{operation.value}:{revision}", operation=operation,
                caller=self.local_profile.identity_did, repository_id=self.repository_id, objective_id=self.manifest_id,
            ),
        )
        self._requests[request.request_id] = request
        return request

    def start(self):
        result = self.service.execute(self.request(Operation.START))
        self._record("start", result.to_dict())
        return result

    def _verify_observation(self):
        self._verify()

    def observe(self) -> dict[str, Any]:
        context_observation = self._verify_observation()
        tree = self.process.snapshot(self.profile)
        result = {
            "schema": "isolated-supervisor-observation@1", "captured_at_ms": _now_ms(),
            "configuration_root": self.manifest_id, "process_tree": tree.to_dict(),
            "healthy": self.process.healthy(self.profile, tree, fencing_epoch=self.lease.fence_epoch, now_ms=_now_ms()),
            "provider_dispatch_allowed": self.manifest["provider_dispatch_allowed"], "task_admission_authority": False,
            "completion_authority": False, "production_activation": False,
        }
        result["native_heartbeat"] = self.process.last_heartbeat_evidence
        if context_observation is not None:
            result["context_observation"] = context_observation
        self._record("observe", result)
        return result

    def stop(self):
        result = self.service.execute(self.request(Operation.STOP))
        self._record("stop", result.to_dict())
        for child in self._children:
            if child.poll() is not None:
                child.wait(timeout=1)
        return result

    def runtime_factory(self):
        """Install three exact, prebound local handlers, with no fake slots.

        These zero-argument handlers qualify this already signed empty source.
        They deliberately reject prompt-to-run plans: admitting task sources
        into this external supervisor needs the native Quack owner join.
        """
        from ..core.multiformats_identity import cid_for_dag_json
        from .run_registry import RunRegistry
        from .runtime_factory import RuntimeEffectError, StandardSupervisorRuntimeFactory

        def evidence_cid(value):
            # Control contracts intentionally retain str enums. Normalize the
            # actual serialized record, including its observation timestamps.
            return cid_for_dag_json(json.loads(json.dumps(value)), for_identity=False)

        def start():
            result = self.start()
            if not result.succeeded:
                raise RuntimeEffectError(f"native local START failed: {result.error}")
            observed = self.observe()
            if not observed["healthy"]:
                raise RuntimeEffectError("native process lost health after START")
            return {
                "receipt_cid": evidence_cid(result.to_dict()), "effect_applied": True,
                "process_cid": evidence_cid(result.to_dict()["data"]["new_process_identity"]),
                "lease_id": self.lease.lease_id, "fencing_generation": self.lease.fence_epoch,
                "state_revision_cid": evidence_cid(result.to_dict()),
                "health_revision_cid": evidence_cid(observed),
                "configuration_root": self.manifest_id, "event_cursor": result.audit_receipt_id,
                "production_activation": False, "task_admission_authority": False,
            }

        def observe():
            observation = self.observe()
            return {
                "receipt_cid": evidence_cid(observation),
                # The effect here is a persisted observation, not a process
                # mutation or promotion of observational health to authority.
                "effect_applied": True, "observation_recorded": True,
                "process_effect_applied": False, **observation,
            }

        def stop():
            result = self.stop()
            if not result.succeeded or self.process.snapshot(self.profile).members:
                raise RuntimeEffectError(f"native local STOP failed: {result.error}")
            return {"receipt_cid": evidence_cid(result.to_dict()), "effect_applied": True,
                    "old_tree_fenced": result.data["old_tree_fenced"], "production_activation": False}

        return StandardSupervisorRuntimeFactory(
            registry=RunRegistry(self.state / "run-registry"),
            handlers={"start": start, "observe": observe, "stop": stop}, production=False,
        )

    def _record(self, name: str, payload: Mapping[str, Any]) -> None:
        from ..core.multiformats_identity import cid_for_dag_json
        encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        cid = cid_for_dag_json(json.loads(encoded), for_identity=False)
        records = self.state / "receipts"
        records.mkdir(mode=0o700, exist_ok=True)
        path = records / f"{cid}.json"
        if path.exists():
            if path.read_text() != encoded:
                raise ValueError("content-addressed local receipt differs")
        else:
            with path.open("x") as output:
                output.write(encoded)
        (self.state / f"{name}-receipt.json").write_text(encoded)

    def _require_no_live_launched_children(self) -> None:
        if any(child.poll() is None for child in getattr(self, "_children", ())):
            raise RuntimeError("stop every live launched child before releasing runtime custody")

    def close(self) -> None:
        self._require_no_live_launched_children()
        if self.process.snapshot(self.profile).members:
            raise RuntimeError("stop the exact supervisor tree before closing its coordinator")
        self.coordinator.release(self.lease, expected_fencing_token=self.lease.fencing_token,
                                 expected_fence_epoch=self.lease.fence_epoch)
        self.coordinator.close()


def _wait_for_stable_empty_supervisor(runtime, *, initial, native_start_tree, root_cid, deadline, evidence):
    """Wait for child churn to settle without adopting another root or owner.

    The initial observation is the immutable healthy observation returned by
    START. Only auxiliary membership/health may reset the stability window;
    the root and native managed daemon keep their exact birth identities.
    """
    from ..core.multiformats_identity import cid_for_dag_json

    tree = ProcessTreeSnapshot.from_dict(initial["process_tree"])
    heartbeat = initial["native_heartbeat"]
    if initial["healthy"] is not True or len(tree.roots) != 1 or not heartbeat:
        raise ValueError("START has no healthy native process anchor")
    root = tree.roots[0]
    if cid_for_dag_json(root.to_dict(), for_identity=False) != root_cid:
        raise ValueError("START root identity differs from its healthy observation")
    owners = [item for item in tree.members if item.pid == heartbeat["daemon_pid"]]
    if (heartbeat["supervisor_pid"] != root.pid or len(owners) != 1
            or owners[0].pid == root.pid or owners[0].parent_pid != root.pid):
        raise ValueError("START native owner identity is not bound to its root")
    owner = owners[0]
    anchors = {root.identity_id, owner.identity_id}
    admitted = ProcessTreeSnapshot.from_dict(native_start_tree)
    if (not anchors <= {item.identity_id for item in admitted.members}
            or len(admitted.roots) != 1 or admitted.roots[0].identity_id != root.identity_id):
        raise ValueError("root or native owner changed after the native START transition")
    initial_heartbeat = last_heartbeat = heartbeat["updated_at_ms"]
    previous_ids = {item.identity_id for item in tree.members}
    stable_ids, stable_since, stable_heartbeat = None, None, None
    started = time.monotonic()
    evidence.update(schema="isolated-supervisor-stability-observation@1",
        root_identity_id=root.identity_id, daemon_identity_id=owner.identity_id,
        initial_heartbeat_ms=initial_heartbeat, stable_window_seconds=.75,
        poll_interval_seconds=.1, samples=[], passed=False, reason="deadline_exceeded")
    while time.monotonic() < deadline:
        observation = runtime.observe()
        now = time.monotonic()
        current = ProcessTreeSnapshot.from_dict(observation["process_tree"])
        ids = {item.identity_id for item in current.members}
        pulse = observation["native_heartbeat"]
        sample = dict(elapsed_seconds=now-started, healthy=observation["healthy"],
            identity_ids=sorted(ids), added_identity_ids=sorted(ids-previous_ids),
            removed_identity_ids=sorted(previous_ids-ids),
            heartbeat_ms=pulse["updated_at_ms"] if pulse else None)
        evidence["samples"].append(sample)
        previous_ids = ids
        # Never reset the anchor on restart, PID reuse, missing owner, changed
        # launch identity or a new root, even if a later sample is healthy.
        if (not anchors <= ids or len(current.roots) != 1
                or current.roots[0].identity_id != root.identity_id):
            evidence["reason"] = "root_or_owner_identity_changed"
            return False
        if pulse:
            if pulse["supervisor_pid"] != root.pid or pulse["daemon_pid"] != owner.pid:
                evidence["reason"] = "native_heartbeat_owner_changed"
                return False
            if pulse["updated_at_ms"] < last_heartbeat:
                evidence["reason"] = "native_heartbeat_regressed"
                return False
            last_heartbeat = pulse["updated_at_ms"]
        if now >= deadline:
            break  # A slow observation cannot acquire success after the bound.
        if observation["healthy"] is True and pulse:
            if stable_ids != ids:
                stable_ids, stable_since, stable_heartbeat = ids, now, last_heartbeat
            if (now-stable_since >= .75 and last_heartbeat > initial_heartbeat
                    and last_heartbeat > stable_heartbeat):
                if time.monotonic() >= deadline:
                    break  # Identity/CID processing must fit the same bound.
                evidence.update(passed=True, reason="stable_tree_and_fresh_native_heartbeat",
                    observed_process_identity_ids=sorted(ids), stable_seconds=now-stable_since)
                return True
        else:
            stable_ids = stable_since = stable_heartbeat = None
        time.sleep(min(.1, max(0., deadline-time.monotonic())))
    return False


def qualify_empty_supervisor(directory: Path, *, timeout_ms: int = 30_000) -> dict[str, Any]:
    """Run and record a real bounded lifecycle qualification with strict success."""
    runtime = IsolatedBenchmarkRuntime.create(directory, timeout_ms=timeout_ms)
    factory = runtime.runtime_factory()
    report: dict[str, Any] = {
        "schema": "isolated-supervisor-lifecycle-qualification@1",
        "repository_root": str(runtime.repository), "state_root": str(runtime.state),
        "repository_id": runtime.repository_id, "configuration_root": runtime.manifest_id,
        "profile_id": runtime.local_profile.profile_id, "lease_id": runtime.lease.lease_id,
        "handler_manifest": dict(factory.handler_manifest()), "production_activation": False,
        "provider_dispatch_allowed": False, "task_admission_authority": False,
        "completion_authority": False, "passed": False,
    }
    started = observed = stopped = False
    deadline = time.monotonic() + timeout_ms / 1000
    try:
        receipt = factory.invoke("start")
        report["start_receipt_cid"] = receipt.receipt_cid
        started = True
        # Reuse START's immutable health observation; taking a fresh baseline
        # could silently adopt a daemon restart between START and polling.
        from ..core.multiformats_identity import cid_for_dag_json
        initial_cid = receipt.values["health_revision_cid"]
        initial = json.loads((runtime.state / "receipts" / f"{initial_cid}.json").read_text())
        if cid_for_dag_json(initial, for_identity=False) != initial_cid:
            raise ValueError("START health observation differs from its immutable receipt")
        native_start = json.loads((runtime.state / "receipts" / f"{receipt.receipt_cid}.json").read_text())
        if cid_for_dag_json(native_start, for_identity=False) != receipt.receipt_cid:
            raise ValueError("native START transition differs from its immutable receipt")
        report["stability_observation"] = evidence = {}
        observed = _wait_for_stable_empty_supervisor(runtime, initial=initial,
            native_start_tree=native_start["data"]["transition"]["new_tree"],
            root_cid=receipt.values["process_cid"], deadline=deadline, evidence=evidence)
        report["observed_process_identity_ids"] = evidence.get("observed_process_identity_ids",
            evidence["samples"][-1]["identity_ids"] if evidence.get("samples") else [])
        report["healthy_stable_process_tree"] = observed
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            receipt = factory.invoke("stop")
            report["stop_receipt_cid"] = receipt.receipt_cid
            stopped = bool(receipt.values["old_tree_fenced"])
        except Exception as exc:
            report["stop_error"] = {"type": type(exc).__name__, "message": str(exc)}
        absent = not runtime.process.snapshot(runtime.profile).members
        report["process_tree_absent_after_stop"] = absent
        report["passed"] = bool(started and observed and stopped and absent)
        try:
            runtime._record("qualification", report)
        finally:
            try:
                factory.registry.close()
            finally:
                if absent:
                    runtime.close()
    return report


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True,
                        help="New disposable qualification directory; must not exist")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    args = parser.parse_args(argv)
    report = qualify_empty_supervisor(args.directory, timeout_ms=args.timeout_ms)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
