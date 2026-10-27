"""Registered host recovery for exactly the worldwide legal-search owner unit.

Systemd owns process containment; native ProgrammaticRecoveryController owns
incident bounds and receipts. This helper never opens or changes task state.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

UNIT = "worldwide-legal-search-supervisor.service"
MODULE = "ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime"


class HostLifecycleError(ValueError):
    pass


def process_identity(pid: int) -> dict[str, Any]:
    if pid <= 0:
        return {"pid": pid, "alive": False, "start_ticks": 0}
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (root / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        groups = [line.split(":", 2)[2] for line in (root / "cgroup").read_text().splitlines()]
        return {"pid": pid, "alive": fields[0] not in {"Z", "X"}, "start_ticks": int(fields[19]), "uid": root.stat().st_uid, "argv": argv, "cgroups": groups}
    except (OSError, ValueError, IndexError):
        return {"pid": pid, "alive": False, "start_ticks": 0}


class RegisteredSystemdLifecycle:
    def __init__(self, *, repository: Path, state_root: Path, run_id: str, config_path: Path | None = None, command=None, clock=time.time, monotonic=time.monotonic, sleep=time.sleep):
        from .worldwide_legal_search_watchdog import read_owned_json

        self.repository, self.state_root = repository.resolve(strict=True), state_root.resolve(strict=True)
        self.config_path = (config_path or self.repository / "config/worldwide_supervisor_host.json").resolve(strict=True)
        config = read_owned_json(self.config_path)
        if config.get("schema") != "worldwide-legal-search.host@1" or Path(config.get("site_root", "")).resolve() != self.repository or Path(config.get("state_root", "")).resolve() != self.state_root:
            raise HostLifecycleError("host registration scope differs from campaign")
        configured_unit = config.get("supervisor_unit", UNIT)
        if configured_unit != UNIT or not run_id:
            raise HostLifecycleError("only the exact registered WWL supervisor unit is permitted")
        if not isinstance(config.get("runtime_argv"), list) or MODULE not in config["runtime_argv"] or str(self.config_path) not in config["runtime_argv"]:
            raise HostLifecycleError("host runtime command does not bind the admitted configuration")
        self.run_id, self.clock, self.monotonic, self.sleep = run_id, clock, monotonic, sleep
        self.command = command or self._command

    @staticmethod
    def _command(argv: list[str], timeout: float) -> str:
        try:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=max(0.1, timeout), check=False)
        except subprocess.TimeoutExpired as error:
            raise HostLifecycleError("registered host lifecycle command exceeded its bound") from error
        if result.returncode:
            raise HostLifecycleError("registered host lifecycle command failed: " + str(result.returncode))
        return result.stdout

    def probe(self) -> dict[str, Any]:
        from .worldwide_legal_search_watchdog import read_owned_json

        raw = self.command(["systemctl", "--user", "show", UNIT, "--property=Id,MainPID,ControlGroup,FragmentPath,ActiveState,SubState,KillMode"], 5)
        values = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        expected_fragment = Path.home() / ".config/systemd/user" / UNIT
        fragment = Path(values.get("FragmentPath") or "/missing-unit")
        if values.get("Id") != UNIT or values.get("KillMode") != "control-group" or fragment != expected_fragment or not fragment.is_file() or fragment.is_symlink() or fragment.stat().st_uid != os.getuid():
            raise HostLifecycleError("WWL unit is not registered with owned cgroup containment")
        content = fragment.read_text()
        if MODULE not in content or str(self.config_path) not in content or str(self.repository) not in content:
            raise HostLifecycleError("WWL unit fragment does not bind this runtime and repository")
        pid = int(values.get("MainPID") or 0)
        group = values.get("ControlGroup") or ""
        if group and (not group.startswith("/") or ".." in Path(group).parts or not group.endswith("/" + UNIT)):
            raise HostLifecycleError("WWL unit control group differs from registered scope")
        process = process_identity(pid)
        alive = bool(process["alive"])
        if alive:
            argv = process.get("argv", [])
            if process.get("uid") != os.getuid() or MODULE not in argv or str(self.config_path) not in argv or not group or not any(value == group or value.startswith(group + "/") for value in process.get("cgroups", [])):
                raise HostLifecycleError("WWL MainPID does not belong to the registered runtime/cgroup")
        snapshot = None
        try:
            snapshot = read_owned_json(self.state_root / "campaign-status.json")
        except (OSError, ValueError):
            pass
        fresh_owner = bool(alive and isinstance(snapshot, dict) and snapshot.get("run_id") == self.run_id and snapshot.get("owner_pid") == pid and type(snapshot.get("observed_at")) in (float, int) and 0 <= self.clock() - snapshot["observed_at"] <= 30)
        return {"unit": UNIT, "registered": True, "main_pid": pid, "start_ticks": process["start_ticks"], "owner_alive": alive,
                "owner_identity_verified": True, "control_group": group, "active_state": values.get("ActiveState", ""),
                "fresh_owner_projection": fresh_owner, "observed_at": self.clock()}

    def _cgroup_empty(self, group: str) -> bool:
        if not group:
            return True
        root = Path("/sys/fs/cgroup") / group.lstrip("/")
        if not root.exists():
            return True
        # Check descendants too: subprocess sessions do not escape a cgroup,
        # and a direct cgroup.procs file does not list subgroup processes.
        try:
            return all(not path.read_text().strip() for path in [root / "cgroup.procs", *root.glob("**/cgroup.procs")])
        except OSError:
            return False

    def recover(self, action: dict[str, Any]) -> dict[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import RescueOperation, prompt_workflow_cid
        from ipfs_accelerate_py.agent_supervisor.rescue.recovery_diagnostics import diagnose_supervisor_incident
        from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import ProgrammaticRecoveryController, ProgrammaticRecoveryPolicy, RecoveryActionObservation
        from .worldwide_legal_search_watchdog import ACTION_ID, canonical, publish_once, read_owned_json

        before = self.probe()
        if not isinstance(action.get("action_id"), str) or ACTION_ID.fullmatch(action["action_id"]) is None or not isinstance(action.get("policy_id"), str) or not action["policy_id"]:
            raise HostLifecycleError("host action must have an exact identity and admitted policy")
        if action.get("run_id") != self.run_id or action.get("kind") not in {"reconcile", "recover"}:
            raise HostLifecycleError("host action run or operation differs from the registered campaign")
        admission_path = self.state_root / "host-recovery/admissions" / (action["action_id"] + ".json")
        try:
            admission = read_owned_json(admission_path)
        except FileNotFoundError:
            admission = {"action": action, "before": before, "run_id": self.run_id, "unit": UNIT, "observed_at_ms": int(self.clock() * 1000)}
            publish_once(admission_path, admission)
        if admission.get("run_id") != self.run_id or admission.get("unit") != UNIT or canonical(admission.get("action")) != canonical(action):
            raise HostLifecycleError("host action identity reused outside its original admission")
        before = admission["before"]
        diagnosis = diagnose_supervisor_incident(
            repository_root=str(self.repository), state_root=str(self.state_root),
            repository_root_cid=prompt_workflow_cid({"repository": str(self.repository)}),
            policy_root=prompt_workflow_cid({"policy_id": action["policy_id"]}),
            run_cid=prompt_workflow_cid({"run_id": self.run_id}),
            process={"lane_id": UNIT, "alive": before["owner_alive"], "failed": not before["owner_alive"], "lane_identity_current": True, "request_action_id": action["action_id"]},
            heartbeat={"lane_id": UNIT, "stale": True, "request_action_id": action["action_id"]},
            health={"healthy": False}, observed_at_ms=admission["observed_at_ms"],
        )
        facts = {"old_owner_fenced": False}

        def restart(context):
            deadline = self.monotonic() + min(35, max(0.1, (context.deadline_at_ms - int(self.clock() * 1000)) / 1000))
            self.probe()
            self.command(["systemctl", "--user", "stop", UNIT], min(10, max(0.1, deadline - self.monotonic())))
            while self.monotonic() < deadline:
                stopped = self.probe()
                old_process = process_identity(before["main_pid"])
                old_alive = old_process["alive"] and old_process["start_ticks"] == before["start_ticks"]
                if stopped["main_pid"] == 0 and not old_alive and self._cgroup_empty(before["control_group"]):
                    facts["old_owner_fenced"] = True
                    break
                self.sleep(0.1)
            else:
                return RecoveryActionObservation(False, reason="registered_old_owner_not_fenced", partial=True)
            self.command(["systemctl", "--user", "start", UNIT], min(10, max(0.1, deadline - self.monotonic())))
            while self.monotonic() < deadline:
                current = self.probe()
                if current["fresh_owner_projection"]:
                    return RecoveryActionObservation(True, context.action.expected_effects, {"healthy": True, "unit": UNIT, "main_pid": current["main_pid"]}, "registered_cgroup_fenced_and_native_owner_ready")
                self.sleep(0.1)
            return RecoveryActionObservation(False, ("old_lane_fenced",), {"healthy": False}, "new_native_owner_readiness_not_yet_verified", partial=True)

        controller = ProgrammaticRecoveryController(self.state_root / "host-recovery", handlers={RescueOperation.RESTART_LANE: restart},
            policy=ProgrammaticRecoveryPolicy(max_attempts_per_action=1, max_total_attempts=1, max_actions=2, deadline_ms=45000),
            clock_ms=lambda: int(self.clock() * 1000))
        result = controller.recover(diagnosis)
        return {"recovered": result.recovered, "receipt": result.terminal_cid, "deduplicated": result.deduplicated,
                "effect_observation": "succeeded" if result.recovered else "unknown", "unit": UNIT,
                "old_owner_fenced": facts["old_owner_fenced"] or bool(result.recovered)}
