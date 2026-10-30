"""Owning adapter controls. No providers, resource grants or services are run."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_runtime as runtime


class CandidateRuntimeAPITests(unittest.TestCase):
    def setUp(self):
        self.private = tempfile.TemporaryDirectory()
        self.addCleanup(self.private.cleanup)
        self.root = Path(self.private.name)
        self.code = self.root / "owner.py"
        self.code.write_text("# exact selected owner fixture\n")
        self.ref = runtime._pin(self.code)

    def owner(self):
        return {"argv": ["/usr/bin/python3", str(self.code), "{freeze_path}", "{receipt_path}"],
                "verify_argv": ["/usr/bin/python3", str(self.code), "--verify", "{receipt_path}"],
                "source_descriptors": [self.ref], "timeout_seconds": 20}

    def policy(self):
        return SimpleNamespace(source_paths=("owner.py",), repository_root=self.root,
                               max_source_bytes=1024, max_candidate_bytes=1024, max_files=10)

    def settings(self):
        adoption = self.owner()
        adoption["argv"].append("{validation_path}")
        return {"schema": runtime.SCHEMA, "source_descriptors": [self.ref],
                "validation_owner": self.owner(), "adoption_owner": adoption,
                "resource_bounds": {"memory_max_bytes": 16 * 1024**3, "cpu_quota_percent": 100,
                                    "tasks_max": 64, "runtime_max_seconds": 2400},
                "checkout_timeout_seconds": 1}

    def test_duplicate_metadata_key_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runtime._json(b'{"passed":false,"passed":true}')

    def test_boolean_descriptor_byte_count_is_not_integer(self):
        with self.assertRaisesRegex(ValueError, "typed"):
            runtime._checked({**self.ref, "bytes": True})

    def test_source_descriptor_mutation_refused(self):
        self.code.write_text("# modified after original selection\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            runtime._checked(self.ref)

    def test_symlink_source_refused_and_private_alias_removed(self):
        alias = self.root / "alias.py"
        alias.symlink_to(self.code)
        try:
            with self.assertRaisesRegex(ValueError, "aliased"):
                runtime._read(alias)
        finally:
            alias.unlink()

    def test_hardlink_source_refused_and_original_nlink_restored(self):
        alias = self.root / "hardlink.py"
        os.link(self.code, alias)
        try:
            with self.assertRaisesRegex(ValueError, "ordinary"):
                runtime._read(self.code)
        finally:
            alias.unlink()
        self.assertEqual(self.code.stat().st_nlink, 1)

    def test_owner_requires_separate_source_bound_verifier(self):
        value = self.owner()
        value["verify_argv"] = ["/bin/true", "{receipt_path}"]
        with self.assertRaisesRegex(ValueError, "pinned source"):
            runtime._owner(value, "validation")

    def test_owner_rejects_embedded_substitutions(self):
        value = self.owner()
        value["argv"][1] = "--path={candidate_root}"
        with self.assertRaisesRegex(ValueError, "substitutions"):
            runtime._owner(value, "validation")

    def test_owner_receipt_argument_is_mandatory(self):
        value = self.owner()
        value["argv"].remove("{receipt_path}")
        with self.assertRaisesRegex(ValueError, "receipt argument"):
            runtime._owner(value, "validation")

    def test_missing_adoption_owner_refuses_before_provider(self):
        value = self.settings()
        value["adoption_owner"] = None
        with self.assertRaisesRegex(ValueError, "adoption owner unavailable"):
            runtime.parse_runtime_policy({"candidate_runtime": value}, self.policy())

    def test_adoption_owner_must_consume_actual_validation(self):
        with self.assertRaisesRegex(ValueError, "consume the accepted validation"):
            runtime._owner(self.owner(), "adoption")

    def test_exact_source_manifest_and_finite_owners_parse(self):
        parsed = runtime.parse_runtime_policy({"candidate_runtime": self.settings()}, self.policy())
        self.assertEqual(parsed.sources, (self.ref,))
        self.assertEqual(parsed.bounds["tasks_max"], 64)

    def test_unselected_manifest_source_refused(self):
        value = self.settings()
        value["source_descriptors"] = []
        with self.assertRaisesRegex(ValueError, "source manifest"):
            runtime.parse_runtime_policy({"candidate_runtime": value}, self.policy())

    def test_runtime_bounds_cannot_remove_original_pid_cap(self):
        value = self.settings()
        value["resource_bounds"]["tasks_max"] = 65
        with self.assertRaisesRegex(ValueError, "tasks_max"):
            runtime.parse_runtime_policy({"candidate_runtime": value}, self.policy())

    def test_too_short_cgroup_deadline_refused_before_charging(self):
        value = self.settings()
        value["resource_bounds"]["runtime_max_seconds"] = 40
        with self.assertRaisesRegex(ValueError, "cannot accommodate owner"):
            runtime.parse_runtime_policy({"candidate_runtime": value}, self.policy())

    def test_too_short_outer_worker_deadline_refused_before_charging(self):
        with self.assertRaisesRegex(ValueError, "cannot accommodate owner"):
            runtime.parse_runtime_policy({"candidate_runtime": self.settings(), "timeout_seconds": 40}, self.policy())

    def _unit(self, *, quota="100000 100000", memory="16777216", pid=None, runtime_limit="40min"):
        pid = os.getpid() if pid is None else pid
        stdout = (f"MainPID={pid}\nInvocationID={'a'*32}\nControlGroup=/test-owned.service\n"
                  f"ActiveState=active\nKillMode=control-group\nRuntimeMaxUSec={runtime_limit}\n"
                  "ActiveEnterTimestampMonotonic=1000000\n")
        files = {"/proc/self/cgroup": "0::/test-owned.service\n",
                 "/sys/fs/cgroup/test-owned.service/memory.max": memory,
                 "/sys/fs/cgroup/test-owned.service/pids.max": "64",
                 "/sys/fs/cgroup/test-owned.service/cpu.max": quota}
        def read(path, *args, **kwargs):
            return files[str(path)]
        with patch.object(Path, "read_text", autospec=True, side_effect=read):
            return runtime._unit_snapshot("test-owned.service", self.settings()["resource_bounds"],
                                          Mock(return_value={"returncode": 0, "stdout": stdout}))

    def test_unit_requires_actual_own_main_pid(self):
        with self.assertRaisesRegex(ValueError, "actual original"):
            self._unit(pid=os.getpid() + 1)

    def test_unit_unlimited_cpu_refused(self):
        with self.assertRaisesRegex(ValueError, "cgroup limits"):
            self._unit(quota="max 100000")

    def test_unit_unlimited_memory_refused(self):
        with self.assertRaisesRegex(ValueError, "cgroup limits"):
            self._unit(memory="max")

    def test_unit_deadline_cannot_exceed_original_limit(self):
        with self.assertRaisesRegex(ValueError, "deadline"):
            self._unit(runtime_limit="41min")

    def test_actual_bounded_unit_observation(self):
        value = self._unit()
        self.assertEqual(value["tasks_max"], 64)
        self.assertEqual(value["runtime_max_seconds"], 2400)

    def test_unknown_worker_flock_is_not_ownership(self):
        lock = self.root / "repair-worker.lock"
        lock.write_text("")
        with patch.object(Path, "read_text", return_value=""):
            with self.assertRaisesRegex(ValueError, "ownership is unknown"):
                runtime._flock_owner(lock)

    def test_passed_boolean_dict_has_no_owner_authority(self):
        value = object.__new__(runtime.CandidateRuntime)
        value.accepted = {}
        with self.assertRaisesRegex(ValueError, "no actual configured"):
            value._owner_guard("validate", {}, {"passed": True})

    def boundary_fixture(self):
        value = object.__new__(runtime.CandidateRuntime)
        value.board = {"id": "board"}
        value.attempt_number = 2
        good = {"schema": "ipfs_accelerate_py/agent-supervisor/provider-filesystem-boundary@1",
                "task_id": "infrastructure-repair:board", "attempt": 2, "stage": "repair",
                "applied": True, "provider_descendants_fenced": True,
                "shared_git_metadata_writable": False}
        return value, good

    def test_boundary_requires_actual_ack_and_same_attempt(self):
        value, good = self.boundary_fixture()
        value._boundary(good)

    def test_boundary_rejects_missing_applied_ack(self):
        value, good = self.boundary_fixture()
        with self.assertRaises(ValueError):
            value._boundary({**good, "applied": False})

    def test_boundary_rejects_boolean_attempt(self):
        value, good = self.boundary_fixture()
        with self.assertRaises(ValueError):
            value._boundary({**good, "attempt": True})

    def test_boundary_rejects_other_attempt(self):
        value, good = self.boundary_fixture()
        with self.assertRaises(ValueError):
            value._boundary({**good, "attempt": 3})

    def test_boundary_rejects_writable_shared_git(self):
        value, good = self.boundary_fixture()
        with self.assertRaises(ValueError):
            value._boundary({**good, "shared_git_metadata_writable": True})

    def test_unknown_provider_closure_refuses_freeze(self):
        value = object.__new__(runtime.CandidateRuntime)
        value._job = Mock()
        value.unit = "unit"
        value.unit_owner = {"cgroup_root": "/unused"}
        value.runtime_policy = SimpleNamespace(bounds={})
        value.runner = Mock()
        value.provider_closed = False
        value.storage_policy = {}
        value.job_path = self.root / "job.json"
        value.policy = SimpleNamespace(state_root=self.root)
        value.fleet = SimpleNamespace(_storage_preflight=Mock(return_value=None))
        with patch.object(runtime, "_unit_snapshot", return_value=value.unit_owner):
            with self.assertRaisesRegex(ValueError, "closure is unknown"):
                value._resource_guard("freeze", {})

    def test_unknown_prior_owner_invocation_is_not_replayed(self):
        invocation = self.root / "original-validate-invocation"
        invocation.mkdir()
        value = object.__new__(runtime.CandidateRuntime)
        value.runtime_policy = SimpleNamespace(validation_owner=SimpleNamespace(timeout_seconds=1))
        value.authority = Mock()
        value.session = SimpleNamespace(metadata_root=self.root, binding={})
        value.command_runner = Mock()
        self.assertIsNone(value._owner_command("validate"))
        value.command_runner.assert_not_called()

    def test_false_terminal_codes_cannot_masquerade_as_zero(self):
        invocation = self.root / "original-validate-invocation"
        invocation.mkdir()
        (invocation / "terminal.json").write_text(json.dumps({"command_returncode": False,
                                                             "verifier_returncode": False}))
        value = object.__new__(runtime.CandidateRuntime)
        value.runtime_policy = SimpleNamespace(validation_owner=SimpleNamespace(timeout_seconds=1))
        value.authority = Mock()
        value.session = SimpleNamespace(metadata_root=self.root, binding={})
        value.command_runner = Mock()
        self.assertIsNone(value._owner_command("validate"))
        value.command_runner.assert_not_called()

    def test_insufficient_owner_deadline_leaves_no_invocation(self):
        value = object.__new__(runtime.CandidateRuntime)
        value.runtime_policy = SimpleNamespace(validation_owner=SimpleNamespace(timeout_seconds=60))
        value.authority = Mock()
        value.session = SimpleNamespace(metadata_root=self.root, binding={})
        value._remaining_seconds = Mock(return_value=45)
        value.command_runner = Mock()
        self.assertIsNone(value._owner_command("validate"))
        self.assertFalse((self.root / "original-validate-invocation").exists())
        value.command_runner.assert_not_called()

    def test_pending_validation_never_starts_provider(self):
        value = object.__new__(runtime.CandidateRuntime)
        value.session = SimpleNamespace(stage="candidate_ready", freeze={"freeze_sha256": "a"*64})
        value._stage = Mock(side_effect=lambda stage, **kw: {"stage": stage, **kw})
        value._owner_command = Mock(return_value=None)
        value._run_provider = Mock()
        self.assertEqual(value.advance()["stage"], "validation_pending")
        value._run_provider.assert_not_called()

    def test_provider_storage_is_bounded_before_freeze(self):
        value = object.__new__(runtime.CandidateRuntime)
        value.policy = SimpleNamespace(max_files=10, max_candidate_bytes=4)
        with self.assertRaisesRegex(ValueError, "byte bound"):
            value._provider_usage((self.root,))

    def _prior_validation(self):
        value = object.__new__(runtime.CandidateRuntime)
        state = self.root / "state"
        token = "a" * 32
        metadata = state / "repair-candidates" / "board" / token
        invocation = metadata / "original-validate-invocation"
        invocation.mkdir(parents=True)
        binding = {"metadata_root": str(metadata), "candidate_root": str(state / "worktrees" / "ptr_lane_0" / token)}
        frozen = {"freeze_sha256": "b" * 64, "source_commitment": "c" * 64}
        runtime._new(metadata / "candidate-ready.json", frozen)
        original_ref = runtime._new(invocation / "original-closeout.json", {"original_owner": "fixture"})
        receipt = {"schema": runtime.candidate.VALIDATION_SCHEMA, **frozen, "passed": True,
                   "source_closed": True, "native_closed": True, "processes_closed": True,
                   "storage_closed": True, "cleanup_errors": [],
                   "original_receipt_reference": original_ref["path"],
                   "original_receipt_descriptor": original_ref}
        receipt_ref = runtime._new(invocation / "receipt.json", receipt)
        command = runtime._owner(self.owner(), "validation")
        policy_value = self.settings()
        values = {"{freeze_path}": str(metadata / "candidate-ready.json"),
                  "{receipt_path}": str(invocation / "receipt.json")}
        argv = [values.get(v, v) for v in command.argv]
        verify = [values.get(v, v) for v in command.verify_argv]
        intent = {"schema": runtime.SCHEMA, "stage": "validate", "argv": argv,
                  "verify_argv": verify, "owner_source_descriptors": list(command.source_descriptors),
                  "runtime_policy_sha256": runtime.candidate.digest(policy_value),
                  "attempt_token": token, "attempt_number": 1, **frozen}
        runtime._new(invocation / "intent.json", intent)
        runtime._new(invocation / "terminal.json", {"command_returncode": 0, "verifier_returncode": 0})
        value.policy = SimpleNamespace(state_root=state, repository_root=self.root)
        value.runtime_policy = SimpleNamespace(validation_owner=command)
        value.config = {"repair_worker": {"candidate_runtime": policy_value}}
        value.board, value.token, value.attempt_number = {"id": "board"}, token, 1
        value.authority = SimpleNamespace(require_current=Mock())
        value.command_runner = Mock(return_value=SimpleNamespace(returncode=0))
        value._remaining_seconds = Mock(return_value=500)
        value.directory = self.root / "runtime"
        value.directory.mkdir()
        value.resuming, value.counter, value.accepted = True, 0, {}
        return value, binding, receipt, receipt_ref, argv, verify

    def test_resume_accepts_genuine_prior_validation_without_owner_or_provider_replay(self):
        value, binding, receipt, ref, owner_argv, verify_argv = self._prior_validation()
        observed = value._owner_guard("validate", binding, receipt)
        self.assertEqual(observed.kind, "validate")
        self.assertEqual(observed.reference, ref["path"])
        self.assertEqual(value.accepted["validate"], (receipt, ref))
        self.assertEqual(value.command_runner.call_count, 1)
        self.assertEqual(value.command_runner.call_args.args[0], verify_argv)
        self.assertNotEqual(value.command_runner.call_args.args[0], owner_argv)
        self.assertEqual(runtime._json(runtime._checked(ref)), receipt)

    def test_resume_rejects_stale_prior_validation_source_before_verifier(self):
        value, binding, receipt, *_ = self._prior_validation()
        self.code.write_text("# source changed after original validation\n")
        with self.assertRaisesRegex(ValueError, "descriptor changed"):
            value._owner_guard("validate", binding, receipt)
        value.command_runner.assert_not_called()

    def test_resume_rejects_mismatched_retained_validation_receipt(self):
        value, binding, receipt, *_ = self._prior_validation()
        with self.assertRaisesRegex(ValueError, "validation differs"):
            value._owner_guard("validate", binding, {**receipt, "source_commitment": "d" * 64})
        value.command_runner.assert_not_called()

    def test_resume_prior_verifier_failure_has_no_authority(self):
        value, binding, receipt, *_ = self._prior_validation()
        value.command_runner.return_value.returncode = 1
        with self.assertRaisesRegex(ValueError, "verifier refused"):
            value._owner_guard("validate", binding, receipt)
        self.assertEqual(value.accepted, {})

    def test_postlease_bootstrap_cannot_publish_before_release(self):
        value = object.__new__(runtime.CandidateRuntime)
        value.checkout_lease_closed = False
        with self.assertRaisesRegex(ValueError, "checkout closeout"):
            value.publish_closed_bootstrap({"stage": "adopted_pending_probe"})

    def test_postlease_bootstrap_is_frozen_and_referenced_by_same_attempt(self):
        value = object.__new__(runtime.CandidateRuntime)
        value.checkout_lease_closed = True
        value.provider_closed = True
        value.board, value.token, value.attempt_number = {"id": "board"}, "a" * 32, 1
        value.unit_owner = {"invocation_id": "b" * 32}
        value.directory = self.root
        value.job_path = self.root / "job.json"
        value.session = SimpleNamespace(binding={"attempt_id": value.token}, freeze={"freeze_sha256": "c"*64})
        job = {"status": "adopted_pending_probe", "candidate_runtime": {"attempt_token": value.token,
               "attempt_number": 1}}
        value._job = Mock(return_value=job)
        @contextmanager
        def lock(_):
            yield True
        writer = Mock()
        value.fleet = SimpleNamespace(lock=lock, write_json=writer)
        ref = value.publish_closed_bootstrap({"stage": "adopted_pending_probe"})
        body = runtime._json(runtime._checked(ref))
        self.assertTrue(body["original_checkout_lease_closed"])
        self.assertTrue(body["provider_known_closed"])
        self.assertEqual(body["invocation_id"], "b"*32)
        self.assertEqual(writer.call_args.args[1]["candidate_runtime"]["bootstrap_result"], ref)
        self.assertFalse(Path(ref["path"]).stat().st_mode & 0o222)

    def test_new_operator_hold_refuses_live_stage_after_outer_preflight(self):
        value = object.__new__(runtime.CandidateRuntime)
        value.job_path = self.root / "job.json"
        value.board, value.token, value.attempt_number = {"id": "board"}, "a"*32, 1
        value.config = {}
        value.job_path.write_text(json.dumps({"board_id": "board", "repair_unit_token": value.token,
                                              "status": "running", "attempts": 1}))
        value.fleet = SimpleNamespace(hold_paths=Mock(return_value=["operatorhold"]),
                                     observation_only=Mock(return_value=False), _recovery_denial=Mock(return_value=None))
        with patch.dict(os.environ, {"IPFS_TASKBOARD_REPAIR_BOARD_ID": "board",
                                    "IPFS_TASKBOARD_REPAIR_JOB_TOKEN": value.token}):
            with self.assertRaisesRegex(ValueError, "hold or recovery policy"):
                value._job()
            self.assertEqual(value._job(execution=False)["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
