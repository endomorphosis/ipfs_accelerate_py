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
        value["resource_bounds"]["tasks_max"] = 513
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

    def _unit(self, *, quota="100000 100000", memory="16777216", pid=None, runtime_limit="40min",
              tasks="64", bounds=None, membership="/test-owned.service"):
        pid = os.getpid() if pid is None else pid
        stdout = (f"MainPID={pid}\nInvocationID={'a'*32}\nControlGroup=/test-owned.service\n"
                  f"ActiveState=active\nKillMode=control-group\nRuntimeMaxUSec={runtime_limit}\n"
                  "ActiveEnterTimestampMonotonic=1000000\n")
        files = {"/proc/self/cgroup": "0::" + membership + "\n",
                 "/sys/fs/cgroup/test-owned.service/memory.max": memory,
                 "/sys/fs/cgroup/test-owned.service/pids.max": tasks,
                 "/sys/fs/cgroup/test-owned.service/cpu.max": quota}
        def read(path, *args, **kwargs):
            return files[str(path)]
        with patch.object(Path, "read_text", autospec=True, side_effect=read):
            return runtime._unit_snapshot("test-owned.service", self.settings()["resource_bounds"] if bounds is None else bounds,
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
        value.config = {"boards": [value.board], "state_dir": str(self.root)}
        value.job_path.write_text(json.dumps({"board_id": "board", "repair_unit_token": value.token,
                                              "status": "running", "attempts": 1}))
        value.fleet = SimpleNamespace(hold_paths=Mock(return_value=["operatorhold"]),
                                     observation_only=Mock(return_value=False), _recovery_denial=Mock(return_value=None))
        with patch.dict(os.environ, {"IPFS_TASKBOARD_REPAIR_BOARD_ID": "board",
                                    "IPFS_TASKBOARD_REPAIR_JOB_TOKEN": value.token}):
            with self.assertRaisesRegex(ValueError, "hold or recovery policy"):
                value._job()
            self.assertEqual(value._job(execution=False)["attempts"], 1)

    def test_explicit_parent_envelope_keeps_native_owners_and_time_finite(self):
        value = self.settings()
        previous_owners = (value["validation_owner"], value["adoption_owner"])
        value["resource_bounds"].update(memory_max_bytes=32 * 1024**3,
            tasks_max=512, runtime_max_seconds=1500)
        parsed = runtime.parse_runtime_policy({"candidate_runtime": value,
            "timeout_seconds": 1500}, self.policy())
        self.assertEqual(parsed.bounds, value["resource_bounds"])
        self.assertEqual(parsed.validation_owner.argv, tuple(previous_owners[0]["argv"]))
        self.assertEqual(parsed.adoption_owner.argv, tuple(previous_owners[1]["argv"]))
        self.assertEqual(parsed.bounds["cpu_quota_percent"], 100)
        self.assertEqual(parsed.bounds["runtime_max_seconds"], 1500)

    def test_parent_memory_above_32_gib_refuses_before_any_owner(self):
        value = self.settings()
        value["resource_bounds"]["memory_max_bytes"] = 32 * 1024**3 + 1
        with self.assertRaisesRegex(ValueError, "memory_max_bytes"):
            runtime.parse_runtime_policy({"candidate_runtime": value}, self.policy())

    def test_parent_cpu_and_time_bounds_are_not_expanded(self):
        for field, overbound in (("cpu_quota_percent", 101), ("runtime_max_seconds", 2401)):
            with self.subTest(field=field):
                value = self.settings()
                value["resource_bounds"][field] = overbound
                with self.assertRaisesRegex(ValueError, field):
                    runtime.parse_runtime_policy({"candidate_runtime": value}, self.policy())

    def test_explicit_parent_observation_requires_same_actual_job_cgroup(self):
        bounds = dict(self.settings()["resource_bounds"], memory_max_bytes=32 * 1024**3,
                      tasks_max=512, runtime_max_seconds=1500)
        value = self._unit(memory=str(32 * 1024**3), tasks="512", bounds=bounds,
                           runtime_limit="1500s")
        self.assertEqual(value["memory_max_bytes"], 32 * 1024**3)
        self.assertEqual(value["tasks_max"], 512)
        self.assertEqual(value["control_group"], "/test-owned.service")
        with self.assertRaisesRegex(ValueError, "outside the original job cgroup"):
            self._unit(memory=str(32 * 1024**3), tasks="512", bounds=bounds,
                       runtime_limit="1500s", membership="/unrelated-fleet.service")


if __name__ == "__main__":
    unittest.main()


# Operational ledger-busy controls: all command/epoch/Source/native statements
# below are explicitly modeled DATA. Only the configured verifier boundary is
# mocked; real owner-command state transitions, files, tickets and queue logic run.
import pytest as _operational_pytest
from contextlib import nullcontext as _operational_nullcontext


def _operational_put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=value if type(value)is bytes else runtime._wire(value)
    path.write_bytes(raw);path.chmod(0o400);return runtime._pin(path)


def _operational_runtime_case(tmp_path):
    root=tmp_path/'operational';root.mkdir();metadata=root/'candidate-metadata';metadata.mkdir()
    invocation=metadata/'original-validate-invocation';invocation.mkdir()
    candidate_root=root/'candidate';candidate_root.mkdir()
    owner=_operational_put(root/'owner.py',b'# Modeled configured owner DATA, never executed.\n')
    command=runtime.OwnerCommand(('/usr/bin/python3',owner['path'],'validate','{freeze_path}','{receipt_path}','{invocation_root}'),
        ('/usr/bin/python3',owner['path'],'verify-validation','{freeze_path}','{receipt_path}','{invocation_root}'),(owner,),1)
    v=object.__new__(runtime.CandidateRuntime);v.counter=0;v.directory=root/'bootstrap';v.directory.mkdir()
    v.token='a'*32;v.attempt_number=1;v.board={'id':'legal-ir'};v.job_path=root/'job.json'
    v.config={'repair_worker':{'candidate_runtime':{'modeled_source_pinned_policy':'DATA'}},'state_dir':str(root)}
    v.config_ref=_operational_put(root/'config.json',v.config)
    v.policy=SimpleNamespace(repository_root=root,state_root=root)
    v.session=SimpleNamespace(metadata_root=metadata,candidate_root=candidate_root,stage='candidate_ready',binding={'candidate':'fictional'},
        freeze={'freeze_sha256':'b'*64,'source_commitment':'c'*64})
    v.runtime_policy=SimpleNamespace(validation_owner=command,validation_ledger_busy_retry=None)
    v.authority=Mock();v._remaining_seconds=Mock(return_value=1000)
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as fleet
    v.fleet=SimpleNamespace(lock=fleet.lock,write_json=fleet.write_json,
        _repair_budget_spent=fleet._repair_budget_spent,_repair_limits=lambda config:{'legal-ir':3})
    _operational_put(v.job_path,dict(status='validation_pending',board_id='legal-ir',attempts=1,repair_unit_token=v.token,
        candidate_runtime={'candidate_metadata':str(metadata),'attempt_token':v.token,'attempt_number':1}))
    v._job=Mock(side_effect=lambda **kw:runtime._json(runtime._read(v.job_path))) # authorization boundary only
    pins={name:_operational_put(root/(name+'.json'),{'fictional':name})for name in('policy','plan','request','parent_source_closeout')}
    material=dict(schema='agent-supervisor/closed-operational-ledger-busy@1',reason='pregrant_ledger_busy',
        binding=v.session.binding,freeze_sha256='b'*64,source_commitment='c'*64,invocation_root=str(invocation),**pins,
        source_closed=True,native_closed=True,storage_status='not_entered',record_created=False,native_grant_created=False,
        productive_child_created=False,semantic_negative=False,retry_authorized=False,storage_release_authorized=False)
    proof=_operational_put(invocation/'operational-ledger-busy.json',material)
    values={'{freeze_path}':str(metadata/'candidate-ready.json'),'{receipt_path}':str(invocation/'receipt.json'),'{invocation_root}':str(invocation)}
    intent={'schema':runtime.SCHEMA,'stage':'validate','argv':[values.get(x,x)for x in command.argv],
        'verify_argv':[values.get(x,x)for x in command.verify_argv],'owner_source_descriptors':[owner],
        'runtime_policy_sha256':runtime.candidate.digest(v.config['repair_worker']['candidate_runtime']),
        'attempt_token':v.token,'attempt_number':1,'freeze_sha256':'b'*64,'source_commitment':'c'*64}
    _operational_put(invocation/'intent.json',intent);_operational_put(invocation/'terminal.json',{'command_returncode':1,'verifier_returncode':None})
    verified=dict(schema='agent-supervisor/verified-operational-ledger-busy@1',operational=proof,disposition=material,
        read_only=True,retry_authorized=False,storage_release_authorized=False)
    v.command_runner=Mock(return_value=SimpleNamespace(returncode=0,stdout=runtime._wire(verified).decode(),stderr=''))
    return SimpleNamespace(value=v,root=root,invocation=invocation,material=material,verified=verified,proof=proof)


def test_known_pregrant_ledger_busy_is_terminal_without_replaying_validation():
    value=object.__new__(runtime.CandidateRuntime)
    value.session=SimpleNamespace(stage='candidate_ready',freeze={'freeze_sha256':'a'*64})
    value._stage=Mock(side_effect=lambda stage,**details:dict(stage=stage,**details))
    value._owner_command=Mock(return_value={'schema':'agent-supervisor/closed-operational-ledger-busy@1'})
    value._ledger_busy_decision=Mock(return_value={'reason':'ledger_busy_profile_not_authorized','retry_authorized':False})
    value._run_provider=Mock()
    with patch.object(runtime.candidate,'record_validation')as positive_acceptance:
        observed=value.advance()
    assert observed['stage']=='validation_needs_new_authorization'
    assert observed['reason']=='ledger_busy_profile_not_authorized'
    positive_acceptance.assert_not_called();value._run_provider.assert_not_called()
    value._owner_command.assert_called_once_with('validate')


def test_persisted_operational_owner_command_and_real_advance_close_without_semantic_acceptance(tmp_path):
    c=_operational_runtime_case(tmp_path);v=c.value
    before=(c.invocation/'terminal.json').read_bytes()
    with patch.object(runtime.candidate,'record_validation')as positive:
        result=v.advance()
        assert result['stage']=='validation_needs_new_authorization'
        assert result['reason']=='ledger_busy_profile_not_authorized'
        assert runtime._json(runtime._read(v.job_path))['status']==result['stage']
        positive.assert_not_called()
    assert v.command_runner.call_count==1
    assert v.command_runner.call_args.args[0][2]=='verify-operational-validation'
    assert (c.invocation/'terminal.json').read_bytes()==before
    # Reopening an already saved proof performs no second verifier process.
    assert v._owner_command('validate')['semantic_negative']is False
    assert v.command_runner.call_count==1


@_operational_pytest.mark.parametrize('mode',['low_window','interrupted_observer','authority_expired'])
def test_operational_marker_that_cannot_be_verified_terminates_without_replay(tmp_path,mode):
    c=_operational_runtime_case(tmp_path);v=c.value
    if mode=='low_window':v._remaining_seconds=Mock(return_value=0)
    elif mode=='interrupted_observer':_operational_put(c.invocation/'operational-observer-intent.json',{'interrupted':'DATA'})
    else:v.authority.require_current.side_effect=ValueError('modeled original epoch expired')
    v._closed_negative_validation=Mock(side_effect=AssertionError('operational marker entered semantic-negative path'))
    result=v.advance()
    assert result['stage']=='validation_needs_new_authorization'
    assert result['operational_disposition']['verified_operational']is False
    v.command_runner.assert_not_called();v._closed_negative_validation.assert_not_called()


@_operational_pytest.mark.parametrize('stamp',[True,-1,float('inf')])
def test_saved_operational_observation_timestamp_cannot_shorten_cooldown(tmp_path,stamp):
    c=_operational_runtime_case(tmp_path);v=c.value;v._owner_command('validate')
    path=c.invocation/'operational-observer.json';body=runtime._json(runtime._read(path));body['observed_at']=stamp
    path.chmod(0o600);path.write_bytes((__import__('json').dumps(body,sort_keys=True)+'\n').encode());path.chmod(0o400)
    with _operational_pytest.raises(ValueError):
        v._owner_command('validate')
    assert v.command_runner.call_count==1


def _operational_decision_scope(tmp_path):
    c=_operational_runtime_case(tmp_path);v=c.value;v._owner_command('validate')
    material,ref,pin,observation=v.known_operational_validation
    observation=dict(observation,observed_at=900.)
    v.known_operational_validation=(material,ref,pin,observation)
    profile={'schema':'agent-supervisor/validation-ledger-busy-retry@1','max_additional_invocations':2,
        'cooldown_seconds':60,'authorization_epoch_sha256':'d'*64,'expires_at':2000.}
    v.runtime_policy.validation_ledger_busy_retry=profile
    epoch=SimpleNamespace(commitment='d'*64,value={'not_before_observed_at':100.,'expires_at':2000.,'max_attempts':3})
    return c,epoch


def test_operational_cooldown_fixed_expiry_and_remaining_global_slots_are_not_reset(tmp_path):
    c,epoch=_operational_decision_scope(tmp_path);v=c.value
    with patch('ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization.parse_authorization_epoch',return_value=epoch):
        first=v._ledger_busy_decision(next_index=1,now=950.)
        assert first['reason']=='ledger_busy_cooldown'and first['eligible_after']==960.
        assert v._ledger_busy_decision(next_index=1,now=960.)['fresh_slot_eligible']is True
        assert v._ledger_busy_decision(next_index=1,now=2000.)['reason']=='ledger_busy_original_authorization_expired'
        assert v._ledger_busy_decision(next_index=3,now=1000.)['reason']=='ledger_busy_fixed_validation_slots_exhausted'
        v._remaining_seconds=Mock(return_value=0)
        assert v._ledger_busy_decision(next_index=1,now=1000.)['reason']=='ledger_busy_original_remaining_window_insufficient'
    assert v.runtime_policy.validation_ledger_busy_retry['expires_at']==2000.


def test_fresh_operational_slot_ticket_precedes_any_invocation_and_cannot_be_recharged(tmp_path):
    c,epoch=_operational_decision_scope(tmp_path);v=c.value;v._remaining_seconds=Mock(return_value=1000)
    # Only the external clock/epoch is fictional. Real O_EXCL ticket/mirror/queue code runs.
    v._ledger_busy_decision=Mock(return_value={'fresh_slot_eligible':True,'epoch_sha256':'d'*64,'expires_at':time_future()})
    invocation=v.session.metadata_root/'original-validate-ledger-busy-01'
    context=v._reserve_ledger_busy_generation(1,invocation)
    assert not invocation.exists()
    assert runtime._pin(v.session.metadata_root/'ledger-busy-slot-01.json')==context['spent_slot_ref']
    job=v._job();assert job['attempts']==1 and v.fleet._repair_budget_spent(job)==2
    before=runtime._read(v.job_path)
    with _operational_pytest.raises(ValueError,match='already spent'):
        v._reserve_ledger_busy_generation(1,invocation)
    assert runtime._read(v.job_path)==before
    assert v.command_runner.call_count==1 # only the earlier modeled read-only observer


def time_future():
    import time
    return time.time()+1000


def test_epoch_expiry_during_queue_wait_spends_no_ticket(tmp_path):
    c,epoch=_operational_decision_scope(tmp_path);v=c.value
    v._ledger_busy_decision=Mock(side_effect=[{'fresh_slot_eligible':True,'epoch_sha256':'d'*64,'expires_at':time_future()},
        {'reason':'ledger_busy_original_authorization_expired','retry_authorized':False}])
    with _operational_pytest.raises(ValueError,match='changed while waiting'):
        v._reserve_ledger_busy_generation(1,v.session.metadata_root/'original-validate-ledger-busy-01')
    assert not (v.session.metadata_root/'ledger-busy-slot-01.json').exists()


def test_preCLI_SourceLeaseTimeout_has_no_ledger_operational_disposition(tmp_path):
    c=_operational_runtime_case(tmp_path);v=c.value
    path=c.invocation/'operational-ledger-busy.json';path.chmod(0o600);path.unlink()
    assert v._closed_operational_validation(v.runtime_policy.validation_owner,c.invocation,{'command_returncode':2,'verifier_returncode':None})is None
    v.command_runner.assert_not_called()



def test_spent_final_slot_delayed_publication_cannot_start_after_fixed_expiry(tmp_path):
    c,epoch=_operational_decision_scope(tmp_path);v=c.value
    expires=time_future()
    v._ledger_busy_decision=Mock(return_value={'fresh_slot_eligible':True,'epoch_sha256':'d'*64,'expires_at':expires})
    fresh=v.session.metadata_root/'original-validate-ledger-busy-01'
    context=v._reserve_ledger_busy_generation(1,fresh)
    v.ledger_busy_selected_context=context;v._ledger_busy_invocation=Mock(return_value=fresh)
    epoch=SimpleNamespace(commitment='d'*64,value={'not_before_observed_at':100.,'expires_at':expires,'max_attempts':2})
    with patch('ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization.parse_authorization_epoch',return_value=epoch),patch.object(runtime.time,'time',return_value=expires+1):
        assert v._owner_command('validate')is None
    assert v.validation_operational_wait['reason']=='ledger_busy_original_remaining_window_insufficient'
    assert (fresh/'intent.json').exists()and not(fresh/'terminal.json').exists()
    assert v.fleet._repair_budget_spent(v._job())==2 and v.command_runner.call_count==1
    assert (v.session.metadata_root/'ledger-busy-slot-01.json').exists()
