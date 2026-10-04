"""Qualification guards around explicit owner/reader/subprocess adapters.

Prompt controls edit only a temporary authored calc fixture under simulated
worker UID. No native owners, scheduler pool, fitting or worker job executes.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import qualify_codebase_signed_successor as harness
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as public


class ContainerResourceAuthorityControls(unittest.TestCase):
    def setUp(self):
        self.namespace = {name: os.readlink("/proc/self/ns/" + name)
                          for name in ("pid", "mnt", "net")}
        self.boundary = {"schema": "supervisor-container-worker-boundary@1",
                         "container_id": "authored-container", "namespaces": self.namespace}
        self.authority = {
            "schema": "successor-dispatch-container-resource-authority@1",
            "state_path": "/results/container-resource-admission.json",
            "persisted_config": {"proof_safety_enabled": True, "lane_reservations": {},
                "total_cpu_slots": 12, "total_memory_mb": 8192,
                "total_child_process_slots": 12, "total_unified_memory_mb": 8192,
                "total_gpu_memory_mb": None},
            "auto_renew_leases": True, "lease_ttl_seconds": 120.0,
            "parent_host_envelope": {"host_scheduler_state_path": "/authored-host/admission.json",
                "host_configuration_pin": {"bytes": 1, "sha256": "0" * 64},
                "container_id": "authored-container", "cpu_limit": 12,
                "memory_limit_bytes": 8 * 1024**3, "pids_limit": 512,
                "host_reservation": {"released": False, "cancelled": False,
                    "requires_gpu": False, "cpu_slots": 12, "memory_mb": 8192,
                    "child_process_slots": 12}},
            "namespace": self.namespace,
            "scope": "local PID accounting inside held host envelope and actual CPU RAM PID cgroup",
            "independent_whole_host_pool": False, "shares_host_pid_state": False,
            "proof_authority": False,
            "initial_resources": {"state_path": "/results/container-resource-admission.json",
                                  "active_lease_count": 0, "waiting_request_count": 0},
        }

    def receive(self, value):
        calls = []
        raw = json.dumps(value, sort_keys=True).encode()
        boundary_raw = json.dumps(self.boundary, sort_keys=True).encode()
        scheduler = SimpleNamespace(config=SimpleNamespace(lease_ttl_seconds=120.0,
                                                          auto_renew_leases=True))

        def read(path, *unused):
            calls.append(("read", str(path)))
            if str(path) == "/results/container-resource-authority.json":
                return raw
            if str(path) == "/opt/ipfs-supervisor/container-boundary.json":
                return boundary_raw
            self.fail("authority guard attempted an unrelated/host read")

        def attach(state, configuration):
            calls.append(("attach", state))
            self.assertEqual(state, "/results/container-resource-admission.json")
            self.assertEqual(configuration, value["persisted_config"])
            return scheduler

        with ExitStack() as stack:
            stack.enter_context(patch.object(harness.full_fixture, "read_absolute", read))
            stack.enter_context(patch.object(candidate_execution, "_root_file",
                lambda unused: hashlib.sha256(boundary_raw).hexdigest()))
            stack.enter_context(patch.object(harness.full_fixture, "shared_scheduler", attach))
            stack.enter_context(patch.object(harness.full_fixture, "assert_clean",
                lambda unused: {"global_active_lease_count": 0, "global_waiting_request_count": 0}))
            result = harness._container_scheduler(Path("/results/container-resource-authority.json"))
        self.assertEqual(result, (scheduler, value, raw))
        self.assertEqual([call for call in calls if call[0] == "attach"],
                         [("attach", "/results/container-resource-admission.json")])
        return calls

    def test_exact_local_authority_never_attaches_to_host_pid_state(self):
        calls = self.receive(self.authority)
        self.assertFalse(any("authored-host" in path for _, path in calls))

    def test_rejects_host_state_and_foreign_container_namespace(self):
        for field, value in (("state_path", "/authored-host/admission.json"),
                             ("namespace", {**self.namespace, "pid": "pid:[foreign]"})):
            with self.subTest(field=field):
                changed = deepcopy(self.authority); changed[field] = value
                with self.assertRaises((AssertionError, ValueError)):
                    self.receive(changed)

    def test_rejects_capacity_type_aliases_and_unbounded_or_invented_gpu_pool(self):
        for field, value in (("total_cpu_slots", True), ("total_cpu_slots", 13),
                ("total_memory_mb", 8193), ("total_child_process_slots", 13),
                ("total_unified_memory_mb", 8191), ("total_gpu_memory_mb", 0)):
            with self.subTest(field=field, value=value):
                changed = deepcopy(self.authority); changed["persisted_config"][field] = value
                with self.assertRaises((AssertionError, ValueError)):
                    self.receive(changed)

    def test_rejects_released_parent_lease_and_root_boundary_container_disagreement(self):
        for mutate in (lambda v: v["parent_host_envelope"]["host_reservation"].update(released=True),
                       lambda v: v["parent_host_envelope"].update(container_id="another-container"),
                       lambda v: v["parent_host_envelope"]["host_reservation"].update(cpu_slots=True)):
            changed = deepcopy(self.authority); mutate(changed)
            with self.assertRaises((AssertionError, ValueError)):
                self.receive(changed)

    def test_rejects_nonfinite_ttl_and_earlier_local_processes(self):
        for mutate in (lambda v: v.update(lease_ttl_seconds=float("nan")),
                       lambda v: v["initial_resources"].update(active_lease_count=1),
                       lambda v: v["initial_resources"].update(waiting_request_count=False)):
            changed = deepcopy(self.authority); mutate(changed)
            with self.assertRaises((AssertionError, ValueError)):
                self.receive(changed)

    def test_rejects_authority_aliases_and_unknown_fields(self):
        for mutate in (lambda v: v.update(proof_authority=0),
                       lambda v: v.update(shares_host_pid_state=0),
                       lambda v: v.update(independent_whole_host_pool=0),
                       lambda v: v.update(trusted=True)):
            changed = deepcopy(self.authority); mutate(changed)
            with self.assertRaises((AssertionError, ValueError)):
                self.receive(changed)


class PublishedPublicCheckControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repository = Path(self.directory.name).resolve()
        self.source = self.repository / "check_offset.py"
        self.source.write_bytes(b"assert 2 + 2 == 4\n")

    def test_retains_actual_argv_exit_utf8_and_exact_source_hashes(self):
        stdout = "public check ✓\n".encode()
        checked = subprocess.CompletedProcess([], 0, stdout, b"")
        with patch.object(harness.subprocess, "run", return_value=checked) as run:
            value = harness._run_published_check(self.repository, "check_offset.py")
        run.assert_called_once_with([harness.sys.executable, "-B", "check_offset.py"],
            cwd=self.repository, check=False, capture_output=True, timeout=10)
        self.assertEqual(value["returncode"], 0)
        self.assertEqual(value["stdout"].encode(), stdout)
        self.assertEqual(value["stdout_bytes"], len(stdout))
        self.assertEqual(value["stdout_sha256"], hashlib.sha256(stdout).hexdigest())
        self.assertEqual(value["source_before"], value["source_after"])
        self.assertEqual(value["source_before"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_failed_or_oversized_public_check_cannot_claim_published_pass(self):
        for checked in (subprocess.CompletedProcess([], 1, b"", b"refused"),
                        subprocess.CompletedProcess([], 0, b"x" * 65_537, b"")):
            with patch.object(harness.subprocess, "run", return_value=checked):
                with self.assertRaises(AssertionError):
                    harness._run_published_check(self.repository, "check_offset.py")

    def test_late_source_mutation_refuses_and_unknown_check_never_launches(self):
        def changed(*unused, **options):
            self.source.write_bytes(b"assert False\n")
            return subprocess.CompletedProcess([], 0, b"", b"")
        with patch.object(harness.subprocess, "run", side_effect=changed):
            with self.assertRaises(AssertionError):
                harness._run_published_check(self.repository, "check_offset.py")
        with patch.object(harness.subprocess, "run") as run:
            with self.assertRaises(AssertionError):
                harness._run_published_check(self.repository, "other.py")
            run.assert_not_called()


class NativeOwnerObservationControls(unittest.TestCase):
    """Regression from the real second Docker run's finite native owner rows."""
    def setUp(self):
        path = Path('/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/signed-successor-worker-qualification-20261003-02/native/owners-before.json')
        self.before = harness.full_fixture.inert.parse(path.read_bytes())

    def test_real_native_finite_float_rows_are_valid_unchanged_observations(self):
        harness._require_unchanged_owners(deepcopy(self.before), self.before)

    def test_changed_owner_generation_and_integer_boolean_alias_refuse(self):
        for value in (0.125, True):
            changed = deepcopy(self.before); changed['registry']['meta'][0][5] = value
            with self.assertRaises((AssertionError, ValueError)):
                harness._require_unchanged_owners(changed, self.before)

    def test_nonfinite_native_observation_refuses(self):
        for value in (float('nan'), float('inf')):
            changed = deepcopy(self.before); changed['registry']['meta'][0][5] = value
            with self.assertRaises((AssertionError, ValueError)):
                harness._require_unchanged_owners(changed, self.before)


class AuthoredWorkerPromptControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name).resolve()
        self.source = self.workspace / "calc.py"
        self.before = b"def increment(n: int) -> int:\n    return n + 2\n"
        self.after = b"def increment(n: int) -> int:\n    return (2 + n)\n"
        self.source.write_bytes(self.before)
        self.receipt = {"schema": "supervisor-public-instruction-inclusion@4",
                        "codebase_successor": {"authored": True}}
        self.arguments = {"artifact": self.workspace / "public.json",
                          "expected_sha256": "0" * 64, "task_cid": "authored-task"}
        self.prompt = " \n" + json.dumps({"objective_id": "SUCCESSOR-FORMAT"}) + (
            "\n\nCODEBASE INVENTORY ADVISORY\nverbatim public context ✓\n")

    def run_worker(self, prompt, loader):
        out = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(harness.os, "getuid", return_value=1001))
            stack.enter_context(patch.object(harness.os, "geteuid", return_value=1001))
            stack.enter_context(patch.object(harness.Path, "cwd", return_value=self.workspace))
            stack.enter_context(patch.object(harness.sys, "stdin",
                SimpleNamespace(buffer=io.BytesIO(prompt.encode("utf-8")))))
            stack.enter_context(patch.object(harness.sys, "stdout", out))
            stack.enter_context(patch.object(public, "load_public_instruction", side_effect=loader))
            result = harness.run_authored_worker(**self.arguments)
        return result, out.getvalue()

    def loader(self, calls, *, refusal_at=None, different_at=None):
        def load(**arguments):
            calls.append(arguments)
            self.assertEqual(self.source.read_bytes(), self.before,
                             "authored worker wrote before both public validations")
            if len(calls) == refusal_at:
                raise ValueError("authored public replay refusal")
            receipt = deepcopy(self.receipt)
            if len(calls) == different_at:
                receipt["changed"] = True
            return "CODEBASE INVENTORY ADVISORY", receipt
        return load

    def test_json_prefix_with_verbatim_suffix_passes_unchanged_prompt_twice_before_write(self):
        calls = []
        result, stdout = self.run_worker(self.prompt, self.loader(calls))
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0]["prompt"], self.prompt)
        self.assertEqual(calls[0]["workspace"], self.workspace)
        self.assertEqual(self.source.read_bytes(), self.after)
        self.assertEqual(json.loads(stdout)["after_sha256"], hashlib.sha256(self.after).hexdigest())

        # The actual07 worker wrote the parentheses-only postimage successfully,
        # then the unchanged native proposal gate rejected its identical AST.
        # Run that real gate on both candidates; suppress only metadata-owner
        # projection, as public replay does, without changing proposal policy.
        from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
            CandidateDiffEntry, DiffChangeKind)
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            public_replay_without_metadata)
        from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
            ImplementationProposal, ORDERED_PROPOSAL_GATES, ProposalFindingCode,
            ProposalValidationPolicy, ProposalValidator)

        binding = {"task_id": "SUCCESSOR-FORMAT", "accepted_plan_id": "plan:authored-successor",
            "repository_id": "repository:authored-successor",
            "repository_tree_id": "tree:authored-successor", "objective_id": "SUCCESSOR-FORMAT",
            "baseline_id": "baseline:authored-successor", "context_id": "context:authored-successor"}
        policy = ProposalValidationPolicy(allowed_paths=("calc.py",),
            **{"expected_" + name.replace("accepted_plan_id", "plan_id"): value
               for name, value in binding.items()})
        validator = ProposalValidator(policy)

        def validate(postimage):
            proposal = ImplementationProposal(**binding, declared_paths=("calc.py",),
                candidate_diff=(CandidateDiffEntry(old_path="calc.py", new_path="calc.py",
                    change_kind=DiffChangeKind.MODIFY, before_source=self.before.decode("utf-8"),
                    after_source=postimage.decode("utf-8")),))
            with public_replay_without_metadata():
                return validator.validate(proposal)

        rejected = validate(b"def increment(n: int) -> int:\n    return (n + 2)\n")
        self.assertFalse(rejected.accepted)
        self.assertEqual(tuple(finding.code for finding in rejected.findings),
                         (ProposalFindingCode.NO_SEMANTIC_CHANGE,))
        accepted = validate(self.source.read_bytes())
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.findings, ())
        self.assertEqual(accepted.receipt.gate_trace, ORDERED_PROPOSAL_GATES)
        self.assertEqual(accepted.receipt.changed_paths, ("calc.py",))
        self.assertEqual(accepted.receipt.expensive_checks_started, 0)
        self.assertEqual(accepted.receipt.proved_requirement_ids, ())
        self.assertIs(accepted.proof_authoritative, False)
        self.assertIs(accepted.completion_authoritative, False)

        checks = {
            "check_type.py": b"from calc import increment\nfor n in (-2, -1, 0, 1, 2):\n    assert type(increment(n)) is int\n",
            "check_offset.py": b"from calc import increment\nfor n in (-2, -1, 0, 1, 2):\n    assert increment(n) == n + 2\n",
        }
        for name, check_source in checks.items():
            (self.workspace / name).write_bytes(check_source)
            checked = harness._run_published_check(self.workspace, name)
            self.assertEqual(checked["returncode"], 0)
            self.assertEqual(checked["source_before"], checked["source_after"])
            self.assertEqual(checked["source_before"]["sha256"], hashlib.sha256(check_source).hexdigest())
        self.assertEqual(self.source.read_bytes(), self.after)

    def test_wrong_objective_refuses_before_public_validation_or_write(self):
        calls = []
        with self.assertRaises(AssertionError):
            self.run_worker(self.prompt.replace("SUCCESSOR-FORMAT", "OTHER"), self.loader(calls))
        self.assertEqual(calls, [])
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_duplicate_objective_refuses_before_public_validation_or_write(self):
        calls = []
        prompt = '{"objective_id":"OTHER","objective_id":"SUCCESSOR-FORMAT"}\nVERBATIM'
        with self.assertRaises(ValueError):
            self.run_worker(prompt, self.loader(calls))
        self.assertEqual(calls, [])
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_nested_duplicate_capsule_field_uses_same_strict_router_decoder(self):
        calls = []
        prompt = '{"objective_id":"SUCCESSOR-FORMAT","metadata":{"x":1,"x":2}}\nVERBATIM'
        with self.assertRaises(ValueError):
            self.run_worker(prompt, self.loader(calls))
        self.assertEqual(calls, [])
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_public_validation_refusal_at_either_read_leaves_source_untouched(self):
        for refusal_at in (1, 2):
            with self.subTest(refusal_at=refusal_at):
                calls = []
                with self.assertRaises(ValueError):
                    self.run_worker(self.prompt, self.loader(calls, refusal_at=refusal_at))
                self.assertEqual(len(calls), refusal_at)
                self.assertEqual(self.source.read_bytes(), self.before)

    def test_changed_closing_receipt_refuses_before_source_write(self):
        calls = []
        with self.assertRaises(AssertionError):
            self.run_worker(self.prompt, self.loader(calls, different_at=2))
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_actual_closed05_native_capsule_with_appended_block_preserves_full_prompt(self):
        path = Path('/home/barberb/lift_coding/artifacts/codebase_ir_terminal_bench/'
            'signed-successor-worker-qualification-20261003-05/native/private/launch/state/run/'
            'admitted_database_portal_attempts/9dca8f5b8786e9ea3fa59090/implementation-logs/'
            'successor-format-base-context-capsule.json')
        original = path.read_bytes()
        self.assertEqual(json.loads(original)["objective_id"], "SUCCESSOR-FORMAT")
        prompt = original.decode("utf-8") + "\nCODEBASE INVENTORY ADVISORY\nretained capsule + authored suffix"
        calls = []
        result, _ = self.run_worker(prompt, self.loader(calls))
        self.assertEqual(result, 0)
        self.assertEqual([row["prompt"] for row in calls], [prompt, prompt])
        self.assertEqual(path.read_bytes(), original)


class FinalReportClosureControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name).resolve()
        self.report = {"qualified": False, "error_type": "OriginalRefusal", "error": "primary refused"}

    def finish(self, *, primary=True, close_error=None, resource_error=None):
        with patch.object(harness.native, "_close", side_effect=close_error) as close, \
                patch.object(harness.full_fixture, "assert_clean", side_effect=resource_error,
                    return_value={"active_lease_count": 0, "waiting_request_count": 0}) as clean:
            harness._finalize_qualification_report(self.output, self.report,
                registry=object(), connection=object(), scheduler=object(),
                started=time.monotonic(), primary_error_active=primary)
        close.assert_called_once()
        clean.assert_called_once()
        return json.loads((self.output / "result.json").read_bytes())

    def test_pending_inner_lease_retains_failed_report_without_masking_primary_refusal(self):
        value = self.finish(resource_error=AssertionError("native lease remains"))
        self.assertFalse(value["qualified"])
        self.assertEqual((value["error_type"], value["error"]), ("OriginalRefusal", "primary refused"))
        self.assertIs(value["final_resource_cleanup_verified"], False)
        self.assertNotIn("final_resources", value)
        self.assertEqual(value["cleanup_errors"][0]["stage"], "verify_named_resource_cleanup")

    def test_cleanup_only_failure_raises_after_persisting_unqualified_result(self):
        self.report = {"qualified": True}
        with self.assertRaisesRegex(AssertionError, "native lease remains"):
            self.finish(primary=False, resource_error=AssertionError("native lease remains"))
        value = json.loads((self.output / "result.json").read_bytes())
        self.assertFalse(value["qualified"])
        self.assertIs(value["final_resource_cleanup_verified"], False)

    def test_native_close_failure_still_checks_resources_and_retains_both_failures(self):
        value = self.finish(close_error=ValueError("owner close refused"),
                            resource_error=AssertionError("native lease remains"))
        self.assertEqual([row["stage"] for row in value["cleanup_errors"]],
                         ["close_native_owners", "verify_named_resource_cleanup"])
        self.assertFalse(value["qualified"])

    def test_proven_clean_counts_are_retained_without_replacing_primary_error(self):
        value = self.finish()
        self.assertEqual(value["final_resources"], {"active_lease_count": 0, "waiting_request_count": 0})
        self.assertIs(value["final_resource_cleanup_verified"], True)
        self.assertEqual(value["error"], "primary refused")


if __name__ == "__main__":
    unittest.main(verbosity=2)
