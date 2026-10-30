"""Pure contract controls, never actual prerequisite or dispatch evidence.

The admitted test runner reads only explicit existing real source documents
and public helper sources. All transformations/refusal controls run in memory;
no pytest, temporary files, JUnit, state, board or repository writes occur.
"""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


packet = module("authored_packet", Path(__file__).with_name("packet_generator.py"))


def control_prerequisites():
    """Literal unit controls only: these are never read by prepare_real_packet."""
    docker = {"schema": "legal-ir-native-docker-fixture-entry-terminal@4",
        "scope": "synthetic-native-acceptance-regression-only", "actual_preparation_finalized": True,
        "provider_calls_performed": False, "real_LIR_task_accepted": False,
        "outcome": {key: True for key in ("actual_native_validation_finalized", "actual_separate_orchestration_finalized",
            "actual_handle_consumed_once", "fixture_board_published", "actual_container_exit_and_cleanup_verified")}}
    docker["outcome"].update(actual_kernel_backend="source-bound-ordinary-A91-Docker",
        provider_calls_performed=False, real_LIR_task_accepted=False)
    retry = {"schema": "legal-ir-attempt-preservation-publication@1", "pushed": True,
        "controlled_policy_testcases": 51, "failures": 0, "errors": 0, "skipped": 0, "canonical_tasks_accepted": 0,
        "new_runtime_setting_enabled": False, "taskboard_or_taskstate_changed": False,
        "branch": "agent/legal-ir-ordinary-worker-20261016", "accelerator_commit": "a" * 40,
        "production_source_sha256": {path: "b" * 64 for path in packet.RETRY_PATHS}}
    history = {"schema": "legal-ir-root-observed-attempt-history@1", "task_id": "LIR-9001",
        "canonical_task_cid": "unit-control-only-not-an-observation", "consumed_attempts": 2,
        "board_namespace": "unit-control-only", "orchestration_namespace": "unit-control-only",
        "runtime_paths": {}, "source_observation_pin": {},
        "history_reset_performed": False}
    return docker, retry, history


class PacketContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projection = packet.load((packet.OLD_ROOT / "tasks.execution.json").read_bytes())
        cls.config = packet.load((packet.OLD_ROOT / "scheduler_config.json").read_bytes())
        cls.board = (packet.OLD_ROOT / "tasks.todo.md").read_bytes()
        cls.identity = module("actual_public_identity_for_contract_tests", packet.ACCELERATOR /
            "ipfs_accelerate_py/agent_supervisor/task_sources/task_identity.py")
        cls.facts = module("actual_factual_helper_for_contract_tests", packet.ADAPTER / "factual_candidate.py")
        cls.sources, cls.frozen = cls.facts.load_source_bundle(packet.OLD_ROOT)

    def revised(self):
        return packet.transform_projection(self.projection, self.config)

    def task_mapping(self, board):
        block = re.split(r"(?m)^## LIR-9002 ", board.decode())[0]
        heading = next(line for line in block.splitlines() if line.startswith("## LIR-9001 "))
        metadata = {}
        for line in block.splitlines():
            if line.startswith("- ") and ":" in line:
                key, value = line[2:].split(":", 1)
                metadata[key.lower()] = value.strip()
        return {"task_id": "LIR-9001", "title": heading.split(" ", 2)[2],
            "outputs": [x.strip() for x in metadata["outputs"].split(",")],
            "acceptance": metadata["acceptance"], "metadata": metadata}

    def test_exact_original_graph_and_all_future_task_bodies_are_preserved(self):
        revised, _ = self.revised()
        for key in ("goal_hierarchy", "legacy_parent_crosswalk"):
            self.assertEqual(revised[key], self.projection[key])
        self.assertEqual(len(revised["tasks"]), 64)
        self.assertEqual(sum(len(t["depends_on"]) for t in revised["tasks"]), 115)
        self.assertEqual(revised["tasks"][1:], self.projection["tasks"][1:])

    def test_lir9001_changes_only_objective_inputs_and_validation(self):
        revised, _ = self.revised()
        old, new = self.projection["tasks"][0], revised["tasks"][0]
        for key in old:
            if key not in {"objective", "inputs", "validation"}:
                self.assertEqual(old[key], new[key])
        self.assertEqual({k:v for k,v in old["inputs"].items() if k != "references"},
                         {k:v for k,v in new["inputs"].items() if k != "references"})

    def test_cap_setting_and_history_lineage_preserve_input_documents(self):
        baseline = copy.deepcopy((self.projection, self.config))
        _, config = self.revised()
        self.assertEqual(config["max_task_attempts"], 3)
        self.assertIs(config["preserve_task_attempts"], True)
        self.assertEqual(config["board_namespace"], self.config["board_namespace"])
        self.assertEqual(config["runtime_paths"], self.config["runtime_paths"])
        self.assertEqual((self.projection, self.config), baseline)

    def test_board_preserves_all_other_blocks_and_semantic_fields(self):
        revised, _ = self.revised()
        board = packet.transform_board(self.board, revised["tasks"][0])
        old_tail = self.board.split(b"## LIR-9002 ", 1)[1]
        self.assertEqual(board.split(b"## LIR-9002 ", 1)[1], old_tail)
        old, new = self.task_mapping(self.board), self.task_mapping(board)
        for key in old["metadata"]:
            if key not in {"objective", "inputs", "validation"}:
                self.assertEqual(old["metadata"][key], new["metadata"][key])

    def test_actual_public_identity_ignores_revised_instructions(self):
        revised, _ = self.revised()
        board = packet.transform_board(self.board, revised["tasks"][0])
        old = self.identity.canonical_task_identity(self.task_mapping(self.board))
        new = self.identity.canonical_task_identity(self.task_mapping(board))
        self.assertEqual(old.canonical_task_cid, new.canonical_task_cid)
        self.assertEqual(old.semantic_fingerprint, new.semantic_fingerprint)

    def test_actual_public_identity_detects_acceptance_or_allowed_path_change(self):
        original = self.task_mapping(self.board)
        baseline = self.identity.canonical_task_identity(original)
        for mutate in ("acceptance", "allowed paths"):
            changed = copy.deepcopy(original)
            if mutate == "acceptance":
                changed["acceptance"] += " A different semantic acceptance contract."
            else:
                changed["metadata"]["allowed paths"] = "a/new/authority.py"
            self.assertNotEqual(self.identity.canonical_task_identity(changed).canonical_task_cid, baseline.canonical_task_cid)

    def test_canonical_identity_and_dedupe_shortcuts_are_refused(self):
        for key in ("canonical_task_cid", "canonical_task_key", "dedupe_key"):
            for nested in (False, True):
                task = {"metadata": {key: "provided"}} if nested else {key: "provided"}
                with self.assertRaisesRegex(packet.PacketRefusal, "shortcut forbidden"):
                    packet.reject_identity_shortcuts(task)

    def test_graph_population_or_dependencies_cannot_be_reduced(self):
        for kind in ("tasks", "goal_hierarchy", "legacy_parent_crosswalk", "edges"):
            projection = copy.deepcopy(self.projection)
            if kind == "edges":
                next(t for t in projection["tasks"] if t["depends_on"])["depends_on"].pop()
            else:
                projection[kind].pop()
            with self.assertRaises(packet.PacketRefusal):
                packet.transform_projection(projection, self.config)

    def test_future_task_cannot_be_unblocked(self):
        projection = copy.deepcopy(self.projection)
        projection["tasks"][1]["status"] = "todo"
        with self.assertRaisesRegex(packet.PacketRefusal, "frontier already changed"):
            packet.transform_projection(projection, self.config)

    def test_board_duplicate_or_missing_revised_field_is_refused(self):
        revised, _ = self.revised()
        for board in (self.board.replace(b"- Validation:", b"- Validation: duplicate\n- Validation:", 1),
                      self.board.replace(b"- Validation:", b"- Unrecognized validator:", 1)):
            with self.assertRaises(packet.PacketRefusal):
                packet.transform_board(board, revised["tasks"][0])

    def test_completion_command_stays_closed_check_report(self):
        revised, _ = self.revised()
        command = revised["tasks"][0]["validation"][0]
        self.assertIn(" --check-report ", command)
        self.assertNotIn("--check-facts", command)
        self.assertEqual(revised["tasks"][0]["completion"], "manual")

    def test_normative_prerequisite_shape_control_grants_nothing(self):
        self.assertIsNone(packet.assess_prerequisite_documents(*control_prerequisites()))

    def test_failed_or_handles_only_docker_qualification_is_refused(self):
        for key in ("actual_native_validation_finalized", "actual_separate_orchestration_finalized",
                    "fixture_board_published", "actual_container_exit_and_cleanup_verified"):
            docker, retry, history = control_prerequisites()
            docker["outcome"][key] = False
            with self.assertRaisesRegex(packet.PacketRefusal, "actual full engineering Docker"):
                packet.assess_prerequisite_documents(docker, retry, history)

    def test_real_task_claim_cannot_replace_engineering_qualification(self):
        docker, retry, history = control_prerequisites()
        docker["real_LIR_task_accepted"] = True
        with self.assertRaisesRegex(packet.PacketRefusal, "actual full engineering Docker"):
            packet.assess_prerequisite_documents(docker, retry, history)

    def test_missing_or_failed_actual_retry_deployment_is_refused(self):
        for key, value in (("pushed", False), ("failures", 1), ("controlled_policy_testcases", 50),
                           ("production_source_sha256", {}), ("taskboard_or_taskstate_changed", True)):
            docker, retry, history = control_prerequisites()
            retry[key] = value
            with self.assertRaisesRegex(packet.PacketRefusal, "actual qualified retry"):
                packet.assess_prerequisite_documents(docker, retry, history)

    def test_history_reset_or_different_consumed_ordinal_is_refused(self):
        for key, value in (("consumed_attempts", 0), ("consumed_attempts", 3), ("history_reset_performed", True)):
            docker, retry, history = control_prerequisites()
            history[key] = value
            with self.assertRaisesRegex(packet.PacketRefusal, "two-consumed-attempt"):
                packet.assess_prerequisite_documents(docker, retry, history)

    def test_pure_transforms_have_no_file_opens_or_process_launches(self):
        with patch("builtins.open", side_effect=AssertionError("filesystem effect")), \
             patch("os.open", side_effect=AssertionError("filesystem effect")), \
             patch("subprocess.run", side_effect=AssertionError("process effect")):
            revised, _ = self.revised()
            packet.transform_board(self.board, revised["tasks"][0])
            packet.assess_prerequisite_documents(*control_prerequisites())

    def test_gaps_reference_actual_fixture_functions_instead_of_claiming_adoption(self):
        self.assertTrue(any("prepare_isolated_fixture" in x for x in packet.MISSING_REAL_BACKEND_FUNCTIONS))
        self.assertTrue(any("prepare_docker_fixture" in x for x in packet.MISSING_REAL_BACKEND_FUNCTIONS))
        self.assertTrue(any("consume_or_publish_fixture" in x for x in packet.MISSING_REAL_BACKEND_FUNCTIONS))

    def checker_memory(self):
        root = Path("/memory/real-source-packet")
        revised, config = self.revised()
        sources = dict(self.sources)
        sources["execution_projection"] = packet.wire(revised)
        sources["execution_board"] = packet.transform_board(self.board, revised["tasks"][0])
        sources["scheduler_config"] = packet.wire(config)
        files = {root / name: raw for name, raw in self.frozen.items()}
        files.update({root / self.facts.SOURCE_PATHS[role]: raw for role, raw in sources.items()})
        files[root / "host_adapter/factual_candidate.py"] = (packet.ADAPTER / "factual_candidate.py").read_bytes()
        files[root / "check_report.py"] = packet.CHECKER_SOURCE.encode()
        template = self.facts.canonical(self.facts.candidate_template(sources, self.frozen))
        files[root / "unaccepted-candidate-template.json"] = template
        files[root / "packet-manifest.json"] = packet.wire({"schema": "legal-ir-real-report-source-packet@1",
            "dispatch_authority": False, "completion_authority": False,
            "files": [{"path": str(path.relative_to(root)), "bytes": len(raw), "sha256": packet.sha(raw)}
                      for path, raw in sorted(files.items())]})
        report = Path("/memory/unaccepted-worker-report.json")
        files[report] = template
        checker = types.ModuleType("unit_packet_checker")
        checker.__file__ = "/memory/check_report.py"
        exec(compile(packet.CHECKER_SOURCE, checker.__file__, "exec"), checker.__dict__)
        return root, report, files, checker

    def invoke_memory_checker(self, root, files, checker, *argv):
        output = io.StringIO()
        with patch.object(checker, "_read", side_effect=lambda path: files[path]), \
             patch.object(checker, "_load_facts", return_value=self.facts), \
             patch.object(self.facts, "read_source_file", side_effect=lambda path: files[path]), \
             contextlib.redirect_stdout(output):
            code = checker.main(["--root", str(root), *argv])
        return code, packet.load(output.getvalue().encode())

    def test_source_preflight_check_all_succeeds_without_acceptance(self):
        root, _, files, checker = self.checker_memory()
        code, result = self.invoke_memory_checker(root, files, checker, "--check-all")
        self.assertEqual(code, 0)
        self.assertIs(result["valid"], True)
        self.assertEqual(result["scope"], "source-integrity-only")
        self.assertIs(result["completion_authority"], False)
        self.assertIs(result["dispatch_authority"], False)
        self.assertTrue(all(value is False for value in result["authority"].values()))

    def test_source_preflight_corruption_is_refused(self):
        root, _, files, checker = self.checker_memory()
        files[root / "authoring/tasks.json"] += b" "
        code, result = self.invoke_memory_checker(root, files, checker, "--check-all")
        self.assertEqual(code, 1)
        self.assertIs(result["valid"], False)
        self.assertIn("source file pin differs", result["errors"])
        self.assertIs(result["completion_authority"], False)

    def test_completion_stays_closed_while_facts_mode_is_only_advisory(self):
        for mode, expected_code in (("--check-report", 3), ("--check-facts", 0)):
            root, report, files, checker = self.checker_memory()
            code, result = self.invoke_memory_checker(root, files, checker, mode, str(report))
            self.assertEqual(code, expected_code)
            self.assertIs(result["completion_authority"], False)


if __name__ == "__main__":
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PacketContractTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
