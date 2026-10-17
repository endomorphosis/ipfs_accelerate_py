"""In-memory controlled correspondence/refusal tests, never actual Git proof.

Only source reads/imports are performed. No temporary files, Git, SDK imports,
native admission, provider calls, acceptance handles or task mutations occur.
"""
import copy
import importlib.util
import io
import json
import re
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


binding = source_module("controlled_real_candidate_binding", Path(__file__).with_name("real_candidate_binding.py"))
identity = source_module("actual_public_identity_for_binding_controls", binding.IDENTITY)
BASE, TREE, OLD, NEW = (character * 40 for character in "1234")
BOARD_PATH = Path("/unit-controls-only/tasks.todo.md")


def commit(parent=BASE):
    return ("tree " + TREE + "\nparent " + parent + "\nauthor Unit Control <control@localhost> 1 +0000\n"
            "committer Unit Control <control@localhost> 1 +0000\n\nControlled metadata only.\n").encode()


def tree_entry(path=binding.REPORT, mode="100644", kind="blob", oid=NEW):
    return f"{mode} {kind} {oid}\t{path}\0".encode()


def delta(path=binding.REPORT, status="A", mode="100644", old_mode="000000", old="0" * 40):
    return f":{old_mode} {mode} {old} {NEW} {status}\0{path}\0".encode()


def parsed_control_board():
    """Controlled parsed objects from independently pinned actual plan bytes.

    The separate public-parser regression uses the genuine production parser.
    These unit objects are not an actual parser, Git or provider observation.
    """
    raw = binding.read_pinned_source(binding.ORIGINAL_BOARD, binding.ORIGINAL_BOARD_SHA256)
    authored = json.loads(binding.read_pinned_source(binding.ORIGINAL_AUTHORING, binding.ORIGINAL_AUTHORING_SHA256))
    rows = authored["tasks"]
    text = raw.decode("utf-8")
    headings = list(re.finditer(r"(?m)^## (LIR-[0-9]{4}) ([^\n]+)$", text))
    if tuple(row["task_id"] for row in rows) != tuple(match.group(1) for match in headings):
        raise AssertionError("Independently frozen authored/board inventory must agree")
    tasks = []
    for index, (match, row) in enumerate(zip(headings, rows)):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        metadata = {}
        for line in text[match.end():end].splitlines():
            if line.startswith("- ") and ":" in line:
                key, value = line[2:].split(":", 1)
                metadata[key.strip().lower()] = value.strip()
        value = types.SimpleNamespace(task_id=row["task_id"], title=match.group(2),
            status=metadata["status"], completion=metadata["completion"], outputs=list(row["outputs"]),
            acceptance=metadata["acceptance"], metadata=metadata, board_namespace=metadata["board namespace"])
        derived = identity.canonical_task_identity(value, board_namespace=value.board_namespace, source_path=BOARD_PATH)
        value.canonical_task_cid, value.canonical_task_key = derived.canonical_task_cid, derived.canonical_task_key
        tasks.append(value)
    return raw, tasks


class BindingMemoryTests(unittest.TestCase):
    def refuse(self, function, reason, *args):
        with self.assertRaisesRegex(binding.BindingRefusal, "^" + reason + "$", msg="Exact controlled refusal required"):
            function(*args)

    def derive(self, raw, tasks):
        return binding.derive_task_binding(raw, BOARD_PATH, tasks, identity.canonical_task_identity, lambda task: tuple(task.outputs))

    def test_exact_single_parent_commit(self):
        self.assertEqual(binding.parse_commit(commit(), BASE), {"tree": TREE, "parent": BASE})

    def test_root_commit_refused(self):
        self.refuse(binding.parse_commit, "candidate_must_have_exact_single_baseline_parent", commit().replace(f"parent {BASE}\n".encode(), b""), BASE)

    def test_merge_commit_refused(self):
        self.refuse(binding.parse_commit, "candidate_must_have_exact_single_baseline_parent", commit().replace(b"\nauthor", f"\nparent {OLD}\nauthor".encode()), BASE)

    def test_stale_baseline_refused(self):
        self.refuse(binding.parse_commit, "candidate_must_have_exact_single_baseline_parent", commit(), OLD)

    def test_duplicate_tree_header_refused(self):
        self.refuse(binding.parse_commit, "candidate_must_have_exact_single_baseline_parent", f"tree {TREE}\n".encode() + commit(), BASE)

    def test_invalid_commit_tree_oid_refused(self):
        self.refuse(binding.parse_commit, "candidate_must_have_exact_single_baseline_parent", commit().replace(TREE.encode(), b"not-an-oid"), BASE)

    def test_oversized_commit_refused(self):
        self.refuse(binding.parse_commit, "bounded_commit_object_required", commit() + b"x" * 65536, BASE)

    def test_regular_report_tree_entry(self):
        self.assertEqual(binding.parse_tree_entry(tree_entry(), binding.REPORT), NEW)

    def test_symlink_report_refused(self):
        self.refuse(binding.parse_tree_entry, "single_regular_candidate_blob_required", tree_entry(mode="120000"), binding.REPORT)

    def test_submodule_report_refused(self):
        self.refuse(binding.parse_tree_entry, "single_regular_candidate_blob_required", tree_entry(mode="160000", kind="commit"), binding.REPORT)

    def test_executable_json_report_refused(self):
        self.refuse(binding.parse_tree_entry, "single_regular_candidate_blob_required", tree_entry(mode="100755"), binding.REPORT)

    def test_wrong_tree_path_refused(self):
        self.refuse(binding.parse_tree_entry, "single_regular_candidate_blob_required", tree_entry(path="other.json"), binding.REPORT)

    def test_multiple_tree_entries_refused(self):
        self.refuse(binding.parse_tree_entry, "single_regular_candidate_blob_required", tree_entry() + tree_entry(path="other.json"), binding.REPORT)

    def test_report_only_addition_delta(self):
        self.assertEqual(binding.parse_report_diff(delta(), binding.REPORT), NEW)

    def test_report_only_modification_delta(self):
        self.assertEqual(binding.parse_report_diff(delta(status="M", old_mode="100644", old=OLD), binding.REPORT), NEW)

    def test_extra_changed_path_refused(self):
        self.refuse(binding.parse_report_diff, "report_only_candidate_required", delta() + delta(path="source.py"), binding.REPORT)

    def test_report_deletion_refused(self):
        self.refuse(binding.parse_report_diff, "report_only_candidate_required", delta(status="D", mode="000000", old_mode="100644", old=OLD), binding.REPORT)

    def test_report_type_change_refused(self):
        self.refuse(binding.parse_report_diff, "report_only_candidate_required", delta(status="T", old_mode="120000", old=OLD), binding.REPORT)

    def test_rename_delta_refused(self):
        self.refuse(binding.parse_report_diff, "report_only_candidate_required", delta(status="R100"), binding.REPORT)

    def test_unchanged_blob_cannot_be_modification(self):
        self.refuse(binding.parse_report_diff, "report_only_candidate_required", delta(status="M", old_mode="100644", old=NEW), binding.REPORT)

    def test_real_public_identity_defines_revision_and_separate_board_sha(self):
        raw, tasks = parsed_control_board()
        observed = self.derive(raw, tasks)
        self.assertEqual(observed["task_revision"], tasks[0].canonical_task_cid)
        self.assertNotEqual(observed["task_revision"], binding.sha(raw))
        self.assertEqual(observed["expected_board_sha256"], binding.sha(raw))
        self.assertTrue(observed["all_other_63_tasks_remain_blocked"])

    def test_revised_instructions_keep_cid_but_change_exact_board_sha(self):
        raw, tasks = parsed_control_board()
        original = self.derive(raw, tasks)
        changed = self.derive(raw + b"\nSource observation comment.\n", tasks)
        self.assertEqual(changed["task_revision"], original["task_revision"])
        self.assertNotEqual(changed["board_sha256"], original["board_sha256"])

    def test_parser_cid_mismatch_refused(self):
        raw, tasks = parsed_control_board()
        tasks[0].canonical_task_cid = "a-controlled-mismatch"
        self.refuse(self.derive, "actual_parser_identity_correspondence_failed", raw, tasks)

    def test_additional_path_authority_cannot_silently_change_daemon_revision(self):
        raw, tasks = parsed_control_board()
        tasks[0].metadata["allowed paths"] = binding.REPORT
        self.refuse(self.derive, "additional_path_authority_revision_not_supported", raw, tasks)

    def test_changed_acceptance_with_stale_cid_refused(self):
        raw, tasks = parsed_control_board()
        tasks[0].acceptance += " A different semantic requirement."
        self.refuse(self.derive, "actual_parser_identity_correspondence_failed", raw, tasks)

    def test_caller_supplied_cid_shortcut_refused(self):
        raw, tasks = parsed_control_board()
        for shortcut in (b"Canonical_Task_Cid", b"canonical task key", b"Dedupe_Key", b"deduplication key"):
            with self.subTest(alias=shortcut):
                changed = raw.replace(b"- Completion: manual", b"  - " + shortcut + b": supplied\n- Completion: manual", 1)
                self.refuse(self.derive, "caller_supplied_task_identity_shortcut", changed, tasks)

    def test_duplicate_metadata_refused_before_parser_identity(self):
        raw, tasks = parsed_control_board()
        raw = raw.replace(b"- Completion: manual", b"- Completion: manual\n- completion: manual", 1)
        self.refuse(self.derive, "duplicate_legacy_task_metadata", raw, tasks)

    def test_nonmanual_completion_refused(self):
        raw, tasks = parsed_control_board()
        tasks[0].completion = "automatic"
        self.refuse(self.derive, "real_task_frontier_or_completion_changed", raw, tasks)

    def test_future_task_activation_refused(self):
        raw, tasks = parsed_control_board()
        tasks[1].status = "todo"
        self.refuse(self.derive, "real_task_frontier_or_completion_changed", raw, tasks)

    def test_missing_or_reordered_task_refused(self):
        raw, tasks = parsed_control_board()
        tasks[1:3] = reversed(tasks[1:3])
        self.refuse(self.derive, "exact_legacy_task_population_required", raw, tasks)

    def test_inventory_is_independently_frozen_actual_nonsequential_plan(self):
        raw, tasks = parsed_control_board()
        self.assertEqual(tuple(task.task_id for task in tasks), binding.verify_original_inventory())
        self.assertEqual(tasks[4].task_id, "LIR-0101")
        self.assertEqual(tasks[-1].task_id, "LIR-9108")
        self.assertEqual(binding.sha(raw), binding.ORIGINAL_BOARD_SHA256)

    def test_invented_contiguous_task_inventory_refused(self):
        raw, tasks = parsed_control_board()
        for number, task in enumerate(tasks, 9001):
            task.task_id = f"LIR-{number}"
        self.refuse(self.derive, "exact_legacy_task_population_required", raw, tasks)

    def test_missing_actual_task_refused(self):
        raw, tasks = parsed_control_board()
        del tasks[4]
        self.refuse(self.derive, "exact_legacy_task_population_required", raw, tasks)

    def test_duplicate_actual_task_refused(self):
        raw, tasks = parsed_control_board()
        tasks[5] = tasks[4]
        self.refuse(self.derive, "exact_legacy_task_population_required", raw, tasks)

    def test_extra_actual_task_refused(self):
        raw, tasks = parsed_control_board()
        tasks.append(copy.deepcopy(tasks[-1]))
        self.refuse(self.derive, "exact_legacy_task_population_required", raw, tasks)

    def test_pure_correspondence_has_no_io_or_native_side_effects(self):
        raw, tasks = parsed_control_board()
        before = copy.deepcopy(tasks[0].__dict__)
        with patch("builtins.open", side_effect=AssertionError("Unexpected file I/O")), patch.object(binding.subprocess, "Popen", side_effect=AssertionError("Unexpected process")):
            self.derive(raw, tasks)
            binding.parse_commit(commit(), BASE)
            binding.parse_tree_entry(tree_entry(), binding.REPORT)
            binding.parse_report_diff(delta(), binding.REPORT)
        self.assertEqual(tasks[0].__dict__, before)
        self.assertTrue(all(value is False for value in binding.NO_AUTHORITY.values()))


if __name__ == "__main__":
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BindingMemoryTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
