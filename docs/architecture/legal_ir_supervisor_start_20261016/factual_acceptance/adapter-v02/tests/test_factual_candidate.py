"""Source-only predicate checks; no provider or native resource grant is mocked."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import factual_candidate as facts

FIXTURE = Path(__file__).parent / "fixtures" / "bootstrap"


class CorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.sources, self.frozen = facts.load_source_bundle(FIXTURE.absolute())
        self.candidate = facts.candidate_template(self.sources, self.frozen)

    def refused(self, candidate, reason):
        with self.assertRaisesRegex(facts.FactualRefusal, reason):
            facts.check_facts(facts.canonical(candidate), self.sources, self.frozen)

    def test_complete_immutable_fixture_derives_exact_population(self):
        evidence = facts.check_facts(facts.canonical(self.candidate), self.sources, self.frozen)
        self.assertEqual(evidence["population"], {"tasks": 64, "goals": 34, "dependency_edges": 115, "legacy_parents": 24, "frozen_files": 18})
        self.assertFalse(evidence["completion_authority"])
        self.assertFalse(evidence["native_validation_owner_bound"])

    def test_one_dependency_change_is_rejected_even_when_count_preserved(self):
        changed = copy.deepcopy(self.candidate)
        row = next(row for row in changed["facts"]["tasks"] if row["depends_on"])
        row["depends_on"][0] = "LIR-9002"
        self.refused(changed, "candidate_facts_differs")

    def test_missing_goal_parent_is_rejected(self):
        changed = copy.deepcopy(self.candidate)
        changed["facts"]["goals"][1]["parent"] = None
        self.refused(changed, "candidate_facts_differs")

    def test_fabricated_repository_head_is_rejected(self):
        changed = copy.deepcopy(self.candidate)
        changed["facts"]["repository_observations"][0]["head"] = "0" * 40
        self.refused(changed, "candidate_facts_differs")

    def test_metadata_snapshot_cannot_be_called_fresh_runtime_observation(self):
        changed = copy.deepcopy(self.candidate)
        changed["facts"]["observation_snapshot"]["status"] = "runtime_authenticated"
        self.refused(changed, "candidate_facts_differs")

    def test_frozen_source_body_change_is_rejected(self):
        altered = dict(self.frozen)
        name = next(iter(altered))
        altered[name] += b"\n"
        with self.assertRaisesRegex(facts.FactualRefusal, "frozen_file_pin_differs"):
            facts.check_facts(facts.canonical(self.candidate), self.sources, altered)

    def test_source_pin_change_is_rejected(self):
        changed = copy.deepcopy(self.candidate)
        changed["source_commitments"]["execution_board"]["sha256"] = "0" * 64
        self.refused(changed, "candidate_source_commitments_differs")

    def test_finding_cannot_point_to_unread_document(self):
        changed = copy.deepcopy(self.candidate)
        changed["findings"][0]["evidence_role"] = "foreign_provider_receipt"
        self.refused(changed, "finding_reference_or_scope_differs")

    def test_free_text_finding_is_rejected(self):
        changed = copy.deepcopy(self.candidate)
        changed["findings"][0]["summary"] = "The model is accurate and may train."
        self.refused(changed, "finding_reference_or_scope_differs")

    def test_unavailable_repository_claim_requires_actual_unavailable_row(self):
        changed = copy.deepcopy(self.candidate)
        expected = facts.derive_expected(self.sources, self.frozen)
        changed["findings"].append(facts._finding("preserve_unavailable_repository", expected))
        self.refused(changed, "unavailable_repository_claim_unsupported")

    def test_duplicate_finding_is_rejected(self):
        changed = copy.deepcopy(self.candidate)
        changed["findings"].append(copy.deepcopy(changed["findings"][0]))
        self.refused(changed, "duplicate_finding")

    def test_missing_unresolved_scope_is_rejected(self):
        changed = copy.deepcopy(self.candidate)
        changed["unresolved_scope"].remove("statutory_semantics_unreviewed")
        self.refused(changed, "candidate_omits_unresolved_scope")

    def test_model_boolean_cannot_create_authority(self):
        for field in self.candidate["authority"]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.candidate)
                changed["authority"][field] = True
                self.refused(changed, "candidate_promotes_authority")

    def test_numeric_zero_does_not_replace_false_authority(self):
        changed = copy.deepcopy(self.candidate)
        changed["authority"]["completion_authority"] = 0
        self.refused(changed, "candidate_promotes_authority")

    def test_count_float_does_not_replace_integer(self):
        changed = copy.deepcopy(self.candidate)
        changed["facts"]["population"]["tasks"] = 64.0
        self.refused(changed, "candidate_facts_differs")

    def test_next_phase_remains_blocked(self):
        changed = copy.deepcopy(self.candidate)
        changed["next_phase"]["activated"] = True
        self.refused(changed, "candidate_activates_next_phase")

    def test_foreign_receipt_field_cannot_create_owner_binding(self):
        changed = copy.deepcopy(self.candidate)
        changed["native_validation_owner_bound"] = True
        self.refused(changed, "candidate_fields_differ")

    def test_duplicate_json_and_nonfinite_numbers_are_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(facts.FactualRefusal):
                facts.strict_json(raw)

    def test_future_task_unblocking_in_source_is_rejected(self):
        source = json.loads(self.sources["execution_projection"])
        source["tasks"][1]["status"] = "todo"
        altered = dict(self.sources, execution_projection=facts.canonical(source))
        with self.assertRaisesRegex(facts.FactualRefusal, "execution_frontier_differs"):
            facts.candidate_template(altered, self.frozen)

    def test_provider_fallback_change_is_rejected(self):
        source = json.loads(self.sources["scheduler_config"])
        source["provider"]["fallback_trigger"] = "any_failure"
        altered = dict(self.sources, scheduler_config=facts.canonical(source))
        with self.assertRaisesRegex(facts.FactualRefusal, "provider_tuple_differs"):
            facts.candidate_template(altered, self.frozen)

    def test_completion_cli_remains_closed_for_valid_facts(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "candidate.json"
            report.write_bytes(facts.canonical(self.candidate))
            with redirect_stdout(io.StringIO()) as output:
                code = facts.main(["--source-root", str(FIXTURE.absolute()), "--check-report", str(report)])
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(output.getvalue())["reason"], "independent_native_validation_owner_binding_not_bound")

    def test_symlink_and_hardlink_source_inputs_are_rejected(self):
        import os
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source.json"
            source.write_bytes(b"{}")
            symlink = base / "symlink.json"
            symlink.symlink_to(source)
            with self.assertRaisesRegex(facts.FactualRefusal, "source_symlink_refused"):
                facts.read_source_file(symlink)
            hardlink = base / "hardlink.json"
            os.link(source, hardlink)
            with self.assertRaisesRegex(facts.FactualRefusal, "bounded_regular_source_required"):
                facts.read_source_file(source)


if __name__ == "__main__":
    unittest.main()
