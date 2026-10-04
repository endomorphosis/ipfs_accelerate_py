"""The qualification runner cannot promote an incomplete smoke result."""
import json
import tempfile
import unittest
from pathlib import Path

from run_supervision_docker import verified_doctor_qualification


class QualificationReceiptTests(unittest.TestCase):
    def test_missing_or_nonpositive_stage_is_rejected(self):
        valid = dict(
            schema="doctor-native-composition-qualification@1",
            tactician_planned=True, sealed_proof_verified=True,
            native_synthesis_admitted=True, graph_impact_closed=True,
            transaction_committed=True, publication_verified=True,
            before_behavior_failed=True, after_behavior_passed=True,
            executed_validation_receipts=12, task_completion_authorized=False,
            terminal_bench_result=False, provider_calls=0,
            expected_base_commit="before", committed_commit="after",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            path.write_text(json.dumps(valid))
            self.assertEqual(verified_doctor_qualification(path), valid)
            for key, value in (
                ("sealed_proof_verified", False), ("publication_verified", "true"),
                ("executed_validation_receipts", True), ("executed_validation_receipts", 0),
                ("task_completion_authorized", True), ("committed_commit", "before"),
                ("schema", "unknown"),
            ):
                with self.subTest(key=key, value=value):
                    path.write_text(json.dumps({**valid, key: value}))
                    with self.assertRaises(ValueError):
                        verified_doctor_qualification(path)
            path.unlink()
            with self.assertRaises(OSError):
                verified_doctor_qualification(path)


if __name__ == "__main__":
    unittest.main()
