import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from index_preflight import qualify


class IndexQualificationTests(unittest.TestCase):
    def test_real_persistence_and_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("def example():\n    return 3\n")
            receipt = qualify(root, root / "state", ["sample.py"])
            self.assertEqual(receipt["status"], "qualified")
            self.assertEqual(receipt["symbols"], 1)
            self.assertEqual(receipt["reopened_lake_counts"]["identity_links"], 1)
            self.assertFalse(receipt["full_system_benchmark"])
            self.assertFalse(receipt["supervisor_consumed_retrieval"])

    def test_missing_ducklake_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("def example():\n    return 3\n")
            with patch("ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index.SupervisorMetaIndex.project_ducklake", return_value={"status": "unavailable"}):
                with self.assertRaisesRegex(RuntimeError, "DuckLake projection failed"):
                    qualify(root, root / "state", ["sample.py"])

    def test_no_source_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "unsafe input"):
                qualify(root, root / "state", ["../secret.py"])
            self.assertFalse((root / "state").exists())


if __name__ == "__main__":
    unittest.main()
