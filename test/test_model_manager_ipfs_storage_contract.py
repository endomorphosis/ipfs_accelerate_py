#!/usr/bin/env python3
"""Contract test for CID-backed model artifact registration in ModelManager."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ipfs_accelerate_py.model_manager import ModelManager, ModelMetadata, IOSpec, ModelType, DataType
except ImportError:
    from model_manager import ModelManager, ModelMetadata, IOSpec, ModelType, DataType


class FakeStorage:
    def __init__(self):
        self.calls = []
        self.objects = {}

    def store(self, data, filename=None, pin=False):
        self.calls.append((data, filename, pin))
        cid = f"cid:{filename}"
        if hasattr(data, "read_bytes"):
            payload = data.read_bytes()
        elif isinstance(data, bytes):
            payload = data
        else:
            payload = str(data).encode("utf-8")
        self.objects[cid] = payload
        return cid

    def retrieve(self, cid):
        return self.objects.get(cid)


class FakeDatasetsManager:
    def __init__(self):
        self.events = []
        self.provenance = []

    def log_event(self, event_type, data, level="INFO", category="GENERAL"):
        self.events.append((event_type, data, level, category))
        return True

    def track_provenance(self, operation, data, record_type="TRANSFORMATION"):
        cid = f"prov:{operation}:{len(self.provenance)}"
        self.provenance.append((operation, data, record_type, cid))
        return cid


class FakeProvenanceLogger:
    def __init__(self):
        self.records = []

    def log_transformation(self, operation, data, input_cid=None, output_cid=None):
        cid = f"legacy:{operation}:{len(self.records)}"
        self.records.append((operation, data, input_cid, output_cid, cid))
        return cid


class TestModelManagerIpfsStorageContract(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.json_path = os.path.join(self.temp_dir, "model_manager.json")
        self.model_path = os.path.join(self.temp_dir, "weights.bin")
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.tokenizer_path = os.path.join(self.temp_dir, "tokenizer.json")

        Path(self.model_path).write_bytes(b"weights")
        Path(self.config_path).write_text('{"hidden_size": 768}', encoding="utf-8")
        Path(self.tokenizer_path).write_text('{"vocab_size": 30522}', encoding="utf-8")

        self.metadata = ModelMetadata(
            model_id="test/model",
            model_name="test-model",
            model_type=ModelType.LANGUAGE_MODEL,
            architecture="TestArchitecture",
            inputs=[IOSpec(name="input_ids", data_type=DataType.TOKENS)],
            outputs=[IOSpec(name="logits", data_type=DataType.LOGITS)],
            revision_id="rev-001",
            parent_model_id="parent/model",
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_artifact_cids_persist_through_registration(self):
        manager = ModelManager(storage_path=self.json_path, use_database=False, enable_ipfs=False)
        fake_storage = FakeStorage()
        fake_datasets = FakeDatasetsManager()
        fake_provenance = FakeProvenanceLogger()
        manager._artifact_storage = fake_storage
        manager._datasets_manager = fake_datasets
        manager._provenance_logger = fake_provenance
        manager._ipfs_backend = None

        success, cid = manager.add_model_with_ipfs_storage(
            self.metadata,
            model_path=self.model_path,
            config_path=self.config_path,
            tokenizer_path=self.tokenizer_path,
        )

        self.assertTrue(success)
        self.assertEqual(cid, "cid:test_model.artifact-manifest.json")

        stored = manager.get_model("test/model")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.model_cid, "cid:test_model.model")
        self.assertEqual(stored.config_cid, "cid:test_model.config")
        self.assertEqual(stored.tokenizer_cid, "cid:test_model.tokenizer")
        self.assertEqual(stored.artifact_cid, "cid:test_model.artifact-manifest.json")
        self.assertEqual(stored.repository_structure["artifact_cid"], "cid:test_model.artifact-manifest.json")
        self.assertEqual(len(fake_storage.calls), 4)
        self.assertTrue(any(event[0] == "model_registered" for event in fake_datasets.events))
        self.assertTrue(any(record[0] == "model_registered" for record in fake_provenance.records))

        restore_dir = os.path.join(self.temp_dir, "restored")
        restored = manager.restore_model_artifacts_from_cids("test/model", restore_dir)
        self.assertEqual(set(restored.keys()), {"model_cid", "config_cid", "tokenizer_cid"})
        self.assertEqual(Path(restored["model_cid"]).read_bytes(), b"weights")
        self.assertEqual(Path(restored["config_cid"]).read_text(encoding="utf-8"), '{"hidden_size": 768}')
        self.assertEqual(Path(restored["tokenizer_cid"]).read_text(encoding="utf-8"), '{"vocab_size": 30522}')

        self.assertTrue(
            manager.mark_model_used(
                "test/model",
                inference_cid="cid:inference-result",
                run_id="run-123",
            )
        )
        self.assertTrue(any(event[0] == "model_accessed" for event in fake_datasets.events))
        self.assertTrue(any(record[0] == "model_accessed" for record in fake_provenance.records))

        manager2 = ModelManager(storage_path=self.json_path, use_database=False, enable_ipfs=False)
        reloaded = manager2.get_model("test/model")
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.model_cid, "cid:test_model.model")
        self.assertEqual(reloaded.config_cid, "cid:test_model.config")
        self.assertEqual(reloaded.tokenizer_cid, "cid:test_model.tokenizer")
        self.assertEqual(reloaded.artifact_cid, "cid:test_model.artifact-manifest.json")
        self.assertIsNotNone(reloaded.model_revision)
        self.assertEqual(reloaded.revision_id, "rev-001")
        self.assertEqual(reloaded.parent_model_id, "parent/model")
        self.assertIsNotNone(reloaded.revision_created_at)
        self.assertEqual(reloaded.last_inference_cid, "cid:inference-result")
        self.assertEqual(reloaded.last_run_id, "run-123")
        self.assertEqual(reloaded.inference_count, 1)
        self.assertIsNotNone(reloaded.last_used_at)

        manager.close()
        manager2.close()


if __name__ == "__main__":
    unittest.main()
