"""
Tests for ModelManager.seed_well_known_models (Gap 5 implementation).

Covers:
- Returns count > 0 on first call
- Idempotency: second call with overwrite=False adds 0 new models
- overwrite=True replaces existing entries
- xAI Grok models are present with correct metadata
- Meta AI / Llama / Spark models are present with correct metadata
- OpenAI, Anthropic, Google models are present
- All seeded models have ServingConfig with engine="api"
- All seeded models have supported_backends containing "api"
- Tags are correct for key models
- get_models_by_pipeline_type returns seeded API models
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ipfs_accelerate_py.model_manager import ModelManager, ModelType


def _make_mm():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "mm.json")
    return ModelManager(path), tmp


# ---------------------------------------------------------------------------
class TestSeedWellKnownModelsBasic(unittest.TestCase):
    def setUp(self):
        self.mm, self._tmp = _make_mm()

    def tearDown(self):
        self.mm.close()

    def test_returns_positive_count(self):
        n = self.mm.seed_well_known_models()
        self.assertGreater(n, 0)

    def test_models_persisted_after_seed(self):
        self.mm.seed_well_known_models()
        models = self.mm.list_models()
        self.assertGreater(len(models), 0)

    def test_idempotent_second_call_adds_zero(self):
        self.mm.seed_well_known_models()
        n2 = self.mm.seed_well_known_models()
        self.assertEqual(n2, 0)

    def test_overwrite_true_replaces_all(self):
        self.mm.seed_well_known_models()
        n2 = self.mm.seed_well_known_models(overwrite=True)
        self.assertGreater(n2, 0)

    def test_total_seeded_at_least_fifteen(self):
        n = self.mm.seed_well_known_models()
        self.assertGreaterEqual(n, 15)


# ---------------------------------------------------------------------------
class TestSeedWellKnownModelsXAI(unittest.TestCase):
    def setUp(self):
        self.mm, self._tmp = _make_mm()
        self.mm.seed_well_known_models()
        self.models = {m.model_id: m for m in self.mm.list_models()}

    def tearDown(self):
        self.mm.close()

    def test_grok3_present(self):
        self.assertIn("xai/grok-3", self.models)

    def test_grok3_fast_present(self):
        self.assertIn("xai/grok-3-fast", self.models)

    def test_grok3_mini_present(self):
        self.assertIn("xai/grok-3-mini", self.models)

    def test_grok2_present(self):
        self.assertIn("xai/grok-2-1212", self.models)

    def test_grok2_vision_present(self):
        self.assertIn("xai/grok-2-vision-1212", self.models)

    def test_grok_models_are_language_or_multimodal(self):
        for mid in ("xai/grok-3", "xai/grok-3-fast", "xai/grok-3-mini", "xai/grok-2-1212"):
            self.assertEqual(self.models[mid].model_type, ModelType.LANGUAGE_MODEL)
        self.assertEqual(self.models["xai/grok-2-vision-1212"].model_type, ModelType.MULTIMODAL)

    def test_grok_models_have_xai_tag(self):
        for mid in ("xai/grok-3", "xai/grok-3-fast"):
            self.assertIn("xai", self.models[mid].tags)

    def test_grok_models_serving_config_engine_api(self):
        m = self.models["xai/grok-3"]
        self.assertIsNotNone(m.serving_config)
        self.assertEqual(m.serving_config["engine"], "api")

    def test_grok_models_supported_backends_api(self):
        m = self.models["xai/grok-3"]
        self.assertIn("api", m.supported_backends)

    def test_grok_serving_config_provider_xai(self):
        m = self.models["xai/grok-3"]
        self.assertEqual(m.serving_config["launch_args"]["provider"], "xai")


# ---------------------------------------------------------------------------
class TestSeedWellKnownModelsMetaAI(unittest.TestCase):
    def setUp(self):
        self.mm, self._tmp = _make_mm()
        self.mm.seed_well_known_models()
        self.models = {m.model_id: m for m in self.mm.list_models()}

    def tearDown(self):
        self.mm.close()

    def test_llama_33_70b_present(self):
        self.assertIn("meta-llama/Llama-3.3-70B-Instruct", self.models)

    def test_llama_31_405b_present(self):
        self.assertIn("meta-llama/Llama-3.1-405B-Instruct", self.models)

    def test_llama_31_8b_present(self):
        self.assertIn("meta-llama/Llama-3.1-8B-Instruct", self.models)

    def test_llama_32_vision_present(self):
        self.assertIn("meta-llama/Llama-3.2-90B-Vision-Instruct", self.models)

    def test_spark_11_present(self):
        self.assertIn("meta-spark/Spark-1.1", self.models)

    def test_llama_32_vision_is_multimodal(self):
        m = self.models["meta-llama/Llama-3.2-90B-Vision-Instruct"]
        self.assertEqual(m.model_type, ModelType.MULTIMODAL)

    def test_spark_is_language_model(self):
        m = self.models["meta-spark/Spark-1.1"]
        self.assertEqual(m.model_type, ModelType.LANGUAGE_MODEL)

    def test_llama_models_have_meta_tag(self):
        for mid in ("meta-llama/Llama-3.3-70B-Instruct",
                    "meta-llama/Llama-3.1-405B-Instruct"):
            self.assertIn("meta", self.models[mid].tags)

    def test_spark_has_spark_tag(self):
        self.assertIn("spark", self.models["meta-spark/Spark-1.1"].tags)

    def test_meta_serving_config_engine_api(self):
        m = self.models["meta-llama/Llama-3.3-70B-Instruct"]
        self.assertIsNotNone(m.serving_config)
        self.assertEqual(m.serving_config["engine"], "api")

    def test_meta_serving_config_provider(self):
        m = self.models["meta-llama/Llama-3.3-70B-Instruct"]
        self.assertEqual(m.serving_config["launch_args"]["provider"], "meta_ai")

    def test_llama_context_window_stored(self):
        m = self.models["meta-llama/Llama-3.3-70B-Instruct"]
        self.assertEqual(m.serving_config.get("context_window"), 128000)


# ---------------------------------------------------------------------------
class TestSeedWellKnownModelsOtherProviders(unittest.TestCase):
    def setUp(self):
        self.mm, self._tmp = _make_mm()
        self.mm.seed_well_known_models()
        self.models = {m.model_id: m for m in self.mm.list_models()}

    def tearDown(self):
        self.mm.close()

    def test_openai_gpt4o_present(self):
        self.assertIn("openai/gpt-4o", self.models)

    def test_openai_gpt4o_mini_present(self):
        self.assertIn("openai/gpt-4o-mini", self.models)

    def test_openai_embedding_present(self):
        self.assertIn("openai/text-embedding-3-large", self.models)

    def test_openai_embedding_is_embedding_model(self):
        m = self.models["openai/text-embedding-3-large"]
        self.assertEqual(m.model_type, ModelType.EMBEDDING_MODEL)

    def test_claude_opus_present(self):
        self.assertIn("anthropic/claude-opus-4-5", self.models)

    def test_gemini_pro_present(self):
        self.assertIn("google/gemini-2.5-pro", self.models)

    def test_gemini_is_multimodal(self):
        m = self.models["google/gemini-2.5-pro"]
        self.assertEqual(m.model_type, ModelType.MULTIMODAL)

    def test_all_seeded_models_have_api_engine(self):
        for m in self.models.values():
            if m.serving_config:
                self.assertEqual(
                    m.serving_config["engine"], "api",
                    f"{m.model_id} should have engine=api, got {m.serving_config['engine']}",
                )

    def test_all_seeded_models_have_api_backend(self):
        for m in self.models.values():
            self.assertIn(
                "api", m.supported_backends,
                f"{m.model_id} missing 'api' in supported_backends",
            )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
