"""
Tests for Meta AI CLI Integration

Covers:
- Module import and instantiation
- Creative Mode (preview approval: headless auto-approve, interactive mock, rejection)
- Vision Chat (image_url forwarded correctly, never cached)
- Standard chat() and generate_code() helpers
- Automatic model selection via suggest_model()
- Global singleton
- list_models() / get_model_info()
- Cache hit / miss behaviour
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers to mock out heavy deps before import
# ---------------------------------------------------------------------------

def _make_mock_cache():
    cache = MagicMock()
    cache.get_chat_completion.return_value = None
    cache.cache_chat_completion.return_value = None
    return cache


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestMetaAICLIIntegration(unittest.TestCase):
    """Unit tests for MetaAICLIIntegration."""

    def setUp(self):
        """Patch heavy external dependencies."""
        # Patch secrets manager so no real credentials are needed
        self.secrets_patcher = patch(
            "ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager"
        )
        mock_sm = self.secrets_patcher.start()
        mock_sm.return_value.get_credential.return_value = None

        # Patch shutil.which so CLI detection doesn't probe the filesystem
        self.which_patcher = patch("shutil.which", return_value=None)
        self.which_patcher.start()

        # Patch subprocess.run used in _check_cli_available
        self.subprocess_patcher = patch("subprocess.run")
        self.subprocess_patcher.start()

        # Patch get_llm_cache / get_global_llm_cache
        self.llm_cache_patcher = patch(
            "ipfs_accelerate_py.cli_integrations.meta_ai_cli_integration.get_llm_cache"
        )
        self.global_llm_cache_patcher = patch(
            "ipfs_accelerate_py.cli_integrations.meta_ai_cli_integration.get_global_llm_cache"
        )
        mock_llm = self.llm_cache_patcher.start()
        mock_global = self.global_llm_cache_patcher.start()
        self._mock_cache = _make_mock_cache()
        mock_llm.return_value = self._mock_cache
        mock_global.return_value = self._mock_cache

        # Now import (after patching)
        from ipfs_accelerate_py.cli_integrations.meta_ai_cli_integration import (
            MetaAICLIIntegration,
            get_meta_ai_cli_integration,
        )
        self.MetaAICLIIntegration = MetaAICLIIntegration
        self.get_meta_ai_cli_integration = get_meta_ai_cli_integration

    def tearDown(self):
        self.secrets_patcher.stop()
        self.which_patcher.stop()
        self.subprocess_patcher.stop()
        self.llm_cache_patcher.stop()
        self.global_llm_cache_patcher.stop()

        # Reset global singleton between tests
        import ipfs_accelerate_py.cli_integrations.meta_ai_cli_integration as mod
        mod._global_meta_ai_cli = None

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_instantiation_default(self):
        """Integration can be created without arguments."""
        integration = self.MetaAICLIIntegration()
        self.assertEqual(integration.get_tool_name(), "Meta AI")
        self.assertFalse(integration.headless)

    def test_instantiation_headless_kwarg(self):
        integration = self.MetaAICLIIntegration(headless=True)
        self.assertTrue(integration.headless)

    def test_instantiation_headless_env(self):
        with patch.dict(os.environ, {"META_AI_HEADLESS": "1"}):
            integration = self.MetaAICLIIntegration(headless=False)
        self.assertTrue(integration.headless)

    def test_get_tool_name(self):
        integration = self.MetaAICLIIntegration()
        self.assertEqual(integration.get_tool_name(), "Meta AI")

    # ------------------------------------------------------------------
    # list_models / get_model_info
    # ------------------------------------------------------------------

    def test_list_models(self):
        integration = self.MetaAICLIIntegration()
        models = integration.list_models()
        self.assertIsInstance(models, list)
        self.assertIn("meta-llama/Llama-3.3-70B-Instruct", models)
        self.assertIn("meta-spark/Spark-1.1", models)
        self.assertIn("meta-llama/Llama-3.2-11B-Vision-Instruct", models)

    def test_get_model_info_known(self):
        integration = self.MetaAICLIIntegration()
        info = integration.get_model_info("meta-spark/Spark-1.1")
        self.assertIsNotNone(info)
        self.assertIn("creative_mode", info["recommended_for"])

    def test_get_model_info_unknown(self):
        integration = self.MetaAICLIIntegration()
        info = integration.get_model_info("does-not-exist")
        self.assertIsNone(info)

    # ------------------------------------------------------------------
    # suggest_model
    # ------------------------------------------------------------------

    def test_suggest_model_creative(self):
        self.assertEqual(
            self.MetaAICLIIntegration.suggest_model("creative"),
            "meta-spark/Spark-1.1",
        )

    def test_suggest_model_vision(self):
        self.assertEqual(
            self.MetaAICLIIntegration.suggest_model("vision"),
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
        )

    def test_suggest_model_reasoning(self):
        self.assertEqual(
            self.MetaAICLIIntegration.suggest_model("reasoning"),
            "meta-llama/Llama-3.1-405B-Instruct",
        )

    def test_suggest_model_subagent(self):
        self.assertEqual(
            self.MetaAICLIIntegration.suggest_model("subagent"),
            "meta-llama/Llama-3.2-1B-Instruct",
        )

    def test_suggest_model_unknown_falls_back_to_default(self):
        from ipfs_accelerate_py.cli_integrations.meta_ai_cli_integration import (
            _DEFAULT_MODEL,
        )
        self.assertEqual(
            self.MetaAICLIIntegration.suggest_model("something_unknown"),
            _DEFAULT_MODEL,
        )

    # ------------------------------------------------------------------
    # Creative Mode
    # ------------------------------------------------------------------

    def test_creative_mode_headless_auto_approves(self):
        integration = self.MetaAICLIIntegration(headless=True)
        mock_raw = {"response": "Once upon a time...", "cached": False, "mode": "SDK"}
        with patch.object(integration, "_execute_with_fallback", return_value=mock_raw):
            result = integration.creative_mode("Write a story")
        self.assertTrue(result["approved"])
        self.assertNotEqual(result["response"], "")
        self.assertEqual(result["model"], "meta-spark/Spark-1.1")

    def test_creative_mode_auto_approve_true(self):
        integration = self.MetaAICLIIntegration(headless=False)
        mock_raw = {"response": "Once upon a time...", "cached": False, "mode": "SDK"}
        with patch.object(integration, "_execute_with_fallback", return_value=mock_raw):
            result = integration.creative_mode("Write a story", auto_approve=True)
        self.assertTrue(result["approved"])

    def test_creative_mode_interactive_approved(self):
        integration = self.MetaAICLIIntegration(headless=False)
        mock_raw = {"response": "A creative story", "cached": False, "mode": "SDK"}
        with patch.object(integration, "_execute_with_fallback", return_value=mock_raw), \
             patch.object(
                 type(integration), "_prompt_creative_approval",
                 staticmethod(lambda preview: True),
             ):
            result = integration.creative_mode("Tell a story", auto_approve=None)
        self.assertTrue(result["approved"])
        self.assertEqual(result["response"], "A creative story")

    def test_creative_mode_interactive_rejected(self):
        integration = self.MetaAICLIIntegration(headless=False)
        mock_raw = {"response": "A creative story", "cached": False, "mode": "SDK"}
        with patch.object(integration, "_execute_with_fallback", return_value=mock_raw), \
             patch.object(
                 type(integration), "_prompt_creative_approval",
                 staticmethod(lambda preview: False),
             ):
            result = integration.creative_mode("Tell a story", auto_approve=None)
        self.assertFalse(result["approved"])
        self.assertEqual(result["response"], "")

    def test_creative_mode_auto_approve_false_overrides_headless(self):
        integration = self.MetaAICLIIntegration(headless=True)
        mock_raw = {"response": "Some content", "cached": False, "mode": "SDK"}
        with patch.object(integration, "_execute_with_fallback", return_value=mock_raw), \
             patch.object(
                 type(integration), "_prompt_creative_approval",
                 staticmethod(lambda preview: False),
             ):
            result = integration.creative_mode("prompt", auto_approve=False)
        self.assertFalse(result["approved"])

    # ------------------------------------------------------------------
    # Vision Chat
    # ------------------------------------------------------------------

    def test_vision_chat_passes_image_url(self):
        integration = self.MetaAICLIIntegration()
        mock_raw = {"response": "An image of IPFS nodes", "cached": False, "mode": "SDK"}
        captured: dict = {}

        def fake_execute_with_fallback(sdk_func, operation, **kwargs):
            captured["operation"] = operation
            captured.update(kwargs)
            return mock_raw

        with patch.object(integration, "_execute_with_fallback", side_effect=fake_execute_with_fallback):
            result = integration.vision_chat(
                "Describe this diagram",
                image_url="https://example.com/ipfs.png",
            )

        self.assertEqual(result["response"], "An image of IPFS nodes")
        self.assertEqual(captured.get("image_url"), "https://example.com/ipfs.png")
        self.assertEqual(captured.get("operation"), "vision_chat")

    def test_vision_chat_uses_vision_model_by_default(self):
        from ipfs_accelerate_py.cli_integrations.meta_ai_cli_integration import (
            _DEFAULT_VISION_MODEL,
        )
        integration = self.MetaAICLIIntegration()
        captured: dict = {}

        def fake_execute_with_fallback(sdk_func, operation, **kwargs):
            captured.update(kwargs)
            return {"response": "ok", "cached": False, "mode": "SDK"}

        with patch.object(integration, "_execute_with_fallback", side_effect=fake_execute_with_fallback):
            integration.vision_chat("What is this?", image_url="https://example.com/x.png")

        self.assertEqual(captured.get("model"), _DEFAULT_VISION_MODEL)
    # ------------------------------------------------------------------
    # Standard chat / generate_code
    # ------------------------------------------------------------------

    def test_chat_returns_response(self):
        integration = self.MetaAICLIIntegration()
        mock_raw = {"response": "Hello from Meta AI", "cached": False, "mode": "SDK"}
        with patch.object(integration, "_execute_with_fallback", return_value=mock_raw):
            result = integration.chat("Hello!")
        self.assertEqual(result["response"], "Hello from Meta AI")
        self.assertFalse(result["cached"])

    def test_generate_code_no_plan(self):
        integration = self.MetaAICLIIntegration()
        captured: dict = {}

        def fake_chat(msg, model, temperature, **kwargs):
            captured["temperature"] = temperature
            return {"response": "def foo(): pass", "cached": False, "mode": "SDK"}

        with patch.object(integration, "chat", side_effect=fake_chat):
            integration.generate_code("Write a function foo")

        self.assertEqual(captured["temperature"], 0.0)

    def test_chat_cache_hit(self):
        integration = self.MetaAICLIIntegration()
        mock_cache = _make_mock_cache()
        mock_cache.get_chat_completion.return_value = "cached response"
        integration.cache = mock_cache
        integration.enable_cache = True

        mock_client = MagicMock()
        with patch.object(integration, "_get_openai_client", return_value=mock_client):
            result = integration._chat_sdk(
                message="What is IPFS?",
                model="meta-llama/Llama-3.3-70B-Instruct",
                temperature=0.7,
            )

        self.assertTrue(result["cached"])
        self.assertEqual(result["response"], "cached response")
        mock_client.chat.completions.create.assert_not_called()

    def test_chat_not_cached(self):
        integration = self.MetaAICLIIntegration()
        integration.enable_cache = True
        self._mock_cache.get_chat_completion.return_value = None

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Live response"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        with patch.object(integration, "_get_openai_client", return_value=mock_client):
            result = integration._chat_sdk(
                message="What is IPFS?",
                model="meta-llama/Llama-3.3-70B-Instruct",
                temperature=0.7,
            )

        self.assertFalse(result["cached"])
        self.assertEqual(result["response"], "Live response")

    # ------------------------------------------------------------------
    # Global singleton
    # ------------------------------------------------------------------

    def test_global_singleton(self):
        import ipfs_accelerate_py.cli_integrations.meta_ai_cli_integration as mod
        mod._global_meta_ai_cli = None
        inst1 = self.get_meta_ai_cli_integration()
        inst2 = self.get_meta_ai_cli_integration()
        self.assertIs(inst1, inst2)


class TestMetaAICLIImport(unittest.TestCase):
    """Verify package-level exports."""

    def test_package_exports(self):
        from ipfs_accelerate_py.cli_integrations import (
            MetaAICLIIntegration,
            get_meta_ai_cli_integration,
        )
        self.assertTrue(callable(MetaAICLIIntegration))
        self.assertTrue(callable(get_meta_ai_cli_integration))

    def test_present_in_all(self):
        import ipfs_accelerate_py.cli_integrations as pkg
        self.assertIn("MetaAICLIIntegration", pkg.__all__)
        self.assertIn("get_meta_ai_cli_integration", pkg.__all__)


# ---------------------------------------------------------------------------
# api_backends/meta_ai.py unit tests
# ---------------------------------------------------------------------------

class TestMetaAIBackend(unittest.TestCase):
    """Unit tests for the meta_ai API backend class."""

    def test_import(self):
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai, ALL_MODELS, CHAT_MODELS
        self.assertIn("meta-llama/Llama-3.3-70B-Instruct", CHAT_MODELS)
        self.assertIn("meta-llama/Llama-3.2-90B-Vision-Instruct", CHAT_MODELS)
        self.assertIn("meta-spark/Spark-1.1", CHAT_MODELS)
        self.assertIsInstance(ALL_MODELS, dict)

    def test_init_no_key(self):
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_cls
        saved = {k: os.environ.pop(k, None) for k in ("META_AI_API_KEY", "ipfs_accelerate_py_META_AI_API_KEY")}
        try:
            client = meta_cls(resources={}, metadata={})
            self.assertIsNone(client.api_key)
            self.assertEqual(client.default_model, "meta-llama/Llama-3.3-70B-Instruct")
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_init_with_metadata(self):
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_cls
        client = meta_cls(
            resources={},
            metadata={
                "api_key": "test-key",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "max_retries": "5",
                "timeout": "45.0",
            },
        )
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.default_model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.timeout, 45.0)

    def test_list_models(self):
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_cls
        client = meta_cls()
        models = client.list_models()
        self.assertIsInstance(models, list)
        self.assertIn("meta-llama/Llama-3.3-70B-Instruct", models)
        self.assertIn("meta-spark/Spark-1.1", models)

    def test_get_model_info_known(self):
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_cls
        client = meta_cls()
        info = client.get_model_info("meta-llama/Llama-3.3-70B-Instruct")
        self.assertIsInstance(info, dict)
        self.assertIn("context_window", info)

    def test_get_model_info_unknown(self):
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_cls
        client = meta_cls()
        self.assertIsNone(client.get_model_info("nonexistent-model"))

    def test_generate_returns_string_on_successful_response(self):
        from unittest.mock import patch
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_cls
        client = meta_cls(metadata={"api_key": "dummy"})
        fake_response = {
            "choices": [{"message": {"content": "Hello from Llama!"}}]
        }
        with patch.object(client, "_make_request", return_value=fake_response):
            result = client.generate("Hello")
        self.assertEqual(result, "Hello from Llama!")

    def test_generate_returns_empty_on_empty_choices(self):
        from unittest.mock import patch
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_cls
        client = meta_cls(metadata={"api_key": "dummy"})
        with patch.object(client, "_make_request", return_value={"choices": []}):
            self.assertEqual(client.generate("Hello"), "")

    def test_embed_returns_list_of_vectors(self):
        from unittest.mock import patch
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_cls
        client = meta_cls(metadata={"api_key": "dummy"})
        fake_response = {"data": [{"embedding": [0.4, 0.5, 0.6]}]}
        with patch.object(client, "_make_request", return_value=fake_response):
            vecs = client.embed(["hello"])
        self.assertEqual(vecs, [[0.4, 0.5, 0.6]])

    def test_make_request_raises_without_key(self):
        from ipfs_accelerate_py.api_backends.meta_ai import meta_ai as meta_cls
        saved = {k: os.environ.pop(k, None) for k in ("META_AI_API_KEY", "ipfs_accelerate_py_META_AI_API_KEY")}
        try:
            client = meta_cls()
            with self.assertRaises(RuntimeError, msg="should raise without API key"):
                client._make_request("chat/completions", {})
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# api_models_registry prefix-mapping tests
# ---------------------------------------------------------------------------

class TestApiModelsRegistryMetaPrefixes(unittest.TestCase):
    """Verify meta-llama/ and meta-spark/ are mapped to the meta_ai backend."""

    def setUp(self):
        from ipfs_accelerate_py.api_backends.api_models_registry import api_models
        self.registry = api_models()

    def test_meta_llama_prefix(self):
        backend = self.registry.get_backend_for_model("meta-llama/Llama-3.3-70B-Instruct")
        self.assertEqual(backend, "meta_ai")

    def test_meta_spark_prefix(self):
        backend = self.registry.get_backend_for_model("meta-spark/Spark-1.1")
        self.assertEqual(backend, "meta_ai")

    def test_openai_prefix_unchanged(self):
        backend = self.registry.get_backend_for_model("openai/gpt-4")
        self.assertEqual(backend, "openai_api")

    def test_unknown_prefix_returns_none(self):
        backend = self.registry.get_backend_for_model("unknownprovider/some-model")
        self.assertIsNone(backend)


# ---------------------------------------------------------------------------
# embeddings_router integration tests for Meta AI
# ---------------------------------------------------------------------------

class TestMetaAIEmbeddingsRouter(unittest.TestCase):
    """Verify the Meta AI provider is wired into the embeddings router."""

    def test_builtin_provider_by_name_meta_ai(self):
        from ipfs_accelerate_py.embeddings_router import _builtin_provider_by_name
        from ipfs_accelerate_py.router_deps import get_default_router_deps
        os.environ["META_AI_API_KEY"] = "dummy"
        try:
            provider = _builtin_provider_by_name("meta_ai", get_default_router_deps())
            self.assertIsNotNone(provider)
            self.assertTrue(hasattr(provider, "embed_texts"))
        finally:
            del os.environ["META_AI_API_KEY"]

    def test_aliases_meta_and_spark(self):
        from ipfs_accelerate_py.embeddings_router import _builtin_provider_by_name
        from ipfs_accelerate_py.router_deps import get_default_router_deps
        os.environ["META_AI_API_KEY"] = "dummy"
        try:
            for alias in ("meta", "spark", "meta_llama", "meta_spark"):
                with self.subTest(alias=alias):
                    self.assertIsNotNone(_builtin_provider_by_name(alias, get_default_router_deps()))
        finally:
            del os.environ["META_AI_API_KEY"]

    def test_no_provider_without_key(self):
        from ipfs_accelerate_py.embeddings_router import _get_meta_ai_embeddings_provider
        saved = {k: os.environ.pop(k, None) for k in ("META_AI_API_KEY", "ipfs_accelerate_py_META_AI_API_KEY")}
        try:
            self.assertIsNone(_get_meta_ai_embeddings_provider())
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_cache_key_includes_meta_ai_vars(self):
        from ipfs_accelerate_py.embeddings_router import _provider_cache_key
        os.environ["META_AI_API_KEY"] = "key-a"
        k1 = _provider_cache_key()
        os.environ["META_AI_API_KEY"] = "key-b"
        k2 = _provider_cache_key()
        self.assertNotEqual(k1, k2)
        del os.environ["META_AI_API_KEY"]


# ---------------------------------------------------------------------------
# multimodal_router integration tests for Meta AI
# ---------------------------------------------------------------------------

class TestMetaAIMultimodalRouter(unittest.TestCase):
    """Verify the Meta AI provider is wired into the multimodal router."""

    def test_builtin_provider_by_name_meta_ai(self):
        from ipfs_accelerate_py.multimodal_router import _builtin_provider_by_name
        from ipfs_accelerate_py.router_deps import get_default_router_deps
        os.environ["META_AI_API_KEY"] = "dummy"
        try:
            provider = _builtin_provider_by_name("meta_ai", get_default_router_deps())
            self.assertIsNotNone(provider)
            self.assertTrue(hasattr(provider, "generate"))
        finally:
            del os.environ["META_AI_API_KEY"]

    def test_no_provider_without_key(self):
        from ipfs_accelerate_py.multimodal_router import _get_meta_ai_multimodal_provider
        saved = {k: os.environ.pop(k, None) for k in ("META_AI_API_KEY", "ipfs_accelerate_py_META_AI_API_KEY")}
        try:
            self.assertIsNone(_get_meta_ai_multimodal_provider())
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_cache_key_includes_meta_ai_vars(self):
        from ipfs_accelerate_py.multimodal_router import _provider_cache_key
        os.environ["META_AI_API_KEY"] = "key-a"
        k1 = _provider_cache_key()
        os.environ["META_AI_API_KEY"] = "key-b"
        k2 = _provider_cache_key()
        self.assertNotEqual(k1, k2)
        del os.environ["META_AI_API_KEY"]


if __name__ == "__main__":
    unittest.main()
