"""
Tests for InferenceBackendManager.configure_provider and auto_discover_api_providers
(Gap 4 implementation).

Covers:
- Alias resolution (_resolve_provider_name)
- configure_provider: success path for xAI and Meta AI
- configure_provider: alias forwarding (grok → xai, spark → meta_ai)
- configure_provider: unknown provider returns error dict
- configure_provider: import error handled gracefully
- configure_provider: api_key from environment variable
- configure_provider: backend registered and retrievable
- configure_provider: idempotent (re-registration updates)
- auto_discover_api_providers: only registers providers with env keys set
- auto_discover_api_providers: no env keys → empty list
- auto_discover_api_providers: multiple keys → multiple registered
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_manager(tmp_dir: str):
    """Create a fresh InferenceBackendManager backed by a temp directory."""
    from ipfs_accelerate_py.inference_backend_manager import InferenceBackendManager
    return InferenceBackendManager(config={
        "state_path": os.path.join(tmp_dir, "registry.json"),
        "enable_health_checks": False,
        "persist_registry": False,
    })


def _mock_backend_cls(name: str = "MockBackend"):
    cls = MagicMock(name=name)
    cls.return_value = MagicMock()
    return cls


# ---------------------------------------------------------------------------
class TestResolveProviderName(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()
        self.mgr = _make_manager(self._tmp)

    def test_canonical_unchanged(self):
        self.assertEqual(self.mgr._resolve_provider_name("xai"), "xai")
        self.assertEqual(self.mgr._resolve_provider_name("meta_ai"), "meta_ai")

    def test_alias_grok(self):
        self.assertEqual(self.mgr._resolve_provider_name("grok"), "xai")

    def test_alias_xai_grok(self):
        self.assertEqual(self.mgr._resolve_provider_name("xai_grok"), "xai")

    def test_alias_spark(self):
        self.assertEqual(self.mgr._resolve_provider_name("spark"), "meta_ai")

    def test_alias_meta(self):
        self.assertEqual(self.mgr._resolve_provider_name("meta"), "meta_ai")

    def test_alias_meta_spark(self):
        self.assertEqual(self.mgr._resolve_provider_name("meta_spark"), "meta_ai")

    def test_alias_meta_llama(self):
        self.assertEqual(self.mgr._resolve_provider_name("meta_llama"), "meta_ai")

    def test_alias_openai_api(self):
        self.assertEqual(self.mgr._resolve_provider_name("openai_api"), "openai")

    def test_unknown_passthrough(self):
        self.assertEqual(self.mgr._resolve_provider_name("totally_unknown"), "totally_unknown")


# ---------------------------------------------------------------------------
class TestConfigureProvider(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()
        self.mgr = _make_manager(self._tmp)

    def _patch_import(self, canonical: str):
        """Return a context manager that patches the backend import for *canonical*."""
        spec = self.mgr._PROVIDER_REGISTRY[canonical]
        mod_path, cls_name = spec[0], spec[1]
        mock_cls = _mock_backend_cls(cls_name)
        mock_mod = MagicMock()
        setattr(mock_mod, cls_name, mock_cls)
        return patch("importlib.import_module", return_value=mock_mod), mock_cls

    # -- xAI ------------------------------------------------------------------
    def test_xai_success(self):
        cm, mock_cls = self._patch_import("xai")
        with cm:
            result = self.mgr.configure_provider("xai", api_key="key-xai")
        self.assertTrue(result["configured"])
        self.assertEqual(result["provider"], "xai")
        self.assertEqual(result["backend_id"], "api_xai")

    def test_xai_backend_is_registered(self):
        cm, mock_cls = self._patch_import("xai")
        with cm:
            self.mgr.configure_provider("xai", api_key="key-xai")
        backend = self.mgr.get_backend("api_xai")
        self.assertIsNotNone(backend)
        self.assertIn("text-generation", backend.capabilities.supported_tasks)

    def test_xai_via_grok_alias(self):
        cm, mock_cls = self._patch_import("xai")
        with cm:
            result = self.mgr.configure_provider("grok", api_key="key-grok")
        self.assertEqual(result["provider"], "xai")
        self.assertTrue(result["configured"])

    def test_xai_via_xai_grok_alias(self):
        cm, mock_cls = self._patch_import("xai")
        with cm:
            result = self.mgr.configure_provider("xai_grok", api_key="key")
        self.assertEqual(result["provider"], "xai")

    # -- Meta AI --------------------------------------------------------------
    def test_meta_ai_success(self):
        cm, mock_cls = self._patch_import("meta_ai")
        with cm:
            result = self.mgr.configure_provider("meta_ai", api_key="key-meta")
        self.assertTrue(result["configured"])
        self.assertEqual(result["provider"], "meta_ai")
        self.assertEqual(result["backend_id"], "api_meta_ai")

    def test_meta_ai_via_spark_alias(self):
        cm, mock_cls = self._patch_import("meta_ai")
        with cm:
            result = self.mgr.configure_provider("spark", api_key="key-spark")
        self.assertEqual(result["provider"], "meta_ai")
        self.assertTrue(result["configured"])

    def test_meta_ai_via_meta_alias(self):
        cm, mock_cls = self._patch_import("meta_ai")
        with cm:
            result = self.mgr.configure_provider("meta", api_key="key-meta")
        self.assertEqual(result["provider"], "meta_ai")

    # -- OpenAI ---------------------------------------------------------------
    def test_openai_success(self):
        cm, mock_cls = self._patch_import("openai")
        with cm:
            result = self.mgr.configure_provider("openai", api_key="sk-test")
        self.assertTrue(result["configured"])
        self.assertEqual(result["provider"], "openai")

    # -- Unknown provider -----------------------------------------------------
    def test_unknown_provider_returns_error(self):
        result = self.mgr.configure_provider("no_such_provider")
        self.assertFalse(result["configured"])
        self.assertIn("error", result)

    def test_unknown_provider_not_registered(self):
        self.mgr.configure_provider("no_such_provider")
        backends = self.mgr.list_backends()
        ids = [b["backend_id"] for b in backends]
        self.assertNotIn("api_no_such_provider", ids)

    # -- Import error ---------------------------------------------------------
    def test_import_error_returns_configured_false(self):
        with patch("importlib.import_module", side_effect=ImportError("no module")):
            result = self.mgr.configure_provider("xai", api_key="k")
        self.assertFalse(result["configured"])
        self.assertIn("error", result)

    # -- API key from environment ----------------------------------------------
    def test_api_key_from_env(self):
        cm, mock_cls = self._patch_import("xai")
        with patch.dict(os.environ, {"XAI_API_KEY": "env-key"}):
            with cm:
                result = self.mgr.configure_provider("xai")
        self.assertTrue(result["configured"])
        # The instance was created with the env key
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["metadata"]["api_key"], "env-key")

    def test_api_key_secondary_env_var(self):
        cm, mock_cls = self._patch_import("xai")
        env = {"XAI_API_KEY": "", "ipfs_accelerate_py_XAI_API_KEY": "secondary-key"}
        with patch.dict(os.environ, env, clear=False):
            with cm:
                result = self.mgr.configure_provider("xai")
        self.assertTrue(result["configured"])
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["metadata"]["api_key"], "secondary-key")

    # -- base_url override ----------------------------------------------------
    def test_base_url_override(self):
        cm, mock_cls = self._patch_import("xai")
        with cm:
            self.mgr.configure_provider("xai", api_key="k", base_url="http://localhost:9999")
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["metadata"]["api_base"], "http://localhost:9999")

    # -- Re-registration ------------------------------------------------------
    def test_re_registration_updates(self):
        """configure_provider is idempotent; calling twice replaces the entry."""
        cm, mock_cls = self._patch_import("xai")
        with cm:
            self.mgr.configure_provider("xai", api_key="k1")
            self.mgr.configure_provider("xai", api_key="k2")
        backends = [b for b in self.mgr.list_backends()
                    if b.backend_id == "api_xai"]
        self.assertEqual(len(backends), 1)


# ---------------------------------------------------------------------------
class TestAutoDiscoverApiProviders(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()
        self.mgr = _make_manager(self._tmp)

    def test_no_env_keys_returns_empty(self):
        clean = {k: "" for k in os.environ if "API_KEY" in k}
        # Patch out ALL known primary env keys to empty
        known_keys = {
            "XAI_API_KEY", "META_AI_API_KEY", "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
            "HF_API_KEY",
            "ipfs_accelerate_py_XAI_API_KEY",
            "ipfs_accelerate_py_META_AI_API_KEY",
            "ipfs_accelerate_py_OPENAI_API_KEY",
            "ipfs_accelerate_py_ANTHROPIC_API_KEY",
            "ipfs_accelerate_py_GEMINI_API_KEY",
            "ipfs_accelerate_py_GROQ_API_KEY",
            "ipfs_accelerate_py_HF_API_KEY",
        }
        patch_env = {k: "" for k in known_keys}
        with patch.dict(os.environ, patch_env, clear=False):
            result = self.mgr.auto_discover_api_providers()
        self.assertEqual(result, [])

    def test_xai_key_in_env_registers_xai(self):
        cm, _ = self._patch_import_for("xai")
        clean_env = {
            "META_AI_API_KEY": "", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "", "GROQ_API_KEY": "", "HF_API_KEY": "",
            "XAI_API_KEY": "xai-test",
        }
        with cm, patch.dict(os.environ, clean_env, clear=False):
            result = self.mgr.auto_discover_api_providers()
        self.assertIn("xai", result)

    def test_multiple_keys_register_multiple(self):
        cm_xai, _ = self._patch_import_for("xai")
        cm_meta, _ = self._patch_import_for("meta_ai")
        clean_env = {
            "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "", "GROQ_API_KEY": "", "HF_API_KEY": "",
            "XAI_API_KEY": "xai-k",
            "META_AI_API_KEY": "meta-k",
        }
        with cm_xai, cm_meta, patch.dict(os.environ, clean_env, clear=False):
            result = self.mgr.auto_discover_api_providers()
        self.assertIn("xai", result)
        self.assertIn("meta_ai", result)

    def _patch_import_for(self, canonical: str):
        spec = self.mgr._PROVIDER_REGISTRY[canonical]
        mod_path, cls_name = spec[0], spec[1]
        mock_cls = _mock_backend_cls(cls_name)
        mock_mod = MagicMock()
        setattr(mock_mod, cls_name, mock_cls)
        return patch("importlib.import_module", return_value=mock_mod), mock_cls


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
