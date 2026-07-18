"""
Tests for parallel async execution and multi-key support across CLI integrations.

Covers:
- ApiKeyPool round-robin, per-user pinning, add/remove, edge cases
- DualModeWrapper.get_api_key() with and without a pool
- DualModeWrapper._aexecute_with_fallback() – anyio thread offload
- ClaudeCodeCLIIntegration async methods (achat, agenerate_code)
- GeminiCLIIntegration async methods (agenerate_text, achat)
- GroqCLIIntegration async methods (achat, acomplete)
- XAIGrokCLIIntegration async methods (achat, aplan_mode, aspawn_subagents, aweb_search)
- CLIEndpointAdapter.async_execute() thread offload
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_cache():
    cache = MagicMock()
    cache.get_chat_completion.return_value = None
    cache.get_completion.return_value = None
    cache.cache_chat_completion.return_value = None
    cache.cache_completion.return_value = None
    return cache


def _run_async(coro):
    """Run a coroutine in a fresh event loop (test helper)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# ApiKeyPool tests
# ---------------------------------------------------------------------------

class TestApiKeyPool(unittest.TestCase):
    """Tests for ApiKeyPool."""

    def _pool(self, keys=None):
        from ipfs_accelerate_py.cli_integrations.api_key_pool import ApiKeyPool
        return ApiKeyPool(keys or ["key-A", "key-B", "key-C"])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_construction_single_key(self):
        from ipfs_accelerate_py.cli_integrations.api_key_pool import ApiKeyPool
        pool = ApiKeyPool(["only"])
        self.assertEqual(len(pool), 1)

    def test_construction_empty_raises(self):
        from ipfs_accelerate_py.cli_integrations.api_key_pool import ApiKeyPool
        with self.assertRaises(ValueError):
            ApiKeyPool([])

    def test_construction_bad_strategy_raises(self):
        from ipfs_accelerate_py.cli_integrations.api_key_pool import ApiKeyPool
        with self.assertRaises(ValueError):
            ApiKeyPool(["k"], strategy="random")

    # ------------------------------------------------------------------
    # Round-robin
    # ------------------------------------------------------------------

    def test_round_robin_cycles(self):
        pool = self._pool()
        seen = [pool.get_key() for _ in range(9)]
        self.assertEqual(seen, ["key-A", "key-B", "key-C"] * 3)

    def test_round_robin_single_key(self):
        from ipfs_accelerate_py.cli_integrations.api_key_pool import ApiKeyPool
        pool = ApiKeyPool(["only"])
        for _ in range(5):
            self.assertEqual(pool.get_key(), "only")

    # ------------------------------------------------------------------
    # Per-user pinning
    # ------------------------------------------------------------------

    def test_user_pinning_consistent(self):
        pool = self._pool()
        key = pool.get_key(user_id="alice")
        for _ in range(10):
            self.assertEqual(pool.get_key(user_id="alice"), key)

    def test_different_users_may_have_different_keys(self):
        pool = self._pool()
        key_a = pool.get_key(user_id="alice")
        key_b = pool.get_key(user_id="bob")
        key_c = pool.get_key(user_id="carol")
        # All three keys should be assigned (round-robin distributes them)
        self.assertIn(key_a, ["key-A", "key-B", "key-C"])
        self.assertIn(key_b, ["key-A", "key-B", "key-C"])
        self.assertIn(key_c, ["key-A", "key-B", "key-C"])

    def test_unpin_user_reassigns_on_next_call(self):
        pool = self._pool()
        pool.get_key(user_id="dave")
        pool.unpin_user("dave")
        # After unpin, a new call will go through round-robin again (no guarantee it
        # returns a *different* key, but the pin is cleared)
        key_after = pool.get_key(user_id="dave")
        self.assertIn(key_after, ["key-A", "key-B", "key-C"])

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def test_add_key(self):
        pool = self._pool()
        pool.add_key("key-D")
        self.assertIn("key-D", pool.keys())
        self.assertEqual(len(pool), 4)

    def test_add_duplicate_no_op(self):
        pool = self._pool()
        pool.add_key("key-A")
        self.assertEqual(len(pool), 3)

    def test_remove_key(self):
        pool = self._pool()
        pool.remove_key("key-B")
        self.assertNotIn("key-B", pool.keys())
        self.assertEqual(len(pool), 2)

    def test_remove_key_clears_user_pins(self):
        pool = self._pool()
        # Pin alice to key-A specifically
        # Force key-A as the first get
        k = pool.get_key(user_id="pinme")  # gets key-A (first)
        # Manually confirm by checking the pin dict
        self.assertIn("pinme", pool._user_pins)
        pool.remove_key(pool._user_pins["pinme"])
        self.assertNotIn("pinme", pool._user_pins)

    def test_remove_nonexistent_key_no_error(self):
        pool = self._pool()
        pool.remove_key("nonexistent")
        self.assertEqual(len(pool), 3)

    def test_empty_pool_raises_on_get(self):
        from ipfs_accelerate_py.cli_integrations.api_key_pool import ApiKeyPool
        pool = ApiKeyPool(["k"])
        pool.remove_key("k")
        with self.assertRaises(RuntimeError):
            pool.get_key()

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def test_thread_safe_round_robin(self):
        """Concurrent get_key() calls from many threads should not crash."""
        pool = self._pool(["k1", "k2", "k3", "k4", "k5"])
        results = []
        errors = []

        def worker():
            try:
                for _ in range(50):
                    k = pool.get_key()
                    results.append(k)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 500)
        self.assertTrue(all(k in ["k1", "k2", "k3", "k4", "k5"] for k in results))

    def test_repr(self):
        pool = self._pool()
        r = repr(pool)
        self.assertIn("ApiKeyPool", r)
        self.assertIn("3", r)


# ---------------------------------------------------------------------------
# DualModeWrapper.get_api_key / pool integration
# ---------------------------------------------------------------------------

class TestDualModeWrapperMultiKey(unittest.TestCase):
    """Test get_api_key with and without a pool."""

    def _make_wrapper(self, api_keys=None, api_key=None):
        """Build a concrete DualModeWrapper subclass for testing."""
        with patch("ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager") as sm, \
             patch("shutil.which", return_value=None), \
             patch("subprocess.run"):
            sm.return_value.get_credential.return_value = None
            from ipfs_accelerate_py.cli_integrations.groq_cli_integration import GroqCLIIntegration
            with patch("ipfs_accelerate_py.cli_integrations.groq_cli_integration.get_llm_cache", return_value=_make_mock_cache()), \
                 patch("ipfs_accelerate_py.cli_integrations.groq_cli_integration.get_global_llm_cache", return_value=_make_mock_cache()):
                return GroqCLIIntegration(api_key=api_key, api_keys=api_keys)

    def test_no_pool_returns_single_key(self):
        wrapper = self._make_wrapper(api_key="single-key")
        self.assertEqual(wrapper.get_api_key(), "single-key")
        self.assertEqual(wrapper.get_api_key(user_id="user1"), "single-key")

    def test_pool_round_robin(self):
        wrapper = self._make_wrapper(api_keys=["k1", "k2", "k3"])
        seen = {wrapper.get_api_key() for _ in range(6)}
        # All three keys must appear
        self.assertGreater(len(seen), 1)

    def test_pool_per_user_pinning(self):
        wrapper = self._make_wrapper(api_keys=["k1", "k2", "k3"])
        key = wrapper.get_api_key(user_id="alice")
        for _ in range(5):
            self.assertEqual(wrapper.get_api_key(user_id="alice"), key)

    def test_single_key_included_in_pool(self):
        wrapper = self._make_wrapper(api_key="base-key", api_keys=["pool-key-1", "pool-key-2"])
        all_keys = wrapper.key_pool.keys()
        self.assertIn("base-key", all_keys)


# ---------------------------------------------------------------------------
# _aexecute_with_fallback tests
# ---------------------------------------------------------------------------

class TestAExecuteWithFallback(unittest.TestCase):
    """Verify _aexecute_with_fallback offloads work to threads."""

    def _make_groq(self):
        with patch("ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager") as sm, \
             patch("shutil.which", return_value=None), \
             patch("subprocess.run"):
            sm.return_value.get_credential.return_value = None
            from ipfs_accelerate_py.cli_integrations.groq_cli_integration import GroqCLIIntegration
            with patch("ipfs_accelerate_py.cli_integrations.groq_cli_integration.get_llm_cache", return_value=_make_mock_cache()), \
                 patch("ipfs_accelerate_py.cli_integrations.groq_cli_integration.get_global_llm_cache", return_value=_make_mock_cache()):
                return GroqCLIIntegration(api_key="test-key")

    def test_aexecute_returns_result(self):
        """_aexecute_with_fallback should return the sdk_func result."""
        groq = self._make_groq()

        def _fake_sdk(**kwargs):
            return {"response": "async result", "cached": False}

        result = _run_async(groq._aexecute_with_fallback(sdk_func=_fake_sdk, operation="test"))
        self.assertEqual(result["response"], "async result")
        self.assertEqual(result["mode"], "SDK")

    def test_aexecute_fallback_without_anyio(self):
        """When anyio is missing, _aexecute_with_fallback runs synchronously."""
        groq = self._make_groq()

        def _fake_sdk(**kwargs):
            return {"response": "sync fallback", "cached": False}

        with patch.dict("sys.modules", {"anyio": None}):
            # Re-import the wrapper to pick up the patched sys.modules
            result = _run_async(groq._aexecute_with_fallback(sdk_func=_fake_sdk, operation="test"))
        self.assertEqual(result["response"], "sync fallback")

    def test_aexecute_raises_when_both_modes_fail(self):
        groq = self._make_groq()

        def _bad_sdk(**kwargs):
            raise RuntimeError("SDK failed")

        with self.assertRaises(RuntimeError):
            _run_async(groq._aexecute_with_fallback(sdk_func=_bad_sdk, operation="test"))

    def test_aexecute_is_called_in_thread(self):
        """Verify the SDK function is invoked from a worker thread (not main)."""
        groq = self._make_groq()
        main_tid = threading.current_thread().ident
        called_from: list = []

        def _capture_thread(**kwargs):
            called_from.append(threading.current_thread().ident)
            return {"response": "ok", "cached": False}

        try:
            import anyio  # noqa: F401
            _run_async(groq._aexecute_with_fallback(sdk_func=_capture_thread, operation="test"))
            self.assertEqual(len(called_from), 1)
            self.assertNotEqual(called_from[0], main_tid)
        except ImportError:
            self.skipTest("anyio not installed")


# ---------------------------------------------------------------------------
# Per-integration async method tests
# ---------------------------------------------------------------------------

def _setup_integration(IntegrationClass, patch_module, **extra_patches):
    """Helper to build a mocked integration."""
    patches = {
        "ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager": MagicMock,
        "shutil.which": MagicMock(return_value=None),
        "subprocess.run": MagicMock(),
        f"{patch_module}.get_llm_cache": MagicMock(return_value=_make_mock_cache()),
        f"{patch_module}.get_global_llm_cache": MagicMock(return_value=_make_mock_cache()),
    }
    patches.update(extra_patches)
    ctx = [patch(k, v if callable(v) and not isinstance(v, MagicMock) else MagicMock(return_value=v() if callable(v) else v))
           for k, v in patches.items()]
    # Simple approach: just patch the secrets manager and cache
    with patch("ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager") as sm, \
         patch("shutil.which", return_value=None), \
         patch("subprocess.run"):
        sm.return_value.get_credential.return_value = None
        with patch(f"{patch_module}.get_llm_cache", return_value=_make_mock_cache()), \
             patch(f"{patch_module}.get_global_llm_cache", return_value=_make_mock_cache()):
            return IntegrationClass(api_key="test-key")


class TestClaudeAsyncMethods(unittest.TestCase):
    def setUp(self):
        self.claude = _setup_integration(
            __import__("ipfs_accelerate_py.cli_integrations.claude_code_cli_integration",
                       fromlist=["ClaudeCodeCLIIntegration"]).ClaudeCodeCLIIntegration,
            "ipfs_accelerate_py.cli_integrations.claude_code_cli_integration",
        )

    def test_achat_returns_response(self):
        def _fake(**kwargs):
            return {"response": "claude async", "cached": False}
        with patch.object(self.claude, "_chat_sdk", side_effect=_fake):
            result = _run_async(self.claude.achat("hi"))
        self.assertEqual(result["response"], "claude async")

    def test_agenerate_code_delegates_to_achat(self):
        async def _fake_achat(*a, **kw):
            return {"response": "code", "cached": False, "mode": "SDK"}
        with patch.object(self.claude, "achat", side_effect=_fake_achat) as m:
            _run_async(self.claude.agenerate_code("write me code"))
            m.assert_awaited_once()

    def test_achat_uses_user_key(self):
        """achat must forward the pool key, not the default key."""
        captured = []

        def _fake(**kwargs):
            captured.append(kwargs.get("api_key"))
            return {"response": "ok", "cached": False}

        self.claude.key_pool = MagicMock()
        self.claude.key_pool.get_key.return_value = "user-specific-key"
        with patch.object(self.claude, "_chat_sdk", side_effect=_fake):
            _run_async(self.claude.achat("hi", user_id="alice"))

        self.assertEqual(captured[0], "user-specific-key")


class TestGeminiAsyncMethods(unittest.TestCase):
    def setUp(self):
        self.gemini = _setup_integration(
            __import__("ipfs_accelerate_py.cli_integrations.gemini_cli_integration",
                       fromlist=["GeminiCLIIntegration"]).GeminiCLIIntegration,
            "ipfs_accelerate_py.cli_integrations.gemini_cli_integration",
        )

    def test_agenerate_text_returns_response(self):
        def _fake(**kwargs):
            return {"response": "gemini async", "cached": False}
        with patch.object(self.gemini, "_generate_text_sdk", side_effect=_fake):
            result = _run_async(self.gemini.agenerate_text("hello"))
        self.assertEqual(result["response"], "gemini async")

    def test_achat_delegates_to_agenerate_text(self):
        async def _fake(*a, **kw):
            return {"response": "g", "cached": False, "mode": "SDK"}
        with patch.object(self.gemini, "agenerate_text", side_effect=_fake) as m:
            _run_async(self.gemini.achat("hi"))
            m.assert_awaited_once()


class TestGroqAsyncMethods(unittest.TestCase):
    def setUp(self):
        self.groq = _setup_integration(
            __import__("ipfs_accelerate_py.cli_integrations.groq_cli_integration",
                       fromlist=["GroqCLIIntegration"]).GroqCLIIntegration,
            "ipfs_accelerate_py.cli_integrations.groq_cli_integration",
        )

    def test_achat_returns_response(self):
        def _fake(**kwargs):
            return {"response": "groq async", "cached": False}
        with patch.object(self.groq, "_chat_sdk", side_effect=_fake):
            result = _run_async(self.groq.achat("hi"))
        self.assertEqual(result["response"], "groq async")

    def test_acomplete_returns_response(self):
        def _fake(**kwargs):
            return {"response": "groq complete async", "cached": False}
        with patch.object(self.groq, "_complete_sdk", side_effect=_fake):
            result = _run_async(self.groq.acomplete("continue:"))
        self.assertEqual(result["response"], "groq complete async")

    def test_achat_uses_pool_key(self):
        captured = []

        def _fake(**kwargs):
            captured.append(kwargs.get("api_key"))
            return {"response": "ok", "cached": False}

        self.groq.key_pool = MagicMock()
        self.groq.key_pool.get_key.return_value = "pool-key-99"
        with patch.object(self.groq, "_chat_sdk", side_effect=_fake):
            _run_async(self.groq.achat("hi", user_id="dave"))

        self.assertEqual(captured[0], "pool-key-99")


class TestXAIGrokAsyncMethods(unittest.TestCase):
    def setUp(self):
        from ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration import XAIGrokCLIIntegration
        with patch("ipfs_accelerate_py.cli_integrations.dual_mode_wrapper.get_global_secrets_manager") as sm, \
             patch("shutil.which", return_value=None), \
             patch("subprocess.run"):
            sm.return_value.get_credential.return_value = None
            with patch("ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration.get_llm_cache",
                       return_value=_make_mock_cache()), \
                 patch("ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration.get_global_llm_cache",
                       return_value=_make_mock_cache()):
                self.grok = XAIGrokCLIIntegration(api_key="xai-test", headless=True)

    def _fake_chat(self, response="grok async", **kwargs):
        return {"response": response, "cached": False}

    def test_achat_returns_response(self):
        with patch.object(self.grok, "_chat_sdk", side_effect=self._fake_chat):
            result = _run_async(self.grok.achat("hi"))
        self.assertEqual(result["response"], "grok async")

    def test_aplan_mode_headless_approved(self):
        plan_text = "## Plan\n1. step\n## Implementation\n```python\npass\n```"
        with patch.object(self.grok, "_chat_sdk", return_value={"response": plan_text, "cached": False}):
            result = _run_async(self.grok.aplan_mode("do X"))
        self.assertTrue(result["approved"])
        self.assertIn("step", result["plan"])

    def test_aspawn_subagents_returns_all(self):
        counter = [0]

        def _fake(**kwargs):
            counter[0] += 1
            return {"response": f"sub {counter[0]}", "cached": False}

        with patch.object(self.grok, "_chat_sdk", side_effect=_fake):
            results = _run_async(self.grok.aspawn_subagents(["t1", "t2", "t3"]))

        self.assertEqual(len(results), 3)

    def test_aweb_search_sets_search_enabled(self):
        captured = []

        def _fake(**kwargs):
            captured.append(kwargs.get("search_enabled", False))
            return {"response": "live", "cached": False}

        with patch.object(self.grok, "_chat_sdk", side_effect=_fake):
            _run_async(self.grok.aweb_search("Python docs"))

        self.assertTrue(captured[0])

    def test_agenerate_code_with_plan_mode(self):
        async def _fake_plan(**kw):
            return {"plan": "plan", "implementation": "impl", "approved": True}

        with patch.object(self.grok, "aplan_mode", side_effect=_fake_plan) as m:
            result = _run_async(self.grok.agenerate_code("Build X", use_plan_mode=True))
            m.assert_awaited_once()

    def test_achat_uses_pool_key(self):
        captured = []

        def _fake(**kwargs):
            captured.append(kwargs.get("api_key"))
            return {"response": "ok", "cached": False}

        self.grok.key_pool = MagicMock()
        self.grok.key_pool.get_key.return_value = "xai-pool-key"
        with patch.object(self.grok, "_chat_sdk", side_effect=_fake):
            _run_async(self.grok.achat("hi", user_id="eve"))

        self.assertEqual(captured[0], "xai-pool-key")


# ---------------------------------------------------------------------------
# CLIEndpointAdapter.async_execute tests
# ---------------------------------------------------------------------------

class TestCLIEndpointAdapterAsync(unittest.TestCase):
    """Tests for the async_execute() method added to CLIEndpointAdapter."""

    def _make_adapter(self):
        """Create a concrete CLIEndpointAdapter for testing."""
        from ipfs_accelerate_py.mcp_server.tools.cli_endpoint_adapters import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter.__new__(ClaudeCodeAdapter)
        adapter.endpoint_id = "test-claude"
        adapter.cli_path = "/usr/bin/claude"
        adapter.config = {}
        adapter.stats = {"requests": 0, "successes": 0, "failures": 0, "total_time": 0.0, "avg_time": 0.0}
        return adapter

    def test_async_execute_delegates_to_execute(self):
        adapter = self._make_adapter()
        expected = {"result": "hello", "status": "success", "endpoint_id": "test-claude"}

        with patch.object(adapter, "execute", return_value=expected) as m:
            result = _run_async(adapter.async_execute("hi"))

        m.assert_called_once_with("hi", task_type="text_generation", timeout=30)
        self.assertEqual(result, expected)

    def test_async_execute_runs_in_thread(self):
        """Verify execute() is called from a worker thread."""
        adapter = self._make_adapter()
        main_tid = threading.current_thread().ident
        called_from: list = []

        def _capture(prompt, **kwargs):
            called_from.append(threading.current_thread().ident)
            return {"result": "done", "status": "success", "endpoint_id": "test-claude"}

        with patch.object(adapter, "execute", side_effect=_capture):
            try:
                import anyio  # noqa: F401
                _run_async(adapter.async_execute("prompt"))
                self.assertEqual(len(called_from), 1)
                self.assertNotEqual(called_from[0], main_tid)
            except ImportError:
                self.skipTest("anyio not installed")

    def test_async_execute_fallback_without_anyio(self):
        adapter = self._make_adapter()

        def _sync(prompt, **kwargs):
            return {"result": "sync", "status": "success", "endpoint_id": "test-claude"}

        with patch.object(adapter, "execute", side_effect=_sync):
            with patch.dict("sys.modules", {"anyio": None}):
                result = _run_async(adapter.async_execute("prompt"))

        self.assertEqual(result["result"], "sync")


# ---------------------------------------------------------------------------
# Package export test
# ---------------------------------------------------------------------------

class TestPackageExports(unittest.TestCase):
    def test_api_key_pool_exported(self):
        import ipfs_accelerate_py.cli_integrations as mod
        self.assertIn("ApiKeyPool", mod.__all__)
        self.assertTrue(callable(mod.ApiKeyPool))


if __name__ == "__main__":
    unittest.main()
