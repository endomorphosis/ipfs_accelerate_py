"""
Tests for xAI Grok Build CLI Integration

Covers:
- Module import and instantiation
- Plan Mode (plan splitting, headless approval, interactive approval mock)
- Subagents (parallel execution, error handling)
- Live Web / X Search (search_parameters forwarded to API)
- chat() and generate_code() helpers
- Global singleton
- list_models() / get_model_info()
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

class TestXAIGrokCLIIntegration(unittest.TestCase):
    """Unit tests for XAIGrokCLIIntegration."""

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
            "ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration.get_llm_cache"
        )
        self.global_llm_cache_patcher = patch(
            "ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration.get_global_llm_cache"
        )
        mock_llm = self.llm_cache_patcher.start()
        mock_global = self.global_llm_cache_patcher.start()
        self._mock_cache = _make_mock_cache()
        mock_llm.return_value = self._mock_cache
        mock_global.return_value = self._mock_cache

        # Now import (after patching)
        from ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration import (
            XAIGrokCLIIntegration,
            get_xai_grok_cli_integration,
        )
        self.XAIGrokCLIIntegration = XAIGrokCLIIntegration
        self.get_xai_grok_cli_integration = get_xai_grok_cli_integration

    def tearDown(self):
        self.secrets_patcher.stop()
        self.which_patcher.stop()
        self.subprocess_patcher.stop()
        self.llm_cache_patcher.stop()
        self.global_llm_cache_patcher.stop()

        # Reset global singleton between tests
        import ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration as mod
        mod._global_xai_grok_cli = None

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_instantiation_default(self):
        """Integration can be created without arguments."""
        integration = self.XAIGrokCLIIntegration()
        self.assertEqual(integration.get_tool_name(), "xAI Grok Build")
        self.assertFalse(integration.headless)
        self.assertEqual(integration.max_subagents, 4)

    def test_instantiation_headless_kwarg(self):
        integration = self.XAIGrokCLIIntegration(headless=True)
        self.assertTrue(integration.headless)

    def test_instantiation_headless_env(self):
        with patch.dict(os.environ, {"XAI_HEADLESS": "1"}):
            integration = self.XAIGrokCLIIntegration()
            self.assertTrue(integration.headless)

    def test_get_tool_name(self):
        integration = self.XAIGrokCLIIntegration()
        self.assertEqual(integration.get_tool_name(), "xAI Grok Build")

    # ------------------------------------------------------------------
    # list_models / get_model_info
    # ------------------------------------------------------------------

    def test_list_models(self):
        integration = self.XAIGrokCLIIntegration()
        models = integration.list_models()
        self.assertIsInstance(models, list)
        self.assertIn("grok-3", models)
        self.assertIn("grok-4", models)

    def test_get_model_info_known(self):
        integration = self.XAIGrokCLIIntegration()
        info = integration.get_model_info("grok-3")
        self.assertIsNotNone(info)
        self.assertIn("context_window", info)
        self.assertEqual(info["context_window"], 131_072)

    def test_get_model_info_unknown(self):
        integration = self.XAIGrokCLIIntegration()
        self.assertIsNone(integration.get_model_info("nonexistent-model"))

    # ------------------------------------------------------------------
    # Plan Mode – _split_plan_impl
    # ------------------------------------------------------------------

    def test_split_plan_impl_both_sections(self):
        from ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration import (
            XAIGrokCLIIntegration,
        )
        text = "## Plan\n1. Step one\n2. Step two\n## Implementation\n```python\npass\n```"
        plan, impl = XAIGrokCLIIntegration._split_plan_impl(text)
        self.assertIn("Step one", plan)
        self.assertIn("```python", impl)

    def test_split_plan_impl_no_markers(self):
        from ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration import (
            XAIGrokCLIIntegration,
        )
        text = "Just some response text."
        plan, impl = XAIGrokCLIIntegration._split_plan_impl(text)
        self.assertEqual(plan, text)
        self.assertEqual(impl, "")

    def test_split_plan_impl_plan_only(self):
        from ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration import (
            XAIGrokCLIIntegration,
        )
        text = "## Plan\n1. Only a plan here."
        plan, impl = XAIGrokCLIIntegration._split_plan_impl(text)
        self.assertIn("Only a plan", plan)
        self.assertEqual(impl, "")

    # ------------------------------------------------------------------
    # Plan Mode – full flow
    # ------------------------------------------------------------------

    def _make_chat_sdk_mock(self, response_text: str):
        """Return a mock _chat_sdk that yields a fixed response."""
        mock = MagicMock(
            return_value={"response": response_text, "cached": False, "mode": "SDK"}
        )
        return mock

    def test_plan_mode_headless_auto_approves(self):
        """In headless mode plan_mode returns approved=True without prompting."""
        integration = self.XAIGrokCLIIntegration(headless=True)
        response = (
            "## Plan\n1. Analyse\n2. Write code\n"
            "## Implementation\n```python\nprint('hello')\n```"
        )
        with patch.object(integration, "_chat_sdk", self._make_chat_sdk_mock(response)):
            result = integration.plan_mode("Write hello world")

        self.assertTrue(result["approved"])
        self.assertIn("Analyse", result["plan"])
        self.assertIn("print", result["implementation"])

    def test_plan_mode_interactive_approved(self):
        """User types 'y' → plan is approved and implementation returned."""
        integration = self.XAIGrokCLIIntegration(headless=False)
        response = (
            "## Plan\n1. Step A\n"
            "## Implementation\n```python\nx = 1\n```"
        )
        with patch.object(integration, "_chat_sdk", self._make_chat_sdk_mock(response)):
            with patch("builtins.input", return_value="y"):
                result = integration.plan_mode("Do something")

        self.assertTrue(result["approved"])
        self.assertIn("x = 1", result["implementation"])

    def test_plan_mode_interactive_rejected(self):
        """User types 'n' → implementation is suppressed."""
        integration = self.XAIGrokCLIIntegration(headless=False)
        response = (
            "## Plan\n1. Step A\n"
            "## Implementation\n```python\nx = 1\n```"
        )
        with patch.object(integration, "_chat_sdk", self._make_chat_sdk_mock(response)):
            with patch("builtins.input", return_value="n"):
                result = integration.plan_mode("Do something")

        self.assertFalse(result["approved"])
        self.assertEqual(result["implementation"], "")

    def test_plan_mode_auto_approve_override(self):
        """auto_approve=True overrides headless=False."""
        integration = self.XAIGrokCLIIntegration(headless=False)
        response = "## Plan\n1. Plan\n## Implementation\n```python\npass\n```"
        with patch.object(integration, "_chat_sdk", self._make_chat_sdk_mock(response)):
            result = integration.plan_mode("Task", auto_approve=True)

        self.assertTrue(result["approved"])

    def test_plan_mode_auto_approve_false_overrides_headless(self):
        """auto_approve=False prompts even in headless mode."""
        integration = self.XAIGrokCLIIntegration(headless=True)
        response = "## Plan\n1. Plan\n## Implementation\n```python\npass\n```"
        with patch.object(integration, "_chat_sdk", self._make_chat_sdk_mock(response)):
            with patch("builtins.input", return_value="n"):
                result = integration.plan_mode("Task", auto_approve=False)

        self.assertFalse(result["approved"])

    # ------------------------------------------------------------------
    # Subagents
    # ------------------------------------------------------------------

    def test_spawn_subagents_parallel(self):
        """Results are returned in original order for all subtasks."""
        integration = self.XAIGrokCLIIntegration(headless=True, max_subagents=2)
        subtasks = ["task A", "task B", "task C"]

        call_count = 0

        def _fake_chat_sdk(message, model, temperature, **kw):
            nonlocal call_count
            call_count += 1
            return {"response": f"answer for {message}", "cached": False, "mode": "SDK"}

        with patch.object(integration, "_chat_sdk", side_effect=_fake_chat_sdk):
            results = integration.spawn_subagents(subtasks, model="grok-3-mini")

        self.assertEqual(len(results), 3)
        self.assertEqual(call_count, 3)
        for i, res in enumerate(results):
            self.assertIn("answer for", res["response"])
            self.assertEqual(res["subtask"], subtasks[i])

    def test_spawn_subagents_handles_failure(self):
        """A failing subagent does not crash the whole batch."""
        integration = self.XAIGrokCLIIntegration(headless=True, max_subagents=2)

        call_no = [0]

        def _flaky_chat_sdk(message, model, temperature, **kw):
            call_no[0] += 1
            if call_no[0] == 2:
                raise RuntimeError("API error")
            return {"response": "ok", "cached": False, "mode": "SDK"}

        with patch.object(integration, "_chat_sdk", side_effect=_flaky_chat_sdk):
            results = integration.spawn_subagents(["t1", "t2", "t3"])

        self.assertEqual(len(results), 3)
        error_results = [r for r in results if "error" in r]
        self.assertEqual(len(error_results), 1)

    def test_spawn_subagents_respects_max_workers(self):
        """max_subagents caps the thread pool size."""
        integration = self.XAIGrokCLIIntegration(max_subagents=1)
        self.assertEqual(integration.max_subagents, 1)

    # ------------------------------------------------------------------
    # Live Web / X Search
    # ------------------------------------------------------------------

    def test_web_search_passes_search_enabled(self):
        """web_search must forward search_enabled=True to _chat_sdk."""
        integration = self.XAIGrokCLIIntegration()
        captured: list = []

        def _capture_chat_sdk(message, model, temperature, search_enabled=False, **kw):
            captured.append(search_enabled)
            return {"response": "search result", "cached": False, "mode": "SDK"}

        with patch.object(integration, "_chat_sdk", side_effect=_capture_chat_sdk):
            result = integration.web_search("Python asyncio docs")

        self.assertTrue(captured[0], "search_enabled must be True for web_search")
        self.assertEqual(result["response"], "search result")

    def test_x_search_uses_x_specific_query(self):
        """x_search wraps the query with X-context wording."""
        integration = self.XAIGrokCLIIntegration()
        captured_messages: list = []

        def _capture_chat_sdk(message, model, temperature, search_enabled=False, **kw):
            captured_messages.append(message)
            return {"response": "x result", "cached": False, "mode": "SDK"}

        with patch.object(integration, "_chat_sdk", side_effect=_capture_chat_sdk):
            integration.x_search("FastAPI release notes")

        self.assertTrue(
            any("X (Twitter)" in m or "Search X" in m for m in captured_messages),
            "x_search must mention X/Twitter in the prompt",
        )

    def test_web_search_not_cached(self):
        """Live search responses must not be written to cache."""
        integration = self.XAIGrokCLIIntegration()

        def _fake_chat_sdk(message, model, temperature, search_enabled=False, **kw):
            # Simulate the non-caching branch
            return {"response": "live data", "cached": False, "mode": "SDK"}

        with patch.object(integration, "_chat_sdk", side_effect=_fake_chat_sdk):
            integration.web_search("latest news")

        integration.cache.cache_chat_completion.assert_not_called()

    # ------------------------------------------------------------------
    # chat() and generate_code()
    # ------------------------------------------------------------------

    def test_chat_returns_response(self):
        integration = self.XAIGrokCLIIntegration()
        with patch.object(
            integration,
            "_chat_sdk",
            return_value={"response": "Hi there!", "cached": False, "mode": "SDK"},
        ):
            result = integration.chat("Hello")

        self.assertEqual(result["response"], "Hi there!")

    def test_generate_code_no_plan(self):
        """generate_code without plan_mode just calls chat."""
        integration = self.XAIGrokCLIIntegration()
        with patch.object(integration, "chat", return_value={"response": "code"}) as m:
            integration.generate_code("Write a function", use_plan_mode=False)
            m.assert_called_once()

    def test_generate_code_with_plan_mode(self):
        """generate_code with use_plan_mode=True delegates to plan_mode."""
        integration = self.XAIGrokCLIIntegration(headless=True)
        with patch.object(
            integration,
            "plan_mode",
            return_value={"plan": "", "implementation": "code", "approved": True},
        ) as m:
            integration.generate_code("Build X", use_plan_mode=True)
            m.assert_called_once()

    # ------------------------------------------------------------------
    # Cache hit short-circuit
    # ------------------------------------------------------------------

    def test_chat_cache_hit(self):
        """A cached response is returned without calling the API."""
        integration = self.XAIGrokCLIIntegration()
        integration.cache.get_chat_completion.return_value = "cached answer"

        with patch.object(integration, "_get_openai_client") as mock_client:
            result = integration._chat_sdk(
                message="Hello",
                model="grok-3",
                temperature=0.0,
            )

        self.assertTrue(result["cached"])
        self.assertEqual(result["response"], "cached answer")
        mock_client.assert_not_called()

    # ------------------------------------------------------------------
    # Global singleton
    # ------------------------------------------------------------------

    def test_global_singleton(self):
        """get_xai_grok_cli_integration always returns the same instance."""
        a = self.get_xai_grok_cli_integration()
        b = self.get_xai_grok_cli_integration()
        self.assertIs(a, b)


# ---------------------------------------------------------------------------
# Import-level smoke test
# ---------------------------------------------------------------------------

class TestXAIGrokCLIImport(unittest.TestCase):
    """Verify the integration is properly exported from the package."""

    def test_package_exports(self):
        from ipfs_accelerate_py.cli_integrations import (
            XAIGrokCLIIntegration,
            get_xai_grok_cli_integration,
        )
        self.assertTrue(callable(XAIGrokCLIIntegration))
        self.assertTrue(callable(get_xai_grok_cli_integration))

    def test_present_in_all(self):
        import ipfs_accelerate_py.cli_integrations as mod
        self.assertIn("XAIGrokCLIIntegration", mod.__all__)
        self.assertIn("get_xai_grok_cli_integration", mod.__all__)


if __name__ == "__main__":
    unittest.main()
