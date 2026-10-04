import unittest
from live import MeteredProvider, generate


class LiveTests(unittest.TestCase):
    def test_router_supports_cli_without_fallback(self):
        calls = []
        def fake(prompt, **kwargs):
            calls.append(kwargs)
            return "{}"
        generate("repair", {"provider": "openai", "model": "test"}, {}, generate_fn=fake)
        self.assertFalse(calls[0]["allow_cross_provider_fallback"])
        self.assertFalse(calls[0]["allow_local_fallback"])
        self.assertEqual(calls[0]["allocation_path"], "api")
        generate("repair", {"provider": "codex_cli", "model": "test"}, {}, generate_fn=fake)
        self.assertEqual(calls[1]["allocation_path"], "cli")
        from uuid import UUID
        UUID(calls[1]["allocation_session_id"])
        self.assertFalse(calls[1]["allow_cross_provider_fallback"])
        self.assertEqual(len(calls), 2)

    def test_cli_usage_including_failure_and_stale_observation(self):
        from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
        def failure(prompt, **kwargs):
            set_last_cli_observation("grok_cli", {"prompt_tokens": 25, "completion_tokens": 5})
            raise RuntimeError("failure after dispatch")
        row = {}
        with self.assertRaises(RuntimeError):
            generate("x", {"provider": "grok_cli", "model": "test"}, row, generate_fn=failure)
        self.assertEqual(row["provider_tokens"], 30)
        row = {"provider_tokens": None}
        generate("x", {"provider": "grok_cli", "model": "test"}, row,
                 generate_fn=lambda *a, **k: "{}")
        self.assertIsNone(row["provider_tokens"])

    def test_discovery_preserves_authenticated_cli_routes(self):
        from unittest.mock import patch
        from live import choose_route
        with patch('ipfs_accelerate_py.llm_allocation.intelligence_index.discover_available_providers',
                   return_value=("codex_cli", "grok_cli", "mock")):
            for provider in ("codex_cli", "grok_cli"):
                route = choose_route(provider, "test-model")
                self.assertEqual(route["provider"], provider)
                self.assertEqual(route["allocation_path"], "cli")

    def test_native_usage_is_preserved_before_parse_failure(self):
        class Provider:
            def chat_completions(self, *args, **kwargs):
                return {"usage": {"prompt_tokens": 21, "completion_tokens": 7},
                        "choices": []}
        measurement = {}
        with self.assertRaises(IndexError):
            MeteredProvider(Provider(), measurement).generate("x")
        self.assertEqual(measurement["provider_tokens"], 28)

    def test_missing_usage_is_not_zero(self):
        class Provider:
            def chat_completions(self, *args, **kwargs):
                return {"choices": [{"message": {"content": "hello"}}]}
        measurement = {"provider_tokens": None}
        self.assertEqual(MeteredProvider(Provider(), measurement).generate("x"), "hello")
        self.assertIsNone(measurement["provider_tokens"])


if __name__ == "__main__":
    unittest.main()
