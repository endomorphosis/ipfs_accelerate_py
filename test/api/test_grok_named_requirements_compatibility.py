"""Exercise the named-session adapter against private synthetic policies."""
from pathlib import Path
import tempfile
import unittest

from ipfs_accelerate_py import llm_router as router


class GrokNamedRequirementsCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.policy = Path(temporary.name) / "requirements.toml"

    def write_policy(self, text):
        self.policy.write_text(text, encoding="utf-8")
        self.policy.chmod(0o600)

    def test_real_zero_policy_and_absent_optional_policy_are_allowed(self):
        self.assertIsNone(router._grok_requirements_allow_zero(self.policy, "grok-4.7"))
        self.write_policy('[cli]\nuse_leader = false\n[model."grok-4.7"]\ntemperature = 0.0\n')
        self.assertIsNone(router._grok_requirements_allow_zero(self.policy, "grok-4.7"))

    def test_real_selected_model_sampling_conflicts_are_refused(self):
        for value in ("0.5", "1", "true", '\"0\"', "nan", "inf"):
            with self.subTest(temperature=value):
                self.write_policy('[model."grok-4.7"]\ntemperature = ' + value + '\n')
                with self.assertRaises(ValueError):
                    router._grok_requirements_allow_zero(self.policy, "grok-4.7")

    def test_real_requirements_layers_are_never_silently_bypassed(self):
        policies = {
            "malformed_toml": '[model."grok-4.7"\n',
            "leader_required": '[cli]\nuse_leader = true\n',
            "invalid_cli_table": 'cli = []\n',
            "invalid_model_table": 'model = []\n',
            "invalid_selected_model_table": '[model]\n"grok-4.7" = []\n',
            "unqualified_experiment": 'experiments = ["unqualified"]\n',
            "unqualified_version_override": 'version_overrides = { unqualified = true }\n',
            "unqualified_campaign": 'campaigns = ["unqualified"]\n',
        }
        for name, policy in policies.items():
            with self.subTest(policy=name):
                self.write_policy(policy)
                with self.assertRaises(ValueError):
                    router._grok_requirements_allow_zero(self.policy, "grok-4.7")


if __name__ == "__main__":
    unittest.main()
