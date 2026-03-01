#!/usr/bin/env python3
"""Tests for canonical MCP server secrets vault."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ipfs_accelerate_py.mcp_server.secrets_vault import SecretsVault


class TestSecretsVault(unittest.TestCase):
    """Validate deterministic vault behavior with explicit master key material."""

    def test_set_get_delete_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault_file = Path(tmp) / "vault.json"
            vault = SecretsVault(
                vault_file=vault_file,
                master_key_b64url="MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
            )

            vault.set("OPENAI_API_KEY", "sk-test-123")
            self.assertEqual(vault.get("OPENAI_API_KEY"), "sk-test-123")
            self.assertIn("OPENAI_API_KEY", vault)

            self.assertTrue(vault.delete("OPENAI_API_KEY"))
            self.assertIsNone(vault.get("OPENAI_API_KEY"))
            self.assertFalse(vault.delete("OPENAI_API_KEY"))

    def test_load_into_env_obeys_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault_file = Path(tmp) / "vault.json"
            vault = SecretsVault(
                vault_file=vault_file,
                master_key_b64url="MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
            )
            vault.set("EXAMPLE_SECRET", "v1")

            os.environ.pop("EXAMPLE_SECRET", None)
            loaded = vault.load_into_env(overwrite=False)
            self.assertEqual(loaded, ["EXAMPLE_SECRET"])
            self.assertEqual(os.environ.get("EXAMPLE_SECRET"), "v1")

            vault.set("EXAMPLE_SECRET", "v2")
            loaded_again = vault.load_into_env(overwrite=False)
            self.assertEqual(loaded_again, [])
            self.assertEqual(os.environ.get("EXAMPLE_SECRET"), "v1")

            loaded_force = vault.load_into_env(overwrite=True)
            self.assertEqual(loaded_force, ["EXAMPLE_SECRET"])
            self.assertEqual(os.environ.get("EXAMPLE_SECRET"), "v2")

            os.environ.pop("EXAMPLE_SECRET", None)

    def test_set_raises_without_master_key_material(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault_file = Path(tmp) / "vault.json"
            prev_key = os.environ.pop("IPFS_MCP_SERVER_SECRETS_MASTER_KEY", None)
            try:
                vault = SecretsVault(vault_file=vault_file)
                with self.assertRaises(RuntimeError):
                    vault.set("MISSING_KEY_SECRET", "value")
            finally:
                if prev_key is not None:
                    os.environ["IPFS_MCP_SERVER_SECRETS_MASTER_KEY"] = prev_key

    def test_get_returns_none_when_key_missing_for_existing_ciphertext(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault_file = Path(tmp) / "vault.json"
            seeded = SecretsVault(
                vault_file=vault_file,
                master_key_b64url="MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
            )
            seeded.set("SEALED_SECRET", "opaque")

            prev_key = os.environ.pop("IPFS_MCP_SERVER_SECRETS_MASTER_KEY", None)
            try:
                reopened = SecretsVault(vault_file=vault_file)
                self.assertIsNone(reopened.get("SEALED_SECRET"))
            finally:
                if prev_key is not None:
                    os.environ["IPFS_MCP_SERVER_SECRETS_MASTER_KEY"] = prev_key

    def test_corrupt_vault_file_recovers_to_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault_file = Path(tmp) / "vault.json"
            vault_file.write_text("{this-is-not-json", encoding="utf-8")

            vault = SecretsVault(
                vault_file=vault_file,
                master_key_b64url="MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
            )

            self.assertEqual(vault.list_names(), [])
            self.assertEqual(len(vault), 0)
            self.assertEqual(vault.info().get("secret_count"), 0)


if __name__ == "__main__":
    unittest.main()
