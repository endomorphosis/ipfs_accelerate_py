"""Contracts for the hosted Meta Model API integration."""

from __future__ import annotations

import json
from pathlib import Path


_META_ENV_NAMES = (
    "MODEL_API_KEY",
    "META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_BASE_URL",
    "ipfs_accelerate_py_META_AI_MODEL",
    "IPFS_ACCELERATE_PY_DISABLE_SECRET_MANAGER",
)


def _clear_meta_environment(monkeypatch) -> None:
    for name in _META_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class _FakeSecretsManager:
    def __init__(self, value: str | None):
        self.value = value
        self.requested: list[str] = []

    def get_credential(self, name: str):
        self.requested.append(name)
        return self.value if name == "meta_ai_api_key" else None


def test_current_endpoint_model_and_legacy_aliases():
    from ipfs_accelerate_py.common.meta_model_api import (
        META_MODEL_API_BASE_URL,
        META_MODEL_API_DEFAULT_MODEL,
        normalize_meta_model_name,
    )

    assert META_MODEL_API_BASE_URL == "https://api.meta.ai/v1"
    assert META_MODEL_API_DEFAULT_MODEL == "muse-spark-1.1"
    assert normalize_meta_model_name(None) == "muse-spark-1.1"
    assert normalize_meta_model_name("meta-spark/Spark-1.1") == "muse-spark-1.1"
    assert normalize_meta_model_name("custom-model") == "custom-model"


def test_credential_precedence_is_explicit_then_environment_then_encrypted_store(
    monkeypatch,
):
    from ipfs_accelerate_py.common.meta_model_api import resolve_meta_model_api_key

    _clear_meta_environment(monkeypatch)
    manager = _FakeSecretsManager("stored-value")

    monkeypatch.setenv("MODEL_API_KEY", "environment-value")
    assert (
        resolve_meta_model_api_key("explicit-value", secrets_manager=manager)
        == "explicit-value"
    )
    assert (
        resolve_meta_model_api_key(secrets_manager=manager)
        == "environment-value"
    )
    assert manager.requested == []

    monkeypatch.delenv("MODEL_API_KEY")
    assert resolve_meta_model_api_key(secrets_manager=manager) == "stored-value"
    assert manager.requested == ["meta_ai_api_key"]


def test_secret_manager_can_be_disabled_for_isolated_processes(monkeypatch):
    from ipfs_accelerate_py.common.meta_model_api import resolve_meta_model_api_key

    _clear_meta_environment(monkeypatch)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_DISABLE_SECRET_MANAGER", "1")
    manager = _FakeSecretsManager("stored-value")

    assert resolve_meta_model_api_key(secrets_manager=manager) is None
    assert manager.requested == []


def test_cache_fingerprint_does_not_contain_secret(monkeypatch):
    import ipfs_accelerate_py.common.meta_model_api as meta_contract

    secret = "synthetic-meta-secret"
    monkeypatch.setattr(
        meta_contract,
        "resolve_meta_model_api_key",
        lambda: secret,
    )
    fingerprint = meta_contract.meta_model_api_key_fingerprint()

    assert fingerprint
    assert secret not in fingerprint
    assert len(fingerprint) == 16


def test_llm_router_uses_encrypted_secret_and_current_wire_contract(monkeypatch):
    import ipfs_accelerate_py.common.secrets_manager as secrets_module
    from ipfs_accelerate_py.llm_router import _get_meta_ai_provider

    _clear_meta_environment(monkeypatch)
    manager = _FakeSecretsManager("synthetic-meta-secret")
    monkeypatch.setattr(
        secrets_module,
        "get_global_secrets_manager",
        lambda: manager,
    )

    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "muse-spark-1.1",
                    "choices": [{"message": {"content": "OK"}}],
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider = _get_meta_ai_provider()
    assert provider is not None
    assert (
        provider.generate(
            "Reply with OK",
            model_name="meta-spark/Spark-1.1",
            max_tokens=512,
            timeout=17,
        )
        == "OK"
    )
    assert captured["url"] == "https://api.meta.ai/v1/chat/completions"
    assert captured["authorization"] == "Bearer synthetic-meta-secret"
    assert captured["timeout"] == 17
    assert captured["payload"]["model"] == "muse-spark-1.1"
    assert captured["payload"]["max_completion_tokens"] == 512
    assert "max_tokens" not in captured["payload"]


def test_api_backend_uses_secret_manager_without_persisting_remote_copy(
    monkeypatch,
    tmp_path: Path,
):
    import ipfs_accelerate_py.common.meta_model_api as meta_contract
    from ipfs_accelerate_py.api_backends.meta_ai import meta_ai
    from ipfs_accelerate_py.common.secrets_manager import SecretsManager

    _clear_meta_environment(monkeypatch)
    manager = SecretsManager(secrets_file=str(tmp_path / "secrets.enc"))
    assert manager.storage is None
    manager.set_credential("meta_ai_api_key", "synthetic-meta-secret")

    monkeypatch.setattr(
        meta_contract,
        "resolve_meta_model_api_key",
        lambda *args, **kwargs: "synthetic-meta-secret",
    )
    client = meta_ai(metadata={"api_key": "synthetic-meta-secret"})

    assert client.api_key == "synthetic-meta-secret"
    assert client.base_url == "https://api.meta.ai/v1"
    assert client.default_model == "muse-spark-1.1"
    assert b"synthetic-meta-secret" not in (tmp_path / "secrets.enc").read_bytes()
