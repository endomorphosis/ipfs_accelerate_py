"""Secrets vault for canonical MCP runtime.

Stores encrypted secret values in a JSON file and can load decrypted values
into `os.environ` for runtime use.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_VAULT_DIR = Path.home() / ".ipfs_accelerate"
_DEFAULT_VAULT_FILE = _DEFAULT_VAULT_DIR / "secrets_vault.json"
_VAULT_FILE_ENV = "IPFS_MCP_SERVER_SECRETS_VAULT_FILE"
_MASTER_KEY_ENV = "IPFS_MCP_SERVER_SECRETS_MASTER_KEY"

_HKDF_SALT = b"ipfs-accelerate-secrets-v1"
_HKDF_INFO = b"encryption-key"


try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    _CRYPTO_AVAILABLE = True
except Exception:  # pragma: no cover - dependency guard
    AESGCM = None  # type: ignore[assignment]
    SHA256 = None  # type: ignore[assignment]
    HKDF = None  # type: ignore[assignment]
    _CRYPTO_AVAILABLE = False


def _b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64_decode(value: str) -> bytes:
    return base64.b64decode(str(value or ""))


def _decode_urlsafe_b64(value: str) -> bytes:
    text = str(value or "").strip()
    padding = "=" * ((4 - (len(text) % 4)) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _derive_enc_key(seed_b64url: str) -> bytes:
    """Derive 32-byte encryption key from a base64url seed string."""
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography package is required for SecretsVault")

    seed_bytes = _decode_urlsafe_b64(seed_b64url)
    kdf = HKDF(algorithm=SHA256(), length=32, salt=_HKDF_SALT, info=_HKDF_INFO)
    return kdf.derive(seed_bytes)


class SecretsVault:
    """Encrypt and persist named secrets for canonical runtime use.

    By default, encryption key material is resolved from one of:
    1. explicit `master_key_b64url` constructor argument
    2. `did_key_manager.export_secret_b64()` when provided
    3. `IPFS_MCP_SERVER_SECRETS_MASTER_KEY` env variable
    """

    def __init__(
        self,
        vault_file: Optional[Path] = None,
        *,
        did_key_manager: Any = None,
        master_key_b64url: Optional[str] = None,
    ) -> None:
        env_path = os.environ.get(_VAULT_FILE_ENV)
        self._vault_file = Path(vault_file) if vault_file is not None else Path(env_path) if env_path else _DEFAULT_VAULT_FILE
        self._did_key_manager = did_key_manager
        self._master_key_b64url = str(master_key_b64url or "").strip()
        self._enc_key: Optional[bytes] = None
        self._data: Dict[str, Any] = {}
        self._load_vault()

    def _resolve_seed_b64url(self) -> str:
        if self._master_key_b64url:
            return self._master_key_b64url

        mgr = self._did_key_manager
        if mgr is None:
            try:
                from .did_key_manager import get_did_key_manager  # type: ignore

                mgr = get_did_key_manager()
            except Exception:
                mgr = None

        if mgr is not None and hasattr(mgr, "export_secret_b64"):
            try:
                seed = str(mgr.export_secret_b64() or "").strip()
                if seed:
                    return seed
            except Exception:
                pass

        env_seed = str(os.environ.get(_MASTER_KEY_ENV, "") or "").strip()
        if env_seed:
            return env_seed

        raise RuntimeError(
            "No secrets master key available; set master_key_b64url or "
            "IPFS_MCP_SERVER_SECRETS_MASTER_KEY"
        )

    def _ensure_enc_key(self) -> bytes:
        if self._enc_key is not None:
            return self._enc_key
        seed_b64url = self._resolve_seed_b64url()
        self._enc_key = _derive_enc_key(seed_b64url)
        return self._enc_key

    def _load_vault(self) -> None:
        if not self._vault_file.exists():
            self._data = {"version": 1, "secrets": {}}
            return
        try:
            self._data = json.loads(self._vault_file.read_text(encoding="utf-8"))
            if not isinstance(self._data, dict):
                self._data = {"version": 1, "secrets": {}}
        except Exception:
            self._data = {"version": 1, "secrets": {}}

    def _save_vault(self) -> None:
        self._vault_file.parent.mkdir(parents=True, exist_ok=True)
        self._vault_file.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        try:
            os.chmod(self._vault_file, 0o600)
        except OSError:
            pass

    def set(self, name: str, value: str) -> None:
        if not _CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package is required for SecretsVault")
        key = str(name or "").strip()
        if not key:
            raise ValueError("Secret name must not be empty")
        if not value:
            raise ValueError("Secret value must not be empty")

        enc_key = self._ensure_enc_key()
        nonce = os.urandom(12)
        ciphertext = AESGCM(enc_key).encrypt(nonce, value.encode("utf-8"), None)
        secrets = self._data.setdefault("secrets", {})
        secrets[key] = {
            "nonce_b64": _b64_encode(nonce),
            "ciphertext_b64": _b64_encode(ciphertext),
        }
        self._save_vault()

    def get(self, name: str) -> Optional[str]:
        if not _CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package is required for SecretsVault")
        key = str(name or "").strip()
        if not key:
            return None

        secrets = self._data.get("secrets", {})
        if key not in secrets:
            return None

        try:
            enc_key = self._ensure_enc_key()
            entry = secrets[key]
            nonce = _b64_decode(entry.get("nonce_b64", ""))
            ciphertext = _b64_decode(entry.get("ciphertext_b64", ""))
            plaintext = AESGCM(enc_key).decrypt(nonce, ciphertext, None)
            return plaintext.decode("utf-8")
        except Exception as exc:
            logger.debug("Failed to decrypt secret %s: %s", key, exc)
            return None

    def delete(self, name: str) -> bool:
        key = str(name or "").strip()
        secrets = self._data.get("secrets", {})
        if key in secrets:
            del secrets[key]
            self._save_vault()
            return True
        return False

    def list_names(self) -> List[str]:
        secrets = self._data.get("secrets", {})
        if not isinstance(secrets, dict):
            return []
        return list(secrets.keys())

    def load_into_env(self, *, overwrite: bool = False) -> List[str]:
        loaded: List[str] = []
        for name in self.list_names():
            if not overwrite and name in os.environ:
                continue
            value = self.get(name)
            if value is None:
                continue
            os.environ[name] = value
            loaded.append(name)
        return loaded

    @property
    def vault_file(self) -> Path:
        return self._vault_file

    def info(self) -> Dict[str, Any]:
        return {
            "vault_file": str(self._vault_file),
            "secret_count": len(self),
            "secret_names": self.list_names(),
            "crypto_available": _CRYPTO_AVAILABLE,
        }

    def __contains__(self, name: str) -> bool:
        return str(name or "").strip() in self._data.get("secrets", {})

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_names())

    def __len__(self) -> int:
        return len(self.list_names())


_default_vault: Optional[SecretsVault] = None


def get_secrets_vault(vault_file: Optional[Path] = None) -> SecretsVault:
    """Return process singleton vault instance."""
    global _default_vault
    if _default_vault is None or vault_file is not None:
        _default_vault = SecretsVault(vault_file=vault_file)
    return _default_vault


def load_env_from_vault(*, overwrite: bool = False) -> List[str]:
    """Load all secrets from singleton vault into environment variables."""
    return get_secrets_vault().load_into_env(overwrite=overwrite)


__all__ = [
    "SecretsVault",
    "get_secrets_vault",
    "load_env_from_vault",
]
