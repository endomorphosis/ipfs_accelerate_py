"""DID key manager for canonical MCP runtime.

Provides persistent Ed25519 ``did:key`` generation/loading and optional UCAN
delegation mint/verify helpers when ``py-ucan`` is available.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import ucan as _ucan_lib  # type: ignore[import-not-found]

    _UCAN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _ucan_lib = None  # type: ignore[assignment]
    _UCAN_AVAILABLE = False


_DEFAULT_KEY_DIR = Path.home() / ".ipfs_accelerate"
_DEFAULT_KEY_FILE = _DEFAULT_KEY_DIR / "did_key.json"
_KEY_FILE_ENV = "IPFS_MCP_SERVER_DID_KEY_FILE"


class DIDKeyManager:
    """Manage persistent Ed25519 DID keys and optional UCAN delegation helpers."""

    def __init__(
        self,
        key_file: Optional[Path] = None,
        *,
        auto_load: bool = True,
    ) -> None:
        env_path = os.environ.get(_KEY_FILE_ENV)
        self._key_file = Path(key_file) if key_file is not None else Path(env_path) if env_path else _DEFAULT_KEY_FILE
        self._keypair: Optional[Any] = None
        self._did: Optional[str] = None

        if auto_load:
            self._load_or_generate_sync()

    @property
    def did(self) -> Optional[str]:
        return self._did

    @property
    def key_file(self) -> Path:
        return self._key_file

    @property
    def ucan_available(self) -> bool:
        return _UCAN_AVAILABLE and self._keypair is not None

    def _load_or_generate_sync(self) -> None:
        if not _UCAN_AVAILABLE:
            logger.warning(
                "py-ucan is not installed — DIDKeyManager is operating in stub mode. "
                "Install with: pip install py-ucan"
            )
            self._did = "did:key:stub-ucan-not-installed"
            return

        if self._key_file.exists():
            self._load_key()
        else:
            self._generate_and_save()

    def _generate_and_save(self) -> None:
        if not _UCAN_AVAILABLE:
            raise RuntimeError("py-ucan is not available — cannot generate key")
        kp = _ucan_lib.EdKeypair.generate()
        priv_model = kp.export()
        data = {
            "version": 1,
            "algorithm": "Ed25519",
            "did": kp.did(),
            "private_key_base64url": priv_model.d,
        }
        self._key_file.parent.mkdir(parents=True, exist_ok=True)
        self._key_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(self._key_file, 0o600)
        except OSError:
            pass
        self._keypair = kp
        self._did = kp.did()

    def _load_key(self) -> None:
        if not _UCAN_AVAILABLE:
            raise RuntimeError("py-ucan is not available — cannot load key")
        try:
            data = json.loads(self._key_file.read_text(encoding="utf-8"))
            secret_b64 = data["private_key_base64url"]
            kp = _ucan_lib.EdKeypair.from_secret_key(secret_b64)
            self._keypair = kp
            self._did = kp.did()
        except Exception as exc:
            logger.error(
                "Failed to load DID key from %s (%s: %s) — regenerating",
                self._key_file,
                type(exc).__name__,
                exc,
            )
            self._generate_and_save()

    def export_secret_b64(self) -> str:
        if not _UCAN_AVAILABLE or self._keypair is None:
            raise RuntimeError("py-ucan is not available — cannot export key")
        return self._keypair.export().d

    def rotate_key(self) -> str:
        if not _UCAN_AVAILABLE:
            raise RuntimeError("py-ucan is not available — cannot rotate key")
        self._generate_and_save()
        assert self._did is not None
        return self._did

    async def mint_delegation(
        self,
        audience_did: str,
        capabilities: Sequence[Tuple[str, str]],
        lifetime_seconds: int = 3600,
    ) -> str:
        if not _UCAN_AVAILABLE or self._keypair is None:
            raise RuntimeError("py-ucan is not available — cannot mint delegation")

        cap_dicts = [{"with": res, "can": ability} for res, ability in capabilities]
        u = await _ucan_lib.build(
            issuer=self._keypair,
            audience=audience_did,
            capabilities=cap_dicts,
            lifetime_in_seconds=lifetime_seconds,
        )
        return u.encode()

    async def verify_delegation(
        self,
        token: str,
        required_capabilities: Sequence[Tuple[str, str]],
    ) -> bool:
        if not _UCAN_AVAILABLE or self._keypair is None:
            raise RuntimeError("py-ucan is not available — cannot verify delegation")

        req_caps = [
            _ucan_lib.RequiredCapability(
                capability=_ucan_lib.Capability(with_=res, can=ability),
                root_issuer=self._did,
            )
            for res, ability in required_capabilities
        ]
        result = await _ucan_lib.verify(
            token,
            audience=self._did,
            required_capabilities=req_caps,
        )
        return isinstance(result, _ucan_lib.VerifyResultOk)

    async def mint_self_delegation(
        self,
        capabilities: Sequence[Tuple[str, str]],
        lifetime_seconds: int = 86_400,
    ) -> str:
        assert self._did is not None
        return await self.mint_delegation(
            audience_did=self._did,
            capabilities=capabilities,
            lifetime_seconds=lifetime_seconds,
        )

    async def sign_delegation_token(
        self,
        token: Any,
        audience_did: Optional[str] = None,
        lifetime_seconds: int = 86_400,
    ) -> str:
        aud = audience_did or getattr(token, "audience", None) or self._did
        caps: List[Tuple[str, str]] = [(c.resource, c.ability) for c in getattr(token, "capabilities", [])]

        if _UCAN_AVAILABLE and self._keypair is not None:
            return await self.mint_delegation(
                audience_did=aud,
                capabilities=caps,
                lifetime_seconds=lifetime_seconds,
            )

        import base64
        import json as _json
        import time

        payload = {
            "iss": self._did,
            "aud": aud,
            "caps": [{"with": c[0], "can": c[1]} for c in caps],
            "exp": int(time.time()) + lifetime_seconds,
            "cid": getattr(token, "cid", "stub"),
        }
        b64 = base64.urlsafe_b64encode(_json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=").decode()
        return f"stub:{b64}"

    async def verify_signed_token(
        self,
        signed_token: str,
        required_capabilities: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> bool:
        if signed_token.startswith("stub:"):
            import base64
            import json as _json

            try:
                b64 = signed_token[5:] + "=="
                _json.loads(base64.urlsafe_b64decode(b64).decode())
                return True
            except Exception:
                return False

        if not _UCAN_AVAILABLE or self._keypair is None:
            return False

        req_caps: List[Any] = []
        if required_capabilities:
            req_caps = [
                _ucan_lib.RequiredCapability(
                    capability=_ucan_lib.Capability(with_=res, can=ability),
                    root_issuer=self._did,
                )
                for res, ability in required_capabilities
            ]
            result = await _ucan_lib.verify(
                signed_token,
                audience=self._did,
                required_capabilities=req_caps,
            )
            return isinstance(result, _ucan_lib.VerifyResultOk)

        try:
            await _ucan_lib.verify(signed_token, audience=self._did, required_capabilities=[])
            return True
        except Exception:
            return False

    def info(self) -> Dict[str, Any]:
        return {
            "did": self._did,
            "key_file": str(self._key_file),
            "ucan_available": self.ucan_available,
        }

    def __repr__(self) -> str:
        return f"DIDKeyManager(did={self._did!r}, key_file={self._key_file!r})"


_default_manager: Optional[DIDKeyManager] = None


def get_did_key_manager(key_file: Optional[Path] = None) -> DIDKeyManager:
    """Return process singleton DID key manager."""
    global _default_manager
    if _default_manager is None or key_file is not None:
        _default_manager = DIDKeyManager(key_file=key_file)
    return _default_manager


__all__ = [
    "DIDKeyManager",
    "get_did_key_manager",
]
