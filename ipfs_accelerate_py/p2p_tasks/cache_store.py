from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _key_digest(key: str) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


@dataclass
class CacheRecord:
    key: str
    value: Any
    created_at: float
    expires_at: float | None

    def expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        t = float(now if now is not None else time.time())
        return t >= float(self.expires_at)


class DiskTTLCache:
    """A tiny, JSON-serializable, TTL-based key/value store.

    Designed for use behind authenticated libp2p RPC calls.

    Storage model:
    - One file per key: `<sha256(key)>.json`
    - Record contains {key, value, created_at, expires_at}
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{_key_digest(key)}.json"

    def get(self, key: str) -> Any | None:
        if not key:
            return None
        path = self._path_for(key)
        with self._lock:
            if not path.exists():
                return None
            try:
                raw = path.read_text(encoding="utf-8")
                obj = json.loads(raw)
            except Exception:
                return None

            if not isinstance(obj, dict):
                return None

            expires_at = obj.get("expires_at")
            try:
                expires_at_f = float(expires_at) if expires_at is not None else None
            except Exception:
                expires_at_f = None

            if expires_at_f is not None and time.time() >= expires_at_f:
                try:
                    path.unlink(missing_ok=True)  # py3.8+: missing_ok
                except TypeError:  # pragma: no cover
                    try:
                        if path.exists():
                            path.unlink()
                    except Exception:
                        pass
                except Exception:
                    pass
                return None

            return obj.get("value")

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def set(self, key: str, value: Any, *, ttl_s: float | None = None) -> Any:
        if not key:
            return value
        now = time.time()
        expires_at = None
        if ttl_s is not None:
            try:
                ttl_value = float(ttl_s)
                if ttl_value > 0:
                    expires_at = now + ttl_value
            except Exception:
                expires_at = None

        rec = {
            "key": str(key),
            "value": value,
            "created_at": now,
            "expires_at": expires_at,
        }

        path = self._path_for(key)
        with self._lock:
            _atomic_write_json(path, rec)
        return value

    def delete(self, key: str) -> bool:
        if not key:
            return False
        path = self._path_for(key)
        with self._lock:
            try:
                path.unlink(missing_ok=True)
                return True
            except TypeError:  # pragma: no cover
                if path.exists():
                    path.unlink()
                    return True
            except Exception:
                return False
        return False


def default_cache_dir() -> Path:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_CACHE_DIR") or os.environ.get(
        "IPFS_DATASETS_PY_TASK_P2P_CACHE_DIR"
    )
    if raw and str(raw).strip():
        return Path(str(raw)).expanduser()
    return Path.home() / ".cache" / "ipfs_accelerate" / "p2p_cache"


def cache_enabled() -> bool:
    return _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE") or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ENABLE_CACHE"))
