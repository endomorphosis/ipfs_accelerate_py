"""
API Key Pool for Multi-User / Multi-Key Parallel Execution

Provides a thread-safe pool of API keys for a single provider so that:
- Multiple concurrent MCP users each get a distinct API key in round-robin
  order, spreading rate-limit pressure evenly across all keys.
- A specific user can be "pinned" to a key for the duration of a session,
  ensuring consistent billing attribution and avoiding context surprises.
- Keys can be added or removed at runtime without restarting the server.

Usage
-----
Single provider, multiple keys::

    from ipfs_accelerate_py.cli_integrations import ApiKeyPool

    pool = ApiKeyPool(["key-A", "key-B", "key-C"])

    # Round-robin – returns "key-A", "key-B", "key-C", "key-A", ...
    key = pool.get_key()

    # Per-user pinning – user "alice" always gets the same key
    key_alice = pool.get_key(user_id="alice")
    key_alice2 = pool.get_key(user_id="alice")  # same as key_alice
    assert key_alice == key_alice2

Multi-provider (one pool per provider)::

    from ipfs_accelerate_py.cli_integrations import XAIGrokCLIIntegration, ApiKeyPool

    xai = XAIGrokCLIIntegration(
        api_keys=["xai-key-1", "xai-key-2", "xai-key-3"],
    )
    # Each async/sync call to achat(user_id="bob") selects the key for "bob"
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional


class ApiKeyPool:
    """
    Thread-safe round-robin pool of API keys with optional per-user pinning.

    Strategy
    --------
    ``"round_robin"`` (default)
        Each call to :meth:`get_key` without a *user_id* advances a shared
        counter and returns ``keys[counter % len(keys)]``.  Calls *with* a
        *user_id* pin the user to the key returned on their first call so
        that every subsequent call returns the same key.

    Parameters
    ----------
    keys:
        Initial list of API keys.  Must contain at least one key.
    strategy:
        Selection strategy – currently only ``"round_robin"`` is supported.
    """

    def __init__(
        self,
        keys: List[str],
        strategy: str = "round_robin",
    ) -> None:
        if not keys:
            raise ValueError("ApiKeyPool requires at least one key")
        if strategy != "round_robin":
            raise ValueError(f"Unsupported strategy: {strategy!r} (only 'round_robin' is supported)")

        self._keys: List[str] = list(keys)
        self._strategy = strategy
        self._counter: int = 0
        self._user_pins: Dict[str, str] = {}  # user_id -> pinned key
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def get_key(self, user_id: Optional[str] = None) -> str:
        """
        Return an API key from the pool.

        If *user_id* is given the user will be pinned to the key chosen on
        their first call.  The pinned key is returned for all subsequent
        calls with the same *user_id*, unless the key has been removed (in
        which case the user is automatically re-pinned to a new key).

        Returns
        -------
        str
            An API key from the pool.

        Raises
        ------
        RuntimeError
            If the pool is empty.
        """
        with self._lock:
            if not self._keys:
                raise RuntimeError("ApiKeyPool is empty – add at least one key before calling get_key()")

            # Honour existing user pin when the pinned key is still present
            if user_id is not None:
                pinned = self._user_pins.get(user_id)
                if pinned and pinned in self._keys:
                    return pinned

            # Round-robin selection
            key = self._keys[self._counter % len(self._keys)]
            self._counter += 1

            # Create / update user pin
            if user_id is not None:
                self._user_pins[user_id] = key

            return key

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def add_key(self, key: str) -> None:
        """Add a key to the pool (no-op if already present)."""
        with self._lock:
            if key not in self._keys:
                self._keys.append(key)

    def remove_key(self, key: str) -> None:
        """
        Remove a key from the pool.

        Any users pinned to the removed key will be automatically re-pinned
        on their next :meth:`get_key` call.
        """
        with self._lock:
            self._keys = [k for k in self._keys if k != key]
            # Clear stale user pins so they get reassigned
            self._user_pins = {u: k for u, k in self._user_pins.items() if k != key}

    def keys(self) -> List[str]:
        """Return a snapshot of the current key list (copies; not live)."""
        with self._lock:
            return list(self._keys)

    def unpin_user(self, user_id: str) -> None:
        """Remove the pin for *user_id* so they receive a new key on the next call."""
        with self._lock:
            self._user_pins.pop(user_id, None)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)

    def __repr__(self) -> str:
        with self._lock:
            return f"ApiKeyPool(keys={len(self._keys)}, strategy={self._strategy!r}, pinned_users={len(self._user_pins)})"
