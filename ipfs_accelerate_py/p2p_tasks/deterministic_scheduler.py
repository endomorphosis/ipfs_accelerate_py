"""Deterministic task assignment helpers for TaskQueue.

This adapts the core assignment concepts from the P2P workflow scheduler:
- Merkle clock (vector clock + deterministic merkle root)
- Peer selection by Hamming distance over hashes

The goal is for a swarm to delegate work deterministically given the same
(view of) peer set + clock state.

This module is dependency-minimal and safe to import in long-lived services.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


def sha256_hex(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def calculate_hamming_distance(hash1: str, hash2: str) -> int:
    """Hamming distance between two hex digests."""

    try:
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
        return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
    except Exception:
        # Worst-case fallback: treat as far.
        return 10**9


@dataclass
class MerkleClock:
    """Vector clock with a deterministic merkle root."""

    node_id: str
    vector: Dict[str, int] = field(default_factory=dict)
    merkle_root: Optional[str] = None

    def __post_init__(self) -> None:
        if self.node_id and self.node_id not in self.vector:
            self.vector[self.node_id] = 0

    def tick(self) -> None:
        if not self.node_id:
            return
        self.vector[self.node_id] = self.vector.get(self.node_id, 0) + 1
        self._update_merkle_root()

    def update(self, other: "MerkleClock") -> None:
        for node_id, ts in (other.vector or {}).items():
            try:
                self.vector[str(node_id)] = max(int(self.vector.get(str(node_id), 0)), int(ts))
            except Exception:
                continue
        self.tick()

    def _update_merkle_root(self) -> None:
        sorted_entries = sorted((self.vector or {}).items())
        clock_data = json.dumps(sorted_entries, sort_keys=True)
        self.merkle_root = sha256_hex(clock_data)

    def get_hash(self) -> str:
        if not self.merkle_root:
            self._update_merkle_root()
        return str(self.merkle_root or "")

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "vector": dict(self.vector or {}), "merkle_root": self.get_hash()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MerkleClock":
        node_id = str(data.get("node_id") or "")
        vector = data.get("vector") if isinstance(data.get("vector"), dict) else {}
        clock = cls(node_id=node_id, vector={str(k): int(v) for k, v in vector.items() if str(k)})
        clock.merkle_root = str(data.get("merkle_root") or "") or None
        return clock


def _normalized_peer_hashes(peers: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for peer_id in peers:
        pid = str(peer_id or "").strip()
        if not pid:
            continue
        out[pid] = sha256_hex(pid)
    return out


def task_hash(*, task_id: str, task_type: str, model_name: str) -> str:
    # Include type + model so different task classes distribute independently.
    return sha256_hex(f"{task_id}:{task_type}:{model_name}")


def select_owner_peer(
    *,
    peer_ids: Iterable[str],
    clock_hash: str,
    task_hash_hex: str,
) -> str:
    """Select the owner peer by minimum Hamming distance."""

    peer_hashes = _normalized_peer_hashes(peer_ids)
    if not peer_hashes:
        return ""

    combined = sha256_hex(f"{clock_hash}:{task_hash_hex}")

    selected_peer = ""
    min_distance = 10**18
    for pid, ph in peer_hashes.items():
        d = calculate_hamming_distance(combined, ph)
        if d < min_distance:
            min_distance = d
            selected_peer = pid

    return selected_peer


def is_peer_stale(*, last_seen: float, timeout_s: float) -> bool:
    try:
        return (time.time() - float(last_seen)) > float(timeout_s)
    except Exception:
        return True
