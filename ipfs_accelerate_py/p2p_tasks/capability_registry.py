"""Persistent peer capability registry for p2p task orchestration."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_registry_path() -> Path:
    return Path.home() / ".cache" / "ipfs_accelerate" / "peer_capability_registry.json"


def _as_str_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    return []


@dataclass
class PeerCapabilityRecord:
    peer_id: str
    multiaddr: str
    last_seen: float = field(default_factory=time.time)
    session: str = ""
    supported_tasks: List[str] = field(default_factory=list)
    hardware_types: List[str] = field(default_factory=list)
    loaded_models: List[str] = field(default_factory=list)
    available_images: List[str] = field(default_factory=list)
    queued: int = 0
    running: int = 0
    queued_by_type: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PeerCapabilityRegistry:
    def __init__(self, path: Optional[str] = None):
        self._path = Path(path).expanduser() if path else _default_registry_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: Dict[str, PeerCapabilityRecord] = {}
        self._load()

    def _serialize(self, record: PeerCapabilityRecord) -> Dict[str, Any]:
        return {
            "peer_id": record.peer_id,
            "multiaddr": record.multiaddr,
            "last_seen": float(record.last_seen),
            "session": record.session,
            "supported_tasks": list(record.supported_tasks),
            "hardware_types": list(record.hardware_types),
            "loaded_models": list(record.loaded_models),
            "available_images": list(record.available_images),
            "queued": int(record.queued),
            "running": int(record.running),
            "queued_by_type": dict(record.queued_by_type),
            "metadata": dict(record.metadata),
        }

    def _deserialize(self, payload: Dict[str, Any]) -> PeerCapabilityRecord:
        return PeerCapabilityRecord(
            peer_id=str(payload.get("peer_id") or ""),
            multiaddr=str(payload.get("multiaddr") or ""),
            last_seen=float(payload.get("last_seen") or time.time()),
            session=str(payload.get("session") or ""),
            supported_tasks=_as_str_list(payload.get("supported_tasks")),
            hardware_types=_as_str_list(payload.get("hardware_types")),
            loaded_models=_as_str_list(payload.get("loaded_models")),
            available_images=_as_str_list(payload.get("available_images")),
            queued=int(payload.get("queued") or 0),
            running=int(payload.get("running") or 0),
            queued_by_type={str(k): int(v) for k, v in dict(payload.get("queued_by_type") or {}).items()},
            metadata=dict(payload.get("metadata") or {}),
        )

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            return

        for item in records:
            if not isinstance(item, dict):
                continue
            try:
                record = self._deserialize(item)
            except Exception:
                continue
            if not record.peer_id:
                continue
            self._records[record.peer_id] = record

    def _save(self) -> None:
        payload = {
            "timestamp": time.time(),
            "records": [self._serialize(record) for record in self._records.values()],
        }
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception:
            return

    @property
    def path(self) -> Path:
        return self._path

    def upsert_from_status(
        self,
        *,
        peer_id: str,
        multiaddr: str,
        status: Dict[str, Any],
    ) -> Optional[PeerCapabilityRecord]:
        pid = str(peer_id or "").strip()
        ma = str(multiaddr or "").strip()
        if not pid or not ma:
            return None

        capabilities = status.get("capabilities") if isinstance(status.get("capabilities"), dict) else {}
        detail = status.get("detail") if isinstance(status.get("detail"), dict) else {}

        supported_tasks = _as_str_list(capabilities.get("supported_task_types") or capabilities.get("supported_tasks"))

        hardware_types: List[str] = []
        runtime = detail.get("runtime") if isinstance(detail.get("runtime"), dict) else {}
        if runtime:
            if runtime.get("cuda_available"):
                hardware_types.append("cuda")
            if runtime.get("mps_available"):
                hardware_types.append("mps")
            if not hardware_types:
                hardware_types.append("cpu")

        loaded_models = _as_str_list(capabilities.get("loaded_models"))
        available_images = _as_str_list(capabilities.get("available_images"))

        queued_by_type = status.get("queued_by_type") if isinstance(status.get("queued_by_type"), dict) else {}

        record = PeerCapabilityRecord(
            peer_id=pid,
            multiaddr=ma,
            last_seen=time.time(),
            session=str(status.get("session") or "").strip(),
            supported_tasks=supported_tasks,
            hardware_types=hardware_types,
            loaded_models=loaded_models,
            available_images=available_images,
            queued=int(status.get("queued") or 0),
            running=int(status.get("running") or 0),
            queued_by_type={str(k): int(v) for k, v in queued_by_type.items()},
            metadata={
                "nat": status.get("nat"),
                "peer_id": pid,
            },
        )

        with self._lock:
            self._records[pid] = record
            self._save()
        return record

    def list_records(self) -> List[PeerCapabilityRecord]:
        with self._lock:
            return list(self._records.values())

    def get_record(self, peer_id: str) -> Optional[PeerCapabilityRecord]:
        with self._lock:
            return self._records.get(str(peer_id or ""))

    def score_peer_for_task(self, *, peer_id: str, task_type: str) -> float:
        with self._lock:
            record = self._records.get(str(peer_id or ""))
        if record is None:
            return 0.0

        score = 0.0
        normalized_task = str(task_type or "").strip().lower()
        supported = {str(t).strip().lower() for t in record.supported_tasks}
        if normalized_task and normalized_task in supported:
            score += 10.0

        queue_penalty = float(max(0, record.queued) + max(0, record.running))
        score -= queue_penalty

        if "cuda" in {x.lower() for x in record.hardware_types}:
            score += 1.0

        age_s = max(0.0, time.time() - float(record.last_seen or 0.0))
        freshness_bonus = max(0.0, 5.0 - min(age_s, 5.0))
        score += freshness_bonus
        return score


__all__ = [
    "PeerCapabilityRegistry",
    "PeerCapabilityRecord",
]
