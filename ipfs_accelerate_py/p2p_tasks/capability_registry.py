"""Persistent peer capability registry for p2p task orchestration.

Audio jobs match against provider, model, voice, codec, locale, device, memory,
and artifact-access advertisements while text-task behavior stays permissive.
This is the capability half of ABBY-VOICE-G016 recovery admission; queue-level
persisted attempt/backoff/lease state and owner heartbeats remain in TaskQueue.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from .task_types import VOICE_TASK_TYPES, canonical_task_type, normalize_task_types


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


def _as_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _first_non_negative_int(*values: Any) -> int:
    for value in values:
        parsed = _as_non_negative_int(value)
        if parsed > 0:
            return parsed
    return 0


def _normalized_values(value: Any) -> set[str]:
    return {item.strip().lower() for item in _as_str_list(value) if item.strip()}


def _capability_values(scope: Mapping[str, Any], *names: str) -> set[str]:
    for name in names:
        if name in scope:
            return _normalized_values(scope.get(name))
    return set()


def _matches_advertised_value(required: str, advertised: set[str]) -> bool:
    normalized = str(required or "").strip().lower()
    return bool(normalized and ("*" in advertised or normalized in advertised))


def _audio_capability_scope(
    capabilities: Mapping[str, Any],
    task_type: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return global and operation-specific audio capability mappings."""

    canonical = canonical_task_type(task_type)
    operation: Mapping[str, Any] = {}
    for key, value in capabilities.items():
        if canonical_task_type(key) == canonical and isinstance(value, Mapping):
            operation = value
            break
    return capabilities, operation


def _scope_values(
    global_scope: Mapping[str, Any],
    operation_scope: Mapping[str, Any],
    *names: str,
) -> set[str]:
    for name in names:
        if name in operation_scope:
            return _normalized_values(operation_scope.get(name))
    return _capability_values(global_scope, *names)


def _required_memory_bytes(payload: Mapping[str, Any]) -> int:
    resources = payload.get("resource_requirements")
    resource_scope = resources if isinstance(resources, Mapping) else {}
    return _first_non_negative_int(
        payload.get("required_memory_bytes"),
        payload.get("min_memory_bytes"),
        resource_scope.get("memory_bytes"),
        resource_scope.get("ram_bytes"),
        resource_scope.get("min_memory_bytes"),
    )


def _artifact_schemes(payload: Mapping[str, Any]) -> set[str]:
    schemes: set[str] = set()
    for name in ("reference_audio", "source_audio"):
        descriptor = payload.get(name)
        if not isinstance(descriptor, Mapping):
            continue
        uri = str(descriptor.get("uri") or "").strip()
        if uri:
            scheme = urlsplit(uri).scheme.strip().lower()
            if scheme:
                schemes.add(scheme)
        elif str(descriptor.get("cid") or "").strip():
            schemes.add("ipfs")
    return schemes


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
    audio_capabilities: Dict[str, Any] = field(default_factory=dict)
    available_memory_bytes: int = 0
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
            "audio_capabilities": dict(record.audio_capabilities),
            "available_memory_bytes": int(record.available_memory_bytes),
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
            audio_capabilities=dict(payload.get("audio_capabilities") or {}),
            available_memory_bytes=_as_non_negative_int(payload.get("available_memory_bytes")),
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
        local_worker = status.get("local_worker") if isinstance(status.get("local_worker"), dict) else {}

        supported_tasks = normalize_task_types(
            [
                *_as_str_list(
                    capabilities.get("supported_task_types")
                    or capabilities.get("supported_tasks")
                    or capabilities.get("task_types")
                ),
                *_as_str_list(local_worker.get("supported_task_types")),
            ],
            expand_aliases=False,
        )

        hardware_types: List[str] = []
        runtime = detail.get("runtime") if isinstance(detail.get("runtime"), dict) else {}
        if runtime:
            if runtime.get("cuda_available"):
                hardware_types.append("cuda")
            if runtime.get("mps_available"):
                hardware_types.append("mps")
            if not hardware_types:
                hardware_types.append("cpu")

        loaded_models = _as_str_list(capabilities.get("loaded_models") or capabilities.get("models"))
        available_images = _as_str_list(capabilities.get("available_images"))
        audio_raw = capabilities.get("audio_capabilities")
        if not isinstance(audio_raw, Mapping):
            audio_raw = capabilities.get("audio")
        if not isinstance(audio_raw, Mapping):
            audio_raw = local_worker.get("audio_capabilities")
        audio_capabilities = dict(audio_raw) if isinstance(audio_raw, Mapping) else {}

        # A flat status remains useful to small deployments, while the nested
        # mapping supports operation-specific TTS/ASR constraints.
        for key in (
            "providers",
            "supported_providers",
            "voices",
            "supported_voices",
            "codecs",
            "supported_codecs",
            "locales",
            "supported_locales",
            "devices",
            "supported_devices",
            "artifact_schemes",
            "supported_artifact_schemes",
        ):
            if key in capabilities and key not in audio_capabilities:
                audio_capabilities[key] = capabilities[key]

        resource_caps = capabilities.get("resources")
        if not isinstance(resource_caps, Mapping):
            resource_caps = {}
        available_memory_bytes = _first_non_negative_int(
            capabilities.get("available_memory_bytes"),
            capabilities.get("memory_available_bytes"),
            capabilities.get("memory_capacity_bytes"),
            audio_capabilities.get("available_memory_bytes"),
            audio_capabilities.get("memory_available_bytes"),
            audio_capabilities.get("memory_capacity_bytes"),
            resource_caps.get("available_memory_bytes"),
            resource_caps.get("memory_available_bytes"),
            resource_caps.get("memory_bytes"),
            runtime.get("available_memory_bytes"),
            runtime.get("memory_available_bytes"),
        )

        queue_status = status.get("queue") if isinstance(status.get("queue"), dict) else {}
        queued_by_type = status.get("queued_by_type")
        if not isinstance(queued_by_type, dict):
            queued_by_type = queue_status.get("queued_by_type")
        if not isinstance(queued_by_type, dict):
            queued_by_type = {}

        record = PeerCapabilityRecord(
            peer_id=pid,
            multiaddr=ma,
            last_seen=time.time(),
            session=str(status.get("session") or "").strip(),
            supported_tasks=supported_tasks,
            hardware_types=hardware_types,
            loaded_models=loaded_models,
            available_images=available_images,
            queued=int(status.get("queued") or queue_status.get("queued") or 0),
            running=int(status.get("running") or queue_status.get("running") or 0),
            queued_by_type={str(k): int(v) for k, v in queued_by_type.items()},
            audio_capabilities=audio_capabilities,
            available_memory_bytes=available_memory_bytes,
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

    def matches_task_requirements(
        self,
        *,
        peer_id: str,
        task_type: str,
        model_name: str = "",
        payload: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Return whether a peer explicitly satisfies a task's requirements.

        Detailed matching is deliberately fail-closed only for canonical audio
        jobs. Legacy text and tool tasks retain task-type-only matching.
        """

        with self._lock:
            record = self._records.get(str(peer_id or ""))
        if record is None:
            return False

        canonical = canonical_task_type(task_type)
        supported = {
            canonical_task_type(value)
            for value in record.supported_tasks
            if canonical_task_type(value)
        }
        if not canonical or canonical not in supported:
            return False
        if canonical not in VOICE_TASK_TYPES:
            return True

        request = payload if isinstance(payload, Mapping) else {}
        resources = request.get("resource_requirements")
        resource_requirements = resources if isinstance(resources, Mapping) else {}
        generation = request.get("generation_settings")
        generation_settings = generation if isinstance(generation, Mapping) else {}
        decoding = request.get("decoding_settings")
        decoding_settings = decoding if isinstance(decoding, Mapping) else {}
        global_scope, operation_scope = _audio_capability_scope(
            record.audio_capabilities,
            canonical,
        )

        requirements = {
            "provider": (
                str(request.get("provider") or "").strip(),
                _scope_values(
                    global_scope,
                    operation_scope,
                    "providers",
                    "supported_providers",
                ),
            ),
            "model": (
                str(model_name or request.get("model_name") or "").strip(),
                _scope_values(
                    global_scope,
                    operation_scope,
                    "models",
                    "loaded_models",
                    "supported_models",
                )
                or {value.strip().lower() for value in record.loaded_models if value.strip()},
            ),
            "voice": (
                str(request.get("voice") or "").strip(),
                _scope_values(
                    global_scope,
                    operation_scope,
                    "voices",
                    "supported_voices",
                ),
            ),
            "codec": (
                str(request.get("codec") or "").strip(),
                _scope_values(
                    global_scope,
                    operation_scope,
                    "codecs",
                    "supported_codecs",
                ),
            ),
            "locale": (
                str(request.get("locale") or "").strip(),
                _scope_values(
                    global_scope,
                    operation_scope,
                    "locales",
                    "supported_locales",
                ),
            ),
            "device": (
                str(
                    request.get("device")
                    or resource_requirements.get("device")
                    or generation_settings.get("device")
                    or decoding_settings.get("device")
                    or ""
                ).strip(),
                _scope_values(
                    global_scope,
                    operation_scope,
                    "devices",
                    "supported_devices",
                )
                or {value.strip().lower() for value in record.hardware_types if value.strip()},
            ),
        }
        mandatory = {"provider", "model"}
        if canonical == "voice.tts":
            mandatory.update({"voice", "codec", "locale"})
        if any(not requirements[name][0] for name in mandatory):
            return False
        for required, advertised in requirements.values():
            if required and not _matches_advertised_value(required, advertised):
                return False

        required_memory = _required_memory_bytes(request)
        if required_memory and record.available_memory_bytes < required_memory:
            return False

        artifact_schemes = _artifact_schemes(request)
        if artifact_schemes:
            advertised_schemes = _scope_values(
                global_scope,
                operation_scope,
                "artifact_schemes",
                "supported_artifact_schemes",
                "artifact_access",
            )
            if "*" not in advertised_schemes and not artifact_schemes.issubset(advertised_schemes):
                return False
        return True

    def score_peer_for_task(
        self,
        *,
        peer_id: str,
        task_type: str,
        model_name: str = "",
        payload: Optional[Mapping[str, Any]] = None,
    ) -> float:
        with self._lock:
            record = self._records.get(str(peer_id or ""))
        if record is None:
            return 0.0

        score = 0.0
        normalized_task = canonical_task_type(task_type)
        supported = {canonical_task_type(t) for t in record.supported_tasks}
        if normalized_task and normalized_task in supported:
            score += 10.0

        if (payload is not None or model_name) and not self.matches_task_requirements(
            peer_id=peer_id,
            task_type=task_type,
            model_name=model_name,
            payload=payload,
        ):
            return float("-inf")

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
