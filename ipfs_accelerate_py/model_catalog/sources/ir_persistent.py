"""Explicit, read-only discovery of persisted IR checkpoint declarations.

This source rereads its caller-selected store on every load/refresh.  It never
constructs ModelManager, imports a decoder, probes a model or installs a router
binding.  The catalog contains declarations with unknown operational state;
the detached sidecar retains the exact IR selectors lost by the generic legacy
projection.  A stored readiness/quality declaration is not a new observation.

DuckDB reads reuse PersistentCatalogSource's read_only connection and bounded
row query.  File witnesses are cooperative endpoint checks, not an atomic
filesystem snapshot or authentication of checkpoint/Hub payloads.  A live
writer may make the read fail; callers should retain their last good snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Tuple

from ..identity import is_secret_value
from .persistent import PersistentCatalogSource
from .static import CatalogSourceResult

SCHEMA = "ir-persistent-catalog-source/v1"
RESOLUTION_SCHEMA = "ir-persisted-binding-resolution/v1"
COMPONENT_SCHEMA = "ir-decoder-component-binding/v1"
MAX_CONFIG_BYTES = 8 * 1024 * 1024
MAX_BINDING_BYTES = 64 * 1024 * 1024
MAX_STORE_BYTES = 4 * 1024 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FAMILIES = {"codebase_ir", "security_ir", "legal_ir", "intent_ir", "ui_ux_ir"}
_SELECTORS = {
    "record_id", "ir_family_id", "dimension", "dimension_role", "schema_version",
    "task_id", "profile_id", "format_id", "checkpoint_sha256", "role",
}
_DECLARATION_FIELDS = _SELECTORS - {"checkpoint_sha256"} | {
    "original_checkpoint_pin", "trained", "initialization_only", "donor",
    "runtime_ready", "teacher_qualified", "proof_authority",
}
_RECORD_PREFIXES = (
    "ir-model-asset-binding/v1:", "ir-model-component-binding/v1:",
    "ir-decoder-checkpoint-record/v1:",
)


class IRPersistentCatalogError(ValueError):
    """A selected store or exact IR binding is malformed or unavailable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IRPersistentCatalogError(message)


def _raw(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "duplicate JSON field in stored IR metadata")
        result[key] = value
    return result


def _primitive(value: Any, depth: int = 0) -> None:
    _require(depth <= 24, "stored IR metadata nesting exceeds bound")
    if type(value) is dict:
        _require(len(value) <= 4096 and all(type(key) is str for key in value),
                 "bounded JSON object required")
        for item in value.values():
            _primitive(item, depth + 1)
    elif type(value) is list:
        _require(len(value) <= 4096, "stored IR metadata array exceeds bound")
        for item in value:
            _primitive(item, depth + 1)
    elif type(value) is float:
        _require(math.isfinite(value), "finite stored IR metadata required")
    else:
        _require(value is None or type(value) in (str, int, bool), "JSON primitives required")


def _json_object(value: Any, maximum: int) -> dict:
    try:
        if type(value) is str:
            _require(len(value.encode("utf-8")) <= maximum, "stored config exceeds byte bound")
            value = json.loads(value, object_pairs_hook=_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    IRPersistentCatalogError("finite JSON metadata required")))
        _primitive(value)
        encoded = _raw(value)
        _require(len(encoded) <= maximum, "stored config exceeds byte bound")
        _require(type(value) is dict, "stored config must be a JSON object")
        return json.loads(encoded)
    except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as error:
        if isinstance(error, IRPersistentCatalogError):
            raise
        raise IRPersistentCatalogError("bounded stored JSON config required") from error


def _text(value: Any, name: str, *, nullable: bool = False, maximum: int = 256):
    if nullable and value is None:
        return None
    _require(type(value) is str and value and "\x00" not in value,
             name + " requires explicit text")
    try:
        _require(len(value.encode("utf-8")) <= maximum and not is_secret_value(value),
                 name + " is oversized or credential-shaped")
    except UnicodeError as error:
        raise IRPersistentCatalogError(name + " must be UTF8") from error
    return value


def _selectors(value: Any) -> dict:
    _require(type(value) is dict and set(value) == _SELECTORS, "closed exact IR selector fields required")
    result = dict(value)
    _text(result["record_id"], "record_id")
    _require(any(result["record_id"].startswith(prefix) and _SHA.fullmatch(
        result["record_id"][len(prefix):]) for prefix in _RECORD_PREFIXES), "known concrete IR record_id required")
    _require(result["ir_family_id"] in _FAMILIES if type(result["ir_family_id"]) is str else False,
             "known IR family required")
    dimension, role = result["dimension"], result["dimension_role"]
    _require((type(dimension) is int and dimension in (8, 384, 768)
              and role in ("input_embedding", "latent"))
             or (dimension is None and role in ("source_tokens", "unbound_component")),
             "exact lane or explicitly detached component required")
    for name in ("schema_version", "task_id", "profile_id", "format_id"):
        _text(result[name], name, nullable=True, maximum=512)
    _text(result["role"], "checkpoint role")
    _require(type(result["checkpoint_sha256"]) is str and _SHA.fullmatch(result["checkpoint_sha256"]),
             "exact checkpoint SHA256 required")
    return result


def _binding(row: dict, declaration: dict) -> dict:
    _require(_DECLARATION_FIELDS <= set(declaration), "complete IR checkpoint declaration required")
    pin = declaration["original_checkpoint_pin"]
    _require(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}, "closed original checkpoint pin required")
    _text(pin["path"], "declared checkpoint path", maximum=4096)
    _require(Path(pin["path"]).is_absolute() and type(pin["bytes"]) is int
             and 0 < pin["bytes"] <= MAX_STORE_BYTES, "absolute bounded checkpoint declaration required")
    selected = _selectors({name: declaration[name] for name in _SELECTORS - {"checkpoint_sha256"}}
                          | {"checkpoint_sha256": pin["sha256"]})
    _require(row.get("model_id") == selected["record_id"], "persistent model_id differs from IR record_id")
    _require(row.get("model_revision") == pin["sha256"] and row.get("revision_id") == pin["sha256"],
             "persisted revision and checkpoint SHA differ")
    for name in ("trained", "donor"):
        _require(declaration[name] is None or type(declaration[name]) is bool, "unknown or boolean declaration required")
    for name in ("initialization_only", "runtime_ready", "teacher_qualified", "proof_authority"):
        _require(type(declaration[name]) is bool, "explicit boolean declaration required")
    _require(not declaration["initialization_only"] or declaration["trained"] is not True,
             "initialization-only checkpoint cannot declare trained")
    identity = {name: selected[name] for name in ("ir_family_id", "dimension", "dimension_role", "role", "checkpoint_sha256")}
    if selected["dimension"] is None:
        _require(declaration.get("asset_binding_schema") == COMPONENT_SCHEMA
                 and "external_lane_binding" in declaration and declaration["external_lane_binding"] is None
                 and selected["profile_id"] is None and selected["format_id"] is None,
                 "detached component must preserve its explicit unbound declaration")
        identity.update(asset_binding_schema=COMPONENT_SCHEMA, external_lane_binding=None)
        expected = "ir-model-component-binding/v1:" + hashlib.sha256(_raw(identity)).hexdigest()
        _require(selected["record_id"] == expected, "detached component identity differs")
    elif selected["record_id"].startswith("ir-model-asset-binding/v1:"):
        expected = "ir-model-asset-binding/v1:" + hashlib.sha256(_raw(identity)).hexdigest()
        _require(selected["record_id"] == expected, "family/lane/checkpoint identity differs")
    else:
        _require(selected["record_id"].startswith("ir-decoder-checkpoint-record/v1:")
                 and selected["ir_family_id"] in ("intent_ir", "security_ir")
                 and selected["dimension"] == 384 and selected["dimension_role"] == "input_embedding"
                 and selected["profile_id"] is not None and selected["format_id"] is not None,
                 "known registered format checkpoint binding required")
    return {**selected, "declaration": declaration,
            "original_checkpoint_bytes_verified": False,
            "runtime_observation_performed": False}


def _authority() -> dict:
    return {"model_manager_constructed": False, "model_loaded": False,
            "inference_executed": False, "training_executed": False,
            "runtime_admitted": False, "teacher_qualified": False,
            "proof_authority": False, "checkpoint_bytes_authenticated": False}


@dataclass(frozen=True)
class IRPersistentCatalogResult(CatalogSourceResult):
    """A catalog-compatible snapshot with a detached, exact selector sidecar."""

    storage_path: str = ""
    binding_snapshot_revision: str = ""
    _binding_payloads: Tuple[str, ...] = field(default=(), repr=False)

    @property
    def ir_bindings(self) -> Tuple[dict, ...]:
        return tuple(json.loads(payload) for payload in self._binding_payloads)

    def to_dict(self) -> dict:
        return {**super().to_dict(), "schema": SCHEMA, "storage_path": self.storage_path,
                "binding_snapshot_revision": self.binding_snapshot_revision,
                "ir_bindings": list(self.ir_bindings), "authority": _authority()}

    def resolve_ir_binding(self, request: dict) -> dict:
        selected = _selectors(request)
        matches = [item for item in self.ir_bindings
                   if all(item[name] == selected[name] for name in _SELECTORS)]
        _require(len(matches) == 1, "exact unique IR binding not found")
        return {"schema": RESOLUTION_SCHEMA, "storage_path": self.storage_path,
                "binding_snapshot_revision": self.binding_snapshot_revision,
                "catalog_revision": self.revision, "selected_binding": matches[0],
                "authority": _authority()}


class IRPersistentCatalogSource(PersistentCatalogSource):
    """Discover exact persisted IR metadata without changing any running jobs.

    ``path`` is required and absolute; no cwd/environment defaults are consulted.
    Non-IR rows are ignored. A missing/wrong store, malformed IR row, duplicate
    record or failed projection rejects the whole load. ``refresh`` performs a
    fresh read. Register under a distinct source name so the legacy manager's
    in-memory source and unrelated catalog sources are retained.
    """

    side_effecting = False

    def __init__(self, *, path: Any, source: str = "ir-models.persisted", max_records: int = 10000,
                 max_config_bytes: int = MAX_CONFIG_BYTES, max_store_bytes: int = MAX_STORE_BYTES):
        _require(isinstance(path, (str, Path)) and Path(path).is_absolute(), "explicit absolute store path required")
        _require(Path(path).suffix.lower() in (".duckdb", ".ddb", ".json", ".jsonl"), "explicit supported store format required")
        _require(type(max_records) is int and 1 <= max_records <= 10000, "bounded positive record count required")
        _require(type(max_config_bytes) is int and 1 <= max_config_bytes <= MAX_CONFIG_BYTES, "bounded config size required")
        _require(type(max_store_bytes) is int and 1 <= max_store_bytes <= MAX_STORE_BYTES, "bounded store size required")
        super().__init__(path=Path(path), source=source, max_records=max_records)
        self.max_config_bytes = max_config_bytes
        self.max_store_bytes = max_store_bytes

    def _witness(self):
        try:
            value = self._path.lstat()
        except OSError as error:
            raise IRPersistentCatalogError("selected IR store is unavailable") from error
        _require(stat.S_ISREG(value.st_mode) and 0 < value.st_size <= self.max_store_bytes,
                 "bounded regular selected IR store required")
        return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def _supplied_value(self):
        if self._path.suffix.lower() not in (".json", ".jsonl"):
            return super()._supplied_value()
        try:
            before = self._witness()
            _require(before[3] <= MAX_CONFIG_BYTES, "explicit JSON store exceeds byte bound")
            fd = os.open(self._path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            chunks, count = [], 0
            witness = lambda value: (value.st_dev, value.st_ino, value.st_mode,
                                    value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            try:
                _require(witness(os.fstat(fd)) == before, "selected JSON descriptor changed before read")
                while True:
                    chunk = os.read(fd, min(65536, MAX_CONFIG_BYTES + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    _require(count <= MAX_CONFIG_BYTES, "explicit JSON store exceeds byte bound during read")
                    chunks.append(chunk)
                _require(witness(os.fstat(fd)) == before, "selected JSON descriptor changed during read")
            finally:
                os.close(fd)
            _require(self._witness() == before, "selected JSON path changed during read")
            text = b"".join(chunks).decode("utf-8")
            def parse(value):
                return json.loads(value, object_pairs_hook=_pairs,
                    parse_constant=lambda _: (_ for _ in ()).throw(
                        IRPersistentCatalogError("finite stored JSON required")))
            try:
                return parse(text)
            except json.JSONDecodeError:
                rows = []
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    rows.append(parse(line))
                    _require(len(rows) <= self.max_records, "selected store exceeds record bound")
                return rows
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise IRPersistentCatalogError("selected JSON store must contain valid UTF8 JSON") from error

    def load(self) -> IRPersistentCatalogResult:
        before = self._witness()
        try:
            supplied = self._supplied_value()
            # Reuse the generic envelope/mapping normalization, not its partial
            # row acceptance: an IR binding cannot survive a malformed peer.
            from .static import _normalize_input
            rows, _ = _normalize_input(supplied, self.max_records)
            _require(len(rows) <= self.max_records, "selected store exceeds record bound")
            bindings, projected, ids = [], [], set()
            binding_bytes = 0
            for row in rows:
                _require(type(row) is dict, "persistent metadata rows must be objects")
                model_id = row.get("model_id")
                config = row.get("huggingface_config")
                is_ir_id = type(model_id) is str and model_id.startswith(_RECORD_PREFIXES)
                if config is None:
                    _require(not is_ir_id, "IR record lacks checkpoint declaration")
                    continue
                config = _json_object(config, self.max_config_bytes)
                declaration = config.get("ir_checkpoint")
                if declaration is None:
                    _require(not is_ir_id, "IR record lacks checkpoint declaration")
                    continue
                _require(type(declaration) is dict, "IR checkpoint declaration must be object")
                binding = _binding(row, declaration)
                binding_bytes += len(_raw(binding))
                _require(binding_bytes <= MAX_BINDING_BYTES, "IR binding sidecar exceeds aggregate byte bound")
                _require(binding["record_id"] not in ids, "duplicate persistent IR record_id")
                ids.add(binding["record_id"])
                labels = {"ir.record-id": binding["record_id"], "ir.family": binding["ir_family_id"],
                          "ir.dimension-role": binding["dimension_role"], "ir.checkpoint-sha256": binding["checkpoint_sha256"],
                          "ir.role": binding["role"],
                          "ir.declaration-sha256": hashlib.sha256(_raw(binding)).hexdigest()}
                for name in ("dimension", "schema_version", "task_id", "profile_id", "format_id"):
                    if binding[name] is not None:
                        labels["ir." + name.replace("_", "-")] = str(binding[name])
                # A metadata inventory contributes no invokable capability.
                projected.append({"provider": "retained-ir", "model_id": binding["record_id"],
                    "model_name": row.get("model_name", binding["record_id"]),
                    "architecture": row.get("architecture"), "model_revision": binding["checkpoint_sha256"],
                    "operations": [], "lifecycle": "declared", "labels": labels})
                bindings.append(binding)
            _require(bool(bindings), "selected store contains no IR checkpoint declarations")
            result = PersistentCatalogSource(projected, source=self.source, max_records=self.max_records).load()
            _require(result.error_count == 0 and len(result.models) == len(bindings), "IR catalog projection must retain every binding")
            model_ids = {item.provenance[0].source_record_id: item.model_id for item in result.models}
            for binding in bindings:
                _require(binding["record_id"] in model_ids, "catalog lost persistent record identity")
                binding["catalog_model_id"] = model_ids[binding["record_id"]]
            bindings.sort(key=lambda item: item["record_id"])
            encoded = _raw(bindings)
            _require(len(encoded) <= MAX_BINDING_BYTES, "IR binding sidecar exceeds aggregate byte bound")
            _require(self._witness() == before, "selected IR store changed during metadata read")
            return IRPersistentCatalogResult(snapshot=result.snapshot, metadata=result.metadata,
                diagnostics=result.diagnostics, redacted_fields=result.redacted_fields,
                storage_path=str(self._path), binding_snapshot_revision=hashlib.sha256(encoded).hexdigest(),
                _binding_payloads=tuple(_raw(item).decode("utf-8") for item in bindings))
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            if isinstance(error, IRPersistentCatalogError):
                raise
            raise IRPersistentCatalogError("selected IR metadata could not be read") from error

    def refresh(self) -> IRPersistentCatalogResult:
        return self.load()

    def resolve_ir_binding(self, request: dict) -> dict:
        # Reject malformed selectors before any store/native access.
        selected = _selectors(request)
        return self.load().resolve_ir_binding(selected)


__all__ = ["IRPersistentCatalogError", "IRPersistentCatalogSource", "IRPersistentCatalogResult"]
