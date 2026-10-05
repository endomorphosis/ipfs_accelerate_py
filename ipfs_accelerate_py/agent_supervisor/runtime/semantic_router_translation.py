"""Reversible representation transport for verified native semantic capsules.

Only typed producer identifier slots are aliased. Python names, source text,
paths, literals, comments, signatures and admission/authority decisions remain
literal. The datasets producer still owns every restored capsule identity.
This transport proves a finite representation round trip, never program
equivalence, admission, execution permission or completion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from ..context.context_contracts import ContextReference, canonical_context_json_bytes
from ..semantic_state.wire import cid_for_payload


TABLE_SCHEMA = "supervisor-semantic-translation-table@1"
TRANSPORT_SCHEMA = "supervisor-semantic-router-input@1"
REPLY_SCHEMA = "supervisor-semantic-router-reply@1"
REF_KEY = "$semantic_ref"
MAX_BYTES = 2_000_000
MAX_ITEMS = 4096
MAX_DEPTH = 40
CAPSULE_FIELDS = frozenset({
    "stable_symbol_id", "version_cid", "source_cid", "symbol_fact_cid", "capsule_cid",
    "relevant_binding_projection_cid", "dependency_stable_ids", "dependency_version_cids",
    "dependency_fact_cids", "dependency_link_ids", "test_refs", "fixture_refs", "proof_obligation_refs",
})
REFERENCE_FIELDS = frozenset({"capsule_cid", "symbol_id", "stable_symbol_id", "version_cid",
    "source_cid", "semantic_state_root_cid", "assessment_cid", "context_cid", "referenced_content_id"})
TABLE_FIELDS = frozenset({"schema", "source_repository", "task_id", "scope_cid", "semantic_root_cid",
    "semantic_artifact", "semantic_sha256", "native_prompt_sha256", "native_core_sha256",
    "native_suffix_sha256", "source_manifest", "symbol_ids", "entries", "replacement_paths",
    "reference_templates", "semantic_positions", "translated_semantic_sha256",
    "semantic_equivalence_claimed", "execution_authority", "completion_authority"})
INSTRUCTIONS = (
    "This is a reversible representation of native semantic evidence. Objects with the sole key "
    "'$semantic_ref' in translated_semantic resolve through translation_table. Source strings, "
    "paths, literals, comments, policy and authority are unchanged. Aliases confer no proof or "
    "permission. Ordinary prose replies remain literal. Structured reference replies must use "
    "supervisor-semantic-router-reply@1 with the exact translation/task/root/scope bindings, "
    "a native residual task_family and a candidate_only native response; alias objects may occur "
    "only in symbol_ids or reference-id lists. Native validators still decide admissibility."
)


# Only owner-authored diagnostics are projected into provider usage receipts.
# Unknown exception text may contain source or provider data and stays private.
_ERROR_REASON_CODES = {
    "ambiguous translation mapping": "mapping_ambiguous",
    "capsule differs from verified producer": "capsule_producer_mismatch",
    "duplicate translation key": "duplicate_key",
    "explicit semantic program selection differs": "program_selection_mismatch",
    "explicit semantic program selection requires its versioned schema": "program_selection_schema_invalid",
    "historical replay cannot decode operational references": "historical_decode_forbidden",
    "historical semantic program reconstruction differs": "historical_program_mismatch",
    "invalid producer block identity": "producer_block_identity_invalid",
    "invalid translation alias": "alias_invalid",
    "invalid translation table identity or authority": "table_identity_or_authority_invalid",
    "malformed reserved translation response": "response_reserved_envelope_malformed",
    "native compiled context JSON required": "native_context_json_invalid",
    "native prompt did not round-trip exactly": "native_prompt_round_trip_mismatch",
    "native prompt exceeds translation bound": "native_prompt_size_exceeded",
    "native prompt has no semantic context nomination": "semantic_nomination_missing",
    "native prompt is not canonical compiled context": "native_context_noncanonical",
    "native prompt reconstruction differs": "native_prompt_reconstruction_mismatch",
    "native semantic chunk binding differs": "native_chunk_binding_mismatch",
    "native structured payload required": "response_structured_payload_missing",
    "nominated semantic artifact differs from native prompt": "nominated_artifact_mismatch",
    "noncanonical translation path": "path_noncanonical",
    "nonregular or oversized translation input": "input_type_or_size_invalid",
    "provider response exceeds translation bound": "response_size_exceeded",
    "raw semantic source differs from retained bytes": "raw_source_mismatch",
    "reference response requires a bounded list": "response_reference_list_invalid",
    "response translation table is stale or foreign": "response_table_stale_or_foreign",
    "restored semantic bytes differ": "restored_semantic_mismatch",
    "retained semantic source binding differs": "retained_source_binding_mismatch",
    "semantic artifact nomination is ambiguous": "semantic_nomination_ambiguous",
    "semantic identifier scope exceeds bound": "identifier_scope_exceeded",
    "semantic manifest differs from complete producer source inventory": "producer_inventory_mismatch",
    "semantic producer belongs to another repository": "producer_repository_mismatch",
    "semantic raw sources escape producer inventory": "raw_source_inventory_escape",
    "semantic reference coverage differs": "reference_coverage_mismatch",
    "semantic repository must be canonical": "repository_noncanonical",
    "semantic source scope identity differs": "source_scope_mismatch",
    "semantic task/schema/source binding differs": "semantic_binding_mismatch",
    "semantic translation source is stale": "source_stale",
    "source changed during translation": "source_changed_during_translation",
    "structured response must remain a native candidate": "response_candidate_authority_invalid",
    "structured response task/root/table binding differs": "response_envelope_binding_mismatch",
    "translated provider prompt exceeds bound": "provider_prompt_size_exceeded",
    "translated response failed native candidate grammar": "response_native_grammar_invalid",
    "translated wire or mapping was tampered": "wire_or_mapping_mismatch",
    "translation exceeds byte bound": "translation_size_exceeded",
    "translation exceeds structure bound": "translation_structure_exceeded",
    "translation input changed during read": "input_changed_during_read",
    "translation table differs from current producer": "table_producer_mismatch",
    "translation table exceeds entry bound": "table_entries_exceeded",
    "unknown or ambiguous semantic reference": "reference_unknown_or_ambiguous",
    "unknown semantic response alias": "response_alias_unknown",
    "unknown semantic response identifier": "response_identifier_unknown",
    "unsupported translation table fields": "table_fields_unsupported",
}


class SemanticTranslationError(ValueError):
    """A source, task, producer or reversible representation binding failed."""

    @property
    def reason_code(self) -> str:
        """Return a closed diagnostic code without exporting exception text."""
        if len(self.args) == 1 and type(self.args[0]) is str:
            return _ERROR_REASON_CODES.get(self.args[0], "unclassified")
        return "unclassified"


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _json(value) -> str:
    return canonical_context_json_bytes(value).decode()


def _parse(text: str):
    if not isinstance(text, str) or len(text.encode()) > MAX_BYTES:
        raise SemanticTranslationError("translation exceeds byte bound")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SemanticTranslationError("duplicate translation key")
            result[key] = value
        return result
    result = json.loads(text, object_pairs_hook=unique)
    count = 0
    def check(value, depth):
        nonlocal count
        count += 1
        if depth > MAX_DEPTH or count > 100_000:
            raise SemanticTranslationError("translation exceeds structure bound")
        if isinstance(value, dict):
            for child in value.values():
                check(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                check(child, depth + 1)
    check(result, 0)
    return result


def _read(root: Path, relative: str) -> bytes:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or str(rel) != relative or relative in {"", "."}:
        raise SemanticTranslationError("noncanonical translation path")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    file_fd = None
    try:
        for part in rel.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(rel.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= MAX_BYTES:
            raise SemanticTranslationError("nonregular or oversized translation input")
        with os.fdopen(file_fd, "rb", closefd=False) as stream:
            raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(file_fd)
        if len(raw) > MAX_BYTES or any(getattr(before, key) != getattr(after, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise SemanticTranslationError("translation input changed during read")
        return raw
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(fd)


@dataclass(frozen=True)
class SemanticTranslationTable:
    """Immutable canonical table, using datasets' typed SymbolMapEntry records."""
    payload_json: str
    translation_cid: str

    def __post_init__(self):
        from ipfs_datasets_py.logic.families.translations import SymbolMapEntry
        payload = _parse(self.payload_json)
        if (not isinstance(payload, dict) or set(payload) != TABLE_FIELDS
                or payload["schema"] != TABLE_SCHEMA or _json(payload) != self.payload_json
                or cid_for_payload(payload) != self.translation_cid
                or any(payload[key] is not False for key in
                       ("semantic_equivalence_claimed", "execution_authority", "completion_authority"))):
            raise SemanticTranslationError("invalid translation table identity or authority")
        entries = tuple(SymbolMapEntry.from_dict(row) for row in payload["entries"])
        if len(entries) > MAX_ITEMS or len(payload["replacement_paths"]) > MAX_ITEMS:
            raise SemanticTranslationError("translation table exceeds entry bound")
        for index, entry in enumerate(entries):
            if entry.disposition.value != "mapped" or entry.target_symbol_ids != (f"s{index}",):
                raise SemanticTranslationError("invalid translation alias")
        if len({entry.source_symbol_id for entry in entries}) != len(entries):
            raise SemanticTranslationError("ambiguous translation mapping")

    def to_dict(self) -> dict:
        return {**_parse(self.payload_json), "translation_cid": self.translation_cid}

    @classmethod
    def from_dict(cls, value: dict) -> "SemanticTranslationTable":
        if not isinstance(value, dict) or set(value) != TABLE_FIELDS | {"translation_cid"}:
            raise SemanticTranslationError("unsupported translation table fields")
        return cls(_json({key: value[key] for key in TABLE_FIELDS}), value["translation_cid"])


@dataclass(frozen=True)
class EncodedSemanticPrompt:
    native_prompt: str
    provider_prompt: str
    table: SemanticTranslationTable
    receipt_json: str
    freshness_checked: bool

    @property
    def receipt(self) -> dict:
        return _parse(self.receipt_json)


@dataclass(frozen=True)
class DecodedSemanticResponse:
    text: str
    receipt_json: str

    @property
    def receipt(self) -> dict:
        return _parse(self.receipt_json)


def _prompt_parts(prompt):
    if not isinstance(prompt, str) or len(prompt.encode()) > MAX_BYTES:
        raise SemanticTranslationError("native prompt exceeds translation bound")
    try:
        wire, end = json.JSONDecoder().raw_decode(prompt)
    except (ValueError, TypeError) as error:
        raise SemanticTranslationError("native compiled context JSON required") from error
    # Canonical native bytes, including duplicate-key rejection, are preserved.
    if _json(wire) != prompt[:end] or not isinstance(wire, dict) or not isinstance(wire.get("evidence"), list):
        raise SemanticTranslationError("native prompt is not canonical compiled context")
    _parse(prompt[:end])
    positions = [index for index, row in enumerate(wire["evidence"])
                 if isinstance(row, dict) and row.get("kind") == "semantic-context"]
    if not positions:
        raise SemanticTranslationError("native prompt has no semantic context nomination")
    refs = [ContextReference.from_dict(wire["evidence"][index]) for index in positions]
    text = "".join(row.summary for row in refs)
    paths = {row.path for row in refs}
    artifact_cid = "sha256:" + _sha(_json({"text": text}))
    for index, row in enumerate(refs):
        if (row.metadata.get("required") is not True or row.metadata.get("chunk_index") != index
                or row.metadata.get("chunk_count") != len(refs)
                or row.metadata.get("artifact_content_id") != artifact_cid
                or row.referenced_content_id != "sha256:" + _sha(row.summary)
                or row.byte_count != len(row.summary.encode())):
            raise SemanticTranslationError("native semantic chunk binding differs")
    if len(paths) != 1 or not next(iter(paths)):
        raise SemanticTranslationError("semantic artifact nomination is ambiguous")
    return wire, prompt[end:], positions, refs, text, next(iter(paths))


def _verified_semantic(root, *, artifact, text, task_id, current):
    from ..semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    from .semantic_context_runtime import load_semantic_worker_context, _source_scope_payload, _validate_program_payload
    from ...mcp_server.mcplusplus.kubo_cid import cid_for_bytes
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticCapsule
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import RepositorySnapshot

    raw = _read(root, artifact)
    if raw != text.encode():
        raise SemanticTranslationError("nominated semantic artifact differs from native prompt")
    payload = _parse(text)
    if (payload.get("schema") not in {"supervisor-semantic-worker-context@1", "supervisor-semantic-worker-context@2"}
            or payload.get("task_id") != task_id or payload.get("completion_authority") is not False
            or not isinstance(payload.get("manifest"), dict) or not 1 <= len(payload["manifest"]) <= 64):
        raise SemanticTranslationError("semantic task/schema/source binding differs")
    blocks = PurePosixPath(artifact).parent / "blocks"
    def block(cid):
        if not isinstance(cid, str) or PurePosixPath(cid).name != cid or cid in {".", ".."}:
            raise SemanticTranslationError("invalid producer block identity")
        return _read(root, str(blocks / cid))
    view = IpfsDatasetsSemanticStateProvider().open_verified_view(payload["semantic_root_cid"], block)
    if view.root.repository_id != cid_for_payload({"repository": str(root)}):
        raise SemanticTranslationError("semantic producer belongs to another repository")
    manifest = payload["manifest"]
    program = None
    if payload["schema"] == "supervisor-semantic-worker-context@2":
        try:
            program = _validate_program_payload(payload)
        except (ValueError, KeyError, TypeError) as error:
            raise SemanticTranslationError("explicit semantic program selection differs") from error
    elif "program_paths" in payload:
        raise SemanticTranslationError("explicit semantic program selection requires its versioned schema")
    if payload.get("scope_cid") != cid_for_payload(_source_scope_payload(manifest, program)):
        raise SemanticTranslationError("semantic source scope identity differs")
    # The native scanner retains its complete captured-file inventory as a
    # root-backed artifact fact. A caller-controlled manifest must not narrow
    # freshness checking while keeping capsules from a larger producer scope.
    snapshot_fact = view.artifact_fact("artifact:snapshot-evidence")
    snapshot = RepositorySnapshot.from_dict(snapshot_fact.artifact.metadata["snapshot"])
    if (snapshot.repository_id != view.root.repository_id
            or snapshot.snapshot_cid != view.root.producer.repository_snapshot_cid
            or snapshot_fact.source_cid != snapshot.snapshot_cid
            or any(entry.is_opaque for entry in snapshot.entries)
            or {entry.path: entry.source_cid for entry in snapshot.entries}
                != {path: binding.get("source_cid") for path, binding in manifest.items()
                    if isinstance(binding, dict) and (program is None or path in program)}
            or any(not isinstance(binding, dict) or set(binding) != {"sha256", "source_cid"}
                   for binding in manifest.values())):
        raise SemanticTranslationError("semantic manifest differs from complete producer source inventory")
    if not isinstance(payload.get("raw_sources"), dict) or not set(payload["raw_sources"]) <= set(manifest):
        raise SemanticTranslationError("semantic raw sources escape producer inventory")
    symbols = []
    for nominated in payload["capsules"]:
        capsule = SemanticCapsule.from_dict(nominated)
        if capsule.to_dict() != view.capsule(capsule.stable_symbol_id).to_dict():
            raise SemanticTranslationError("capsule differs from verified producer")
        symbols.append(capsule.stable_symbol_id)
    retained_sources = {}
    for path, binding in payload["manifest"].items():
        original = block(binding["source_cid"])
        retained_sources[path] = original
        if _sha(original) != binding["sha256"] or cid_for_bytes(original) != binding["source_cid"]:
            raise SemanticTranslationError("retained semantic source binding differs")
        if current and _read(root, path) != original:
            raise SemanticTranslationError("semantic translation source is stale")
        if path in payload.get("raw_sources", {}) and payload["raw_sources"][path] != original.decode():
            raise SemanticTranslationError("raw semantic source differs from retained bytes")
    if current:
        # Preserve the existing dispatch loader's bounds and refresh contract.
        load_semantic_worker_context(repository=root, artifact=artifact,
            expected_sha256=_sha(raw), task_id=task_id)
    elif program is not None:
        try:
            _validate_program_payload(payload, sources=retained_sources, repository=root)
        except (ValueError, KeyError, TypeError) as error:
            raise SemanticTranslationError("historical semantic program reconstruction differs") from error
    return payload, tuple(sorted(symbols))


def _slots(payload):
    """Closed identifier locations; never descend into source/literal metadata."""
    result = []
    def fields(record, prefix, allowed):
        for key in sorted(set(record) & allowed):
            value = record[key]
            if isinstance(value, str):
                result.append(((*prefix, key), value))
            elif isinstance(value, list):
                result.extend(((*prefix, key, index), item) for index, item in enumerate(value)
                              if isinstance(item, str))
    for index, capsule in enumerate(payload["capsules"]):
        fields(capsule, ("capsules", index), CAPSULE_FIELDS)
    for index, admission in enumerate(payload.get("admissions", [])):
        fields(admission, ("admissions", index), {"assessment_cid"})
        fields(admission.get("ref", {}), ("admissions", index, "ref"), REFERENCE_FIELDS)
    # ContextPack references are typed records; summary and metadata stay literal.
    pack = payload.get("pack", {})
    for key in ("references", "evidence"):
        for index, reference in enumerate(pack.get(key, [])):
            fields(reference, ("pack", key, index), REFERENCE_FIELDS)
    if len(result) > MAX_ITEMS:
        raise SemanticTranslationError("semantic identifier scope exceeds bound")
    return result


def _set(payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _get(payload, path):
    for key in path:
        payload = payload[key]
    return payload


def _encode(*, prompt: str, repository: Path, current: bool) -> EncodedSemanticPrompt:
    from ipfs_datasets_py.logic.families.translations import SymbolMapEntry
    root = Path(repository).absolute()
    if root.resolve(strict=True) != root:
        raise SemanticTranslationError("semantic repository must be canonical")
    wire, suffix, positions, refs, text, artifact = _prompt_parts(prompt)
    semantic, symbols = _verified_semantic(root, artifact=artifact, text=text,
                                           task_id=wire["objective_id"], current=current)
    slots = _slots(semantic)
    counts = Counter(value for _, value in slots)
    identities = sorted(value for value, count in counts.items() if count > 1 and len(value) >= 40)
    aliases = {identity: f"s{index}" for index, identity in enumerate(identities)}
    entries = [SymbolMapEntry(source_symbol_id=value, target_symbol_ids=(alias,),
        reason="lossless producer identifier representation only").to_dict() for value, alias in aliases.items()]
    translated = _parse(text)
    replaced = []
    for path, value in slots:
        if value in aliases:
            _set(translated, path, {REF_KEY: aliases[value]})
            replaced.append(list(path))
    templates = [{key: value for key, value in wire["evidence"][index].items() if key != "summary"}
                 for index in positions]
    core = {key: value for key, value in wire.items() if key != "evidence"}
    payload = {"schema": TABLE_SCHEMA, "source_repository": str(root), "task_id": wire["objective_id"],
        "scope_cid": semantic["scope_cid"], "semantic_root_cid": semantic["semantic_root_cid"],
        "semantic_artifact": artifact, "semantic_sha256": _sha(text), "native_prompt_sha256": _sha(prompt),
        "native_core_sha256": _sha(_json(core)), "native_suffix_sha256": _sha(suffix),
        "source_manifest": semantic["manifest"], "symbol_ids": list(symbols), "entries": entries,
        "replacement_paths": replaced, "reference_templates": templates, "semantic_positions": positions,
        "translated_semantic_sha256": _sha(_json(translated)), "semantic_equivalence_claimed": False,
        "execution_authority": False, "completion_authority": False}
    table = SemanticTranslationTable(_json(payload), cid_for_payload(payload))
    remaining = {**wire, "evidence": [row for index, row in enumerate(wire["evidence"]) if index not in positions]}
    transport = {"schema": TRANSPORT_SCHEMA, "translation_cid": table.translation_cid,
        "translation_table": {alias: value for value, alias in aliases.items()},
        "native_context": remaining, "translated_semantic": translated, "native_suffix": suffix,
        "instructions": INSTRUCTIONS, "execution_authority": False, "completion_authority": False}
    provider_prompt = _json(transport)
    if len(provider_prompt.encode()) > MAX_BYTES:
        raise SemanticTranslationError("translated provider prompt exceeds bound")
    receipt = {"schema": "supervisor-semantic-router-encoding@1", "translation_cid": table.translation_cid,
        "task_id": wire["objective_id"], "scope_cid": semantic["scope_cid"],
        "semantic_root_cid": semantic["semantic_root_cid"], "semantic_sha256": _sha(text),
        "native_prompt_sha256": _sha(prompt), "native_prompt_bytes": len(prompt.encode()),
        "provider_prompt_sha256": _sha(provider_prompt), "provider_prompt_bytes": len(provider_prompt.encode()),
        "identifier_mappings": len(entries), "identifier_occurrences": len(replaced),
        "freshness_checked": current, "semantic_equivalence_claimed": False,
        "execution_authority": False, "completion_authority": False}
    encoded = EncodedSemanticPrompt(prompt, provider_prompt, table, _json(receipt), current)
    if _restore(encoded.provider_prompt, table) != prompt:
        raise SemanticTranslationError("native prompt did not round-trip exactly")
    if current:
        for path, binding in semantic["manifest"].items():
            if _sha(_read(root, path)) != binding["sha256"]:
                raise SemanticTranslationError("source changed during translation")
    return encoded


def encode_semantic_router_prompt(*, prompt: str, repository: Path) -> EncodedSemanticPrompt:
    """Verify current sources and producer capsules before a provider dispatch."""
    return _encode(prompt=prompt, repository=repository, current=True)


def replay_semantic_router_prompt_for_audit(*, prompt: str, repository: Path) -> EncodedSemanticPrompt:
    """Reconstruct historical wire bytes from immutable artifacts; no freshness."""
    return _encode(prompt=prompt, repository=repository, current=False)


def _restore(provider_prompt: str, table: SemanticTranslationTable) -> str:
    wire, mapping = _parse(provider_prompt), _parse(table.payload_json)
    fields = {"schema", "translation_cid", "translation_table", "native_context", "translated_semantic",
              "native_suffix", "instructions", "execution_authority", "completion_authority"}
    aliases = {row["target_symbol_ids"][0]: row["source_symbol_id"] for row in mapping["entries"]}
    if (set(wire) != fields or wire["schema"] != TRANSPORT_SCHEMA
            or wire["translation_cid"] != table.translation_cid or wire["translation_table"] != aliases
            or wire["instructions"] != INSTRUCTIONS or wire["execution_authority"] is not False
            or wire["completion_authority"] is not False
            or _sha(_json(wire["translated_semantic"])) != mapping["translated_semantic_sha256"]):
        raise SemanticTranslationError("translated wire or mapping was tampered")
    semantic = wire["translated_semantic"]
    for path in mapping["replacement_paths"]:
        value = _get(semantic, path)
        if not isinstance(value, dict) or set(value) != {REF_KEY} or value[REF_KEY] not in aliases:
            raise SemanticTranslationError("unknown or ambiguous semantic reference")
        _set(semantic, path, aliases[value[REF_KEY]])
    text = _json(semantic)
    if _sha(text) != mapping["semantic_sha256"]:
        raise SemanticTranslationError("restored semantic bytes differ")
    raw, offset, refs = text.encode(), 0, []
    for template in mapping["reference_templates"]:
        end = offset + template["byte_count"]
        row = {**template, "summary": raw[offset:end].decode()}
        ContextReference.from_dict(row)
        refs.append(row)
        offset = end
    if offset != len(raw):
        raise SemanticTranslationError("semantic reference coverage differs")
    native = wire["native_context"]
    for position, reference in zip(mapping["semantic_positions"], refs):
        native["evidence"].insert(position, reference)
    result = _json(native) + wire["native_suffix"]
    if _sha(result) != mapping["native_prompt_sha256"]:
        raise SemanticTranslationError("native prompt reconstruction differs")
    return result


def restore_semantic_router_prompt(*, provider_prompt: str, table: SemanticTranslationTable,
                                   repository: Path) -> str:
    """Revalidate a current table and restore exact native prompt bytes."""
    native = _restore(provider_prompt, table)
    verified = encode_semantic_router_prompt(prompt=native, repository=repository)
    if verified.table != table or verified.provider_prompt != provider_prompt:
        raise SemanticTranslationError("translation table differs from current producer")
    return native


def decode_semantic_router_response(*, response: str, encoded: EncodedSemanticPrompt,
                                   repository: Path) -> DecodedSemanticResponse:
    """Keep prose literal; expand only an explicit native candidate envelope."""
    from ..residual_intelligence.structured_decoding import DecodeStatus, decode_structured_output, grammar_for
    if not isinstance(response, str) or len(response.encode()) > MAX_BYTES:
        raise SemanticTranslationError("provider response exceeds translation bound")
    try:
        value = _parse(response)
    except (ValueError, TypeError) as error:
        # A malformed explicit protocol response must not escape strict
        # validation by being relabelled as ordinary prose. The permissive
        # parser is used only to identify that reserved schema, never to decode.
        try:
            claimed = json.loads(response)
        except (ValueError, TypeError):
            claimed = None
        if isinstance(claimed, dict) and claimed.get("schema") == REPLY_SCHEMA:
            raise SemanticTranslationError("malformed reserved translation response") from error
        value = None
    receipt = {"schema": "supervisor-semantic-router-decoding@1", "translation_cid": encoded.table.translation_cid,
        "input_sha256": _sha(response), "input_bytes": len(response.encode()),
        "structured_translation": False, "freshness_checked": False,
        "candidate_only": True, "execution_authority": False, "completion_authority": False}
    if not isinstance(value, dict) or value.get("schema") != REPLY_SCHEMA:
        return DecodedSemanticResponse(response, _json({**receipt, "output_sha256": _sha(response)}))
    if not encoded.freshness_checked:
        raise SemanticTranslationError("historical replay cannot decode operational references")
    current = encode_semantic_router_prompt(prompt=encoded.native_prompt, repository=repository)
    if current.table != encoded.table or current.provider_prompt != encoded.provider_prompt:
        raise SemanticTranslationError("response translation table is stale or foreign")
    binding = _parse(encoded.table.payload_json)
    if (set(value) != {"schema", "translation_cid", "task_id", "semantic_root_cid", "scope_cid", "task_family", "response"}
            or value["translation_cid"] != encoded.table.translation_cid
            or any(value[key] != binding[key] for key in ("task_id", "semantic_root_cid", "scope_cid"))):
        raise SemanticTranslationError("structured response task/root/table binding differs")
    body = value["response"]
    if not isinstance(body, dict) or body.get("candidate_only") is not True:
        raise SemanticTranslationError("structured response must remain a native candidate")
    aliases = {row["target_symbol_ids"][0]: row["source_symbol_id"] for row in binding["entries"]}
    wire, *_ = _prompt_parts(encoded.native_prompt)
    allowed = set(aliases.values()) | {row["reference_id"] for row in wire["evidence"]}
    def resolve(items, *, symbols=False):
        if not isinstance(items, list) or len(items) > 64:
            raise SemanticTranslationError("reference response requires a bounded list")
        result = []
        for item in items:
            if isinstance(item, dict) and set(item) == {REF_KEY}:
                if not isinstance(item[REF_KEY], str) or item[REF_KEY] not in aliases:
                    raise SemanticTranslationError("unknown semantic response alias")
                item = aliases[item[REF_KEY]]
            if not isinstance(item, str) or item not in (set(binding["symbol_ids"]) if symbols else allowed):
                raise SemanticTranslationError("unknown semantic response identifier")
            result.append(item)
        return result
    if "evidence_references" in body:
        body["evidence_references"] = resolve(body["evidence_references"])
    structured = body.get("structured_payload")
    if not isinstance(structured, dict):
        raise SemanticTranslationError("native structured payload required")
    for key in tuple(structured):
        if key == "symbol_ids" or key.endswith("reference_ids"):
            structured[key] = resolve(structured[key], symbols=key == "symbol_ids")
    decoded = decode_structured_output(_json(body), grammar_for(value["task_family"]))
    if decoded.status is not DecodeStatus.VALID:
        raise SemanticTranslationError("translated response failed native candidate grammar")
    output = _json(body)
    return DecodedSemanticResponse(output, _json({**receipt, "structured_translation": True,
        "freshness_checked": True, "output_sha256": _sha(output), "output_bytes": len(output.encode()),
        "native_grammar_validated": True}))
