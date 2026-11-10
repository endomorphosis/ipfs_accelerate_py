"""Pure, lossless presentation of complete source-bound packet business data.

This codec neither opens native owners nor admits packet freshness. An owner
must independently bind its exact original live packet and this issued object.
Opaque integrity wrappers move to a pinned sidecar; payloads, reference roles,
requirements, effects, uncertainty and unknown business fields remain visible.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import weakref

MODE = PROFILE = "lossless_model_view@1"
VIEW_SCHEMA = "repository-world-packet-presentation@1"
SIDECAR_SCHEMA = "repository-world-packet-presentation-sidecar@1"
RECEIPT_SCHEMA = "repository-world-packet-presentation-receipt@1"
MAX_BYTES = 262144
MAX_SIDECAR_BYTES = 1024 * 1024
MAX_DEPTH = 32
MAX_NODES = 65536
MAX_FACTS = 2048
MAX_MEMBERS = 512
MAX_NORMALIZED_SCALAR_BYTES = 2 * 1024 * 1024
MIN_SHARED_BYTES = 128
MIN_JSON_DATA_BYTES = 128
FALSE_FIELDS = frozenset({"proof_authority", "execution_authority", "completion_authority",
    "omission_authority", "semantic_alignment_verified", "source_freshness_verified",
    "compiler_signature_verified", "world_dispatch_ready", "provider_launch",
    "provider_ingestion_verified", "model_consumption_verified", "model_tokens_measured",
    "total_token_savings_measured"})
EXPLANATION = (
    "Lossless business view: {$fact:N} means shared_facts[N]; {$json:V} is the exact "
    "canonical JSON DATA string of V; {$literal:[[key,value],...]} escapes literal "
    "marker keys. Roles and repeated occurrences remain distinct. Only packet/member "
    "integrity IDs and digests are in the pinned host sidecar. This adds no authority."
)
_TRANSPORT_SCHEMA = "repository-world-packet@1"
_PACKET_SCHEMA = "ipfs_accelerate_py/agent-supervisor/delta-task-provider-packet@1"
_PACKET_FIELDS = frozenset({"schema", "packet_id", "task_cid", "repository_id", "tree_id",
    "context_cid", "plan_cid", "policy_id", "policy_digest", "schema_revision", "evidence_digest",
    "counterexample_digest", "effect_scope", "requirements", "members", "frontier", "completeness",
    "data_label", "treat_as", "authority", "semantic_authority", "write_authority", "completion_authority",
    "nomination_only", "omitted_authority", "contains_credentials", "contains_secrets"})
_MEMBER_FIELDS = frozenset({"item_id", "kind", "digest", "summary", "path", "payload", "expansion_handle"})
_PACKET_CUSTODY = ("packet_id", "evidence_digest", "counterexample_digest")
_MEMBER_CUSTODY = ("item_id", "digest")
_MARKERS = frozenset({"$fact", "$json", "$literal"})
_SEAL = object()
_ISSUED = weakref.WeakKeyDictionary()
_LOCK = threading.RLock()


class RepositoryWorldPacketPresentationError(ValueError):
    """Closed bounded refusal codes with no source or prompt text."""


def _need(value, code):
    if not value:
        raise RepositoryWorldPacketPresentationError("world_presentation_" + code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _string_measure(value):
    _need(type(value) is str and len(value) <= MAX_SIDECAR_BYTES, "string_bound")
    size, escaped = 2, 2
    for char in value:
        point = ord(char)
        _need(not 0xD800 <= point <= 0xDFFF, "invalid_unicode")
        if char in '"\\':
            size += 2; escaped += 2
        elif char in "\b\f\n\r\t":
            size += 2; escaped += 1
        elif point < 32:
            size += 6; escaped += 1
        else:
            size += 1 if point < 128 else 2 if point < 2048 else 3 if point < 65536 else 4
        _need(size <= MAX_SIDECAR_BYTES, "string_bound")
    return size, escaped, 1, 0


def _combine(kind, parts):
    # Measures canonical UTF8 bytes, quote/backslash occurrences, virtual
    # materialization nodes and depth BEFORE allocating expanded JSON objects.
    size = 2 + max(0, len(parts) - 1)
    escaped, nodes, depth = 0, 1, 0
    for part in parts:
        size += part[0]; escaped += part[1]; nodes += part[2]; depth = max(depth, part[3] + 1)
    _need(nodes <= MAX_NODES and depth <= MAX_DEPTH and size <= MAX_SIDECAR_BYTES, "structure_bound")
    return size, escaped, nodes, depth


def _measure(value, depth=0):
    _need(depth <= MAX_DEPTH, "depth_bound")
    if type(value) is str: return _string_measure(value)
    if value is None: return 4, 0, 1, 0
    if type(value) is bool: return (4 if value else 5), 0, 1, 0
    if type(value) is int:
        _need(value.bit_length() <= 4096, "integer_bound")
        return len(str(value)), 0, 1, 0
    if type(value) is float:
        _need(math.isfinite(value), "nonfinite_number")
        return len(json.dumps(value, allow_nan=False)), 0, 1, 0
    if type(value) is list:
        _need(len(value) <= MAX_NODES, "nodes_bound")
        return _combine("list", [_measure(item, depth + 1) for item in value])
    _need(type(value) is dict and len(value) <= MAX_NODES and all(type(k) is str for k in value), "json_type")
    parts = []
    for key, item in value.items():
        key_m, item_m = _string_measure(key), _measure(item, depth + 1)
        parts.append((key_m[0] + 1 + item_m[0], key_m[1] + item_m[1], key_m[2] + item_m[2], max(key_m[3], item_m[3])))
    return _combine("object", parts)


def _wire(value, maximum=MAX_BYTES):
    _need(_measure(value)[0] <= maximum, "byte_bound")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8", "strict")


def _preflight(text):
    depth, nodes = 0, 0
    quoted = escaped = token = False
    for char in text:
        if quoted:
            if escaped: escaped = False
            elif char == "\\": escaped = True
            elif char == '"': quoted = False
            continue
        if char == '"': quoted = True; token = False; nodes += 1
        elif char in "[{": depth += 1; nodes += 1; token = False
        elif char in "]}": depth -= 1; token = False
        elif char in " \n\r\t,:": token = False
        elif not token: token = True; nodes += 1
        _need(0 <= depth <= MAX_DEPTH and nodes <= MAX_NODES, "json_preflight_bound")
    _need(not quoted and depth == 0, "invalid_json")


def _pairs(items):
    result = {}
    for key, value in items:
        _need(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _integer(text):
    _need(len(text) <= 1024, "integer_bound")
    return int(text)


def _number(text):
    value = float(text)
    _need(math.isfinite(value), "nonfinite_number")
    return value


def _decode(raw, maximum=MAX_BYTES):
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "byte_bound")
    try:
        text = raw.decode("utf-8", "strict")
        _preflight(text)
        value = json.loads(text, object_pairs_hook=_pairs, parse_int=_integer, parse_float=_number,
            parse_constant=lambda _v: (_ for _ in ()).throw(RepositoryWorldPacketPresentationError("world_presentation_nonfinite_number")))
        _need(_measure(value)[0] <= maximum, "byte_bound")
        return value
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise RepositoryWorldPacketPresentationError("world_presentation_invalid_json") from error


def _prompt_bytes(prompt):
    _need(type(prompt) is str and 0 < len(prompt) <= MAX_BYTES, "prompt_bound")
    try: raw = prompt.encode("utf-8", "strict")
    except UnicodeError as error: raise RepositoryWorldPacketPresentationError("world_presentation_invalid_unicode") from error
    _need(len(raw) <= MAX_BYTES, "prompt_bound")
    return raw


def _source(prompt):
    prefix, separator, body = prompt.partition("\n")
    if not separator or not body.startswith("{"): return None, "unsupported_transport"
    value = _decode(body.encode("utf-8"))
    if type(value) is not dict or set(value) != {"schema", "delivery_mode", "packet"}:
        return None, "unsupported_transport_fields"
    if value["schema"] != _TRANSPORT_SCHEMA or value["delivery_mode"] != "self_contained":
        return None, "unsupported_transport_schema"
    if _wire(value).decode() != body: return None, "noncanonical_transport"
    packet = value["packet"]
    if type(packet) is not dict or set(packet) != _PACKET_FIELDS or packet.get("schema") != _PACKET_SCHEMA:
        return None, "unsupported_packet_fields"
    members = packet.get("members")
    if type(members) is not list or len(members) > MAX_MEMBERS or any(type(row) is not dict or set(row) != _MEMBER_FIELDS for row in members):
        return None, "unsupported_member_fields"
    if not re.fullmatch(r"baguqeera[0-9a-f]{52}", packet["packet_id"] if type(packet["packet_id"]) is str else ""):
        return None, "unsupported_integrity_identity"
    for key in ("evidence_digest", "counterexample_digest"):
        if type(packet[key]) is not str or packet[key] and not re.fullmatch(r"sha256:[0-9a-f]{64}", packet[key]):
            return None, "unsupported_integrity_identity"
    for row in members:
        if (type(row["item_id"]) is not str or not re.fullmatch(r"delta-item:sha256:[0-9a-f]{64}", row["item_id"])
                or type(row["digest"]) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", row["digest"])):
            return None, "unsupported_integrity_identity"
    return (prefix, value), None


def _business(packet):
    custody = [["packet", key, packet[key]] for key in _PACKET_CUSTODY]
    business = {key: value for key, value in packet.items() if key not in _PACKET_CUSTODY and key != "members"}
    business["members"] = []
    for index, row in enumerate(packet["members"]):
        custody.extend(["member", index, key, row[key]] for key in _MEMBER_CUSTODY)
        business["members"].append({key: value for key, value in row.items() if key not in _MEMBER_CUSTODY})
    return business, custody


class _Group:
    __slots__ = ("index", "kind", "value", "children", "measure", "count")
    def __init__(self, index, kind, value, children, measure):
        self.index, self.kind, self.value, self.children, self.measure, self.count = index, kind, value, children, measure, 1


class _Builder:
    def __init__(self):
        self.groups = {}; self.nodes = self.scalar_bytes = self.json_strings = 0
        self.large_strings = self.large_string_bytes = 0
        self.pool = []; self.indices = {}; self.references = 0

    def node(self, value, depth=0, layer=0):
        self.nodes += 1
        _need(self.nodes <= MAX_NODES and depth <= MAX_DEPTH, "normalized_structure_bound")
        kind, children, scalar = type(value).__name__, (), None
        if type(value) is str:
            measure = _string_measure(value)
            scalar = _wire(value)
            self.scalar_bytes += len(scalar)
            _need(self.scalar_bytes <= MAX_NORMALIZED_SCALAR_BYTES, "normalized_scalar_bound")
            if layer == 0 and len(scalar) >= MIN_SHARED_BYTES:
                self.large_strings += 1; self.large_string_bytes += len(scalar)
            decoded = None
            if len(value) >= MIN_JSON_DATA_BYTES and value[:1] in {"{", "["}:
                try:
                    decoded = _decode(value.encode("utf-8"))
                    if type(decoded) not in {dict, list} or _wire(decoded).decode() != value: decoded = None
                except RepositoryWorldPacketPresentationError:
                    decoded = None  # This is an opaque business string, not a JSON transport.
            if decoded is not None:
                kind, children = "json", (self.node(decoded, depth + 1, layer + 1),)
                self.json_strings += 1
                key = (kind, children[0].index)
            else: key = (kind, scalar)
        elif type(value) is dict:
            children = tuple((key, self.node(value[key], depth + 1, layer)) for key in sorted(value))
            measure = _combine("object", [( _string_measure(key)[0] + 1 + child.measure[0],
                _string_measure(key)[1] + child.measure[1], 1 + child.measure[2], child.measure[3]) for key, child in children])
            key = (kind, tuple((key, child.index) for key, child in children))
        elif type(value) is list:
            children = tuple(self.node(item, depth + 1, layer) for item in value)
            measure = _combine("list", [child.measure for child in children]); key = (kind, tuple(child.index for child in children))
        else:
            measure = _measure(value); scalar = _wire(value); key = (kind, scalar)
        original = self.groups.get(key)
        if original is not None:
            original.count += 1
            return original
        group = _Group(len(self.groups), kind, value, children, measure)
        self.groups[key] = group
        return group

    def emit(self, group, *, shared=True):
        if (shared and group.count > 1 and group.measure[0] >= MIN_SHARED_BYTES
                and group.measure[0] * group.count > group.measure[0] + 16 * group.count + 8):
            if group.index not in self.indices:
                rendered = self.literal(group, shared=shared)
                _need(len(self.pool) < MAX_FACTS, "shared_fact_bound")
                # Each fact's encoded value is bounded before table retention.
                _wire(rendered)
                self.indices[group.index] = len(self.pool); self.pool.append(rendered)
            self.references += 1
            return {"$fact": self.indices[group.index]}
        return self.literal(group, shared=shared)

    def literal(self, group, *, shared):
        if group.kind == "dict":
            if any(key in _MARKERS for key, _child in group.children):
                return {"$literal": [[key, self.emit(child, shared=shared)] for key, child in group.children]}
            return {key: self.emit(child, shared=shared) for key, child in group.children}
        if group.kind == "list": return [self.emit(child, shared=shared) for child in group.children]
        if group.kind == "json": return {"$json": self.emit(group.children[0], shared=shared)}
        return group.value


def _build(prompt):
    original = _prompt_bytes(prompt)
    source, reason = _source(prompt)
    stats = dict(shared_fact_count=0, shared_reference_count=0, json_data_string_count=0,
        original_business_bytes=0, presented_business_bytes=0, host_custody_field_count=0,
        host_custody_bytes=0, original_large_string_occurrence_count=0, original_large_string_bytes=0)
    custody = []
    view = prompt
    if source is not None:
        prefix, native = source
        business, custody = _business(native["packet"])
        try:
            builder = _Builder(); root = builder.node(business)
            packed = builder.emit(root)
            wire = dict(schema=VIEW_SCHEMA, profile=PROFILE, packet=packed, shared_facts=builder.pool)
            # Compare complete view cost to the same typed view without interning.
            literal = dict(schema=VIEW_SCHEMA, profile=PROFILE, packet=builder.emit(root, shared=False), shared_facts=[])
            if _measure(wire)[0] >= _measure(literal)[0]:
                wire = literal; builder.pool = []; builder.references = 0
            candidate = prefix + "\n" + EXPLANATION + "\n" + _wire(wire).decode()
            _need(len(candidate.encode()) <= MAX_BYTES, "view_byte_bound")
        except RepositoryWorldPacketPresentationError as error:
            reason = str(error).removeprefix("world_presentation_")
            source = None; custody = []
        if source is not None:
            view = candidate
            stats.update(shared_fact_count=len(builder.pool), shared_reference_count=builder.references,
                json_data_string_count=builder.json_strings, original_business_bytes=len(_wire(business)),
                presented_business_bytes=len(_wire(wire)), host_custody_field_count=len(custody),
                host_custody_bytes=len(_wire(custody)), original_large_string_occurrence_count=builder.large_strings,
                original_large_string_bytes=builder.large_string_bytes)
    sidecar = dict(schema=SIDECAR_SCHEMA, profile=PROFILE, model_view_mode=MODE if source is not None else "original_fallback",
        fallback_reason=reason, original_prompt=prompt, original_prompt_sha256=_sha(original), original_prompt_bytes=len(original),
        presented_prompt_sha256=_sha(view.encode()), presented_prompt_bytes=len(view.encode()), custody=custody, stats=stats)
    sidecar_bytes = _wire(sidecar, MAX_SIDECAR_BYTES)
    receipt = dict(schema=RECEIPT_SCHEMA, profile=PROFILE, model_view_mode=sidecar["model_view_mode"],
        codec_supported=source is not None, fallback_reason=reason,
        **{key: sidecar[key] for key in ("original_prompt_sha256", "original_prompt_bytes", "presented_prompt_sha256", "presented_prompt_bytes")},
        sidecar_sha256=_sha(sidecar_bytes), sidecar_bytes=len(sidecar_bytes), **stats,
        sidecar_contains_original_prompt=True, presentation_structural_only=True, **dict.fromkeys(FALSE_FIELDS, False))
    return view, sidecar_bytes, _wire(receipt)


def _unpack_measure(value, facts, available, depth=0):
    _need(depth <= MAX_DEPTH, "expanded_depth_bound")
    if type(value) is dict and any(key in _MARKERS for key in value):
        _need(len(value) == 1, "marker_shape")
        if "$fact" in value:
            index = value["$fact"]
            _need(type(index) is int and 0 <= index < available, "shared_reference")
            return facts[index]
        if "$json" in value:
            child = _unpack_measure(value["$json"], facts, available, depth + 1)
            result = (2 + child[0] + child[1], 2 + 2 * child[1], child[2] + 1, child[3] + 1)
            _need(result[0] <= MAX_BYTES and result[2] <= MAX_NODES and result[3] <= MAX_DEPTH, "expanded_structure_bound")
            return result
        pairs = value["$literal"]
        _need(type(pairs) is list and len(pairs) <= MAX_NODES, "literal_shape")
        _need(all(type(pair) is list and len(pair) == 2 and type(pair[0]) is str for pair in pairs)
              and len({pair[0] for pair in pairs}) == len(pairs), "literal_shape")
        parts = []
        for key, child in pairs:
            left, right = _string_measure(key), _unpack_measure(child, facts, available, depth + 1)
            parts.append((left[0] + 1 + right[0], left[1] + right[1], left[2] + right[2], right[3]))
        return _combine("dict", parts)
    if type(value) is list:
        return _combine("list", [_unpack_measure(child, facts, available, depth + 1) for child in value])
    if type(value) is dict:
        parts = []
        for key, child in value.items():
            left, right = _string_measure(key), _unpack_measure(child, facts, available, depth + 1)
            parts.append((left[0] + 1 + right[0], left[1] + right[1], left[2] + right[2], right[3]))
        return _combine("dict", parts)
    return _measure(value)


def _unpack(value, facts):
    if type(value) is dict and "$fact" in value: return facts[value["$fact"]]
    if type(value) is dict and "$json" in value: return _wire(_unpack(value["$json"], facts)).decode()
    if type(value) is dict and "$literal" in value: return {key: _unpack(child, facts) for key, child in value["$literal"]}
    if type(value) is dict: return {key: _unpack(child, facts) for key, child in value.items()}
    if type(value) is list: return [_unpack(child, facts) for child in value]
    return value


def replay_repository_world_packet_presentation(*, view_prompt, sidecar_bytes, expected_sidecar_sha256, expected_original_sha256):
    """Reconstruct original bytes from visible facts plus independently pinned custody."""
    raw_view = _prompt_bytes(view_prompt)
    _need(type(sidecar_bytes) is bytes and len(sidecar_bytes) <= MAX_SIDECAR_BYTES, "sidecar_bound")
    for digest in (expected_sidecar_sha256, expected_original_sha256):
        _need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "independent_pin_required")
    _need(_sha(sidecar_bytes) == expected_sidecar_sha256, "sidecar_pin")
    sidecar = _decode(sidecar_bytes, MAX_SIDECAR_BYTES)
    _need(type(sidecar) is dict and set(sidecar) == {"schema", "profile", "model_view_mode", "fallback_reason",
        "original_prompt", "original_prompt_sha256", "original_prompt_bytes", "presented_prompt_sha256", "presented_prompt_bytes", "custody", "stats"}
        and sidecar["schema"] == SIDECAR_SCHEMA and sidecar["profile"] == PROFILE, "sidecar_schema")
    original = sidecar["original_prompt"]; raw_original = _prompt_bytes(original)
    _need(_sha(raw_original) == expected_original_sha256 == sidecar["original_prompt_sha256"]
          and type(sidecar["original_prompt_bytes"]) is int and len(raw_original) == sidecar["original_prompt_bytes"], "original_pin")
    _need(_sha(raw_view) == sidecar["presented_prompt_sha256"] and type(sidecar["presented_prompt_bytes"]) is int
          and len(raw_view) == sidecar["presented_prompt_bytes"], "view_pin")
    expected_view, expected_sidecar, _receipt = _build(original)
    _need(expected_view == view_prompt and expected_sidecar == sidecar_bytes, "fixed_projection_mismatch")
    if sidecar["model_view_mode"] == "original_fallback":
        _need(view_prompt == original, "fallback_mismatch")
        return original
    prefix, native = _source(original)[0]
    start = prefix + "\n" + EXPLANATION + "\n"
    _need(view_prompt.startswith(start), "guidance_mismatch")
    view = _decode(view_prompt[len(start):].encode())
    _need(type(view) is dict and set(view) == {"schema", "profile", "packet", "shared_facts"}
          and view["schema"] == VIEW_SCHEMA and view["profile"] == PROFILE, "view_schema")
    pool = view["shared_facts"]
    _need(type(pool) is list and len(pool) <= MAX_FACTS, "shared_fact_bound")
    measures = []
    for index, fact in enumerate(pool):
        measured = _unpack_measure(fact, measures, index)
        _need(measured[0] <= MAX_BYTES, "expanded_byte_bound")
        measures.append(measured)
    measured = _unpack_measure(view["packet"], measures, len(pool))
    _need(measured[0] <= MAX_BYTES and measured[2] <= MAX_NODES and measured[3] <= MAX_DEPTH, "expanded_structure_bound")
    facts = []  # No materialization precedes the complete expansion measurements.
    for fact in pool: facts.append(_unpack(fact, facts))
    business = _unpack(view["packet"], facts)
    expected_business, expected_custody = _business(native["packet"])
    _need(_wire(business) == _wire(expected_business) and sidecar["custody"] == expected_custody, "business_population_mismatch")
    # A shared fact may occur as two distinct member rows, or as an inert
    # nested business value. Custody belongs to each physical wrapper occurrence;
    # restore it on isolated shallow rows without mutating any shared payload.
    business = dict(business)
    business["members"] = [dict(row) for row in business["members"]]
    for row in sidecar["custody"]:
        if row[0] == "packet": business[row[1]] = row[2]
        else: business["members"][row[1]][row[2]] = row[3]
    restored = prefix + "\n" + _wire(dict(schema=_TRANSPORT_SCHEMA, delivery_mode="self_contained", packet=business)).decode()
    _need(restored == original and _sha(restored.encode()) == expected_original_sha256, "inverse_mismatch")
    return restored


class PreparedRepositoryWorldPacketPresentation:
    __slots__ = ("prompt", "sidecar_bytes", "_receipt_bytes", "_seal", "__weakref__")

    def __init__(self, seal, *, prompt, sidecar_bytes, receipt_bytes):
        _need(seal is _SEAL and type(self) is PreparedRepositoryWorldPacketPresentation, "original_issuance_required")
        for key, value in (("prompt", prompt), ("sidecar_bytes", sidecar_bytes), ("_receipt_bytes", receipt_bytes), ("_seal", seal)):
            object.__setattr__(self, key, value)

    def __setattr__(self, _key, _value):
        raise AttributeError("immutable packet presentation")

    def __delattr__(self, _key):
        raise AttributeError("immutable packet presentation")

    def _seal_material(self):
        return self._seal, self.prompt, self.sidecar_bytes, self._receipt_bytes

    def _retained(self):
        _need(type(self) is PreparedRepositoryWorldPacketPresentation and getattr(self, "_seal", None) is _SEAL,
              "original_presentation_required")
        with _LOCK:
            original = _ISSUED.get(self)
            _need(original is not None, "presentation_not_issued")
            _need(all(hasattr(self, key) for key in ("prompt", "sidecar_bytes", "_receipt_bytes", "_seal")),
                  "original_presentation_replaced")
            current = PreparedRepositoryWorldPacketPresentation._seal_material(self)
            _need(len(original) == len(current) and all(a is b for a, b in zip(original, current)), "original_presentation_replaced")

    def receipt(self):
        PreparedRepositoryWorldPacketPresentation._retained(self)
        return json.loads(self._receipt_bytes)

    def validate_original(self, original_prompt):
        PreparedRepositoryWorldPacketPresentation._retained(self)
        receipt = PreparedRepositoryWorldPacketPresentation.receipt(self)
        raw = _prompt_bytes(original_prompt)
        _need(_sha(raw) == receipt["original_prompt_sha256"] and len(raw) == receipt["original_prompt_bytes"], "original_pin")
        restored = replay_repository_world_packet_presentation(view_prompt=self.prompt, sidecar_bytes=self.sidecar_bytes,
            expected_sidecar_sha256=receipt["sidecar_sha256"], expected_original_sha256=receipt["original_prompt_sha256"])
        _need(restored == original_prompt, "inverse_mismatch")
        PreparedRepositoryWorldPacketPresentation._retained(self)
        return receipt


def prepare_repository_world_packet_presentation(prompt):
    view, sidecar, receipt = _build(prompt)
    result = PreparedRepositoryWorldPacketPresentation(_SEAL, prompt=view, sidecar_bytes=sidecar, receipt_bytes=receipt)
    with _LOCK: _ISSUED[result] = PreparedRepositoryWorldPacketPresentation._seal_material(result)
    PreparedRepositoryWorldPacketPresentation.validate_original(result, prompt)
    return result
