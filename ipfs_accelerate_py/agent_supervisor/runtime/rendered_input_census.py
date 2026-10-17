"""Bounded, content-free census of an already verified router user message.

The caller owns delivery evidence. This module reads no files, changes no prompt
and grants no omission or execution authority. Canonical JSON repetition is a
separate observation from literal input bytes, and neither measures saved tokens.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

SCHEMA = "supervisor-rendered-input-census@1"
MAX_INPUT_BYTES = 256_000
MAX_NODES = 20_000
MAX_DEPTH = 32
MAX_SERIALIZATION_BYTES = 8 * 1024 * 1024
MAX_GROUPS = 32
MAX_RESULT_BYTES = 65_536


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _invalid():
    # No exception text includes untrusted keys, paths or prompt contents.
    raise ValueError("rendered input census requires bounded valid input")


def _utf8(value, limit):
    if type(value) is not str or len(value) > limit:
        _invalid()
    try:
        raw = value.encode("utf-8")
    except UnicodeError:
        _invalid()
    if len(raw) > limit:
        _invalid()
    return raw


def _json_prefix(text: str):
    """Bound nesting before the decoder allocates; suffixes are opaque text."""
    begin = len(text) - len(text.lstrip(" \t\r\n"))
    if begin == len(text) or text[begin] != "{":
        _invalid()
    quoted = escaped = False
    nesting = 0
    end = None
    for index in range(begin, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            nesting += 1
            if nesting > MAX_DEPTH + 1:
                _invalid()
        elif char in "]}":
            nesting -= 1
            if nesting == 0:
                end = index + 1
                break
    if end is None:
        _invalid()

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                _invalid()
            value[key] = item
        return value

    def constant(_):
        _invalid()

    try:
        value = json.loads(text[begin:end], object_pairs_hook=unique, parse_constant=constant)
    except (ValueError, RecursionError, OverflowError):
        _invalid()
    if type(value) is not dict:
        _invalid()
    return value, text[:end], text[end:]


class _EncodingBudget:
    def __init__(self):
        self.used = 0

    def encode(self, value):
        try:
            raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
            _invalid()
        self.used += len(raw)
        if self.used > MAX_SERIALIZATION_BYTES:
            _invalid()
        return raw


class _Occupied:
    """Disjoint DFS intervals, with bounded logarithmic overlap queries."""
    def __init__(self, size):
        self.tree = [0] * (size + 1)

    def _prefix(self, end):
        total = 0
        while end:
            total += self.tree[end]
            end -= end & -end
        return total

    def intersects(self, start, end):
        return self._prefix(end) != self._prefix(start)

    def reserve(self, start, end):
        # Selected intervals never overlap, so each node is updated at most once.
        for node in range(start + 1, end + 1):
            while node < len(self.tree):
                self.tree[node] += 1
                node += node & -node


def _overlaps(value, budget):
    containers, strings = defaultdict(list), defaultdict(int)
    pending = [(False, value, 0, None)]
    nodes = 0
    while pending:
        closing, current, depth, record = pending.pop()
        if closing:
            record[1] = nodes
            continue
        start = nodes
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            _invalid()
        if type(current) in (dict, list):
            raw = budget.encode(current)
            record = [start, None]
            if current and len(raw) >= 64:
                # Retain bytes in the key: hash equality alone is insufficient.
                containers[(_sha(raw), raw)].append(record)
            pending.append((True, None, depth, record))
            children = list(current.values()) if type(current) is dict else current
            pending.extend((False, child, depth + 1, None) for child in reversed(children))
        elif type(current) is str:
            raw = budget.encode(current)
            if len(raw) >= 128:
                strings[(_sha(raw), raw)] += 1

    occupied = _Occupied(nodes)
    selected = []
    selected_count = total = 0
    for (digest, raw), intervals in sorted(containers.items(),
            key=lambda item: (-len(item[0][1]), item[0][0], item[0][1])):
        available = [(start, end) for start, end in intervals if not occupied.intersects(start, end)]
        if len(available) < 2:
            continue
        for start, end in available:
            occupied.reserve(start, end)
        repeated = (len(available) - 1) * len(raw)
        total += repeated
        selected_count += 1
        if len(selected) < MAX_GROUPS:
            selected.append({"value_sha256": digest, "canonical_bytes": len(raw),
                "occurrences": len(available), "repeated_value_bytes": repeated})
    structured = {"groups": selected, "selected_repeated_value_bytes": total,
        "groups_truncated": selected_count > MAX_GROUPS, "selected_group_count": selected_count,
        "minimum_canonical_bytes": 64, "selection": "largest_first_disjoint_including_representatives",
        "literal_input_byte_coverage": None, "safe_to_remove_bytes": None}
    repeated_strings = []
    string_count = string_total = 0
    for (digest, raw), count in sorted(strings.items(), key=lambda item: (-len(item[0][1]), item[0][0], item[0][1])):
        if count < 2:
            continue
        repeated = (count - 1) * len(raw)
        string_total += repeated
        string_count += 1
        if len(repeated_strings) < MAX_GROUPS:
            repeated_strings.append({"value_sha256": digest, "canonical_bytes": len(raw),
                "occurrences": count, "repeated_value_bytes": repeated})
    long_strings = {"groups": repeated_strings, "selected_repeated_value_bytes": string_total,
        "groups_truncated": string_count > MAX_GROUPS, "selected_group_count": string_count,
        "minimum_canonical_bytes": 128, "additive_with_json_subtrees": False,
        "literal_input_byte_coverage": None, "safe_to_remove_bytes": None}
    return structured, long_strings, nodes


def _partition(value, budget, canonical_size):
    fields = ("goal", "scope", "authority", "acceptance", "native_context",
        "translated_semantic", "translation_table", "instructions")
    result = {key: 0 for key in (*fields, "evidence_summaries", "evidence_metadata",
        "evidence_other", "other_top_level_values", "keys_and_framing")}
    total = 0
    for key, item in value.items():
        size = len(budget.encode(item))
        total += size
        if key == "evidence":
            if type(item) is list and all(type(row) is dict for row in item):
                for row in item:
                    for field, category in (("summary", "evidence_summaries"), ("metadata", "evidence_metadata")):
                        if field in row:
                            result[category] += len(budget.encode(row[field]))
            result["evidence_other"] += size - result["evidence_summaries"] - result["evidence_metadata"]
        else:
            result[key if key in fields else "other_top_level_values"] += size
    result["keys_and_framing"] = canonical_size - total
    return result


def census_rendered_input(*, model_prompt: str, workspace_advisory: str, invocation_id: str,
                          input_representation: str | None = None) -> dict:
    """Measure one exact, caller-verified user message; never a session total.

    The JSON prefix is the literal router context after its exact advisory.
    Explicit world input additionally separates its fixed guidance and checks
    the canonical envelope; the caller still owns complete packet validation.
    Canonical subtree and long-string equality are independently nonadditive
    observations. Output contains no caller-provided keys, paths or bodies.
    """
    model = _utf8(model_prompt, MAX_INPUT_BYTES)
    advisory = _utf8(workspace_advisory, 8192)
    identity = _utf8(invocation_id, 512)
    if not model or not identity or not model_prompt.startswith(workspace_advisory):
        _invalid()
    if input_representation is not None and (
            type(input_representation) is not str or input_representation != "world_task_packet"):
        _invalid()
    router = model_prompt[len(workspace_advisory):]
    guidance = b""
    if input_representation == "world_task_packet":
        from .repository_world_packet import GUIDANCE, SCHEMA as WORLD_SCHEMA
        from ..prompt.delta_task_packet import PROVIDER_FACING_PACKET_SCHEMA
        preamble = GUIDANCE + "\n"
        if not router.startswith(preamble):
            _invalid()
        guidance = preamble.encode("utf-8")
        router = router[len(preamble):]
    value, prefix, suffix = _json_prefix(router)
    if input_representation == "world_task_packet":
        packet = value.get("packet")
        if (set(value) != {"schema", "delivery_mode", "packet"}
                or value["schema"] != WORLD_SCHEMA or value["delivery_mode"] != "self_contained"
                or type(packet) is not dict or packet.get("schema") != PROVIDER_FACING_PACKET_SCHEMA
                or type(packet.get("members")) is not list or len(packet["members"]) > 512
                or any(type(member) is not dict for member in packet["members"])
                or any(packet.get(name) is not False for name in (
                    "semantic_authority", "write_authority", "completion_authority",
                    "contains_credentials", "contains_secrets"))):
            _invalid()
    budget = _EncodingBudget()
    structured, long_strings, nodes = _overlaps(value, budget)
    canonical = budget.encode(value)
    if input_representation == "world_task_packet" and prefix.encode("utf-8") != canonical:
        _invalid()
    partition = _partition(value, budget, len(canonical))
    prefix_raw, suffix_raw = prefix.encode(), suffix.encode()
    result = {"schema": SCHEMA, "invocation_id_sha256": _sha(identity),
        "model_prompt_sha256": _sha(model), "model_prompt_bytes": len(model),
        "disjoint_byte_partition": {"workspace_advisory": len(advisory),
            **({"world_packet_guidance": len(guidance)} if input_representation is not None else {}),
            "json_prefix": len(prefix_raw), "router_suffix": len(suffix_raw)},
        "byte_partition_reconciles": len(advisory) + len(guidance) + len(prefix_raw) + len(suffix_raw) == len(model),
        "json_prefix": {"sha256": _sha(prefix_raw), "bytes": len(prefix_raw),
            "canonical_bytes": len(canonical), "is_canonical": prefix_raw == canonical},
        "canonical_byte_partition": partition,
        "canonical_partition_reconciles": sum(partition.values()) == len(canonical),
        "repeated_json_subtrees": structured, "repeated_long_strings": long_strings,
        "json_nodes": nodes, "serialization_work_bytes": budget.used,
        "content_tokens": None, "provider_usage_tokens": None, "safe_to_remove_bytes": None,
        "raw_content_exported": False, "prompt_modified": False,
        "omission_authority": False, "execution_authority": False, "completion_authority": False,
        "scope": "one verified router user message; excludes CLI/system context and later tool history",
        "nested_json_strings_inspected": False, "literal_duplicate_substrings_mined": False,
        "overlap_totals_are_additive": False, "token_savings_established": False}
    if input_representation is not None:
        result.update(input_representation=input_representation,
            world_packet_guidance={"sha256": _sha(guidance), "bytes": len(guidance)})
    if len(json.dumps(result, sort_keys=True, allow_nan=False).encode()) > MAX_RESULT_BYTES:
        _invalid()
    # This is the already selected, caller-verified message. Observing its
    # declared raw-source slots never restores an omitted candidate or searches
    # arbitrary source-like text. Optional enrichment yields its budget before
    # the existing census, and cannot change the input audit's verdict.
    try:
        from .inline_source_observation import observe_inline_sources, observe_world_inline_sources
        options = dict(model_prompt=model_prompt, workspace_advisory=workspace_advisory,
                       invocation_id=invocation_id)
        inline = (observe_world_inline_sources(**options)
                  if input_representation == "world_task_packet" else observe_inline_sources(**options))
        enriched = {**result, "inline_sources": inline}
        if input_representation == "world_task_packet":
            # Repetition accounting never decodes nested strings. The optional
            # source join can inspect one declared complete semantic artifact;
            # an unavailable result does not establish whether it was decoded.
            enriched.update(nested_json_strings_inspected=True if inline["status"] == "observed" else None,
                nested_json_inspection_scope="declared_complete_semantic_context_only",
                repetition_census_nested_json_strings_inspected=False)
        if len(json.dumps(enriched, sort_keys=True, allow_nan=False).encode()) <= MAX_RESULT_BYTES:
            result = enriched
    except Exception:
        # Content-bearing exceptions remain private. Absence is unknown to the
        # downstream join, never an empty source population.
        pass
    return result
