"""Actual local tokenization of complete authored MCP replies; no providers."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_token_budget as budget
from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import load_owner_prompt_tokenizer
from test.api.test_local_prompt_tokenizer import make_binding, MODEL


@pytest.fixture
def local(tmp_path):
    artifact, digest, _descriptor = make_binding(tmp_path)
    return load_owner_prompt_tokenizer(artifact=artifact, expected_sha256=digest,
                                      provider="codex_cli", model=MODEL)


def wire(content, representation, *, encoding="utf-8", extra=None):
    payload = dict(generation=1, path="source.py", source_sha256="a" * 64,
        representation=representation, encoding=encoding, content=content, proof_authority=False)
    if extra:
        payload.update(extra)
    response = dict(jsonrpc="2.0", id=7, result=dict(content=[dict(type="text",
        text=json.dumps(payload, separators=(",", ":"), ensure_ascii=False))], isError=False))
    return json.dumps(response, separators=(",", ":"), ensure_ascii=False) + "\n"


def choose(local, original, compact, **kwargs):
    return budget.select_repository_reply(original_reply=original, compact_reply=compact,
        requested_representation=kwargs.pop("requested_representation", "compact"), tokenizer=local, **kwargs)


def test_complete_reply_strict_reduction_uses_verified_owner_tokenizer(local):
    original, compact = wire("hello world " * 30, "original"), wire("hello", "compact")
    receipt = choose(local, original, compact)
    assert receipt["selected_representation"] == "compact"
    assert receipt["measurement_status"] == "measured"
    assert receipt["selected_tokens"] == local.count(compact)
    assert receipt["candidates"]["original"]["tokens"] == local.count(original)
    assert receipt["selected_reply_sha256"] == hashlib.sha256(compact.encode()).hexdigest()
    assert receipt["selected_reply_bytes"] == len(compact.encode())
    assert receipt["tokenizer_binding"] == local.binding()
    assert receipt["tokenizer_provenance"] == local.provenance()
    assert receipt["provider_usage_measured"] is receipt["provider_framing_measured"] is False
    assert receipt["provider_tokenizer_equivalence_verified"] is False
    assert receipt["total_session_token_savings"] is None
    assert receipt["proof_authority"] is receipt["completion_authority"] is False
    assert "hello" not in json.dumps(receipt)


def test_byte_smaller_but_token_larger_retains_original(local):
    original = wire("abcdefghijklmnopqrstuvwxyz" * 15, "original")
    compact = wire("a + b + c + d", "compact")
    assert len(compact.encode()) < len(original.encode())
    assert local.count(compact) > local.count(original)
    receipt = choose(local, original, compact)
    assert receipt["selected_representation"] == "original"
    assert receipt["selection_reason"] == "original_not_larger"


def test_ties_prefer_original_even_when_compact_has_fewer_bytes(local):
    original, compact = wire("longsingleword" * 15, "original"), wire("hello", "compact")
    assert len(compact) < len(original) and local.count(compact) == local.count(original)
    receipt = choose(local, original, compact)
    assert receipt["selected_representation"] == "original"
    assert receipt["measurement_status"] == "measured"


def test_full_envelope_overhead_can_reverse_a_content_only_win(local):
    original = wire("hello world", "original")
    compact = wire("hello", "compact", extra={"representation_dictionary": ["world"] * 20})
    assert local.count("hello") < local.count("hello world")
    assert local.count(compact) > local.count(original)
    receipt = choose(local, original, compact)
    assert receipt["selected_representation"] == "original"
    assert receipt["candidates"]["compact"]["tokens"] == local.count(compact)


def test_exact_envelope_and_newline_are_counted_once_each(local, monkeypatch):
    original, compact = wire("hello world café 法律", "original"), wire("hello", "compact")
    count = type(local).count
    calls = []
    def observed(self, text):
        calls.append(text)
        return count(self, text)
    monkeypatch.setattr(type(local), "count", observed)
    receipt = choose(local, original, compact)
    assert calls == [original, compact]
    assert all(text.endswith("\n") and '"jsonrpc":"2.0"' in text for text in calls)
    assert receipt["candidates"]["original"]["reply_bytes"] == len(original.encode())


def test_explicit_original_honors_request_without_unused_count_or_candidate(local, monkeypatch):
    def forbidden(*_args):
        raise AssertionError("explicit original must not tokenize")
    monkeypatch.setattr(type(local), "count", forbidden)
    original = wire("hello world", "original")
    receipt = choose(local, original, object(), requested_representation="original")
    assert receipt["selected_representation"] == "original"
    assert receipt["selection_reason"] == "original_requested"
    assert receipt["measurement_status"] == "not_requested"
    assert receipt["selected_tokens"] is None
    assert set(receipt["candidates"]) == {"original"}


@pytest.mark.parametrize("tokenizer", [None, object(), lambda text: 1])
def test_missing_or_foreign_counter_keeps_original_without_savings(tokenizer):
    receipt = choose(tokenizer, wire("hello world", "original"), wire("hello", "compact"))
    assert receipt["selected_representation"] == "original"
    assert receipt["measurement_status"] == ("unavailable" if tokenizer is None else "failed")
    assert receipt["selected_tokens"] is None
    assert all(row["tokens"] is None for row in receipt["candidates"].values())


@pytest.mark.parametrize("bad", [None, True, False, 0, -1, 1.0, float("nan"), float("inf"), "2", budget.MAX_REPLY_TOKENS + 1])
def test_invalid_count_never_releases_partial_measurement(local, monkeypatch, bad):
    calls = iter([5, bad])
    monkeypatch.setattr(type(local), "count", lambda *_args: next(calls))
    receipt = choose(local, wire("hello world", "original"), wire("hello", "compact"))
    assert receipt["selected_representation"] == "original"
    assert receipt["measurement_status"] == "failed"
    assert all(row["tokens"] is None for row in receipt["candidates"].values())
    assert receipt["tokenizer_binding"] is receipt["tokenizer_provenance"] is None


def test_counter_exception_is_content_free_and_falls_back(local, monkeypatch):
    def broken(*_args):
        raise RuntimeError("private content: DO_NOT_RETAIN")
    monkeypatch.setattr(type(local), "count", broken)
    receipt = choose(local, wire("hello world", "original"), wire("hello", "compact"))
    assert receipt["selection_reason"] == "tokenizer_failed"
    assert "DO_NOT_RETAIN" not in json.dumps(receipt)


@pytest.mark.parametrize("target", ["descriptor", "asset"])
def test_changed_local_assets_after_load_refuse_compact(local, target):
    provenance = local.provenance()
    path = Path(provenance["descriptor_path" if target == "descriptor" else "asset_path"])
    path.write_bytes(path.read_bytes() + b" ")
    receipt = choose(local, wire("hello world", "original"), wire("hello", "compact"))
    assert receipt["measurement_status"] == "failed"
    assert receipt["selected_representation"] == "original"
    assert receipt["selected_tokens"] is None


def test_change_between_counts_clears_both_counts(local, monkeypatch):
    count = type(local).count
    calls = []
    asset = Path(local.provenance()["asset_path"])
    def mutate(self, text):
        calls.append(text)
        result = count(self, text)
        if len(calls) == 1:
            asset.write_bytes(asset.read_bytes() + b" ")
        return result
    monkeypatch.setattr(type(local), "count", mutate)
    receipt = choose(local, wire("hello world", "original"), wire("hello", "compact"))
    assert len(calls) == 2
    assert receipt["measurement_status"] == "failed"
    assert all(row["tokens"] is None for row in receipt["candidates"].values())


def test_unchanged_binary_passthrough_stays_original_and_exact(local):
    raw = b"\x00\xff\xfe\x01authored binary\n"
    encoded = base64.b64encode(raw).decode()
    original, compact = wire(encoded, "original", encoding="base64"), wire(encoded, "compact", encoding="base64")
    receipt = choose(local, original, compact)
    assert receipt["selected_representation"] == "original"
    assert receipt["selected_reply_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    payload = json.loads(json.loads(original)["result"]["content"][0]["text"])
    assert base64.b64decode(payload["content"]) == raw


def test_unavailable_compact_does_not_create_a_comparison(local):
    receipt = choose(local, wire("hello", "original"), None)
    assert receipt["selection_reason"] == "compact_unavailable"
    assert receipt["measurement_status"] == "not_requested"


def test_large_valid_wire_falls_back_at_existing_tokenizer_bound(local, monkeypatch):
    calls = []
    monkeypatch.setattr(type(local), "count", lambda *_args: calls.append(True) or 1)
    original = wire("x" * budget.MAX_PROMPT_BYTES, "original")
    receipt = choose(local, original, wire("hello", "compact"))
    assert budget.MAX_PROMPT_BYTES < len(original.encode()) < budget.MAX_REPLY_BYTES
    assert receipt["selection_reason"] == "tokenizer_input_bound"
    assert receipt["measurement_status"] == "unavailable" and not calls
    assert receipt["selected_representation"] == "original"


@pytest.mark.parametrize("text", [None, b"{}\n", "", "\n", "{}", "{}\n\n", "{\n}\n", "\ud800\n",
    "x" * budget.MAX_REPLY_BYTES + "\n", "é" * (budget.MAX_REPLY_BYTES // 2 + 1) + "\n"])
def test_invalid_or_unbounded_wire_refused_before_counter(local, text):
    with pytest.raises(ValueError):
        choose(local, text, wire("hello", "compact"))


@pytest.mark.parametrize("value", [None, True, [], "auto", "minified"])
def test_representation_selection_is_closed(local, value):
    with pytest.raises(ValueError, match="explicit original or compact"):
        choose(local, wire("hello", "original"), None, requested_representation=value)


def binding():
    return dict(artifact="/authored/owner-tokenizer.json", expected_sha256="a" * 64,
                provider="codex_cli", model="authored-model")


def test_tokenizer_binding_validation_is_inert_and_returns_copy():
    value = binding()
    result = budget.validate_tokenizer_binding(value)
    assert result == value and result is not value


@pytest.mark.parametrize("change", [
    {"artifact": "relative.json"}, {"artifact": "/a/../b"}, {"artifact": "/a//b"},
    {"artifact": "/a/./b"}, {"artifact": "https://example.invalid/file.json"},
    {"artifact": "/file\n.json"}, {"artifact": "/file\\name"}, {"artifact": "/\ud800"},
    {"artifact": "/" + "a" * 4096}, {"artifact": "/" + "é" * 2500},
    {"provider": ""}, {"provider": True}, {"provider": " codex_cli"}, {"provider": "x\n"},
    {"model": ""}, {"model": "x" * 257}, {"model": "\ud800"},
    {"expected_sha256": "A" * 64}, {"expected_sha256": True}, {"extra": 1},
])
def test_invalid_binding_does_not_admit_asset_or_network_lookup(change):
    with pytest.raises(ValueError):
        budget.validate_tokenizer_binding({**binding(), **change})


@pytest.mark.parametrize("value", [None, True, [], {}, {"artifact": "/file"}])
def test_closed_tokenizer_binding(value):
    with pytest.raises(ValueError, match="closed repository tokenizer binding"):
        budget.validate_tokenizer_binding(value)


@pytest.fixture
def audit_path(tmp_path):
    directory = tmp_path / "private-audit"
    directory.mkdir(mode=0o700)
    return directory / "token-selections.jsonl"


def measured(local):
    return choose(local, wire("hello world " * 20, "original"), wire("hello", "compact"))


def owner_config(local):
    source = local.provenance()
    return dict(artifact=source["descriptor_path"], expected_sha256=source["descriptor_sha256"],
                provider=local.provider, model=local.model)


def closed_audit(path, receipts):
    audit = budget.SelectionAudit(path)
    for receipt in receipts:
        audit.record(receipt)
    audit.close()
    return audit


def replace_population(path, rows):
    """Keep the non-authoritative journal checksum valid for semantic forgeries."""
    chunks = [json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode() + b"\n" for row in rows]
    footer = dict(schema=budget.AUDIT_SCHEMA, event="closed", read_count=len(chunks),
                  records_sha256=hashlib.sha256(b"".join(chunks)).hexdigest())
    path.write_bytes(b"".join(chunks) + json.dumps(footer, separators=(",", ":")).encode() + b"\n")


def test_audit_replays_complete_population_without_reply_contents(local, audit_path):
    compact = measured(local)
    original = choose(local, wire("longsingleword" * 5, "original"), wire("a + b + c", "compact"))
    explicit = choose(local, wire("hello", "original"), None, requested_representation="original")
    unavailable = choose(None, wire("hello", "original"), wire("hello", "compact"))
    absent_compact = choose(local, wire("hello", "original"), None)
    closed_audit(audit_path, [compact, original, explicit, unavailable, absent_compact])
    assert stat.S_IMODE(audit_path.stat().st_mode) == 0o600
    summary = budget.summarize_selection_audit(audit_path, tokenizer_binding=owner_config(local))
    assert summary == dict(schema=budget.SUMMARY_SCHEMA, status="valid", audit_closed=True,
        read_count=5, compact_selected=1, original_selected=4, measured_comparisons=2,
        unmeasured_fallbacks=2, explicit_original=1,
        selected_reply_tokens=compact["selected_tokens"] + original["selected_tokens"],
        provider_usage_measured=False)
    assert budget.validate_selection_summary(summary) == summary
    raw = audit_path.read_text()
    assert "hello world" not in raw and "longsingleword" not in raw
    assert "records_sha256" in raw


def test_closed_empty_audit_has_no_adoption_or_token_claim(audit_path):
    closed_audit(audit_path, [])
    summary = budget.summarize_selection_audit(audit_path)
    assert summary["status"] == "valid" and summary["audit_closed"] is True
    assert all(summary[key] == 0 for key in budget._SUMMARY_COUNTERS)
    assert summary["provider_usage_measured"] is False


def test_open_or_missing_audit_never_qualifies(local, audit_path):
    missing = budget.summarize_selection_audit(audit_path)
    assert missing["status"] == "missing" and missing["audit_closed"] is False
    audit = budget.SelectionAudit(audit_path)
    try:
        audit.record(measured(local))
        opened = budget.summarize_selection_audit(audit_path)
        assert opened["status"] == "invalid" and opened["read_count"] is None
        with pytest.raises(ValueError):
            budget.validate_selection_summary(opened)
    finally:
        audit.close()
    assert budget.summarize_selection_audit(audit_path)["status"] == "valid"
    with pytest.raises(ValueError):
        budget.validate_selection_summary(missing)


def test_close_is_idempotent_and_record_after_close_is_refused(local, audit_path):
    audit = closed_audit(audit_path, [measured(local)])
    before = audit_path.read_bytes()
    audit.close()
    assert audit_path.read_bytes() == before
    with pytest.raises(ValueError, match="closed"):
        audit.record(measured(local))


@pytest.mark.parametrize("key,value", [
    ("selected_representation", "original"), ("selected_tokens", True),
    ("selected_tokens", 0), ("selected_reply_sha256", "b" * 64),
    ("selected_reply_bytes", True), ("measurement_status", "unavailable"),
    ("selection_reason", "original_not_larger"), ("provider_usage_measured", True),
    ("provider_framing_measured", True), ("provider_tokenizer_equivalence_verified", True),
    ("proof_authority", True), ("completion_authority", True),
    ("total_session_token_savings", 3), ("count_scope", "body_only"),
    ("max_reply_tokens", True), ("max_tokenizer_input_bytes", True),
    ("source_contents", "DO_NOT_RETAIN"),
])
def test_semantic_receipt_forgery_rejected_even_with_recomputed_footer(local, audit_path, key, value):
    closed_audit(audit_path, [measured(local)])
    row = json.loads(audit_path.read_bytes().splitlines()[0])
    row["selection"][key] = value
    replace_population(audit_path, [row])
    summary = budget.summarize_selection_audit(audit_path)
    assert summary["status"] == "invalid" and summary["read_count"] is None


@pytest.mark.parametrize("change", ["candidate_count", "candidate_digest", "candidate_bytes", "extra_candidate",
    "partial_pair", "false_tie", "tokenizer_revision", "tokenizer_asset_hash", "tokenizer_extra",
    "provenance_extra", "provenance_claim", "relative_asset", "schema", "sequence", "row_extra"])
def test_nested_audit_inconsistencies_fail_closed(local, audit_path, change):
    closed_audit(audit_path, [measured(local)])
    row = json.loads(audit_path.read_bytes().splitlines()[0])
    selection = row["selection"]
    if change == "candidate_count":
        selection["candidates"]["original"]["tokens"] = budget.MAX_REPLY_TOKENS + 1
    elif change == "candidate_digest":
        selection["candidates"]["original"]["reply_sha256"] = "A" * 64
    elif change == "candidate_bytes":
        selection["candidates"]["original"]["reply_bytes"] = budget.MAX_PROMPT_BYTES + 1
    elif change == "extra_candidate":
        selection["candidates"]["other"] = dict(selection["candidates"]["original"])
    elif change == "partial_pair":
        selection["candidates"]["original"]["tokens"] = None
    elif change == "false_tie":
        selection["candidates"]["original"]["tokens"] = selection["selected_tokens"]
    elif change == "tokenizer_revision":
        selection["tokenizer_binding"]["tokenizer_revision"] = "different"
    elif change == "tokenizer_asset_hash":
        selection["tokenizer_provenance"]["asset_sha256"] = "b" * 64
    elif change == "tokenizer_extra":
        selection["tokenizer_binding"]["source"] = "DO_NOT_RETAIN"
    elif change == "provenance_extra":
        selection["tokenizer_provenance"]["source"] = "DO_NOT_RETAIN"
    elif change == "provenance_claim":
        selection["tokenizer_provenance"]["completion_authority"] = True
    elif change == "relative_asset":
        selection["tokenizer_provenance"]["asset_path"] = "asset.json"
    elif change == "schema":
        row["schema"] = "unknown"
    elif change == "sequence":
        row["sequence"] = True
    elif change == "row_extra":
        row["read_success"] = True
    replace_population(audit_path, [row])
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"


def test_duplicate_or_gapped_sequences_are_not_a_complete_population(local, audit_path):
    closed_audit(audit_path, [measured(local)])
    row = json.loads(audit_path.read_bytes().splitlines()[0])
    replace_population(audit_path, [row, row])
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"
    row["sequence"] = 2
    replace_population(audit_path, [row])
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"


@pytest.mark.parametrize("change", ["truncate", "checksum", "read_count", "duplicate_footer", "trailing_data", "duplicate_key", "nonfinite"])
def test_incomplete_or_contradictory_audit_closure_is_invalid(local, audit_path, change):
    closed_audit(audit_path, [measured(local)])
    raw = audit_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    footer = json.loads(lines[-1])
    if change == "truncate":
        raw = raw[:-1]
    elif change == "checksum":
        footer["records_sha256"] = "b" * 64
        raw = lines[0] + json.dumps(footer).encode() + b"\n"
    elif change == "read_count":
        footer["read_count"] = True
        raw = lines[0] + json.dumps(footer).encode() + b"\n"
    elif change == "duplicate_footer":
        raw += lines[-1]
    elif change == "trailing_data":
        raw += b"{}\n"
    elif change == "duplicate_key":
        raw = lines[0] + lines[-1].replace(b'"event":"closed"', b'"event":"closed","event":"closed"')
    elif change == "nonfinite":
        raw = lines[0] + lines[-1].replace(b'"read_count":1', b'"read_count":NaN')
    audit_path.write_bytes(raw)
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"


@pytest.mark.parametrize("change", [{"provider": "different"}, {"model": "different"},
    {"artifact": "/different/descriptor.json"}, {"expected_sha256": "b" * 64}])
def test_replay_must_match_invocation_tokenizer_binding(local, audit_path, change):
    closed_audit(audit_path, [measured(local)])
    assert budget.summarize_selection_audit(audit_path, tokenizer_binding={**owner_config(local), **change})["status"] == "invalid"


def test_mixed_tokenizer_owners_are_rejected(local, audit_path):
    closed_audit(audit_path, [measured(local)])
    row = json.loads(audit_path.read_bytes().splitlines()[0])
    other = copy.deepcopy(row)
    other["sequence"] = 2
    other["selection"]["tokenizer_binding"]["model"] = "different"
    replace_population(audit_path, [row, other])
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"


def test_record_bound_closes_without_valid_footer(local, audit_path, monkeypatch):
    monkeypatch.setattr(budget, "MAX_AUDIT_RECORDS", 1)
    audit = budget.SelectionAudit(audit_path)
    receipt = measured(local)
    audit.record(receipt)
    with pytest.raises(ValueError, match="record bound"):
        audit.record(receipt)
    audit.close()
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"


def test_total_byte_bound_is_enforced_before_appending(local, audit_path, monkeypatch):
    audit = budget.SelectionAudit(audit_path)
    receipt = measured(local)
    audit.record(receipt)
    before = audit_path.read_bytes()
    monkeypatch.setattr(budget, "MAX_AUDIT_BYTES", len(before) + 511)
    with pytest.raises(ValueError, match="byte bound"):
        audit.record(receipt)
    assert audit_path.read_bytes() == before
    audit.close()
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"


def test_invalid_record_poisons_journal_instead_of_disappearing(local, audit_path):
    audit = budget.SelectionAudit(audit_path)
    receipt = measured(local)
    audit.record(receipt)
    receipt["selected_tokens"] = False
    with pytest.raises(ValueError):
        audit.record(receipt)
    audit.close()
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory", "public_file", "public_parent", "parent_symlink"])
def test_nonprivate_or_nonregular_audit_refused(local, audit_path, tmp_path, kind):
    if kind in {"hardlink", "public_file", "public_parent"}:
        closed_audit(audit_path, [measured(local)])
        if kind == "hardlink":
            os.link(audit_path, audit_path.with_name("alias"))
        elif kind == "public_file":
            audit_path.chmod(0o644)
        else:
            audit_path.parent.chmod(0o755)
    elif kind == "symlink":
        other = audit_path.with_name("other")
        closed_audit(other, [measured(local)])
        audit_path.symlink_to(other)
    elif kind == "fifo":
        os.mkfifo(audit_path, mode=0o600)
    elif kind == "directory":
        audit_path.mkdir(mode=0o700)
    elif kind == "parent_symlink":
        closed_audit(audit_path, [measured(local)])
        alias = tmp_path / "alias-parent"
        alias.symlink_to(audit_path.parent, target_is_directory=True)
        audit_path = alias / audit_path.name
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"
    with pytest.raises((OSError, ValueError)):
        budget.SelectionAudit(audit_path)


def test_existing_audit_is_never_overwritten(local, audit_path):
    closed_audit(audit_path, [measured(local)])
    before = audit_path.read_bytes()
    with pytest.raises(FileExistsError):
        budget.SelectionAudit(audit_path)
    assert audit_path.read_bytes() == before


def test_replaced_audit_path_is_detected_before_write(local, audit_path):
    audit = budget.SelectionAudit(audit_path)
    audit_path.rename(audit_path.with_name("replaced"))
    audit_path.write_bytes(b"")
    audit_path.chmod(0o600)
    with pytest.raises(ValueError, match="changed"):
        audit.record(measured(local))
    audit.close()
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"


@pytest.mark.parametrize("change", [{"read_count": True}, {"compact_selected": 2},
    {"original_selected": 1}, {"measured_comparisons": 0}, {"unmeasured_fallbacks": 1},
    {"explicit_original": 1}, {"selected_reply_tokens": 0}, {"selected_reply_tokens": budget.MAX_REPLY_TOKENS + 1},
    {"provider_usage_measured": True}, {"audit_closed": False}, {"status": "missing"},
    {"schema": "unknown"}, {"extra": 1}])
def test_summary_rejects_contradictory_counts_and_authority(local, audit_path, change):
    closed_audit(audit_path, [measured(local)])
    summary = budget.summarize_selection_audit(audit_path)
    with pytest.raises(ValueError):
        budget.validate_selection_summary({**summary, **change})


def test_audit_read_byte_bound_is_enforced_without_unbounded_reads(audit_path):
    audit_path.write_bytes(b" " * (budget.MAX_AUDIT_BYTES + 1))
    audit_path.chmod(0o600)
    assert budget.summarize_selection_audit(audit_path)["status"] == "invalid"
