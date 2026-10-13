"""Closed repository treatment and retained adoption joins; no provider calls."""
from copy import deepcopy
import gzip
import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_source_policy as policy


@pytest.fixture(params=policy.REPOSITORY_SOURCE_POLICIES[1:])
def mode(request):
    return request.param


def dispatch_report(mode):
    """Authored full receipt, including a failed tool call and successful edit."""
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_policy import create_binding
    binding = create_binding(workspace=Path("/authored/candidate"), requested_policy=mode)
    binding["usage"] = dict(status="observed", initializations=1, tool_calls=8,
        successful_reads=3, compact_reads=2 if mode == policy.COMPACT_REPOSITORY_SOURCE_POLICY else 0,
        canonical_edits=1, uncertain_edits=0, refreshes=2, errors=1, returned_bytes=2300,
        store_cleanup_complete=True, artifacts_retained_after_invocation=False)
    identity = "authored-coding-invocation"
    row = dict(phase="coding", purpose="coding", provider="codex_cli", invocation_id=identity,
        status="provider_returned", workspace="/authored/candidate", repository_source_policy=binding,
        model_prompt_sha256="a" * 64, model_prompt_bytes=2300,
        usage=dict(codex_repository_source_configured=True,
                   codex_repository_source_config_sha256=binding["config_sha256"]))
    return dict(repository_source_policy=mode, provider_invocations=[{"phase": "planning"}, row],
        context_input_audit=dict(status="verified_all_observed_model_inputs",
            all_observed_coding_inputs_verified=True, coding_receipts_seen=1, usable_coding_receipts=1,
            read_errors=[], invocations=[dict(invocation_id=identity,
                model_input_verified=True, status="verified_exact_model_input")],
            matches=[dict(invocation_id=identity, model_input_verified=True,
                model_prompt_sha256="a" * 64, model_prompt_bytes=2300,
                model_input_checks=dict(repository_source_helper_source=True, model_prompt_sha256=True,
                    model_prompt_bytes=True, workspace_advisory_sha256=True, workspace_advisory_bytes=True))]))


def test_explicit_modes_and_legacy_omission(mode):
    assert policy.repository_source_selection("legacy") == {}
    assert policy.repository_source_selection(mode) == {"repository_source_policy": mode}
    assert policy.validate_repository_source_policy(mode, arm="full", provider="codex_cli") == mode
    policy.require_repository_source_archive(None, "legacy")


@pytest.mark.parametrize("value", [None, True, {}, [], "repository-source-foreign@1"])
def test_closed_policy(value):
    with pytest.raises(ValueError, match="supported repository source policy"):
        policy.validate_repository_source_policy(value)


@pytest.mark.parametrize("kwargs", [{"arm": "no-index"}, {"provider": "grok_cli"},
    {"source_view_policy": "source-view-compact@1"}, {"source_view_policy": None}])
def test_only_full_codex_route_without_old_helper(mode, kwargs):
    with pytest.raises(ValueError, match="full Codex coding route"):
        policy.validate_repository_source_policy(mode, **kwargs)


def test_current_source_and_datasets_api_archive_join(mode):
    inventory = policy.repository_source_inventory()
    assert len(inventory["files"]) == len(policy.SOURCE_PATHS) + len(policy.DATASETS_SOURCE_PATHS)
    policy.require_repository_source_archive(inventory, mode)


@pytest.mark.parametrize("relative", policy.SOURCE_PATHS + policy.DATASETS_SOURCE_PATHS)
def test_every_launcher_runtime_and_native_owner_source_is_bound(relative):
    inventory = policy.repository_source_inventory()
    prefix = "datasets/" if relative.startswith("ipfs_datasets_py/") else "source/"
    row = next(item for item in inventory["files"] if item["path"] == prefix + relative)
    row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_repository_source_archive(inventory, policy.COMPACT_REPOSITORY_SOURCE_POLICY)


@pytest.mark.parametrize("change", ["missing", "duplicate", "size", "boolean_size", "missing_size", "renamed"])
def test_native_owner_inventory_is_strict(change, mode):
    inventory = policy.repository_source_inventory()
    target = next(row for row in inventory["files"] if row["path"].endswith("/codebase_ir.py"))
    if change == "missing": inventory["files"].remove(target)
    elif change == "duplicate": inventory["files"].append(dict(target))
    elif change == "size": target["bytes"] += 1
    elif change == "boolean_size": target["bytes"] = True
    elif change == "missing_size": target.pop("bytes")
    elif change == "renamed": target["path"] = "datasets/foreign.py"
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_repository_source_archive(inventory, mode)


@pytest.mark.parametrize("value", [None, True, {"files": None}, {"files": {}}, {"files": "foreign"}])
def test_missing_or_malformed_archive(value, mode):
    with pytest.raises(ValueError, match="current runtime source inventory"):
        policy.require_repository_source_archive(value, mode)


@pytest.mark.parametrize("change", ["class", "method", "parameter", "scan_limit"])
def test_matching_but_incompatible_datasets_owner_is_refused(monkeypatch, change):
    sources = policy._current_sources()
    name = "datasets/ipfs_datasets_py/logic/software_contracts/codebase_ir.py"
    raw = sources[name]
    if change == "class": raw = raw.replace(b"class RepositoryCodebaseIndex:", b"class ForeignOwner:")
    elif change == "method": raw = raw.replace(b"def prepare_current(", b"def stale_prepare(")
    elif change == "parameter": raw = raw.replace(b"expected_head", b"foreign_head")
    elif change == "scan_limit": raw = raw.replace(b"max_entries", b"unbounded_entries")
    assert raw != sources[name]
    sources[name] = raw
    monkeypatch.setattr(policy, "_current_sources", lambda: sources)
    inventory = {"files": [{"path": path, "bytes": len(content),
                           "sha256": hashlib.sha256(content).hexdigest()} for path, content in sources.items()]}
    with pytest.raises(ValueError, match="current datasets owner APIs"):
        policy.require_repository_source_archive(inventory, policy.COMPACT_REPOSITORY_SOURCE_POLICY)


def source_archive(tmp_path, *, mutation=None):
    sources = policy._current_sources()
    manifest = policy.repository_source_inventory()
    members = list(sources.items())
    if mutation == "missing": members.pop()
    elif mutation == "duplicate": members.append(members[0])
    elif mutation == "late_duplicate": members.extend([("weights/padding.bin", b"x" * 65536), members[0]])
    elif mutation == "changed": members[0] = (members[0][0], b"!" + members[0][1][1:])
    elif mutation == "size": members[0] = (members[0][0], members[0][1] + b" ")
    elif mutation == "unknown": members.append(("unlisted.py", b"not indexed"))
    elif mutation in {"alias", "traversal", "absolute", "backslash"}:
        prefix = {"alias": "./", "traversal": "../", "absolute": "/", "backslash": "source\\"}[mutation]
        members[0] = (prefix + members[0][0], members[0][1])
    if mutation == "late_duplicate":
        manifest["files"].append({"path": "weights/padding.bin", "bytes": 65536,
                                   "sha256": hashlib.sha256(b"x" * 65536).hexdigest()})
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for index, (name, raw) in enumerate(members):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(raw), 0o644
            if index == 0:
                if mutation == "mode": info.mode = 0o600
                elif mutation == "symlink": info.type, info.linkname, info.size = tarfile.SYMTYPE, "foreign.py", 0
                elif mutation == "hardlink": info.type, info.linkname, info.size = tarfile.LNKTYPE, "foreign.py", 0
                elif mutation == "directory": info.type, info.size = tarfile.DIRTYPE, 0
                elif mutation == "foreign_pax": info.pax_headers = {"mtime": "0"}
                elif mutation == "pax_sparse": info.pax_headers = {"GNU.sparse.major": "1", "GNU.sparse.minor": "0"}
            archive.addfile(info, io.BytesIO(raw) if info.isfile() else None)
    encoded = gzip.compress(raw_tar.getvalue())
    if mutation == "truncated_gzip": encoded = encoded[:-4]
    elif mutation == "bad_gzip_crc": encoded = encoded[:-8] + bytes([encoded[-8] ^ 1]) + encoded[-7:]
    elif mutation == "after_end_duplicate": encoded += gzip.compress(raw_tar.getvalue())
    elif mutation == "excess_padding": encoded += gzip.compress(b"\0" * (tarfile.RECORDSIZE + 1))
    path = tmp_path / "runtime.tar.gz"
    path.write_bytes(encoded)
    manifest["archive_sha256"] = hashlib.sha256(encoded).hexdigest()
    return path, manifest


def test_actual_archive_members_join_current_source_and_owner(tmp_path, mode):
    path, manifest = source_archive(tmp_path)
    policy.require_repository_source_archive(manifest, mode, archive=path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "late_duplicate", "changed", "size", "unknown",
    "alias", "traversal", "absolute", "backslash", "mode", "symlink", "hardlink", "directory", "foreign_pax",
    "truncated_gzip", "bad_gzip_crc", "after_end_duplicate", "excess_padding", "pax_sparse"])
def test_current_inventory_cannot_authenticate_different_archive_members(tmp_path, mutation):
    path, manifest = source_archive(tmp_path, mutation=mutation)
    # The outer archive hash and current inventory are both honest here. Their
    # separate validity must not substitute for the inventory/member join.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["archive_sha256"]
    policy.require_repository_source_archive(manifest, policy.COMPACT_REPOSITORY_SOURCE_POLICY)
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_repository_source_archive(manifest, policy.COMPACT_REPOSITORY_SOURCE_POLICY, archive=path)


def test_small_pax_sparse_header_cannot_allocate_a_large_body_map(tmp_path):
    path, manifest = source_archive(tmp_path, mutation="pax_sparse")
    with pytest.raises(ValueError) as error:
        policy.require_repository_source_archive(manifest, policy.COMPACT_REPOSITORY_SOURCE_POLICY, archive=path)
    assert str(error.value.__cause__) == "archive sparse members not supported"


def test_corrupt_deflate_is_a_qualification_refusal(tmp_path):
    # Valid gzip header followed by the reserved DEFLATE block type.
    encoded = b"\x1f\x8b\x08\x00" + b"\0" * 6 + b"\x07" + b"\0" * 8
    path = tmp_path / "runtime.tar.gz"
    path.write_bytes(encoded)
    manifest = policy.repository_source_inventory()
    manifest["archive_sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_repository_source_archive(manifest, policy.COMPACT_REPOSITORY_SOURCE_POLICY, archive=path)


@pytest.mark.parametrize("kind,size", [(tarfile.XHDTYPE, 1024**3), (tarfile.XGLTYPE, 32),
    (tarfile.GNUTYPE_LONGNAME, 1024**3), (tarfile.GNUTYPE_SPARSE, 0)])
def test_hidden_tar_metadata_is_refused_before_body_allocation(tmp_path, kind, size):
    info = tarfile.TarInfo("hidden-header")
    info.type, info.size = kind, size
    # Deliberately no advertised payload: metadata must be refused by the
    # header hook, before tarfile attempts its potentially enormous read.
    raw = info.tobuf(format=tarfile.USTAR_FORMAT) + b"\0" * 1024
    path = tmp_path / "runtime.tar.gz"
    path.write_bytes(gzip.compress(raw))
    manifest = policy.repository_source_inventory()
    manifest["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="newly qualified runtime archive") as error:
        policy.require_repository_source_archive(manifest, policy.COMPACT_REPOSITORY_SOURCE_POLICY, archive=path)
    assert str(error.value.__cause__) in {"archive extended header exceeds bound", "archive builder regular members required"}


def test_archive_changed_after_inventory_hash_is_refused(tmp_path):
    path, manifest = source_archive(tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_repository_source_archive(manifest, policy.COMPACT_REPOSITORY_SOURCE_POLICY, archive=path)


def test_archive_read_bounds_reject_before_member_body(tmp_path, monkeypatch):
    path, manifest = source_archive(tmp_path)
    monkeypatch.setattr(policy, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_repository_source_archive(manifest, policy.COMPACT_REPOSITORY_SOURCE_POLICY, archive=path)


def test_actual_reads_qualify_without_proof_completion_or_savings_claim(mode):
    report = dispatch_report(mode)
    original = deepcopy(report)
    result = policy.repository_source_dispatch_observation(report, mode)
    assert result["effective_repository_tools_observed"] is True
    assert result["status"] == "verified_repository_tool_use"
    row = result["invocations"][0]
    assert row["native_tools_configured"] is True
    assert row["server_initialized"] is True
    assert row["actual_reads_observed"] is True
    assert row["compact_reads_observed"] is (mode == policy.COMPACT_REPOSITORY_SOURCE_POLICY)
    assert row["confirmed_canonical_edit_observed"] is True
    assert row["errors"] == 1
    assert result["token_savings_established"] is result["proof_authority"] is result["completion_authority"] is False
    assert report == original


@pytest.mark.parametrize("mutation", ["declared", "missing_binding", "schema", "instruction",
    "source_digest", "config_digest", "native_config", "native_configured", "old_helper", "foreign_provider",
    "duplicate_coding", "duplicate_audit", "missing_audit", "source_check", "prompt_digest", "advisory_check",
    "planning_policy", "unreceipted", "unknown_phase", "malformed_phase", "malformed_id", "malformed_hash",
    "no_coding", "read_error", "foreign_match", "audit_count", "audit_boolean_count", "extra_failed_attempt",
    "missing_status", "malformed_status"])
def test_dispatch_refuses_forged_or_incomplete_binding_population(mutation, mode):
    report = dispatch_report(mode)
    coding, audit = report["provider_invocations"][1], report["context_input_audit"]
    binding = coding["repository_source_policy"]
    if mutation == "declared": report["repository_source_policy"] = "legacy"
    elif mutation == "missing_binding": coding.pop("repository_source_policy")
    elif mutation == "schema": binding["schema"] = "foreign"
    elif mutation == "instruction": binding["instruction_sha256"] = "0" * 64
    elif mutation == "source_digest": binding["helper_sha256"] = "0" * 64
    elif mutation == "config_digest": binding["config_sha256"] = "0" * 64
    elif mutation == "native_config": coding["usage"]["codex_repository_source_config_sha256"] = "0" * 64
    elif mutation == "native_configured": coding["usage"]["codex_repository_source_configured"] = 1
    elif mutation == "old_helper": coding["source_view_policy"] = None
    elif mutation == "foreign_provider": coding["provider"] = "grok_cli"
    elif mutation == "duplicate_coding": report["provider_invocations"].append(deepcopy(coding))
    elif mutation == "duplicate_audit": audit["invocations"].append(deepcopy(audit["invocations"][0]))
    elif mutation == "missing_audit": report.pop("context_input_audit")
    elif mutation == "source_check": audit["matches"][0]["model_input_checks"]["repository_source_helper_source"] = False
    elif mutation == "prompt_digest": audit["matches"][0]["model_prompt_sha256"] = "0" * 64
    elif mutation == "advisory_check": audit["matches"][0]["model_input_checks"]["workspace_advisory_sha256"] = False
    elif mutation == "planning_policy": report["provider_invocations"][0]["repository_source_policy"] = None
    elif mutation == "unreceipted": report["unreceipted_provider_attempt"] = True
    elif mutation == "unknown_phase": report["provider_invocations"].append({"phase": "foreign"})
    elif mutation == "malformed_phase": report["provider_invocations"].append({"phase": []})
    elif mutation == "malformed_id": coding["invocation_id"] = {}
    elif mutation == "malformed_hash": coding["model_prompt_sha256"] = "foreign"
    elif mutation == "no_coding": report["provider_invocations"] = [{"phase": "planning"}]
    elif mutation == "read_error": audit["read_errors"] = ["authored"]
    elif mutation == "foreign_match": audit["matches"].append({"invocation_id": "unobserved"})
    elif mutation == "audit_count": audit["coding_receipts_seen"] = 2
    elif mutation == "audit_boolean_count": audit["usable_coding_receipts"] = True
    elif mutation == "missing_status": coding.pop("status")
    elif mutation == "malformed_status": coding["status"] = []
    elif mutation == "extra_failed_attempt":
        report["provider_invocations"].append({**deepcopy(coding), "invocation_id": "failed-attempt", "status": "failed"})
    result = policy.repository_source_dispatch_observation(report, mode)
    assert result["effective_repository_tools_observed"] is False
    assert result["status"] == "unverified"
    if mutation == "extra_failed_attempt": assert result["coding_receipts_seen"] == 2
    if mutation in {"source_digest", "native_config", "native_configured", "missing_audit"}:
        assert result["invocations"][0]["confirmed_canonical_edit_observed"] is False


@pytest.mark.parametrize("change", ["bool_counter", "negative", "float", "unbounded", "missing_counter",
    "extra_field", "malformed_status", "too_many_initializations", "compact_exceeds_reads", "calls_too_small",
    "excess_errors", "refreshes_exceed_successes", "bytes_with_zero_calls", "missing_bytes", "tiny_bytes",
    "excess_bytes", "uninitialized_reads", "cleanup_incomplete", "artifacts_retained", "cleanup_error",
    "uncertain_edit", "unavailable_counts", "not_observed_counts", "unrefreshed_successful_edit"])
def test_contradictory_or_unqualified_server_metrics_cannot_qualify(change, mode):
    report = dispatch_report(mode)
    usage = report["provider_invocations"][1]["repository_source_policy"]["usage"]
    if change == "bool_counter": usage["tool_calls"] = True
    elif change == "negative": usage["errors"] = -1
    elif change == "float": usage["errors"] = 1.0
    elif change == "unbounded": usage["tool_calls"] = 513
    elif change == "missing_counter": usage.pop("compact_reads")
    elif change == "extra_field": usage["claimed_success"] = True
    elif change == "malformed_status": usage["status"] = []
    elif change == "too_many_initializations": usage["initializations"] = 2
    elif change == "compact_exceeds_reads": usage["compact_reads"] = 4
    elif change == "calls_too_small": usage["tool_calls"] = 3
    elif change == "excess_errors": usage["errors"] = 9
    elif change == "refreshes_exceed_successes": usage["refreshes"] = 7
    elif change == "bytes_with_zero_calls": usage["tool_calls"] = 0
    elif change == "missing_bytes": usage["returned_bytes"] = 0
    elif change == "tiny_bytes": usage["returned_bytes"] = 1
    elif change == "excess_bytes": usage["returned_bytes"] = 2**31
    elif change == "uninitialized_reads": usage["initializations"] = 0
    elif change == "cleanup_incomplete": usage.update(store_cleanup_complete=False, artifacts_retained_after_invocation=None)
    elif change == "artifacts_retained": usage["artifacts_retained_after_invocation"] = True
    elif change == "cleanup_error": usage["store_cleanup_error"] = "cleanup_unavailable"
    elif change == "uncertain_edit": usage["uncertain_edits"] = 1
    elif change == "unavailable_counts": usage["status"] = "unavailable"
    elif change == "not_observed_counts": usage["status"] = "not_observed"
    elif change == "unrefreshed_successful_edit": usage.update(canonical_edits=2, refreshes=0)
    result = policy.repository_source_dispatch_observation(report, mode)
    assert result["effective_repository_tools_observed"] is False


@pytest.mark.parametrize("state", ["missing", "unavailable", "zero", "initialized", "initialization_failed", "list_only"])
def test_unavailable_and_zero_adoption_remain_distinguishable(state, mode):
    report = dispatch_report(mode)
    binding = report["provider_invocations"][1]["repository_source_policy"]
    if state == "missing": binding.pop("usage")
    else:
        usage = binding["usage"]
        for key in policy._COUNTERS: usage[key] = None if state == "unavailable" else 0
        usage["status"] = "unavailable" if state == "unavailable" else "not_observed" if state == "zero" else "observed"
        if state in {"initialized", "list_only"}: usage["initializations"] = 1
        if state == "initialization_failed": usage["errors"] = 1
        if state == "list_only": usage.update(tool_calls=1, returned_bytes=100)
    result = policy.repository_source_dispatch_observation(report, mode)
    row = result["invocations"][0]
    assert row["native_tools_configured"] is True
    assert row["server_initialized"] is (state in {"initialized", "list_only"})
    assert row["actual_reads_observed"] is False
    assert row["server_audit_valid"] is (state != "missing")
    assert result["effective_repository_tools_observed"] is False


def test_compact_policy_requires_actual_compact_reads():
    mode = policy.COMPACT_REPOSITORY_SOURCE_POLICY
    report = dispatch_report(mode)
    report["provider_invocations"][1]["repository_source_policy"]["usage"]["compact_reads"] = 0
    result = policy.repository_source_dispatch_observation(report, mode)
    assert result["invocations"][0]["actual_reads_observed"] is True
    assert result["effective_repository_tools_observed"] is False


def test_failed_provider_attempt_is_preserved_and_can_observe_tools(mode):
    report = dispatch_report(mode)
    coding = report["provider_invocations"][1]
    coding.update(status="failed", error_type="TimeoutExpired", failure_phase="provider_invocation")
    coding["usage"].update(timed_out=True, exit_code=None)
    result = policy.repository_source_dispatch_observation(report, mode)
    assert result["coding_receipts_seen"] == 1
    assert result["invocations"][0]["provider_attempt_status"] == "failed"
    assert result["effective_repository_tools_observed"] is True
    assert result["completion_authority"] is False


def test_different_policies_cannot_share_a_selection_label(mode):
    other = policy.COMPACT_REPOSITORY_SOURCE_POLICY if mode == policy.RAW_REPOSITORY_SOURCE_POLICY else policy.RAW_REPOSITORY_SOURCE_POLICY
    report = dispatch_report(other)
    report["repository_source_policy"] = mode
    result = policy.repository_source_dispatch_observation(report, mode)
    assert result["effective_repository_tools_observed"] is False
    assert result["invocations"][0]["policy_binding_valid"] is False


def test_duplicate_server_configuration_cannot_qualify_distinct_attempts(mode):
    report = dispatch_report(mode)
    report["provider_invocations"].append({**deepcopy(report["provider_invocations"][1]), "invocation_id": "second"})
    audit = report["context_input_audit"]
    audit.update(coding_receipts_seen=2, usable_coding_receipts=2)
    for name in ("invocations", "matches"):
        audit[name].append({**deepcopy(audit[name][0]), "invocation_id": "second"})
    result = policy.repository_source_dispatch_observation(report, mode)
    assert result["effective_repository_tools_observed"] is False
    assert "repository_configuration_identities_missing_or_duplicate" in result["problems"]


def token_dispatch():
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_policy import create_binding
    mode = policy.COMPACT_REPOSITORY_SOURCE_POLICY
    report = dispatch_report(mode)
    row = report["provider_invocations"][1]
    usage = row["repository_source_policy"]["usage"]
    binding = create_binding(workspace=Path(row["workspace"]), requested_policy=mode, tokenizer={
        "artifact": "/authored/owner-tokenizer.json", "expected_sha256": "b" * 64,
        "provider": "codex_cli", "model": "pinned"})
    binding["usage"] = usage
    usage["compact_reads"] = 0
    usage["token_selection"] = dict(schema="repository-source-token-selection-summary@1", status="valid",
        audit_closed=True, read_count=3, compact_selected=0, original_selected=3,
        measured_comparisons=2, unmeasured_fallbacks=0, explicit_original=1,
        selected_reply_tokens=100, provider_usage_measured=False)
    row["repository_source_policy"] = binding
    row["usage"]["codex_repository_source_config_sha256"] = binding["config_sha256"]
    return report


def test_measured_original_selection_counts_as_observed_policy_without_claiming_savings():
    report = token_dispatch()
    result = policy.repository_source_dispatch_observation(report, policy.COMPACT_REPOSITORY_SOURCE_POLICY)
    observed = result["invocations"][0]
    assert observed["compact_reads_observed"] is False
    assert observed["local_token_selection_observed"] is True
    assert result["effective_repository_tools_observed"] is True
    assert result["token_savings_established"] is False


@pytest.mark.parametrize("mutation", ["missing", "null", "open", "read_count", "compact_count",
    "unmeasured", "extra", "bool_count", "explicit_only", "unbound"])
def test_incomplete_or_contradictory_selection_cannot_qualify(mutation):
    report = token_dispatch()
    binding = report["provider_invocations"][1]["repository_source_policy"]
    selection = binding["usage"]["token_selection"]
    if mutation == "missing": binding["usage"].pop("token_selection")
    elif mutation == "null": binding["usage"]["token_selection"] = None
    elif mutation == "open": selection["audit_closed"] = False
    elif mutation == "read_count": selection.update(read_count=4, original_selected=4, explicit_original=2)
    elif mutation == "compact_count": selection.update(compact_selected=1, original_selected=2)
    elif mutation == "unmeasured": selection.update(measured_comparisons=1, unmeasured_fallbacks=1)
    elif mutation == "extra": selection["source"] = "forbidden extra content"
    elif mutation == "bool_count": selection["read_count"] = True
    elif mutation == "explicit_only": selection.update(measured_comparisons=0, explicit_original=3, selected_reply_tokens=0)
    else:
        report["provider_invocations"][1]["usage"]["codex_repository_source_config_sha256"] = "0" * 64
    result = policy.repository_source_dispatch_observation(report, policy.COMPACT_REPOSITORY_SOURCE_POLICY)
    assert result["effective_repository_tools_observed"] is False
