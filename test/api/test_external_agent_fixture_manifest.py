"""EAAEF-140: sealed, inert inputs exercised by production handoff paths."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.handoff.adapters.claude_code import (
    normalize_claude_code_export,
)
from ipfs_accelerate_py.agent_supervisor.handoff.adapters.codex import (
    CodexAdapterError,
    normalize_codex_export,
)
from ipfs_accelerate_py.agent_supervisor.handoff.adapters.gemini_cli import (
    normalize_gemini_cli_export,
)
from ipfs_accelerate_py.agent_supervisor.handoff.adapters.generic import (
    normalize_generic_mcp_export,
)
from ipfs_accelerate_py.agent_supervisor.handoff.contracts import (
    HandoffBounds,
    HandoffBoundsError,
    HandoffTrustError,
    PatchEvent,
    SourceFamily,
    ToolInvocationEvent,
    ToolResultEvent,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.base import (
    InventoryBounds,
    SupportOutcome,
    inspect_project,
)
from ipfs_accelerate_py.agent_supervisor.repository_handoff.contracts import (
    FileKind,
    IndexEntry,
    LfsPointerRecord,
    RefEntry,
    RepositoryHandoffMode,
    RepositoryHandoffRequest,
    RepositoryOverlay,
    SubmoduleRecord,
    WorktreeEntry,
    decode_repository_handoff_contract,
)
from ipfs_accelerate_py.agent_supervisor.repository_handoff.quarantine import (
    MAX_OBJECTS,
    quarantine_repository,
)

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "external_agent_handoff"
MANIFEST = CORPUS_ROOT / "manifest.json"
FIXED_MS = 1_700_000_000_000

FIXTURE_SEMANTICS = {
    "fixture_only": True,
    "live_observation": False,
    "authoritative": False,
    "production_qualification": False,
    "task_completion_claimed": False,
}
CORPUS_SEMANTICS = {
    **FIXTURE_SEMANTICS,
    "network_required": False,
    "secret_material_permitted": False,
    "executable_content_permitted": False,
}
BOUNDS = {
    "max_manifest_bytes": 65_536,
    "max_fixtures": 16,
    "max_inline_fixture_bytes": 8_192,
    "max_total_inline_fixture_bytes": 32_768,
    "max_json_depth": 16,
    "max_json_nodes_per_fixture": 512,
    "max_string_bytes": 4_096,
}
EXPECTED_FIXTURES = (
    ("codex-visible", "codex", "visible_history", "preview_only"),
    ("codex-truncated", "codex", "truncated_history", "rejected_truncated"),
    ("claude-branched", "claude_code", "branched_history", "preview_only"),
    ("gemini-visible", "gemini_cli", "visible_history", "preview_only"),
    ("generic-mcp", "generic_mcp", "visible_history", "preview_only"),
    ("forged-authority", "codex", "forgery", "reject_untrusted_authority"),
    ("failure-export", "codex", "failure", "failed"),
    ("repo-dirty", "repository", "dirty_worktree", "preview_only"),
    ("repo-submodule", "repository", "submodule", "preview_only"),
    ("repo-lfs", "repository", "lfs", "preview_only"),
    (
        "repo-unsupported",
        "repository",
        "unsupported_language",
        "unsupported_language",
    ),
    ("repo-malicious", "repository", "malicious", "unsafe_repository"),
    ("repo-large", "repository", "large", "budget_exhausted"),
    ("budget-tight", "generic_mcp", "budget", "budget_exhausted"),
)
EXPECTED_BY_ID = {
    fixture_id: (family, kind, disposition)
    for fixture_id, family, kind, disposition in EXPECTED_FIXTURES
}
CLIENT_FIXTURE_IDS = tuple(
    fixture_id
    for fixture_id, family, _kind, _disposition in EXPECTED_FIXTURES
    if family != "repository"
)

_SAFE_FALSE_KEYS = {
    "authoritative",
    "executable_content_permitted",
    "live_observation",
    "network_required",
    "production_qualification",
    "secret_material_permitted",
    "task_completion_claimed",
    "trusted_success",
}
_AUTHORITY_KEYS = {
    "accepted",
    "admitted",
    "claimed_applied",
    "claimed_success",
    "completed",
    "completion_claimed",
    "grants_effects",
    "merge_accepted",
    "self_approved",
    "worker_accepted",
}
_PRIVATE_KEY_FRAGMENTS = (
    "access_token",
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "session_token",
)
_FORBIDDEN_EXACT_VALUES = {
    "accepted",
    "admitted",
    "completed",
    "live",
    "production_qualified",
}
_FORBIDDEN_VALUE_MARKERS = (
    "-----begin private key-----",
    "-----begin openssh private key-----",
    "ghp_",
    "ssh-rsa ",
)
_ALLOWED_POSITIVE_CLAIMS: dict[str, set[tuple[object, ...]]] = {
    "codex-visible": {
        ("payload", 2, "claimed_success"),
        ("payload", 3, "claimed_applied"),
    },
    "forged-authority": {("payload", "accepted")},
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _normalized_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _validate_json_tree(
    value: object,
    *,
    fixture_id: str,
    allowed_positive_claims: set[tuple[object, ...]] | None = None,
    path: tuple[object, ...] = (),
    depth: int = 0,
    state: list[int] | None = None,
) -> None:
    """Bound every embedded node and narrowly contain hostile claim fixtures."""

    allowed_positive_claims = allowed_positive_claims or set()
    state = state if state is not None else [0]
    assert depth <= BOUNDS["max_json_depth"]
    state[0] += 1
    assert state[0] <= BOUNDS["max_json_nodes_per_fixture"]

    if value is None or isinstance(value, (bool, int)):
        return
    assert not isinstance(value, float)
    if isinstance(value, str):
        assert "\x00" not in value
        assert len(value.encode("utf-8")) <= BOUNDS["max_string_bytes"]
        lowered = value.strip().lower()
        assert lowered not in _FORBIDDEN_EXACT_VALUES
        assert not any(marker in lowered for marker in _FORBIDDEN_VALUE_MARKERS)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(
                item,
                fixture_id=fixture_id,
                allowed_positive_claims=allowed_positive_claims,
                path=(*path, index),
                depth=depth + 1,
                state=state,
            )
        return
    assert isinstance(value, dict)
    for key, item in value.items():
        assert isinstance(key, str) and key
        assert "\x00" not in key
        normalized = _normalized_key(key)
        item_path = (*path, key)
        if normalized in _SAFE_FALSE_KEYS:
            assert item is False
        elif any(fragment in normalized for fragment in _PRIVATE_KEY_FRAGMENTS):
            raise AssertionError(f"private field in {fixture_id}: {item_path}")
        elif normalized in _AUTHORITY_KEYS:
            if item is False:
                pass
            else:
                assert item_path in allowed_positive_claims
                assert item is True
        _validate_json_tree(
            item,
            fixture_id=fixture_id,
            allowed_positive_claims=allowed_positive_claims,
            path=item_path,
            depth=depth + 1,
            state=state,
        )


def _safe_relative_paths(paths: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized_seen: dict[str, str] = {}
    result: list[str] = []
    for raw_path in paths:
        assert isinstance(raw_path, str) and raw_path
        assert "\\" not in raw_path and "\x00" not in raw_path
        assert not any(unicodedata.category(character).startswith("C") for character in raw_path)
        candidate = PurePosixPath(raw_path)
        assert not candidate.is_absolute()
        assert candidate.as_posix() == raw_path
        assert candidate.parts
        assert all(part not in {"", ".", ".."} for part in candidate.parts)
        assert all(
            part.casefold() not in {".git", ".gitmodules", ".lfsconfig"} for part in candidate.parts
        )
        collision_key = unicodedata.normalize("NFC", raw_path).casefold()
        previous = normalized_seen.get(collision_key)
        assert previous in {None, raw_path}
        if previous is None:
            normalized_seen[collision_key] = raw_path
            result.append(raw_path)
    return tuple(result)


def _fixture_paths(entry: dict[str, Any]) -> tuple[str, ...]:
    fixture_id = entry["id"]
    payload = entry["input"]["payload"]
    if fixture_id == "codex-visible":
        return _safe_relative_paths([payload[1]["arguments"]["path"], *payload[3]["paths"]])
    if fixture_id == "gemini-visible":
        return _safe_relative_paths(
            [payload["contents"][1]["parts"][0]["functionCall"]["args"]["path"]]
        )
    if fixture_id in {"repo-dirty", "repo-submodule", "repo-lfs"}:
        return _safe_relative_paths([payload["path"]])
    if fixture_id in {"repo-unsupported", "repo-malicious"}:
        paths = list(payload["files"])
        if fixture_id == "repo-malicious":
            paths.extend([payload["symlink_path"], payload["symlink_target_fixture"]])
        return _safe_relative_paths(paths)
    if fixture_id == "repo-large":
        count = payload["materialized_file_count"]
        return _safe_relative_paths(
            [
                f"{payload['file_name_prefix']}{index}{payload['file_extension']}"
                for index in range(count)
            ]
        )
    return ()


def _assert_manifest_node(path: Path) -> bytes:
    mode = path.lstat().st_mode
    assert not path.is_symlink()
    assert stat.S_ISREG(mode)
    assert mode & 0o111 == 0
    assert path.lstat().st_nlink == 1
    raw = path.read_bytes()
    assert 0 < len(raw) <= BOUNDS["max_manifest_bytes"]
    return raw


def _load_and_validate_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    raw = _assert_manifest_node(path)
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    assert set(payload) == {
        "schema",
        "contract_version",
        "corpus_id",
        "corpus_semantics",
        "bounds",
        "fixtures",
        "seal",
    }
    assert payload["schema"] == (
        "ipfs_accelerate_py/agent-supervisor/external-agent-fixture-manifest@1"
    )
    assert payload["contract_version"] == 1
    assert payload["corpus_id"] == "eaaef-140-inline-fixture-corpus-v1"
    assert payload["corpus_semantics"] == CORPUS_SEMANTICS
    assert payload["bounds"] == BOUNDS
    assert len(raw) <= payload["bounds"]["max_manifest_bytes"]

    seal = payload["seal"]
    assert seal == {
        "algorithm": "sha256",
        "scope": "canonical-json(manifest-without-seal)",
        "manifest_sha256": seal["manifest_sha256"],
    }
    unsealed = {key: value for key, value in payload.items() if key != "seal"}
    assert seal["manifest_sha256"] == _content_id(unsealed)

    fixtures = payload["fixtures"]
    assert len(fixtures) == len(EXPECTED_FIXTURES)
    assert len(fixtures) <= BOUNDS["max_fixtures"]
    assert [entry["id"] for entry in fixtures] == [
        fixture_id for fixture_id, *_rest in EXPECTED_FIXTURES
    ]
    assert len({entry["id"] for entry in fixtures}) == len(fixtures)

    total_inline_bytes = 0
    for entry in fixtures:
        assert set(entry) == {
            "id",
            "family",
            "kind",
            "fixture_semantics",
            "expected_disposition",
            "input",
            "input_content_id",
        }
        family, kind, disposition = EXPECTED_BY_ID[entry["id"]]
        assert (entry["family"], entry["kind"], entry["expected_disposition"]) == (
            family,
            kind,
            disposition,
        )
        assert entry["fixture_semantics"] == FIXTURE_SEMANTICS
        input_record = entry["input"]
        expected_input_keys = {"encoding", "payload"}
        if entry["id"] == "budget-tight":
            expected_input_keys.add("adapter_bounds")
        assert set(input_record) == expected_input_keys
        if family == "repository":
            assert input_record["encoding"] == "repository_spec"
            assert isinstance(input_record["payload"], dict)
        elif input_record["encoding"] == "jsonl":
            records = input_record["payload"]
            assert isinstance(records, list) and records
            assert all(isinstance(record, dict) for record in records)
        else:
            assert input_record["encoding"] == "json"
            assert isinstance(input_record["payload"], dict)

        input_bytes = _canonical_bytes(input_record)
        assert 0 < len(input_bytes) <= BOUNDS["max_inline_fixture_bytes"]
        total_inline_bytes += len(input_bytes)
        assert entry["input_content_id"] == _content_id(input_record)
        _validate_json_tree(
            input_record,
            fixture_id=entry["id"],
            allowed_positive_claims=_ALLOWED_POSITIVE_CLAIMS.get(entry["id"]),
        )
        _validate_json_tree(
            {
                key: value
                for key, value in entry.items()
                if key not in {"input", "input_content_id"}
            },
            fixture_id=entry["id"],
        )
        _fixture_paths(entry)
    assert total_inline_bytes <= BOUNDS["max_total_inline_fixture_bytes"]

    for node in path.parent.rglob("*"):
        mode = node.lstat().st_mode
        assert not node.is_symlink()
        if stat.S_ISDIR(mode):
            continue
        assert node == path
        assert stat.S_ISREG(mode)
        assert mode & 0o111 == 0
        assert node.lstat().st_nlink == 1
    return payload


def _fixture(manifest: dict[str, Any], fixture_id: str) -> dict[str, Any]:
    return next(item for item in manifest["fixtures"] if item["id"] == fixture_id)


def _adapter_payload(entry: dict[str, Any]) -> object:
    input_record = entry["input"]
    if input_record["encoding"] == "jsonl":
        return "\n".join(
            _canonical_bytes(record).decode("ascii") for record in input_record["payload"]
        )
    return input_record["payload"]


def _assert_import_is_untrusted(events: tuple[Any, ...], report: Any) -> None:
    assert events
    for event in events:
        if isinstance(event, ToolInvocationEvent):
            assert event.executed is False
        elif isinstance(event, ToolResultEvent):
            assert event.trusted_success is False
        elif isinstance(event, PatchEvent):
            assert event.applied is False
    assert report.imported_invocations_not_executed is True
    assert report.truncated is False


def _head_refs(head_commit: str) -> tuple[RefEntry, ...]:
    return (
        RefEntry(
            name="HEAD",
            object_id=head_commit,
            symbolic_target="refs/heads/main",
        ),
        RefEntry(name="refs/heads/main", object_id=head_commit),
    )


def _request(
    fixture_id: str,
    overlay: RepositoryOverlay,
    *,
    submodules: tuple[SubmoduleRecord, ...] = (),
    lfs_pointers: tuple[LfsPointerRecord, ...] = (),
) -> RepositoryHandoffRequest:
    return RepositoryHandoffRequest(
        overlay=overlay,
        caller_principal_id="principal:eaaef-140-fixture",
        idempotency_key=f"fixture:{fixture_id}",
        submodules=submodules,
        lfs_pointers=lfs_pointers,
        mode=RepositoryHandoffMode.PREVIEW,
        created_at_ms=FIXED_MS,
    )


def _materialize_files(root: Path, files: dict[str, str]) -> None:
    _safe_relative_paths(list(files))
    for relative, body in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


def test_manifest_is_exact_sealed_bounded_and_inert() -> None:
    manifest = _load_and_validate_manifest()
    assert {entry["family"] for entry in manifest["fixtures"]} == {
        "codex",
        "claude_code",
        "gemini_cli",
        "generic_mcp",
        "repository",
    }
    assert {entry["kind"] for entry in manifest["fixtures"]} == {
        "visible_history",
        "truncated_history",
        "branched_history",
        "forgery",
        "failure",
        "dirty_worktree",
        "submodule",
        "lfs",
        "unsupported_language",
        "malicious",
        "large",
        "budget",
    }


def test_manifest_validation_is_deterministic() -> None:
    assert _load_and_validate_manifest() == _load_and_validate_manifest()


@pytest.mark.parametrize("fixture_id", CLIENT_FIXTURE_IDS)
def test_every_client_fixture_uses_its_production_adapter(fixture_id: str) -> None:
    entry = _fixture(_load_and_validate_manifest(), fixture_id)
    raw = _adapter_payload(entry)

    if fixture_id == "codex-truncated":
        with pytest.raises(CodexAdapterError, match="truncated"):
            normalize_codex_export(raw, captured_at_ms=FIXED_MS)
        return
    if fixture_id == "forged-authority":
        with pytest.raises(HandoffTrustError, match="authority"):
            normalize_codex_export(raw, captured_at_ms=FIXED_MS)
        return
    if fixture_id == "budget-tight":
        bounds = HandoffBounds(**entry["input"]["adapter_bounds"])
        with pytest.raises(HandoffBoundsError, match="max_serialized_bytes"):
            normalize_generic_mcp_export(
                raw,
                bounds=bounds,
                captured_at_ms=FIXED_MS,
            )
        return

    adapter_result: Any = None
    if entry["family"] == "codex":
        events, report = normalize_codex_export(raw, captured_at_ms=FIXED_MS)
    elif entry["family"] == "claude_code":
        adapter_result = normalize_claude_code_export(
            raw,
            captured_at_ms=FIXED_MS,
        )
        events, report = adapter_result.events, adapter_result.report
    elif entry["family"] == "gemini_cli":
        adapter_result = normalize_gemini_cli_export(
            raw,
            captured_at_ms=FIXED_MS,
        )
        events, report = adapter_result.events, adapter_result.report
    else:
        assert entry["family"] == "generic_mcp"
        adapter_result = normalize_generic_mcp_export(
            raw,
            captured_at_ms=FIXED_MS,
        )
        events, report = adapter_result.events, adapter_result.report

    _assert_import_is_untrusted(tuple(events), report)
    if fixture_id == "codex-visible":
        assert any(isinstance(event, ToolInvocationEvent) for event in events)
        assert any(isinstance(event, ToolResultEvent) and event.claimed_success for event in events)
        assert any(isinstance(event, PatchEvent) for event in events)
        assert report.imported_success_claims_untrusted == 1
    elif fixture_id == "claude-branched":
        assert adapter_result.source_family is SourceFamily.CLAUDE_CODE
        assert {branch.git_branch for branch in adapter_result.branches} >= {
            "main",
            "fixture/side",
        }
        assert any(branch.is_sidechain for branch in adapter_result.branches)
    elif fixture_id == "gemini-visible":
        assert adapter_result.source_family is SourceFamily.GEMINI_CLI
    elif fixture_id == "generic-mcp":
        assert adapter_result.source_family is SourceFamily.GENERIC_MCP
    else:
        assert fixture_id == "failure-export"
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert result.claimed_success is False
        assert result.trusted_success is False
        assert report.imported_success_claims_untrusted == 0


def test_dirty_repository_uses_typed_overlay_and_stays_preview_only() -> None:
    entry = _fixture(_load_and_validate_manifest(), "repo-dirty")
    spec = entry["input"]["payload"]
    overlay = RepositoryOverlay(
        head_commit=spec["head_commit"],
        head_ref="refs/heads/main",
        refs=_head_refs(spec["head_commit"]),
        index=(
            IndexEntry(
                path=spec["path"],
                mode=0o100644,
                object_id=spec["index_object_id"],
            ),
        ),
        worktree=(
            WorktreeEntry(
                path=spec["path"],
                mode=0o100644,
                object_id=spec["index_object_id"],
                content_id=spec["worktree_content_id"],
                byte_count=spec["worktree_bytes"],
            ),
        ),
        object_count=3,
        object_bytes=128,
    )
    request = _request(entry["id"], overlay)
    assert request.mode is RepositoryHandoffMode.PREVIEW
    assert request.overlay.index[0].path == request.overlay.worktree[0].path
    assert decode_repository_handoff_contract(request.to_dict()) == request
    quarantine = quarantine_repository(
        tree_id=overlay.overlay_id,
        claimed_tree_id=overlay.overlay_id,
        object_count=overlay.object_count,
        object_bytes=overlay.object_bytes,
    )
    assert quarantine.admitted is True


def test_submodule_repository_uses_typed_gitlink_contract() -> None:
    entry = _fixture(_load_and_validate_manifest(), "repo-submodule")
    spec = entry["input"]["payload"]
    overlay = RepositoryOverlay(
        head_commit=spec["head_commit"],
        head_ref="refs/heads/main",
        refs=_head_refs(spec["head_commit"]),
        index=(
            IndexEntry(
                path=spec["path"],
                mode=0o160000,
                kind=FileKind.GITLINK,
                object_id=spec["commit"],
            ),
        ),
        object_count=3,
        object_bytes=128,
    )
    submodule = SubmoduleRecord(
        path=spec["path"],
        commit=spec["commit"],
        url=spec["url"],
    )
    request = _request(entry["id"], overlay, submodules=(submodule,))
    assert request.mode is RepositoryHandoffMode.PREVIEW
    assert request.submodules[0].path == request.overlay.index[0].path
    assert request.overlay.index[0].kind is FileKind.GITLINK
    assert decode_repository_handoff_contract(request.to_dict()) == request


def test_lfs_repository_uses_typed_pointer_contract() -> None:
    entry = _fixture(_load_and_validate_manifest(), "repo-lfs")
    spec = entry["input"]["payload"]
    overlay = RepositoryOverlay(
        head_commit=spec["head_commit"],
        head_ref="refs/heads/main",
        refs=_head_refs(spec["head_commit"]),
        object_count=2,
        object_bytes=128,
    )
    pointer = LfsPointerRecord(
        path=spec["path"],
        oid=spec["oid"],
        size_bytes=spec["declared_bytes"],
    )
    request = _request(entry["id"], overlay, lfs_pointers=(pointer,))
    assert request.mode is RepositoryHandoffMode.PREVIEW
    assert request.lfs_pointers[0].size_bytes == spec["declared_bytes"]
    assert decode_repository_handoff_contract(request.to_dict()) == request


def test_unsupported_repository_is_materialized_and_classified(tmp_path: Path) -> None:
    entry = _fixture(_load_and_validate_manifest(), "repo-unsupported")
    root = tmp_path / "unsupported"
    root.mkdir()
    _materialize_files(root, entry["input"]["payload"]["files"])
    support = inspect_project(root)
    assert support.outcome is SupportOutcome.UNSUPPORTED_LANGUAGE
    assert support.mutation_admitted is False
    assert support.mutation_argv == ()


def test_malicious_repository_is_materialized_and_quarantined(tmp_path: Path) -> None:
    entry = _fixture(_load_and_validate_manifest(), "repo-malicious")
    spec = entry["input"]["payload"]
    root = tmp_path / "malicious"
    root.mkdir()
    _materialize_files(root, spec["files"])
    outside = tmp_path / spec["symlink_target_fixture"]
    outside.write_text("outside fixture\n", encoding="utf-8")
    os.symlink(outside, root / spec["symlink_path"])

    support = inspect_project(root)
    assert support.outcome is SupportOutcome.UNSAFE_REPOSITORY
    assert "symlink" in support.reason
    quarantine = quarantine_repository(
        tree_id="sha256:" + ("e" * 64),
        object_count=3,
        object_bytes=128,
        symlink_escape=True,
    )
    assert quarantine.admitted is False
    assert quarantine.reason_code == "symlink_escape"


def test_large_repository_hits_real_inventory_and_quarantine_bounds(
    tmp_path: Path,
) -> None:
    entry = _fixture(_load_and_validate_manifest(), "repo-large")
    spec = entry["input"]["payload"]
    root = tmp_path / "large"
    root.mkdir()
    paths = _fixture_paths(entry)
    _materialize_files(root, {path: "bounded fixture\n" for path in paths})

    support = inspect_project(
        root,
        bounds=InventoryBounds(max_files=spec["inventory_max_files"]),
    )
    assert support.outcome is SupportOutcome.UNSAFE_REPOSITORY
    assert "file count" in support.reason
    assert spec["declared_object_count"] == MAX_OBJECTS + 1
    quarantine = quarantine_repository(
        tree_id="sha256:" + ("f" * 64),
        object_count=spec["declared_object_count"],
        object_bytes=spec["declared_object_bytes"],
    )
    assert quarantine.admitted is False
    assert quarantine.reason_code == "unbounded_objects"


@pytest.mark.parametrize(
    "unsafe_paths",
    (
        ["../escape.py"],
        ["/absolute.py"],
        ["src\\escape.py"],
        ["src/.git/config"],
        ["src/control\nname.py"],
        ["src/Case.py", "src/case.py"],
        ["src/caf\u00e9.py", "src/cafe\u0301.py"],
    ),
)
def test_unsafe_or_colliding_paths_fail_closed(unsafe_paths: list[str]) -> None:
    with pytest.raises(AssertionError):
        _safe_relative_paths(unsafe_paths)


@pytest.mark.parametrize(
    "unsafe_tree",
    (
        {"nested": {"private_key": "fixture-value"}},
        {"nested": {"accepted": True}},
        {"nested": {"status": "completed"}},
        {"nested": {"authoritative": True}},
    ),
)
def test_recursive_private_authority_and_completion_markers_fail_closed(
    unsafe_tree: dict[str, Any],
) -> None:
    with pytest.raises(AssertionError):
        _validate_json_tree(unsafe_tree, fixture_id="negative")


def test_manifest_content_or_seal_drift_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "manifest.json"
    shutil.copy2(MANIFEST, copied)
    payload = json.loads(copied.read_text(encoding="utf-8"))
    payload["corpus_id"] = "drifted"
    copied.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        _load_and_validate_manifest(copied)


def test_symlink_manifest_fails_closed(tmp_path: Path) -> None:
    linked = tmp_path / "manifest.json"
    linked.symlink_to(MANIFEST)
    with pytest.raises(AssertionError):
        _load_and_validate_manifest(linked)


def test_executable_manifest_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "manifest.json"
    shutil.copy2(MANIFEST, copied)
    copied.chmod(copied.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(AssertionError):
        _load_and_validate_manifest(copied)


def test_hardlinked_manifest_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "manifest.json"
    alias = tmp_path / "alias.json"
    shutil.copy2(MANIFEST, copied)
    os.link(copied, alias)
    with pytest.raises(AssertionError):
        _load_and_validate_manifest(copied)


def test_nonregular_manifest_fails_closed(tmp_path: Path) -> None:
    fifo = tmp_path / "manifest.json"
    os.mkfifo(fifo)
    with pytest.raises(AssertionError):
        _load_and_validate_manifest(fifo)
