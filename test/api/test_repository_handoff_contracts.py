"""Deterministic tests for EAAEF-020 repository handoff and overlay contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.repository_handoff.contracts import (
    ABSOLUTE_MAX_OBJECTS,
    ATTRIBUTE_AND_MODE_RECORD_INTERFACE,
    CONTRACT_VERSION,
    HOOK_POLICY_INTERFACE,
    LFS_POINTER_RECORD_INTERFACE,
    LFS_POINTER_VERSION,
    NESTED_REPO_RECORD_INTERFACE,
    ORIGIN_AND_SHALLOW_BOUNDS_INTERFACE,
    REPOSITORY_HANDOFF_CONTRACT_FAMILY,
    REPOSITORY_HANDOFF_CONTRACT_VERSION,
    REPOSITORY_HANDOFF_REFUSAL_INTERFACE,
    REPOSITORY_HANDOFF_REQUEST_INTERFACE,
    REPOSITORY_HANDOFF_REQUEST_SCHEMA,
    REPOSITORY_OVERLAY_INTERFACE,
    SCHEMA_VERSION,
    SPARSE_CHECKOUT_RECORD_INTERFACE,
    SUBMODULE_RECORD_INTERFACE,
    AttributeAndModeRecord,
    FileKind,
    HookPolicy,
    IndexEntry,
    LfsPointerRecord,
    NestedGitDirKind,
    NestedRepoRecord,
    OriginAndShallowBounds,
    RefEntry,
    RefusalReason,
    RepositoryHandoffBounds,
    RepositoryHandoffBoundsError,
    RepositoryHandoffContractError,
    RepositoryHandoffIdentityError,
    RepositoryHandoffMode,
    RepositoryHandoffRefusal,
    RepositoryHandoffRefusalError,
    RepositoryHandoffRequest,
    RepositoryHandoffVersionError,
    RepositoryOverlay,
    SparseCheckoutRecord,
    SubmoduleIgnore,
    SubmoduleRecord,
    UntrackedEntry,
    WorktreeEntry,
    canonical_repository_handoff_json_bytes,
    decode_repository_handoff_contract,
    refusal_from_error,
)


FIXED_MS = 1_700_000_000_000
GIT_A = "a" * 40
GIT_B = "b" * 40
GIT_C = "c" * 40
GIT_D = "d" * 40
SHA_A = "sha256:" + ("a" * 64)
SHA_B = "sha256:" + ("b" * 64)
SHA_C = "sha256:" + ("c" * 64)
SHA_D = "sha256:" + ("d" * 64)
SHA_E = "sha256:" + ("e" * 64)
DEFAULT_OVERFLOW = RepositoryHandoffBounds().max_objects + 1
DEFAULT_BYTE_OVERFLOW = RepositoryHandoffBounds().max_object_bytes + 1


def _head_refs(*, detached: bool = False, head_ref: str = "refs/heads/main") -> tuple[RefEntry, ...]:
    if detached:
        return (RefEntry(name="HEAD", object_id=GIT_A),)
    return (
        RefEntry(name="HEAD", object_id=GIT_A, symbolic_target=head_ref),
        RefEntry(name=head_ref, object_id=GIT_A),
        RefEntry(name="refs/tags/v1", object_id=GIT_A),
    )


def _overlay(**changes: object) -> RepositoryOverlay:
    values: dict[str, object] = {
        "head_commit": GIT_A,
        "head_ref": "refs/heads/main",
        "detached": False,
        "refs": _head_refs(),
        "index": (
            IndexEntry(path="README.md", mode=0o100644, object_id=GIT_B),
            IndexEntry(path="vendor/lib", mode=0o160000, object_id=GIT_C),
        ),
        "worktree": (
            WorktreeEntry(
                path="README.md",
                mode=0o100644,
                content_id=SHA_A,
                object_id=GIT_B,
                byte_count=12,
            ),
            WorktreeEntry(
                path="docs/link.md",
                mode=0o120000,
                kind=FileKind.SYMLINK,
                symlink_target="README.md",
                byte_count=9,
            ),
        ),
        "untracked": (
            UntrackedEntry(
                path="scratch.txt",
                mode=0o100644,
                content_id=SHA_B,
                byte_count=4,
            ),
        ),
        "object_count": 4,
        "object_bytes": 128,
    }
    values.update(changes)
    return RepositoryOverlay(**values)


def _request(overlay: RepositoryOverlay | None = None, **changes: object) -> RepositoryHandoffRequest:
    overlay = overlay or _overlay()
    values: dict[str, object] = {
        "overlay": overlay,
        "caller_principal_id": "principal:operator",
        "idempotency_key": "idempotency:repo-handoff-1",
        "hook_policy": HookPolicy(present_hook_names=("pre-commit", "pre-push")),
        "origin_and_shallow": OriginAndShallowBounds(
            origin_url="https://github.com/example/repo.git",
            object_count=overlay.object_count,
            object_bytes=overlay.object_bytes,
        ),
        "sparse_checkout": SparseCheckoutRecord(
            enabled=True,
            cone=True,
            patterns=("src", "README.md"),
        ),
        "submodules": (
            SubmoduleRecord(
                path="vendor/lib",
                commit=GIT_C,
                url="../lib.git",
                name="lib",
            ),
        ),
        "nested_repos": (
            NestedRepoRecord(
                path="tools/vendor-src",
                head_commit=GIT_D,
                git_dir_kind=NestedGitDirKind.GIT_DIR,
            ),
        ),
        "lfs_pointers": (
            LfsPointerRecord(path="assets/model.bin", oid=SHA_C, size_bytes=2048),
        ),
        "attributes_and_modes": (
            AttributeAndModeRecord(
                path="README.md",
                mode=0o100644,
                attributes={"text": "auto", "eol": "lf"},
            ),
        ),
        "session_id": SHA_D,
        "object_bundle_id": SHA_E,
        "mode": RepositoryHandoffMode.PREVIEW,
        "created_at_ms": FIXED_MS,
    }
    values.update(changes)
    return RepositoryHandoffRequest(**values)


def test_frozen_contract_family_covers_required_interfaces() -> None:
    assert REPOSITORY_HANDOFF_CONTRACT_VERSION == 1
    assert CONTRACT_VERSION == 1
    assert SCHEMA_VERSION == 1
    expected = {
        "request": REPOSITORY_HANDOFF_REQUEST_INTERFACE,
        "overlay": REPOSITORY_OVERLAY_INTERFACE,
        "submodule": SUBMODULE_RECORD_INTERFACE,
        "nested_repo": NESTED_REPO_RECORD_INTERFACE,
        "lfs_pointer": LFS_POINTER_RECORD_INTERFACE,
        "sparse_checkout": SPARSE_CHECKOUT_RECORD_INTERFACE,
        "hook_policy": HOOK_POLICY_INTERFACE,
        "attribute_and_mode": ATTRIBUTE_AND_MODE_RECORD_INTERFACE,
        "origin_and_shallow": ORIGIN_AND_SHALLOW_BOUNDS_INTERFACE,
        "refusal": REPOSITORY_HANDOFF_REFUSAL_INTERFACE,
    }
    assert dict(REPOSITORY_HANDOFF_CONTRACT_FAMILY) == expected
    assert all(value.endswith("@1") for value in expected.values())


def test_nested_contracts_round_trip_and_preserve_content_identity() -> None:
    overlay = _overlay()
    request = _request(overlay)
    values = (
        RepositoryHandoffBounds(),
        overlay,
        request.hook_policy,
        request.origin_and_shallow,
        request.sparse_checkout,
        request.submodules[0],
        request.nested_repos[0],
        request.lfs_pointers[0],
        request.attributes_and_modes[0],
        request,
        RepositoryHandoffRefusal(
            reason=RefusalReason.SYMLINK_ESCAPE,
            message="symlink escape refused",
            created_at_ms=FIXED_MS,
        ),
    )
    for value in values:
        restored = type(value).from_json(value.to_json())
        assert restored == value
        assert restored.content_id == value.content_id
        assert decode_repository_handoff_contract(json.loads(value.to_json())) == value
        payload = json.loads(value.to_json())
        assert payload["contract_version"] == 1
        assert payload["schema"].endswith("@1")
        assert payload["interface"].endswith("@1")


def test_serialization_and_identity_are_deterministic_across_input_order() -> None:
    first = AttributeAndModeRecord(
        path="README.md",
        mode=0o100644,
        attributes={"eol": "lf", "text": "auto"},
    )
    second = AttributeAndModeRecord(
        path="README.md",
        mode=0o100644,
        attributes={"text": "auto", "eol": "lf"},
    )
    assert first.to_json() == second.to_json()
    assert first.content_id == second.content_id
    assert first.to_json() == json.dumps(
        json.loads(first.to_json()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert canonical_repository_handoff_json_bytes(first) == first.canonical_bytes()


def test_records_are_frozen() -> None:
    overlay = _overlay()
    with pytest.raises(FrozenInstanceError):
        overlay.head_commit = GIT_B  # type: ignore[misc]
    request = _request(overlay)
    with pytest.raises(FrozenInstanceError):
        request.mode = RepositoryHandoffMode.RECONSTRUCT  # type: ignore[misc]


def test_unknown_schema_and_version_are_rejected() -> None:
    payload = _overlay().to_dict()
    payload["schema"] = "ipfs_accelerate_py/agent-supervisor/repository-overlay@2"
    with pytest.raises(RepositoryHandoffVersionError):
        RepositoryOverlay.from_dict(payload)
    payload = _request().to_dict()
    payload["interface"] = "RepositoryHandoffRequest@2"
    with pytest.raises(RepositoryHandoffVersionError):
        RepositoryHandoffRequest.from_dict(payload)
    payload = _overlay().to_dict()
    payload["contract_version"] = 2
    with pytest.raises(RepositoryHandoffVersionError):
        RepositoryOverlay.from_dict(payload)
    with pytest.raises(RepositoryHandoffVersionError):
        decode_repository_handoff_contract(
            {"schema": "UnknownRecord@1", "contract_version": 1}
        )


def test_unknown_top_level_fields_are_rejected() -> None:
    payload = _overlay().to_dict()
    payload["extra"] = "nope"
    with pytest.raises(RepositoryHandoffContractError, match="unsupported fields"):
        RepositoryOverlay.from_dict(payload)


def test_forged_content_identity_is_rejected() -> None:
    payload = _overlay().to_dict()
    payload["content_id"] = SHA_A
    with pytest.raises(RepositoryHandoffIdentityError):
        RepositoryOverlay.from_dict(payload)
    payload = _request().to_dict()
    payload["request_id"] = SHA_A
    with pytest.raises(RepositoryHandoffIdentityError):
        RepositoryHandoffRequest.from_dict(payload)


def test_hidden_chain_of_thought_and_private_material_are_rejected() -> None:
    with pytest.raises(RepositoryHandoffContractError, match="hidden chain-of-thought"):
        AttributeAndModeRecord(
            path="README.md",
            mode=0o100644,
            attributes={"thinking": "secret scratchpad"},
        )
    with pytest.raises(RepositoryHandoffRefusalError, match="private material") as denied:
        AttributeAndModeRecord(
            path="README.md",
            mode=0o100644,
            attributes={"api_key": "sk-test"},
        )
    assert denied.value.reason is RefusalReason.PRIVATE_MATERIAL
    payload = _overlay().to_dict()
    payload["raw_bytes"] = "blob"
    with pytest.raises(RepositoryHandoffContractError, match="unsupported fields"):
        RepositoryOverlay.from_dict(payload)


def test_floats_and_malformed_json_fail_closed() -> None:
    with pytest.raises(RepositoryHandoffContractError, match="floats"):
        AttributeAndModeRecord(
            path="README.md",
            mode=0o100644,
            attributes={"score": 1.5},  # type: ignore[dict-item]
        )
    with pytest.raises(RepositoryHandoffContractError, match="malformed"):
        RepositoryOverlay.from_json("{")
    with pytest.raises(RepositoryHandoffContractError, match="object"):
        RepositoryOverlay.from_json("[]")
    with pytest.raises(RepositoryHandoffContractError):
        decode_repository_handoff_contract("not-an-object")  # type: ignore[arg-type]


def test_overlay_accounts_for_head_refs_index_worktree_and_untracked() -> None:
    overlay = _overlay()
    payload = overlay.to_dict()
    assert payload["head_name"] == "HEAD"
    assert payload["head_ref"] == "refs/heads/main"
    assert payload["head_commit"] == GIT_A
    assert payload["detached"] is False
    assert [item["name"] for item in payload["refs"]] == [
        "HEAD",
        "refs/heads/main",
        "refs/tags/v1",
    ]
    assert payload["index"][0]["path"] == "README.md"
    assert payload["index"][1]["kind"] == FileKind.GITLINK.value
    assert payload["worktree"][1]["kind"] == FileKind.SYMLINK.value
    assert payload["untracked"][0]["path"] == "scratch.txt"
    assert "body" not in json.dumps(payload)
    detached = _overlay(
        detached=True,
        head_ref="",
        refs=_head_refs(detached=True),
        index=(),
        worktree=(),
        untracked=(),
        object_count=1,
        object_bytes=64,
    )
    assert detached.detached is True
    assert detached.head_ref == ""


def test_symlink_escape_is_a_typed_refusal() -> None:
    with pytest.raises(RepositoryHandoffRefusalError, match="symlink escape") as escaped:
        _overlay(
            worktree=(
                WorktreeEntry(
                    path="evil",
                    mode=0o120000,
                    kind=FileKind.SYMLINK,
                    symlink_target="../outside",
                ),
            )
        )
    assert escaped.value.reason is RefusalReason.SYMLINK_ESCAPE
    with pytest.raises(RepositoryHandoffRefusalError, match="symlink escape"):
        UntrackedEntry(
            path="link",
            mode=0o120000,
            kind=FileKind.SYMLINK,
            symlink_target="/etc/passwd",
        )
    refusal = refusal_from_error(escaped.value, created_at_ms=FIXED_MS)
    assert refusal.reason is RefusalReason.SYMLINK_ESCAPE
    assert refusal.interface == REPOSITORY_HANDOFF_REFUSAL_INTERFACE
    restored = RepositoryHandoffRefusal.from_json(refusal.to_json())
    assert restored == refusal


def test_enabled_hooks_are_refused_and_must_be_disabled_on_import() -> None:
    policy = HookPolicy(present_hook_names=("pre-commit",))
    assert policy.hooks_enabled is False
    assert policy.import_hooks_disabled is True
    assert policy.to_dict()["hooks_enabled"] is False
    with pytest.raises(RepositoryHandoffRefusalError, match="disabled on import") as enabled:
        HookPolicy(hooks_enabled=True)
    assert enabled.value.reason is RefusalReason.ENABLED_HOOKS
    with pytest.raises(RepositoryHandoffRefusalError, match="disabled on import"):
        HookPolicy(import_hooks_disabled=False)
    with pytest.raises(RepositoryHandoffRefusalError, match="hooksPath"):
        HookPolicy(core_hooks_path="/tmp/hooks")
    payload = policy.to_dict()
    payload["hooks_enabled"] = True
    with pytest.raises(RepositoryHandoffRefusalError) as decoded:
        HookPolicy.from_dict(payload)
    assert decoded.value.reason is RefusalReason.ENABLED_HOOKS


def test_unbounded_objects_are_a_typed_refusal() -> None:
    with pytest.raises(RepositoryHandoffRefusalError, match="unbounded objects") as overlay_denied:
        _overlay(unbounded_objects=True)
    assert overlay_denied.value.reason is RefusalReason.UNBOUNDED_OBJECTS
    with pytest.raises(RepositoryHandoffRefusalError, match="exceed declared bounds"):
        _overlay(object_count=DEFAULT_OVERFLOW)
    tight = RepositoryHandoffBounds(max_objects=2, max_object_bytes=64)
    with pytest.raises(RepositoryHandoffRefusalError, match="exceed declared bounds"):
        OriginAndShallowBounds(
            object_count=8,
            object_bytes=8,
            max_objects=2,
            max_object_bytes=64,
            bounds=tight,
        )
    with pytest.raises(RepositoryHandoffRefusalError, match="unbounded objects"):
        OriginAndShallowBounds(unbounded_objects=True)
    with pytest.raises(RepositoryHandoffRefusalError, match="LFS object exceeds"):
        LfsPointerRecord(path="big.bin", oid=SHA_C, size_bytes=DEFAULT_BYTE_OVERFLOW)


def test_submodules_nested_repos_lfs_sparse_origin_and_modes() -> None:
    request = _request()
    assert request.schema == REPOSITORY_HANDOFF_REQUEST_SCHEMA
    assert request.submodules[0].commit == GIT_C
    assert request.submodules[0].ignore is SubmoduleIgnore.NONE
    assert request.nested_repos[0].git_dir_kind is NestedGitDirKind.GIT_DIR
    assert request.lfs_pointers[0].version == LFS_POINTER_VERSION
    assert request.sparse_checkout.cone is True
    assert request.origin_and_shallow.shallow is False
    assert request.attributes_and_modes[0].kind is FileKind.REGULAR
    assert request.overlay_id == request.overlay.content_id
    assert request.request_id.startswith("b")
    shallow = OriginAndShallowBounds(
        origin_url="git@github.com:example/repo.git",
        shallow=True,
        depth=1,
        filter_spec="blob:none",
        promisor=True,
        unshallow_required=True,
        object_count=4,
        object_bytes=128,
    )
    assert shallow.depth == 1
    assert decode_repository_handoff_contract(shallow.to_dict()) == shallow


def test_host_path_origin_and_git_dir_escape_are_refused() -> None:
    with pytest.raises(RepositoryHandoffRefusalError, match="host filesystem path") as origin_denied:
        OriginAndShallowBounds(origin_url="file:///tmp/repo.git")
    assert origin_denied.value.reason is RefusalReason.HOST_PATH_ORIGIN
    with pytest.raises(RepositoryHandoffRefusalError, match="host filesystem path"):
        SubmoduleRecord(path="vendor/lib", commit=GIT_C, url="/tmp/lib.git")
    with pytest.raises(RepositoryHandoffRefusalError, match="nested git escape"):
        _overlay(
            index=(IndexEntry(path=".git/config", mode=0o100644, object_id=GIT_B),)
        )
    payload = _request().to_dict()
    payload["host_path"] = "/tmp/checkout"
    with pytest.raises(RepositoryHandoffRefusalError, match="host filesystem"):
        RepositoryHandoffRequest.from_dict(payload)


def test_paths_must_be_repository_relative_and_git_object_ids_exact() -> None:
    with pytest.raises(RepositoryHandoffContractError, match="repository-relative"):
        IndexEntry(path="../secret.py", mode=0o100644, object_id=GIT_B)
    with pytest.raises(RepositoryHandoffContractError, match="repository-relative"):
        SparseCheckoutRecord(enabled=True, patterns=("../etc",))
    with pytest.raises(RepositoryHandoffContractError, match="Git object id"):
        _overlay(head_commit="not-a-git-oid")
    with pytest.raises(RepositoryHandoffContractError, match="sha256 or CIDv1"):
        _request(session_id="session:not-addressed")
    with pytest.raises(RepositoryHandoffContractError, match="Git file mode"):
        IndexEntry(path="README.md", mode=0o100666, object_id=GIT_B)


def test_bounds_reject_absolute_and_relative_overflow() -> None:
    with pytest.raises(RepositoryHandoffBoundsError):
        RepositoryHandoffBounds(max_objects=ABSOLUTE_MAX_OBJECTS + 1)
    with pytest.raises(RepositoryHandoffBoundsError):
        RepositoryHandoffBounds(max_text_bytes=8_000, max_record_bytes=4_000)
    tight = RepositoryHandoffBounds(
        max_refs=1,
        max_index_entries=1,
        max_worktree_entries=1,
        max_untracked=1,
        max_record_bytes=32_768,
        max_serialized_bytes=65_536,
        max_unknown_field_bytes=2_048,
    )
    with pytest.raises(RepositoryHandoffBoundsError):
        _overlay(bounds=tight)


def test_request_rejects_provider_selection_and_keeps_identities_distinct() -> None:
    request = _request()
    assert request.mode is RepositoryHandoffMode.PREVIEW
    assert request.overlay_id != request.request_id
    assert request.overlay_id != request.session_id
    assert request.hook_policy.hook_policy_id != request.origin_and_shallow.origin_bounds_id
    payload = request.to_dict()
    payload["provider_id"] = "grok"
    with pytest.raises(RepositoryHandoffContractError, match="cannot select a provider"):
        RepositoryHandoffRequest.from_dict(payload)
    reconstruct = _request(mode=RepositoryHandoffMode.RECONSTRUCT)
    assert reconstruct.mode is RepositoryHandoffMode.RECONSTRUCT
    inspect = _request(mode=RepositoryHandoffMode.INSPECT)
    assert inspect.to_dict()["mode"] == "inspect"


def test_decode_accepts_interface_without_schema_and_rejects_inconsistent_head() -> None:
    payload = _overlay().to_dict()
    payload.pop("schema")
    restored = decode_repository_handoff_contract(payload)
    assert isinstance(restored, RepositoryOverlay)
    assert restored.overlay_id == _overlay().overlay_id
    with pytest.raises(RepositoryHandoffIdentityError, match="head_commit"):
        _overlay(
            refs=(
                RefEntry(name="HEAD", object_id=GIT_B, symbolic_target="refs/heads/main"),
                RefEntry(name="refs/heads/main", object_id=GIT_A),
            )
        )
    with pytest.raises(RepositoryHandoffContractError, match="HEAD"):
        RepositoryOverlay(head_commit=GIT_A, refs=(), head_ref="refs/heads/main")


def test_public_records_are_references_not_object_bodies() -> None:
    request = _request()
    encoded = request.to_json()
    assert "raw_bytes" not in encoded
    assert "blob_bytes" not in encoded
    assert LFS_POINTER_VERSION in encoded
    assert request.lfs_pointers[0].oid.startswith("sha256:")
    worktree = request.overlay.worktree[0]
    assert worktree.content_id == SHA_A
    assert "body" not in worktree.to_dict()
    payload = request.to_dict()
    payload.pop("schema")
    restored = decode_repository_handoff_contract(payload)
    assert isinstance(restored, RepositoryHandoffRequest)
    assert restored.request_id == request.request_id
