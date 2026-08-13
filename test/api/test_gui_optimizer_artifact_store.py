"""VGO-054: host-owned evidence artifact store tests.

Acceptance coverage:

* artifact bytes resolve only through verified CIDs
* stored bytes rehash before any return
* identical payloads share one CID
* corrupt or truncated blobs fail closed
* browser-selected and escaping paths reject
* reuse requires exact repository/component/scenario/extractor/checker identities
* reuse never becomes current verification authority
* closed wire inputs reject unknown fields, nulls, and path keys
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.artifact_store import (
    BROAD_ROOTS,
    GUI_EVIDENCE_ARTIFACT_STORE_INTERFACE,
    ArtifactKind,
    ArtifactReuseGate,
    ArtifactStoreReasonCode,
    GuiArtifactStoreError,
    GuiEvidenceArtifactStore,
    artifact_cid_for_bytes,
    artifact_digest_for_bytes,
    default_evidence_artifact_store,
)
from ipfs_datasets_py.logic.gui_optimizer.identity import parse_cid_v1

REVISION = "a" * 40
SCREENSHOT = b"\x89PNG\r\n\x1a\n" + b"vgo-screenshot-bytes"


def _gate(**overrides: Any) -> dict[str, str]:
    payload = {
        "repository_id": "repository:verified-gui-optimizer",
        "repository_revision": REVISION,
        "component_id": "comp:goal-form",
        "scenario_id": "scenario:keyboard-desktop",
        "extractor_id": "extractor:playwright@1",
        "extractor_version": "playwright@1.0.0",
        "checker_id": "checker:visual-regression@1",
        "checker_version": "visual-regression@1.0.0",
    }
    payload.update(overrides)
    return payload


def _store(tmp_path: Path) -> GuiEvidenceArtifactStore:
    return default_evidence_artifact_store(tmp_path / "artifacts")


def test_put_get_rehashes_bytes_through_verified_cid(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.put(
        SCREENSHOT,
        kind=ArtifactKind.SCREENSHOT,
        binding=_gate(),
    )
    parsed = parse_cid_v1(record.cid)
    assert parsed["cid"] == record.cid
    assert record.digest == artifact_digest_for_bytes(SCREENSHOT)
    assert record.cid == artifact_cid_for_bytes(SCREENSHOT)
    assert record.is_current_authority is False
    body, loaded = store.get(record.cid, kind=ArtifactKind.SCREENSHOT)
    assert body == SCREENSHOT
    assert loaded.cid == record.cid
    assert store.rehash(record.cid).digest == record.digest
    again = store.put(SCREENSHOT, kind=ArtifactKind.SCREENSHOT, binding=_gate())
    assert again.cid == record.cid


def test_get_rejects_caller_path_and_resolves_only_by_cid(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.put(SCREENSHOT, kind="screenshot", binding=_gate())
    with pytest.raises(GuiArtifactStoreError) as exc:
        store.get_from_mapping(
            {
                "cid": record.cid,
                "path": str(store.host_path_for_cid(record.cid)),
            }
        )
    assert exc.value.reason_code == ArtifactStoreReasonCode.BROWSER_PATH_FORBIDDEN.value
    confined = store.host_path_for_cid(record.cid)
    assert confined.is_relative_to(store.host_root)
    assert confined.name == f"{record.cid}.bin"


def test_path_escape_and_browser_roots_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(GuiArtifactStoreError) as browser:
        GuiEvidenceArtifactStore(host_root="file:///tmp/artifacts")
    assert (
        browser.value.reason_code
        == ArtifactStoreReasonCode.BROWSER_PATH_FORBIDDEN.value
    )
    with pytest.raises(GuiArtifactStoreError) as traversal:
        GuiEvidenceArtifactStore(host_root=str(tmp_path / ".." / "escaped"))
    assert (
        traversal.value.reason_code
        == ArtifactStoreReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    )
    with pytest.raises(GuiArtifactStoreError) as broad:
        GuiEvidenceArtifactStore(host_root="/tmp")
    assert broad.value.reason_code == ArtifactStoreReasonCode.BROAD_ROOT_FORBIDDEN.value
    assert "/tmp" in BROAD_ROOTS
    store = _store(tmp_path)
    with pytest.raises(GuiArtifactStoreError) as put_path:
        store.put_from_mapping(
            {
                "kind": "trace",
                "binding": _gate(),
                "text": '{"ok":true}',
                "host_path": "/etc/passwd",
            }
        )
    assert (
        put_path.value.reason_code
        == ArtifactStoreReasonCode.BROWSER_PATH_FORBIDDEN.value
    )


def test_corrupt_and_truncated_bytes_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.put(SCREENSHOT, kind=ArtifactKind.SCREENSHOT, binding=_gate())
    blob = store.host_path_for_cid(record.cid)
    blob.write_bytes(SCREENSHOT + b"tamper")
    with pytest.raises(GuiArtifactStoreError) as corrupt:
        store.get(record.cid)
    assert corrupt.value.reason_code == ArtifactStoreReasonCode.REHASH_MISMATCH.value
    blob.write_bytes(b"")
    with pytest.raises(GuiArtifactStoreError) as truncated:
        store.get(record.cid)
    assert truncated.value.reason_code == ArtifactStoreReasonCode.TRUNCATED_ARTIFACT.value


def test_reuse_requires_exact_gate_and_is_never_authority(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.put(
        {"observation": "missing-name", "node_id": "goal-input"},
        kind=ArtifactKind.ACCESSIBILITY,
        binding=_gate(),
    )
    body, reused = store.reuse(record.cid, _gate(), kind=ArtifactKind.ACCESSIBILITY)
    assert b"missing-name" in body
    assert reused.is_current_authority is False
    assert store.is_current_authority(record.cid) is False
    mismatch = _gate(checker_version="visual-regression@9.9.9")
    with pytest.raises(GuiArtifactStoreError) as exc:
        store.reuse(record.cid, mismatch)
    assert exc.value.reason_code == ArtifactStoreReasonCode.REUSE_GATE_MISMATCH.value


def test_manifest_rehashes_every_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    shot = store.put(SCREENSHOT, kind=ArtifactKind.SCREENSHOT, binding=_gate())
    trace = store.put(
        {"events": ["focus", "type"], "scenario_id": "scenario:keyboard-desktop"},
        kind=ArtifactKind.TRACE,
        binding=_gate(checker_id="checker:interaction@1"),
    )
    manifest = store.put_manifest(
        run_id="run:agent-supervisor-1",
        artifacts=[shot, trace.cid],
        binding=_gate(component_id="comp:run-manifest"),
    )
    loaded = store.get_manifest(manifest.cid, required_gate=_gate(component_id="comp:run-manifest"))
    assert loaded.artifact_cids == tuple(sorted((shot.cid, trace.cid)))
    assert loaded.run_id == "run:agent-supervisor-1"
    assert all(store.rehash(cid).cid == cid for cid in loaded.artifact_cids)


def test_closed_wire_inputs_reject_unknown_null_and_wrong_containers(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    with pytest.raises(GuiArtifactStoreError) as unknown:
        store.put_from_mapping(
            {
                "kind": "baseline",
                "binding": _gate(),
                "payload": {"ok": True},
                "vendor": "hidden",
            }
        )
    assert unknown.value.reason_code == ArtifactStoreReasonCode.UNKNOWN_FIELD.value
    with pytest.raises(GuiArtifactStoreError) as null_gate:
        ArtifactReuseGate.from_mapping({**_gate(), "component_id": None})
    assert (
        null_gate.value.reason_code
        == ArtifactStoreReasonCode.INVALID_ARTIFACT_STORE_INPUT.value
    )
    with pytest.raises(GuiArtifactStoreError) as tuple_payload:
        store.put(("not", "bytes"), kind="trace", binding=_gate())  # type: ignore[arg-type]
    assert (
        tuple_payload.value.reason_code
        == ArtifactStoreReasonCode.INVALID_COLLECTION_TYPE.value
    )
    with pytest.raises(GuiArtifactStoreError):
        store.get("not-a-cid")
    assert store.interface == GUI_EVIDENCE_ARTIFACT_STORE_INTERFACE
    assert ArtifactReuseGate.from_mapping(_gate()).matches(
        ArtifactReuseGate.from_any(_gate())
    )
