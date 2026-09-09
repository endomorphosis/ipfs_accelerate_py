"""PCCE-054 truthful clean-install/no-go contract tests."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "proof_context" / "test_clean_install.py"
LOCK_ROOT = PROJECT_ROOT / "packaging" / "proof_context" / "locks"
ENVIRONMENT_ROOT = LOCK_ROOT / "cpython312-linux-aarch64"


def _load_harness():
    spec = importlib.util.spec_from_file_location("pcce054_clean_install", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


def _bind_synthetic_manifest(monkeypatch: pytest.MonkeyPatch, manifest_path: Path) -> None:
    identity = HARNESS._identity_for_bytes(manifest_path.read_bytes())
    monkeypatch.setattr(
        HARNESS,
        "FROZEN_ARTIFACT_MANIFEST_IDENTITY",
        {"sha256": identity["sha256"], "cid_v1_raw": identity["cid_v1_raw"]},
    )


def _install_resolver_variant(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    mutate,
    *,
    bind_expected: bool,
) -> None:
    receipt_path = ENVIRONMENT_ROOT / f"{profile}.resolver.json"
    value = json.loads(receipt_path.read_text())
    mutate(value)
    raw = HARNESS._canonical_json_bytes(value)
    if bind_expected:
        frozen = copy.deepcopy(HARNESS.FROZEN_PROFILE_INPUT_IDENTITIES)
        identity = HARNESS._identity_for_bytes(raw)
        frozen[profile]["resolver_receipt"] = {
            "sha256": identity["sha256"],
            "cid_v1_raw": identity["cid_v1_raw"],
        }
        monkeypatch.setattr(HARNESS, "FROZEN_PROFILE_INPUT_IDENTITIES", frozen)
    original = HARNESS._load_canonical_object

    def load_variant(path: Path):
        if path.resolve() == receipt_path.resolve():
            return copy.deepcopy(value), raw
        return original(path)

    monkeypatch.setattr(HARNESS, "_load_canonical_object", load_variant)


def _descriptor(
    tmp_path: Path,
    *,
    distribution: str,
    version: str,
    kind: str,
    filename: str,
    available: bool,
) -> dict[str, object]:
    payload = f"{distribution}\0{version}\0{kind}\n".encode()
    digest = HARNESS._sha256_bytes(payload)
    if available:
        (tmp_path / filename).write_bytes(payload)
    return {
        "bytes_available": available,
        "bytes_verified": available,
        "cid_binding_status": (
            "bytes-verified" if available else "identity-derived-bytes-unavailable"
        ),
        "cid_v1_raw": HARNESS._raw_cid_v1_from_sha256(digest),
        "distribution": distribution,
        "filename": filename,
        "kind": kind,
        "sha256": digest,
        "size": len(payload) if available else None,
        "source_commit": "1" * 40,
        "version": version,
    }


def _artifact_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    descriptors = [
        _descriptor(
            tmp_path,
            distribution="ipfs-datasets-py",
            version="0.2.0",
            kind="wheel",
            filename="ipfs_datasets_py-0.2.0-cp312-cp312-linux_aarch64.whl",
            available=True,
        ),
        _descriptor(
            tmp_path,
            distribution="ipfs-datasets-py",
            version="0.2.0",
            kind="sdist",
            filename="ipfs_datasets_py-0.2.0.tar.gz",
            available=True,
        ),
        _descriptor(
            tmp_path,
            distribution="ipfs-kit-py",
            version="0.3.0",
            kind="wheel",
            filename="ipfs_kit_py-0.3.0-py3-none-any.whl",
            available=True,
        ),
        _descriptor(
            tmp_path,
            distribution="ipfs-kit-py",
            version="0.3.0",
            kind="sdist",
            filename="ipfs_kit_py-0.3.0.tar.gz",
            available=False,
        ),
        _descriptor(
            tmp_path,
            distribution="ipfs-accelerate-py",
            version="0.0.45",
            kind="wheel",
            filename="ipfs_accelerate_py-0.0.45-py3-none-any.whl",
            available=True,
        ),
        _descriptor(
            tmp_path,
            distribution="ipfs-accelerate-py",
            version="0.0.45",
            kind="sdist",
            filename="ipfs_accelerate_py-0.0.45.tar.gz",
            available=True,
        ),
        _descriptor(
            tmp_path,
            distribution="mcp-plus-plus-contracts",
            version="0.1.0",
            kind="wheel",
            filename="mcp_plus_plus_contracts-0.1.0-py3-none-any.whl",
            available=True,
        ),
        _descriptor(
            tmp_path,
            distribution="mcp-plus-plus-contracts",
            version="0.1.0",
            kind="sdist",
            filename="mcp_plus_plus_contracts-0.1.0.tar.gz",
            available=True,
        ),
    ]
    descriptors[3]["sha256"] = "8db7299f2cc144814d6b1b01a8476ba2daa67830856513f2863c6fac4af3ed15"
    descriptors[3]["cid_v1_raw"] = "bafkreienw4uz6lgbisau22y3agueo25c3kthqmefmuj7fbr4n6wev47ncu"
    manifest: dict[str, object] = {
        "artifact_byte_availability_status": HARNESS.BYTE_AVAILABILITY_STATUS,
        "artifact_clean_install_status": HARNESS.CLEAN_INSTALL_NO_GO,
        "artifacts": descriptors,
        "cid_policy": "CIDv1 with raw multicodec and sha2-256 multihash",
        "identity_policy": "SHA-256 over the exact admitted archive bytes",
        "resolution_status": HARNESS.RESOLUTION_STATUS,
        "schema": HARNESS.ARTIFACT_SCHEMA,
        "semantic_surrogates": [],
        "source_commits": {},
    }
    path = tmp_path / "artifact_hashes.json"
    path.write_bytes(HARNESS._canonical_json_bytes(manifest))
    return path, tmp_path, manifest


def _frozen_pairs() -> dict[tuple[str, str], dict[str, dict[str, object]]]:
    inputs = json.loads((LOCK_ROOT / "inputs.json").read_text())
    pairs: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for item in inputs["artifacts"]:
        key = (HARNESS._normalized_name(item["distribution"]), item["version"])
        pairs.setdefault(key, {})[item["kind"]] = item
    return pairs


def test_raw_cid_binds_the_exact_sha256() -> None:
    digest = HARNESS._sha256_bytes(b"PCCE-054\n")
    cid = HARNESS._raw_cid_v1_from_sha256(digest)

    HARNESS._verify_raw_cid_v1(cid, digest)
    assert cid.startswith("bafkrei")
    with pytest.raises(HARNESS.EvidenceError, match="does not bind"):
        HARNESS._verify_raw_cid_v1(cid, "0" * 64)
    with pytest.raises(HARNESS.EvidenceError, match="base32-lower"):
        HARNESS._verify_raw_cid_v1(cid.upper(), digest)


def test_artifact_manifest_verifies_every_available_byte_and_raw_cid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, artifact_root, _ = _artifact_fixture(tmp_path)
    _bind_synthetic_manifest(monkeypatch, manifest_path)

    manifest, pairs, evidence = HARNESS._validate_artifact_manifest(manifest_path, artifact_root)

    assert manifest["semantic_surrogates"] == []
    assert len(pairs) == 4
    assert evidence["byte_verified_count"] == 7
    assert evidence["identity_derived_unavailable_count"] == 1
    assert evidence["frozen_pcce053_identity_binding"] == "passed"
    HARNESS._verify_raw_cid_v1(evidence["cid_v1_raw"], evidence["sha256"])


def test_omitted_artifact_root_is_manifest_only_and_never_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _, _ = _artifact_fixture(tmp_path)
    _bind_synthetic_manifest(monkeypatch, manifest_path)

    _, pairs, evidence = HARNESS._validate_artifact_manifest(manifest_path, None)

    assert len(pairs) == 4
    assert evidence["artifact_root"] is None
    assert evidence["artifact_root_input"] == "not-provided"
    assert evidence["byte_verified_count"] == 0
    assert evidence["manifest_declared_byte_verified_count"] == 7
    assert (
        evidence["artifact_bytes_verification_status"]
        == "artifact-bytes-not-verified-artifact-root-not-provided"
    )


def test_self_consistent_unfrozen_artifact_manifest_is_rejected(tmp_path: Path) -> None:
    manifest_path, artifact_root, _ = _artifact_fixture(tmp_path)

    with pytest.raises(HARNESS.EvidenceError, match="exact frozen PCCE-053 identity"):
        HARNESS._validate_artifact_manifest(manifest_path, artifact_root)


def test_artifact_manifest_rejects_surrogates_and_tampered_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, artifact_root, manifest = _artifact_fixture(tmp_path)
    manifest["semantic_surrogates"] = [{"reason": "same-version-rebuild"}]
    manifest_path.write_bytes(HARNESS._canonical_json_bytes(manifest))
    _bind_synthetic_manifest(monkeypatch, manifest_path)
    with pytest.raises(HARNESS.EvidenceError, match="semantic surrogate"):
        HARNESS._validate_artifact_manifest(manifest_path, artifact_root)

    manifest["semantic_surrogates"] = []
    manifest_path.write_bytes(HARNESS._canonical_json_bytes(manifest))
    _bind_synthetic_manifest(monkeypatch, manifest_path)
    (artifact_root / "ipfs_kit_py-0.3.0-py3-none-any.whl").write_bytes(b"surrogate")
    with pytest.raises(HARNESS.EvidenceError, match="bytes do not match"):
        HARNESS._validate_artifact_manifest(manifest_path, artifact_root)


def test_unavailable_identity_cannot_be_filled_by_an_unadmitted_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, artifact_root, _ = _artifact_fixture(tmp_path)
    _bind_synthetic_manifest(monkeypatch, manifest_path)
    (artifact_root / "ipfs_kit_py-0.3.0.tar.gz").write_bytes(b"semantic rebuild")

    with pytest.raises(HARNESS.EvidenceError, match="marks artifact bytes unavailable"):
        HARNESS._validate_artifact_manifest(manifest_path, artifact_root)


def test_frozen_locks_are_exact_and_every_profile_is_source_only_no_go() -> None:
    pairs = _frozen_pairs()
    expected = {
        "core": {"python-baseconv-1.2.2.tar.gz", "varint-1.0.2.tar.gz"},
        "verification": {"python-baseconv-1.2.2.tar.gz", "varint-1.0.2.tar.gz"},
        "codex": {"python-baseconv-1.2.2.tar.gz", "varint-1.0.2.tar.gz"},
        "local-model": {
            "llama_cpp_python-0.3.35.tar.gz",
            "python-baseconv-1.2.2.tar.gz",
            "varint-1.0.2.tar.gz",
        },
        "evaluation": {
            "llama_cpp_python-0.3.35.tar.gz",
            "python-baseconv-1.2.2.tar.gz",
            "varint-1.0.2.tar.gz",
        },
    }

    for profile in HARNESS.PROFILE_ORDER:
        evidence = HARNESS._validate_profile_inputs(profile, pairs)
        selected = {item["filename"] for item in evidence["selected_source_archives"]}
        assert selected == expected[profile]
        assert evidence["receipt"]["artifact_clean_install_status"] == HARNESS.CLEAN_INSTALL_NO_GO
        assert evidence["requirement_risk_ledger"]["policy_status"] == "passed"
        assert evidence["requirement_risk_ledger"]["selected_unsafe_requirements_by_class"] == {
            "editable": [],
            "local-path": [],
            "mutable-vcs": [],
            "unadmitted-direct-url": [],
        }
        assert (
            evidence["requirement_risk_ledger"][
                "selected_unsafe_vcs_direct_editable_path_requirements"
            ]
            == []
        )
        HARNESS._verify_raw_cid_v1(evidence["lock"]["cid_v1_raw"], evidence["lock"]["sha256"])
        assert evidence["lock"]["frozen_pcce053_identity_binding"] == "passed"
        HARNESS._verify_raw_cid_v1(
            evidence["resolver_receipt"]["cid_v1_raw"],
            evidence["resolver_receipt"]["sha256"],
        )
        assert evidence["resolver_receipt"]["frozen_pcce053_identity_binding"] == "passed"


def test_resolver_semantic_surrogate_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs = _frozen_pairs()
    _install_resolver_variant(
        monkeypatch,
        "core",
        lambda value: value["packages"][0].__setitem__(
            "resolution_surrogate", {"sha256": "0" * 64}
        ),
        bind_expected=True,
    )
    with pytest.raises(HARNESS.EvidenceError, match="semantic surrogate"):
        HARNESS._validate_profile_inputs("core", pairs)


def test_self_consistent_unfrozen_resolver_receipt_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _frozen_pairs()
    _install_resolver_variant(
        monkeypatch,
        "core",
        lambda value: value["packages"][0].__setitem__(
            "resolution_surrogate", {"sha256": "0" * 64}
        ),
        bind_expected=False,
    )
    with pytest.raises(HARNESS.EvidenceError, match="exact frozen PCCE-053 identity"):
        HARNESS._validate_profile_inputs("core", pairs)


def test_nonpassing_requirement_risk_ledger_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _frozen_pairs()

    def make_unsafe(value):
        value["requirement_risk_ledger"]["policy_status"] = "failed"
        value["requirement_risk_ledger"]["selected_unsafe_requirements_by_class"]["mutable-vcs"] = [
            "example @ git+https://example.invalid/repository.git@main"
        ]

    _install_resolver_variant(monkeypatch, "core", make_unsafe, bind_expected=True)
    with pytest.raises(HARNESS.EvidenceError, match="risk policy did not pass"):
        HARNESS._validate_profile_inputs("core", pairs)


def test_source_path_guard_and_isolated_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    source = PROJECT_ROOT.resolve()
    outside = source.parent / "pcce054-outside"

    HARNESS._assert_source_path_isolation(
        cwd=outside,
        sys_path=["/usr/lib/python3.12", "/tmp/site-packages"],
        source_roots=[source],
    )
    with pytest.raises(HARNESS.EvidenceError, match="cwd is inside source tree"):
        HARNESS._assert_source_path_isolation(
            cwd=source / "test", sys_path=[], source_roots=[source]
        )
    with pytest.raises(HARNESS.EvidenceError, match="sys.path reaches source tree"):
        HARNESS._assert_source_path_isolation(
            cwd=outside, sys_path=[str(source)], source_roots=[source]
        )

    environment = HARNESS._offline_environment(
        {
            "PYTHONHOME": "/source/python",
            "PYTHONPATH": "/source/repo",
            "PIP_INDEX_URL": "https://example.invalid/simple",
            "PIP_EXTRA_INDEX_URL": "https://example.invalid/extra",
        }
    )
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert "PIP_INDEX_URL" not in environment
    assert "PIP_EXTRA_INDEX_URL" not in environment
    assert environment["PIP_NO_INDEX"] == "1"

    monkeypatch.setattr(HARNESS, "ACCELERATOR_ROOT", source)
    trace = HARNESS._capture_source_path_trace()
    assert trace["status"] == "passed"
    assert trace["isolated_flag"] == 1
    HARNESS._verify_raw_cid_v1(
        trace["transcript_identity"]["cid_v1_raw"],
        trace["transcript_identity"]["sha256"],
    )


def test_report_records_no_execution_and_exit_five_when_qualification_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    report = {
        "schema": HARNESS.REPORT_SCHEMA,
        "qualification_status": "no-go",
        "profiles": [{"profile": "core", "qualification_status": "no-go"}],
    }
    monkeypatch.setattr(HARNESS, "build_report", lambda **_kwargs: report)
    output = tmp_path / "matrix.json"

    exit_code = HARNESS.main(
        [
            "--artifacts",
            str(tmp_path / "artifact_hashes.json"),
            "--artifact-root",
            str(tmp_path),
            "--output",
            str(output),
            "--require-qualified",
        ]
    )

    assert exit_code == 5
    assert json.loads(output.read_text()) == report
    assert json.loads(capsysbinary.readouterr().out) == report


def test_workflow_is_immutable_arm64_and_fails_on_current_no_go() -> None:
    text = (PROJECT_ROOT / ".github" / "workflows" / "proof-context-clean-install.yml").read_text()
    action_refs = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", text)

    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)
    assert "runs-on: ubuntu-24.04-arm" in text
    assert "repository: ${{ job.workflow_repository }}" in text
    assert "ref: ${{ job.workflow_sha }}" in text
    assert "repository: ${{ github.repository }}" not in text
    assert "ref: ${{ github.sha }}" not in text
    assert "run-id: ${{ inputs.artifact_run_id }}" in text
    assert "--artifact-root" in text
    assert "--require-qualified" in text
    assert "--network none" in text
    assert "continue-on-error" not in text
    assert "/home/" not in text
    assert text.index("--require-qualified") < text.index("docker buildx build")


def test_dockerfile_has_pinned_arm64_base_offline_wheels_and_numeric_user() -> None:
    text = (PROJECT_ROOT / "docker" / "proof-context" / "Dockerfile").read_text()
    from_lines = [line for line in text.splitlines() if line.startswith("FROM ")]

    assert text.startswith(
        "# syntax=docker/dockerfile:1.7@sha256:"
        "a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e\n"
    )
    expected_base = (
        "FROM docker.io/library/python:3.12.3-slim-bookworm@sha256:"
        "c27c26153fdf6863da2a1d85b474d1be004c42cc8ea2fd647004a8d4007a34d5 AS "
    )
    assert from_lines == [
        expected_base + "proof-context-build",
        expected_base + "proof-context-runtime",
    ]
    assert "COPY --from=proof_context_artifacts" in text
    assert "--no-index" in text
    assert "--only-binary=:all:" in text
    assert "--require-hashes" in text
    assert "--require-qualified" in text
    assert "USER 65532:65532" in text
    assert "COPY ." not in text
    assert not re.search(r"\b(?:apt|apk|curl|wget)\b", text)
