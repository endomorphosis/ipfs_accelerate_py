"""Hermetic PCAR-025 cross-repository read-only contract audit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.cross_repository_audit import (
    CLOSED_DISPOSITIONS,
    CLOSED_DISPOSITION_ORDER,
    CROSS_REPOSITORY_AUDIT_EVIDENCE,
    CROSS_REPOSITORY_AUDIT_SCHEMA,
    CROSS_REPOSITORY_AUDIT_VERSION,
    DEFAULT_REQUIRED_GITLINKS,
    DEFAULT_SCOPE_SPECS,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    INVENTORY_RELATIVE_PATH,
    PROPOSAL_PACKET_SCHEMA,
    TASK_ID,
    WRITE_POLICY,
    ContractCompatibilityDisposition,
    CrossRepositoryAuditError,
    CrossRepositoryContractAudit,
    CrossRepositoryContractAuditor,
    CrossRepositoryEscapeError,
    CrossRepositoryWriteError,
    ProposalPacket,
    SiblingScopeSpec,
    audit_cross_repository_contracts,
    canonical_audit_json,
    classify_compatibility,
    logical_path_under,
    normalize_relative_path,
)

ROOT = Path(__file__).resolve().parents[3]
INV = ROOT / "docs/architecture/architecture_refactorer_inventory"
AUDIT_PATH = INV / "cross_repository_contract_audit.json"
BOOTSTRAP_PATH = INV / "cross_repository_contracts.bootstrap.json"
SEALED_PATH = INV / "sealed_current_tree_baseline.json"


def _canonical(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _load_audit() -> dict:
    raw = AUDIT_PATH.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert raw == _canonical(payload)
    return payload


def _spec(
    *,
    repository: str,
    gitlink_path: str,
    pin: str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    published_paths: tuple[str, ...] = ("published.py",),
    local_paths: tuple[str, ...] = ("local.py",),
    published_authority: bool = True,
    local_authority: bool = False,
    consumption_kind: str = "direct_import",
    adapter_tokens: tuple[str, ...] = (),
    published_schema_tokens: tuple[str, ...] = ("published/schema@1",),
    local_schema_tokens: tuple[str, ...] = ("published/schema@1",),
    published_version: str = "1",
    local_version: str = "1",
    concern: str = "fixture concern",
) -> SiblingScopeSpec:
    return SiblingScopeSpec(
        repository=repository,
        gitlink_path=gitlink_path,
        required_pin=pin,
        published_concern=concern,
        published_paths=published_paths,
        local_consumer_paths=local_paths,
        published_authority_claim=published_authority,
        local_authority_claim=local_authority,
        consumption_kind=consumption_kind,
        adapter_tokens=adapter_tokens,
        published_schema_tokens=published_schema_tokens,
        local_schema_tokens=local_schema_tokens,
        published_version_token=published_version,
        local_version_token=local_version,
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_closed_dispositions() -> None:
    bootstrap = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    assert list(CLOSED_DISPOSITION_ORDER) == bootstrap["closed_dispositions"]
    assert CLOSED_DISPOSITIONS == set(bootstrap["closed_dispositions"])
    assert {item.value for item in ContractCompatibilityDisposition} == CLOSED_DISPOSITIONS
    with pytest.raises(ValueError):
        ContractCompatibilityDisposition("maybe")
    with pytest.raises(CrossRepositoryAuditError, match="cannot permit sibling writes"):
        ProposalPacket.from_mapping(
            {
                "disposition": "compatible",
                "gitlink_path": "ipfs_datasets_py",
                "local_adapter_alternative": "adapt locally",
                "packet_id": "p",
                "published_concern": "identity",
                "requested_change": "none",
                "schema": PROPOSAL_PACKET_SCHEMA,
                "sibling_write_permitted": True,
                "target_repository": "ipfs_datasets_py",
            }
        )


def test_three_sibling_scopes() -> None:
    payload = _load_audit()
    report = CrossRepositoryContractAudit.from_mapping(payload)
    assert payload["schema"] == CROSS_REPOSITORY_AUDIT_SCHEMA
    assert payload["schema"].endswith("cross-repository-contract-audit@1")
    assert payload["version"] == CROSS_REPOSITORY_AUDIT_VERSION
    assert payload["evidence"] == CROSS_REPOSITORY_AUDIT_EVIDENCE
    assert payload["task_id"] == TASK_ID
    assert payload["authority"] is False
    assert payload["write_policy"] == WRITE_POLICY
    assert payload["effect_class"] == EFFECT_CLASS
    assert payload["extractor_identity"] == EXTRACTOR_IDENTITY
    assert payload["closed_dispositions"] == list(CLOSED_DISPOSITION_ORDER)
    assert len(payload["scopes"]) == 3
    assert len(DEFAULT_SCOPE_SPECS) == 3
    repositories = [item["repository"] for item in payload["scopes"]]
    assert repositories == ["ipfs_datasets_py", "ipfs_kit_py", "MCP++"]
    gitlink_paths = [item["gitlink_path"] for item in payload["scopes"]]
    assert gitlink_paths == [
        "ipfs_datasets_py",
        "ipfs_kit_py",
        "ipfs_accelerate_py/mcplusplus",
    ]
    concerns = {item["repository"]: item["published_concern"] for item in payload["scopes"]}
    assert concerns["ipfs_datasets_py"] == "semantic and content identities"
    assert concerns["ipfs_kit_py"] == "storage and proof-seal authority"
    assert concerns["MCP++"] == "wire and profile schemas"
    for scope in payload["scopes"]:
        assert scope["disposition"] in CLOSED_DISPOSITIONS
        assert isinstance(scope["unavailable"], bool)
        if scope["unavailable"]:
            assert scope["disposition"] == "unavailable"
        assert scope["published_contracts"]
        assert all(contract["present"] for contract in scope["published_contracts"])
        assert all(set(contract) == {"path", "present"} for contract in scope["published_contracts"])
        assert all((ROOT / contract["path"]).is_file() for contract in scope["published_contracts"])
        assert all((ROOT / contract["path"]).is_file() for contract in scope["local_consumers"])
    live = audit_cross_repository_contracts(ROOT)
    assert live.to_dict() == payload
    assert CrossRepositoryContractAudit.from_mapping(live.to_dict()) == live
    assert report.to_dict() == payload


def test_gitlink_source_identity() -> None:
    payload = _load_audit()
    sealed = json.loads(SEALED_PATH.read_text(encoding="utf-8"))
    bootstrap = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    assert payload["required_gitlinks"] == DEFAULT_REQUIRED_GITLINKS
    for target in bootstrap["targets"]:
        gitlink_path = {
            "ipfs_datasets_py": "ipfs_datasets_py",
            "ipfs_kit_py": "ipfs_kit_py",
            "MCP++": "ipfs_accelerate_py/mcplusplus",
        }[target["repository"]]
        assert payload["required_gitlinks"][gitlink_path] == target["gitlink"]
        sealed_entry = sealed["gitlinks"][gitlink_path]
        assert sealed_entry["gitlink"] == target["gitlink"]
        assert sealed_entry["required_pin"] is True
    by_path = {item["gitlink_path"]: item for item in payload["scopes"]}
    for path, pin in DEFAULT_REQUIRED_GITLINKS.items():
        scope = by_path[path]
        assert scope["required_pin"] == pin
        assert scope["observed_gitlink"] == pin
        assert scope["checkout_head"] == pin
        assert scope["checkout_matches_pin"] is True
        assert scope["unavailable"] is False
        for contract in scope["published_contracts"]:
            assert contract["present"] is True
            path = ROOT / contract["path"]
            assert path.is_file()
            assert not path.is_symlink()


def test_live_dispositions_and_proposal_packet() -> None:
    payload = _load_audit()
    by_repo = {item["repository"]: item for item in payload["scopes"]}
    assert by_repo["ipfs_datasets_py"]["disposition"] == "duplicate_authority"
    assert by_repo["ipfs_kit_py"]["disposition"] == "compatible"
    assert by_repo["MCP++"]["disposition"] == "adapter_required"
    assert payload["proposal_packets"]
    packet = payload["proposal_packets"][0]
    assert packet["schema"] == PROPOSAL_PACKET_SCHEMA
    assert packet["sibling_write_permitted"] is False
    assert packet["target_repository"] == "ipfs_datasets_py"
    assert packet["disposition"] == "duplicate_authority"
    assert "Do not write the sibling gitlink" in packet["requested_change"]
    rebuilt = ProposalPacket.from_mapping(packet)
    assert rebuilt.to_dict() == packet
    datasets = by_repo["ipfs_datasets_py"]["comparison"]
    assert datasets["local_authority_claim"] is True
    assert datasets["published_authority_claim"] is True
    assert datasets["adapter_bound"] is False
    kit = by_repo["ipfs_kit_py"]["comparison"]
    assert kit["adapter_bound"] is True
    assert kit["consumption_kind"] == "direct_import"
    mcp = by_repo["MCP++"]["comparison"]
    assert mcp["adapter_bound"] is True
    assert mcp["consumption_kind"] == "runtime_adapter"
    assert "mcp++/execution/envelope@1" in mcp["shared_schema_tokens"]
    assert kit["shared_schema_tokens"] == ["ArtifactKind", "ProofSealStore"]
    assert datasets["shared_schema_tokens"] == []


def test_classify_closed_vocabulary_and_unavailable_stays_unavailable() -> None:
    unavailable = classify_compatibility(
        published_present=False,
        gitlink_available=False,
        published_version="1",
        local_version="1",
        published_markers=("a@1",),
        local_markers=("a@1",),
        published_authority_claim=True,
        local_authority_claim=True,
        adapter_bound=True,
        consumption_kind="direct_import",
    )
    assert unavailable is ContractCompatibilityDisposition.UNAVAILABLE
    still_unavailable = classify_compatibility(
        published_present=False,
        gitlink_available=True,
        published_version="1",
        local_version="1",
        published_markers=("a@1",),
        local_markers=("a@1",),
        published_authority_claim=False,
        local_authority_claim=False,
        adapter_bound=True,
        consumption_kind="direct_import",
    )
    assert still_unavailable is ContractCompatibilityDisposition.UNAVAILABLE
    assert (
        classify_compatibility(
            published_present=True,
            gitlink_available=True,
            published_version="2.0.0",
            local_version="1.4.0",
            published_markers=("a@2",),
            local_markers=("a@1",),
            published_authority_claim=True,
            local_authority_claim=False,
            adapter_bound=True,
            consumption_kind="direct_import",
        )
        is ContractCompatibilityDisposition.VERSION_INCOMPATIBLE
    )
    assert (
        classify_compatibility(
            published_present=True,
            gitlink_available=True,
            published_version="1",
            local_version="1",
            published_markers=("datasets-id@1",),
            local_markers=("accelerate-id@1",),
            published_authority_claim=True,
            local_authority_claim=True,
            adapter_bound=False,
            consumption_kind="parallel_authority",
        )
        is ContractCompatibilityDisposition.DUPLICATE_AUTHORITY
    )
    assert (
        classify_compatibility(
            published_present=True,
            gitlink_available=True,
            published_version="1",
            local_version="1",
            published_markers=("wire@1",),
            local_markers=("wire-local@1",),
            published_authority_claim=True,
            local_authority_claim=False,
            adapter_bound=True,
            consumption_kind="runtime_adapter",
        )
        is ContractCompatibilityDisposition.SCHEMA_DRIFT
    )
    assert (
        classify_compatibility(
            published_present=True,
            gitlink_available=True,
            published_version="1",
            local_version="1",
            published_markers=("mcp++/execution/envelope@1",),
            local_markers=("mcp++/execution/envelope@1",),
            published_authority_claim=True,
            local_authority_claim=False,
            adapter_bound=True,
            consumption_kind="runtime_adapter",
        )
        is ContractCompatibilityDisposition.ADAPTER_REQUIRED
    )
    assert (
        classify_compatibility(
            published_present=True,
            gitlink_available=True,
            published_version="1",
            local_version="1",
            published_markers=("ipfs_kit_py/proof_seal_store",),
            local_markers=("ipfs_kit_py/proof_seal_store",),
            published_authority_claim=True,
            local_authority_claim=False,
            adapter_bound=True,
            consumption_kind="direct_import",
        )
        is ContractCompatibilityDisposition.COMPATIBLE
    )


def test_unknown_fields_rejected() -> None:
    payload = _load_audit()
    with pytest.raises(CrossRepositoryAuditError, match="unknown cross-repository audit field"):
        CrossRepositoryContractAudit.from_mapping({**payload, "hidden": True})
    packet = payload["proposal_packets"][0]
    with pytest.raises(CrossRepositoryAuditError, match="unknown cross-repository audit field"):
        ProposalPacket.from_mapping({**packet, "extra": 1})
    with pytest.raises(CrossRepositoryAuditError, match="missing cross-repository audit field"):
        CrossRepositoryContractAudit.from_mapping(
            {key: value for key, value in payload.items() if key != "scopes"}
        )


def test_read_only_write_rejected_before_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    auditor = CrossRepositoryContractAuditor(ROOT)
    opens: list[tuple[tuple, dict]] = []
    real_open = open

    def tracking_open(*args, **kwargs):
        opens.append((args, kwargs))
        return real_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    victim = "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/content.py"
    original = (ROOT / victim).read_bytes()
    with pytest.raises(CrossRepositoryWriteError, match="before I/O"):
        auditor.refuse_sibling_write(victim)
    with pytest.raises(CrossRepositoryWriteError, match="before I/O"):
        auditor.write_published(victim, b"mutated")
    write_opens = []
    for args, kwargs in opens:
        mode = kwargs.get("mode")
        if mode is None and len(args) > 1:
            mode = args[1]
        if isinstance(mode, str) and any(flag in mode for flag in "wxa+"):
            write_opens.append((args, kwargs))
    assert write_opens == []
    assert (ROOT / victim).read_bytes() == original
    with pytest.raises(CrossRepositoryEscapeError, match="repository-relative"):
        auditor.refuse_sibling_write("../escape.py")
    with pytest.raises(CrossRepositoryEscapeError, match="repository-relative"):
        normalize_relative_path("/abs/path")
    assert logical_path_under(
        "ipfs_kit_py/ipfs_kit_py/proof_seal_store/contracts.py", "ipfs_kit_py"
    )
    assert not logical_path_under(
        "ipfs_accelerate_py/mcp_server/mcplusplus/envelope.py",
        "ipfs_accelerate_py/mcplusplus",
    )


def test_symlink_and_submodule_escape_rejected_before_io(tmp_path: Path) -> None:
    datasets = tmp_path / "ipfs_datasets_py"
    kit = tmp_path / "ipfs_kit_py"
    mcp = tmp_path / "ipfs_accelerate_py" / "mcplusplus"
    datasets.mkdir()
    kit.mkdir()
    mcp.mkdir(parents=True)
    _write(datasets / "published.py", 'SCHEMA = "published/schema@1"\n')
    _write(kit / "published.py", 'SCHEMA = "kit/schema@1"\nCONTRACT_VERSION = 1\n')
    _write(mcp / "published.py", 'SCHEMA = "mcp++/wire@1"\n')
    _write(
        tmp_path / "local_datasets.py",
        'SCHEMA = "accelerate/schema@1"\n',
    )
    _write(
        tmp_path / "local_kit.py",
        'from ipfs_kit_py.proof_seal_store.contracts import ArtifactKind\nSCHEMA = "kit/schema@1"\n',
    )
    _write(
        tmp_path / "local_mcp.py",
        'INTERFACE = "RuntimeEnvelopeAdapter@1"\nSCHEMA = "mcp++/wire@1"\n',
    )
    outside = tmp_path / "outside-secret.py"
    outside.write_text("secret", encoding="utf-8")
    (datasets / "link.py").symlink_to(outside)
    (kit / ".git").write_text("gitdir: /tmp/fake-kit.git\n", encoding="utf-8")
    (tmp_path / "owned-link.py").symlink_to(kit / "published.py")
    specs = (
        _spec(
            repository="datasets",
            gitlink_path="ipfs_datasets_py",
            local_paths=("local_datasets.py",),
            consumption_kind="parallel_authority",
            local_authority=True,
            published_schema_tokens=("published/schema@1",),
            local_schema_tokens=("accelerate/schema@1",),
        ),
        _spec(
            repository="kit",
            gitlink_path="ipfs_kit_py",
            local_paths=("local_kit.py",),
            consumption_kind="direct_import",
            adapter_tokens=("from ipfs_kit_py.proof_seal_store.contracts import",),
            published_schema_tokens=("kit/schema@1",),
            local_schema_tokens=("kit/schema@1",),
        ),
        _spec(
            repository="mcp",
            gitlink_path="ipfs_accelerate_py/mcplusplus",
            local_paths=("local_mcp.py",),
            consumption_kind="runtime_adapter",
            adapter_tokens=("RuntimeEnvelopeAdapter@1",),
            published_schema_tokens=("mcp++/wire@1",),
            local_schema_tokens=("mcp++/wire@1",),
        ),
    )
    auditor = CrossRepositoryContractAuditor(tmp_path, scopes=specs, require_git=False)
    with pytest.raises(CrossRepositoryEscapeError, match="symlink escape"):
        auditor.read_published("ipfs_datasets_py/link.py")
    with pytest.raises(CrossRepositoryWriteError, match="submodule escape|sibling write"):
        auditor.write_published("ipfs_kit_py/published.py", b"nope")
    with pytest.raises(CrossRepositoryEscapeError, match="symlink escape"):
        auditor.write_published("owned-link.py", b"nope")
    kit_original = (kit / "published.py").read_bytes()
    datasets_original = (datasets / "published.py").read_bytes()
    assert kit_original == b'SCHEMA = "kit/schema@1"\nCONTRACT_VERSION = 1\n'
    assert datasets_original == b'SCHEMA = "published/schema@1"\n'


def test_unavailable_fixture_stays_unavailable(tmp_path: Path) -> None:
    kit = tmp_path / "ipfs_kit_py"
    mcp = tmp_path / "ipfs_accelerate_py" / "mcplusplus"
    kit.mkdir()
    mcp.mkdir(parents=True)
    _write(kit / "published.py", 'SCHEMA = "kit/schema@1"\n')
    _write(mcp / "published.py", 'SCHEMA = "mcp++/wire@1"\n')
    _write(tmp_path / "local_kit.py", 'from ipfs_kit_py.proof_seal_store.contracts import ArtifactKind\nSCHEMA = "kit/schema@1"\n')
    _write(tmp_path / "local_mcp.py", 'INTERFACE = "RuntimeEnvelopeAdapter@1"\nSCHEMA = "mcp++/wire@1"\n')
    _write(tmp_path / "local_missing.py", 'SCHEMA = "published/schema@1"\n')
    specs = (
        _spec(
            repository="missing-datasets",
            gitlink_path="ipfs_datasets_py",
            local_paths=("local_missing.py",),
            consumption_kind="direct_import",
        ),
        _spec(
            repository="kit",
            gitlink_path="ipfs_kit_py",
            local_paths=("local_kit.py",),
            consumption_kind="direct_import",
            adapter_tokens=("from ipfs_kit_py.proof_seal_store.contracts import",),
            published_schema_tokens=("kit/schema@1",),
            local_schema_tokens=("kit/schema@1",),
        ),
        _spec(
            repository="mcp",
            gitlink_path="ipfs_accelerate_py/mcplusplus",
            local_paths=("local_mcp.py",),
            consumption_kind="runtime_adapter",
            adapter_tokens=("RuntimeEnvelopeAdapter@1",),
            published_schema_tokens=("mcp++/wire@1",),
            local_schema_tokens=("mcp++/wire@1",),
        ),
    )
    report = CrossRepositoryContractAuditor(tmp_path, scopes=specs, require_git=False).audit()
    by_repo = {item["repository"]: item for item in report.scopes}
    assert by_repo["missing-datasets"]["disposition"] == "unavailable"
    assert by_repo["missing-datasets"]["unavailable"] is True
    payload = report.to_dict()
    rebuilt = CrossRepositoryContractAudit.from_mapping(payload)
    assert rebuilt.scopes[0]["disposition"] == "unavailable"
    with pytest.raises(CrossRepositoryAuditError, match="unavailable stays unavailable"):
        bad = dict(payload["scopes"][0])
        bad["disposition"] = "compatible"
        CrossRepositoryContractAudit.from_mapping({**payload, "scopes": [bad, *payload["scopes"][1:]]})


def test_proposal_packet_does_not_write_sibling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    datasets = tmp_path / "ipfs_datasets_py"
    kit = tmp_path / "ipfs_kit_py"
    mcp = tmp_path / "ipfs_accelerate_py" / "mcplusplus"
    datasets.mkdir()
    kit.mkdir()
    mcp.mkdir(parents=True)
    _write(datasets / "published.py", 'SCHEMA = "datasets/id@1"\n')
    _write(kit / "published.py", 'SCHEMA = "kit/schema@1"\n')
    _write(mcp / "published.py", 'SCHEMA = "mcp++/wire@1"\n')
    _write(tmp_path / "local_datasets.py", 'SCHEMA = "accelerate/id@1"\n')
    _write(tmp_path / "local_kit.py", 'from ipfs_kit_py.proof_seal_store.contracts import ArtifactKind\nSCHEMA = "kit/schema@1"\n')
    _write(tmp_path / "local_mcp.py", 'INTERFACE = "RuntimeEnvelopeAdapter@1"\nSCHEMA = "mcp++/wire@1"\n')
    specs = (
        _spec(
            repository="datasets",
            gitlink_path="ipfs_datasets_py",
            local_paths=("local_datasets.py",),
            consumption_kind="parallel_authority",
            local_authority=True,
            published_schema_tokens=("datasets/id@1",),
            local_schema_tokens=("accelerate/id@1",),
        ),
        _spec(
            repository="kit",
            gitlink_path="ipfs_kit_py",
            local_paths=("local_kit.py",),
            consumption_kind="direct_import",
            adapter_tokens=("from ipfs_kit_py.proof_seal_store.contracts import",),
            published_schema_tokens=("kit/schema@1",),
            local_schema_tokens=("kit/schema@1",),
        ),
        _spec(
            repository="mcp",
            gitlink_path="ipfs_accelerate_py/mcplusplus",
            local_paths=("local_mcp.py",),
            consumption_kind="runtime_adapter",
            adapter_tokens=("RuntimeEnvelopeAdapter@1",),
            published_schema_tokens=("mcp++/wire@1",),
            local_schema_tokens=("mcp++/wire@1",),
        ),
    )
    auditor = CrossRepositoryContractAuditor(tmp_path, scopes=specs, require_git=False)
    opens: list[object] = []
    real_open = open

    def tracking_open(*args, **kwargs):
        opens.append((args, kwargs))
        return real_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    original = (datasets / "published.py").read_bytes()
    packet = auditor.propose_shared_change(
        packet_id="pcar-025-fixture",
        target_repository="datasets",
        gitlink_path="ipfs_datasets_py",
        published_concern="fixture concern",
        disposition="duplicate_authority",
        requested_change="Do not write the sibling gitlink.",
        local_adapter_alternative="Adapt locally.",
    )
    assert packet.sibling_write_permitted is False
    assert packet.target_repository == "datasets"
    report = auditor.audit()
    assert report.proposal_packets
    assert report.proposal_packets[0].sibling_write_permitted is False
    assert (datasets / "published.py").read_bytes() == original
    write_opens = []
    for args, kwargs in opens:
        mode = kwargs.get("mode")
        if mode is None and len(args) > 1:
            mode = args[1]
        if isinstance(mode, str) and any(flag in mode for flag in "wxa+"):
            write_opens.append((args, kwargs))
    assert write_opens == []
    with pytest.raises(CrossRepositoryWriteError):
        auditor.write_inventory("ipfs_datasets_py/stolen.json")
    assert not (datasets / "stolen.json").exists()


def test_inventory_path_is_the_only_local_write(tmp_path: Path) -> None:
    for name in ("ipfs_datasets_py", "ipfs_kit_py"):
        (tmp_path / name).mkdir()
        _write(tmp_path / name / "published.py", 'SCHEMA = "x@1"\n')
    mcp = tmp_path / "ipfs_accelerate_py" / "mcplusplus"
    mcp.mkdir(parents=True)
    _write(mcp / "published.py", 'SCHEMA = "x@1"\n')
    _write(tmp_path / "local.py", 'SCHEMA = "x@1"\nfrom ipfs_kit_py.proof_seal_store.contracts import ArtifactKind\nINTERFACE = "RuntimeEnvelopeAdapter@1"\n')
    specs = (
        _spec(repository="a", gitlink_path="ipfs_datasets_py", local_paths=("local.py",), consumption_kind="parallel_authority", local_authority=True),
        _spec(
            repository="b",
            gitlink_path="ipfs_kit_py",
            local_paths=("local.py",),
            adapter_tokens=("from ipfs_kit_py.proof_seal_store.contracts import",),
            published_schema_tokens=("x@1",),
            local_schema_tokens=("x@1",),
        ),
        _spec(
            repository="c",
            gitlink_path="ipfs_accelerate_py/mcplusplus",
            local_paths=("local.py",),
            consumption_kind="runtime_adapter",
            adapter_tokens=("RuntimeEnvelopeAdapter@1",),
            published_schema_tokens=("x@1",),
            local_schema_tokens=("x@1",),
        ),
    )
    auditor = CrossRepositoryContractAuditor(tmp_path, scopes=specs, require_git=False)
    written = auditor.write_inventory()
    assert written.as_posix().endswith(INVENTORY_RELATIVE_PATH)
    raw = written.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert raw == canonical_audit_json(payload)
    assert payload["write_policy"] == "deny"
    with pytest.raises(CrossRepositoryWriteError):
        auditor.write_inventory("docs/architecture/architecture_refactorer_inventory/other.json")
