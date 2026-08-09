"""DCR-024: classify and deduplicate deterministic repair findings.

Acceptance:
* Duplicate dag.put/get-style tasks collapse only when semantic keys match.
* expected_only, missing, ambiguous, and unobserved remain nonpassing.
* Independent protocol/schema/authority/liveness/identity/mediation/
  implementation defects are preserved even when names coincide.
* Earliest broken edge is selected along the mandatory consumer path.
* Findings CID reconstructs from canonical bytes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph import (
    ContractAuthority,
    ConsumerPathInput,
    StageEndpoint,
    SourceSpan,
    build_mcp_contract_graph,
    canonical_graph_cid,
    materialize_mcp_contract_graph,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_identity import (
    ContractDirection,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_mismatch import (
    CONTRACT_MISMATCH_INTERFACE,
    CONTRACT_MISMATCH_SCHEMA,
    CONTRACT_VERSION,
    DCR_ARTIFACT_PATH,
    DCR_TASK_ID,
    MISMATCH_EVIDENCE_TERM,
    MISMATCH_FINDINGS_INTERFACE,
    MISMATCH_FINDINGS_SCHEMA,
    NONPASSING_MISMATCH_CLASSES,
    REPAIR_FINDING_KEY_INTERFACE,
    REPAIR_FINDING_KEY_SCHEMA,
    ContractMismatch,
    McpContractMismatchError,
    McpContractMismatchFindings,
    MismatchClass,
    RepairFindingKey,
    build_mismatch_findings,
    canonical_mismatch_cid,
    classify_and_deduplicate,
    classify_mismatches,
    deduplicate_findings,
    earliest_broken_edge,
    ensure_mcp_contract_mismatch_findings_artifact,
    load_mcp_contract_mismatch_findings,
    materialize_mcp_contract_mismatch_findings,
    write_mcp_contract_mismatch_findings,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer import (
    LIVE_CONTRACT_TRANSCRIPT_SCHEMA,
    LIVE_OBSERVATION_EVIDENCE_TERM,
    load_mcp_live_transcript,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


_REPO_ROOT = _repo_root()
_ARTIFACT = _REPO_ROOT / DCR_ARTIFACT_PATH

# Prefer committed artifact; materialize when missing so validation can proceed.
ensure_mcp_contract_mismatch_findings_artifact(repo_root=_REPO_ROOT, force=False)


def _endpoint(
    stage: str,
    *,
    key: str | None = None,
    authority: ContractAuthority = ContractAuthority.REVIEWED_DECLARATION,
    owning_root: str = "swissknife",
    identity_cid: str = "",
    **payload: object,
) -> StageEndpoint:
    return StageEndpoint(
        stage=stage,
        stable_key=key or f"{stage}:fixture",
        label=f"{stage}-label",
        authority=authority,
        owning_root=owning_root,
        payload=payload,
        source_refs=(f"src/{stage}.ts",),
        span=SourceSpan(path=f"src/{stage}.ts", root_id=owning_root),
        identity_cid=identity_cid,
    )


def _full_endpoints(
    *,
    prefix: str = "fixture",
    provider_root: str = "external/ipfs_accelerate",
) -> tuple[StageEndpoint, ...]:
    runtime_cid = canonical_graph_cid({"fixture": prefix, "role": "runtime"})
    stages = []
    for stage in (
        "ui_action",
        "descriptor",
        "orb_idl",
        "mcp_method_schema",
        "mediator",
        "route",
        "dispatcher",
        "handler",
        "effect",
        "receipt",
        "runtime_identity",
    ):
        if stage in {
            "dispatcher",
            "handler",
            "effect",
            "receipt",
            "runtime_identity",
        }:
            authority = ContractAuthority.SOURCE_OBSERVATION
            root = provider_root
        elif stage == "mediator":
            authority = ContractAuthority.POLICY
            root = "swissknife"
        elif stage == "mcp_method_schema":
            authority = ContractAuthority.REVIEWED_DECLARATION
            root = "Mcp-Plus-Plus"
        else:
            authority = ContractAuthority.REVIEWED_DECLARATION
            root = "swissknife"
        extra: dict = {}
        if stage == "runtime_identity":
            extra["identity_cid"] = runtime_cid
        stages.append(
            _endpoint(
                stage,
                key=f"{stage}:{prefix}",
                authority=authority,
                owning_root=root,
                **extra,
            )
        )
    return tuple(stages)


def _consumer(
    consumer_id: str = "swissknife/ipfs_accelerate_py/echo",
    *,
    package: str = "ipfs_accelerate_py",
    operation: str = "tools.call.echo",
    endpoints: tuple[StageEndpoint, ...] | None = None,
    profile: str = "mcp++/default",
    transport: str = "stdio",
    aliases: tuple[str, ...] = ("echo",),
) -> ConsumerPathInput:
    return ConsumerPathInput(
        consumer_id=consumer_id,
        package=package,
        operation=operation,
        owning_root="swissknife",
        transport=transport,
        profile=profile,
        aliases=aliases,
        declaration={
            "method": "tools/call",
            "tool": aliases[0] if aliases else "echo",
            "schema_root": f"schemas/{aliases[0] if aliases else 'echo'}.json",
            "input_schema": {"type": "object"},
        },
        endpoints=endpoints if endpoints is not None else _full_endpoints(),
    )


def _minimal_transcript(
    *,
    tools: dict[str, list[str]] | None = None,
    passed: bool = True,
) -> dict:
    tools = tools or {
        "accelerate": ["model_catalog_health"],
        "datasets": ["logic_health"],
        "kit": ["iroh_diagnostics"],
    }
    exchanges = []
    for role, listed in sorted(tools.items()):
        package = {
            "accelerate": "ipfs_accelerate_py",
            "datasets": "ipfs_datasets_py",
            "kit": "ipfs_kit_py",
        }[role]
        for kind in ("initialize", "tools/list", "tools/call"):
            details: dict = {}
            if kind == "tools/list":
                details = {"tool_count": len(listed), "tools": list(listed)}
            exchanges.append(
                {
                    "role": role,
                    "package": package,
                    "kind": kind,
                    "method": kind if kind != "tools/call" else "tools/call",
                    "terminal_state": "passed",
                    "details": details,
                    "jsonrpc_version": "2.0",
                    "mediated": True,
                    "model_calls": 0,
                }
            )
    return {
        "schema": LIVE_CONTRACT_TRANSCRIPT_SCHEMA,
        "interface": "LiveContractTranscript@1",
        "evidence_term": LIVE_OBSERVATION_EVIDENCE_TERM,
        "service_id": "deterministic-contract-repair-mcp-runtime-v1",
        "roles_observed": sorted(tools),
        "passed": passed,
        "model_calls": 0,
        "exchanges": exchanges,
        "process_witness": {"witness_cid": "baguqeeratestwitness000000000000000000000000000000000000000"},
    }


# ---------------------------------------------------------------------------
# Interfaces / symbols
# ---------------------------------------------------------------------------


def test_interfaces_and_symbols() -> None:
    assert CONTRACT_MISMATCH_INTERFACE == "ContractMismatch@1"
    assert REPAIR_FINDING_KEY_INTERFACE == "RepairFindingKey@1"
    assert MISMATCH_FINDINGS_INTERFACE == "McpContractMismatchFindings@1"
    assert MISMATCH_EVIDENCE_TERM == "dcr/mismatch@1"
    assert CONTRACT_VERSION == 1
    assert DCR_TASK_ID == "DCR-024"
    assert callable(deduplicate_findings)
    assert callable(classify_mismatches)
    assert set(NONPASSING_MISMATCH_CLASSES) == {
        "expected_only",
        "missing",
        "ambiguous",
        "unobserved",
    }


def test_materialized_artifact_verifies_cid() -> None:
    findings = materialize_mcp_contract_mismatch_findings(repo_root=_REPO_ROOT)
    payload = findings.to_dict()
    assert payload["schema"] == MISMATCH_FINDINGS_SCHEMA
    assert payload["interface"] == MISMATCH_FINDINGS_INTERFACE
    assert payload["evidence_term"] == MISMATCH_EVIDENCE_TERM
    assert payload["model_calls"] == 0
    assert findings.verifies_cid() is True
    assert findings.findings_cid == canonical_mismatch_cid(findings._root_payload())
    assert findings.findings_cid == content_identity(findings._root_payload())
    assert payload["canonical_digest"].startswith("sha256:")
    # Digest is never accepted as findings CID.
    assert findings.findings_cid != payload["canonical_digest"]


def test_committed_artifact_round_trip() -> None:
    ensure_mcp_contract_mismatch_findings_artifact(repo_root=_REPO_ROOT, force=True)
    loaded = load_mcp_contract_mismatch_findings(repo_root=_REPO_ROOT)
    assert loaded.verifies_cid() is True
    assert loaded.schema == MISMATCH_FINDINGS_SCHEMA
    assert _ARTIFACT.is_file()
    raw = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert raw["findings_cid"] == loaded.findings_cid


# ---------------------------------------------------------------------------
# Graph blocker classification
# ---------------------------------------------------------------------------


def test_expected_only_and_missing_are_nonpassing() -> None:
    endpoints = (
        _endpoint("ui_action", key="ui:eo"),
        _endpoint("descriptor", key="desc:eo"),
        _endpoint("orb_idl", key="orb:eo"),
        _endpoint(
            "mcp_method_schema",
            key="method:eo",
            owning_root="Mcp-Plus-Plus",
        ),
        _endpoint(
            "mediator",
            key="med:eo",
            authority=ContractAuthority.POLICY,
        ),
        _endpoint("route", key="route:eo"),
    )
    graph = build_mcp_contract_graph(
        snapshot_id="snap:expected-only",
        consumers=(_consumer("expected:only", endpoints=endpoints),),
    )
    findings = classify_and_deduplicate(graph, _minimal_transcript())
    classes = {item.mismatch_class for item in findings}
    assert MismatchClass.EXPECTED_ONLY in classes or MismatchClass.MISSING in classes
    for item in findings:
        if item.mismatch_class.value in NONPASSING_MISMATCH_CLASSES:
            assert item.nonpassing is True
            assert item.mismatch_class.value in {
                "expected_only",
                "missing",
                "ambiguous",
                "unobserved",
            }


def test_ambiguous_handler_is_nonpassing_ambiguous() -> None:
    endpoints = list(_full_endpoints(prefix="amb"))
    endpoints.append(
        _endpoint(
            "handler",
            key="handler:amb:alt",
            authority=ContractAuthority.SOURCE_OBSERVATION,
            owning_root="external/ipfs_accelerate",
        )
    )
    graph = build_mcp_contract_graph(
        snapshot_id="snap:ambiguous",
        consumers=(_consumer("amb:consumer", endpoints=tuple(endpoints)),),
    )
    findings = classify_and_deduplicate(graph)
    amb = [
        item
        for item in findings
        if item.edge_kind == "dispatcher_to_handler"
        and item.mismatch_class is MismatchClass.AMBIGUOUS
    ]
    assert amb
    assert all(item.nonpassing for item in amb)


def test_earliest_broken_edge_is_first_mandatory_blocker() -> None:
    endpoints = (
        _endpoint("ui_action", key="ui:early"),
        _endpoint("descriptor", key="desc:early"),
        _endpoint("orb_idl", key="orb:early"),
        _endpoint(
            "mcp_method_schema",
            key="method:early",
            owning_root="Mcp-Plus-Plus",
        ),
        _endpoint(
            "mediator",
            key="med:early",
            authority=ContractAuthority.POLICY,
        ),
        _endpoint("route", key="route:early"),
    )
    graph = build_mcp_contract_graph(
        snapshot_id="snap:earliest",
        consumers=(_consumer("early:consumer", endpoints=endpoints),),
    )
    blocker = earliest_broken_edge(graph, "early:consumer")
    assert blocker is not None
    # First broken mandatory link after route is route_to_dispatcher.
    assert blocker.edge_kind == "route_to_dispatcher"
    catalog = build_mismatch_findings(
        graph, _minimal_transcript(), require_shared_epoch=True
    )
    assert catalog.earliest_by_consumer["early:consumer"] == "route_to_dispatcher"


# ---------------------------------------------------------------------------
# Deduplication semantics
# ---------------------------------------------------------------------------


def test_duplicate_semantic_keys_collapse_only_when_exact() -> None:
    key_put = RepairFindingKey(
        package="ipfs_kit_py",
        operation="tools.call.dag.put",
        direction=ContractDirection.REQUEST,
        schema_root="schemas/dag.put.json",
        profile="mcp++/default",
        transport="stdio",
        mismatch_class=MismatchClass.IMPLEMENTATION,
        edge_kind="dispatcher_to_handler",
        snapshot_id="snap:dedupe",
    )
    key_get = RepairFindingKey(
        package="ipfs_kit_py",
        operation="tools.call.dag.get",
        direction=ContractDirection.REQUEST,
        schema_root="schemas/dag.get.json",
        profile="mcp++/default",
        transport="stdio",
        mismatch_class=MismatchClass.IMPLEMENTATION,
        edge_kind="dispatcher_to_handler",
        snapshot_id="snap:dedupe",
    )
    # Same semantic key twice → collapse.
    a = ContractMismatch(
        finding_key=key_put,
        mismatch_class=MismatchClass.IMPLEMENTATION,
        package="ipfs_kit_py",
        operation="tools.call.dag.put",
        direction=ContractDirection.REQUEST,
        consumer_id="swissknife/ipfs_kit_py/dag.put",
        edge_kind="dispatcher_to_handler",
        stage="handler",
        expected_edge={"edge_kind": "dispatcher_to_handler"},
        observed_edge={"resolution": "unresolved"},
        counterexample_seed={"n": 1},
        reason_code="missing_handler",
    )
    b = ContractMismatch(
        finding_key=key_put,
        mismatch_class=MismatchClass.IMPLEMENTATION,
        package="ipfs_kit_py",
        operation="tools.call.dag.put",
        direction=ContractDirection.REQUEST,
        consumer_id="swissknife/ipfs_kit_py/dag.put",
        edge_kind="dispatcher_to_handler",
        stage="handler",
        expected_edge={"edge_kind": "dispatcher_to_handler"},
        observed_edge={"resolution": "unresolved"},
        counterexample_seed={"n": 2, "richer": True},
        reason_code="missing_handler",
        blocker_id="",
    )
    # Distinct operation (dag.get) must remain separate.
    c = ContractMismatch(
        finding_key=key_get,
        mismatch_class=MismatchClass.IMPLEMENTATION,
        package="ipfs_kit_py",
        operation="tools.call.dag.get",
        direction=ContractDirection.REQUEST,
        consumer_id="swissknife/ipfs_kit_py/dag.get",
        edge_kind="dispatcher_to_handler",
        stage="handler",
        expected_edge={"edge_kind": "dispatcher_to_handler"},
        observed_edge={"resolution": "unresolved"},
        counterexample_seed={"n": 1},
        reason_code="missing_handler",
    )
    deduped = deduplicate_findings((a, b, c))
    assert len(deduped) == 2
    ops = {item.operation for item in deduped}
    assert ops == {"tools.call.dag.put", "tools.call.dag.get"}
    put = next(item for item in deduped if item.operation == "tools.call.dag.put")
    # Richer seed wins among exact key matches.
    assert put.counterexample_seed.get("richer") is True


def test_independent_classes_preserved_when_names_coincide() -> None:
    """Protocol and schema defects for the same package/operation stay distinct."""

    base = dict(
        package="ipfs_accelerate_py",
        operation="tools.call.shared.name",
        direction=ContractDirection.REQUEST,
        schema_root="schemas/shared.name.json",
        profile="mcp++/default",
        transport="http",
        edge_kind="mcp_method_schema_to_mediator",
        snapshot_id="snap:classes",
    )
    protocol_key = RepairFindingKey(
        **base, mismatch_class=MismatchClass.PROTOCOL  # type: ignore[arg-type]
    )
    schema_key = RepairFindingKey(
        **base, mismatch_class=MismatchClass.SCHEMA  # type: ignore[arg-type]
    )
    authority_key = RepairFindingKey(
        **base, mismatch_class=MismatchClass.AUTHORITY  # type: ignore[arg-type]
    )
    findings = []
    for key, cls, reason in (
        (protocol_key, MismatchClass.PROTOCOL, "protocol_defect"),
        (schema_key, MismatchClass.SCHEMA, "schema_defect"),
        (authority_key, MismatchClass.AUTHORITY, "authority_defect"),
    ):
        findings.append(
            ContractMismatch(
                finding_key=key,
                mismatch_class=cls,
                package=base["package"],
                operation=base["operation"],
                direction=ContractDirection.REQUEST,
                consumer_id="swissknife/ipfs_accelerate_py/shared.name",
                edge_kind=base["edge_kind"],
                stage="mediator",
                expected_edge={"class": cls.value},
                observed_edge={"class": cls.value},
                counterexample_seed={"reason": reason},
                reason_code=reason,
            )
        )
    deduped = deduplicate_findings(findings)
    assert len(deduped) == 3
    assert {item.mismatch_class for item in deduped} == {
        MismatchClass.PROTOCOL,
        MismatchClass.SCHEMA,
        MismatchClass.AUTHORITY,
    }


# ---------------------------------------------------------------------------
# Live unobserved
# ---------------------------------------------------------------------------


def test_unobserved_tools_are_nonpassing() -> None:
    graph = materialize_mcp_contract_graph()
    # Live catalog does not advertise the reference tools.
    transcript = _minimal_transcript(
        tools={
            "accelerate": ["model_catalog_health"],
            "datasets": ["logic_health"],
            "kit": ["iroh_diagnostics"],
        }
    )
    findings = classify_and_deduplicate(graph, transcript)
    unobserved = [
        item for item in findings if item.mismatch_class is MismatchClass.UNOBSERVED
    ]
    assert unobserved
    assert all(item.nonpassing for item in unobserved)
    # expected.only.tool must appear as expected_only and/or unobserved.
    classes_for_expected = {
        item.mismatch_class
        for item in findings
        if "expected.only" in item.operation or "expected.only" in item.consumer_id
    }
    assert classes_for_expected & {
        MismatchClass.EXPECTED_ONLY,
        MismatchClass.MISSING,
        MismatchClass.UNOBSERVED,
    }


def test_reference_catalog_has_expected_only_findings() -> None:
    findings = materialize_mcp_contract_mismatch_findings(repo_root=_REPO_ROOT)
    assert findings.model_calls == 0
    assert findings.graph_cid
    assert findings.transcript_epoch
    # Graph reference includes expected.only.tool blockers.
    expected_related = [
        item
        for item in findings.findings
        if "expected.only" in item.consumer_id or "expected.only" in item.operation
    ]
    assert expected_related
    assert all(item.nonpassing for item in expected_related)
    # Earliest edge recorded for the expected-only consumer.
    assert any(
        "expected.only" in consumer for consumer in findings.earliest_by_consumer
    )


def test_forged_findings_cid_is_rejected() -> None:
    findings = materialize_mcp_contract_mismatch_findings(repo_root=_REPO_ROOT)
    payload = findings.to_dict()
    payload["findings_cid"] = "baguqeeratampered000000000000000000000000000000000000000000"
    with pytest.raises(McpContractMismatchError, match="findings_cid"):
        McpContractMismatchFindings.from_dict(payload)


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    findings = materialize_mcp_contract_mismatch_findings(repo_root=_REPO_ROOT)
    destination = tmp_path / "mcp_contract_mismatch_findings.json"
    write_mcp_contract_mismatch_findings(destination, findings=findings)
    loaded = load_mcp_contract_mismatch_findings(destination)
    assert loaded.findings_cid == findings.findings_cid
    assert len(loaded.findings) == len(findings.findings)
    assert loaded.verifies_cid() is True


def test_repair_finding_key_id_is_content_addressed() -> None:
    key = RepairFindingKey(
        package="ipfs_kit_py",
        operation="tools.call.ipfs.add",
        direction=ContractDirection.REQUEST,
        schema_root="schemas/ipfs.add.json",
        profile="mcp++/default",
        transport="stdio",
        mismatch_class=MismatchClass.IMPLEMENTATION,
        edge_kind="handler_to_effect",
        snapshot_id="snap:key",
    )
    assert key.schema == REPAIR_FINDING_KEY_SCHEMA
    assert key.interface == REPAIR_FINDING_KEY_INTERFACE
    assert key.key_id == content_identity(key._identity_payload())
    assert key.key_id.startswith("b")


def test_mismatch_carries_evidence_subset() -> None:
    graph = materialize_mcp_contract_graph()
    transcript = load_mcp_live_transcript(repo_root=_REPO_ROOT)
    catalog = build_mismatch_findings(graph, transcript)
    assert catalog.findings
    for item in catalog.findings:
        assert item.schema == CONTRACT_MISMATCH_SCHEMA
        assert item.interface == CONTRACT_MISMATCH_INTERFACE
        assert isinstance(item.expected_edge, Mapping) or item.expected_edge is not None
        assert item.counterexample_seed is not None
        assert item.canonical_key == item.finding_key.key_id
        assert item.finding_id.startswith("b")
