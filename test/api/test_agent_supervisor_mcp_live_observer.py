"""DCR-023: live MCP initialize/list/call/profile/logic observation tests.

Acceptance:
* All three reviewed services (accelerate, datasets, kit) are observed.
* Discovery/RPC failures are typed (never empty success).
* Process-local and datasets MCP logic_tools/cec_prove results are
  canonically equivalent.
* Profiles A–F are probed; malformed/unknown calls fail closed.
* Transcript CIDs reconstruct from canonical bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer import (
    DCR_ARTIFACT_PATH,
    DCR_TASK_ID,
    JSONRPC_VERSION,
    LIVE_CONTRACT_TRANSCRIPT_INTERFACE,
    LIVE_CONTRACT_TRANSCRIPT_SCHEMA,
    LIVE_OBSERVATION_EVIDENCE_TERM,
    LOGIC_CEC_PROVE_TOOL,
    MCP_LIVE_OBSERVATION_INTERFACE,
    MCP_PLUS_PROFILES_A_F,
    SAFE_TOOLS_CALL,
    McpLiveObserver,
    McpLiveObserverError,
    ObservationKind,
    ObservationTerminalState,
    ensure_mcp_live_transcript_artifact,
    load_mcp_live_transcript,
    materialize_mcp_live_transcript,
    observation_content_cid,
    observe_mcp_live_contracts,
    write_mcp_live_transcript,
)
from ipfs_accelerate_py.agent_supervisor.analysis.runtime_service_identity import (
    REQUIRED_SERVICE_ROLES,
    is_pseudo_cid,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


_REPO_ROOT = _repo_root()
_ARTIFACT = _REPO_ROOT / DCR_ARTIFACT_PATH

# Prefer the committed artifact bytes.  Only materialize when missing so a
# provider checkout that already includes the declared output does not churn
# tree identity under fail-closed validation.  ensure also removes accidental
# undeclared one-shot helpers from earlier attempts.
ensure_mcp_live_transcript_artifact(repo_root=_REPO_ROOT, force=False)
_TRANSCRIPT = materialize_mcp_live_transcript(repo_root=_REPO_ROOT)



def test_observer_symbols_and_interfaces_exist() -> None:
    assert callable(McpLiveObserver)
    assert LIVE_CONTRACT_TRANSCRIPT_INTERFACE == "LiveContractTranscript@1"
    assert MCP_LIVE_OBSERVATION_INTERFACE == "McpLiveObservation@1"
    assert LIVE_OBSERVATION_EVIDENCE_TERM == "dcr/live-observation@1"
    assert DCR_TASK_ID == "DCR-023"
    assert set(SAFE_TOOLS_CALL) == set(REQUIRED_SERVICE_ROLES)
    assert len(MCP_PLUS_PROFILES_A_F) == 6


def test_all_three_services_are_observed() -> None:
    transcript = observe_mcp_live_contracts(repo_root=_REPO_ROOT)
    payload = transcript.to_dict()

    assert set(transcript.roles_observed) == set(REQUIRED_SERVICE_ROLES)
    roles_in_exchanges = {item.role for item in transcript.exchanges}
    assert roles_in_exchanges == set(REQUIRED_SERVICE_ROLES)
    assert payload["service_id"] == "deterministic-contract-repair-mcp-runtime-v1"
    assert payload["schema"] == LIVE_CONTRACT_TRANSCRIPT_SCHEMA
    assert payload["interface"] == LIVE_CONTRACT_TRANSCRIPT_INTERFACE
    assert payload["evidence_term"] == LIVE_OBSERVATION_EVIDENCE_TERM
    assert payload["model_calls"] == 0
    assert transcript.model_calls == 0


def test_initialize_list_call_malformed_unknown_and_profiles_captured() -> None:
    transcript = observe_mcp_live_contracts(repo_root=_REPO_ROOT)

    for role in REQUIRED_SERVICE_ROLES:
        kinds = {item.kind for item in transcript.exchanges if item.role == role}
        assert ObservationKind.INITIALIZE.value in kinds
        assert ObservationKind.TOOLS_LIST.value in kinds
        assert ObservationKind.TOOLS_CALL.value in kinds
        assert ObservationKind.MALFORMED_CALL.value in kinds
        assert ObservationKind.UNKNOWN_CALL.value in kinds
        assert ObservationKind.PROFILE_PROBE.value in kinds

        init = [
            item
            for item in transcript.exchanges
            if item.role == role and item.kind == ObservationKind.INITIALIZE.value
        ][0]
        assert init.jsonrpc_version == JSONRPC_VERSION
        assert init.terminal_state == ObservationTerminalState.PASSED.value
        assert "initialize_observed" in init.reason_codes

        malformed = [
            item
            for item in transcript.exchanges
            if item.role == role and item.kind == ObservationKind.MALFORMED_CALL.value
        ][0]
        assert malformed.terminal_state == ObservationTerminalState.REFUTED.value
        assert "malformed_params_fail_closed" in malformed.reason_codes

        unknown = [
            item
            for item in transcript.exchanges
            if item.role == role and item.kind == ObservationKind.UNKNOWN_CALL.value
        ][0]
        assert unknown.terminal_state == ObservationTerminalState.REFUTED.value
        assert "unknown_operation_fail_closed" in unknown.reason_codes

        profiles = [
            item
            for item in transcript.exchanges
            if item.role == role and item.kind == ObservationKind.PROFILE_PROBE.value
        ]
        assert len(profiles) == len(MCP_PLUS_PROFILES_A_F)
        probed = {item.details.get("profile") for item in profiles}
        assert probed == set(MCP_PLUS_PROFILES_A_F)
        assert set(transcript.profile_results[role]) == set(MCP_PLUS_PROFILES_A_F)


def test_allowlisted_safe_tools_call_zero_model_calls() -> None:
    transcript = observe_mcp_live_contracts(repo_root=_REPO_ROOT)
    for role, operation in SAFE_TOOLS_CALL.items():
        calls = [
            item
            for item in transcript.exchanges
            if item.role == role and item.kind == ObservationKind.TOOLS_CALL.value
        ]
        assert calls, f"missing safe tools/call for {role}"
        call = calls[0]
        assert call.model_calls == 0
        assert call.details.get("operation") == operation
        # Must not convert failures into empty success without reason codes.
        if call.terminal_state == ObservationTerminalState.PASSED.value:
            assert "tools_call_allowlisted" in call.reason_codes
        else:
            assert call.terminal_state in {
                ObservationTerminalState.FAILED.value,
                ObservationTerminalState.REFUTED.value,
            }
            assert call.reason_codes


def test_transport_errors_are_typed_not_empty_success() -> None:
    transcript = observe_mcp_live_contracts(repo_root=_REPO_ROOT)
    loopbacks = [
        item
        for item in transcript.exchanges
        if item.kind == ObservationKind.LOOPBACK_PROBE.value
    ]
    assert len(loopbacks) == len(REQUIRED_SERVICE_ROLES)
    for item in loopbacks:
        assert item.transport == "loopback_http"
        assert item.mediated is False
        if item.terminal_state == ObservationTerminalState.TRANSPORT_ERROR.value:
            assert "transport_error_not_empty_success" in item.reason_codes
            # Response must carry a JSON-RPC error object, not empty result.
            response_text = item.response_bytes.get("utf8") or ""
            assert "error" in response_text
            assert '"result":{}' not in response_text.replace(" ", "")
        elif item.terminal_state == ObservationTerminalState.PASSED.value:
            assert "health_is_liveness_only" in item.reason_codes
        else:
            pytest.fail(f"unexpected loopback terminal state: {item.terminal_state}")


def test_process_local_and_mcp_logic_results_are_canonically_equivalent() -> None:
    transcript = observe_mcp_live_contracts(repo_root=_REPO_ROOT)
    logic = transcript.logic_equivalence
    assert logic.get("tool") == LOGIC_CEC_PROVE_TOOL
    assert logic.get("canonically_equivalent") is True
    assert not is_pseudo_cid(logic.get("process_local_cid"))
    assert not is_pseudo_cid(logic.get("mcp_result_cid"))
    assert logic["process_local_cid"] == logic["mcp_result_cid"]

    cec = [
        item
        for item in transcript.exchanges
        if item.kind == ObservationKind.LOGIC_CEC_PROVE.value
    ]
    assert len(cec) == 2
    surfaces = {item.details.get("surface") for item in cec}
    assert surfaces == {"process_local", "mcp_tools_call"}


def test_process_witness_bound_for_each_role() -> None:
    transcript = observe_mcp_live_contracts(repo_root=_REPO_ROOT)
    witness = transcript.process_witness
    assert witness.get("passed") is True
    roles = {item["role"] for item in witness.get("roles", [])}
    assert roles == set(REQUIRED_SERVICE_ROLES)
    assert not is_pseudo_cid(witness.get("witness_cid"))
    for exchange in transcript.exchanges:
        assert exchange.process_witness_cid
        assert not is_pseudo_cid(exchange.process_witness_cid)
        assert not is_pseudo_cid(exchange.receipt_cid)
        assert not is_pseudo_cid(exchange.local_cid)
        assert not is_pseudo_cid(exchange.schema_identity)
        assert exchange.request_bytes.get("sha256", "").startswith("sha256:")
        assert exchange.response_bytes.get("sha256", "").startswith("sha256:")


def test_transcript_cid_reconstructs_from_canonical_bytes() -> None:
    transcript = observe_mcp_live_contracts(repo_root=_REPO_ROOT)
    payload = transcript.to_dict()
    assert payload["transcript_cid"] == transcript.transcript_cid
    assert not is_pseudo_cid(payload["transcript_cid"])
    # Recompute identity excluding the transcript_cid field itself.
    recomputed = observation_content_cid(
        {
            key: value
            for key, value in payload.items()
            if key not in {"transcript_cid", "policies"}
        }
    )
    # transcript_cid is derived from the identity payload (without itself/policies).
    assert transcript.transcript_cid == observation_content_cid(
        {
            "schema": transcript.schema,
            "interface": transcript.interface,
            "version": transcript.version,
            "evidence_term": transcript.evidence_term,
            "task_id": transcript.task_id,
            "passed": transcript.passed,
            "service_id": transcript.service_id,
            "roles_observed": list(transcript.roles_observed),
            "exchanges": [item.to_dict() for item in transcript.exchanges],
            "process_witness": dict(transcript.process_witness),
            "logic_equivalence": dict(transcript.logic_equivalence),
            "profile_results": dict(transcript.profile_results),
            "reason_codes": list(transcript.reason_codes),
            "model_calls": transcript.model_calls,
        }
    )
    assert recomputed  # sanity: non-empty


def test_artifact_written_and_loadable() -> None:
    # The declared artifact must already exist as a committed generated output.
    # Tests must not rewrite it (tree identity is validated fail-closed).
    assert _ARTIFACT.is_file(), (
        f"missing declared artifact {_ARTIFACT}; "
        "run write_mcp_live_transcript during implementation"
    )
    loaded = load_mcp_live_transcript(repo_root=_REPO_ROOT)
    assert loaded["schema"] == LIVE_CONTRACT_TRANSCRIPT_SCHEMA
    assert loaded["interface"] == LIVE_CONTRACT_TRANSCRIPT_INTERFACE
    assert loaded["evidence_term"] == LIVE_OBSERVATION_EVIDENCE_TERM
    assert loaded["task_id"] == DCR_TASK_ID
    assert set(loaded["roles_observed"]) == set(REQUIRED_SERVICE_ROLES)
    assert loaded["logic_equivalence"]["canonically_equivalent"] is True
    assert loaded["model_calls"] == 0
    assert loaded["policies"]["transport_error_becomes_empty_success"] is False
    assert loaded["policies"]["infer_missing_calls"] is False
    assert loaded["policies"]["local_loopback_only"] is True
    # File JSON must parse and retain exchange evidence fields.
    raw = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(raw["exchanges"], list)
    assert len(raw["exchanges"]) >= len(REQUIRED_SERVICE_ROLES) * 5
    for exchange in raw["exchanges"]:
        assert exchange["interface"] == MCP_LIVE_OBSERVATION_INTERFACE
        assert exchange["jsonrpc_version"] == JSONRPC_VERSION
        assert "request_bytes" in exchange
        assert "response_bytes" in exchange
        assert "receipt_cid" in exchange


def test_write_helper_can_emit_transcript(tmp_path: Path) -> None:
    """write_mcp_live_transcript works, but only to non-declared paths in tests."""

    out = tmp_path / "mcp-live-transcript.json"
    transcript = materialize_mcp_live_transcript(repo_root=_REPO_ROOT)
    written = write_mcp_live_transcript(out, transcript=transcript, repo_root=_REPO_ROOT)
    assert written == out
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema"] == LIVE_CONTRACT_TRANSCRIPT_SCHEMA
    assert loaded["passed"] is True


def test_transcript_passes_with_zero_model_calls() -> None:
    transcript = _TRANSCRIPT
    assert transcript.passed is True
    assert transcript.model_calls == 0
    assert "logic_cec_prove_canonically_equivalent" in transcript.reason_codes
    assert "roles_observed" in transcript.reason_codes
    assert "zero_model_calls" in transcript.reason_codes


def test_missing_transcript_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    with pytest.raises(McpLiveObserverError) as exc:
        load_mcp_live_transcript(missing, repo_root=_REPO_ROOT)
    assert exc.value.reason_code == "transcript_missing"
