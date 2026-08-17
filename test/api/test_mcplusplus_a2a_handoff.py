"""MCPP-056: A2A reference adapter and two-agent handoff tests.

Acceptance:
* Two independently instantiated agents complete a handoff.
* Cancel writes Event DAG records.
* Malformed extension fails closed.

Also covers retry, streaming, and unsupported-profile fail-closed paths
(MCPP-G100 evidence criteria / MCPP-056 effects).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.a2a_adapter import (
    ALLOWED_PROFILES,
    DEFAULT_PROFILES,
    ERR_EXTENSION_REQUIRED,
    ERR_MALFORMED_EXTENSION,
    ERR_MALFORMED_EXTENSION_URI,
    ERR_MISSING_RECEIPT_CID,
    ERR_NOT_ACTIVATED,
    ERR_PROFILE_NOT_SUBSET,
    ERR_TASK_NOT_CANCELABLE,
    ERR_UNSUPPORTED_PROFILE,
    EXTENSION_URI,
    INTERFACE,
    METADATA_KEY_PREFIX,
    SCHEMA_TERMINAL_EVIDENCE,
    TASK_ID,
    TERMINAL_TASK_STATES,
    WORKING_ALIAS,
    A2AExtensionError,
    A2ATaskAdapter,
    TaskState,
    classify_extension_uri,
    map_result_status_to_task_state,
    parse_a2a_extensions_header,
    validate_activation,
    validate_agent_extension,
    validate_profile_request,
    validate_task_metadata,
    validate_terminal_evidence,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.event_dag import EventDAGStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> A2ATaskAdapter:
    return A2ATaskAdapter()


@pytest.fixture
def two_agents(adapter: A2ATaskAdapter):
    client = adapter.create_agent(agent_id="client-a", name="Client Agent A")
    server = adapter.create_agent(agent_id="server-b", name="Server Agent B")
    return client, server


def _vector_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "mcplusplus"
        / "conformance"
        / "vectors"
        / "a2a"
    )


def _load_vector(name: str) -> Dict[str, Any]:
    path = _vector_root() / f"{name}.json"
    assert path.is_file(), f"missing vector suite: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Interface / constants
# ---------------------------------------------------------------------------


class TestA2AAdapterInterface:
    def test_interface_constants(self, adapter: A2ATaskAdapter) -> None:
        assert adapter.interface == INTERFACE == "A2ATaskAdapter@1"
        assert adapter.extension_uri == EXTENSION_URI
        assert adapter.working_alias == WORKING_ALIAS
        assert adapter.task_id == TASK_ID == "MCPP-056"
        assert EXTENSION_URI == "https://mcplusplus.io/extensions/execution/v1"
        assert WORKING_ALIAS == "io.mcplusplus.execution@1"
        assert WORKING_ALIAS != EXTENSION_URI

    def test_agents_are_independent_instances(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        assert client is not server
        assert client.agent_id != server.agent_id
        assert client.event_dag is not server.event_dag
        assert isinstance(client.event_dag, EventDAGStore)
        assert isinstance(server.event_dag, EventDAGStore)

    def test_agent_card_declares_confirmed_uri(self, two_agents) -> None:
        _client, server = two_agents
        card = server.agent_card()
        extensions = card["capabilities"]["extensions"]
        assert len(extensions) == 1
        assert extensions[0]["uri"] == EXTENSION_URI
        result = validate_agent_extension(extensions[0])
        assert result.ok, result.errors
        # Alias may appear only inside params, never as uri.
        assert extensions[0]["params"]["alias"] == WORKING_ALIAS


# ---------------------------------------------------------------------------
# Two-agent handoff
# ---------------------------------------------------------------------------


class TestTwoAgentHandoff:
    def test_two_agents_complete_handoff(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        receipt = adapter.handoff(
            client,
            server,
            text="please run repo.status on the workspace",
            method="repo.status",
            requested_profiles=["A", "B"],
        )

        assert receipt["interface"] == INTERFACE
        assert receipt["extension_uri"] == EXTENSION_URI
        assert receipt["client_agent_id"] == client.agent_id
        assert receipt["server_agent_id"] == server.agent_id
        assert receipt["client_did"] != receipt["server_did"]

        task = receipt["task"]
        assert task["id"]
        assert task["status"]["state"] == TaskState.COMPLETED.value
        assert task["extension_uri"] == EXTENSION_URI

        # Namespaced metadata carries evidence keys.
        meta = task["metadata"]
        assert f"{METADATA_KEY_PREFIX}receipt_cid" in meta
        assert f"{METADATA_KEY_PREFIX}event_cid" in meta
        assert f"{METADATA_KEY_PREFIX}envelope_cid" in meta
        assert f"{METADATA_KEY_PREFIX}output_cid" in meta

        evidence = receipt["terminal_evidence"]
        assert evidence["schema"] == SCHEMA_TERMINAL_EVIDENCE
        assert evidence["task_state"] == TaskState.COMPLETED.value
        assert evidence["portable"] is True
        assert evidence["receipt_cid"]
        assert evidence["event_cid"]
        assert evidence["envelope_cid"]
        assert validate_terminal_evidence(evidence).ok

        # Event DAG lineage is non-empty on the server agent only.
        lineage = receipt["event_lineage"]
        assert len(lineage) >= 2
        assert server.event_dag.has_event(evidence["event_cid"])
        assert client.event_dag.stats()["event_count"] == 0

        # Artifacts carry output_cid under extension namespace.
        assert task["artifacts"]
        art_meta = task["artifacts"][0]["metadata"]
        assert f"{METADATA_KEY_PREFIX}output_cid" in art_meta

    def test_handoff_uses_a2a_extensions_header_string(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        receipt = adapter.handoff(
            client,
            server,
            a2a_extensions=EXTENSION_URI,
        )
        assert receipt["task"]["status"]["state"] == TaskState.COMPLETED.value
        assert EXTENSION_URI in receipt["activated_extensions"]

    def test_handoff_rejects_missing_activation(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        with pytest.raises(A2AExtensionError) as exc:
            adapter.handoff(
                client,
                server,
                a2a_extensions=["https://example.com/extensions/geolocation/v1"],
            )
        assert exc.value.code in {ERR_NOT_ACTIVATED, ERR_MALFORMED_EXTENSION}

    def test_discover_validates_card_extension(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        _client, server = two_agents
        card = adapter.discover(server)
        assert card["name"] == server.name
        assert card["capabilities"]["extensions"][0]["uri"] == EXTENSION_URI


# ---------------------------------------------------------------------------
# Cancel → Event DAG
# ---------------------------------------------------------------------------


class TestCancelEventDag:
    def test_cancel_writes_event_dag_records(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        receipt = adapter.handoff(
            client,
            server,
            hold_open=True,
            execute=False,
        )
        task_id = receipt["task"]["id"]
        assert receipt["task"]["status"]["state"] == TaskState.WORKING.value

        # Pre-cancel: submitted + working events, no cancel.
        before = server.event_dag.stats()["event_count"]
        assert before >= 2
        assert server.cancel_events(task_id) == []

        canceled = adapter.cancel(server, task_id, reason="operator-abort")
        view = canceled["task"]
        assert view["status"]["state"] == TaskState.CANCELED.value

        cancel_events = canceled["cancel_events"]
        assert len(cancel_events) == 1
        payload = cancel_events[0]
        assert payload["kind"] == "task.canceled"
        assert payload["task_id"] == task_id
        assert payload["state"] == TaskState.CANCELED.value
        assert payload["extension_uri"] == EXTENSION_URI
        assert payload["durable_cancel_id"]
        assert payload["parents"], "cancel event must parent prior working event"

        # Event DAG grew and lineage reaches the cancel node.
        assert server.event_dag.stats()["event_count"] == before + 1
        lineage = canceled["event_lineage"]
        assert lineage[-1] == view["metadata"][f"{METADATA_KEY_PREFIX}event_cid"]
        assert server.event_dag.has_event(lineage[-1])

        # Durable cancel journal is present and content-addressed.
        durable = canceled["durable_cancels"]
        assert payload["durable_cancel_id"] in durable
        assert durable[payload["durable_cancel_id"]]["task_id"] == task_id

    def test_cancel_terminal_task_fails_closed(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        receipt = adapter.handoff(client, server)
        task_id = receipt["task"]["id"]
        with pytest.raises(A2AExtensionError) as exc:
            adapter.cancel(server, task_id)
        assert exc.value.code == ERR_TASK_NOT_CANCELABLE

    def test_cancel_event_payload_is_retrievable(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        receipt = adapter.handoff(client, server, hold_open=True)
        task_id = receipt["task"]["id"]
        adapter.cancel(server, task_id, reason="stream-cancel")
        event_cid = server.get_task(task_id).last_event_cid
        assert event_cid
        stored = server.event_dag.get_event(event_cid)
        assert stored is not None
        assert stored["kind"] == "task.canceled"
        assert stored["event_cid"] == event_cid


# ---------------------------------------------------------------------------
# Malformed extension fails closed
# ---------------------------------------------------------------------------


class TestMalformedExtensionFailClosed:
    def test_reverse_dns_alias_rejected_as_wire_uri(self) -> None:
        ok, code = classify_extension_uri(WORKING_ALIAS)
        assert ok is False
        assert code == ERR_MALFORMED_EXTENSION_URI
        result = validate_agent_extension({"uri": WORKING_ALIAS})
        assert result.ok is False
        assert result.code == ERR_MALFORMED_EXTENSION_URI

    def test_missing_uri_rejected(self) -> None:
        result = validate_agent_extension({"description": "no uri"})
        assert result.ok is False
        assert result.code == ERR_MALFORMED_EXTENSION

    def test_foreign_https_uri_rejected(self) -> None:
        result = validate_agent_extension(
            {"uri": "https://example.com/extensions/execution/v1"}
        )
        assert result.ok is False
        assert result.code == ERR_MALFORMED_EXTENSION_URI

    def test_reserved_a2a_org_prefix_rejected(self) -> None:
        result = validate_agent_extension(
            {"uri": "https://a2a-protocol.org/extensions/execution/v1"}
        )
        assert result.ok is False
        assert result.code == ERR_MALFORMED_EXTENSION_URI

    def test_future_v2_uri_rejected(self) -> None:
        result = validate_agent_extension(
            {"uri": "https://mcplusplus.io/extensions/execution/v2"}
        )
        assert result.ok is False
        assert result.code == ERR_MALFORMED_EXTENSION_URI

    def test_activation_alias_only_fails_closed(self) -> None:
        result = validate_activation(
            {
                "schema": "mcp++/a2a/activation@1",
                "a2a_extensions": [WORKING_ALIAS],
                "mcp_plus_plus_execution_activated": True,
            }
        )
        assert result.ok is False
        assert result.code == ERR_MALFORMED_EXTENSION_URI

    def test_activation_claims_active_without_uri(self) -> None:
        result = validate_activation(
            {
                "schema": "mcp++/a2a/activation@1",
                "a2a_extensions": ["https://example.com/extensions/geolocation/v1"],
                "mcp_plus_plus_execution_activated": True,
            }
        )
        assert result.ok is False
        assert result.code == ERR_MALFORMED_EXTENSION

    def test_portable_completed_missing_receipt(self) -> None:
        result = validate_terminal_evidence(
            {
                "schema": SCHEMA_TERMINAL_EVIDENCE,
                "extension_uri": EXTENSION_URI,
                "task_id": "task-bad",
                "task_state": "completed",
                "portable": True,
                "envelope_cid": "bafkreidpgkdasegkb6zkedd73ikdmzvqtw7y3njdqgk4scsyn62uf7ymvu",
            }
        )
        assert result.ok is False
        assert result.code == ERR_MISSING_RECEIPT_CID

    def test_non_a2a_task_state_rejected(self) -> None:
        result = validate_terminal_evidence(
            {
                "schema": SCHEMA_TERMINAL_EVIDENCE,
                "extension_uri": EXTENSION_URI,
                "task_state": "succeeded",
                "portable": False,
            }
        )
        assert result.ok is False
        assert result.code == ERR_MALFORMED_EXTENSION

    def test_task_metadata_unknown_property(self) -> None:
        result = validate_task_metadata(
            {
                "schema": "mcp++/a2a/task-metadata@1",
                "envelope_cid": "bafkreidpgkdasegkb6zkedd73ikdmzvqtw7y3njdqgk4scsyn62uf7ymvu",
                "mcpp_private_status": "running",
            }
        )
        assert result.ok is False
        assert result.code == ERR_MALFORMED_EXTENSION

    def test_adapter_reject_malformed_extension(
        self, adapter: A2ATaskAdapter
    ) -> None:
        with pytest.raises(A2AExtensionError) as exc:
            adapter.reject_malformed_extension({"uri": WORKING_ALIAS})
        assert exc.value.code == ERR_MALFORMED_EXTENSION_URI

    def test_malformed_vector_suite_expected_failures(
        self, adapter: A2ATaskAdapter
    ) -> None:
        suite = _load_vector("malformed")
        assert suite.get("valid") is False
        for case in suite.get("cases") or []:
            result = adapter.evaluate_vector_case(case)
            assert result.ok, (
                f"malformed case {case.get('id')} should be rejected by adapter "
                f"and scored as suite-pass: {result.errors} code={result.code}"
            )


# ---------------------------------------------------------------------------
# Unsupported profile fails closed
# ---------------------------------------------------------------------------


class TestUnsupportedProfileFailClosed:
    def test_unknown_profile_letter(self) -> None:
        result = validate_profile_request(["A", "B"], ["Z"])
        assert result.ok is False
        assert result.code == ERR_UNSUPPORTED_PROFILE

    def test_requested_not_advertised(self) -> None:
        result = validate_profile_request(["A", "B", "C"], ["H"])
        assert result.ok is False
        assert result.code == ERR_PROFILE_NOT_SUBSET

    def test_handoff_rejects_unsupported_profile(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        with pytest.raises(A2AExtensionError) as exc:
            adapter.handoff(
                client,
                server,
                requested_profiles=["Z"],  # type: ignore[list-item]
            )
        assert exc.value.code == ERR_UNSUPPORTED_PROFILE

    def test_handoff_rejects_profile_not_advertised(
        self, adapter: A2ATaskAdapter
    ) -> None:
        client = adapter.create_agent(agent_id="c1", profiles=["A", "B"])
        server = adapter.create_agent(agent_id="s1", profiles=["A", "B"])
        with pytest.raises(A2AExtensionError) as exc:
            adapter.handoff(
                client,
                server,
                requested_profiles=["A", "C"],
            )
        assert exc.value.code == ERR_PROFILE_NOT_SUBSET

    def test_unsupported_profile_vector_suite(
        self, adapter: A2ATaskAdapter
    ) -> None:
        suite = _load_vector("unsupported-profile")
        assert suite.get("valid") is False
        for case in suite.get("cases") or []:
            result = adapter.evaluate_vector_case(case)
            assert result.ok, (
                f"unsupported-profile case {case.get('id')} expected rejection: "
                f"{result.errors} code={result.code}"
            )

    def test_allowed_profiles_closed_set(self) -> None:
        assert ALLOWED_PROFILES == frozenset("ABCDEFGH")
        for letter in DEFAULT_PROFILES:
            assert letter in ALLOWED_PROFILES


# ---------------------------------------------------------------------------
# Retry + streaming
# ---------------------------------------------------------------------------


class TestRetryAndStreaming:
    def test_retry_after_failure_links_event_dag(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        first = adapter.handoff(client, server, fail=True)
        assert first["task"]["status"]["state"] == TaskState.FAILED.value
        failed_id = first["task"]["id"]
        prior_event = first["terminal_evidence"]["event_cid"]

        retried = server.retry_task(
            failed_id,
            a2a_extensions=[EXTENSION_URI],
            execute=True,
            fail=False,
        )
        assert retried["status"]["state"] == TaskState.COMPLETED.value
        assert retried["attempt"] == 2
        assert retried["id"] != failed_id

        lineage = server.event_lineage(retried["id"])
        assert prior_event in lineage or any(
            prior_event in (server.event_dag.get_event(cid) or {}).get("parents", [])
            for cid in lineage
        )
        # Retry event kind exists in DAG.
        snap = server.event_dag.export_snapshot()
        kinds = {
            (item.get("payload") or {}).get("kind")
            for item in snap.get("events") or []
        }
        assert "task.retry" in kinds
        assert "task.terminal" in kinds

    def test_streaming_emits_status_then_terminal(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        receipt = adapter.handoff(client, server)
        task_id = receipt["task"]["id"]
        events = list(server.stream_task(task_id))
        assert events
        states = [e["state"] for e in events]
        assert TaskState.WORKING.value in states or TaskState.COMPLETED.value in states
        assert events[-1]["state"] == TaskState.COMPLETED.value
        assert events[-1]["kind"] == "terminal"
        # Stream closes on terminal (last event is terminal).
        assert events[-1]["state"] in TERMINAL_TASK_STATES

    def test_streaming_while_open_then_cancel(
        self, adapter: A2ATaskAdapter, two_agents
    ) -> None:
        client, server = two_agents
        receipt = adapter.handoff(client, server, hold_open=True)
        task_id = receipt["task"]["id"]
        mid = list(server.stream_task(task_id))
        assert any(e["state"] == TaskState.WORKING.value for e in mid)
        adapter.cancel(server, task_id)
        final = list(server.stream_task(task_id))
        assert final[-1]["state"] == TaskState.CANCELED.value


# ---------------------------------------------------------------------------
# Mapping / activation helpers
# ---------------------------------------------------------------------------


class TestMappingsAndActivation:
    def test_result_status_mapping_table(self) -> None:
        assert map_result_status_to_task_state("succeeded") == "completed"
        assert map_result_status_to_task_state("failed") == "failed"
        assert map_result_status_to_task_state("timed_out") == "failed"
        assert map_result_status_to_task_state("cancelled") == "canceled"
        assert map_result_status_to_task_state("rejected") == "rejected"
        with pytest.raises(A2AExtensionError):
            map_result_status_to_task_state("running")

    def test_parse_a2a_extensions_header(self) -> None:
        assert parse_a2a_extensions_header(EXTENSION_URI) == [EXTENSION_URI]
        multi = parse_a2a_extensions_header(
            f"https://example.com/ext/v1, {EXTENSION_URI}"
        )
        assert EXTENSION_URI in multi
        assert len(multi) == 2

    def test_activation_success_echo(self) -> None:
        result = validate_activation([EXTENSION_URI])
        assert result.ok
        assert result.metadata["mcp_plus_plus_execution_activated"] is True

    def test_extension_required_agent(
        self, adapter: A2ATaskAdapter
    ) -> None:
        client = adapter.create_agent(agent_id="c-req")
        server = adapter.create_agent(
            agent_id="s-req", extension_required=True
        )
        with pytest.raises(A2AExtensionError) as exc:
            # Empty activation list fails structural validation first.
            server.activate([], require_execution=True)
        assert exc.value.code in {
            ERR_EXTENSION_REQUIRED,
            ERR_MALFORMED_EXTENSION,
            ERR_NOT_ACTIVATED,
        }
        # Successful required activation still works with proper URI.
        act = server.activate([EXTENSION_URI], require_execution=True)
        assert act["activated"] is True
        # Handoff still succeeds when activated.
        receipt = adapter.handoff(client, server)
        assert receipt["task"]["status"]["state"] == TaskState.COMPLETED.value

    def test_well_formed_vector_suite_passes(
        self, adapter: A2ATaskAdapter
    ) -> None:
        suite = _load_vector("well-formed")
        assert suite.get("valid") is True
        for case in suite.get("cases") or []:
            result = adapter.evaluate_vector_case(case)
            assert result.ok, f"{case.get('id')}: {result.errors}"


# ---------------------------------------------------------------------------
# End-to-end acceptance bundle (single high-signal test)
# ---------------------------------------------------------------------------


def test_mcpp_056_acceptance_bundle(adapter: A2ATaskAdapter) -> None:
    """Bundle the three acceptance bullets into one integration path."""
    client = adapter.create_agent(agent_id="acc-client", name="Acceptance Client")
    server = adapter.create_agent(agent_id="acc-server", name="Acceptance Server")
    assert client.event_dag is not server.event_dag

    # 1) Two agents complete a handoff.
    done = adapter.handoff(client, server, text="acceptance handoff")
    assert done["task"]["status"]["state"] == "completed"
    assert done["terminal_evidence"]["receipt_cid"]
    assert done["terminal_evidence"]["event_cid"]

    # 2) Cancel writes Event DAG records (separate open task).
    open_task = adapter.handoff(client, server, hold_open=True)
    tid = open_task["task"]["id"]
    before = server.event_dag.stats()["event_count"]
    canceled = adapter.cancel(server, tid, reason="acceptance-cancel")
    assert canceled["task"]["status"]["state"] == "canceled"
    assert canceled["cancel_events"]
    assert server.event_dag.stats()["event_count"] == before + 1
    assert canceled["cancel_events"][0]["kind"] == "task.canceled"

    # 3) Malformed extension fails closed.
    bad = validate_agent_extension({"uri": WORKING_ALIAS})
    assert bad.ok is False
    assert bad.code == ERR_MALFORMED_EXTENSION_URI
    with pytest.raises(A2AExtensionError):
        adapter.reject_malformed_extension(
            {"uri": "io.modelcontextprotocol/tasks"}
        )
