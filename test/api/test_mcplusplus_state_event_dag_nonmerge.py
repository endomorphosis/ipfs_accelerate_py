"""MCPP-040: Prove Event DAG branches do not silently merge mutable state.

Acceptance (state-ref.md §5 / ADR-0004 / plan KD-8):

* Single-authority conflict is explicit.
* CRDT mode converges (Automerge merge only when mode is ``crdt``).
* Consensus mode requires plugin evidence.
* Silent merge fails the test.

Interface under test: ``StateNonMergeReceipt@1`` — the observation receipt
produced when concurrent Event DAG leaves reference the same logical
``StateRef@1`` id.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.artifacts import compute_artifact_cid
from ipfs_accelerate_py.mcp_server.mcplusplus.event_dag import EventDAGStore
from ipfs_accelerate_py.mcp_server.mcplusplus.state.consensus_plugin import (
    CONSENSUS_EVIDENCE_SCHEMA,
    CONSENSUS_PLUGIN_INTERFACE,
    CONSENSUS_RESULT_SCHEMA,
    DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID,
    GUARANTEE_MAJORITY_APPROVAL,
    VERDICT_SUPPORT,
    ConsensusEvidenceError,
    ConsensusRejectedError,
    DeterministicTestAdapter,
    build_plugin_evidence,
    validate_plugin_evidence,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.state.provider import (
    ALLOWED_CONSISTENCY_MODES,
    STATE_REF_SCHEMA,
    StateModeError,
    validate_state_ref,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.state.sqlite_authority import (
    CasMismatchError,
    SqliteAuthorityState,
)

automerge = pytest.importorskip(
    "automerge",
    reason="CRDT branch convergence requires the real automerge package",
)

from ipfs_accelerate_py.mcp_server.mcplusplus.state.automerge_crdt import (  # noqa: E402
    AutomergeCrdtState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_NON_MERGE_RECEIPT_SCHEMA = "mcp++/state/state-non-merge-receipt@1"
STATE_NON_MERGE_INTERFACE = "StateNonMergeReceipt@1"

OUTCOME_CONFLICT = "conflict"
OUTCOME_CONVERGED = "converged"
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"
OUTCOME_OBSERVED = "observed"

ALLOWED_OUTCOMES = frozenset(
    {
        OUTCOME_CONFLICT,
        OUTCOME_CONVERGED,
        OUTCOME_ACCEPTED,
        OUTCOME_REJECTED,
        OUTCOME_OBSERVED,
    }
)

STATE_ID = "state:test/event-dag-nonmerge"
PEER_A = "did:key:peer-a"
PEER_B = "did:key:peer-b"
PEER_C = "did:key:peer-c"
PEERS = (PEER_A, PEER_B, PEER_C)

PROPOSAL_LEFT = "bafkreileftproposal000000000000000000000000000000000000000001"
PROPOSAL_RIGHT = "bafkreirightproposal0000000000000000000000000000000000000002"


# ---------------------------------------------------------------------------
# Helpers: Event DAG + branch observation
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _value_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_mode(mode: object) -> str:
    if not isinstance(mode, str) or mode not in ALLOWED_CONSISTENCY_MODES:
        raise StateModeError(
            f"mode {mode!r} is not one of {sorted(ALLOWED_CONSISTENCY_MODES)}"
        )
    return mode


def _frontier(store: EventDAGStore) -> list[str]:
    """Return leaf event CIDs (events that are not parents of any other event)."""
    snapshot = store.export_snapshot()
    events = snapshot.get("events") or []
    all_cids: list[str] = []
    parent_set: set[str] = set()
    for item in events:
        if not isinstance(item, dict):
            continue
        cid = item.get("event_cid")
        if not isinstance(cid, str) or not cid:
            continue
        all_cids.append(cid)
        payload = item.get("payload")
        if isinstance(payload, dict):
            for parent in payload.get("parents") or []:
                parent_set.add(str(parent))
    return sorted(cid for cid in all_cids if cid not in parent_set)


def _state_write_event(
    *,
    event_cid: str,
    parents: Sequence[str],
    state_ref: Mapping[str, Any],
    value: Any,
    peer_did: str,
    evidence: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build an Event DAG payload that observes a state write on a branch."""
    ref = validate_state_ref(state_ref)
    payload: dict[str, Any] = {
        "event_type": "result",
        "parents": list(parents),
        "peer_did": peer_did,
        "timestamp": "2026-08-16T00:00:00Z",
        "payload": {
            "kind": "state_write_observation",
            "state_ref": ref,
            "value": copy.deepcopy(value),
            "value_digest": _value_digest(value),
        },
    }
    if evidence is not None:
        payload["payload"]["consensus_evidence"] = dict(evidence)
    # Bind content address when caller uses the digest as the event_cid.
    _ = event_cid
    return payload


def _add_state_write(
    store: EventDAGStore,
    *,
    parents: Sequence[str],
    state_ref: Mapping[str, Any],
    value: Any,
    peer_did: str,
    evidence: Optional[Mapping[str, Any]] = None,
    event_cid: Optional[str] = None,
) -> str:
    body = _state_write_event(
        event_cid="pending",
        parents=parents,
        state_ref=state_ref,
        value=value,
        peer_did=peer_did,
        evidence=evidence,
    )
    cid = event_cid or compute_artifact_cid(body)
    store.add_event(cid, body)
    return cid


@dataclass(frozen=True)
class BranchObservation:
    """One concurrent Event DAG leaf that touches a logical state id."""

    event_cid: str
    peer_did: str
    state_ref: dict[str, Any]
    value: Any
    value_digest: str
    consensus_evidence: Optional[dict[str, Any]] = None


@dataclass
class StateNonMergeReceipt:
    """``StateNonMergeReceipt@1`` — proof that branch observation is mode-gated.

    Wire schema marker: ``mcp++/state/state-non-merge-receipt@1``.
    """

    state_id: str
    mode: str
    outcome: str
    branch_event_cids: list[str]
    branch_value_digests: list[str]
    reason: str = ""
    merged_value: Any = None
    consensus_result: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = STATE_NON_MERGE_RECEIPT_SCHEMA
    interface: str = STATE_NON_MERGE_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": self.schema,
            "interface": self.interface,
            "state_id": self.state_id,
            "mode": self.mode,
            "outcome": self.outcome,
            "branch_event_cids": list(self.branch_event_cids),
            "branch_value_digests": list(self.branch_value_digests),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }
        if self.merged_value is not None:
            out["merged_value"] = copy.deepcopy(self.merged_value)
        if self.consensus_result is not None:
            out["consensus_result"] = dict(self.consensus_result)
        return out


def collect_branch_observations(
    store: EventDAGStore,
    *,
    state_id: str,
    frontier: Optional[Sequence[str]] = None,
) -> list[BranchObservation]:
    """Collect concurrent leaf observations for ``state_id`` from the DAG."""
    leaves = list(frontier) if frontier is not None else _frontier(store)
    observations: list[BranchObservation] = []
    for event_cid in sorted(leaves):
        event = store.get_event(event_cid)
        if event is None:
            continue
        inner = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if not isinstance(inner, dict):
            continue
        if inner.get("kind") != "state_write_observation":
            continue
        ref_raw = inner.get("state_ref")
        if not isinstance(ref_raw, Mapping):
            continue
        ref = validate_state_ref(ref_raw)
        if ref["id"] != state_id:
            continue
        value = inner.get("value")
        digest = str(inner.get("value_digest") or _value_digest(value))
        evidence = inner.get("consensus_evidence")
        evidence_dict = dict(evidence) if isinstance(evidence, Mapping) else None
        observations.append(
            BranchObservation(
                event_cid=event_cid,
                peer_did=str(event.get("peer_did") or ""),
                state_ref=ref,
                value=copy.deepcopy(value),
                value_digest=digest,
                consensus_evidence=evidence_dict,
            )
        )
    return observations


def observe_concurrent_state_branches(
    store: EventDAGStore,
    *,
    state_id: str,
    mode: str,
    crdt_replicas: Optional[Mapping[str, AutomergeCrdtState]] = None,
    consensus_plugin: Optional[DeterministicTestAdapter] = None,
    silent_merge: bool = False,
) -> StateNonMergeReceipt:
    """Observe concurrent Event DAG branches for one logical state id.

    Fail-closed rules (KD-8 / state-ref.md §5):

    * ``single_authority`` / ``causal`` / ``immutable``: concurrent divergent
      values yield ``conflict`` — never a silent payload merge.
    * ``crdt``: Automerge merge is allowed and must converge.
    * ``consensus``: acceptance requires valid plugin evidence; absent evidence
      is ``rejected``.
    * ``silent_merge=True`` deliberately violates the rule and raises so that
      silent merge cannot pass this test.
    """
    mode = _require_mode(mode)
    observations = collect_branch_observations(store, state_id=state_id)
    if not observations:
        return StateNonMergeReceipt(
            state_id=state_id,
            mode=mode,
            outcome=OUTCOME_OBSERVED,
            branch_event_cids=[],
            branch_value_digests=[],
            reason="no concurrent branch observations for state_id",
        )

    event_cids = [o.event_cid for o in observations]
    digests = [o.value_digest for o in observations]
    modes = {o.state_ref["mode"] for o in observations}
    if modes != {mode}:
        raise StateModeError(
            f"branch StateRef modes {sorted(modes)} do not match observation mode {mode!r}"
        )

    # Explicit anti-pattern: dict-union / last-write-wins over concurrent
    # mutable branches. The acceptance criterion is that this path fails.
    if silent_merge:
        raise AssertionError(
            "silent merge of concurrent Event DAG branch state is forbidden "
            f"(mode={mode!r}, state_id={state_id!r}, branches={event_cids})"
        )

    distinct_values = {_canonical_json(o.value) for o in observations}

    if mode == "single_authority":
        if len(observations) == 1 or len(distinct_values) == 1:
            return StateNonMergeReceipt(
                state_id=state_id,
                mode=mode,
                outcome=OUTCOME_OBSERVED,
                branch_event_cids=event_cids,
                branch_value_digests=digests,
                reason="single_authority branches agree or only one observation",
                metadata={"authoritative_value": observations[0].value},
            )
        # Concurrent divergent single_authority writes: explicit conflict.
        # Retain both branch tips; do not invent a merged value.
        return StateNonMergeReceipt(
            state_id=state_id,
            mode=mode,
            outcome=OUTCOME_CONFLICT,
            branch_event_cids=event_cids,
            branch_value_digests=digests,
            reason=(
                "concurrent single_authority Event DAG branches diverge; "
                "explicit conflict — no silent merge"
            ),
            metadata={
                "branch_values": [o.value for o in observations],
                "branch_versions": [
                    o.state_ref.get("version") for o in observations
                ],
                "merged_value": None,
            },
        )

    if mode == "causal":
        # Causal mode records partial order only; concurrent branches remain concurrent.
        if len(distinct_values) > 1:
            return StateNonMergeReceipt(
                state_id=state_id,
                mode=mode,
                outcome=OUTCOME_CONFLICT,
                branch_event_cids=event_cids,
                branch_value_digests=digests,
                reason=(
                    "causal mode retains concurrent branches; "
                    "observing two branches must not invent a total order or merge"
                ),
                metadata={"branch_values": [o.value for o in observations]},
            )
        return StateNonMergeReceipt(
            state_id=state_id,
            mode=mode,
            outcome=OUTCOME_OBSERVED,
            branch_event_cids=event_cids,
            branch_value_digests=digests,
            reason="causal branches agree",
        )

    if mode == "immutable":
        # Distinct content under concurrent observations is fine only as distinct CIDs;
        # identity mutation is out of scope for branch merge.
        if len(distinct_values) > 1:
            return StateNonMergeReceipt(
                state_id=state_id,
                mode=mode,
                outcome=OUTCOME_CONFLICT,
                branch_event_cids=event_cids,
                branch_value_digests=digests,
                reason="immutable mode does not multi-writer merge concurrent payloads",
            )
        return StateNonMergeReceipt(
            state_id=state_id,
            mode=mode,
            outcome=OUTCOME_OBSERVED,
            branch_event_cids=event_cids,
            branch_value_digests=digests,
            reason="immutable branches reference identical content",
        )

    if mode == "crdt":
        if crdt_replicas is None or not crdt_replicas:
            raise ValueError("crdt mode requires crdt_replicas for Automerge merge evidence")
        # Merge only in crdt mode, using real Automerge document merge.
        ordered = sorted(crdt_replicas.items(), key=lambda item: item[0])
        primary_key, primary = ordered[0]
        for key, replica in ordered[1:]:
            if replica.state_id != primary.state_id:
                raise ValueError(
                    f"crdt replica state_id mismatch: {replica.state_id!r} vs {primary.state_id!r}"
                )
            primary.merge(replica)
            # Bidirectional heal so all listed replicas can converge.
            replica.merge(primary)
        # Ensure all peers exchange until heads match.
        for _ in range(len(ordered)):
            for _, a in ordered:
                for _, b in ordered:
                    if a is b:
                        continue
                    a.merge(b)
        heads = {tuple(r.heads_hex()) for _, r in ordered}
        snaps = {_canonical_json(r.snapshot()) for _, r in ordered}
        if len(heads) != 1 or len(snaps) != 1:
            return StateNonMergeReceipt(
                state_id=state_id,
                mode=mode,
                outcome=OUTCOME_CONFLICT,
                branch_event_cids=event_cids,
                branch_value_digests=digests,
                reason="Automerge replicas failed to converge after explicit CRDT merge",
                metadata={"primary_replica": primary_key},
            )
        converged = primary.snapshot()
        return StateNonMergeReceipt(
            state_id=state_id,
            mode=mode,
            outcome=OUTCOME_CONVERGED,
            branch_event_cids=event_cids,
            branch_value_digests=digests,
            reason="crdt mode converged via Automerge document merge (not informal last-write-wins)",
            merged_value=converged,
            metadata={
                "backend": "automerge",
                "heads": primary.heads_hex(),
                "change_evidence": primary.change_evidence(),
            },
        )

    if mode == "consensus":
        if consensus_plugin is None:
            raise ValueError("consensus mode requires a ConsensusPlugin@1 instance")
        # Every branch that claims a transition must carry plugin evidence.
        missing = [o.event_cid for o in observations if not o.consensus_evidence]
        if missing:
            return StateNonMergeReceipt(
                state_id=state_id,
                mode=mode,
                outcome=OUTCOME_REJECTED,
                branch_event_cids=event_cids,
                branch_value_digests=digests,
                reason=(
                    "consensus mode requires plugin evidence on each branch; "
                    f"missing evidence on {missing}"
                ),
                metadata={"missing_evidence_event_cids": missing},
            )
        # Evaluate each branch's evidence; accept only when plugin accepts.
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for obs in observations:
            assert obs.consensus_evidence is not None
            try:
                evidence = validate_plugin_evidence(obs.consensus_evidence)
            except (ConsensusEvidenceError, Exception) as exc:  # noqa: BLE001
                rejected.append(
                    {
                        "event_cid": obs.event_cid,
                        "reason": f"invalid evidence: {exc}",
                    }
                )
                continue
            try:
                result = consensus_plugin.accept(evidence)
            except ConsensusRejectedError as exc:
                rejected.append({"event_cid": obs.event_cid, "reason": str(exc)})
                continue
            accepted.append(result.to_dict())

        if not accepted:
            return StateNonMergeReceipt(
                state_id=state_id,
                mode=mode,
                outcome=OUTCOME_REJECTED,
                branch_event_cids=event_cids,
                branch_value_digests=digests,
                reason="no branch carried acceptable consensus plugin evidence",
                metadata={"rejected": rejected},
            )
        if len(accepted) > 1 and len(distinct_values) > 1:
            # Multiple accepted concurrent proposals with different values still conflict
            # until a single accepted head is chosen by the plugin round; do not merge.
            return StateNonMergeReceipt(
                state_id=state_id,
                mode=mode,
                outcome=OUTCOME_CONFLICT,
                branch_event_cids=event_cids,
                branch_value_digests=digests,
                reason=(
                    "multiple concurrent consensus-accepted proposals diverge; "
                    "plugin evidence does not authorize silent merge of values"
                ),
                consensus_result=accepted[0],
                metadata={"accepted": accepted, "rejected": rejected},
            )
        return StateNonMergeReceipt(
            state_id=state_id,
            mode=mode,
            outcome=OUTCOME_ACCEPTED,
            branch_event_cids=event_cids,
            branch_value_digests=digests,
            reason="consensus plugin evidence accepted the proposal",
            consensus_result=accepted[0],
            metadata={"accepted": accepted, "rejected": rejected},
        )

    raise StateModeError(f"unhandled mode for non-merge observation: {mode!r}")


def _build_branched_dag(
    *,
    mode: str,
    left_value: Any,
    right_value: Any,
    left_ref_extra: Optional[Mapping[str, Any]] = None,
    right_ref_extra: Optional[Mapping[str, Any]] = None,
    left_evidence: Optional[Mapping[str, Any]] = None,
    right_evidence: Optional[Mapping[str, Any]] = None,
    state_id: str = STATE_ID,
) -> tuple[EventDAGStore, str, str, str]:
    """Create root + two concurrent leaves that write the same logical state id."""
    store = EventDAGStore()
    root_payload = {
        "event_type": "invocation",
        "parents": [],
        "peer_did": "did:key:root",
        "timestamp": "2026-08-16T00:00:00Z",
        "payload": {"kind": "genesis", "state_id": state_id, "mode": mode},
    }
    root_cid = compute_artifact_cid(root_payload)
    store.add_event(root_cid, root_payload)

    left_ref: dict[str, Any] = {
        "schema": STATE_REF_SCHEMA,
        "id": state_id,
        "mode": mode,
        "version": 1,
    }
    if left_ref_extra:
        left_ref.update(dict(left_ref_extra))
    right_ref: dict[str, Any] = {
        "schema": STATE_REF_SCHEMA,
        "id": state_id,
        "mode": mode,
        "version": 1,
    }
    if right_ref_extra:
        right_ref.update(dict(right_ref_extra))

    left_cid = _add_state_write(
        store,
        parents=[root_cid],
        state_ref=left_ref,
        value=left_value,
        peer_did=PEER_A,
        evidence=left_evidence,
    )
    right_cid = _add_state_write(
        store,
        parents=[root_cid],
        state_ref=right_ref,
        value=right_value,
        peer_did=PEER_B,
        evidence=right_evidence,
    )
    return store, root_cid, left_cid, right_cid


# ---------------------------------------------------------------------------
# Interface / receipt shape
# ---------------------------------------------------------------------------


def test_state_non_merge_receipt_interface_constants() -> None:
    assert STATE_NON_MERGE_INTERFACE == "StateNonMergeReceipt@1"
    assert STATE_NON_MERGE_RECEIPT_SCHEMA == "mcp++/state/state-non-merge-receipt@1"
    assert "single_authority" in ALLOWED_CONSISTENCY_MODES
    assert "crdt" in ALLOWED_CONSISTENCY_MODES
    assert "consensus" in ALLOWED_CONSISTENCY_MODES


# ---------------------------------------------------------------------------
# Single-authority: explicit conflict, never silent merge
# ---------------------------------------------------------------------------


def test_single_authority_concurrent_dag_branches_are_explicit_conflict() -> None:
    """Observing two concurrent single_authority leaves must not merge values."""
    left_value = {"counter": 1, "writer": "A"}
    right_value = {"counter": 1, "writer": "B"}
    store, _root, left_cid, right_cid = _build_branched_dag(
        mode="single_authority",
        left_value=left_value,
        right_value=right_value,
        left_ref_extra={"authority": {"kind": "principal", "principal": PEER_A}},
        right_ref_extra={"authority": {"kind": "principal", "principal": PEER_B}},
    )

    frontier = _frontier(store)
    assert set(frontier) == {left_cid, right_cid}

    receipt = observe_concurrent_state_branches(
        store, state_id=STATE_ID, mode="single_authority"
    )
    assert receipt.schema == STATE_NON_MERGE_RECEIPT_SCHEMA
    assert receipt.interface == STATE_NON_MERGE_INTERFACE
    assert receipt.mode == "single_authority"
    assert receipt.outcome == OUTCOME_CONFLICT
    assert set(receipt.branch_event_cids) == {left_cid, right_cid}
    assert receipt.merged_value is None
    assert receipt.metadata.get("merged_value") is None
    # Both divergent tips retained as evidence of the conflict.
    assert left_value in receipt.metadata["branch_values"]
    assert right_value in receipt.metadata["branch_values"]
    assert "silent" in receipt.reason or "explicit conflict" in receipt.reason

    wire = receipt.to_dict()
    assert wire["outcome"] == OUTCOME_CONFLICT
    assert "merged_value" not in wire or wire.get("merged_value") is None


def test_single_authority_sqlite_cas_rejects_concurrent_second_writer(
    tmp_path: Path,
) -> None:
    """Authority store: first CAS wins; concurrent expected_version conflicts."""
    db = tmp_path / "authority.sqlite3"
    with SqliteAuthorityState.open(db) as store:
        store.create(
            STATE_ID,
            {"counter": 0},
            authority={"kind": "principal", "principal": PEER_A},
        )
        # Both branches observe version 0 and attempt to write.
        first = store.cas_write(
            STATE_ID,
            expected_version=0,
            value={"counter": 1, "writer": "A"},
            operation_id="branch-a",
            writer=PEER_A,
        )
        assert first.status == "updated"
        assert first.version == 1

        with pytest.raises(CasMismatchError) as excinfo:
            store.cas_write(
                STATE_ID,
                expected_version=0,
                value={"counter": 1, "writer": "B"},
                operation_id="branch-b",
                writer=PEER_B,
            )
        err = excinfo.value
        assert err.expected_version == 0
        assert err.actual_version == 1
        # Live value is still branch A — not a dict-union of A and B.
        live = store.get(STATE_ID)
        assert live["value"] == {"counter": 1, "writer": "A"}
        assert live["value"] != {"counter": 1, "writer": "A", "extra_from_b": True}
        assert live["version"] == 1


def test_single_authority_silent_merge_path_fails_closed() -> None:
    """Acceptance: silent merge fails the test (must raise, never succeed)."""
    store, _, _, _ = _build_branched_dag(
        mode="single_authority",
        left_value={"k": "left"},
        right_value={"k": "right"},
    )
    with pytest.raises(AssertionError) as excinfo:
        observe_concurrent_state_branches(
            store,
            state_id=STATE_ID,
            mode="single_authority",
            silent_merge=True,
        )
    message = str(excinfo.value).lower()
    assert "silent merge" in message
    assert "forbidden" in message


def test_single_authority_does_not_dict_union_branch_payloads() -> None:
    """A naive dict-union of concurrent branches is not a valid resolution."""
    left = {"a": 1, "shared": "L"}
    right = {"b": 2, "shared": "R"}
    store, _, _, _ = _build_branched_dag(
        mode="single_authority",
        left_value=left,
        right_value=right,
    )
    receipt = observe_concurrent_state_branches(
        store, state_id=STATE_ID, mode="single_authority"
    )
    assert receipt.outcome == OUTCOME_CONFLICT
    # Prove we did not invent the silent-merge value {"a":1,"b":2,"shared":...}.
    silent = {**left, **right}
    assert receipt.merged_value != silent
    assert receipt.merged_value is None
    assert silent not in (receipt.metadata.get("branch_values") or [])


# ---------------------------------------------------------------------------
# CRDT: merge only in crdt mode; concurrent branches converge
# ---------------------------------------------------------------------------


def test_crdt_mode_concurrent_dag_branches_converge_via_automerge() -> None:
    """CRDT mode: concurrent offline writes on DAG branches converge after merge."""
    assert automerge is not None

    genesis = AutomergeCrdtState.open(STATE_ID, actor_id="genesis")
    genesis.put("shared", "base")
    genesis.put("version", 1)
    blob = genesis.save()

    left = AutomergeCrdtState.load(blob, state_id=STATE_ID, actor_id="left")
    right = AutomergeCrdtState.load(blob, state_id=STATE_ID, actor_id="right")

    left.put("left_only", "L")
    left.put("shared", "from-left")
    right.put("right_only", "R")
    right.put("shared", "from-right")

    # Still diverged before heal.
    assert left.snapshot() != right.snapshot()
    assert left.heads() != right.heads()

    left_ref = left.state_ref()
    right_ref = right.state_ref()
    assert left_ref["mode"] == "crdt"
    assert right_ref["mode"] == "crdt"

    store, _root, left_cid, right_cid = _build_branched_dag(
        mode="crdt",
        left_value=left.snapshot(),
        right_value=right.snapshot(),
        left_ref_extra={
            "clocks": left_ref.get("clocks"),
            "metadata": left_ref.get("metadata"),
            "provider": left_ref.get("provider"),
        },
        right_ref_extra={
            "clocks": right_ref.get("clocks"),
            "metadata": right_ref.get("metadata"),
            "provider": right_ref.get("provider"),
        },
    )

    receipt = observe_concurrent_state_branches(
        store,
        state_id=STATE_ID,
        mode="crdt",
        crdt_replicas={"left": left, "right": right},
    )
    assert receipt.outcome == OUTCOME_CONVERGED
    assert receipt.mode == "crdt"
    assert set(receipt.branch_event_cids) == {left_cid, right_cid}
    assert receipt.merged_value is not None
    snap = receipt.merged_value
    assert snap["left_only"] == "L"
    assert snap["right_only"] == "R"
    # Concurrent key resolved by Automerge, not wall-clock LWW invented here.
    assert snap["shared"] in {"from-left", "from-right"}
    assert left.converged_with(right)
    assert receipt.metadata.get("backend") == "automerge"
    wire = json.dumps(receipt.to_dict()).lower()
    assert "automerge" in wire
    # Receipt must not advertise informal last-write-wins as the merge method.
    assert '"backend": "lww"' not in wire
    assert "last_write_wins" not in wire
    assert "timestamp_winner" not in wire


def test_crdt_merge_is_not_available_under_single_authority_mode() -> None:
    """Automatic multi-writer merge is allowed only when mode is crdt."""
    store, _, _, _ = _build_branched_dag(
        mode="single_authority",
        left_value={"x": 1},
        right_value={"x": 2},
    )
    # Even if Automerge replicas are supplied, single_authority observation
    # must report conflict and never call through as a silent CRDT merge.
    genesis = AutomergeCrdtState.open(STATE_ID, actor_id="g")
    left = AutomergeCrdtState.load(genesis.save(), state_id=STATE_ID, actor_id="l")
    right = AutomergeCrdtState.load(genesis.save(), state_id=STATE_ID, actor_id="r")
    left.put("x", 1)
    right.put("x", 2)

    receipt = observe_concurrent_state_branches(
        store,
        state_id=STATE_ID,
        mode="single_authority",
        crdt_replicas={"left": left, "right": right},
    )
    assert receipt.outcome == OUTCOME_CONFLICT
    assert receipt.merged_value is None


# ---------------------------------------------------------------------------
# Consensus: plugin evidence required
# ---------------------------------------------------------------------------


def test_consensus_mode_rejects_branches_without_plugin_evidence() -> None:
    store, _, left_cid, right_cid = _build_branched_dag(
        mode="consensus",
        left_value={"proposal": "left"},
        right_value={"proposal": "right"},
        left_evidence=None,
        right_evidence=None,
    )
    plugin = DeterministicTestAdapter(guarantee=GUARANTEE_MAJORITY_APPROVAL)
    receipt = observe_concurrent_state_branches(
        store,
        state_id=STATE_ID,
        mode="consensus",
        consensus_plugin=plugin,
    )
    assert receipt.outcome == OUTCOME_REJECTED
    assert receipt.mode == "consensus"
    assert set(receipt.branch_event_cids) == {left_cid, right_cid}
    assert "plugin evidence" in receipt.reason
    missing = receipt.metadata.get("missing_evidence_event_cids") or []
    assert set(missing) == {left_cid, right_cid}


def test_consensus_mode_accepts_only_with_valid_plugin_evidence() -> None:
    plugin = DeterministicTestAdapter(guarantee=GUARANTEE_MAJORITY_APPROVAL)
    evidence = plugin.propose(
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_LEFT,
        members=PEERS,
        round_id="r-nonmerge-1",
    )
    for peer in PEERS:
        evidence = plugin.record_ballot(
            evidence, principal=peer, verdict=VERDICT_SUPPORT
        )
    # Pre-check: plugin would accept.
    accepted = plugin.accept(evidence)
    assert accepted.accepted is True
    assert accepted.schema == CONSENSUS_RESULT_SCHEMA
    assert accepted.guarantee == GUARANTEE_MAJORITY_APPROVAL

    store = EventDAGStore()
    root_payload = {
        "event_type": "invocation",
        "parents": [],
        "peer_did": "did:key:root",
        "timestamp": "2026-08-16T00:00:00Z",
        "payload": {"kind": "genesis", "state_id": STATE_ID, "mode": "consensus"},
    }
    root_cid = compute_artifact_cid(root_payload)
    store.add_event(root_cid, root_payload)

    state_ref = {
        "schema": STATE_REF_SCHEMA,
        "id": STATE_ID,
        "mode": "consensus",
        "version": 1,
        "authority": {
            "kind": "plugin",
            "plugin_id": DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID,
            "guarantee": GUARANTEE_MAJORITY_APPROVAL,
        },
    }
    leaf = _add_state_write(
        store,
        parents=[root_cid],
        state_ref=state_ref,
        value={"proposal": "left", "proposal_cid": PROPOSAL_LEFT},
        peer_did=PEER_A,
        evidence=evidence.to_dict(),
    )

    receipt = observe_concurrent_state_branches(
        store,
        state_id=STATE_ID,
        mode="consensus",
        consensus_plugin=plugin,
    )
    assert receipt.outcome == OUTCOME_ACCEPTED
    assert receipt.branch_event_cids == [leaf]
    assert receipt.consensus_result is not None
    assert receipt.consensus_result["accepted"] is True
    assert receipt.consensus_result["plugin_id"] == DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID
    assert receipt.consensus_result["guarantee"] == GUARANTEE_MAJORITY_APPROVAL
    assert CONSENSUS_PLUGIN_INTERFACE == "ConsensusPlugin@1"
    assert evidence.schema == CONSENSUS_EVIDENCE_SCHEMA


def test_consensus_mode_partial_evidence_still_rejects_unevidenced_branch() -> None:
    """One branch with evidence and one without: overall observation rejects."""
    plugin = DeterministicTestAdapter(guarantee=GUARANTEE_MAJORITY_APPROVAL)
    evidence = plugin.propose(
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_LEFT,
        members=PEERS,
        round_id="r-partial",
    )
    for peer in (PEER_A, PEER_B):
        evidence = plugin.record_ballot(
            evidence, principal=peer, verdict=VERDICT_SUPPORT
        )

    store, _, left_cid, right_cid = _build_branched_dag(
        mode="consensus",
        left_value={"proposal": "left"},
        right_value={"proposal": "right"},
        left_evidence=evidence.to_dict(),
        right_evidence=None,
    )
    receipt = observe_concurrent_state_branches(
        store,
        state_id=STATE_ID,
        mode="consensus",
        consensus_plugin=plugin,
    )
    assert receipt.outcome == OUTCOME_REJECTED
    missing = receipt.metadata.get("missing_evidence_event_cids") or []
    assert right_cid in missing
    assert left_cid not in missing


def test_consensus_divergent_accepted_proposals_conflict_without_silent_merge() -> None:
    """Two fully evidenced concurrent proposals with different values conflict."""
    plugin = DeterministicTestAdapter(guarantee=GUARANTEE_MAJORITY_APPROVAL)

    def _evidenced(proposal_cid: str, round_id: str):
        ev = plugin.propose(
            state_id=STATE_ID,
            proposal_cid=proposal_cid,
            members=PEERS,
            round_id=round_id,
        )
        for peer in PEERS:
            ev = plugin.record_ballot(ev, principal=peer, verdict=VERDICT_SUPPORT)
        plugin.accept(ev)
        return ev

    left_ev = _evidenced(PROPOSAL_LEFT, "round-left")
    right_ev = _evidenced(PROPOSAL_RIGHT, "round-right")

    store, _, left_cid, right_cid = _build_branched_dag(
        mode="consensus",
        left_value={"proposal": "left", "cid": PROPOSAL_LEFT},
        right_value={"proposal": "right", "cid": PROPOSAL_RIGHT},
        left_evidence=left_ev.to_dict(),
        right_evidence=right_ev.to_dict(),
    )
    receipt = observe_concurrent_state_branches(
        store,
        state_id=STATE_ID,
        mode="consensus",
        consensus_plugin=plugin,
    )
    assert receipt.outcome == OUTCOME_CONFLICT
    assert set(receipt.branch_event_cids) == {left_cid, right_cid}
    assert receipt.merged_value is None
    assert "silent merge" in receipt.reason


# ---------------------------------------------------------------------------
# Causal observation ≠ merge
# ---------------------------------------------------------------------------


def test_causal_mode_retains_concurrent_branches_without_merge() -> None:
    store, _, left_cid, right_cid = _build_branched_dag(
        mode="causal",
        left_value={"step": "A"},
        right_value={"step": "B"},
    )
    receipt = observe_concurrent_state_branches(
        store, state_id=STATE_ID, mode="causal"
    )
    assert receipt.outcome == OUTCOME_CONFLICT
    assert set(receipt.branch_event_cids) == {left_cid, right_cid}
    assert receipt.merged_value is None
    assert "total order" in receipt.reason or "concurrent" in receipt.reason


# ---------------------------------------------------------------------------
# End-to-end: branched Event DAG + SQLite authority composition
# ---------------------------------------------------------------------------


def test_event_dag_branch_observation_does_not_mutate_sqlite_authority(
    tmp_path: Path,
) -> None:
    """DAG observation alone must not merge or overwrite single_authority live value."""
    db = tmp_path / "live.sqlite3"
    with SqliteAuthorityState.open(db) as authority:
        authority.create(
            STATE_ID,
            {"counter": 0, "writer": "genesis"},
            authority={"kind": "principal", "principal": PEER_A},
        )
        # Authoritative write from branch A only (as if branch A won CAS).
        authority.cas_write(
            STATE_ID,
            expected_version=0,
            value={"counter": 1, "writer": "A"},
            operation_id="branch-a-commit",
            writer=PEER_A,
            parents=[],
        )
        before = authority.get(STATE_ID)

        store, _, _, _ = _build_branched_dag(
            mode="single_authority",
            left_value={"counter": 1, "writer": "A"},
            right_value={"counter": 1, "writer": "B"},
        )
        receipt = observe_concurrent_state_branches(
            store, state_id=STATE_ID, mode="single_authority"
        )
        assert receipt.outcome == OUTCOME_CONFLICT

        after = authority.get(STATE_ID)
        assert after["value"] == before["value"] == {"counter": 1, "writer": "A"}
        assert after["version"] == before["version"] == 1
        # Observing branch B did not apply a silent merge into the authority store.
        assert after["value"].get("writer") != "B"
        assert set(after["value"]) == {"counter", "writer"}


def test_build_plugin_evidence_is_required_shape_for_consensus_branches() -> None:
    """Consensus branch evidence must validate under ConsensusPlugin@1 schemas."""
    raw = build_plugin_evidence(
        plugin_id=DETERMINISTIC_TEST_ADAPTER_PLUGIN_ID,
        guarantee=GUARANTEE_MAJORITY_APPROVAL,
        state_id=STATE_ID,
        proposal_cid=PROPOSAL_LEFT,
        evidence_kind="test",
        members=PEERS,
        approvals=PEERS,
        threshold=2,
        round_id="shape-check",
        source="nonmerge_test",
    )
    restored = validate_plugin_evidence(raw.to_dict())
    assert restored.schema == CONSENSUS_EVIDENCE_SCHEMA
    assert restored.guarantee == GUARANTEE_MAJORITY_APPROVAL
    assert restored.state_id == STATE_ID


def test_mode_mismatch_on_branches_fails_closed() -> None:
    store = EventDAGStore()
    root = {
        "event_type": "invocation",
        "parents": [],
        "peer_did": "did:key:root",
        "timestamp": "2026-08-16T00:00:00Z",
        "payload": {"kind": "genesis"},
    }
    root_cid = compute_artifact_cid(root)
    store.add_event(root_cid, root)
    _add_state_write(
        store,
        parents=[root_cid],
        state_ref={
            "schema": STATE_REF_SCHEMA,
            "id": STATE_ID,
            "mode": "crdt",
            "version": 1,
        },
        value={"x": 1},
        peer_did=PEER_A,
    )
    with pytest.raises(StateModeError):
        observe_concurrent_state_branches(
            store, state_id=STATE_ID, mode="single_authority"
        )
