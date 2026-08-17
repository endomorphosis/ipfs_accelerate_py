"""Tests for StateProvider@1 and ImmutableCidState@1 (MCPP-036).

Acceptance:
* Writes are append-only.
* Fetch verifies CID.
* Mutation of an existing CID is rejected.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.artifacts import ArtifactStore
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes
from ipfs_accelerate_py.mcp_server.mcplusplus.state.immutable_cid import (
    IMMUTABLE_CID_STATE_INTERFACE,
    IMMUTABLE_MODE,
    IMMUTABLE_PROVIDER_ID,
    ImmutableCidState,
    canonicalize_payload,
    cid_for_payload,
    create_immutable_cid_state,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.state.provider import (
    ALLOWED_CONSISTENCY_MODES,
    STATE_PROVIDER_INTERFACE,
    STATE_REF_SCHEMA,
    StateIntegrityError,
    StateModeError,
    StateMutationError,
    StateNotFoundError,
    StateProvider,
    StateWriteResult,
    is_portable_cid,
    validate_state_ref,
)


def _payload(value: str = "hello") -> dict:
    return {"schema": "mcp++/test/immutable-value@1", "value": value}


class TestStateProviderContract:
    def test_interface_constants(self) -> None:
        assert STATE_PROVIDER_INTERFACE == "StateProvider@1"
        assert STATE_REF_SCHEMA == "mcp++/state/state-ref@1"
        assert ALLOWED_CONSISTENCY_MODES == {
            "immutable",
            "single_authority",
            "causal",
            "crdt",
            "consensus",
        }

    def test_immutable_provider_is_state_provider(self) -> None:
        provider = create_immutable_cid_state()
        assert isinstance(provider, StateProvider)
        assert provider.mode == IMMUTABLE_MODE
        assert provider.provider_id == IMMUTABLE_PROVIDER_ID
        assert provider.interface == IMMUTABLE_CID_STATE_INTERFACE

    def test_validate_state_ref_requires_mode(self) -> None:
        with pytest.raises(StateModeError):
            validate_state_ref({"schema": STATE_REF_SCHEMA, "id": "state:x"})

    def test_validate_state_ref_rejects_unknown_mode(self) -> None:
        with pytest.raises(StateModeError):
            validate_state_ref(
                {"schema": STATE_REF_SCHEMA, "id": "state:x", "mode": "lww"}
            )

    def test_validate_state_ref_accepts_immutable(self) -> None:
        ref = validate_state_ref(
            {
                "schema": STATE_REF_SCHEMA,
                "id": "state:demo/immutable-config",
                "mode": "immutable",
            }
        )
        assert ref["mode"] == "immutable"
        assert ref["id"] == "state:demo/immutable-config"


class TestAppendOnlyWrites:
    def test_write_stores_new_cid(self) -> None:
        state = ImmutableCidState()
        payload = _payload("alpha")
        result = state.write(payload)

        assert isinstance(result, StateWriteResult)
        assert result.created is True
        assert result.mode == "immutable"
        assert result.provider == IMMUTABLE_PROVIDER_ID
        assert is_portable_cid(result.cid)
        assert result.cid == cid_for_payload(payload)
        assert result.byte_length == len(canonicalize_payload(payload))
        assert state.has(result.cid)

    def test_identical_write_is_idempotent_not_duplicate(self) -> None:
        state = ImmutableCidState()
        payload = _payload("same")
        first = state.write(payload)
        second = state.write(payload)

        assert first.cid == second.cid
        assert first.created is True
        assert second.created is False
        assert state.stats()["block_count"] == 1
        assert state.stats()["idempotent_hits"] == 1

    def test_writes_append_distinct_payloads(self) -> None:
        state = ImmutableCidState()
        a = state.write(_payload("one"))
        b = state.write(_payload("two"))

        assert a.cid != b.cid
        assert a.created and b.created
        assert state.stats()["block_count"] == 2
        assert state.has(a.cid) and state.has(b.cid)

    def test_raw_bytes_write_is_append_only(self) -> None:
        state = ImmutableCidState()
        data = b"raw-immutable-block"
        result = state.put_bytes(data)
        again = state.put_bytes(data)

        assert result.cid == cid_for_bytes(data)
        assert again.created is False
        assert state.fetch(result.cid) == data

    def test_write_with_matching_expected_cid(self) -> None:
        state = ImmutableCidState()
        payload = _payload("expected")
        cid = cid_for_payload(payload)
        result = state.write(payload, expected_cid=cid)
        assert result.cid == cid
        assert result.created is True

    def test_write_with_mismatched_expected_cid_fails(self) -> None:
        state = ImmutableCidState()
        payload = _payload("expected")
        other = cid_for_payload(_payload("other"))
        with pytest.raises(StateIntegrityError):
            state.write(payload, expected_cid=other)


class TestFetchVerifiesCid:
    def test_fetch_returns_bytes_and_verifies(self) -> None:
        state = ImmutableCidState()
        payload = _payload("fetch-me")
        result = state.write(payload)
        data = state.fetch(result.cid)
        assert data == canonicalize_payload(payload)
        assert cid_for_bytes(data) == result.cid

    def test_fetch_json_round_trip(self) -> None:
        state = ImmutableCidState()
        payload = _payload("json-round")
        result = state.write(payload)
        loaded = state.fetch_json(result.cid)
        assert loaded == payload

    def test_fetch_missing_cid_raises(self) -> None:
        state = ImmutableCidState()
        missing = cid_for_bytes(b"does-not-exist")
        with pytest.raises(StateNotFoundError):
            state.fetch(missing)

    def test_fetch_detects_tampered_store(self) -> None:
        blocks: dict[str, bytes] = {}
        state = ImmutableCidState(blocks=blocks)
        result = state.write(_payload("honest"))
        # Simulate external corruption of the underlying block map.
        blocks[result.cid] = b"tampered-bytes-not-matching-cid"
        with pytest.raises(StateIntegrityError):
            state.fetch(result.cid)

    def test_fetch_json_rejects_non_object(self) -> None:
        state = ImmutableCidState()
        result = state.write(b'[1,2,3]')
        with pytest.raises(StateIntegrityError):
            state.fetch_json(result.cid)


class TestMutationRejected:
    def test_cannot_overwrite_existing_cid_with_different_bytes(self) -> None:
        """Direct block collision: same CID key, different content."""
        blocks: dict[str, bytes] = {}
        state = ImmutableCidState(blocks=blocks)
        original = b"original-value"
        cid = cid_for_bytes(original)
        state.put_bytes(original)

        # Force a collision by pre-seating different bytes under that CID then
        # attempting a legitimate write of the original identity — but the
        # store already holds wrong bytes, so a re-write of correct content
        # that addresses to the same CID must still reject mismatched store.
        blocks[cid] = b"foreign-content-under-cid"
        with pytest.raises((StateMutationError, StateIntegrityError)):
            # Content addresses to cid but store holds different bytes.
            # write() sees existing != data → StateMutationError.
            # If caller goes through fetch first, integrity fails.
            state.put_bytes(original)

    def test_mutate_existing_rejects_foreign_payload(self) -> None:
        state = ImmutableCidState()
        result = state.write(_payload("v1"))
        with pytest.raises(StateIntegrityError):
            state.mutate_existing(result.cid, _payload("v2"))

    def test_mutate_existing_idempotent_when_identical(self) -> None:
        state = ImmutableCidState()
        payload = _payload("stable")
        result = state.write(payload)
        again = state.mutate_existing(result.cid, payload)
        assert again.created is False
        assert again.cid == result.cid

    def test_publish_advances_root_without_mutating_prior_cid(self) -> None:
        state = ImmutableCidState()
        first_ref = state.publish("state:demo/cfg", _payload("v1"))
        first_cid = first_ref["root_cid"]
        second_ref = state.publish("state:demo/cfg", _payload("v2"))
        second_cid = second_ref["root_cid"]

        assert first_cid != second_cid
        assert state.fetch_json(first_cid) == _payload("v1")
        assert state.fetch_json(second_cid) == _payload("v2")
        assert second_ref["parents"] == [first_cid]
        assert state.get_ref("state:demo/cfg")["root_cid"] == second_cid

    def test_artifact_store_side_index_is_append_only(self) -> None:
        store = ArtifactStore()
        state = ImmutableCidState(artifact_store=store)
        payload = _payload("side")
        result = state.write(payload)
        assert store.get(result.cid) == payload

        # Inject conflicting dict under the same CID in the side store.
        store.put(result.cid, _payload("conflict"))
        with pytest.raises(StateMutationError):
            # Idempotent path re-checks the artifact store.
            state.write(payload)


class TestStateRefBinding:
    def test_bind_ref_requires_immutable_mode(self) -> None:
        state = ImmutableCidState()
        with pytest.raises(StateModeError):
            state.bind_ref(
                {
                    "schema": STATE_REF_SCHEMA,
                    "id": "state:x",
                    "mode": "single_authority",
                }
            )

    def test_bind_ref_verifies_root_cid(self) -> None:
        state = ImmutableCidState()
        result = state.write(_payload("rooted"))
        bound = state.bind_ref(
            {
                "schema": STATE_REF_SCHEMA,
                "id": "state:demo/immutable-config",
                "mode": "immutable",
                "root_cid": result.cid,
            }
        )
        assert bound["root_cid"] == result.cid
        assert bound["provider"] == IMMUTABLE_PROVIDER_ID
        assert bound["authority"] == {"kind": "none"}
        assert state.get_ref("state:demo/immutable-config")["root_cid"] == result.cid

    def test_bind_ref_missing_root_fails(self) -> None:
        state = ImmutableCidState()
        missing = cid_for_bytes(b"absent")
        with pytest.raises(StateNotFoundError):
            state.bind_ref(
                {
                    "schema": STATE_REF_SCHEMA,
                    "id": "state:missing",
                    "mode": "immutable",
                    "root_cid": missing,
                }
            )

    def test_open_ref_alias(self) -> None:
        state = ImmutableCidState()
        result = state.write(_payload("open"))
        opened = state.open_ref(
            {
                "id": "state:open",
                "mode": "immutable",
                "root_cid": result.cid,
            }
        )
        assert opened["schema"] == STATE_REF_SCHEMA
        assert opened["mode"] == "immutable"


class TestFactoryAndStats:
    def test_factory_creates_isolated_providers(self) -> None:
        a = create_immutable_cid_state()
        b = create_immutable_cid_state()
        cid = a.write(_payload("only-a")).cid
        assert a.has(cid)
        assert not b.has(cid)

    def test_stats_surface(self) -> None:
        state = create_immutable_cid_state(artifact_store=ArtifactStore())
        state.write(_payload("s"))
        stats = state.stats()
        assert stats["mode"] == "immutable"
        assert stats["block_count"] == 1
        assert stats["artifact_store_attached"] is True
        assert stats["interface"] == IMMUTABLE_CID_STATE_INTERFACE

    def test_export_blocks_is_defensive_copy(self) -> None:
        state = ImmutableCidState()
        result = state.write(b"export-me")
        exported = state.export_blocks()
        exported[result.cid] = b"mutated"
        assert state.fetch(result.cid) == b"export-me"
