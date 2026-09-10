"""ASEH-033: current minimal pack selection and exact stale-identity rejection."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
    CONTEXT_PACK_FRESHNESS_INTERFACE,
    DATASETS_CONTEXT_PACK_AUTHORITY,
    EXACT_FRESHNESS_FIELDS,
    KIT_CONTEXT_PACK_STORE_AUTHORITY,
    ContextPackError,
    ContextPacker,
    CurrentPackIdentity,
    StaleIdentityError,
    encode_context_pack_envelope,
    evaluate_exact_freshness,
    load_datasets_context_pack_authority,
    verify_datasets_semantic_identity,
    verify_kit_bytes,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack_selector import (
    CONTEXT_PACK_SELECTOR_INTERFACE,
    ContextPackSelector,
    apply_installed_kit_contract_vectors,
    installed_datasets_context_pack_schema,
    record_reuse_or_invalidation,
    select_current_minimal_pack,
    select_minimal_adequate_pack,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.proof_context.context_pack import (
    INTERFACE as DATASETS_INTERFACE,
    build_context_pack,
    build_minimal_semantic_pack,
)
from ipfs_kit_py.proof_context.state_store import open_context_pack_store


TREE = "16ef68abe8a35a3033dfaf1ed4e8d6132600df8f"
STALE_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OBJECTIVE = "ASEH-033"
OBJECTIVE_REVISION = "baguqeerazrumjlclkl2morwckg4j7jrk2txy324iqxb6p3onhaezvwmvftza"
TOOLCHAIN = "python3.12"
ENVIRONMENT = "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
SCHEMA_IDENTITY = "ipfs-datasets.proof-context.context-pack@0.1"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _freshness_bindings(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "file_and_symbol_identities": [],
        "schema_identities": [SCHEMA_IDENTITY],
        "toolchain_identities": [TOOLCHAIN],
        "environment_requirements": [ENVIRONMENT],
        "reusable_until_conditions": ["tree-unchanged"],
    }
    payload.update(overrides)
    return payload


def _helper_dep() -> dict[str, object]:
    return {
        "symbol": "helper",
        "cid": _cid("helper"),
        "path": "helper.py",
        "meaning": "pure helper used by target",
    }


def _build_minimal(**overrides: object) -> Any:
    fields: dict[str, object] = {
        "repository_state_cid": _cid("repo-state"),
        "task_id": "ASEH-033",
        "target_source_cid": _cid("target"),
        "surrounding_source_cid": _cid("surround"),
        "test_source_cid": _cid("test"),
        "scanned_tree_oid": TREE,
        "source_tree_oid": TREE,
        "dependencies": [_helper_dep()],
        "objective_identity": OBJECTIVE,
        "objective_revision": OBJECTIVE_REVISION,
        "policy_identity": _cid("aseh-033-policy"),
        "freshness_bindings": _freshness_bindings(),
        "invalidation": {
            "invalidation_triggers": ["tree-changed", "policy-changed"],
            "reusable_until_conditions": ["tree-unchanged"],
        },
    }
    fields.update(overrides)
    return build_minimal_semantic_pack(**fields)


def _build_bloated() -> Any:
    helper = _cid("helper")
    unused = _cid("unused-whole-repo")
    return build_context_pack(
        repository_state_cid=_cid("repo-state"),
        task_id="ASEH-033",
        target_source_cid=_cid("target"),
        surrounding_source_cid=_cid("surround"),
        test_source_cid=_cid("test"),
        scanned_tree_oid=TREE,
        source_tree_oid=TREE,
        capsule_cids=(helper, unused),
        objective_identity=OBJECTIVE,
        objective_revision=OBJECTIVE_REVISION,
        policy_identity=_cid("aseh-033-policy"),
        freshness_bindings=_freshness_bindings(),
        invalidation={
            "invalidation_triggers": ["tree-changed", "policy-changed"],
            "reusable_until_conditions": ["tree-unchanged"],
        },
    )


def _store_pack(store: Any, record: Any, *, current: bool = False, cache_key: str | None = None):
    data = encode_context_pack_envelope(record.to_dict())
    if cache_key is not None:
        reference = store.put_candidate(data, cache_key=cache_key)
    else:
        reference = store.put_verified_bytes(data)
    if current:
        pointer = store.current_root()
        if pointer is None:
            store.compare_and_swap_current_root(new_cid=reference.cid, generation=0)
        else:
            store.compare_and_swap_current_root(
                new_cid=reference.cid,
                expected_parent_cid=pointer.seal_cid,
                generation=pointer.generation + 1,
                kind=reference.kind,
            )
    return reference, data


def test_installed_packages_and_immutable_vectors_are_the_authorities() -> None:
    authority = load_datasets_context_pack_authority()
    assert authority.producer == DATASETS_CONTEXT_PACK_AUTHORITY
    assert authority.interface == DATASETS_INTERFACE
    assert ContextPacker.V01_PRODUCTION_AUTHORITY is False
    assert ContextPackSelector.V01_PRODUCTION_AUTHORITY is False
    assert ContextPackSelector.DATASETS_AUTHORITY == DATASETS_CONTEXT_PACK_AUTHORITY
    assert ContextPackSelector.KIT_AUTHORITY == KIT_CONTEXT_PACK_STORE_AUTHORITY
    assert ContextPackSelector.FRESHNESS_FIELDS == EXACT_FRESHNESS_FIELDS
    schema = installed_datasets_context_pack_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["interface"]["const"] == DATASETS_INTERFACE
    results = apply_installed_kit_contract_vectors()
    assert results
    assert any(item["status"] == "accept" for item in results)
    assert any(item["status"] == "reject" for item in results)
    selector_source = inspect.getsource(
        inspect.getmodule(select_current_minimal_pack)
    )
    pack_source = inspect.getsource(inspect.getmodule(verify_datasets_semantic_identity))
    for source in (selector_source, pack_source):
        assert "tests.proof_context" not in source
        assert "tests.test_context_pack_store" not in source
        assert "tests/proof_context" not in source


def test_current_minimal_pack_is_selected_and_reuse_is_recorded(tmp_path) -> None:
    store = open_context_pack_store(tmp_path)
    minimal = _build_minimal()
    bloated = _build_bloated()
    assert len(minimal.capsule_cids) < len(bloated.capsule_cids)
    current = CurrentPackIdentity.from_envelope(minimal.to_dict())
    _store_pack(store, bloated, cache_key="pack:bloated")
    minimal_ref, _ = _store_pack(
        store, minimal, current=True, cache_key="pack:minimal"
    )
    selection = select_current_minimal_pack(store, current)
    assert selection.selected is not None
    assert selection.selected.datasets_pack_cid == minimal.pack_cid
    assert selection.selected.kit_cid == minimal_ref.cid
    assert selection.selected.is_current_root is True
    assert selection.admission.reused is True
    assert selection.admission.disposition == "reuse"
    assert selection.admission.datasets_pack_cid == minimal.pack_cid
    assert selection.admission.kit_cid == minimal_ref.cid
    record = record_reuse_or_invalidation(selection)
    assert record.record_cid == selection.admission.record_cid
    assert record.datasets_pack_cid == minimal.pack_cid
    store.close()


def test_selects_minimal_adequate_pack_over_bloated_current_root(tmp_path) -> None:
    store = open_context_pack_store(tmp_path)
    minimal = _build_minimal()
    bloated = _build_bloated()
    current = CurrentPackIdentity.from_envelope(minimal.to_dict())
    bloated_ref, _ = _store_pack(store, bloated, current=True, cache_key="pack:bloated")
    minimal_ref, _ = _store_pack(store, minimal, cache_key="pack:minimal")
    selection = select_minimal_adequate_pack(store, current)
    assert selection.selected is not None
    assert selection.selected.datasets_pack_cid == minimal.pack_cid
    assert selection.selected.kit_cid == minimal_ref.cid
    assert selection.selected.is_current_root is False
    assert selection.admission.reused is False
    assert selection.admission.disposition == "selected"
    assert selection.admission.current_root_cid == bloated_ref.cid
    assert selection.admission.capsule_count == len(minimal.capsule_cids)
    store.close()


def test_datasets_identity_is_verified_and_never_reminted(tmp_path) -> None:
    store = open_context_pack_store(tmp_path)
    record = _build_minimal()
    envelope = record.to_dict()
    verified = verify_datasets_semantic_identity(envelope)
    assert verified["pack_cid"] == record.pack_cid
    assert verified["producer"] == DATASETS_CONTEXT_PACK_AUTHORITY
    _store_pack(store, record, current=True)
    selection = select_current_minimal_pack(
        store, CurrentPackIdentity.from_envelope(envelope)
    )
    assert selection.selected is not None
    assert selection.selected.datasets_pack_cid == record.pack_cid
    assert selection.selected.kit_cid != record.pack_cid
    forged = dict(envelope)
    forged["pack_cid"] = _cid("forged-pack")
    with pytest.raises(Exception):
        verify_datasets_semantic_identity(forged)
    store.close()


def test_kit_bytes_and_root_are_verified_independently_of_semantics(tmp_path) -> None:
    store = open_context_pack_store(tmp_path)
    record = _build_minimal()
    reference, data = _store_pack(store, record, current=True)
    assert verify_kit_bytes(store, data, claimed_cid=reference.cid) == reference.cid
    with pytest.raises(ContextPackError, match="kit CID"):
        verify_kit_bytes(store, data, claimed_cid=_cid("other-bytes"))
    masquerade = open_context_pack_store(tmp_path / "masquerade")
    opaque = masquerade.put_verified_bytes(b'{"storage":"is-not-semantic-proof"}')
    masquerade.compare_and_swap_current_root(new_cid=opaque.cid, generation=0)
    current = CurrentPackIdentity.from_envelope(record.to_dict())
    with pytest.raises(ContextPackError, match="semantic"):
        select_current_minimal_pack(masquerade, current)
    assert store.current_root().seal_cid == reference.cid
    store.close()
    masquerade.close()


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("tree", {"scanned_tree_oid": STALE_TREE, "source_tree_oid": STALE_TREE}),
        ("objective", {"objective_revision": "stale-objective-revision"}),
        ("policy", {"policy_identity": _cid("stale-policy")}),
        (
            "interface",
            {
                "freshness_bindings": _freshness_bindings(
                    schema_identities=["stale.interface@1"]
                )
            },
        ),
        (
            "toolchain",
            {"freshness_bindings": _freshness_bindings(toolchain_identities=["pypy"])},
        ),
        (
            "environment",
            {
                "freshness_bindings": _freshness_bindings(
                    environment_requirements=["PATH=/tmp/user-writable"]
                )
            },
        ),
    ],
)
def test_exact_stale_identity_rejection(tmp_path, field: str, overrides: dict[str, object]) -> None:
    store = open_context_pack_store(tmp_path)
    fresh = _build_minimal()
    stale = _build_minimal(**overrides)
    current = CurrentPackIdentity.from_envelope(fresh.to_dict())
    verdict = evaluate_exact_freshness(stale.to_dict(), current)
    assert verdict.fresh is False
    assert field in verdict.stale_fields
    _store_pack(store, stale, current=True, cache_key=f"pack:stale-{field}")
    with pytest.raises(StaleIdentityError) as excinfo:
        select_current_minimal_pack(store, current)
    assert field in excinfo.value.stale_fields
    recorded = select_current_minimal_pack(
        store, current, require_selection=False
    )
    assert recorded.selected is None
    assert recorded.admission.disposition == "rejected"
    assert recorded.admission.reused is False
    assert field in recorded.admission.stale_fields
    assert any(item.stale_fields for item in recorded.invalidated)
    store.close()


def test_fixture_pack_cannot_masquerade_as_live(tmp_path) -> None:
    store = open_context_pack_store(tmp_path)
    live = _build_minimal()
    fixture = _build_minimal(
        identity_kind="fixture",
        evidence_kind="fixture",
        execution_mode="simulated",
    )
    current = CurrentPackIdentity.from_envelope(live.to_dict())
    _store_pack(store, fixture, current=True, cache_key="pack:fixture")
    with pytest.raises(StaleIdentityError):
        select_current_minimal_pack(store, current)
    recorded = select_current_minimal_pack(
        store, current, require_selection=False
    )
    assert recorded.selected is None
    assert recorded.invalidated
    assert "fixture_as_live" in recorded.invalidated[0].masquerade_reasons
    store.close()


def test_missing_evidence_is_not_adequate_and_is_not_reused(tmp_path) -> None:
    store = open_context_pack_store(tmp_path)
    incomplete = _build_minimal(missing_evidence=["named-missing-contract"])
    current = CurrentPackIdentity.from_envelope(incomplete.to_dict())
    _store_pack(store, incomplete, current=True, cache_key="pack:incomplete")
    with pytest.raises(StaleIdentityError, match="no current minimal adequate"):
        select_current_minimal_pack(store, current)
    recorded = select_current_minimal_pack(
        store, current, require_selection=False
    )
    assert recorded.selected is None
    assert recorded.admission.reused is False
    assert "inadequate_coverage" in recorded.inspected[0].invalidation_reasons
    store.close()


def test_selector_exposes_freshness_interface_and_does_not_claim_construction() -> None:
    selector = ContextPackSelector()
    assert selector.INTERFACE == CONTEXT_PACK_SELECTOR_INTERFACE
    assert selector.V01_PRODUCTION_AUTHORITY is False
    assert CONTEXT_PACK_FRESHNESS_INTERFACE.endswith("FreshnessAdmission@1")
    assert EXACT_FRESHNESS_FIELDS == (
        "tree",
        "objective",
        "policy",
        "interface",
        "toolchain",
        "environment",
    )
