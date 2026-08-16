"""Focused SemanticWorkBinding@1 checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.work_binding import (
    INTERFACE,
    SCHEMA,
    WorkBindingError,
    example_binding,
    field_ownership_table,
    parse_work_binding,
)


def test_binding_roundtrip_and_identity() -> None:
    first = example_binding()
    second = parse_work_binding(dict(first))
    assert first["schema"] == SCHEMA
    assert first["interface"] == INTERFACE
    assert first["binding_cid"] == second["binding_cid"]
    assert first["target_capsule_cids"]


def test_stale_mixed_and_forged_references_rejected() -> None:
    payload = dict(example_binding())
    payload["accepted_plan_cid"] = "ACCEPTED_LGSWF-006_SOURCE_HEAD"
    payload.pop("binding_cid", None)
    with pytest.raises(WorkBindingError, match="sentinel"):
        parse_work_binding(payload)
    payload = dict(example_binding())
    payload["target_capsule_cids"] = ("forged",)
    payload.pop("binding_cid", None)
    with pytest.raises(WorkBindingError, match="forged"):
        parse_work_binding(payload)
    payload = dict(example_binding())
    payload["capsule_body"] = {"copied": True}
    payload.pop("binding_cid", None)
    with pytest.raises(WorkBindingError, match="must not copy"):
        parse_work_binding(payload)


def test_ownership_table_and_absent_facts_need_typed_hold() -> None:
    table = field_ownership_table()
    assert any(row["field"] == "semantic_state_root_cid" for row in table)
    payload = dict(example_binding())
    payload["raw_source_requirements"] = None
    payload.pop("binding_cid", None)
    with pytest.raises(WorkBindingError, match="fabricated"):
        parse_work_binding(payload)
