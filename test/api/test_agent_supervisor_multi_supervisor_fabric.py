"""Focused LGSWF-062 multi-supervisor fabric checks."""

from __future__ import annotations

import pytest

import importlib.util
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric import issue_fence
from ipfs_accelerate_py.agent_supervisor.runtime.work_partitioning import partition_tasks

_OPS = (
    Path(__file__).resolve().parents[2]
    / "scripts/ops/agent_supervisor/configured_board_scheduler.py"
)
_SPEC = importlib.util.spec_from_file_location("lgswf_ops_configured_board", _OPS)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
lgswf_coordinate_fabric = _MOD.lgswf_coordinate_fabric


def _ok(**overrides):
    payload = {
        "quack_endpoint": "quack://state-owner",
        "state_server_identity": "StateServerIdentity@1",
        "capability": "quack-state-owner",
        "remote_ready": True,
        "start_count": 1,
        "stop_count": 1,
        "partitions": partition_tasks(["T1", "T2"], ["S1", "S2"]),
        "packets": ("P1",),
        "epoch": 2,
        "result_key": "k1",
        "result": "accepted",
    }
    payload.update(overrides)
    return payload


def test_propagates_endpoint_identity_and_one_logical_result() -> None:
    fence = issue_fence({"supervisor_id": "S1", "capability": "dispatch", "epoch": 2})
    result = lgswf_coordinate_fabric(_ok())
    assert result["endpoint"] == "quack://state-owner"
    assert result["state_server_identity"] == "StateServerIdentity@1"
    assert result["started_once"] is True
    assert result["stopped_once"] is True
    assert result["local_file_readiness_authoritative"] is False
    assert result["logical_results"] == {"k1": "accepted"}
    assert result["multiprocess_mutation"] is False
    assert fence["fenced"] is True


def test_missing_capability_and_local_readiness_fail_closed() -> None:
    with pytest.raises(ValueError, match="capability"):
        lgswf_coordinate_fabric(_ok(capability="duckdb-file"))
    with pytest.raises(ValueError, match="local file"):
        lgswf_coordinate_fabric(_ok(remote_ready=False, local_file_readiness=True))
    with pytest.raises(ValueError, match="LGSWF-072"):
        lgswf_coordinate_fabric(_ok(production_multiprocess_mutation=True))
