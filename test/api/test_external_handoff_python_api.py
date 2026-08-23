"""Deterministic tests for the EAAEF-110 in-process Python handoff API."""

from __future__ import annotations

import inspect

import pytest

from ipfs_accelerate_py.agent_supervisor import api as handoff_api
from ipfs_accelerate_py.agent_supervisor.api.external_handoff import (
    HANDOFF_API_OPERATIONS,
    OPERATIONS,
    ExternalHandoffAPI,
    ExternalHandoffAPIError,
    ExternalHandoffAuthorityError,
    ExternalHandoffReceipt,
    ExternalHandoffRequest,
    HandoffApiVerdict,
    WorkerSelfApprovalError,
    discover_external_handoff_api,
    export_result,
)


OPERATOR = "principal:operator"
WORKER = "principal:worker"
REVIEWER = "principal:reviewer"
SESSION = "session:example"
REPO = "repo:example"


def _start_request(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "principal_id": OPERATOR,
        "worker_principal_id": WORKER,
        "session_id": SESSION,
        "repository_id": REPO,
        "objective_id": "objective:handoff",
        "idempotency_key": "idem:start-1",
    }
    values.update(changes)
    return values


def _control_request(started: ExternalHandoffReceipt, **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "principal_id": OPERATOR,
        "worker_principal_id": WORKER,
        "run_id": started.run_id,
        "authority_id": started.authority_id,
        "session_id": SESSION,
    }
    values.update(changes)
    return values


def _assert_identities(receipt: ExternalHandoffReceipt, *, operation: str) -> None:
    assert receipt.operation == operation
    assert receipt.status == "ok"
    assert receipt.content_id
    assert receipt.receipt_id == receipt.content_id
    assert receipt.request_id
    assert receipt.run_id
    assert receipt.authority_id
    assert receipt.content_id != receipt.request_id
    assert receipt.run_id != receipt.authority_id
    assert receipt.identities["run_id"] == receipt.run_id
    assert receipt.identities["request_id"] == receipt.request_id
    assert receipt.identities["authority_id"] == receipt.authority_id
    restored = ExternalHandoffReceipt.from_dict(receipt.to_dict())
    assert restored.content_id == receipt.content_id


@pytest.mark.parametrize("operation", HANDOFF_API_OPERATIONS)
def test_each_operation_exists_on_module_and_api(operation: str) -> None:
    assert operation in OPERATIONS
    assert callable(getattr(handoff_api, operation))
    assert callable(getattr(ExternalHandoffAPI, operation))
    assert callable(getattr(ExternalHandoffAPI(), operation))


@pytest.mark.parametrize("operation", HANDOFF_API_OPERATIONS)
def test_each_operation_returns_canonical_identities(operation: str) -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(_start_request())
    bound = _control_request(started)
    if operation == "handoff":
        receipt = started
    elif operation == "preview":
        receipt = api.preview(_start_request(idempotency_key="idem:preview-1", session_id="session:preview"))
    elif operation == "steer":
        receipt = api.steer({**bound, "instruction": "narrow the owned-file patch"})
    elif operation == "pause":
        receipt = api.pause(bound)
    elif operation == "resume":
        api.pause(bound)
        receipt = api.resume(bound)
    elif operation == "approve":
        receipt = api.approve({**bound, "reviewer_principal_id": REVIEWER})
    elif operation == "reject":
        receipt = api.reject({**bound, "reviewer_principal_id": REVIEWER})
    elif operation == "export":
        receipt = api.export(bound)
    else:
        receipt = getattr(api, operation)(bound)
    _assert_identities(receipt, operation=operation)


def test_preview_is_distinct_from_handoff() -> None:
    api = ExternalHandoffAPI()
    shared = _start_request(idempotency_key="idem:shared")
    admitted = api.handoff(shared)
    previewed = api.preview(shared)
    assert admitted.operation == "handoff"
    assert previewed.operation == "preview"
    assert admitted.verdict == HandoffApiVerdict.ADMITTED.value
    assert previewed.verdict == HandoffApiVerdict.PREVIEW_ONLY.value
    assert admitted.run_status == "running"
    assert previewed.run_status == "preview_only"
    assert admitted.run_id != previewed.run_id
    assert admitted.content_id != previewed.content_id
    assert admitted.request_id != previewed.request_id
    assert admitted.authority_id != previewed.authority_id


def test_approve_rejects_worker_self_approval() -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(_start_request())
    bound = _control_request(started)
    with pytest.raises(WorkerSelfApprovalError, match="self-approval") as worker_reviewer:
        api.approve({**bound, "reviewer_principal_id": WORKER})
    assert worker_reviewer.value.reason_code == "worker_self_approval"
    with pytest.raises(WorkerSelfApprovalError, match="self-approval") as worker_caller:
        api.approve(
            {
                **bound,
                "principal_id": WORKER,
                "reviewer_principal_id": REVIEWER,
            }
        )
    assert worker_caller.value.reason_code == "worker_self_approval"
    with pytest.raises(WorkerSelfApprovalError, match="independent reviewer") as missing:
        api.approve(bound)
    assert missing.value.reason_code == "missing_reviewer"


def test_independent_reviewer_may_approve() -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(_start_request())
    receipt = api.approve(
        _control_request(started, reviewer_principal_id=REVIEWER)
    )
    _assert_identities(receipt, operation="approve")
    assert receipt.run_status == "approved"
    assert receipt.reviewer_principal_id == REVIEWER
    assert receipt.worker_principal_id == WORKER
    assert receipt.reviewer_principal_id != receipt.worker_principal_id


@pytest.mark.parametrize("operation", ("cancel", "pause", "resume", "steer"))
def test_control_ops_require_matching_run_and_authority(operation: str) -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(_start_request())
    other = api.handoff(_start_request(idempotency_key="idem:other", session_id="session:other"))
    payload: dict[str, object] = _control_request(started)
    if operation == "steer":
        payload["instruction"] = "keep owned files only"
    if operation == "resume":
        api.pause(_control_request(started))

    with pytest.raises(ExternalHandoffAPIError, match="run_id") as missing_run:
        getattr(api, operation)({**payload, "run_id": ""})
    assert missing_run.value.reason_code == "malformed"

    with pytest.raises(ExternalHandoffAuthorityError) as missing_authority:
        getattr(api, operation)({**payload, "authority_id": ""})
    assert missing_authority.value.reason_code == "authority_mismatch"

    with pytest.raises(ExternalHandoffAPIError, match="unknown run") as unknown_run:
        getattr(api, operation)({**payload, "run_id": other.authority_id})
    assert unknown_run.value.reason_code == "unknown_run"

    with pytest.raises(ExternalHandoffAuthorityError, match="authority id") as mismatch:
        getattr(api, operation)({**payload, "authority_id": other.authority_id})
    assert mismatch.value.reason_code == "authority_mismatch"

    receipt = getattr(api, operation)(payload)
    _assert_identities(receipt, operation=operation)
    assert receipt.run_id == started.run_id
    assert receipt.authority_id == started.authority_id


def test_request_dict_and_dataclass_are_accepted() -> None:
    api = ExternalHandoffAPI()
    as_dict = api.handoff(_start_request())
    as_dataclass = api.preview(
        ExternalHandoffRequest(
            operation="preview",
            principal_id=OPERATOR,
            worker_principal_id=WORKER,
            session_id="session:dataclass",
            repository_id=REPO,
            idempotency_key="idem:dataclass",
        )
    )
    _assert_identities(as_dict, operation="handoff")
    _assert_identities(as_dataclass, operation="preview")
    assert as_dict.verdict != as_dataclass.verdict


def test_unknown_operation_fails_closed() -> None:
    with pytest.raises(ExternalHandoffAPIError, match="unknown") as err:
        ExternalHandoffRequest(
            operation="mutate_production",
            principal_id=OPERATOR,
        )
    assert err.value.reason_code == "unknown_operation"
    with pytest.raises(ExternalHandoffAPIError, match="unsupported fields"):
        ExternalHandoffRequest.from_dict(
            {
                "operation": "status",
                "principal_id": OPERATOR,
                "docker_socket": "/var/run/docker.sock",
            }
        )


def test_export_result_alias_and_catalog_are_in_process() -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(_start_request())
    exported = export_result(_control_request(started), api=api)
    _assert_identities(exported, operation="export")
    assert exported.export_id
    assert exported.identities["export_id"] == exported.export_id
    catalog = discover_external_handoff_api()
    assert catalog["operations"] == list(HANDOFF_API_OPERATIONS)
    assert catalog["preview_is_handoff"] is False
    assert catalog["self_approval"] is False
    assert catalog["live_quack"] is False
    assert catalog["live_docker"] is False
    source = inspect.getsource(inspect.getmodule(ExternalHandoffAPI))
    assert "import docker" not in source
    assert "import quack" not in source
    assert "from docker" not in source
    assert "from quack" not in source
