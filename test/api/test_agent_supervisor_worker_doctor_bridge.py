"""WPD-030: Worker Doctor bridge from validation failures.

Acceptance (from the sealed WPD board):

* Known failure classes produce Doctor inspect requests with exact roots.
* Unknown failure classes yield ``abstain_review``.
* The bridge never opens an LLM / network / remote model-provider surface.
* Evidence subset: mapping table, no network, typed abstention.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
    DoctorMode,
    DoctorOperation,
)
from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service import (
    DoctorOperationRequest,
    assert_no_llm_surface_loaded,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    ImplementationDisposition,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.worker_doctor_bridge import (
    KNOWN_FAILURE_CLASS_MAP,
    REASON_KNOWN_FAILURE_INSPECT,
    REASON_KNOWN_FAILURE_PLAN,
    REASON_MISSING_ROOTS,
    REASON_UNKNOWN_FAILURE_ABSTAIN,
    WORKER_DOCTOR_BRIDGE_EVIDENCE,
    WORKER_DOCTOR_BRIDGE_INTERFACE,
    WORKER_DOCTOR_BRIDGE_VERSION,
    WorkerDoctorBridge,
    WorkerDoctorBridgeInputError,
    WorkerFailureClass,
    WorkerFailureRecord,
    build_worker_doctor_bridge,
    failure_class_mapping_table,
    known_failure_classes,
    map_worker_failure,
    normalize_failure_class,
)


def _roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "file_root_id": "file-root:fixture",
        "ast_root_id": "ast:fixture",
        "graph_id": "graph:fixture",
        "corpus_id": "corpus:fixture",
        "index_id": "index:fixture",
        "model_id": "model:fixture",
        "cache_id": "cache:fixture",
        "operator_registry_id": "operators:fixture",
        "translator_id": "translator:fixture",
        "solver_id": "solver:fixture",
        "kernel_id": "kernel:fixture",
        "toolchain_id": "toolchain:fixture",
        "policy_id": "policy:fixture",
        "sandbox_id": "sandbox:fixture",
        "environment_id": "environment:fixture",
        "lease_id": "lease:fixture",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def _failure(
    failure_class: str,
    *,
    roots: DoctorAuthorityRoots | None = None,
    **changes: Any,
) -> WorkerFailureRecord:
    values: dict[str, Any] = {
        "failure_class": failure_class,
        "task_cid": "task:fixture-1",
        "roots": roots if roots is not None else _roots(),
        "write_paths": ("pkg/module.py",),
        "finding_ids": ("finding:1",),
        "reason_codes": ("pytest_exit_1",),
        "attempt": 1,
    }
    values.update(changes)
    return WorkerFailureRecord(**values)


# ---------------------------------------------------------------------------
# Interface / cold import / mapping table evidence
# ---------------------------------------------------------------------------


def test_interface_and_evidence_identity_are_stable() -> None:
    assert WORKER_DOCTOR_BRIDGE_INTERFACE == "WorkerDoctorBridge@1"
    assert WORKER_DOCTOR_BRIDGE_VERSION == 1
    assert WORKER_DOCTOR_BRIDGE_EVIDENCE == "wpd/worker-doctor-bridge@1"
    discovery = WorkerDoctorBridge.discovery()
    assert discovery["interface"] == WORKER_DOCTOR_BRIDGE_INTERFACE
    assert discovery["evidence_key"] == WORKER_DOCTOR_BRIDGE_EVIDENCE
    assert discovery["llm_router_enabled"] is False
    assert discovery["automatic_fallback"] is False
    assert discovery["network_access"] is False
    assert discovery["provider_hooks"] == 0
    assert discovery["unknown_disposition"] == "abstain_review"
    assert "validation_failure" in discovery["known_failure_classes"]
    assert discovery["mapping_table"]["validation_failure"] == "inspect"
    assert discovery["mapping_table"]["scope_failure"] == "inspect"
    assert discovery["mapping_table"]["proof_failure"] == "inspect"


def test_mapping_table_covers_closed_vocabulary() -> None:
    table = failure_class_mapping_table()
    assert set(table) == known_failure_classes()
    assert set(table) == {item.value for item in WorkerFailureClass}
    for failure_class, operation in table.items():
        assert operation == "inspect"
        assert KNOWN_FAILURE_CLASS_MAP[WorkerFailureClass(failure_class)] is (
            DoctorOperation.INSPECT
        )


def test_cold_import_does_not_load_llm_or_network_clients() -> None:
    # Substrings must not appear as import targets.  English plurals such as
    # "Doctor operation records" are fine; bare ``import requests`` is not.
    llm_import_roots = (
        "openai",
        "anthropic",
        "litellm",
        "groq",
        "together",
        "requests",
        "httpx",
        "aiohttp",
    )
    import importlib
    import re

    importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.worker_doctor_bridge"
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        worker_doctor_bridge as mod,
    )

    source = inspect.getsource(mod)
    for root in llm_import_roots:
        # Reject ``import root``, ``from root ...``, and dotted submodules.
        pattern = re.compile(
            rf"(?m)^\s*(?:import|from)\s+{re.escape(root)}(?:\.|\s|$)"
        )
        assert pattern.search(source) is None, f"forbidden import of {root!r}"
        # Also reject the bare module token as a non-comment identifier import
        # alias (covers ``importlib.import_module("openai")`` style loads).
        assert f'"{root}"' not in source
        assert f"'{root}'" not in source
    assert_no_llm_surface_loaded()


# ---------------------------------------------------------------------------
# Known failure classes → Doctor inspect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure_class",
    [
        "validation_failure",
        "scope_failure",
        "proof_failure",
        "merge_failure",
        "merge_admission_failure",
        "path_escape",
        "exact_roots_mismatch",
        "contract_gap",
    ],
)
def test_known_failure_classes_produce_doctor_inspect(failure_class: str) -> None:
    roots = _roots()
    result = map_worker_failure(_failure(failure_class, roots=roots))

    assert result.known is True
    assert result.mapping_table_hit is True
    assert result.provider_hook_count == 0
    assert result.authorizes_provider is False
    assert result.disposition is None
    assert result.reason_code == REASON_KNOWN_FAILURE_INSPECT
    assert result.produces_doctor_inspect is True
    assert result.doctor_operation == DoctorOperation.INSPECT.value

    request = result.doctor_request
    assert isinstance(request, DoctorOperationRequest)
    assert request.operation == DoctorOperation.INSPECT.value
    assert request.mode is DoctorMode.REPORT_ONLY
    assert request.roots is not None
    assert request.roots.content_id == roots.content_id
    assert request.llm_router_invoked is False
    assert request.remote_model_provider_invoked is False
    assert request.network_access is False
    assert request.model_invocation_count == 0
    assert request.provider_invocation_count == 0
    assert failure_class in request.reason_codes
    assert "pkg/module.py" in request.write_paths
    assert "finding:1" in request.finding_ids


def test_failure_class_aliases_normalize_to_known() -> None:
    assert normalize_failure_class("validation") is WorkerFailureClass.VALIDATION_FAILURE
    assert normalize_failure_class("scope_escape") is WorkerFailureClass.SCOPE_FAILURE
    assert normalize_failure_class("proof") is WorkerFailureClass.PROOF_FAILURE
    result = map_worker_failure(_failure("validation"))
    assert result.known is True
    assert result.failure_class == "validation_failure"
    assert result.produces_doctor_inspect is True


def test_plan_override_for_eligible_known_class() -> None:
    result = map_worker_failure(
        _failure("validation_failure"),
        operation="plan",
    )
    assert result.known is True
    assert result.reason_code == REASON_KNOWN_FAILURE_PLAN
    assert result.doctor_operation == DoctorOperation.PLAN.value
    assert result.doctor_request is not None
    assert result.doctor_request.operation == DoctorOperation.PLAN.value
    assert result.doctor_request.mode is DoctorMode.PLAN
    assert result.provider_hook_count == 0


def test_plan_override_rejected_for_non_eligible_class() -> None:
    with pytest.raises(WorkerDoctorBridgeInputError, match="not plan-eligible"):
        map_worker_failure(_failure("path_escape"), operation="plan")


def test_repair_operation_not_admitted() -> None:
    with pytest.raises(WorkerDoctorBridgeInputError, match="inspect or plan"):
        map_worker_failure(_failure("validation_failure"), operation="repair")


def test_mapping_binds_exact_roots_and_lease() -> None:
    roots = _roots(lease_id="lease:exact-1", tree_id="tree:exact-1")
    result = map_worker_failure(
        _failure(
            "proof_failure",
            roots=roots,
            lease_id="",
            write_paths=("src/a.py", "src/b.py"),
        )
    )
    request = result.doctor_request
    assert request is not None
    assert request.roots is not None
    assert request.roots.tree_id == "tree:exact-1"
    assert request.lease_id == "lease:exact-1"
    assert set(request.write_paths) == {"src/a.py", "src/b.py"}


# ---------------------------------------------------------------------------
# Unknown → abstain_review
# ---------------------------------------------------------------------------


def test_unknown_failure_class_abstains_review() -> None:
    result = map_worker_failure(_failure("totally_unknown_widget_failure"))

    assert result.known is False
    assert result.mapping_table_hit is False
    assert result.disposition is ImplementationDisposition.ABSTAIN_REVIEW
    assert result.abstained is True
    assert result.reason_code == REASON_UNKNOWN_FAILURE_ABSTAIN
    assert result.doctor_request is None
    assert result.doctor_operation == ""
    assert result.provider_hook_count == 0
    assert result.authorizes_provider is False
    assert result.produces_doctor_inspect is False


def test_known_class_without_exact_roots_abstains() -> None:
    result = map_worker_failure(
        WorkerFailureRecord(
            failure_class="validation_failure",
            task_cid="task:no-roots",
            roots=None,
        )
    )
    assert result.known is False
    assert result.disposition is ImplementationDisposition.ABSTAIN_REVIEW
    assert result.reason_code == REASON_MISSING_ROOTS
    assert result.doctor_request is None
    assert result.mapping_table_hit is True


def test_require_exact_roots_can_be_relaxed() -> None:
    bridge = build_worker_doctor_bridge(require_exact_roots_for_known=False)
    result = bridge.map_failure(
        WorkerFailureRecord(
            failure_class="validation_failure",
            task_cid="task:relaxed",
            roots=None,
        )
    )
    assert result.known is True
    assert result.produces_doctor_inspect is True
    assert result.doctor_request is not None
    assert result.doctor_request.roots is None


# ---------------------------------------------------------------------------
# Never opens LLM / body-free surface
# ---------------------------------------------------------------------------


def test_bridge_never_authorizes_provider_or_llm() -> None:
    assert_no_llm_surface_loaded()
    bridge = build_worker_doctor_bridge()
    for failure_class in sorted(known_failure_classes()):
        result = bridge.map_failure(_failure(failure_class))
        assert result.provider_hook_count == 0
        assert result.authorizes_provider is False
        payload = result.to_dict()
        assert payload["provider_hook_count"] == 0
        assert payload["llm_router_invoked"] is False
        assert payload["network_access"] is False
        assert payload["authorizes_provider"] is False
    # Unknown path too
    unknown = bridge.map_failure(_failure("mystery_failure"))
    assert unknown.provider_hook_count == 0
    assert unknown.authorizes_provider is False
    assert_no_llm_surface_loaded()


def test_source_bodies_and_secrets_rejected() -> None:
    with pytest.raises(WorkerDoctorBridgeInputError, match="secrets or source bodies"):
        WorkerFailureRecord(
            failure_class="validation_failure",
            roots=_roots(),
            metadata={"source_body": "def evil(): pass"},
        )
    with pytest.raises(WorkerDoctorBridgeInputError, match="secrets or source bodies"):
        WorkerFailureRecord.from_dict(
            {
                "failure_class": "validation_failure",
                "roots": _roots().to_dict(),
                "api_key": "sekrit",
            }
        )


def test_path_escape_on_write_paths_rejected() -> None:
    with pytest.raises(WorkerDoctorBridgeInputError, match="relative repository path"):
        WorkerFailureRecord(
            failure_class="validation_failure",
            roots=_roots(),
            write_paths=("../outside.py",),
        )


def test_mapping_from_dict_round_trip() -> None:
    record = _failure("scope_failure")
    result = map_worker_failure(record.to_dict())
    assert result.known is True
    assert result.produces_doctor_inspect is True
    assert result.doctor_request is not None
    # Request itself is body-free and serializable
    request_dict = result.doctor_request.to_dict()
    rebuilt = DoctorOperationRequest.from_dict(request_dict)
    assert rebuilt.operation == DoctorOperation.INSPECT.value
    assert rebuilt.roots is not None
    assert rebuilt.roots.content_id == record.roots.content_id  # type: ignore[union-attr]


def test_result_payload_is_body_free_and_content_addressed() -> None:
    result = map_worker_failure(_failure("validation_failure"))
    payload = result.to_dict()
    assert "source" not in payload
    assert "prompt" not in payload
    assert result.content_id
    # Stable for identical mapping
    again = map_worker_failure(_failure("validation_failure"))
    # Incident id binds attempt/task; content_id of result includes request_id
    # which is content-addressed — just ensure both are non-empty opaque tokens.
    assert again.content_id
    assert " " not in again.content_id


def test_missing_failure_class_rejected() -> None:
    with pytest.raises(WorkerDoctorBridgeInputError, match="failure_class"):
        WorkerFailureRecord(failure_class="")


def test_sys_modules_does_not_require_llm_after_map() -> None:
    # Ambient host may already have packages; ensure the bridge path does not
    # *require* loading them as a side effect of mapping.
    before = {
        name
        for name in sys.modules
        if name.split(".", 1)[0]
        in {"openai", "anthropic", "litellm", "requests", "httpx"}
    }
    map_worker_failure(_failure("validation_failure"))
    map_worker_failure(_failure("unknown_xyz"))
    after = {
        name
        for name in sys.modules
        if name.split(".", 1)[0]
        in {"openai", "anthropic", "litellm", "requests", "httpx"}
    }
    assert after == before
