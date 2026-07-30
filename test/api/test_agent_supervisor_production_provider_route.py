"""SCA-615 production-wire bounded Grok proposal and independent Codex review.

Acceptance (fail-closed):

* A production model-assisted task invokes only the typed packet route.
* Grok cannot self-review.
* Codex receives only the bounded proposal/evidence slice.
* The final applied patch and merge bind to the admitted review chain.
* Absent/degraded/stale/cross-task receipts remain pending.
* Deterministic-only tasks invoke no model.
* No provider receives the repository corpus.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.model_catalog.identity import (
    model_identity,
    provider_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA,
    PRODUCTION_PROVIDER_ROUTE_INTERFACE,
    PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA,
    SCAEV615ROUTE,
    ImplementationProviderRouter,
    ProductionContractPacket,
    ProductionReceiptDisposition,
    ProviderBounds,
    ProviderReason,
    ProviderRequest,
    ProviderRole,
    ProviderRoutingError,
    ReviewPresence,
    RouteStatus,
    bind_applied_patch_to_review_chain,
    build_production_contract_packet,
    build_production_provider_route_evaluation,
    evaluate_production_provider_receipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    MODEL_ASSISTED_PROVIDER_ROUTE_EVENT,
    McpPlusPlusLlmGenerateProvider,
    PRODUCTION_PROVIDER_ROUTE_BINDING_EVENT,
    PRODUCTION_PROVIDER_ROUTE_EVENT,
    PRODUCTION_PROVIDER_ROUTE_PENDING_EVENT,
    PortalTask,
    PortalTaskState,
    TodoImplementationDaemon,
    _production_provider_response_schema,
    parse_task_file,
)


EVALUATION_RELATIVE_PATH = Path(
    "data/agent_supervisor/swissknife_contract_assurance/evaluation/"
    "production-provider-route.json"
)

SNAPSHOT = "git-commit:sca-615-fixture"
PATH = (
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/todo_daemon/"
    "implementation_daemon.py"
)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _commit_and_bind(
    daemon: TodoImplementationDaemon,
    task: PortalTask,
    route_payload: dict[str, Any],
    *,
    attempt: int = 1,
):
    repo = daemon.repo_root
    _git(repo, "add", PATH)
    _git(repo, "commit", "-m", f"{task.task_id}: bind reviewed implementation")
    implementation_commit = _git_output(repo, "rev-parse", "HEAD")
    baseline_ref = _git_output(repo, "rev-parse", "HEAD^")
    return daemon._bind_production_route_to_implementation_commit(
        task=task,
        attempt=attempt,
        baseline_ref=baseline_ref,
        implementation_commit=implementation_commit,
        route_payload=route_payload,
    )


def _daemon(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TodoImplementationDaemon:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Production Route Test")
    _git(repo, "config", "user.email", "production-route@example.invalid")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Production provider route tasks\n", encoding="utf-8")
    (repo / ".gitignore").write_text("state/\n", encoding="utf-8")
    target = repo / PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# baseline\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")

    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## SCA-",
        implement=True,
        implementation_command="raw-model-command-must-not-run",
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_mutation",
        lambda _kind, _payload, action: action(),
    )
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_ALLOW_RAW_MODEL_COMMAND",
        raising=False,
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_PRODUCTION_PROVIDER_ROUTE", "1")
    return daemon


def _task(**overrides: Any) -> PortalTask:
    payload = {
        "task_id": "SCA-615",
        "title": "Production-wire bounded Grok proposal and independent Codex review",
        "status": "ready",
        "completion": "manual",
        "priority": "P0",
        "track": "production-provider-routing",
        "outputs": [PATH],
        "validation": [
            "python3 -m pytest external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_production_provider_route.py -q"
        ],
        "acceptance": (
            "A production model-assisted task invokes only the typed packet route"
        ),
        "metadata": {
            "Provider role": "grok-implement, codex-review",
            "Context budget tokens": "4096",
        },
    }
    payload.update(overrides)
    return PortalTask(**payload)


def _events(daemon: TodoImplementationDaemon) -> list[dict[str, Any]]:
    if not daemon.events_path.exists():
        return []
    return [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _accept(proposal):
    return {"accepted": True, "reason_code": f"admitted:{proposal.role.value}"}


def _grok(request):
    assert request["role"] == ProviderRole.GROK_IMPLEMENT.value
    assert request["response_contract"]["repository_write_allowed"] is False
    provider_input = request["provider_input"]
    assert "contract_packet" in provider_input
    encoded = json.dumps(provider_input, sort_keys=True)
    assert "repository_corpus" not in encoded
    assert "source_code" not in encoded
    assert "workspace_path" not in encoded
    return {
        "proposal": {
            "patch": f"diff --git a/{PATH} b/{PATH}\n",
            "declared_paths": [PATH],
            "files": [
                {
                    "path": PATH,
                    "content": "# production-route-applied\n",
                }
            ],
        }
    }


def _codex(request):
    assert request["role"] == ProviderRole.CODEX_REVIEW.value
    assert request["response_contract"]["repository_write_allowed"] is False
    provider_input = request["provider_input"]
    assert "contract_packet" not in provider_input
    assert "admitted_implementation_proposal" in provider_input
    assert "evidence_slice" in provider_input
    assert provider_input["admitted_implementation_proposal"][
        "completion_authoritative"
    ] is False
    encoded = json.dumps(provider_input, sort_keys=True)
    assert "repository_corpus" not in encoded
    assert "source_code" not in encoded
    return {"decision": "approve", "findings": []}


_grok.provider_identity = "mcp++:xai:grok-test"
_grok.model_identity = "grok-test"
_grok.last_session_identity = "session:grok-test"
_codex.provider_identity = "mcp++:openai:codex-test"
_codex.model_identity = "codex-test"
_codex.last_session_identity = "session:codex-test"


class _McpResponse:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.status = 200
        self.body = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.headers = {"Content-Length": str(len(self.body))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self, limit: int) -> bytes:
        return self.body[:limit]


def _mcp_provider_request() -> ProviderRequest:
    prompt = b'{"contract_packet":"bounded"}'
    return ProviderRequest(
        role=ProviderRole.GROK_IMPLEMENT,
        packet_id="packet:mcp-test",
        snapshot_id=SNAPSHOT,
        task_id="SCA-615",
        payload={"contract_packet": "bounded"},
        bounds=ProviderBounds(
            max_prompt_tokens=64,
            max_prompt_bytes=1024,
            max_response_bytes=4096,
            timeout_seconds=10,
        ),
        response_contract={"repository_write_allowed": False},
        prompt=prompt,
        prompt_tokens=8,
    )


def _mcp_success_envelope(request: ProviderRequest) -> dict[str, Any]:
    request_id = (
        f"sca615:{request.role.value}:"
        f"{hashlib.sha256(request.prompt).hexdigest()}"
    )
    generated = '{"proposal":{"files":[]}}'
    catalog_provider_id = provider_identity("grok_cli")
    catalog_model_id = model_identity(
        catalog_provider_id,
        "grok-4.5",
    )
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "success": True,
            "status": "success",
            "catalog_revision": "catalog:test",
            "text": generated,
            "selected_binding": {
                "binding_id": "binding:grok",
                "router": "llm_router",
                "provider_id": catalog_provider_id,
                "model_id": catalog_model_id,
                "operations": ["text.generate"],
            },
            "receipt": {
                "selected_binding_id": "binding:grok",
                "catalog_revision": "catalog:test",
                "operation": "text.generate",
                "fallback": {"allowed": False, "used": False},
                "input": {
                    "count": 1,
                    "text_bytes": len(request.prompt),
                },
                "output": {"bytes": len(generated.encode("utf-8"))},
            },
        },
    }


def _mcp_failure_envelope(
    request_id: str,
    error_code: str,
    *,
    unsafe_message: str,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "success": False,
            "status": "error",
            "error": {
                "code": error_code,
                "message": unsafe_message,
                "cause": "UnsafeProviderDetail",
            },
            "error_code": error_code,
            "error_type": error_code,
        },
    }


def test_mcpplusplus_provider_pins_route_and_rejects_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _mcp_provider_request()
    envelope = _mcp_success_envelope(request)
    observed: dict[str, Any] = {}

    def fake_urlopen(http_request, *, timeout):
        observed["body"] = json.loads(http_request.data)
        observed["timeout"] = timeout
        return _McpResponse(envelope)

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "implementation_daemon.urlopen",
        fake_urlopen,
    )
    provider = McpPlusPlusLlmGenerateProvider(
        role=ProviderRole.GROK_IMPLEMENT,
        endpoint_url="http://127.0.0.1:9002/mcp",
        provider_selector="grok_cli",
        model_selector="grok-4.5",
    )

    generated = provider(request)

    assert generated == envelope["result"]["text"]
    arguments = observed["body"]["params"]["arguments"]
    assert arguments["provider"] == "grok_cli"
    assert arguments["model"] == "grok-4.5"
    assert arguments["allow_fallback"] is False
    assert arguments["response_schema"]["required"] == ["proposal"]
    proposal_branches = arguments["response_schema"]["properties"]["proposal"]["anyOf"]
    assert proposal_branches[0]["required"] == ["patch", "declared_paths"]
    assert proposal_branches[0]["additionalProperties"] is False
    assert proposal_branches[1]["required"] == ["files", "declared_paths"]
    assert proposal_branches[1]["additionalProperties"] is False
    assert provider.last_session_identity

    envelope["result"]["receipt"]["fallback"]["used"] = True
    with pytest.raises(
        ProviderRoutingError,
        match="routing receipt is invalid",
    ):
        provider(request)


@pytest.mark.parametrize(
    ("error_code", "expected_reason"),
    [
        ("timeout", ProviderReason.PROVIDER_TIMEOUT.value),
        (
            "output_limit_exceeded",
            ProviderReason.PROVIDER_RESPONSE_TOO_LARGE.value,
        ),
        (
            "invalid_router_output",
            ProviderReason.PROVIDER_RESPONSE_MALFORMED.value,
        ),
        ("input_limit_exceeded", ProviderReason.PROMPT_TOO_LARGE.value),
        ("invalid_request", ProviderReason.PACKET_MALFORMED.value),
        ("no_match", ProviderReason.GROK_UNAVAILABLE.value),
        ("router_error", ProviderReason.PROVIDER_FAILURE.value),
        pytest.param(
            "t" * 65,
            ProviderReason.PROVIDER_FAILURE.value,
            id="overlong-code-fails-generic",
        ),
    ],
)
def test_mcpplusplus_provider_classifies_only_bounded_tool_error_codes(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
    expected_reason: str,
) -> None:
    request = _mcp_provider_request()
    request_id = (
        f"sca615:{request.role.value}:"
        f"{hashlib.sha256(request.prompt).hexdigest()}"
    )
    unsafe_message = (
        "provider-secret=must-not-reflect prompt="
        + request.prompt.decode("utf-8")
    )
    envelope = _mcp_failure_envelope(
        request_id,
        error_code,
        unsafe_message=unsafe_message,
    )

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "implementation_daemon.urlopen",
        lambda _request, *, timeout: _McpResponse(envelope),
    )
    provider = McpPlusPlusLlmGenerateProvider(
        role=ProviderRole.GROK_IMPLEMENT,
        endpoint_url="http://127.0.0.1:9002/mcp",
        provider_selector="grok_cli",
        model_selector="grok-4.5",
    )

    with pytest.raises(ProviderRoutingError) as failure:
        provider(request)

    assert failure.value.reason_code == expected_reason
    assert unsafe_message not in str(failure.value)
    assert request.prompt.decode("utf-8") not in str(failure.value)


def test_mcpplusplus_failed_route_receipt_preserves_pinned_identities_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_error_marker = "wire-error-secret-must-not-persist"
    unsafe_prompt_marker = "wire-prompt-secret-must-not-persist"

    def fake_urlopen(http_request, *, timeout):
        request_payload = json.loads(http_request.data)
        prompt = request_payload["params"]["arguments"]["prompt"]
        unsafe_message = f"{unsafe_error_marker}:{prompt}"
        return _McpResponse(
            _mcp_failure_envelope(
                request_payload["id"],
                "timeout",
                unsafe_message=unsafe_message,
            )
        )

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "implementation_daemon.urlopen",
        fake_urlopen,
    )
    provider = McpPlusPlusLlmGenerateProvider(
        role=ProviderRole.GROK_IMPLEMENT,
        endpoint_url="http://127.0.0.1:9002/mcp",
        provider_selector="grok_cli",
        model_selector="grok-4.5",
    )
    packet = build_production_contract_packet(
        task_id="SCA-615",
        snapshot_id=SNAPSHOT,
        write_paths=[PATH],
        validation_commands=["true"],
        acceptance_criteria=unsafe_prompt_marker,
    )

    result = ImplementationProviderRouter(
        grok_provider=provider,
        admission_gate=_accept,
    ).route(packet, current_snapshot_id=SNAPSHOT)

    assert result.status is RouteStatus.REJECTED
    assert result.reason_code == ProviderReason.PROVIDER_TIMEOUT.value
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.status == "failed"
    assert attempt.provider_identity == (
        "mcp++:llm_generate:provider=grok_cli"
    )
    assert attempt.model_identity == "grok_cli:grok-4.5"
    assert attempt.session_identity == ""
    assert attempt.prompt_bytes > 0
    assert attempt.response_bytes == 0

    receipt = result.provider_receipt.to_dict()
    encoded_receipt = json.dumps(receipt, sort_keys=True)
    assert unsafe_error_marker not in encoded_receipt
    assert unsafe_prompt_marker not in encoded_receipt
    assert receipt["attempts"][0]["prompt_embedded"] is False
    assert receipt["attempts"][0]["response_embedded"] is False
    assert "error" not in receipt["attempts"][0]


def test_mcpplusplus_codex_review_schema_is_strict_output_compatible() -> None:
    schema = _production_provider_response_schema(ProviderRole.CODEX_REVIEW)

    def assert_strict_objects(node: Any) -> None:
        if isinstance(node, Mapping):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", ())) == set(
                    node.get("properties", {})
                )
            for value in node.values():
                assert_strict_objects(value)
        elif isinstance(node, list):
            for value in node:
                assert_strict_objects(value)

    assert schema["required"] == ["decision", "findings", "proposal"]
    proposal = schema["properties"]["proposal"]
    assert proposal["anyOf"][-1] == {"type": "null"}
    assert proposal["anyOf"][0]["required"] == [
        "patch",
        "declared_paths",
    ]
    assert proposal["anyOf"][1]["required"] == [
        "files",
        "declared_paths",
    ]
    for branch in proposal["anyOf"][:-1]:
        assert branch["type"] == "object"
        assert branch["additionalProperties"] is False
        assert set(branch["required"]) == set(branch["properties"])
        files = branch["properties"].get("files")
        if files is not None:
            replacement = files["items"]
            assert replacement["additionalProperties"] is False
            assert set(replacement["required"]) == set(replacement["properties"])
    assert_strict_objects(schema)


def test_production_model_assisted_invokes_only_typed_packet_route(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    workspace = daemon.repo_root
    writes_via_raw = []

    def forbid_raw(*_args, **_kwargs):
        writes_via_raw.append("raw")
        raise AssertionError("raw model command must not run")

    monkeypatch.setattr(daemon, "_build_implementation_command", forbid_raw)

    result = daemon.run_production_model_assisted_route(
        task,
        attempt=1,
        workspace_path=workspace,
        snapshot_id=SNAPSHOT,
        apply=True,
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
    )

    assert result["raw_model_command_invoked"] is False
    assert result["typed_packet_route_only"] is True
    assert result["returncode"] == 0
    assert result["route_result"].status is RouteStatus.SUCCEEDED
    assert result["binding"] is None
    assert result["event"]["review_chain_binding_pending"] is True
    assert result["pending"] is False
    assert writes_via_raw == []

    applied = (workspace / PATH).read_text(encoding="utf-8")
    assert "production-route-applied" in applied
    binding = _commit_and_bind(daemon, task, result)
    assert binding.implementation_commit == _git_output(
        workspace, "rev-parse", "HEAD"
    )
    assert binding.implementation_tree_id == _git_output(
        workspace, "rev-parse", "HEAD^{tree}"
    )
    assert binding.changed_paths == (PATH,)

    events = _events(daemon)
    assert any(item.get("type") == PRODUCTION_PROVIDER_ROUTE_EVENT for item in events)
    assert any(
        item.get("type") == MODEL_ASSISTED_PROVIDER_ROUTE_EVENT for item in events
    )
    assert any(
        item.get("type") == PRODUCTION_PROVIDER_ROUTE_BINDING_EVENT for item in events
    )
    production = next(
        item for item in events if item.get("type") == PRODUCTION_PROVIDER_ROUTE_EVENT
    )
    assert production["typed_packet_route_only"] is True
    assert production["raw_model_command_invoked"] is False
    assert production["provider"]
    assert production["packet"]
    assert production["review_chain"]
    assert production["provider_receipt"]


def test_production_directory_scope_allows_declared_descendants(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    directory = "src/scoped"
    child = f"{directory}/contract.py"
    scoped = daemon.repo_root / directory
    scoped.mkdir(parents=True)
    (scoped / "baseline.py").write_text("# baseline\n", encoding="utf-8")
    _git(daemon.repo_root, "add", directory)
    _git(daemon.repo_root, "commit", "-m", "seed directory scope")
    baseline_ref = _git_output(daemon.repo_root, "rev-parse", "HEAD")
    task = _task(outputs=[directory], validation=[])

    def grok(_request):
        return {
            "proposal": {
                "patch": f"diff --git a/{child} b/{child}\n",
                "declared_paths": [child],
                "files": [{"path": child, "content": "# scoped\n"}],
            }
        }

    def codex(_request):
        return {"decision": "approve", "findings": []}

    grok.provider_identity = "mcp++:xai:grok-directory"
    grok.model_identity = "grok-directory"
    grok.last_session_identity = "session:grok-directory"
    codex.provider_identity = "mcp++:openai:codex-directory"
    codex.model_identity = "codex-directory"
    codex.last_session_identity = "session:codex-directory"

    route_payload = daemon.run_production_model_assisted_route(
        task,
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=SNAPSHOT,
        apply=True,
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=_accept,
    )

    assert route_payload["returncode"] == 0
    packet_scope = route_payload[
        "contract_packet"
    ].provider_input_payload["scope"]
    assert packet_scope["write_directory_paths"] == [directory]
    allowed_paths, allowed_directories = (
        daemon._production_packet_write_scope(
            route_payload["contract_packet"]
        )
    )
    assert daemon._production_path_in_write_scope(
        child,
        allowed=set(allowed_paths),
        allowed_directories=allowed_directories,
    )
    assert not daemon._production_path_in_write_scope(
        "src/scoped-sibling/contract.py",
        allowed=set(allowed_paths),
        allowed_directories=allowed_directories,
    )
    assert (daemon.repo_root / child).read_text(encoding="utf-8") == (
        "# scoped\n"
    )

    _git(daemon.repo_root, "add", child)
    _git(daemon.repo_root, "commit", "-m", "apply directory-scoped proposal")
    implementation_commit = _git_output(
        daemon.repo_root,
        "rev-parse",
        "HEAD",
    )
    binding = daemon._bind_production_route_to_implementation_commit(
        task=task,
        attempt=1,
        baseline_ref=baseline_ref,
        implementation_commit=implementation_commit,
        route_payload=route_payload,
    )
    assert binding.changed_paths == (child,)


def test_production_route_forbids_raw_implementation_command(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    with pytest.raises(RuntimeError, match="typed packet route"):
        daemon._build_implementation_command(daemon.repo_root, task=task)


def test_grok_cannot_self_review_on_production_route(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)

    def same(request):
        if request["role"] == ProviderRole.GROK_IMPLEMENT.value:
            return {
                "proposal": {
                    "patch": "x",
                    "declared_paths": [PATH],
                    "files": [{"path": PATH, "content": "x\n"}],
                }
            }
        return {"decision": "approve"}

    result = daemon.run_production_model_assisted_route(
        _task(),
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=SNAPSHOT,
        apply=True,
        grok_provider=same,
        codex_provider=same,
        admission_gate=_accept,
    )
    assert result["route_result"].status is RouteStatus.REJECTED
    assert (
        result["route_result"].reason_code
        == ProviderReason.SELF_REVIEW_FORBIDDEN.value
    )
    assert result["binding"] is None
    assert result["pending"] is True
    assert daemon.model_assisted_authoritative_completion_allowed(
        result["route_result"]
    ) is False


def test_codex_receives_only_bounded_proposal_evidence_slice(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    seen: dict[str, Any] = {}

    def codex(request):
        seen["input"] = request["provider_input"]
        assert set(request["provider_input"]) == {
            "admitted_implementation_proposal",
            "evidence_slice",
        }
        slice_ = request["provider_input"]["evidence_slice"]
        assert "packet_id" in slice_
        assert "scope" in slice_
        assert "acceptance" in slice_
        assert "goal_ids" in slice_
        # Full goal corpus / counterexample bodies must not appear.
        encoded = json.dumps(request["provider_input"], sort_keys=True)
        assert "counterexample" not in encoded
        assert "repository_corpus" not in encoded
        return {"decision": "approve", "findings": []}

    codex.provider_identity = "mcp++:openai:codex-bounded-slice"
    codex.model_identity = "codex-bounded-slice"
    codex.last_session_identity = "session:codex-bounded-slice"

    result = daemon.run_production_model_assisted_route(
        _task(),
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=SNAPSHOT,
        apply=True,
        grok_provider=_grok,
        codex_provider=codex,
        admission_gate=_accept,
    )
    assert result["route_result"].status is RouteStatus.SUCCEEDED
    assert "admitted_implementation_proposal" in seen["input"]


def test_applied_patch_and_merge_bind_to_admitted_review_chain(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    result = daemon.run_production_model_assisted_route(
        task,
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:sca-615:1",
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
    )
    assert result["binding"] is None
    binding = _commit_and_bind(daemon, task, result)
    payload = binding.to_dict()
    assert payload["schema"] == PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA
    assert payload["provider_result_admitted"] is True
    assert payload["review_presence"] == ReviewPresence.INDEPENDENT.value
    assert payload["write_performed"] is True
    assert payload["writer_lease_id"] == "lease:sca-615:1"
    assert payload["receipt_id"]
    assert payload["review_chain_digest"]
    assert payload["selected_proposal_digest"]
    assert payload["completion_authoritative"] is False

    # Merge metadata carries the same admitted review-chain binding.
    assert daemon._last_production_review_chain_binding is binding
    metadata_probe = {
        "task_id": task.task_id,
    }
    # Simulate the enqueue attachment path.
    production_binding = daemon._last_production_review_chain_binding
    assert production_binding.task_id == task.task_id
    metadata_probe["admitted_review_chain_binding"] = production_binding.to_dict()
    assert metadata_probe["admitted_review_chain_binding"]["receipt_id"] == (
        binding.receipt_id
    )


def test_merge_gate_revalidates_review_binding_against_git(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    route_payload = daemon.run_production_model_assisted_route(
        task,
        attempt=1,
        workspace_path=daemon.repo_root,
        snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:sca-615:merge-gate",
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
    )
    binding = _commit_and_bind(daemon, task, route_payload)
    implementation_commit = binding.implementation_commit
    merge_tree = _git_output(
        daemon.repo_root,
        "rev-parse",
        f"{implementation_commit}^{{tree}}",
    )
    metadata = {
        "production_provider_route": True,
        "baseline_ref": _git_output(daemon.repo_root, "rev-parse", "HEAD^"),
        "provider_execution_receipt": route_payload["receipt"].to_dict(),
        "admitted_review_chain_binding": binding.to_dict(),
    }

    evidence = daemon._production_provider_review_merge_evidence(
        metadata=metadata,
        task=task,
        implementation_commit=implementation_commit,
        merge_commit=implementation_commit,
        repository_tree_id=f"git-tree:{merge_tree}",
    )

    assert evidence["admitted"] is True
    assert evidence["binding"]["merge_commit"] == implementation_commit
    assert evidence["binding"]["merge_tree_id"] == merge_tree
    assert evidence["gate_evidence"]["provider_review"][
        "review_receipt_id"
    ] == binding.receipt_id

    metadata["admitted_review_chain_binding"] = {
        **binding.to_dict(),
        "implementation_tree_id": "forged-tree",
    }
    rejected = daemon._production_provider_review_merge_evidence(
        metadata=metadata,
        task=task,
        implementation_commit=implementation_commit,
        merge_commit=implementation_commit,
        repository_tree_id=f"git-tree:{merge_tree}",
    )
    assert rejected["admitted"] is False
    assert rejected["reason"] == ProviderReason.REVIEW_CHAIN_UNBOUND.value


@pytest.mark.parametrize(
    "kind",
    ["absent", "degraded", "stale", "cross_task"],
)
def test_absent_degraded_stale_cross_task_receipts_remain_pending(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()

    if kind == "absent":
        result = daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=False,
            grok_provider=_grok,
            admission_gate=_accept,
        )
        assert result["route_result"].review_presence == ReviewPresence.ABSENT.value
        assert result["pending"] is True
        assert result["binding"] is None
        assert result["disposition"] is ProductionReceiptDisposition.PENDING_ABSENT
    elif kind == "degraded":
        def unavailable_codex(_request):
            raise RuntimeError("review unavailable")

        unavailable_codex.provider_identity = "mcp++:openai:unavailable"
        unavailable_codex.model_identity = "codex-unavailable"
        unavailable_codex.last_session_identity = "session:codex-unavailable"
        result = daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=False,
            grok_provider=_grok,
            codex_provider=unavailable_codex,
            admission_gate=_accept,
        )
        assert result["route_result"].review_presence == ReviewPresence.DEGRADED.value
        assert result["pending"] is True
        assert result["binding"] is None
        assert result["disposition"] is ProductionReceiptDisposition.PENDING_DEGRADED
    elif kind == "stale":
        result = daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=True,
            grok_provider=_grok,
            codex_provider=_codex,
            admission_gate=_accept,
        )
        disposition, reason = evaluate_production_provider_receipt(
            result["receipt"],
            expected_task_id=task.task_id,
            expected_snapshot_id="git-commit:other",
            current_snapshot_id="git-commit:other",
        )
        assert disposition is ProductionReceiptDisposition.PENDING_STALE
        assert reason == ProviderReason.RECEIPT_STALE.value
        assert daemon.production_provider_receipt_allows_merge(
            result["receipt"],
            expected_task_id=task.task_id,
            expected_snapshot_id="git-commit:other",
        ) is False
        return
    else:  # cross_task
        result = daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=True,
            grok_provider=_grok,
            codex_provider=_codex,
            admission_gate=_accept,
        )
        disposition, reason = evaluate_production_provider_receipt(
            result["receipt"],
            expected_task_id="SCA-OTHER",
            expected_snapshot_id=SNAPSHOT,
        )
        assert disposition is ProductionReceiptDisposition.PENDING_CROSS_TASK
        assert reason == ProviderReason.RECEIPT_CROSS_TASK.value
        assert daemon.production_provider_receipt_allows_merge(
            result["receipt"],
            expected_task_id="SCA-OTHER",
            expected_snapshot_id=SNAPSHOT,
        ) is False
        return

    assert daemon.model_assisted_authoritative_completion_allowed(
        result["route_result"]
    ) is False
    pending_events = [
        item
        for item in _events(daemon)
        if item.get("type") == PRODUCTION_PROVIDER_ROUTE_PENDING_EVENT
    ]
    assert pending_events
    assert all(item.get("pending") is True for item in pending_events)
    assert all(item.get("completion_authoritative") is False for item in pending_events)


def test_deterministic_only_tasks_invoke_no_model(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(
        task_id="SCA-DET",
        metadata={"Provider role": "deterministic-only"},
    )
    assert daemon._task_uses_typed_local_execution(task) is True
    assert daemon._production_provider_route_enabled(task) is False
    with pytest.raises(RuntimeError, match="deterministic-only"):
        daemon.run_production_model_assisted_route(
            task,
            attempt=1,
            workspace_path=daemon.repo_root,
            snapshot_id=SNAPSHOT,
            apply=True,
            grok_provider=_grok,
            codex_provider=_codex,
            admission_gate=_accept,
        )


def test_no_provider_receives_repository_corpus() -> None:
    seen: list[dict[str, Any]] = []

    def grok(request):
        seen.append(request.to_dict())
        assert request["role"] == ProviderRole.GROK_IMPLEMENT.value
        return {
            "proposal": {
                "patch": "bounded",
                "declared_paths": [PATH],
            }
        }

    def codex(request):
        seen.append(request.to_dict())
        assert request["role"] == ProviderRole.CODEX_REVIEW.value
        return {"decision": "approve"}

    packet = build_production_contract_packet(
        task_id="SCA-615",
        snapshot_id=SNAPSHOT,
        write_paths=[PATH],
        validation_commands=["true"],
        acceptance_criteria="no corpus",
    )
    # Ensure broad keys are rejected before any provider call.
    with pytest.raises(Exception):
        build_production_contract_packet(
            task_id="SCA-615",
            snapshot_id=SNAPSHOT,
            write_paths=[PATH],
            extra_goal={"repository_corpus": "entire-tree"},
        )

    # Distinct callables: Grok must never self-review.
    router = ImplementationProviderRouter(
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=_accept,
    )
    result = router.route(packet, current_snapshot_id=SNAPSHOT)
    assert result.status is RouteStatus.SUCCEEDED
    encoded = json.dumps(seen, sort_keys=True)
    assert "repository_corpus" not in encoded
    assert "source_code" not in encoded
    assert "workspace_path" not in encoded
    assert "full_repository" not in encoded
    # Codex path must not include the implementer's full contract packet.
    codex_request = next(
        item for item in seen if item["role"] == ProviderRole.CODEX_REVIEW.value
    )
    assert "contract_packet" not in codex_request["provider_input"]


def test_bind_applied_patch_requires_independent_admitted_review() -> None:
    packet = build_production_contract_packet(
        task_id="SCA-615",
        snapshot_id=SNAPSHOT,
        write_paths=[PATH],
    )
    router = ImplementationProviderRouter(
        grok_provider=_grok,
        admission_gate=_accept,
    )
    fallback = router.route(packet, current_snapshot_id=SNAPSHOT)
    assert bind_applied_patch_to_review_chain(fallback) is None

    full = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
        writer=lambda _proposal, _lease: None,
    ).route(
        packet,
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:bound",
    )
    binding = bind_applied_patch_to_review_chain(
        full,
        writer_lease_id="lease:bound",
        implementation_commit="a" * 40,
        implementation_tree_id="b" * 40,
        changed_paths=[PATH],
    )
    assert binding is not None
    assert binding.write_performed is True
    assert binding.writer_lease_id == "lease:bound"


def test_production_evaluation_artifact_exists_and_covers_acceptance() -> None:
    search_roots = (Path.cwd(), *Path(__file__).resolve().parents)
    evaluation_path = next(
        (
            root / EVALUATION_RELATIVE_PATH
            for root in search_roots
            if (root / EVALUATION_RELATIVE_PATH).is_file()
        ),
        None,
    )
    if evaluation_path is None:
        pytest.skip("root-level SCA-615 evaluation artifact is not packaged")
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert payload["schema"] == PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA
    assert payload["interface"] == PRODUCTION_PROVIDER_ROUTE_INTERFACE
    assert SCAEV615ROUTE in payload["evidence"]["requirement_ids"]
    acceptance = payload["acceptance"]
    assert acceptance["typed_packet_route_only"] is True
    assert acceptance["grok_cannot_self_review"] is True
    assert acceptance["codex_bounded_slice_only"] is True
    assert acceptance["apply_merge_bound_to_review_chain"] is True
    assert acceptance["deterministic_only_no_model"] is True
    assert acceptance["no_repository_corpus"] is True
    assert payload["production_route"]["raw_model_command_forbidden"] is True
    assert payload["corpus_isolation"]["provider_receives_repository_corpus"] is False
    assert payload["deterministic_only"]["invokes_no_model"] is True
    case_ids = {item.get("id") for item in payload.get("cases") or []}
    assert "happy-path-admitted" in case_ids
    assert "absent-review-pending" in case_ids
    assert "cross-task-receipt-pending" in case_ids
    assert "stale-receipt-pending" in case_ids


def test_daemon_builds_bounded_production_packet(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    packet = daemon.build_production_contract_packet_for_task(
        _task(),
        snapshot_id=SNAPSHOT,
        attempt=2,
    )
    assert isinstance(packet, ProductionContractPacket)
    assert packet.task_id == "SCA-615"
    assert packet.snapshot_id == SNAPSHOT
    payload = dict(packet.provider_input_payload)
    assert payload["authority"]["completion_authoritative"] is False
    assert PATH in payload["scope"]["write_paths"]
    assert "repository_corpus" not in json.dumps(payload)


def test_production_packet_forwards_only_compiler_selected_targeted_evidence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    target = daemon.repo_root / PATH
    target.write_text(
        "# baseline\nTARGETED_SOURCE_MARKER = 'repair-me'\n",
        encoding="utf-8",
    )
    discovery = daemon.repo_root / "data" / "discovery" / "sca-615.md"
    discovery.parent.mkdir(parents=True)
    discovery.write_text(
        "# Finding\nMissing evidence: SCAEV615ROUTE\n",
        encoding="utf-8",
    )
    task = _task(
        metadata={
            "Provider role": "grok-implement, codex-review",
            "Context budget tokens": "4096",
            "Discovery evidence": "data/discovery/sca-615.md",
            "Missing evidence": "SCAEV615ROUTE",
        }
    )

    context = daemon._compile_implementation_context(task, attempt=1)
    selected = context.capsule.evidence
    assert any(
        reference.kind == "task-discovery-evidence"
        for reference in selected
    )
    assert any(
        "TARGETED_SOURCE_MARKER" in reference.summary
        for reference in selected
    )
    snapshot = f"git-commit:{context.capsule.tree_id}"
    packet = daemon.build_production_contract_packet_for_task(
        task,
        snapshot_id=snapshot,
        attempt=1,
        context_capsule=context.capsule,
    )
    payload = dict(packet.provider_input_payload)
    handles = payload["evidence_handles"]
    encoded_handles = json.dumps(handles, sort_keys=True)
    assert "TARGETED_SOURCE_MARKER" in encoded_handles
    assert "SCAEV615ROUTE" in encoded_handles
    assert payload["goal"]["context_capsule_id"] == (
        context.capsule.content_id
    )
    assert payload["goal"]["obligation_ids"] == ["SCAEV615ROUTE"]
    assert len(encoded_handles.encode("utf-8")) < 16_384

    captured: dict[str, Any] = {}

    def grok(request):
        captured["grok"] = request
        return _grok(request)

    def codex(request):
        captured["codex"] = request
        return _codex(request)

    result = ImplementationProviderRouter(
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=_accept,
    ).route(
        packet,
        current_snapshot_id=snapshot,
        apply=False,
    )

    assert result.provider_result_admitted is True
    assert "TARGETED_SOURCE_MARKER" in json.dumps(
        captured["grok"]["provider_input"],
        sort_keys=True,
    )
    assert "TARGETED_SOURCE_MARKER" in json.dumps(
        captured["codex"]["provider_input"]["evidence_slice"],
        sort_keys=True,
    )
    assert "TARGETED_SOURCE_MARKER" not in json.dumps(
        result.provider_receipt.to_dict(),
        sort_keys=True,
    )


def test_daemon_bridges_verified_compiled_context_as_ids_and_handles(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    bounded_summary = (
        "task=SCA-615; vector=cid:vector; merge=cid:merge; "
        f"artifact=sha256:{'a' * 64}"
    )
    monkeypatch.setattr(
        daemon,
        "_render_todo_vector_context",
        lambda _task: bounded_summary,
    )
    monkeypatch.setattr(
        daemon,
        "_load_todo_vector_context",
        lambda _task: {},
    )
    compiled = daemon._compile_implementation_context(task, attempt=1)
    snapshot = f"git-commit:{compiled.capsule.tree_id}"

    packet = daemon.build_production_contract_packet_for_task(
        task,
        snapshot_id=snapshot,
        attempt=1,
    )

    payload = dict(packet.provider_input_payload)
    contract_ids = set(payload["goal"]["contract_ids"])
    selected = compiled.capsule.evidence[0]
    assert compiled.capsule.capsule_id in contract_ids
    assert compiled.receipt.receipt_id in contract_ids
    assert selected.reference_content_id in contract_ids
    assert selected.referenced_content_id in contract_ids
    binding = payload["expansion_handles"][0]
    assert binding["status"] == "verified"
    assert binding["task_id"] == task.task_id
    assert binding["production_snapshot_id"] == snapshot
    selected_handle = next(
        item
        for item in payload["expansion_handles"]
        if item.get("reference_id") == selected.reference_id
    )
    assert selected_handle["disposition"] == "selected"
    assert "summary" not in selected_handle
    assert bounded_summary not in json.dumps(payload, sort_keys=True)


def test_daemon_omits_compiled_context_from_a_different_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    daemon._compile_implementation_context(task, attempt=1)

    packet = daemon.build_production_contract_packet_for_task(
        task,
        snapshot_id=SNAPSHOT,
        attempt=1,
    )

    payload = dict(packet.provider_input_payload)
    assert payload["goal"]["contract_ids"] == []
    assert payload["goal"]["obligation_ids"] == []
    assert payload["expansion_handles"][0]["status"] == (
        "omitted_snapshot_mismatch"
    )
    assert payload["expansion_handles"][0]["context_snapshot_id"] != SNAPSHOT


def test_daemon_keeps_immutable_context_valid_when_source_head_advances(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    compiled = daemon._compile_implementation_context(task, attempt=1)
    compiled_snapshot = f"git-commit:{compiled.capsule.tree_id}"

    monkeypatch.setattr(
        daemon,
        "_implementation_repository_and_tree_ids",
        lambda _task: (
            compiled.capsule.repository_id,
            "newer-source-head-after-context-compilation",
        ),
    )

    packet = daemon.build_production_contract_packet_for_task(
        task,
        snapshot_id=compiled_snapshot,
        attempt=1,
    )

    payload = dict(packet.provider_input_payload)
    assert compiled.capsule.capsule_id in payload["goal"]["contract_ids"]
    assert payload["expansion_handles"][0]["status"] == "verified"
    assert payload["expansion_handles"][0]["context_tree_id"] == (
        compiled.capsule.tree_id
    )


def test_daemon_rejects_cross_task_or_tampered_compiled_context(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    compiled = daemon._compile_implementation_context(task, attempt=1)
    snapshot = f"git-commit:{compiled.capsule.tree_id}"

    with pytest.raises(
        ProviderRoutingError,
        match="stale or cross-task",
    ) as cross_task:
        daemon.build_production_contract_packet_for_task(
            _task(task_id="SCA-OTHER"),
            snapshot_id=snapshot,
            attempt=1,
        )
    assert cross_task.value.reason_code == ProviderReason.PACKET_MALFORMED.value

    object.__setattr__(compiled.capsule, "objective_id", "SCA-TAMPERED")
    with pytest.raises(
        ProviderRoutingError,
        match="canonical verification",
    ) as tampered:
        daemon.build_production_contract_packet_for_task(
            task,
            snapshot_id=snapshot,
            attempt=1,
        )
    assert tampered.value.reason_code == ProviderReason.PACKET_MALFORMED.value


def test_build_production_provider_route_evaluation_helper() -> None:
    packet = build_production_contract_packet(
        task_id="SCA-615",
        snapshot_id=SNAPSHOT,
        write_paths=[PATH],
    )
    result = ImplementationProviderRouter(
        grok_provider=_grok,
        codex_provider=_codex,
        admission_gate=_accept,
        writer=lambda _p, _l: None,
    ).route(
        packet,
        current_snapshot_id=SNAPSHOT,
        apply=True,
        writer_lease_id="lease:eval",
    )
    binding = bind_applied_patch_to_review_chain(
        result,
        writer_lease_id="lease:eval",
        implementation_commit="c" * 40,
        implementation_tree_id="d" * 40,
        changed_paths=[PATH],
    )
    evaluation = build_production_provider_route_evaluation(
        route_result=result,
        binding=binding,
        deterministic_only_model_calls=0,
        raw_model_command_invoked=False,
        corpus_exposed_to_provider=False,
    )
    assert evaluation["schema"] == PRODUCTION_PROVIDER_ROUTE_EVALUATION_SCHEMA
    assert evaluation["acceptance"]["typed_packet_route_only"] is True
    assert evaluation["route_result"]["provider_result_admitted"] is True
    assert evaluation["evaluation_id"]


@pytest.mark.parametrize(
    ("typed_route", "expected_attempt_count", "expected_refund_count"),
    [(True, 2, 1), (False, 3, 0)],
)
def test_restart_refunds_only_evidenced_typed_malformed_provider_attempt_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    typed_route: bool,
    expected_attempt_count: int,
    expected_refund_count: int,
) -> None:
    """An already charged attempt-3 receipt is replay-safe on restart."""

    daemon = _daemon(tmp_path, monkeypatch)
    daemon.implement = False
    daemon.max_task_attempts = 3
    daemon.todo_path.write_text(
        f"""# Production provider route tasks

## SCA-615 Production typed provider serialization

- Status: todo
- Completion: manual
- Priority: P0
- Track: production-provider-routing
- Depends on:
- Outputs: {PATH}
- Validation: python3 -m pytest test/api/test_agent_supervisor_production_provider_route.py -q
- Acceptance: Return one bounded provider proposal.
- Provider role: grok-implement, codex-review
""",
        encoding="utf-8",
    )
    task = parse_task_file(daemon.todo_path, "## SCA-")[0]
    identity = daemon._identity_for_task(task)
    snapshot = f"git-commit:{_git_output(daemon.repo_root, 'rev-parse', 'HEAD')}"
    packet = daemon.build_production_contract_packet_for_task(
        task,
        snapshot_id=snapshot,
        attempt=3,
    )

    def malformed_provider(_request):
        return "I'll inspect the scoped files and draft a proposal."

    malformed_provider.provider_identity = "mcp++:xai:grok-malformed"
    malformed_provider.model_identity = "grok-malformed"
    malformed_provider.last_session_identity = "session:grok-malformed"
    route_result, _event, receipt_path = (
        daemon.route_model_assisted_contract_packet(
            packet,
            current_snapshot_id=snapshot,
            task=task,
            attempt=3,
            grok_provider=malformed_provider,
            apply=False,
        )
    )
    assert (
        route_result.reason_code
        == ProviderReason.PROVIDER_RESPONSE_MALFORMED.value
    )
    assert route_result.write_performed is False
    assert route_result.provider_result_admitted is False

    log_path = daemon.implementation_log_dir / "sca-615-attempt-3.log"
    execution = (
        "Execution: production typed packet route "
        f"({PRODUCTION_PROVIDER_ROUTE_INTERFACE})"
        if typed_route
        else "Command: grok --mode agent"
    )
    log_path.write_text(
        execution
        + "\n\n"
        + json.dumps(
            {
                "provider_result_admitted": False,
                "raw_model_command_invoked": False,
                "returncode": 1,
                "typed_packet_route_only": True,
                "write_performed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    state = PortalTaskState(
        task_identities={task.task_id: identity.to_dict()},
        implementation_attempts={task.task_id: 3},
        implementation_attempts_by_cid={
            identity.canonical_task_cid: 3
        },
        last_implementation_task_id=task.task_id,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_returncode=1,
        last_implementation_log_path=str(log_path),
        last_implementation_commit="",
    )
    state.save(daemon.state_path)

    first = daemon.run_once()
    first_state = PortalTaskState.load(daemon.state_path)
    second = daemon.run_once()
    second_state = PortalTaskState.load(daemon.state_path)

    assert len(first["typed_provider_malformed_refunds"]) == (
        expected_refund_count
    )
    assert second.get("typed_provider_malformed_refunds", []) == []
    assert first_state.implementation_attempts[task.task_id] == (
        expected_attempt_count
    )
    assert second_state.implementation_attempts[task.task_id] == (
        expected_attempt_count
    )
    if typed_route:
        receipt_id = route_result.provider_receipt.receipt_id
        assert first["typed_provider_malformed_refunds"][0][
            "receipt_id"
        ] == receipt_id
        assert first_state.typed_provider_malformed_refund_receipts == {
            identity.canonical_task_cid: receipt_id
        }
        assert receipt_path.name == "sca-615-attempt-3-provider-receipt.json"
        refund_events = [
            item
            for item in _events(daemon)
            if item.get("type")
            == "typed_provider_malformed_attempt_refunded"
        ]
        assert len(refund_events) == 1
    else:
        assert (
            first_state.typed_provider_malformed_refund_receipts == {}
        )


def _persist_pre_policy_production_provider_failure(
    daemon: TodoImplementationDaemon,
    *,
    log_returncode: Any = 1,
    prior_malformed_marker: bool = False,
    durable_attempt_count: int = 3,
    failure_role: str = "grok",
) -> tuple[PortalTask, Any, Any, Path]:
    """Persist the exact failed-attempt shape observed before retry deferrals."""

    daemon.implement = False
    daemon.max_task_attempts = 3
    daemon.todo_path.write_text(
        f"""# Production provider route tasks

## SCA-615 Production provider operational restart repair

- Status: todo
- Completion: manual
- Priority: P0
- Track: production-provider-routing
- Depends on:
- Outputs: {PATH}
- Validation: python3 -m pytest test/api/test_agent_supervisor_production_provider_route.py -q
- Acceptance: Retry an operational provider failure without consuming the task budget.
- Provider role: grok-implement, codex-review
""",
        encoding="utf-8",
    )
    task = parse_task_file(daemon.todo_path, "## SCA-")[0]
    identity = daemon._identity_for_task(task)
    snapshot = f"git-commit:{_git_output(daemon.repo_root, 'rev-parse', 'HEAD')}"
    packet = daemon.build_production_contract_packet_for_task(
        task,
        snapshot_id=snapshot,
        attempt=3,
    )

    def failed_provider(_request):
        raise RuntimeError("simulated production provider transport failure")

    failed_provider.provider_identity = "mcp++:xai:grok-failed"
    failed_provider.model_identity = "grok-failed"
    failed_provider.last_session_identity = "session:grok-failed"
    grok_provider = failed_provider
    codex_provider = None
    if failure_role == "codex":
        failed_provider.provider_identity = "mcp++:openai:codex-failed"
        failed_provider.model_identity = "codex-failed"
        failed_provider.last_session_identity = "session:codex-failed"
        grok_provider = _grok
        codex_provider = failed_provider
    route_result, _event, receipt_path = (
        daemon.route_model_assisted_contract_packet(
            packet,
            current_snapshot_id=snapshot,
            task=task,
            attempt=3,
            grok_provider=grok_provider,
            codex_provider=codex_provider,
            admission_gate=_accept,
            apply=False,
        )
    )
    assert route_result.reason_code == ProviderReason.PROVIDER_FAILURE.value
    assert route_result.write_performed is False
    assert route_result.provider_result_admitted is False
    failed_attempt = route_result.attempts[-1]
    expected_failed_role = (
        ProviderRole.CODEX_REVIEW
        if failure_role == "codex"
        else ProviderRole.GROK_IMPLEMENT
    )
    assert failed_attempt.role is expected_failed_role
    assert failed_attempt.status == "failed"
    assert failed_attempt.prompt_bytes > 0
    assert failed_attempt.response_bytes == 0
    assert failed_attempt.response_digest == ""
    assert failed_attempt.provider_identity == failed_provider.provider_identity
    assert failed_attempt.model_identity == failed_provider.model_identity
    assert failed_attempt.session_identity == ""

    log_path = daemon.implementation_log_dir / "sca-615-attempt-3.log"
    log_path.write_text(
        "Execution: production typed packet route "
        f"({PRODUCTION_PROVIDER_ROUTE_INTERFACE})\n\n"
        + json.dumps(
            {
                "provider_result_admitted": False,
                "raw_model_command_invoked": False,
                "returncode": log_returncode,
                "typed_packet_route_only": True,
                "write_performed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    malformed_markers = (
        {
            identity.canonical_task_cid: content_identity(
                {
                    "canonical_task_cid": identity.canonical_task_cid,
                    "reason": ProviderReason.PROVIDER_RESPONSE_MALFORMED.value,
                }
            )
        }
        if prior_malformed_marker
        else {}
    )
    state = PortalTaskState(
        task_identities={task.task_id: identity.to_dict()},
        implementation_attempts={task.task_id: durable_attempt_count},
        implementation_attempts_by_cid={
            identity.canonical_task_cid: durable_attempt_count
        },
        typed_provider_malformed_refund_receipts=malformed_markers,
        last_implementation_task_id=task.task_id,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_returncode=1,
        last_implementation_log_path=str(log_path),
        last_implementation_commit="",
    )
    state.save(daemon.state_path)
    daemon.task_queue.register_task(
        identity,
        priority=task.priority,
        track=task.track,
    )
    daemon.task_queue.record_failure(
        identity.canonical_task_cid,
        reason="pre-policy provider_failure",
    )
    daemon.task_queue.save()
    assert daemon.task_queue.is_cooled_down(identity.canonical_task_cid)
    return task, identity, route_result, receipt_path


def _rewrite_provider_receipt_numeric_field(
    receipt_path: Path,
    *,
    field_name: str,
    value: Any,
) -> None:
    """Keep both content identities valid while corrupting one numeric field."""

    integrated = json.loads(receipt_path.read_text(encoding="utf-8"))
    integration = dict(integrated.pop("daemon_integration"))
    integration.pop("integration_receipt_id", None)
    integrated["attempts"][0][field_name] = value
    receipt_body = {
        key: item
        for key, item in integrated.items()
        if key != "receipt_id"
    }
    integrated["receipt_id"] = content_identity(receipt_body)
    integration["provider_receipt_id"] = integrated["receipt_id"]
    with_integration = dict(integrated)
    with_integration["daemon_integration"] = integration
    integration["integration_receipt_id"] = content_identity(
        with_integration
    )
    integrated["daemon_integration"] = integration
    receipt_path.write_text(
        json.dumps(integrated, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_restart_refunds_pre_policy_provider_failure_once_with_malformed_marker(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task, identity, route_result, receipt_path = (
        _persist_pre_policy_production_provider_failure(
            daemon,
            prior_malformed_marker=True,
        )
    )
    original_state = PortalTaskState.load(daemon.state_path)
    prior_malformed = dict(
        original_state.typed_provider_malformed_refund_receipts
    )

    first = daemon.run_once()
    first_state = PortalTaskState.load(daemon.state_path)
    second = daemon.run_once()
    second_state = PortalTaskState.load(daemon.state_path)

    assert first["typed_provider_malformed_refunds"] == []
    assert len(first["production_provider_operational_refunds"]) == 1
    assert second.get("production_provider_operational_refunds", []) == []
    refund = first["production_provider_operational_refunds"][0]
    assert refund["task_id"] == task.task_id
    assert refund["canonical_task_cid"] == identity.canonical_task_cid
    assert refund["provider_reason"] == ProviderReason.PROVIDER_FAILURE.value
    assert refund["attempt"] == 3
    assert refund["previous_attempt_count"] == 3
    assert refund["refunded_attempt_count"] == 2
    assert refund["receipt_id"] == route_result.provider_receipt.receipt_id
    assert refund["receipt_path"] == str(receipt_path)
    assert refund["write_performed"] is False
    assert refund["provider_result_admitted"] is False
    assert first_state.implementation_attempts[task.task_id] == 2
    assert second_state.implementation_attempts[task.task_id] == 2
    assert first_state.implementation_attempts_by_cid[
        identity.canonical_task_cid
    ] == 2
    assert (
        first_state.production_provider_operational_refund_receipts
        == {
            identity.canonical_task_cid: (
                route_result.provider_receipt.receipt_id
            )
        }
    )
    assert (
        second_state.production_provider_operational_refund_receipts
        == first_state.production_provider_operational_refund_receipts
    )
    assert first_state.typed_provider_malformed_refund_receipts == (
        prior_malformed
    )
    assert second_state.typed_provider_malformed_refund_receipts == (
        prior_malformed
    )
    queue_entry = daemon.task_queue.entries[
        daemon.task_queue.resolve_key(identity.canonical_task_cid)
    ]
    assert queue_entry.consecutive_failures == 0
    assert queue_entry.selection_penalty == 0
    assert queue_entry.cooldown_until == 0.0
    refund_events = [
        item
        for item in _events(daemon)
        if item.get("type")
        == "production_provider_operational_attempt_refunded"
    ]
    assert len(refund_events) == 1
    assert refund_events[0]["receipt_id"] == (
        route_result.provider_receipt.receipt_id
    )


def test_restart_preserves_current_nonconsuming_codex_failure_attempt_count(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prospectively restored attempt-3 receipt cannot receive another refund."""

    daemon = _daemon(tmp_path, monkeypatch)
    task, identity, route_result, _receipt_path = (
        _persist_pre_policy_production_provider_failure(
            daemon,
            durable_attempt_count=2,
            failure_role="codex",
        )
    )
    assert [attempt.role for attempt in route_result.attempts] == [
        ProviderRole.GROK_IMPLEMENT,
        ProviderRole.CODEX_REVIEW,
    ]
    assert route_result.attempts[0].status == "succeeded"
    assert route_result.attempts[1].status == "failed"

    result = daemon.run_once()
    state = PortalTaskState.load(daemon.state_path)

    assert result.get("typed_provider_malformed_refunds", []) == []
    assert result.get("production_provider_operational_refunds", []) == []
    assert state.implementation_attempts[task.task_id] == 2
    assert state.implementation_attempts_by_cid[
        identity.canonical_task_cid
    ] == 2
    assert state.production_provider_operational_refund_receipts == {}


@pytest.mark.parametrize(
    ("corrupt_location", "corrupt_value"),
    [
        ("log_returncode", "1"),
        ("prompt_bytes", "1024"),
        ("response_bytes", []),
    ],
)
def test_restart_provider_failure_corrupt_numeric_evidence_declines_safely(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_location: str,
    corrupt_value: Any,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task, identity, _route_result, receipt_path = (
        _persist_pre_policy_production_provider_failure(
            daemon,
            log_returncode=(
                corrupt_value
                if corrupt_location == "log_returncode"
                else 1
            ),
        )
    )
    if corrupt_location != "log_returncode":
        _rewrite_provider_receipt_numeric_field(
            receipt_path,
            field_name=corrupt_location,
            value=corrupt_value,
        )

    result = daemon.run_once()
    state = PortalTaskState.load(daemon.state_path)

    assert result["typed_provider_malformed_refunds"] == []
    assert result["production_provider_operational_refunds"] == []
    assert state.implementation_attempts[task.task_id] == 3
    assert state.implementation_attempts_by_cid[
        identity.canonical_task_cid
    ] == 3
    assert state.production_provider_operational_refund_receipts == {}
    assert not [
        item
        for item in _events(daemon)
        if item.get("type")
        == "production_provider_operational_attempt_refunded"
    ]
