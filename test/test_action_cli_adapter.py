"""Sandbox, injection, and environment-leak tests for the CLI action adapter."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from ipfs_accelerate_py.action_runtime import (
    ActionCatalog,
    ActionDescriptor,
    ActionExecutor,
    ActionProposal,
    ActionStatus,
    FailClosedPolicy,
    RiskClass,
    SideEffectClass,
    propose_from_voice_route,
)
from ipfs_accelerate_py.action_runtime.adapters.cli import (
    CLIActionAdapter,
    CLIActionRegistration,
    CLISandboxPolicy,
    build_argv,
)
from ipfs_accelerate_py.action_runtime.catalog import resolve_reviewed_executable
from ipfs_accelerate_py.action_runtime.contracts import ActionDecisionKind
from ipfs_accelerate_py.action_runtime.voice_bridge import VoiceActionBridge
from ipfs_accelerate_py.voice_router import VoiceResponsePlan, VoiceTurnRequest, process_voice_turn


TRUE_BIN = Path(shutil.which("true") or "/usr/bin/true")
ECHO_BIN = Path(shutil.which("echo") or "/usr/bin/echo")


def _descriptor(descriptor_id: str = "voice.cli.open_app_surface.v1") -> ActionDescriptor:
    return ActionDescriptor(
        descriptor_id=descriptor_id,
        logical_action="open_app_surface",
        adapter="cli",
        risk_class=RiskClass.READ,
        side_effect_class=SideEffectClass.LOCAL_READ,
        requires_confirmation=True,
        allowed_channels=("voice", "chat", "test"),
        allowed_tenants=("*",),
    )


def _registration(
    descriptor_id: str = "voice.cli.open_app_surface.v1",
    *,
    executable: Path = TRUE_BIN,
    fixed_argv_prefix: tuple[str, ...] = (),
    argument_slots: tuple[str, ...] = (),
    sandbox: CLISandboxPolicy | None = None,
) -> CLIActionRegistration:
    return CLIActionRegistration(
        descriptor_id=descriptor_id,
        executable=executable,
        fixed_argv_prefix=fixed_argv_prefix,
        argument_slots=argument_slots,
        sandbox=sandbox or CLISandboxPolicy(timeout_seconds=2.0),
    )


def _proposal(
    descriptor_id: str = "voice.cli.open_app_surface.v1",
    **kwargs: object,
) -> ActionProposal:
    base = {
        "proposal_id": "prop-test-1",
        "descriptor_id": descriptor_id,
        "logical_action": "open_app_surface",
        "arguments": {},
        "route": "app_surface_navigation",
        "channel": "voice",
        "tenant_id": "211-ai",
    }
    base.update(kwargs)
    return ActionProposal(**base)  # type: ignore[arg-type]


def test_resolve_reviewed_executable_rejects_relative_and_pack_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        resolve_reviewed_executable("true")
    with pytest.raises(ValueError, match="absolute"):
        resolve_reviewed_executable("./bin/evil")
    missing = tmp_path / "missing-bin"
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_reviewed_executable(missing)


def test_build_argv_rejects_injection_markers() -> None:
    reg = _registration(argument_slots=("surface",))
    with pytest.raises(ValueError, match="injection|disallowed|characters"):
        build_argv(reg, {"surface": "wallet;rm"})
    with pytest.raises(ValueError, match="injection|disallowed|characters"):
        build_argv(reg, {"surface": "wallet && id"})
    with pytest.raises(ValueError, match="injection|disallowed|characters"):
        build_argv(reg, {"surface": "$(whoami)"})
    with pytest.raises(ValueError, match="injection|disallowed|characters"):
        build_argv(reg, {"surface": "a b"})
    with pytest.raises(ValueError, match="traversal"):
        build_argv(reg, {"surface": "../etc/passwd"})
    with pytest.raises(ValueError, match="unexpected"):
        build_argv(reg, {"surface": "wallet", "extra": "nope"})


def test_build_argv_accepts_safe_slot() -> None:
    reg = _registration(executable=ECHO_BIN, fixed_argv_prefix=("-n",), argument_slots=("surface",))
    argv = build_argv(reg, {"surface": "wallet_docs"})
    assert argv[0] == str(ECHO_BIN.resolve())
    assert argv[-1] == "wallet_docs"


def test_sandbox_policy_rejects_secret_env_keys() -> None:
    with pytest.raises(ValueError, match="secret-shaped"):
        CLISandboxPolicy(allowed_env={"HF_TOKEN": "x"})


def test_fail_closed_policy_denies_unknown_and_requires_grant() -> None:
    catalog = ActionCatalog([_descriptor()])
    policy = FailClosedPolicy(catalog=catalog)
    unknown = _proposal(descriptor_id="not.registered")
    decision = policy.decide(unknown)
    assert decision.kind is ActionDecisionKind.DENY
    assert decision.reason == "unknown_descriptor"

    known = _proposal()
    decision = policy.decide(known)
    assert decision.kind is ActionDecisionKind.CONFIRM
    assert decision.reason == "confirmation_required"
    assert not decision.permits_execution


def test_cli_adapter_does_not_execute_without_permit() -> None:
    catalog = ActionCatalog([_descriptor()])
    policy = FailClosedPolicy(catalog=catalog)
    adapter = CLIActionAdapter([_registration()])
    executor = ActionExecutor(catalog=catalog, policy=policy, cli_adapter=adapter)

    decision, receipt = executor.execute(_proposal())
    assert not decision.permits_execution
    assert receipt.status is ActionStatus.DENIED


def test_cli_adapter_executes_true_with_explicit_grant() -> None:
    catalog = ActionCatalog([_descriptor()])
    policy = FailClosedPolicy(catalog=catalog)
    policy.grant(descriptor_id="voice.cli.open_app_surface.v1")
    adapter = CLIActionAdapter([_registration()])
    executor = ActionExecutor(catalog=catalog, policy=policy, cli_adapter=adapter)

    decision, receipt = executor.execute(_proposal())
    assert decision.permits_execution
    assert receipt.status is ActionStatus.SUCCEEDED
    assert receipt.exit_code == 0
    assert receipt.adapter == "cli"
    assert receipt.stdout_digest
    assert "stdout" not in receipt.public_result  # raw output not leaked


def test_environment_isolation_with_helper_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "should-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "also-should-not-leak")

    helper = tmp_path / "check_env.py"
    helper.write_text(
        "import json,os\n"
        "keys=sorted(os.environ)\n"
        "print(json.dumps({"
        "'has_secret': any('SECRET' in k or 'TOKEN' in k for k in keys),"
        "'keys': keys}))\n",
        encoding="utf-8",
    )
    import sys

    # Executable is python absolute path; script path as fixed argv is absolute and safe.
    script_token = str(helper.resolve())
    # Slots/paths with only safe charset - absolute paths use / and alnum.
    catalog = ActionCatalog([_descriptor()])
    policy = FailClosedPolicy(catalog=catalog)
    policy.grant(descriptor_id="voice.cli.open_app_surface.v1")
    reg = CLIActionRegistration(
        descriptor_id="voice.cli.open_app_surface.v1",
        executable=sys.executable,
        fixed_argv_prefix=(script_token,),
        sandbox=CLISandboxPolicy(timeout_seconds=3.0, isolate_environment=True),
    )
    adapter = CLIActionAdapter([reg])
    executor = ActionExecutor(catalog=catalog, policy=policy, cli_adapter=adapter)
    decision, receipt = executor.execute(_proposal())
    assert decision.permits_execution
    assert receipt.status is ActionStatus.SUCCEEDED, receipt.to_dict()
    # Child should not inherit ambient SECRET/TOKEN keys. We only have digests;
    # re-run with a recording runner assertion via ProcessRunner direct check.
    from ipfs_accelerate_py.cli_runtime.process_runner import ProcessRunner, ProcessSpec

    runner = ProcessRunner(base_env={})
    result = runner.run(
        ProcessSpec(
            argv=[str(Path(sys.executable).resolve()), script_token],
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            env_overlay=False,
            timeout_seconds=3.0,
        )
    )
    payload = json.loads(result.stdout)
    assert payload["has_secret"] is False
    assert "SUPER_SECRET_TOKEN" not in payload["keys"]
    assert "AWS_SECRET_ACCESS_KEY" not in payload["keys"]


def test_proposal_rejects_executable_smuggling() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        ActionProposal(
            proposal_id="p1",
            descriptor_id="d1",
            logical_action="open_app_surface",
            arguments={"executable": "/bin/sh"},
        )
    with pytest.raises(ValueError, match="not allowed"):
        ActionProposal(
            proposal_id="p1",
            descriptor_id="d1",
            logical_action="open_app_surface",
            arguments={"command": "rm -rf /"},
        )


def test_voice_bridge_proposes_app_surface_from_library_route() -> None:
    catalog = ActionCatalog([_descriptor()])
    bridge = VoiceActionBridge(catalog=catalog)
    proposal = bridge.propose(
        route="app_surface_navigation",
        transcript="Open the wallet documents surface",
        template_id="lib-frame-1",
        channel="voice",
        confidence=0.91,
    )
    assert proposal is not None
    assert proposal.descriptor_id == "voice.cli.open_app_surface.v1"
    assert proposal.logical_action == "open_app_surface"
    assert "executable" not in proposal.arguments


def test_voice_library_route_plus_cli_adapter_end_to_end() -> None:
    """Response-library route proposes; grant permits; CLI true runs."""

    dag_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "phone_dialog_generation"
        / "slotted_response_dag.json"
    )
    if not dag_path.is_file():
        pytest.skip("slotted response DAG library not present")

    exemplars = json.loads(dag_path.read_text(encoding="utf-8"))["nodes"]["uniqueExemplars"]
    sample = next(ex for ex in exemplars if ex.get("route") == "app_surface_navigation")
    user = sample["user"]
    assistant = " ".join(sample["assistant"].replace("**", "").split())

    class _Templates:
        def retrieve(self, transcript: str, **_: object) -> VoiceResponsePlan:
            return VoiceResponsePlan(
                template_id=sample["id"],
                template=assistant,
                metadata={"route": "app_surface_navigation", "recordId": sample["recordId"]},
            )

    import io
    import struct
    import wave

    def _wav() -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            samples = b"".join(
                struct.pack("<h", 1_000 if i % 2 else -1_000) for i in range(160)
            )
            handle.writeframes(samples)
        return buf.getvalue()

    class _TTS:
        def synthesize(self, text: str, **_: object) -> bytes:
            return _wav()

    voice_result = process_voice_turn(
        VoiceTurnRequest(transcript=user, request_id="lib-cli-e2e"),
        template_provider=_Templates(),
        tts_provider=_TTS(),
    )
    assert voice_result.status == "completed"
    assert voice_result.response_text == assistant

    catalog = ActionCatalog([_descriptor()])
    policy = FailClosedPolicy(catalog=catalog)
    bridge = VoiceActionBridge(catalog=catalog)
    proposal = bridge.propose(
        route="app_surface_navigation",
        transcript=user,
        template_id=sample["id"],
        channel="voice",
        confidence=0.95,
        evidence=tuple(sample.get("evidenceDocIds") or ()),
    )
    assert proposal is not None

    # Without grant: confirm/deny only.
    decision = policy.decide(proposal)
    assert not decision.permits_execution

    # With grant: CLI executes.
    policy.grant(proposal_id=proposal.proposal_id)
    adapter = CLIActionAdapter([_registration()])
    executor = ActionExecutor(catalog=catalog, policy=policy, cli_adapter=adapter)
    decision, receipt = executor.execute(proposal)
    assert decision.permits_execution
    assert receipt.status is ActionStatus.SUCCEEDED
    assert receipt.exit_code == 0


def test_route_without_mapping_returns_none() -> None:
    assert propose_from_voice_route(route="grounded_211_answer") is None
    assert propose_from_voice_route(route="live_agent") is None
