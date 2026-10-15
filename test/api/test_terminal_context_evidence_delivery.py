"""Closed replay-bundle transport with authored counters and cancellable doubles.

These archives exercise the real allowlist/hash verifier, not native admission,
provider execution, Docker, or historical audit success.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import shlex
import tarfile
from pathlib import Path

import pytest
from harbor.models.agent.context import AgentContext

from benchmarks.agent_supervisor.container_coding import terminal_context_evidence as evidence
from test.api.test_terminal_adapter_deadline import Clock, Environment, _agent, _install, _report, adapter


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def _bundle(state, *, private_member=False):
    leaves = {
        "replay-request.json": ("replay-request", b'{"receipts":[],"decision_report":{}}'),
        "state/prepared.json": ("prepared-state", b"{}"),
        "repository/.runtime/raw-parent.json": ("raw-retrieval", b'{"authored":true}'),
    }
    if private_member:
        leaves["state/native-rollout.json"] = ("provider-transcript", b"PRIVATE-TRANSCRIPT-MARKER")
    manifest = {"schema": evidence.SCHEMA, "profile": evidence.PROFILE,
        "repository_origin": "/app", "state_origin": state,
        "workspace_root_origin": "/opt/ipfs-supervisor/worktrees",
        "native_prompt_bindings": [], "public_baseline_commit": "0" * 40,
        "public_source_count": 0,
        "files": [{"path": name, "role": role, "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}
                  for name, (role, raw) in sorted(leaves.items())],
        "content_bytes": sum(len(raw) for _, raw in leaves.values()), "provider_calls": 0,
        "current_source_freshness_claimed": False, "private_state_tree_copied": False,
        "provider_transcripts_included": False, "execution_authority": False,
        "completion_authority": False}
    encoded = _encode(manifest)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        for name, raw in [("manifest.json", encoded),
                          *[(name, raw) for name, (_, raw) in sorted(leaves.items())]]:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            stream.addfile(member, io.BytesIO(raw))
    raw = archive.getvalue()
    receipt = {"schema": evidence.RECEIPT_SCHEMA, "status": "exported",
        "artifact": state + "-context-evidence.tar", "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest_bytes": len(encoded), "file_count": len(leaves),
        "content_bytes": manifest["content_bytes"], "repository_origin": "/app",
        "state_origin": state, "workspace_root_origin": "/opt/ipfs-supervisor/worktrees",
        "profile": evidence.PROFILE, "provider_calls": 0,
        "execution_authority": False, "completion_authority": False}
    return raw, receipt


def _complete_report():
    report = _report(usage=True)
    report["task_completed"] = True
    native = report["provider_invocations"][0]["native_rollout_usage"]
    native.update(task_complete_observed=True, usage_lifecycle_complete=True,
                  completion_basis="native_lifecycle_after_latest_usage@1", lifecycle_malformed_records=0)
    return report


class EvidenceEnvironment(Environment):
    def __init__(self, clock, *, change=None, private_member=False, **kwargs):
        super().__init__(clock, report=_complete_report(), **kwargs)
        self.change = change
        self.private_member = private_member
        self.bundle = None
        self.state = None

    async def exec(self, **kwargs):
        argv = shlex.split(kwargs["command"])
        self.state = argv[argv.index("--state") + 1]
        self.bundle, receipt = _bundle(self.state, private_member=self.private_member)
        if self.change is not None:
            self.change(receipt)
        self.report["context_replay_evidence"] = receipt
        return await super().exec(**kwargs)

    async def download_file(self, source, target):
        if source == self.state + "-context-evidence.tar":
            self.downloads.append(source)
            await self.perform("context_replay_evidence_export")
            Path(target).write_bytes(self.bundle)
        else:
            await super().download_file(source, target)


def _run(tmp_path, monkeypatch, environment):
    _install(monkeypatch, environment.clock, environment)
    agent, context = _agent(tmp_path), AgentContext()
    asyncio.run(agent.run("Literal public instruction", environment, context))
    return agent, context


def _assert_accounting_retained(context):
    assert context.metadata["task_completed"] is True
    assert context.metadata["usage"]["observed_complete_sessions"] is True
    assert context.metadata["usage"]["total_tokens"] == 120
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (100, 80, 20)
    assert context.metadata["signed_admission_exported"] is True
    assert context.metadata["public_output_evidence"]["status"] == "authored public output"


def test_verified_closed_archive_is_retained_without_rewriting_report(tmp_path, monkeypatch):
    clock = Clock()
    costs = {"instruction_upload": 1, "public_input_capture": 2, "driver": 250,
             "report_export": 4, "context_replay_evidence_export": 4,
             "admission_export": 3, "public_output_export": 2}
    environment = EvidenceEnvironment(clock, costs=costs)
    agent, context = _run(tmp_path, monkeypatch, environment)
    retained = context.metadata["context_replay_evidence"]
    assert retained["verified"] is True and retained["status"] == "delivered"
    archive = Path(retained["artifact"])
    assert archive == agent.logs_dir / "context-replay-evidence.tar"
    assert archive.read_bytes() == environment.bundle
    assert archive.stat().st_mode & 0o777 == 0o444 and archive.stat().st_nlink == 1
    assert json.loads((agent.logs_dir / "context-replay-evidence-receipt.json").read_text()) == retained
    assert json.loads((agent.logs_dir / "supervisor-result.json").read_text()) == environment.report
    assert environment.report["context_replay_evidence"]["status"] == "exported"
    assert environment.downloads == [environment.state + "-result.json",
        environment.state + "-context-evidence.tar", environment.state + "/admission.json"]
    assert environment.events == list(costs)
    deadline = context.metadata["adapter_deadline"]
    assert deadline["finalization_reserve_seconds"] == 25
    assert deadline["seconds"] == sum(costs.values()) == 266
    assert deadline["stages"]["context_replay_evidence_export"]["timeout_seconds"] == 5
    assert not list(agent.logs_dir.glob(".context-replay-evidence-*.tar"))
    _assert_accounting_retained(context)


@pytest.mark.parametrize("changes", [
    {"artifact": "/opt/ipfs-supervisor/state/foreign-context-evidence.tar"},
    {"state_origin": "/opt/ipfs-supervisor/state/foreign"},
    {"execution_authority": True}, {"bytes": evidence.MAX_ARCHIVE_BYTES + 1},
    {"sha256": "PRIVATE-DIGEST-MARKER"}, {"private_field": "PRIVATE-RECEIPT-MARKER"},
])
def test_invalid_owner_receipt_is_rejected_before_any_bundle_download(tmp_path, monkeypatch, changes):
    environment = EvidenceEnvironment(Clock(), change=lambda receipt: receipt.update(changes))
    agent, context = _run(tmp_path, monkeypatch, environment)
    assert environment.state + "-context-evidence.tar" not in environment.downloads
    assert context.metadata["context_replay_evidence"]["verified"] is False
    assert context.metadata["context_replay_evidence"]["error_type"] == "ValueError"
    assert "PRIVATE-" not in json.dumps(context.metadata["context_replay_evidence"])
    assert not (agent.logs_dir / "context-replay-evidence.tar").exists()
    _assert_accounting_retained(context)


@pytest.mark.parametrize("changes", [{"sha256": "0" * 64}, {"manifest_sha256": "0" * 64},
    {"bytes": 1}, {"manifest_bytes": 1}, {"file_count": 1}, {"content_bytes": 1}])
def test_hash_size_and_manifest_receipt_mismatches_do_not_publish(tmp_path, monkeypatch, changes):
    environment = EvidenceEnvironment(Clock(), change=lambda receipt: receipt.update(changes))
    agent, context = _run(tmp_path, monkeypatch, environment)
    assert environment.state + "-context-evidence.tar" in environment.downloads
    assert context.metadata["context_replay_evidence"]["verified"] is False
    assert not (agent.logs_dir / "context-replay-evidence.tar").exists()
    assert not (agent.logs_dir / "context-replay-evidence-receipt.json").exists()
    assert not list(agent.logs_dir.glob(".context-replay-evidence-*.tar"))
    _assert_accounting_retained(context)


def test_valid_digest_cannot_admit_provider_transcript_member(tmp_path, monkeypatch):
    environment = EvidenceEnvironment(Clock(), private_member=True)
    agent, context = _run(tmp_path, monkeypatch, environment)
    assert context.metadata["context_replay_evidence"]["verified"] is False
    assert "PRIVATE-TRANSCRIPT-MARKER" not in json.dumps(context.metadata)
    assert not (agent.logs_dir / "context-replay-evidence.tar").exists()
    assert not list(agent.logs_dir.glob(".context-replay-evidence-*.tar"))
    _assert_accounting_retained(context)


def test_owner_capture_unavailable_preserves_completion_usage_without_download(tmp_path, monkeypatch):
    def unavailable(receipt):
        receipt.clear()
        receipt.update(schema=evidence.RECEIPT_SCHEMA, status="unavailable",
                       reason="export_timed_out", error_type="TimeoutError")
    environment = EvidenceEnvironment(Clock(), change=unavailable)
    _, context = _run(tmp_path, monkeypatch, environment)
    assert "context_replay_evidence_export" not in environment.events
    assert context.metadata["context_replay_evidence"] == {
        "status": "unavailable", "reason": "owner_export_unavailable", "verified": False}
    _assert_accounting_retained(context)


@pytest.mark.parametrize("error", [TimeoutError("PRIVATE-TRANSPORT-MARKER"),
                                  OSError("PRIVATE-TRANSPORT-MARKER")])
def test_download_failure_preserves_completion_usage_and_remaining_exports(tmp_path, monkeypatch, error):
    environment = EvidenceEnvironment(Clock(), failures={"context_replay_evidence_export": error})
    _, context = _run(tmp_path, monkeypatch, environment)
    assert context.metadata["context_replay_evidence"]["verified"] is False
    assert context.metadata["context_replay_evidence"]["error_type"] == type(error).__name__
    assert "PRIVATE-TRANSPORT-MARKER" not in json.dumps(context.metadata)
    assert environment.events[-2:] == ["admission_export", "public_output_export"]
    _assert_accounting_retained(context)


def test_delivery_starts_no_download_when_shared_budget_is_reserved(tmp_path, monkeypatch):
    environment = EvidenceEnvironment(Clock(), costs={"driver": 281, "report_export": 4})
    _, context = _run(tmp_path, monkeypatch, environment)
    assert "context_replay_evidence_export" not in environment.events
    stage = context.metadata["adapter_deadline"]["stages"]["context_replay_evidence_export"]
    assert stage["status"] == "not_started_budget_exhausted"
    assert context.metadata["context_replay_evidence"]["error_type"] == "_AdapterBudgetExhausted"
    _assert_accounting_retained(context)


def test_delivery_cancellation_stops_transport_finalizes_and_reraises(tmp_path, monkeypatch):
    entered = asyncio.Event()
    stopped = []

    class CancellableEnvironment(EvidenceEnvironment):
        async def download_file(self, source, target):
            if source == self.state + "-context-evidence.tar":
                Path(target).write_bytes(b"authored incomplete transfer")
            await super().download_file(source, target)

        async def perform(self, name):
            await super().perform(name)
            if name == "context_replay_evidence_export":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.append(name)

    environment = CancellableEnvironment(Clock())
    _install(monkeypatch, environment.clock, environment)
    agent, context = _agent(tmp_path), AgentContext()

    async def scenario():
        task = asyncio.create_task(agent.run("Literal public instruction", environment, context))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not [other for other in asyncio.all_tasks() if other is not asyncio.current_task()]

    asyncio.run(scenario())
    assert stopped == ["context_replay_evidence_export"]
    assert context.metadata["adapter_deadline"]["stages"]["context_replay_evidence_export"]["status"] == "cancelled"
    assert context.metadata["context_replay_evidence"]["verified"] is False
    assert not (agent.logs_dir / "context-replay-evidence.tar").exists()
    assert not list(agent.logs_dir.glob(".context-replay-evidence-*.tar"))
    _assert_accounting_retained(context)


def test_actual_delivery_wait_timeout_stops_partial_transfer_and_retains_usage(tmp_path, monkeypatch):
    stopped = []

    class StalledEnvironment(EvidenceEnvironment):
        async def download_file(self, source, target):
            if source != self.state + "-context-evidence.tar":
                return await super().download_file(source, target)
            self.downloads.append(source)
            await self.perform("context_replay_evidence_export")
            Path(target).write_bytes(b"authored incomplete transfer")
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(True)

    environment = StalledEnvironment(Clock())
    monkeypatch.setattr(adapter, "PUBLIC_EXPORT_SECONDS", 0.01)
    agent, context = _run(tmp_path, monkeypatch, environment)
    assert stopped == [True]
    assert context.metadata["adapter_deadline"]["stages"]["context_replay_evidence_export"]["status"] == "timed_out"
    assert context.metadata["context_replay_evidence"]["error_type"] == "TimeoutError"
    assert context.metadata["context_replay_evidence"]["verified"] is False
    assert not (agent.logs_dir / "context-replay-evidence.tar").exists()
    assert not list(agent.logs_dir.glob(".context-replay-evidence-*.tar"))
    _assert_accounting_retained(context)


@pytest.mark.parametrize("name", ["context-replay-evidence.tar", "context-replay-evidence-receipt.json"])
def test_existing_artifact_is_never_replaced_or_used_as_current_evidence(tmp_path, monkeypatch, name):
    environment = EvidenceEnvironment(Clock())
    _install(monkeypatch, environment.clock, environment)
    agent, context = _agent(tmp_path), AgentContext()
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    previous = agent.logs_dir / name
    previous.write_bytes(b"authored prior evidence")
    asyncio.run(agent.run("Literal public instruction", environment, context))
    assert previous.read_bytes() == b"authored prior evidence"
    assert environment.state + "-context-evidence.tar" not in environment.downloads
    assert context.metadata["context_replay_evidence"]["verified"] is False
    _assert_accounting_retained(context)
