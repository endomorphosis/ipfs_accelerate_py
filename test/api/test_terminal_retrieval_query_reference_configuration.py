"""Retrieval query opt-in reaches signed task-bound owner custody.

Authored public bundles, archive bytes and process doubles exercise the actual
option/admission APIs. No provider, deployment or task verifier is invoked.
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext

from benchmarks.agent_supervisor.container_coding import (
    full_supervisor_benchmark as benchmark,
    full_supervisor_harbor_agent as adapter,
    terminal_container_supervisor as driver,
    terminal_doctor_dispatch as doctor,
)
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from test.api.test_terminal_decision_archive import _decision_runtime_archive
from test.api.test_terminal_prompt_tokenizer_forwarding import supervisor
from test.api.test_terminal_timeout_configuration import _authored_prepare_dataset

NOMINATION = {"semantic_decision_nomination_artifact": "/owner/nomination.json",
              "semantic_decision_nomination_sha256": "a" * 64,
              "disable_semantic_typed_nomination": True}
REFERENCE = {"semantic_retrieval_query_reference": True}
SIGNED = {"semantic_decision_context_bundle_artifact": ".runtime/context bundle.json",
          "semantic_decision_context_bundle_sha256": "b" * 64,
          "semantic_decision_task_cid": "admitted-task-cid"}
COMMAND = {"router": Path("/router"), "model": "pinned-model", "reasoning": "high",
           "timeout": 30, "semantic_repository": Path("/app"), "doctor": None}


def test_false_optin_keeps_default_and_plain_nomination_shapes(tmp_path):
    for arm in ("full", "no-index"):
        args = (tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", arm)
        assert benchmark._prompt_options(arm, semantic_retrieval_query_reference=False) == benchmark._prompt_options(arm)
        assert benchmark.config_for(*args, semantic_retrieval_query_reference=False) == benchmark.config_for(*args)
    ordinary = doctor.implementation_argv(**COMMAND, **NOMINATION, semantic_decision_task_id="TASK")
    assert doctor.implementation_argv(**COMMAND, **NOMINATION, semantic_decision_task_id="TASK",
        semantic_retrieval_query_reference=False) == ordinary
    assert not any("context-bundle" in arg or arg == "--semantic-decision-task-cid" for arg in ordinary)


@pytest.mark.parametrize("host_source", [False, True])
def test_optin_options_are_closed_and_retained_change_is_detected(tmp_path, host_source):
    nomination = dict(NOMINATION)
    if host_source:
        nomination.pop("semantic_decision_nomination_artifact")
        nomination["semantic_decision_nomination_source"] = "/host/frozen-nomination.json"
    args = (tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full")
    plain = benchmark.config_for(*args, **nomination)
    chosen = benchmark.config_for(*args, **nomination, **REFERENCE)
    assert chosen["agents"][0]["kwargs"] == {**plain["agents"][0]["kwargs"], **REFERENCE}
    options = benchmark._prompt_options("full", **nomination, **REFERENCE)
    prepared = {"arm": "full", "prompt_comparison_options": options}
    observed = benchmark._prompt_option_observation(prepared, chosen["agents"][0]["kwargs"])
    assert observed["prompt_comparison_config_unchanged"] is True
    assert observed["semantic_retrieval_query_reference_requested"] is True
    changed = benchmark._prompt_option_observation(prepared, plain["agents"][0]["kwargs"])
    assert changed["prompt_comparison_options_valid"] is True
    assert changed["prompt_comparison_config_unchanged"] is False
    assert "semantic_retrieval_query_reference_requested" not in changed


@pytest.mark.parametrize("arm,options", [
    ("full", {**NOMINATION, "semantic_retrieval_query_reference": value})
    for value in (None, 0, 1, "true")
] + [
    ("no-index", {**NOMINATION, **REFERENCE}),
    ("full", REFERENCE),
    ("full", {**NOMINATION, **REFERENCE, "disable_semantic_typed_nomination": False}),
    ("full", {**REFERENCE, "semantic_decision_policy_artifact": "/owner/policy.json",
              "semantic_decision_policy_sha256": "b" * 64}),
    ("full", {**NOMINATION, **REFERENCE, "semantic_native_input": True}),
    ("full", {**NOMINATION, **REFERENCE, "semantic_context_projection": True,
              "semantic_slot_policy_revision": "closed-typed-identifiers@2"}),
])
def test_invalid_optin_refuses_before_host_or_driver_effects(tmp_path, monkeypatch, arm, options):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid retrieval option reached source, archive, state or process work")
    unread = SimpleNamespace(resolve=forbidden)
    monkeypatch.setattr(benchmark, "read_nomination_source", forbidden)
    monkeypatch.setattr(benchmark.subprocess, "run", forbidden)
    monkeypatch.setattr(driver.os, "geteuid", forbidden)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=unread, output=tmp_path / "output", archive=unread, arm=arm, **options)
    with pytest.raises(ValueError):
        adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
            runtime_archive=str(tmp_path / "missing"), arm=arm, **options)
    with pytest.raises(ValueError):
        driver.run(instruction=Path("/instruction"), state=Path("/state"), arm=arm, **options)
    assert not (tmp_path / "output").exists() and not (tmp_path / "logs").exists()


def test_signed_owner_receives_one_exact_bundle_and_admitted_task_group():
    plain = doctor.implementation_argv(**COMMAND, **NOMINATION, semantic_decision_task_id="TASK")
    argv = doctor.implementation_argv(**COMMAND, **NOMINATION, **REFERENCE, **SIGNED,
                                      semantic_decision_task_id="TASK")
    assert argv == plain + ["--semantic-decision-context-bundle-artifact", SIGNED["semantic_decision_context_bundle_artifact"],
        "--semantic-decision-context-bundle-sha256", SIGNED["semantic_decision_context_bundle_sha256"],
        "--semantic-decision-task-cid", SIGNED["semantic_decision_task_cid"]]
    assert shlex.split(shlex.join(argv)) == argv
    assert "--semantic-retrieval-query-reference" not in argv
    assert argv[argv.index("--semantic-decision-typed-nomination") + 1] == "disabled"


@pytest.mark.parametrize("changes", [
    {key: None} for key in SIGNED
] + [
    {"semantic_decision_context_bundle_artifact": value}
    for value in ("/app/bundle.json", "../bundle.json", ".runtime/../bundle.json", ".runtime//bundle.json", "bundle.json/")
] + [
    {"semantic_decision_context_bundle_sha256": "B" * 64},
    {"semantic_decision_task_cid": " foreign "},
    {"semantic_retrieval_query_reference": False},
    {"disable_semantic_typed_nomination": False},
])
def test_signed_inputs_refuse_missing_noncanonical_or_unopted_group(changes):
    with pytest.raises(ValueError):
        doctor.implementation_argv(**COMMAND, semantic_decision_task_id="TASK",
                                  **({**NOMINATION, **REFERENCE, **SIGNED, **changes}))


def _bundle(root, *, task_cid="task", task_id="TASK", metadata=None):
    root.mkdir(parents=True, exist_ok=True)
    return write_task_context_bundle(repository=root, output=root / ".runtime/context-bundle.json",
        prepared=[{"schema": "supervisor-task-context-preparation@1", "task_cid": task_cid,
                   "task_id": task_id, "metadata": metadata if metadata is not None else {
                       "Code retrieval artifact": ".runtime/pinned-retrieval.json",
                       "Code retrieval sha256": "e" * 64}}])


@pytest.mark.parametrize("fault", [None, "task-cid", "task-key", "bundle-sha", "bundle-bytes", "no-retrieval"])
def test_actual_bundle_authentication_derives_owner_inputs_without_raw_self_discovery(tmp_path, fault):
    root = tmp_path / "source"
    bundle = _bundle(root, metadata={} if fault == "no-retrieval" else None)
    kwargs = {"repository": root, "bundle": dict(bundle), "task_cid": "task", "task_id": "TASK"}
    if fault == "task-cid": kwargs["task_cid"] = "foreign"
    elif fault == "task-key": kwargs["task_id"] = "FOREIGN"
    elif fault == "bundle-sha": kwargs["bundle"]["sha256"] = "0" * 64
    elif fault == "bundle-bytes": (root / bundle["artifact"]).write_text("{}")
    if fault:
        with pytest.raises(ValueError): driver._retrieval_query_owner_bindings(**kwargs)
    else:
        assert driver._retrieval_query_owner_bindings(**kwargs) == {
            "semantic_decision_context_bundle_artifact": bundle["artifact"],
            "semantic_decision_context_bundle_sha256": bundle["sha256"], "semantic_decision_task_cid": "task"}
    assert not (root / ".runtime/pinned-retrieval.json").exists()


def test_actual_driver_derives_signed_group_from_authenticated_context_after_admission(supervisor, tmp_path, monkeypatch):
    state, planned, signed = supervisor
    public_root = tmp_path / "public-source"
    bundle = _bundle(public_root)
    monkeypatch.setattr(driver.preparation, "context", lambda **_: {"context_bundle": bundle})
    actual = driver._retrieval_query_owner_bindings
    calls = []
    def authenticate(**kwargs):
        calls.append(kwargs)
        assert kwargs["repository"] == Path("/app")
        return actual(**{**kwargs, "repository": public_root})
    monkeypatch.setattr(driver, "_retrieval_query_owner_bindings", authenticate)
    result = driver.run(instruction=Path("/instruction"), state=state, arm="full",
        doctor_dispatch_policy="model-only", **NOMINATION, **REFERENCE)
    assert result["task_completed"] is True and len(planned) == len(signed) == len(calls) == 1
    assert calls[0]["bundle"] == bundle and calls[0]["task_cid"] == "task" and calls[0]["task_id"] == "TASK"
    assert result["semantic_retrieval_query_owner_context"] == {
        "context_bundle": bundle, "task_cid": "task", "task_id": "TASK"}
    for flag, expected in (("--semantic-decision-context-bundle-artifact", bundle["artifact"]),
                           ("--semantic-decision-context-bundle-sha256", bundle["sha256"]),
                           ("--semantic-decision-task-cid", "task")):
        assert signed[0].count(flag) == 1 and signed[0][signed[0].index(flag) + 1] == expected
        assert flag not in planned[0]


@pytest.mark.parametrize("selected", [False, True])
def test_harbor_forwards_optin_once_without_unsigned_bundle_inputs(tmp_path, monkeypatch, selected):
    archive = tmp_path / "archive"; archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({"archive_sha256": "authored", "learned_requirements": []}))
    commands = []
    class Environment:
        async def upload_file(self, source, target): pass
        async def exec(self, **kwargs):
            commands.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(stdout="", stderr="", return_code=0)
        async def download_file(self, source, target): raise FileNotFoundError(source)
    async def capture(*args, **kwargs): return {"status": "authored"}
    monkeypatch.setattr(adapter, "capture_public_inputs", capture)
    monkeypatch.setattr(adapter, "export_public_outputs", capture)
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
        runtime_archive=str(archive), arm="full", semantic_retrieval_query_reference=selected, **NOMINATION)
    context = AgentContext()
    asyncio.run(agent.run("Authored public task.", Environment(), context))
    assert len(commands) == 1
    assert commands[0].count("--semantic-retrieval-query-reference") == int(selected)
    assert not any("context-bundle" in value or value == "--semantic-decision-task-cid" for value in commands[0])
    assert (context.metadata.get("semantic_retrieval_query_reference_requested") is True) is selected


def _reference_archive(tmp_path):
    archive, manifest = _decision_runtime_archive(tmp_path)
    with tarfile.open(archive / "runtime.tar.gz", "r:gz") as original:
        rows = [(m.name, original.extractfile(m).read()) for m in original]
    extra = {k: v for k, v in benchmark._decision_runtime_sources(retrieval_query_reference=True).items()
             if k not in benchmark._decision_runtime_sources()}
    for name, source in extra.items():
        raw = source.read_bytes(); rows.append((name, raw))
        manifest["files"].append({"path": name, "bytes": len(raw), "sha256": benchmark._hash(source), "mode": 0o644})
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as writer:
        for name, raw in rows:
            member = tarfile.TarInfo(name); member.size, member.mode = len(raw), 0o644
            writer.addfile(member, io.BytesIO(raw))
    manifest["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    benchmark._json(archive / "manifest.json", manifest)
    return archive


def test_optin_capabilities_are_extra_and_missing_archive_refuses_before_output(tmp_path, monkeypatch):
    plain = benchmark._decision_runtime_sources()
    chosen = benchmark._decision_runtime_sources(retrieval_query_reference=True)
    assert set(chosen) - set(plain) == {benchmark.DECISION_RUNTIME_ARCHIVE_PREFIX + name for name in (
        "semantic_retrieval_query_reference.py", "code_retrieval_context.py", "task_context_bundle.py")}
    archive, _ = _decision_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path); output = tmp_path / "output"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("missing retrieval sources reached dry-run"))
    with pytest.raises(ValueError, match="selected decision"):
        benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **NOMINATION, **REFERENCE)
    assert not output.exists()


@pytest.mark.parametrize("missing_source", [
    benchmark.DECISION_RUNTIME_ARCHIVE_PREFIX + "task_context_bundle.py",
    benchmark.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + "terminal_context_evidence.py",
])
def test_prepare_and_execute_pin_the_opted_in_current_sources(tmp_path, monkeypatch, missing_source):
    archive = _reference_archive(tmp_path); dataset = _authored_prepare_dataset(tmp_path); output = tmp_path / "prepared"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout="Authored dry-run.", stderr=""))
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **NOMINATION, **REFERENCE)
    assert prepared["runtime_decision_source_sha256"] == {
        k: benchmark._hash(v) for k, v in benchmark._decision_runtime_sources(retrieval_query_reference=True).items()}
    assert prepared["semantic_retrieval_query_reference_requested"] is True
    prepared["runtime_decision_source_sha256"].pop(missing_source)
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("missing required source pin reached invocation"))
    with pytest.raises(ValueError, match="decision source pins"): benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_selected_source_population_refuses_extra_source_before_hashing(tmp_path, monkeypatch):
    archive = _reference_archive(tmp_path)
    manifest = json.loads((archive / "manifest.json").read_text())
    selected = benchmark._decision_runtime_sources(retrieval_query_reference=True)
    extra = tmp_path / "extra.py"
    extra.write_text("# Authored source beyond the declared maximum.\n")
    selected["source/authored-extra.py"] = extra
    hashed = []
    original_hash = benchmark._hash
    def record_hash(source):
        hashed.append(Path(source))
        return original_hash(source)
    monkeypatch.setattr(benchmark, "_hash", record_hash)
    with pytest.raises(ValueError, match="selected decision"):
        benchmark._require_current_timeout_runtime(archive, manifest, selected_sources=selected)
    assert extra not in hashed


def test_cli_forwards_explicit_optin_and_keeps_default_keyword_absent(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"provider_calls": 0})
    argv = ["full-supervisor", "prepare", "--output", str(tmp_path / "output"),
            "--dataset", str(tmp_path / "dataset"), "--archive", str(tmp_path / "archive")]
    for selected in (False, True):
        monkeypatch.setattr(sys, "argv", argv + (["--semantic-retrieval-query-reference"] if selected else []))
        benchmark.main(); assert json.loads(capsys.readouterr().out) == {"provider_calls": 0}
        assert ("semantic_retrieval_query_reference" in calls[-1]) is selected
    driver_result = {"task_completed": True, "arm": "full", "seconds": 0, "provider_invocations": []}
    monkeypatch.setattr(driver, "run", lambda **kwargs: calls.append(kwargs) or driver_result)
    for selected in (False, True):
        monkeypatch.setattr(sys, "argv", ["terminal-driver", "--instruction", "/instruction", "--state", "/state", "--arm", "full"]
            + (["--semantic-retrieval-query-reference"] if selected else []))
        assert driver.main() == 0
        assert json.loads(capsys.readouterr().out) == driver_result
        assert ("semantic_retrieval_query_reference" in calls[-1]) is selected
