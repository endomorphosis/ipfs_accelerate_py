"""CLI custody and real native read-set composition, with no provider calls."""
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from benchmarks.agent_supervisor import repository_read_set as subject
from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
from ipfs_accelerate_py.agent_supervisor.runtime import repository_context_read_set as read_sets
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter, render_model_prompt

REQUIREMENTS = "Keep this exact authored intent/security/UI obligation café unresolved.\n"
POLICY = b"# MANDATORY_ORIGINAL_COMMENT\ndef preserve_policy():\n    return True\n"
TARGET = b"# SELECTED_ORIGINAL_COMMENT\ndef invoice_total(subtotal, tax):\n    return subtotal + tax\n"
PRIVATE = "PRIVATE_FAILURE_OR_SOURCE_MUST_NOT_APPEAR"


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "repository"
    root.mkdir(mode=0o700)
    (root / "policy.py").write_bytes(POLICY)
    (root / "ledger.py").write_bytes(TARGET)
    (root / "unrelated.py").write_text("# OMITTED_ORIGINAL_COMMENT\ndef unrelated():\n    return 0\n")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(REQUIREMENTS, encoding="utf-8")
    return dict(root=root, query="invoice_total", required_paths=["policy.py"],
        requirements_file=requirements, task_id="authored-read-set", provider="grok_cli",
        model="grok-4.7", max_input_tokens=100000, output=tmp_path / "delivery", limit=8)


@pytest.fixture
def native_index(tmp_path, monkeypatch):
    """Only admission is authored; the actual index/session/producer are used."""
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_index
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import policy_sha256
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        GlobalResourceScheduler, ResourceSchedulerConfig,
    )
    fixture_scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "resources.json",
        proof_resource_sampler=lambda: ProofHostResources(8, 16384, 16384),
        lane_reservations={}, auto_renew_leases=False,
        proof_backoff_seconds=0, poll_interval_seconds=0.005))
    fixture_scheduler.snapshot()
    persisted_policy = json.loads(fixture_scheduler.state_path.read_bytes())["config"]
    original, materializations, supplied_owners = repository_source_index.NativeRepositoryIndex, [], []

    def actual_index(path, *, scheduler=None):
        materializations.append(path)
        supplied_owners.append(scheduler)
        return original(path, scheduler=fixture_scheduler if scheduler is None else scheduler)

    monkeypatch.setattr(repository_source_index, "NativeRepositoryIndex", actual_index)
    yield dict(paths=materializations, owners=supplied_owners, state=fixture_scheduler.state_path,
        policy_sha256=policy_sha256(persisted_policy))
    assert fixture_scheduler.snapshot()["active_lease_count"] == 0
    assert json.loads(fixture_scheduler.state_path.read_bytes())["config"] == persisted_policy


def tokenizer(monkeypatch, *, kind="estimate", fail=False, mutate=None):
    seen, routes = [], []

    def resolve(*, provider, model):
        routes.append((provider, model))
        def count(text):
            seen.append(text)
            if mutate is not None:
                mutate()
            if fail:
                raise RuntimeError(PRIVATE)
            return len(text.encode("utf-8"))
        counter = PromptTokenCounter(provider=provider, model=model, tokenizer_id="authored-byte-counter",
            tokenizer_revision="authored@1", special_token_policy="literal",
            count_tokens=count, count_kind=kind, compatibility_basis="authored-local-counter")
        return counter, dict(offline=True, count_kind=kind, compatibility_basis="authored-local-counter")

    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", resolve)
    return seen, routes


@pytest.mark.parametrize("kind,provider,model", [
    ("estimate", "grok_cli", "grok-4.7"), ("tokenizer", "custom_backend", "fixture-model:1"),
])
def test_cli_real_native_compilation_preserves_literals_and_original_files(
        case, native_index, monkeypatch, capsys, kind, provider, model):
    case.update(provider=provider, model=model)
    case.update(scheduler_state=native_index["state"], scheduler_policy_sha256=native_index["policy_sha256"])
    seen, routes = tokenizer(monkeypatch, kind=kind)
    arguments = []
    for key, value in case.items():
        if key == "required_paths":
            for path in value:
                arguments.extend(["--required-path", path])
        else:
            arguments.extend(["--" + key.replace("_", "-"), str(value)])
    assert subject.main(arguments) == 0
    stdout = json.loads(capsys.readouterr().out)
    prompt = (case["output"] / "prompt.txt").read_bytes()
    raw_receipt = (case["output"] / "receipt.json").read_bytes()
    receipt = json.loads(raw_receipt)
    core = receipt["read_set"]
    _, advisory = render_model_prompt(prompt="", purpose="coding", workspace=case["root"])
    assert prompt.decode().startswith(advisory + read_sets.GUIDANCE)
    body = json.loads(prompt.decode()[len(advisory) + len(read_sets.GUIDANCE):])
    assert body["required_texts"] == [REQUIREMENTS]
    assert {row["path"]: row["content"] for row in body["sources"]} == {
        "policy.py": POLICY.decode(), "ledger.py": TARGET.decode()}
    assert b"OMITTED_ORIGINAL_COMMENT" not in prompt
    assert prompt.decode() in seen and routes == [(provider, model)]
    count_field = "selected_estimated_input_tokens" if kind == "estimate" else "selected_input_tokens"
    other_field = "selected_input_tokens" if kind == "estimate" else "selected_estimated_input_tokens"
    assert core[count_field] == len(prompt) and core[other_field] is None
    assert core["model_prompt_sha256"] == receipt["prompt_sha256"] == hashlib.sha256(prompt).hexdigest()
    assert core["model_prompt_bytes"] == receipt["prompt_bytes"] == len(prompt)
    assert set(core["selected_paths"]) == {"policy.py", "ledger.py"}
    assert receipt["requirements_sha256"] == hashlib.sha256(REQUIREMENTS.encode()).hexdigest()
    assert receipt["source_current_at_close"] is True
    assert stdout["receipt_sha256"] == hashlib.sha256(raw_receipt).hexdigest()
    assert set(path.name for path in case["output"].iterdir()) == {"prompt.txt", "receipt.json"}
    assert stat.S_IMODE(case["output"].stat().st_mode) == 0o700
    for path in case["output"].iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600 and path.stat().st_nlink == 1
    assert native_index["paths"] and all(not path.parent.exists() for path in native_index["paths"])
    assert all(owner is not None and owner.state_path == native_index["state"] for owner in native_index["owners"])
    binding = receipt["resource_scheduler_binding"]
    assert binding == dict(mode="existing_pinned", policy_sha256=native_index["policy_sha256"],
        state_path_sha256=hashlib.sha256(str(native_index["state"]).encode()).hexdigest(),
        policy_reconfiguration_requested=False, state_creation_requested=False)
    for body in (REQUIREMENTS, POLICY.decode(), TARGET.decode(), "OMITTED_ORIGINAL_COMMENT"):
        assert body not in raw_receipt.decode() and body not in json.dumps(stdout)
    assert receipt["provider_invoked"] is receipt["task_acceptance"] is receipt["billing_total_verified"] is False
    assert receipt["whole_session_token_savings"] is None


@pytest.mark.parametrize("change", ["budget", "duplicate", "control", "absolute", "parent", "git", "limit",
    "query", "provider", "output_inside", "existing", "output_symlink", "ancestor_symlink",
    "requirements_symlink", "requirements_hardlink", "requirements_large", "requirements_utf8"])
def test_invalid_cli_population_or_filesystem_refuses_before_tokenizer(case, tmp_path, monkeypatch, change):
    calls = []
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", lambda **kw: calls.append(kw))
    if change == "budget": case["max_input_tokens"] = True
    elif change == "duplicate": case["required_paths"] *= 2
    elif change == "control": case["required_paths"] = ["bad\nname.py"]
    elif change == "absolute": case["required_paths"] = ["/outside.py"]
    elif change == "parent": case["required_paths"] = ["../outside.py"]
    elif change == "git": case["required_paths"] = [".git/HEAD"]
    elif change == "limit": case["limit"] = 33
    elif change == "query": case["query"] = "x" * 4097
    elif change == "provider": case["provider"] = " provider "
    elif change == "output_inside": case["output"] = case["root"] / "delivery"
    elif change == "existing": case["output"].mkdir()
    elif change == "output_symlink": case["output"].symlink_to(case["root"], target_is_directory=True)
    elif change == "ancestor_symlink":
        link = tmp_path / "alias"
        link.symlink_to(case["root"], target_is_directory=True)
        case["output"] = link / "delivery"
    elif change == "requirements_symlink":
        link = tmp_path / "linked.txt"
        link.symlink_to(case["requirements_file"])
        case["requirements_file"] = link
    elif change == "requirements_hardlink": os.link(case["requirements_file"], tmp_path / "second.txt")
    elif change == "requirements_large": case["requirements_file"].write_bytes(b"x" * 16385)
    elif change == "requirements_utf8": case["requirements_file"].write_bytes(b"\xff")
    with pytest.raises((ValueError, OSError, UnicodeError)):
        subject.run_repository_read_set(**case)
    assert calls == []


@pytest.mark.parametrize("change", ["counter_failure", "source_changed", "requirements_changed", "budget"])
def test_failed_compilation_publishes_no_partial_prompt_or_private_exception(
        case, native_index, monkeypatch, capsys, change):
    mutate = None
    if change == "source_changed":
        mutate = lambda: (case["root"] / "ledger.py").write_text("# changed source\n")
    elif change == "requirements_changed":
        mutate = lambda: case["requirements_file"].write_text("changed literal requirement\n")
    elif change == "budget":
        case["max_input_tokens"] = 1
    tokenizer(monkeypatch, fail=change == "counter_failure", mutate=mutate)
    with pytest.raises(Exception):
        subject.run_repository_read_set(**case)
    assert not case["output"].exists()
    assert native_index["paths"] and all(not path.parent.exists() for path in native_index["paths"])
    assert PRIVATE not in capsys.readouterr().out


def test_cli_closed_failure_and_existing_output_are_preserved(case, monkeypatch, capsys):
    case["output"].mkdir()
    sentinel = case["output"] / "keep.txt"
    sentinel.write_text(PRIVATE)
    assert subject.main(["--root", str(case["root"]), "--query", case["query"],
        "--requirements-file", str(case["requirements_file"]), "--task-id", case["task_id"],
        "--provider", case["provider"], "--model", case["model"], "--max-input-tokens", "1000",
        "--output", str(case["output"])]) == 2
    assert sentinel.read_text() == PRIVATE
    result = json.loads(capsys.readouterr().out)
    assert result == dict(schema=subject.SCHEMA, status="refused", reason="invalid_input_or_local_failure")


def test_publication_failure_removes_only_its_new_output(case, monkeypatch):
    original, calls = os.write, []
    def fail_second(descriptor, raw):
        calls.append(descriptor)
        if len(calls) == 2:
            raise OSError(PRIVATE)
        return original(descriptor, raw)
    monkeypatch.setattr(subject.os, "write", fail_second)
    with pytest.raises(OSError):
        subject._publish(case["output"], case["root"], b"private prompt", b"{}\n")
    assert not case["output"].exists()
    assert (case["root"] / "ledger.py").read_bytes() == TARGET


@pytest.mark.parametrize("change", ["missing_state", "missing_policy", "wrong_policy", "malformed_policy",
    "absent_state", "symlink_state", "hardlink_state", "directory_state", "empty_state"])
def test_existing_scheduler_requires_exact_pair_and_current_policy_before_native_work(
        case, native_index, tmp_path, monkeypatch, change):
    case.update(scheduler_state=native_index["state"], scheduler_policy_sha256=native_index["policy_sha256"])
    before = native_index["state"].read_bytes()
    seen, routes = tokenizer(monkeypatch)
    if change == "missing_state": case["scheduler_state"] = None
    elif change == "missing_policy": case["scheduler_policy_sha256"] = None
    elif change == "wrong_policy": case["scheduler_policy_sha256"] = "0" * 64
    elif change == "malformed_policy": case["scheduler_policy_sha256"] = "invalid"
    elif change == "absent_state": case["scheduler_state"] = tmp_path / "not-created.json"
    elif change == "symlink_state":
        path = tmp_path / "alias.json"
        path.symlink_to(native_index["state"])
        case["scheduler_state"] = path
    elif change == "hardlink_state": os.link(native_index["state"], tmp_path / "second.json")
    elif change == "directory_state": case["scheduler_state"] = case["root"]
    elif change == "empty_state":
        path = tmp_path / "empty.json"
        path.write_bytes(b"")
        case["scheduler_state"] = path
    with pytest.raises(Exception):
        subject.run_repository_read_set(**case)
    assert native_index["paths"] == [] and seen == [] and routes == []
    assert native_index["state"].read_bytes() == before
    assert not case["output"].exists()
    if change == "absent_state":
        assert not case["scheduler_state"].exists()
        assert not case["scheduler_state"].with_name(case["scheduler_state"].name + ".lock").exists()
    if change == "hardlink_state":
        (tmp_path / "second.json").unlink()  # Restore only the test-authored alias before fixture teardown.


def test_omitted_scheduler_options_preserve_native_default_selection(case, native_index, monkeypatch):
    tokenizer(monkeypatch)
    subject.run_repository_read_set(**case)
    assert native_index["owners"] == [None]
    receipt = json.loads((case["output"] / "receipt.json").read_bytes())
    assert receipt["resource_scheduler_binding"] == dict(mode="native_default",
        state_path_sha256=None, policy_sha256=None, policy_reconfiguration_requested=False,
        state_creation_requested=None)
