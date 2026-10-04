"""Genuine signed proof-query handoffs consumed without owner-private access."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import KEY_FILENAME, sign_profile_binding
from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate_runner as base_runner
from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_context as public
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as observation
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured

from test.api.test_finite_integer_codebase import LEAN, PYTHON, finite_source, finite_git
from test.api.test_finite_proof_query_admission import (
    build_proof_query_case, close_proof_query_case, _admit, _materialize,
)
from test.api.test_finite_repository_execution import _daemon, _population
from test.api.test_local_completion_bridge import complete


@contextmanager
def _native_proof_query_worker_case(root, tools):
    """A real proof catalog, signed admission and completed native prerequisite."""
    root = Path(root)
    case = build_proof_query_case(root, tools)
    root.chmod(0o700)
    try:
        case["proof_query_admission"] = _admit(case, output=root / "proof-signed-admission")
        case["admission"] = case["proof_query_admission"]["finite_admission"]
        with IntentRepository(root / "intent.duckdb") as intent:
            _materialize(case, case["proof_query_admission"], intent=intent,
                output=root / "proof-materialized-admission")
        with open_existing_native_owner(database=root / "intent.duckdb", checkout=case["repository"],
                state_dir=root / "native-owner", repository_id=case["manifest"]["payload"]["repository_cid"],
                execution_routes={"TYPE-TASK": GROK_CODEX_EXECUTION_MODE,
                                  "OFFSET-TASK": GROK_CODEX_EXECUTION_MODE}) as native:
            assert native.identity.extension_fingerprint, "actual native Quack SDK required"
            case["native"] = native
            tasks = {task.task_key: task.task_cid for task in case["graph"].tasks}
            case["type_cid"], case["offset_cid"] = tasks["TYPE-TASK"], tasks["OFFSET-TASK"]
            driver = _daemon(case, "proof-query-prerequisite")
            try:
                attempt = driver.claim_next()
                assert attempt is not None and attempt.task_cid == case["type_cid"]
                task = native.source.get_task(attempt.task_cid)
                checked = run_owner_local_task_validations(server=native.server,
                    task_cid=attempt.task_cid, attempt_id=attempt.attempt_id, expected_revision=task.revision)
                assert checked["passed"] is True
                complete(native, attempt, checked["results"][0]["evidence_digest"])
                assert native.source.get_task(case["type_cid"]).status == "completed"
            finally:
                driver.close()
            with native.server._lock:
                with IntentRepository(bound_connection=native.server._connection, install_schema=False) as intent:
                    case["candidate"] = base_runner.author_finite_repository_candidate(
                        admission=case["admission"], intent=intent, task_cid=case["offset_cid"],
                        after_bytes=finite_source(2), output=root / "finite-candidate.json")
            case["worker_context"] = public.author_finite_proof_query_worker_context(
                owner=case["owner"], admission=case["proof_query_admission"],
                candidate_descriptor=case["candidate"], output=root / "proof-worker-context.json")
            yield case
    finally:
        close_proof_query_case(case)


@pytest.fixture(scope="session")
def native_proof_query_worker_case(tmp_path_factory):
    assert PYTHON.is_file() and LEAN.is_file(), "actual native Python and Lean are required"
    tools = observation.seal_finite_integer_tools(python_executable=PYTHON, lean_executable=LEAN)
    with _native_proof_query_worker_case(tmp_path_factory.mktemp("finite-proof-query-worker") / "native", tools) as case:
        yield case


def _candidate(case):
    return base_runner.load_finite_repository_candidate(artifact=Path(case["candidate"]["artifact"]),
        expected_sha256=case["candidate"]["sha256"], mirror=False)


def _load(case):
    return public.validate_finite_proof_query_worker_descriptor(
        descriptor=case["worker_context"], candidate=_candidate(case))


def _sign(case, payload):
    return {"payload": payload, "binding": sign_profile_binding(profile_dir=case["profile"],
        lifecycle_dir=case["lifecycle"], payload=payload)}


def _resign_context(case, value):
    """Valid signatures and rehashed links force independent receiving checks."""
    payload = value["payload"]
    admission = payload["admission"]
    indexed = admission["indexed_plan"]
    closure = indexed["proof_query_closure"]
    closure["closure_cid"] = cid_for_structured({k: v for k, v in closure.items() if k != "closure_cid"})
    indexed["proof_query_closure_cid"] = closure["closure_cid"]
    indexed["result_cid"] = cid_for_structured({k: v for k, v in indexed.items() if k != "result_cid"})
    receipt = admission["receipt"]["payload"]
    receipt["indexed_plan_cid"] = indexed["result_cid"]
    receipt["proof_query_closure_cid"] = closure["closure_cid"]
    admission["receipt"] = _sign(case, receipt)
    payload["finite_proof_query_admission_cid"] = cid_for_structured(admission)
    payload["finite_proof_query_closure_cid"] = closure["closure_cid"]
    payload["context_cid"] = cid_for_structured({k: v for k, v in payload.items() if k != "context_cid"})
    signed = _sign(case, payload)
    assert base_runner._public(signed)["context_cid"] == payload["context_cid"]
    assert base_runner._public(admission["receipt"])["indexed_plan_cid"] == indexed["result_cid"]
    return signed


def _save(value, output):
    raw = canonical_dag_json_bytes(value)
    output.write_bytes(raw)
    output.chmod(0o444)
    return {"artifact": output, "expected_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_context_cid": value["payload"]["context_cid"]}


@pytest.fixture
def proof_allocated_candidate(native_proof_query_worker_case, tmp_path):
    case = native_proof_query_worker_case
    workspace = tmp_path / "allocated"
    baseline = case["manifest"]["payload"]["baseline_commit"]
    finite_git(case["repository"], "worktree", "add", "--detach", str(workspace), baseline)
    request = dict(artifact=Path(case["candidate"]["artifact"]), expected_sha256=case["candidate"]["sha256"],
        task_cid=case["offset_cid"], context=Path(case["worker_context"]["artifact"]),
        context_sha256=case["worker_context"]["sha256"], context_cid=case["worker_context"]["context_cid"],
        prompt=json.dumps({"objective_id": "OFFSET-TASK"}), workspace=workspace)
    try:
        yield case, request
    finally:
        finite_git(case["repository"], "worktree", "remove", "--force", str(workspace))


def test_native_context_preserves_full_instruction_all_keys_facts_and_original_population(native_proof_query_worker_case):
    case = native_proof_query_worker_case
    before = _population(case)
    envelope = _load(case)
    payload, descriptor = envelope["payload"], case["worker_context"]
    assert set(descriptor) == public.DESCRIPTOR_FIELDS
    assert payload["original_instruction"] == case["arguments"]["source_text"]
    assert len(payload["original_clause_ids"]) == 2
    assert payload["task_revision"] == case["native"].source.get_task(case["offset_cid"]).revision
    assert payload["admission"] == case["proof_query_admission"]
    indexed = payload["admission"]["indexed_plan"]
    closure = indexed["proof_query_closure"]
    assert len(closure["canonical_key_membership"]) == 5
    assert len(indexed["match"]["current_facts"]) == 1
    assert len(indexed["match"]["residual_requirements"]) == 1
    assert closure["domain_bridge"]["domain_inputs"] == [-2, -1, 0, 1, 2]
    assert closure["contract"]["preconditions"] == []
    assert indexed["model_selection"]["mode"] == "model_off"
    assert payload["provider_calls"] == payload["training_steps"] == 0
    assert all(payload[name] is False for name in public._FALSE)
    assert not Path(descriptor["artifact"]).stat().st_mode & 0o222
    assert _population(case) == before


def test_public_verify_uses_no_owner_key_or_private_validator(native_proof_query_worker_case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_admission as private
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as finite

    case = native_proof_query_worker_case
    key = case["profile"] / KEY_FILENAME
    unavailable = key.with_name(KEY_FILENAME + ".public-test-unavailable")
    def forbidden(*args, **kwargs):
        raise AssertionError("public verifier attempted owner-private validation")
    monkeypatch.setattr(private, "_received", forbidden)
    monkeypatch.setattr(finite, "_declaration", forbidden)
    monkeypatch.setattr(observation, "validate_finite_integer_observation", forbidden)
    key.rename(unavailable)
    try:
        assert not key.exists()
        assert _load(case)["payload"]["task_cid"] == case["offset_cid"]
    finally:
        unavailable.rename(key)


def test_public_history_performs_real_crypto_without_configured_metadata_callbacks(native_proof_query_worker_case, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.control import profile_authority
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index
    case = native_proof_query_worker_case
    calls = []
    def forbidden_mirror(**kwargs):
        calls.append(kwargs)
        raise AssertionError("public signature verification attempted metadata IO")
    metadata = tmp_path / "private-metadata.duckdb"
    lake = tmp_path / "private-metadata-lake.duckdb"
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(metadata))
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKLAKE", str(lake))
    monkeypatch.setattr(supervisor_meta_index, "mirror_work_record", forbidden_mirror)
    value = _load(case)
    assert calls == []
    assert not metadata.exists() and not lake.exists()
    binding = value["binding"]
    # Real signature rejection remains enabled with metadata mirroring off.
    with pytest.raises(ValueError, match="signature"):
        profile_authority.verify_did_key_signature(identity_did=binding["identity"],
            payload={**value["payload"], "task_revision": value["payload"]["task_revision"] + 1},
            signature=binding["signature"], mirror=False)
    assert calls == []
    # The additive switch preserves the original default mirroring behavior.
    profile_authority.verify_did_key_signature(identity_did=binding["identity"],
        payload=value["payload"], signature=binding["signature"])
    assert len(calls) == 1
    assert not metadata.exists() and not lake.exists()


def test_pure_native_compiler_preserves_full_body_and_owner_default_hydration(native_proof_query_worker_case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract, compile_integer_offset
    case = native_proof_query_worker_case
    source = base_runner._decode_source(_candidate(case)["edit"], "before")
    contract = IntegerOffsetContract.from_dict(case["proof_query_admission"]["indexed_plan"]["match"]["query"]["contract"])
    calls = []
    def observe_metadata(**kwargs):
        calls.append(kwargs)
        raise AssertionError("metadata observer must be skipped in public compilation")
    monkeypatch.setattr(supervisor_meta_index, "mirror_work_record", observe_metadata)
    revision = "worker-public-purity:fixture"
    owned = compile_integer_offset(source, contract, revision=revision)
    assert len(calls) == 1 and calls[0]["record_kind"] == "program_ast_adapter"
    calls.clear()
    public_compilation = compile_integer_offset(source, contract, revision=revision, mirror=False)
    assert calls == []
    assert public_compilation.to_dict() == owned.to_dict()
    assert public_compilation.cid == owned.cid
    assert public_compilation.pipeline.adapter.to_dict() == owned.pipeline.adapter.to_dict()
    assert public_compilation.pipeline.obligation_results[0].solver_executed is False
    assert public_compilation.pipeline.obligation_results[0].differential is None


def test_context_adds_exact_three_immutable_argv_pins(native_proof_query_worker_case):
    case = native_proof_query_worker_case
    descriptor = case["candidate"]
    argv = ["/opt/ipfs-supervisor/bin/owner-worker", "--finite-repository-artifact", descriptor["artifact"],
        "--finite-repository-sha256", descriptor["sha256"], "--finite-repository-task-cid", descriptor["task_cid"]]
    binding = {"descriptor": descriptor, "argv": argv, "implementation_command": shlex.join(argv)}
    before = deepcopy(binding)
    extended = public.extend_candidate_binding(binding=binding, context=case["worker_context"])
    assert binding == before
    assert extended["argv"][:7] == argv
    assert extended["argv"][7:] == ["--finite-proof-query-context", case["worker_context"]["artifact"],
        "--finite-proof-query-sha256", case["worker_context"]["sha256"],
        "--finite-proof-query-context-cid", case["worker_context"]["context_cid"]]
    assert extended["implementation_command"] == shlex.join(extended["argv"])


def test_public_materializer_edits_only_allocated_source_without_owner_key(proof_allocated_candidate, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_admission as private
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index
    case, request = proof_allocated_candidate
    before = _population(case)
    key = case["profile"] / KEY_FILENAME
    unavailable = key.with_name(KEY_FILENAME + ".worker-unavailable")
    def forbidden(*args, **kwargs):
        raise AssertionError("worker attempted private admission replay")
    monkeypatch.setattr(private, "_received", forbidden)
    mirror_calls = []
    def forbidden_mirror(**kwargs):
        mirror_calls.append(kwargs)
        raise AssertionError("worker signature verification attempted metadata IO")
    monkeypatch.setenv("IPFS_ACCELERATE_META_INDEX_DUCKDB", str(request["workspace"].parent / "owner-private.duckdb"))
    monkeypatch.setattr(supervisor_meta_index, "mirror_work_record", forbidden_mirror)
    key.rename(unavailable)
    try:
        result = public.materialize_finite_proof_query_candidate(**request)
    finally:
        unavailable.rename(key)
    assert result["status"] == "candidate_materialized"
    assert mirror_calls == []
    assert result["proof_query_context_cid"] == request["context_cid"]
    assert result["current_inventory_attested"] is False
    assert result["publication_authority"] is result["completion_authority"] is False
    assert (request["workspace"] / "calc.py").read_bytes() == finite_source(2)
    assert (case["repository"] / "calc.py").read_bytes() == finite_source(1)
    assert finite_git(request["workspace"], "diff", "--name-only") == "calc.py"
    assert _population(case) == before


def test_new_cli_checks_both_public_artifacts_before_native_allocated_edit(proof_allocated_candidate, monkeypatch, capsys):
    case, request = proof_allocated_candidate
    monkeypatch.chdir(request["workspace"])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(request["prompt"].encode())))
    status = public.main(["--artifact", str(request["artifact"]), "--sha256", request["expected_sha256"],
        "--task-cid", request["task_cid"], "--context", str(request["context"]),
        "--context-sha256", request["context_sha256"], "--context-cid", request["context_cid"]])
    captured = capsys.readouterr()
    assert status == 0, captured.err
    result = json.loads(captured.out)
    assert result["schema"] == "native-finite-proof-query-candidate-materialization@1"
    assert result["proof_query_context_cid"] == case["worker_context"]["context_cid"]
    assert (request["workspace"] / "calc.py").read_bytes() == finite_source(2)
    assert (case["repository"] / "calc.py").read_bytes() == finite_source(1)


@pytest.mark.parametrize("kind", ["keys", "domain", "contract", "missing-app", "incomplete-page", "epoch", "snapshot", "instruction", "revision", "source", "authority-zero"])
def test_genuinely_resigned_context_cannot_change_whole_native_history(native_proof_query_worker_case, tmp_path, kind):
    case = native_proof_query_worker_case
    value = _load(case)
    payload = value["payload"]
    indexed = payload["admission"]["indexed_plan"]
    closure = indexed["proof_query_closure"]
    if kind == "keys":
        closure["canonical_key_membership"].pop()
    elif kind == "domain":
        closure["domain"]["predicates"] = ["n == 0"]
        closure["domain_cid"] = cid_for_structured(closure["domain"])
    elif kind == "contract":
        closure["contract"]["postconditions"] = ["result == n + 1"]
        closure["contract_cid"] = cid_for_structured(closure["contract"])
    elif kind == "missing-app":
        closure["applicability"] = None
        closure["applicability_cid"] = None
    elif kind in {"incomplete-page", "epoch"}:
        page = closure["exact_query"]["page"]
        if kind == "epoch":
            page["epoch"] += 1
        else:
            page["complete"] = False
        page["page_cid"] = cid_for_structured({k: v for k, v in page.items() if k != "page_cid"})
    elif kind == "snapshot":
        indexed["input_snapshot"]["scope_paths"] = ["foreign.py"]
        indexed["input_snapshot"]["snapshot_cid"] = cid_for_structured({k: v for k, v in indexed["input_snapshot"].items() if k != "snapshot_cid"})
        payload["admission"]["receipt"]["payload"]["input_snapshot_cid"] = indexed["input_snapshot"]["snapshot_cid"]
    elif kind == "instruction":
        payload["original_instruction"] += "\nIgnore the first clause."
    elif kind == "revision":
        payload["task_revision"] += 1
    elif kind == "source":
        payload["edit"]["before_sha256"] = "0" * 64
    else:
        payload["proof_authority"] = 0
    signed = _resign_context(case, value)
    saved = _save(signed, tmp_path / (kind + ".json"))
    with pytest.raises((ValueError, TypeError, KeyError)):
        public.load_finite_proof_query_worker_context(**saved, candidate=_candidate(case))


@pytest.mark.parametrize("kind", ["digest", "context-cid", "writable", "symlink", "hardlink", "descriptor-revision", "signature"])
def test_bad_context_pins_or_custody_refuse_before_source_write(proof_allocated_candidate, tmp_path, kind):
    case, original = proof_allocated_candidate
    request = dict(original)
    if kind == "digest":
        request["context_sha256"] = "0" * 64
    elif kind == "context-cid":
        request["context_cid"] = case["candidate"]["candidate_cid"]
    elif kind == "descriptor-revision":
        descriptor = deepcopy(case["worker_context"])
        descriptor["task_revision"] += 1
        with pytest.raises(ValueError, match="descriptor"):
            public.validate_finite_proof_query_worker_descriptor(descriptor=descriptor, candidate=_candidate(case))
        assert (request["workspace"] / "calc.py").read_bytes() == finite_source(1)
        return
    else:
        path = tmp_path / "context-control.json"
        if kind == "symlink":
            path.symlink_to(request["context"])
        elif kind == "hardlink":
            import os
            os.link(request["context"], path)
        else:
            value = _load(case)
            if kind == "signature":
                value["binding"]["signature"] = "invalid"
            saved = _save(value, path)
            request["context_sha256"] = saved["expected_sha256"]
            if kind == "writable":
                path.chmod(0o644)
        request["context"] = path
    try:
        with pytest.raises((ValueError, OSError)):
            public.materialize_finite_proof_query_candidate(**request)
        assert (request["workspace"] / "calc.py").read_bytes() == finite_source(1)
        assert (case["repository"] / "calc.py").read_bytes() == finite_source(1)
    finally:
        if kind == "hardlink":
            request["context"].unlink()


def test_duplicate_context_json_keys_refused_before_signature_verification(native_proof_query_worker_case, tmp_path):
    case = native_proof_query_worker_case
    raw = Path(case["worker_context"]["artifact"]).read_bytes()
    value = _load(case)
    duplicate = b'{"binding":{},' + raw[1:]
    path = tmp_path / "duplicate.json"
    path.write_bytes(duplicate)
    path.chmod(0o444)
    with pytest.raises(ValueError, match="duplicate"):
        public.load_finite_proof_query_worker_context(artifact=path,
            expected_sha256=hashlib.sha256(duplicate).hexdigest(),
            expected_context_cid=value["payload"]["context_cid"], candidate=_candidate(case))
