"""Bounded initial capsules retain full source and candidate symbol coverage.

Real semantic producers, router dispatch, source broker and a registered child
use Unix sockets. Only the provider and container UID/filesystem admission are
authored. Candidate capsules are regenerated structural facts, not restoration
of omitted baseline capsule bytes or learned Source384 predictions.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_context_client as candidate
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from test.api.test_semantic_router_integration import provider  # noqa: F401
from test.api.test_worker_source_views import git


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _prompt(root, artifact):
    refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact.relative_to(root).as_posix(),
        repository_id="repo:capsule-budget", tree_id="tree:capsule-budget", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:capsule-budget", tree_id="tree:capsule-budget", objective_id="BUDGET-1",
        objective_revision="sha256:task", policy_id="policy:budget", policy_revision="sha256:policy",
        caller="supervisor:budget", stage="implementation", goal={"id": "BUDGET-1"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["labels.py"], "forbidden_paths": ["instruction.md"]},
        acceptance={"criteria": ["Preserve signatures; native validation remains pending"]}, evidence=refs)
    return render_context_capsule(compiled.capsule)


@pytest.fixture
def cases(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    original = "".join(f"def normalize_label_{i}(value):\n    return value + {i}\n\n" for i in range(8)).encode()
    instruction = b"Preserve all eight public functions and their signatures. Do not edit this instruction.\n"
    (root / "labels.py").write_bytes(original)
    (root / "instruction.md").write_bytes(instruction)
    git(root, "init", "-q")
    git(root, "add", "labels.py", "instruction.md")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "Authored capsule population")
    values = {}
    for cap in (1, 2, 8):
        output = root / ".runtime" / f"semantic-{cap}"
        prepared = prepare_semantic_context(repository=root, paths=["labels.py", "instruction.md"],
            required_raw_paths=["instruction.md"], program_paths=["labels.py"],
            objective=instruction.decode().strip(), task_id="BUDGET-1", output=output,
            worker_query="normalize_label_0", worker_capsule_limit=cap,
            worker_max_bytes=65536, context_input_tokens=32768)
        artifact = output / "worker-context.json"
        payload = json.loads(artifact.read_bytes())
        index = json.loads((output / "blocks" / payload["worker_projection"]["capsule_index_cid"]).read_bytes())
        values[cap] = dict(prepared=prepared, output=output, payload=payload,
            prompt=_prompt(root, artifact), index=index, artifact=artifact)
    workspace = tmp_path / "allocated"
    git(root, "worktree", "add", "--detach", "-q", str(workspace))
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    return root, workspace, original, instruction, values


# A real registered child performs requests, including kernel ancestry checks.
# The fixture does not claim the Docker owner/worker cross-UID boundary.
CHILD = r'''
import importlib.util, json, pathlib, socket, sys
spec = importlib.util.spec_from_file_location("capsule_source_transport", sys.argv[1])
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
for line in sys.stdin:
    value = json.loads(line)
    if "write" in value:
        pathlib.Path("labels.py").write_text(value["write"])
        response = {"written": True}
    else:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(30)
            channel.connect(sys.argv[2])
            protocol.send_frame(channel, value)
            response = protocol.receive_frame(channel)
    print(json.dumps(response), flush=True)
'''


def _exchange(process, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    assert line, "registered socket worker exited without a response"
    return json.loads(line)


def _request(service, generation=None, artifact=None, selection=None, cursor=None, max_bytes=8192):
    if artifact is None:
        query = dict(path="labels.py", view="original", cursor=None, max_bytes=max_bytes,
            expected_source_sha256=None, expected_view_sha256=None, source_state="working-candidate",
            expected_workspace_generation=generation)
        schema = source.REQUEST_SCHEMA_V2
    else:
        query = dict(artifact=artifact, selection=selection or dict(path=None, symbol_id=None,
            function_id=None, kind=None), expected_workspace_generation=generation,
            cursor=cursor, max_bytes=max_bytes)
        schema = candidate.REQUEST_SCHEMA_V2
    return source.validate_request(dict(schema=schema, session_id=service.descriptor["session_id"],
        request_id=uuid.uuid4().hex, query=query))


def _response(service, process, **kwargs):
    request = _request(service, **kwargs)
    response = _exchange(process, request)
    return source.validate_response(response, request=request, descriptor=service.descriptor)


def _artifact(service, process, generation, artifact, selection=None):
    import base64
    chunks, cursor = [], None
    for _ in range(32):
        response = _response(service, process, generation=generation, artifact=artifact,
            selection=selection, cursor=cursor)
        assert response["ok"], response
        page = response["page"]
        chunks.append(page["content"].encode() if page["encoding"] == "utf-8"
                      else base64.b64decode(page["content"], validate=True))
        cursor = page["next_cursor"]
        if cursor is None:
            raw = b"".join(chunks)
            assert len(raw) == page["artifact_bytes"] and _sha(raw) == page["artifact_sha256"]
            return json.loads(raw)
    pytest.fail("bounded candidate artifact did not terminate")


def _symbol_selection(symbol):
    return dict(path="labels.py", symbol_id=symbol, function_id=None, kind=None)


@pytest.mark.parametrize("cap", [1, 2, 8])
def test_bounded_capsules_retain_source_scope_and_retrieve_current_omitted_symbols(
        cases, provider, monkeypatch, record_property, cap):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SemanticCapsule, verify_block_bytes,
    )

    root, workspace, original, instruction, values = cases
    selected = values[cap]
    payload, baseline = selected["payload"], values[8]["payload"]
    projection = payload["worker_projection"]
    assert projection["full_capsule_count"] == 9  # Eight functions plus the module capsule.
    assert projection["selected_capsule_count"] == len(payload["capsules"]) == cap
    assert projection["omitted_capsule_count"] == 9 - cap
    assert selected["index"] == values[8]["index"]
    assert len(selected["index"]["pairs"]) == 9
    for key in ("manifest", "required_raw_paths", "raw_sources", "program_paths", "scope_cid",
                "semantic_root_cid", "reconstruction", "completion_authority"):
        assert payload[key] == baseline[key]
    assert payload["raw_sources"] == {"instruction.md": instruction.decode()}
    assert payload["worker_projection"]["raw_source_fetch_required"] == {
        "labels.py": payload["manifest"]["labels.py"]}
    for key in ("authority", "scope", "acceptance"):
        assert json.loads(selected["prompt"])[key] == json.loads(values[8]["prompt"])[key]
    retained = {path.name: _sha(path.read_bytes()) for path in (selected["output"] / "blocks").iterdir()}
    selected_names = {row["qualified_name"].split(".")[-1] for row in projection["selected_symbols"]}
    target = "normalize_label_6"
    assert (target not in selected_names) == (cap < 8)
    full_capsules = []
    for symbol_id, capsule_cid in selected["index"]["pairs"]:
        raw = (selected["output"] / "blocks" / capsule_cid).read_bytes()
        verify_block_bytes(capsule_cid, raw)
        capsule = SemanticCapsule.from_dict(dict(json.loads(raw), capsule_cid=capsule_cid))
        assert capsule.stable_symbol_id == symbol_id
        full_capsules.append(capsule.to_dict())
    baseline_symbol = next(row["stable_symbol_id"] for row in full_capsules
        if any(facet.get("version_evidence", {}).get("name") == target
               for facet in row["metadata"]["facets"]))
    monkeypatch.setattr(guard, "_owner_boundary", lambda path:
        (dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "b" * 64))
    monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary",
        lambda **kwargs: {"authored_uid_boundary": True})
    state, observed = provider
    observations = {}
    with tempfile.TemporaryDirectory(prefix="capsule-budget-") as directory:
        boundary = Path(directory) / "boundary.json"
        boundary.write_text("authored boundary")
        (boundary.parent / "worker-observations").mkdir(mode=0o750)
        service = broker.WorkerSourceBroker(repository=root, workspace=workspace, boundary=boundary,
            prompt=selected["prompt"], profile="identity@1", timeout_seconds=120)
        process = None
        try:
            def descriptor(path, digest, *, workspace):
                raw = Path(path).read_bytes()
                value = source.validate_descriptor(json.loads(raw))
                assert raw == source.canonical_json(value) and _sha(raw) == digest
                assert value["workspace"] == str(workspace)
                return deepcopy(value)
            monkeypatch.setattr(source, "load_descriptor", descriptor)
            service.worker_uid = os.getuid()  # Only same-UID fixture admission is authored.
            process = subprocess.Popen([sys.executable, "-I", "-c", CHILD, source.__file__,
                str(service.socket_path)], cwd=workspace, text=True, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            service.note_spawned(process)

            def during_provider():
                assert os.environ[source.DESCRIPTOR_ENV] == str(service.descriptor_path)
                assert os.environ[source.DIGEST_ENV] == service.descriptor_sha256
                working = _response(service, process)
                assert working["ok"]
                generation = working["page"]["workspace_generation"]
                catalog = _artifact(service, process, generation, "catalog")
                row = next(row for row in catalog["rows"] if row["path"] == "labels.py")
                names = {name for capsule in row["capsules"] for name in capsule["names"]}
                assert {f"normalize_label_{i}" for i in range(8)} <= names
                symbol = next(item["stable_symbol_id"] for item in row["capsules"] if target in item["names"])
                # The candidate index has its own source-generation scope.
                # Workers must discover candidate IDs, not reuse baseline IDs.
                assert symbol != baseline_symbol
                wrong_scope = _artifact(service, process, generation, "capsules",
                    _symbol_selection(baseline_symbol))
                assert wrong_scope["counts"]["matched_count"] == 0
                assert wrong_scope["status"] == "no_matching_rows"
                fetched = _artifact(service, process, generation, "capsules", _symbol_selection(symbol))
                assert fetched["counts"]["matched_count"] == 1
                assert fetched["rows"][0]["stable_symbol_id"] == symbol
                assert fetched["coverage"]["source_sha256"] == _sha(original)
                for key in ("baseline_evidence_applicable", "proof_authority", "completion_authority"):
                    assert fetched[key] is False
                assert fetched["model_inference_calls"] == fetched["provider_calls"] == 0
                assert fetched["training_steps"] == fetched["source_obligations_discharged"] == 0
                # Deliberately retain a partial cursor; an edit must not let it
                # retrieve prior-generation rows even after a fresh capture.
                partial = _response(service, process, generation=generation, artifact="capsules",
                    selection=_symbol_selection(symbol), max_bytes=2048)
                assert partial["ok"] and partial["page"]["next_cursor"] is not None
                cursor = partial["page"]["next_cursor"]
                changed = original.replace(b"return value + 6", b"return value + 60")
                assert _exchange(process, {"write": changed.decode()}) == {"written": True}
                stale = _response(service, process, generation=generation, artifact="capsules",
                    selection=_symbol_selection(symbol), cursor=cursor, max_bytes=2048)
                assert stale["ok"] is False and stale["page"] is None
                fresh = _response(service, process)
                assert fresh["ok"] and fresh["page"]["workspace_generation"] != generation
                new_generation = fresh["page"]["workspace_generation"]
                assert not _response(service, process, generation=generation, artifact="catalog")["ok"]
                assert not _response(service, process, generation=new_generation, artifact="capsules",
                    selection=_symbol_selection(symbol), cursor=cursor, max_bytes=2048)["ok"]
                current = _artifact(service, process, new_generation, "catalog")
                current_row = next(row for row in current["rows"] if row["path"] == "labels.py")
                new_symbol = next(item["stable_symbol_id"] for item in current_row["capsules"]
                    if target in item["names"])
                fresh_capsule = _artifact(service, process, new_generation, "capsules", _symbol_selection(new_symbol))
                assert fresh_capsule["counts"]["matched_count"] == 1
                assert fresh_capsule["coverage"]["source_sha256"] == _sha(changed)
                assert fresh_capsule["binding"] != fetched["binding"]
                observations.update(full_candidate_symbols=len(names), omitted_target_retrieved=cap < 8,
                    old_generation_refused=True, stale_cursor_refused=True,
                    baseline_identity_not_substituted=True,
                    fresh_source_sha256=fresh_capsule["coverage"]["source_sha256"])

            state["after_generate"] = during_provider
            output, receipt = runner.run(prompt=selected["prompt"], provider="codex_cli", model="pinned",
                timeout=1, max_output_tokens=128, semantic_repository=root, container_boundary=boundary,
                container_boundary_sha256="b" * 64, source_view_profile="identity@1",
                source_view_descriptor=service.descriptor_path,
                source_view_descriptor_sha256=service.descriptor_sha256)
            assert output == "literal summary" and len(observed) == 1
            assert "worker_candidate_context_client.py" in observed[0][0]
            assert receipt["source_view_retrieval"]["session_id"] == service.descriptor["session_id"]
            assert receipt["model_prompt_sha256"] == _sha(observed[0][0].encode())
            assert receipt["completion_authority"] is False
            assert (root / "labels.py").read_bytes() == original
            assert (root / "instruction.md").read_bytes() == instruction
            assert (workspace / "instruction.md").read_bytes() == instruction
            assert retained == {path.name: _sha(path.read_bytes())
                for path in (selected["output"] / "blocks").iterdir()}
        finally:
            if process is not None:
                process.stdin.close()
                process.wait(timeout=10)
            service.close()
        audit = json.loads(service.audit_path.read_bytes())
        assert audit["accounting_complete"] is audit["service_stopped"] is True
        assert any(row["request_operation"] == "candidate_context" and row["candidate_context_page"]
                   and row["delivered"] for row in audit["requests"])
        assert audit["retrieval_provider_calls"] == audit["source_obligations_discharged"] == 0
    record_property("capsule_retrieval_qualification", json.dumps(dict(cap=cap,
        selected_capsules=cap, full_baseline_index_capsules=9, external_provider_calls=0,
        learned_source384_retrieval_qualified=False, baseline_capsule_byte_restoration_claimed=False,
        native_router_and_registered_socket_worker=True, **observations), sort_keys=True))
