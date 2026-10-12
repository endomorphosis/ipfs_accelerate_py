"""Native semantic scope admission reaches candidate retrieval unchanged.

Only the test container boundary and peer identity are authored. Source scans,
semantic artifacts, prompt validation, working generations and candidate
capsule/formal producers are the real implementations; no model is invoked.
"""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as broker
from test.api.test_semantic_router_translation import _program_prompt
from test.api.test_worker_candidate_context_broker import answer as candidate_answer
from test.api.test_worker_source_broker import allocated, native, service, _bytes, _request  # noqa: F401
from test.api.test_worker_source_working_broker import answer as source_answer


def artifact(service, generation, kind):
    chunks, cursor, offset = [], None, 0
    for _ in range(64):
        response = candidate_answer(service, generation, artifact=kind, cursor=cursor)
        assert response["ok"], response
        page = response["page"]
        assert page["workspace_generation"] == generation and page["offset"] == offset
        assert page["baseline_evidence_applicable"] is False
        chunks.append(_bytes(page))
        offset, cursor = page["end_offset"], page["next_cursor"]
        if cursor is None:
            raw = b"".join(chunks)
            assert len(raw) == page["artifact_bytes"]
            assert hashlib.sha256(raw).hexdigest() == page["artifact_sha256"]
            return json.loads(raw)
    pytest.fail("bounded candidate artifact did not terminate")


def test_native_semantic_artifact_larger_than_transport_frame_still_starts(service, monkeypatch):
    table = json.loads(service._table_bytes)
    raw = (service.repository / table["semantic_artifact"]).read_bytes()
    assert len(raw) > 8192  # Source request framing is not the semantic artifact bound.
    assert hashlib.sha256(raw).hexdigest() == table["semantic_sha256"]
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    legacy = service._answer(_request(service, view="original"), None)
    assert legacy["ok"] and _bytes(legacy["page"]) == (service.repository / "mod.py").read_bytes()
    working = source_answer(service)
    assert working["ok"]
    summary = artifact(service, working["page"]["workspace_generation"], "summary")
    assert summary["program_paths"] == ["mod.py"]
    assert summary["program_files"] == 1 and summary["capsule_count"] > 0


@pytest.mark.parametrize("program", [[], ["mod.py"]], ids=["support-only", "explicit-subset"])
def test_explicit_native_program_selection_preserved_by_broker(service, native, monkeypatch, program):
    repository, semantic_artifact, prompt = _program_prompt(native, program=program)
    payload = json.loads(semantic_artifact.read_bytes())
    assert payload["schema"] == "supervisor-semantic-worker-context@2"
    assert payload["program_paths"] == program
    support = (repository / "harness.py").read_bytes()
    assert b"def excluded_harness_symbol" in support
    (service.workspace / "harness.py").write_bytes(support)
    candidate = broker.WorkerSourceBroker(repository=repository, workspace=service.workspace,
        boundary=service.boundary, prompt=prompt, profile="python-comments@1", timeout_seconds=90)
    try:
        monkeypatch.setattr(candidate, "_peer", lambda channel: None)
        assert candidate.candidate_context.program_paths == tuple(program)
        legacy = candidate._answer(_request(candidate, path="harness.py", view="original"), None)
        assert legacy["ok"] and _bytes(legacy["page"]) == support
        working = source_answer(candidate, path="harness.py", view="original")
        assert working["ok"] and _bytes(working["page"]) == support
        generation = working["page"]["workspace_generation"]
        summary = artifact(candidate, generation, "summary")
        assert summary["program_paths"] == program
        assert summary["program_files"] == len(program)
        assert summary["source_files"] == 2 and summary["support_files"] == 2 - len(program)
        coverage = {row["path"]: row for row in summary["coverage"]}
        assert coverage["harness.py"]["role"] == "support"
        assert coverage["harness.py"]["semantic_status"] == "not_scanned_support"
        assert coverage["harness.py"]["formal_status"] == "not_selected_support"
        capsules = artifact(candidate, generation, "capsules")
        formal = artifact(candidate, generation, "formal")
        assert set(formal["reports"]) == set(program)
        assert all(row["source_slice_path"] in program for row in capsules["capsules"])
        assert "excluded_harness_symbol" not in json.dumps(capsules["capsules"])
        if not program:
            assert capsules["capsules"] == [] and formal["reports"] == {}
            assert summary["capsule_count"] == summary["formal_report_count"] == 0
            assert summary["pending_obligation_count"] == 0
        else:
            assert summary["capsule_count"] > 0 and summary["formal_report_count"] == 1
        assert summary["baseline_evidence_applicable"] is False
        assert summary["source_obligations_discharged"] == summary["model_inference_calls"] == 0
    finally:
        candidate.close()
