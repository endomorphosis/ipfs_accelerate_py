"""Selective transport keeps native rows and refuses stale or foreign inputs."""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_artifacts as builder
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_selection as module


SOURCE = b"import numpy as np\ndef select(values):\n    return np.argmax(values)\n\ndef clip(values):\n    return values[0]\n"
SOURCES = {"first.py": SOURCE,
    "other.py": b"import numpy as np\ndef other(values):\n    return np.argmin(values)\n",
    "instruction.md": b"Do not execute source.\n"}


def query(path=None, symbol_id=None, function_id=None, kind=None):
    return dict(path=path, symbol_id=symbol_id, function_id=function_id, kind=kind)


def build(sources, paths=None, generation="a" * 64):
    return builder.build_candidate_artifacts(sources=sources,
        program_paths=tuple(sorted(sources)) if paths is None else paths,
        workspace_generation=generation)


@pytest.fixture(scope="module")
def artifacts():
    return build(SOURCES, paths=("first.py", "other.py"))


def select(artifacts, artifact="catalog", selection=None, sources=None):
    raw = module.select_candidate_artifact(artifacts=artifacts, sources=SOURCES if sources is None else sources,
        artifact=artifact, selection=query() if selection is None else selection)
    result = json.loads(raw)
    assert builder._wire(result) == raw
    return result


def test_catalog_discovers_exact_ids_paths_and_coverage_without_source_or_projections(artifacts):
    result = select(artifacts)
    native = {key: json.loads(value) for key, value in artifacts.items()}
    assert result["schema"] == module.SCHEMA
    assert [row["path"] for row in result["rows"]] == sorted(SOURCES)
    first = result["rows"][0]
    assert first["source_sha256"] == hashlib.sha256(SOURCE).hexdigest()
    assert first["function_coverage"] == native["formal"]["reports"]["first.py"]["coverage"]["function_coverage"]
    assert {item["stable_symbol_id"] for item in first["capsules"]} == {
        row["stable_symbol_id"] for row in native["capsules"]["capsules"] if row["source_slice_path"] == "first.py"}
    assert any(item["names"] == ["select"] for item in first["capsules"])
    assert first["obligation_kinds"]["library_semantics_and_effects"] == 1
    assert result["unsupported_families"] == native["formal"]["profile"]["unsupported_families"]
    assert result["counts"] == dict(total_count=3, matched_count=3, filtered_out_count=0,
        out_of_path_count=0, returned_count=3, remaining_count=0)
    assert "projections" not in result and "producer_sources" not in result
    assert SOURCE.decode() not in json.dumps(result)
    assert len(builder._wire(result)) < len(artifacts["formal"])


def test_single_path_catalog_does_not_return_other_file_symbols_or_functions(artifacts):
    result = select(artifacts, selection=query("first.py"))
    assert len(result["rows"]) == 1 and result["rows"][0]["path"] == "first.py"
    assert result["counts"]["total_count"] == 1 and result["counts"]["out_of_path_count"] == 2
    assert all(row["name"] != "other" for row in result["rows"][0]["function_coverage"])


def test_catalog_is_independent_of_captured_mapping_insertion_order(artifacts):
    normal = select(artifacts)
    reversed_sources = dict(reversed(list(SOURCES.items())))
    assert select(artifacts, sources=reversed_sources) == normal


def test_selected_capsule_is_exact_native_row_with_full_scope_binding(artifacts):
    capsules = json.loads(artifacts["capsules"])["capsules"]
    native = next(row for row in capsules if row["source_slice_path"] == "first.py" and row["signature"])
    result = select(artifacts, "capsules", query("first.py", symbol_id=native["stable_symbol_id"]))
    assert result["rows"] == [native]
    assert builder._wire(result["rows"][0]) == builder._wire(native)
    assert result["counts"] == dict(total_count=3, matched_count=1, filtered_out_count=2,
        out_of_path_count=2, returned_count=1, remaining_count=0)
    summary = json.loads(artifacts["summary"])
    binding = result["binding"]
    for key in ("workspace_generation", "candidate_scope_cid", "semantic_root_cid", "producer_binding_sha256"):
        assert binding[key] == summary[key]
    assert binding["source_hashes_sha256"] == builder._sha(builder._wire(summary["source_hashes"]))
    assert binding["parent_artifact_sha256"] == {key: builder._sha(raw) for key, raw in artifacts.items()}
    assert all(type(result[key]) is type(value) and result[key] == value for key, value in builder.AUTHORITY.items())


def test_exact_function_and_kind_use_native_pager_and_leave_other_rows_out(artifacts, monkeypatch):
    from ipfs_datasets_py.logic.software_verification import numerical_obligation_retrieval as pager
    report = json.loads(artifacts["formal"])["reports"]["first.py"]
    fid = next(row["function_id"] for row in report["coverage"]["function_coverage"] if row["name"] == "select")
    expected = [row for row in report["pending_obligations"] if fid in row["function_ids"]
        and row["kind"] == "library_semantics_and_effects"]
    calls, original = [], pager.page_numerical_obligations
    def wrapped(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(pager, "page_numerical_obligations", wrapped)
    result = select(artifacts, "obligations", query("first.py", function_id=fid, kind="library_semantics_and_effects"))
    assert len(expected) == 1 and result["rows"] == expected
    assert result["counts"]["matched_count"] == 1
    assert result["counts"]["total_count"] == len(report["pending_obligations"])
    assert result["counts"]["filtered_out_count"] == len(report["pending_obligations"]) - 1
    assert result["counts"]["out_of_path_count"] > 0
    assert calls and all(call["source_text"] == SOURCE.decode() and call["path"] == "first.py"
        and call["function_id"] == fid and call["kind"] == "library_semantics_and_effects" for call in calls)
    assert len(builder._wire(result)) < len(artifacts["formal"])


def test_multiple_native_pages_are_collected_without_omitting_or_rewriting_rows(artifacts, monkeypatch):
    from ipfs_datasets_py.logic.software_verification import numerical_obligation_retrieval as pager
    original, calls = pager.page_numerical_obligations, []
    monkeypatch.setattr(pager, "MAX_PAGE_BYTES", 2500)
    def wrapped(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(pager, "page_numerical_obligations", wrapped)
    result = select(artifacts, "obligations", query("first.py"))
    assert len(calls) > 1 and calls[0]["cursor"] is None
    assert all(call["cursor"] is not None for call in calls[1:])
    expected = json.loads(artifacts["formal"])["reports"]["first.py"]["pending_obligations"]
    assert result["rows"] == expected
    assert result["counts"]["matched_count"] == result["counts"]["returned_count"] == len(expected)
    assert result["counts"]["remaining_count"] == 0


@pytest.mark.parametrize("artifact,selection", [
    ("obligations", query("first.py", function_id="function:absent")),
    ("obligations", query("first.py", kind="unknown_kind")),
    ("capsules", query("first.py", symbol_id="unknown_symbol")),
])
def test_nonmatching_exact_filter_is_explicit_not_a_safety_claim(artifacts, artifact, selection):
    result = select(artifacts, artifact, selection)
    assert result["status"] == "no_matching_rows" and result["rows"] == []
    assert result["counts"]["total_count"] == result["counts"]["filtered_out_count"] > 0
    assert result["source_obligations_discharged"] == 0
    assert result["omission_authority"] is result["proof_authority"] is False


@pytest.mark.parametrize("artifact", ["capsules", "obligations"])
def test_support_file_is_bound_but_does_not_gain_program_facts(artifacts, artifact):
    result = select(artifacts, artifact, query("instruction.md"))
    assert result["status"] == "not_program" and result["rows"] == []
    assert result["coverage"]["role"] == "support"
    assert result["counts"]["total_count"] == result["counts"]["matched_count"] == 0
    assert result["counts"]["out_of_path_count"] > 0


@pytest.mark.parametrize("artifact", sorted(module.ARTIFACTS))
def test_out_of_scope_path_refuses(artifacts, artifact):
    with pytest.raises(ValueError, match="outside admitted"):
        select(artifacts, artifact, query("absent.py"))


def test_unsupported_partial_and_zero_obligations_keep_native_coverage():
    sources = {"bad.py": b"def broken(:\n", "foreign.js": b"let a = 1;",
        "empty.py": b"", "constant.py": b"def constant():\n    return 1\n",
        "loop.py": b"def loop(values):\n    for item in values:\n        print(item)\n"}
    artifacts = build(sources)
    for path in ("bad.py", "foreign.js", "empty.py"):
        result = select(artifacts, "obligations", query(path), sources)
        assert result["status"] == "unsupported" and result["rows"] == []
        assert result["coverage"]["formal_status"] == "unsupported"
        assert result["unsupported_families"]
    result = select(artifacts, "obligations", query("constant.py"), sources)
    assert result["status"] == "no_pending_obligations" and result["rows"] == []
    assert result["coverage"]["formal_status"] == "translated"
    result = select(artifacts, "catalog", query("loop.py"), sources)
    assert result["rows"][0]["formal_status"] == "partial"
    assert result["rows"][0]["structural_diagnostics"]
    assert result["rows"][0]["function_coverage"][0]["unsupported_constructs"]


def test_empty_program_scope_returns_only_support_catalog():
    sources = {"readme.md": b"support"}
    artifacts = build(sources, paths=())
    result = select(artifacts, sources=sources)
    assert result["rows"][0]["role"] == "support"
    assert result["rows"][0]["capsules"] == [] and result["rows"][0]["obligation_kinds"] == {}


def test_changed_argmax_argmin_requires_new_native_generation(artifacts):
    changed = dict(SOURCES, **{"first.py": SOURCE.replace(b"argmax", b"argmin")})
    with pytest.raises(ValueError, match="source identity"):
        select(artifacts, "obligations", query("first.py"), changed)
    fresh = build(changed, paths=("first.py", "other.py"), generation="b" * 64)
    old = select(artifacts, "obligations", query("first.py", kind="library_semantics_and_effects"))
    new = select(fresh, "obligations", query("first.py", kind="library_semantics_and_effects"), changed)
    assert old["rows"][0]["roles"]["imported_callee"] == "numpy.argmax"
    assert new["rows"][0]["roles"]["imported_callee"] == "numpy.argmin"
    assert old["binding"] != new["binding"]
    assert new["binding"]["workspace_generation"] == "b" * 64


def _rewrite(artifacts, change, *, relink=False):
    decoded = {key: json.loads(raw) for key, raw in artifacts.items()}
    change(decoded)
    result = {key: builder._wire(value) for key, value in decoded.items()}
    if relink:
        decoded["summary"]["artifacts"] = {key: dict(sha256=builder._sha(result[key]), bytes=len(result[key]))
            for key in ("capsules", "formal")}
        result["summary"] = builder._wire(decoded["summary"])
    return result


@pytest.mark.parametrize("change", [
    lambda values: values["formal"].update(workspace_generation="b" * 64),
    lambda values: values["capsules"].update(semantic_root_cid="invalid"),
    lambda values: values["summary"].update(proof_authority=True),
    lambda values: values["summary"].update(source_files=100),
    lambda values: values["summary"].update(producer_binding_sha256="0" * 64),
    lambda values: values["formal"]["reports"]["first.py"].update(report_sha256="0" * 64),
    lambda values: values["capsules"]["capsules"][0].update(docstring_hint="forged"),
    lambda values: values["summary"]["coverage"][0].update(capsule_count=100),
    lambda values: values["formal"]["profile"].update(unsupported_families=[]),
])
def test_tampered_parent_identity_or_native_payload_refuses_even_relinked(artifacts, change):
    altered = _rewrite(artifacts, change, relink=True)
    with pytest.raises(ValueError):
        select(altered)


def test_source_replay_refuses_report_changed_with_recomputed_report_and_parent_hashes(artifacts):
    def forge(values):
        report = values["formal"]["reports"]["first.py"]
        report["pending_obligations"][0]["reason"] = "Invented obligation"
        report["report_sha256"] = builder._sha(builder._wire({key: value for key, value in report.items()
            if key != "report_sha256"}))
        values["summary"]["coverage"][0]["formal_report_sha256"] = report["report_sha256"]
    altered = _rewrite(artifacts, forge, relink=True)
    with pytest.raises(ValueError, match="exact source replay"):
        select(altered, "obligations", query("first.py"))


def test_cross_path_capsule_with_recomputed_native_cid_still_refuses_source_binding(artifacts):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    def forge(values):
        row = values["capsules"]["capsules"][0]
        row["source_slice_path"] = "other.py"
        row["capsule_cid"] = cid_for_structured({key: value for key, value in row.items() if key != "capsule_cid"})
    altered = _rewrite(artifacts, forge, relink=True)
    with pytest.raises(ValueError, match="capsule source differs"):
        select(altered, "capsules", query("other.py"))


def test_parent_digest_and_canonical_bytes_required(artifacts):
    altered = dict(artifacts)
    altered["formal"] = artifacts["formal"].replace(b'"proof_authority":false', b'"proof_authority":true', 1)
    with pytest.raises(ValueError):
        select(altered)
    altered = dict(artifacts, summary=b" " + artifacts["summary"])
    with pytest.raises(ValueError, match="canonical"):
        select(altered)


@pytest.mark.parametrize("artifact,selection", [
    ("formal", query()), ("catalog", {}), ("catalog", dict(query(), extra=None)),
    ("catalog", query(symbol_id="symbol")), ("catalog", query(function_id="function:1")),
    ("catalog", query(kind="kind")), ("capsules", query()), ("obligations", query()),
    ("capsules", query("first.py", function_id="function:1")),
    ("capsules", query("first.py", kind="kind")),
    ("obligations", query("first.py", symbol_id="symbol")),
    ("catalog", query("../first.py")), ("catalog", query("/first.py")),
    ("catalog", query("a//first.py")), ("catalog", query(".git/config")),
    ("catalog", query("a\\first.py")), ("catalog", query("")),
    ("catalog", query("x" * 4097)), ("obligations", query("first.py", kind="x" * 513)),
    ("obligations", query("first.py", kind=True)),
    ("obligations", query("first.py", kind="bad\nkind")),
])
def test_invalid_selection_refuses_before_parent_or_datasets_work(monkeypatch, artifact, selection):
    monkeypatch.setattr(module, "_parents", lambda *_: pytest.fail("invalid selection reached native work"))
    with pytest.raises(ValueError):
        module.select_candidate_artifact(artifacts={}, sources={}, artifact=artifact, selection=selection)


def test_validation_returns_independent_copy():
    original = query("first.py")
    actual = module.validate_selection("catalog", original)
    assert actual == original and actual is not original
    original["path"] = "other.py"
    assert actual["path"] == "first.py"


def test_output_bound_refuses_whole_result_instead_of_silently_dropping_rows(artifacts, monkeypatch):
    # Parent bound must still permit the native artifacts while output alone is
    # checked with a lowered bound after the validated parent cache is read.
    original = module._parents
    def lower_after_parent(*args):
        value = original(*args)
        monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 64)
        return value
    monkeypatch.setattr(module, "_parents", lower_after_parent)
    with pytest.raises(ValueError, match="selection output byte bound"):
        select(artifacts)


def test_selector_opens_no_metadata_owner(artifacts, monkeypatch, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index as meta
    monkeypatch.setattr(meta, "_connect", lambda *_: pytest.fail("selection opened metadata owner"))
    database = tmp_path / "metadata.duckdb"
    monkeypatch.setenv(meta.ENV_DUCKDB, str(database))
    result = select(artifacts, "obligations", query("first.py"))
    assert not database.exists() and result["rows"]
    assert result["model_inference_calls"] == result["solver_calls"] == result["prover_calls"] == 0
