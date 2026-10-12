"""Real datasets capsule/logic generation over inert working source bytes."""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_artifacts as module


SOURCE = b"import numpy as np\ndef select(values):\n    return np.argmax(values)\n"


def build(sources=None, *, program_paths=None, generation="a" * 64):
    sources = {"program.py": SOURCE} if sources is None else sources
    paths = tuple(sorted(sources)) if program_paths is None else program_paths
    return module.build_candidate_artifacts(sources=sources, program_paths=paths,
        workspace_generation=generation)


@pytest.fixture(scope="module")
def baseline():
    return build()


def decode(result):
    return {key: json.loads(raw) for key, raw in result.items()}


def test_native_capsules_and_logic_are_unchanged_source_bound_payloads(baseline):
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticCapsule
    from ipfs_datasets_py.logic.software_verification.numerical_logic import formalize_numerical_source
    values = decode(baseline)
    assert set(values) == {"summary", "capsules", "formal"}
    assert all(module._wire(value) == baseline[key] for key, value in values.items())
    summary, capsules, formal = (values[key] for key in ("summary", "capsules", "formal"))
    assert summary["source_hashes"] == {"program.py": hashlib.sha256(SOURCE).hexdigest()}
    assert summary["capsule_count"] == len(capsules["capsules"]) >= 2
    assert summary["formal_report_count"] == 1
    assert summary["pending_obligation_count"] > 0
    assert formal["reports"]["program.py"] == formalize_numerical_source(SOURCE.decode(), path="program.py")
    assert all(SemanticCapsule.from_dict(row).to_dict() == row for row in capsules["capsules"])
    assert summary["coverage"][0]["families_with_artifacts"] == ["program", "first_order", "higher_order"]
    assert summary["coverage"][0]["formal_status"] == "translated"
    for key in ("capsules", "formal"):
        assert summary["artifacts"][key] == {"bytes": len(baseline[key]),
            "sha256": hashlib.sha256(baseline[key]).hexdigest()}
    for value in values.values():
        assert value["workspace_generation"] == "a" * 64
        assert value["source_state"] == "working-candidate"
        assert value["producer_binding_sha256"] == hashlib.sha256(module._wire(summary["producer_sources"])).hexdigest()
        assert all(value[key] == expected for key, expected in module.AUTHORITY.items())


def test_changed_signature_and_formula_rebuild_without_baseline_proof_reuse(baseline):
    old = decode(baseline)
    changed = SOURCE.replace(b"select(values)", b"select(values, offset=0)").replace(b"argmax", b"argmin")
    new = decode(build({"program.py": changed}, generation="b" * 64))
    assert old["summary"]["semantic_root_cid"] != new["summary"]["semantic_root_cid"]
    before = [row for row in old["capsules"]["capsules"] if row["signature"]]
    after = [row for row in new["capsules"]["capsules"] if row["signature"]]
    assert before and after and before != after
    old_report, new_report = [row["formal"]["reports"]["program.py"] for row in (old, new)]
    assert "SelectionMax" in json.dumps(old_report["projections"]["lean"])
    assert "SelectionMin" in json.dumps(new_report["projections"]["lean"])
    assert old_report["projections"] != new_report["projections"]
    assert new_report["coverage"]["source_obligations_discharged"] == 0
    assert all(row["status"] == "pending" for row in new_report["pending_obligations"])
    assert not new["summary"]["baseline_evidence_applicable"]


def test_repeat_build_from_same_generation_is_byte_identical(baseline):
    assert build() == baseline


def test_invalid_syntax_binary_empty_nonpython_and_grammar_size_are_accounted():
    sources = {"bad.py": b"def broken(:\n", "binary.py": b"\xff\0", "empty.py": b"",
        "foreign.js": b"function f() { return 1; }", "large.py": b"#" + b"x" * 65536 + b"\n"}
    values = decode(build(sources))
    coverage = {row["path"]: row for row in values["summary"]["coverage"]}
    assert set(coverage) == set(sources)
    assert all(row["formal_status"] == "unsupported" for row in coverage.values())
    assert coverage["bad.py"]["semantic_status"] == "opaque"
    assert set(values["formal"]["reports"]) == {"bad.py"}
    assert {row["path"] for row in values["formal"]["frontiers"]} == set(sources) - {"bad.py"}
    assert values["formal"]["reports"]["bad.py"]["structural"]["diagnostics"]
    assert values["summary"]["formal_frontier_count"] == 4


def test_partial_grammar_keeps_native_diagnostics_and_unsupported_families():
    values = decode(build({"loop.py": b"def f(values):\n    for x in values:\n        print(x)\n"}))
    report = values["formal"]["reports"]["loop.py"]
    assert report["summary"]["structural_status"] == "partial"
    assert report["structural"]["diagnostics"]
    assert report["unsupported_families"]
    assert report["summary"]["partial_or_unsupported_functions"] == 1


def test_support_sources_are_bound_but_not_scanned_or_formalized():
    sources = {"program.py": SOURCE, "support.py": b"def broken(:\n", "instruction.md": b"Keep intent.\n"}
    values = decode(build(sources, program_paths=("program.py",)))
    summary = values["summary"]
    assert summary["source_files"] == 3 and summary["program_files"] == 1 and summary["support_files"] == 2
    support = [row for row in summary["coverage"] if row["role"] == "support"]
    assert len(support) == 2
    assert all(row["semantic_status"] == "not_scanned_support" and row["formal_status"] == "not_selected_support"
        and row["capsule_count"] == 0 for row in support)
    assert set(values["formal"]["reports"]) == {"program.py"}
    assert all(row["source_slice_path"] == "program.py" for row in values["capsules"]["capsules"])


def test_empty_program_scope_has_no_invented_capsules_or_formal_facts():
    values = decode(build({"instruction.md": b"Only support."}, program_paths=()))
    assert values["capsules"]["capsules"] == []
    assert values["formal"]["reports"] == {}
    assert values["summary"]["program_files"] == values["summary"]["pending_obligation_count"] == 0


def test_target_not_executed_and_metadata_owner_not_opened(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index as meta
    marker, database, lake = (tmp_path / name for name in ("executed", "metadata.duckdb", "lake"))
    monkeypatch.setenv(meta.ENV_DUCKDB, str(database))
    monkeypatch.setenv(meta.ENV_DUCKLAKE, str(lake))
    monkeypatch.setattr(meta, "_connect", lambda *_: pytest.fail("candidate opened metadata owner"))
    source = (f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "def f(x):\n    return x + 1\n").encode()
    values = decode(build({"program.py": source}))
    assert not marker.exists() and not database.exists() and not lake.exists()
    assert values["summary"]["model_inference_calls"] == values["summary"]["prover_calls"] == 0


@pytest.mark.parametrize("sources,paths,generation", [
    ({}, (), "a" * 64), ({"x.py": "text"}, ("x.py",), "a" * 64),
    ({"../x.py": b"x"}, ("../x.py",), "a" * 64),
    ({"/x.py": b"x"}, ("/x.py",), "a" * 64),
    ({"a//x.py": b"x"}, ("a//x.py",), "a" * 64),
    ({".git/config": b"x"}, (".git/config",), "a" * 64),
    ({"x.py": b"x"}, ["x.py"], "a" * 64),
    ({"x.py": b"x"}, ("missing.py",), "a" * 64),
    ({"x.py": b"x"}, ("x.py", "x.py"), "a" * 64),
    ({"a.py": b"x", "b.py": b"x"}, ("b.py", "a.py"), "a" * 64),
    ({"x.py": b"x"}, ("x.py",), "A" * 64),
    ({"x.py": b"x"}, ("x.py",), "invalid"),
    ({f"{i}.py": b"x" for i in range(65)}, (), "a" * 64),
    ({"x.py": b"x" * (module.MAX_SOURCE_BYTES + 1)}, (), "a" * 64),
])
def test_invalid_scope_refused_before_native_work(monkeypatch, sources, paths, generation):
    monkeypatch.setattr(module, "_producer_binding", lambda: pytest.fail("invalid input reached producers"))
    with pytest.raises(ValueError):
        module.build_candidate_artifacts(sources=sources, program_paths=paths, workspace_generation=generation)


def test_output_bound_is_refusal_not_truncation(monkeypatch):
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 64)
    with pytest.raises(ValueError, match="output byte bound"):
        build()


def test_native_symbol_bound_is_refusal_not_partial_scan():
    source = "".join(f"def f_{index}():\n    return {index}\n" for index in range(257)).encode()
    with pytest.raises(ValueError, match="symbol bound"):
        build({"program.py": source})


def test_producer_changes_during_build_are_refused(monkeypatch):
    calls = iter(({"producer": "a" * 64}, {"producer": "b" * 64}))
    monkeypatch.setattr(module, "_producer_binding", lambda: next(calls))
    with pytest.raises(ValueError, match="producer sources changed"):
        build()
