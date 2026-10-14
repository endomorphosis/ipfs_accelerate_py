"""Qualification receipts must preserve scope, input pins and execution identity."""
import copy
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import zlib

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/legal_autoformalization_source_alignment.py"
spec = importlib.util.spec_from_file_location("span_section_driver_test", SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    controller = tmp_path / "controller"
    datasets = tmp_path / "datasets"
    controller.mkdir()
    datasets.mkdir()
    (controller / "config").mkdir()
    board_name = "docs/recovery.todo.md"
    board = controller / board_name
    board.parent.mkdir()
    initial = "# Fictional recovery fixture\n\n" + "".join(
        "## " + task + " Fictional task\n\n- Status: todo\n"
        "- Validation: fictional-task-command\n- Outputs: fictional-result\n\n"
        for task in driver.TASKS)
    board.write_text(initial, encoding="utf-8")
    template = tmp_path / "initial-board-template.md"
    template.write_text(initial, encoding="utf-8")
    (controller / "config/legal_autoformalization_source_alignment_scheduler.json").write_text(
        json.dumps({"taskboard_path": board_name}))
    (datasets / "owner.py").write_text("PROFILE = 'fictional-source-contract'\n")
    (datasets / "test_owner.py").write_text("def test_fictional_fixture():\n    assert 'é'.encode() == b'\\xc3\\xa9'\n")
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                  "commit", "-qm", "Freeze fixture"]):
        subprocess.run(["git", *args], cwd=datasets, check=True, capture_output=True)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"status": "bounded_literal_parent_views_occurrence_inventory_and_sentence_candidates_complete"}))
    cfg = {"schema": "legal-autoformalization-source-alignment-sources/v1", "campaign_id": "fixture",
        "datasets": {"root": str(datasets), "commit": driver.git(datasets, "rev-parse", "HEAD")},
        "prior_span_reconstruction_completion": driver.identity(prior),
        "initial_board_template": driver.identity(template),
        "jobs": {"LAC-S11-044": {"test_paths": ["test_owner.py"], "code_paths": ["owner.py"],
                                 "minimum_tests": 1, "timeout_seconds": 30}},
        "immutable_inputs": [{"path": name, "sha256": driver.identity(datasets / name)["sha256"]}
                             for name in ("owner.py", "test_owner.py")]}
    sources = controller / "sources.json"
    sources.write_text(json.dumps(cfg))
    monkeypatch.setattr(driver, "ROOT", controller)
    return datasets, sources, cfg


@pytest.mark.parametrize("subset", range(8))
def test_only_eight_native_status_projections_have_stable_bindings(frozen, subset):
    _, sources, cfg = frozen
    initial = Path(cfg["initial_board_template"]["path"]).read_bytes()
    before = driver.bindings("LAC-S11-044", sources)[3]
    lines = initial.splitlines(keepends=True)
    statuses = [i for i, line in enumerate(lines) if line == b"- Status: todo\n"]
    for offset, index in enumerate(statuses):
        if subset & (1 << offset):
            lines[index] = b"- Status: completed\n"
    (driver.ROOT / "docs/recovery.todo.md").write_bytes(b"".join(lines))
    after = driver.bindings("LAC-S11-044", sources)[3]
    assert before == after
    assert after["taskboard"] == {
        "profile": driver.TASKBOARD_PROJECTION_PROFILE,
        "repository_path": "docs/recovery.todo.md",
        "initial_board_template": cfg["initial_board_template"],
        "allowed_states": ["todo", "completed"]}


def test_completed_native_board_replays_existing_receipt_without_rewriting(frozen, monkeypatch):
    _, sources, _ = frozen
    driver.materialize("LAC-S11-044", sources)
    output = driver.output("LAC-S11-044")
    original = output.read_bytes(), output.stat().st_mtime_ns
    board = driver.ROOT / "docs/recovery.todo.md"
    board.write_bytes(board.read_bytes().replace(b"- Status: todo\n", b"- Status: completed\n"))
    result = driver.read_json(output)
    assert driver.validate(result, "LAC-S11-044", sources)["valid"] is True
    monkeypatch.setattr(driver, "execute", lambda *args: pytest.fail("existing receipt must replay"))
    assert driver.materialize("LAC-S11-044", sources)["materialized"] is False
    assert (output.read_bytes(), output.stat().st_mtime_ns) == original


def test_mutable_board_in_immutable_population_is_rejected_before_dispatch(frozen):
    _, sources, cfg = frozen
    board = driver.ROOT / "docs/recovery.todo.md"
    cfg["immutable_inputs"].append({"repository": "accelerator", "path": "docs/recovery.todo.md",
                                    "sha256": driver.identity(board)["sha256"]})
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Mutable configured taskboard"):
        driver.bindings("LAC-S11-044", sources)
    assert not driver.output("LAC-S11-044").exists()


def test_immutable_alias_resolving_to_mutable_board_is_rejected(frozen):
    _, sources, cfg = frozen
    board = driver.ROOT / "docs/recovery.todo.md"
    (driver.ROOT / "board-alias.md").symlink_to(board)
    cfg["immutable_inputs"].append({"repository": "accelerator", "path": "board-alias.md",
                                    "sha256": driver.identity(board)["sha256"]})
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Mutable configured taskboard"):
        driver.bindings("LAC-S11-044", sources)


@pytest.mark.parametrize("mutation", ["failed", "extra-text", "task-id", "extra-status", "reordered-tasks",
                                       "command", "output", "whitespace"])
def test_resealed_source_cannot_admit_unapproved_board_projection(frozen, mutation):
    _, sources, _ = frozen
    board = driver.ROOT / "docs/recovery.todo.md"
    initial = board.read_bytes()
    changed = {
        "failed": initial.replace(b"- Status: todo\n", b"- Status: failed\n", 1),
        "extra-text": initial + b"An unrelated edit.\n",
        "task-id": initial.replace(b"LAC-S11-011", b"LAC-S11-099"),
        "extra-status": initial + b"- Status: completed\n",
        "reordered-tasks": initial.replace(b"LAC-S11-011", b"TEMP").replace(
            b"LAC-S11-024", b"LAC-S11-011").replace(b"TEMP", b"LAC-S11-024"),
        "command": initial.replace(b"fictional-task-command", b"other-command", 1),
        "output": initial.replace(b"fictional-result", b"other-result", 1),
        "whitespace": initial.replace(b"- Status: todo\n", b"- Status: todo \n", 1)}[mutation]
    board.write_bytes(changed)
    with pytest.raises(ValueError, match="status-only projection"):
        driver.bindings("LAC-S11-044", sources)


@pytest.mark.parametrize("mutation", ["missing", "extra-key", "wrong-sha", "wrong-bytes", "bool-bytes", "over-bound"])
def test_independent_initial_template_pin_is_required_closed_and_bounded(frozen, mutation):
    _, sources, cfg = frozen
    if mutation == "missing":
        del cfg["initial_board_template"]
    elif mutation == "extra-key":
        cfg["initial_board_template"]["self_authenticated"] = True
    elif mutation == "wrong-sha":
        cfg["initial_board_template"]["sha256"] = "0" * 64
    elif mutation == "wrong-bytes":
        cfg["initial_board_template"]["bytes"] += 1
    elif mutation == "bool-bytes":
        cfg["initial_board_template"]["bytes"] = True
    else:
        cfg["initial_board_template"]["bytes"] = driver.MAX_TASKBOARD_BYTES + 1
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="taskboard template pin"):
        driver.bindings("LAC-S11-044", sources)


@pytest.mark.parametrize("mutation", ["precompleted", "missing-task", "duplicate-status", "unknown-header", "empty-header"])
def test_rehashed_initial_template_must_be_exactly_three_initial_todo_tasks(frozen, mutation):
    _, sources, cfg = frozen
    template = Path(cfg["initial_board_template"]["path"])
    initial = template.read_bytes()
    changed = {
        "precompleted": initial.replace(b"- Status: todo\n", b"- Status: completed\n", 1),
        "missing-task": initial[:initial.index(b"## LAC-S11-044")],
        "duplicate-status": initial + b"- Status: todo\n",
        "unknown-header": initial + b"## OTHER Task\n",
        "empty-header": initial + b"## \n"}[mutation]
    template.write_bytes(changed)
    cfg["initial_board_template"] = driver.identity(template)
    sources.write_text(json.dumps(cfg))
    (driver.ROOT / "docs/recovery.todo.md").write_bytes(changed)
    with pytest.raises(ValueError, match="Initial taskboard template"):
        driver.bindings("LAC-S11-044", sources)


@pytest.mark.parametrize("target", ["template", "board"])
def test_taskboard_custody_refuses_symlink_aliases(frozen, target):
    _, sources, cfg = frozen
    if target == "template":
        original = Path(cfg["initial_board_template"]["path"])
        alias = original.parent / "template-alias.md"
        alias.symlink_to(original)
        cfg["initial_board_template"]["path"] = str(alias)
        sources.write_text(json.dumps(cfg))
    else:
        board = driver.ROOT / "docs/recovery.todo.md"
        board.unlink()
        board.symlink_to(cfg["initial_board_template"]["path"])
    with pytest.raises(ValueError, match="alias|escaped pinned repository"):
        driver.bindings("LAC-S11-044", sources)


@pytest.fixture
def alignment_input_fixture(frozen):
    """Small fixed-owner stand-ins exercise the real bounded child dispatcher.

    Actual source semantics are qualified by the separately pinned production
    owner suites; these fixtures check file custody, FULL pins and API wiring.
    """
    datasets, _, _ = frozen
    inputs = datasets.parent / "alignment-inputs"
    inputs.mkdir()
    digest = lambda value: driver.sha(driver.wire(value))
    def put(name, value):
        path = inputs / name
        path.write_bytes(driver.wire(value) + b"\n")
        return driver.identity(path)
    descriptor = {"profile": "fictional-parent", "revision": "a" * 40}
    selected = {"rows": [{"row_ordinal": 4, "raw_row_sha256": "b" * 64}]}
    preflight = {"row_count": 1}
    parent = {"selected_rows_report": selected, "fixture": "parent", "authority": {"legal": False}}
    sentence = {"fixture": "sentence", "parent_sha256": digest(parent), "authority": {"legal": False}}
    raw = inputs / "official.html"
    raw.write_bytes(b"Fictional official comparison bytes")
    view = {"fixture": "official-view", "authority": {"legal": False}}
    document_descriptor = {"legal_id": "usc:us:10:1", "edition": "2024"}
    view_file = put("official-view.json", view)
    document = {"raw_path": str(raw), "descriptor": document_descriptor,
        "expected_descriptor_sha256": digest(document_descriptor), "expected_raw_sha256": driver.identity(raw)["sha256"],
        "expected_raw_bytes": raw.stat().st_size, "view_report_path": view_file["path"],
        "view_report_sha256": view_file["sha256"], "view_report_bytes": view_file["bytes"],
        "expected_view_report_sha256": digest(view)}
    doc_population = digest([{key: document[key] for key in ("expected_descriptor_sha256",
        "expected_raw_sha256", "expected_raw_bytes", "expected_view_report_sha256")}])
    alignment = {"fixture": "alignment", "parent_sha256": digest(parent),
        "document_population_sha256": doc_population, "authority": {"legal": False}}
    context = {"fixture": "context", "sentence_sha256": digest(sentence),
        "alignment_sha256": digest(alignment), "authority": {"legal": False}}
    handoff = {"fixture": "handoff", "context_sha256": digest(context), "status": "pending",
        "authority": {"legal": False}}
    payload = {"schema": "legal-source-alignment-population-inputs/v1",
        "parent_descriptor": descriptor, "expected_parent_descriptor_sha256": digest(descriptor),
        "selected_rows_report_file": put("selected.json", selected), "expected_selected_rows_report_sha256": digest(selected),
        "preflight_report_file": put("preflight.json", preflight), "expected_preflight_sha256": digest(preflight),
        "row_ordinals": [4], "expected_row_sha256s": ["b" * 64],
        "parent_text_views_report_file": put("parent.json", parent), "expected_parent_text_views_report_sha256": digest(parent),
        "sentence_candidates_report_file": put("sentence.json", sentence), "expected_sentence_candidates_report_sha256": digest(sentence),
        "official_documents": [document], "expected_document_population_sha256": doc_population,
        "alignment_report_file": put("alignment.json", alignment), "expected_original_source_alignment_report_sha256": digest(alignment),
        "context_report_file": put("context.json", context), "expected_candidate_context_inventory_report_sha256": digest(context),
        "expected_source_review_handoff_report_sha256": digest(handoff)}
    owner = datasets / "ipfs_datasets_py/logic/autoformal"
    owner.mkdir(parents=True)
    for package in [datasets / "ipfs_datasets_py", datasets / "ipfs_datasets_py/logic", owner]:
        (package / "__init__.py").write_text("")
    values = {"parent": parent, "sentence": sentence, "alignment": alignment, "context": context,
              "handoff": handoff, "selected": selected, "preflight": preflight, "descriptor": descriptor}
    common = '''import hashlib,json
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def need(ok):
 if not ok:raise ValueError("Independent fixed fixture owner contract differs")
VALUES=json.loads(%r)
def parent_inputs(args):
 need(set(args)=={"expected_selected_rows_report_sha256","parent_descriptor","expected_parent_descriptor_sha256","preflight_report","expected_preflight_sha256","row_ordinals","expected_row_sha256s"})
 need(args["expected_selected_rows_report_sha256"]==digest(VALUES["selected"]) and args["parent_descriptor"]==VALUES["descriptor"] and args["expected_parent_descriptor_sha256"]==digest(VALUES["descriptor"]))
 need(args["preflight_report"]==VALUES["preflight"] and args["expected_preflight_sha256"]==digest(VALUES["preflight"]) and args["row_ordinals"]==[4] and args["expected_row_sha256s"]==["b"*64])
def sentence_inputs(args):
 need(set(args)=={"expected_parent_text_views_report_sha256","parent_text_views_inputs"} and args["expected_parent_text_views_report_sha256"]==digest(VALUES["parent"]))
 parent_inputs(args["parent_text_views_inputs"])
def alignment_inputs(args):
 need(set(args)=={"expected_parent_text_views_report_sha256","parent_text_views_inputs","document_inputs","expected_document_population_sha256"} and args["expected_parent_text_views_report_sha256"]==digest(VALUES["parent"]))
 parent_inputs(args["parent_text_views_inputs"])
 documents=args["document_inputs"]
 need(len(documents)==1 and set(documents[0])=={"descriptor","raw_bytes","expected_descriptor_sha256","expected_raw_sha256","expected_raw_bytes","view_report","expected_view_report_sha256"})
 d=documents[0];need(digest(d["descriptor"])==d["expected_descriptor_sha256"] and digest(d["view_report"])==d["expected_view_report_sha256"] and hashlib.sha256(d["raw_bytes"]).hexdigest()==d["expected_raw_sha256"] and len(d["raw_bytes"])==d["expected_raw_bytes"])
 need(args["expected_document_population_sha256"]==VALUES["alignment"]["document_population_sha256"])
def context_inputs(args):
 need(set(args)=={"expected_sentence_candidates_report_sha256","sentence_inputs","expected_alignment_report_sha256","alignment_inputs"})
 need(args["expected_sentence_candidates_report_sha256"]==digest(VALUES["sentence"]) and args["expected_alignment_report_sha256"]==digest(VALUES["alignment"]))
 sentence_inputs(args["sentence_inputs"]);alignment_inputs(args["alignment_inputs"])
''' % json.dumps(values)
    (owner / "_fixture_contract.py").write_text(common)
    modules = {
        "legal_parent_text_views": '''from ._fixture_contract import *
def validate_parent_text_views(report,selected_rows_report,*,expected_report_sha256,**kwargs):
 parent_inputs(kwargs);need(report==VALUES["parent"] and selected_rows_report==VALUES["selected"] and digest(report)==expected_report_sha256);return {"valid":True,"validation_receipt_only":True}
''',
        "legal_source_view_inventory": '''from ._fixture_contract import *
def view_document_population_sha256(documents):return digest([{key:d[key] for key in ("expected_descriptor_sha256","expected_raw_sha256","expected_raw_bytes","expected_view_report_sha256")} for d in documents])
''',
        "legal_parent_source_alignment": '''from ._fixture_contract import *
def prepare_parent_source_alignment(report,**kwargs):
 alignment_inputs(kwargs);need(report==VALUES["parent"]);return VALUES["alignment"]
parent_source_alignment_report_sha256=digest
''',
        "legal_parent_sentence_candidates": '''from ._fixture_contract import *
def validate_parent_sentence_candidates(report,parent,*,expected_report_sha256,**kwargs):
 sentence_inputs(kwargs);need(report==VALUES["sentence"] and parent==VALUES["parent"] and digest(report)==expected_report_sha256);return {"valid":True,"validation_receipt_only":True}
''',
        "legal_candidate_context_inventory": '''from ._fixture_contract import *
def prepare_candidate_context_inventory(sentence,alignment,**kwargs):
 context_inputs(kwargs);need(sentence==VALUES["sentence"] and alignment==VALUES["alignment"]);return VALUES["context"]
candidate_context_inventory_report_sha256=digest
''',
        "legal_source_review_handoff": '''from ._fixture_contract import *
def prepare_source_review_handoff(report,*,expected_candidate_context_inventory_report_sha256,context_inputs):
 need(report==VALUES["context"] and digest(report)==expected_candidate_context_inventory_report_sha256)
 need(set(context_inputs)=={"sentence_candidates_report","alignment_report","expected_sentence_candidates_report_sha256","sentence_inputs","expected_alignment_report_sha256","alignment_inputs"})
 need(context_inputs["sentence_candidates_report"]==VALUES["sentence"] and context_inputs["alignment_report"]==VALUES["alignment"])
 globals()["context_inputs"]({key:value for key,value in context_inputs.items() if key not in {"sentence_candidates_report","alignment_report"}})
 return VALUES["handoff"]
source_review_handoff_report_sha256=digest
'''}
    for name, source in modules.items():
        (owner / (name + ".py")).write_text(source)
    master = inputs / "master.json"
    def job(kind):
        master.write_bytes(driver.wire(payload) + b"\n")
        pin = driver.identity(master)
        return {"real_input": {"spec": {"kind": kind, "path": pin["path"],
            "file_bytes": pin["bytes"], "file_sha256": pin["sha256"]}}}
    return datasets, payload, values, job, put


@pytest.mark.parametrize("kind,expected", [("original_source_alignment", "alignment"),
    ("candidate_context_inventory", "context"), ("source_review_handoff", "handoff")])
def test_source_alignment_dispatcher_replays_exact_fixed_owner_bundles(alignment_input_fixture, kind, expected):
    datasets, _, values, job, _ = alignment_input_fixture
    result = driver.run_real_input(datasets, job(kind))
    assert result == {"schema": "legal-source-alignment-real-input/v1", "kind": kind, "report": values[expected]}
    assert not any(result["report"]["authority"].values())
    assert driver.unpack_real_input_report(driver.pack_real_input_report(result)) == result


@pytest.mark.parametrize("field,kind", [("expected_parent_text_views_report_sha256", "original_source_alignment"),
    ("expected_document_population_sha256", "original_source_alignment"),
    ("expected_original_source_alignment_report_sha256", "original_source_alignment"),
    ("expected_sentence_candidates_report_sha256", "candidate_context_inventory"),
    ("expected_candidate_context_inventory_report_sha256", "candidate_context_inventory"),
    ("expected_source_review_handoff_report_sha256", "source_review_handoff")])
def test_rehashed_master_cannot_replace_independent_full_population_pins(alignment_input_fixture, field, kind):
    datasets, payload, _, job, _ = alignment_input_fixture
    payload[field] = "0" * 64
    with pytest.raises(ValueError, match="Real-input observation failed"):
        driver.run_real_input(datasets, job(kind))


@pytest.mark.parametrize("field,kind", [("alignment_report_file", "original_source_alignment"),
    ("context_report_file", "source_review_handoff")])
def test_resealed_cached_report_cannot_replace_fixed_owner_replay(alignment_input_fixture, field, kind):
    datasets, payload, _, job, put = alignment_input_fixture
    forged = driver.read_json(payload[field]["path"])
    forged["fixture"] = "provider-resealed-incomplete-population"
    payload[field] = put("forged.json", forged)
    with pytest.raises(ValueError, match="Retained .* differs from fixed-owner replay"):
        driver.run_real_input(datasets, job(kind))


@pytest.mark.parametrize("mutation", ["extra-field", "oversized-report-population", "raw-file-alias"])
def test_official_input_population_is_closed_and_bounded_before_retention(alignment_input_fixture, mutation):
    datasets, payload, _, job, _ = alignment_input_fixture
    document = payload["official_documents"][0]
    if mutation == "extra-field":
        document["infer_source_revision"] = True
        message = "Closed official comparison document"
    elif mutation == "oversized-report-population":
        document["view_report_bytes"] = 16 * 1024 * 1024
        document["view_report_path"] = "/unavailable-refused-before-capture"
        payload["official_documents"].append(copy.deepcopy(document))
        message = "before capture"
    else:
        raw = Path(document["raw_path"])
        alias = raw.parent / "raw-alias.html"
        alias.symlink_to(raw)
        document["raw_path"] = str(alias)
        message = "input alias refused"
    with pytest.raises(ValueError, match=message):
        driver.run_real_input(datasets, job("original_source_alignment"))


@pytest.fixture
def referenced_report(tmp_path, monkeypatch):
    # Force the transport boundary cheaply; real reports are qualified separately
    # against the unchanged native caps and complete pinned production baselines.
    monkeypatch.setattr(driver, "MAX_COMPRESSED_REPORT_BYTES", 32)
    def build(kind="original_source_alignment", name="baseline.json"):
        report = {"schema": "legal-source-alignment-real-input/v1", "kind": kind,
            "report": {"rows": [{"ordinal": i, "text": "Literal é evidence " + str(i)} for i in range(16)],
                "unknown_metadata": ["retained", {"nested": "entire population"}], "authority": {"legal": False}}}
        path = tmp_path / name
        path.write_bytes(driver.wire(report) + b"\n")
        baseline = {**driver.identity(path), "canonical_json_sha256": driver.sha(driver.wire(report))}
        return report, baseline
    return build


@pytest.mark.parametrize("kind", driver.SOURCE_ALIGNMENT_KINDS)
def test_reference_fallback_losslessly_recovers_every_full_report_field(referenced_report, kind):
    report, baseline = referenced_report(kind)
    envelope = driver.pack_real_input_report(report, trusted_baseline=baseline)
    assert set(envelope) == {"schema", "encoding", "baseline", "report_sha256", "report_bytes", "report_schema", "kind"}
    assert envelope["schema"] == driver.REPORT_REFERENCE_SCHEMA
    assert envelope["encoding"] == "pinned-baseline-reference"
    assert envelope["report_bytes"] == len(driver.wire(report))
    assert envelope["baseline"] == baseline
    assert driver.unpack_real_input_report(envelope, trusted_baseline=baseline) == report
    assert len(driver.wire(envelope)) < driver.MAX_RESULT_BYTES
    job = {"real_input": {"spec": {"kind": kind}, "baseline": baseline}}
    driver.validate_real_input({"kind": kind, "report": envelope, "report_sha256": driver.sha(driver.wire(report))}, job)


def test_fallback_compares_whole_generated_payload_even_when_hash_helper_matches(referenced_report, monkeypatch):
    report, baseline = referenced_report()
    other = copy.deepcopy(report)
    other["report"]["rows"].pop()
    real_sha = driver.sha
    # An apparent equality of digest strings must not replace whole byte equality.
    def colliding_sha(raw):
        if raw == driver.wire(other):
            return baseline["canonical_json_sha256"]
        return real_sha(raw)
    monkeypatch.setattr(driver, "sha", colliding_sha)
    with pytest.raises(ValueError, match="Generated complete report differs"):
        driver.pack_real_input_report(other, trusted_baseline=baseline)


@pytest.mark.parametrize("damage", ["missing-trust", "foreign-path", "raw-hash", "canonical-hash", "bytes", "extra-pin-key"])
def test_reference_cannot_choose_its_own_baseline_before_file_open(referenced_report, monkeypatch, damage):
    report, baseline = referenced_report()
    envelope = driver.pack_real_input_report(report, trusted_baseline=baseline)
    trusted = baseline
    if damage == "missing-trust":
        trusted = None
    elif damage == "foreign-path":
        envelope["baseline"]["path"] = str(Path(baseline["path"]).parent / "provider-chosen.json")
    elif damage == "raw-hash":
        envelope["baseline"]["sha256"] = "0" * 64
    elif damage == "canonical-hash":
        envelope["baseline"]["canonical_json_sha256"] = "0" * 64
    elif damage == "bytes":
        envelope["baseline"]["bytes"] += 1
    else:
        envelope["baseline"]["self_authenticated"] = True
    def deny_open(*args, **kwargs):
        pytest.fail("untrusted reference must not open any file")
    monkeypatch.setattr(driver.os, "open", deny_open)
    with pytest.raises(ValueError):
        driver.unpack_real_input_report(envelope, trusted_baseline=trusted)


@pytest.mark.parametrize("damage", ["report-sha", "report-size", "boolean-size", "schema", "kind", "extra-key", "encoding"])
def test_resealed_reference_metadata_cannot_change_full_pinned_payload(referenced_report, damage):
    report, baseline = referenced_report()
    envelope = driver.pack_real_input_report(report, trusted_baseline=baseline)
    if damage == "report-sha": envelope["report_sha256"] = "0" * 64
    elif damage == "report-size": envelope["report_bytes"] -= 1
    elif damage == "boolean-size": envelope["report_bytes"] = True
    elif damage == "schema": envelope["report_schema"] = "pretend-legal-authority/v1"
    elif damage == "kind": envelope["kind"] = "source_review_handoff"
    elif damage == "encoding": envelope["encoding"] = "unverified-pointer"
    else: envelope["passed"] = True
    with pytest.raises(ValueError):
        driver.unpack_real_input_report(envelope, trusted_baseline=baseline)


def test_modified_baseline_body_cannot_be_recovered_from_old_reference(referenced_report):
    report, baseline = referenced_report()
    envelope = driver.pack_real_input_report(report, trusted_baseline=baseline)
    changed = copy.deepcopy(report)
    changed["report"]["rows"][0]["text"] = "changed evidence"
    Path(baseline["path"]).write_bytes(driver.wire(changed) + b"\n")
    with pytest.raises(ValueError, match="Trusted baseline"):
        driver.unpack_real_input_report(envelope, trusted_baseline=baseline)


@pytest.mark.parametrize("alias", ["symlink", "raw-path", "dot-path", "repeated-separator"])
def test_independently_pinned_baseline_custody_refuses_aliases(referenced_report, alias):
    report, baseline = referenced_report()
    target = Path(baseline["path"])
    if alias == "symlink":
        path = target.parent / "baseline-alias.json"
        path.symlink_to(target)
    elif alias == "raw-path":
        directory = target.parent / "subdirectory"
        directory.mkdir()
        path = directory / ".." / target.name
    elif alias == "dot-path":
        path = str(target.parent) + "/./" + target.name
    else:
        path = str(target.parent) + "//" + target.name
    baseline["path"] = str(path)
    with pytest.raises(ValueError, match="alias refused"):
        driver.pack_real_input_report(report, trusted_baseline=baseline)


@pytest.mark.parametrize("damage", ["raw-cap", "canonical-pin", "raw-pin", "boolean-bytes", "raised-bytes", "wrong-kind", "wrong-schema"])
def test_reference_fallback_requires_exact_bounded_external_file_pins(referenced_report, monkeypatch, damage):
    report, baseline = referenced_report()
    if damage == "raw-cap": monkeypatch.setattr(driver, "MAX_BYTES", len(driver.wire(report)) - 1)
    elif damage == "canonical-pin": baseline["canonical_json_sha256"] = "0" * 64
    elif damage == "raw-pin": baseline["sha256"] = "0" * 64
    elif damage == "boolean-bytes": baseline["bytes"] = True
    elif damage == "raised-bytes": baseline["bytes"] = driver.MAX_BYTES + 1
    else:
        changed = copy.deepcopy(report)
        changed["kind" if damage == "wrong-kind" else "schema"] = "foreign-stage-profile"
        Path(baseline["path"]).write_bytes(driver.wire(changed) + b"\n")
        baseline = {**driver.identity(baseline["path"]), "canonical_json_sha256": driver.sha(driver.wire(changed))}
    with pytest.raises(ValueError):
        driver.pack_real_input_report(report, trusted_baseline=baseline)


def test_older_profile_cannot_use_stage11_reference_even_with_matching_file_pin(referenced_report):
    report, baseline = referenced_report()
    report["schema"] = "legal-source-views-real-input/v1"
    report["kind"] = "source_view_inventory"
    path = Path(baseline["path"])
    path.write_bytes(driver.wire(report) + b"\n")
    baseline = {**driver.identity(path), "canonical_json_sha256": driver.sha(driver.wire(report))}
    with pytest.raises(ValueError, match="Compressed full real-input report exceeds"):
        driver.pack_real_input_report(report, trusted_baseline=baseline)


def test_reference_is_not_substituted_for_complete_subprocess_execution(alignment_input_fixture, monkeypatch):
    datasets, payload, values, job_builder, put = alignment_input_fixture
    job = job_builder("original_source_alignment")
    job.update({"test_paths": ["test_owner.py"], "minimum_tests": 1})
    full = {"schema": "legal-source-alignment-real-input/v1", "kind": "original_source_alignment", "report": values["alignment"]}
    baseline = put("baseline.json", full)
    baseline["canonical_json_sha256"] = driver.sha(driver.wire(full))
    job["real_input"]["baseline"] = baseline
    monkeypatch.setattr(driver, "MAX_COMPRESSED_REPORT_BYTES", 32)
    calls = []
    def actual_child(*args):
        calls.append(1)
        raise ValueError("fixed owner child failed; baseline is not execution")
    monkeypatch.setattr(driver, "run_real_input", actual_child)
    with pytest.raises(ValueError, match="baseline is not execution"):
        driver.execute(datasets, job)
    assert calls == [1]


def test_frozen_qualification_executes_real_test_and_never_rewrites(frozen, monkeypatch):
    _, sources, _ = frozen
    first = driver.materialize("LAC-S11-044", sources)
    assert first["valid"] is True and first["test_count"] == 1
    target = driver.output("LAC-S11-044")
    before = target.read_bytes(), target.stat().st_mtime_ns
    monkeypatch.setattr(driver, "execute", lambda *args: pytest.fail("must not rerun during materialize"))
    assert driver.materialize("LAC-S11-044", sources)["materialized"] is False
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_tracked_source_mutation_refuses_qualification(frozen):
    datasets, sources, _ = frozen
    driver.materialize("LAC-S11-044", sources)
    (datasets / "owner.py").write_text("PROFILE = 'changed'\n")
    with pytest.raises(ValueError, match="tracked changes"):
        driver.materialize("LAC-S11-044", sources)


def test_tampered_receipt_and_scope_are_rejected_without_repair(frozen):
    _, sources, _ = frozen
    driver.materialize("LAC-S11-044", sources)
    target = driver.output("LAC-S11-044")
    original = json.loads(target.read_text())
    for field, value in (("legal_semantic_acceptance", True), ("corpus_coverage_complete", True)):
        altered = copy.deepcopy(original)
        altered[field] = value
        target.write_text(json.dumps(altered))
        before = target.read_bytes()
        with pytest.raises(ValueError, match="acceptance scope"):
            driver.materialize("LAC-S11-044", sources)
        assert target.read_bytes() == before
    altered = copy.deepcopy(original)
    altered["test_execution"]["passed_count"] = 17
    with pytest.raises(ValueError, match="denominator"):
        driver.validate(altered, "LAC-S11-044", sources)


def test_source_path_escape_and_missing_population_are_rejected(frozen):
    _, sources, cfg = frozen
    cfg["jobs"]["LAC-S11-044"]["code_paths"] = ["../outside.py"]
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="relative source path"):
        driver.materialize("LAC-S11-044", sources)
    cfg["jobs"]["LAC-S11-044"]["code_paths"] = ["owner.py"]
    cfg["jobs"]["LAC-S11-044"]["minimum_tests"] = 2
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="population shrank"):
        driver.materialize("LAC-S11-044", sources)
    assert not driver.output("LAC-S11-044").exists()


def test_wrong_interpreter_and_output_alias_are_rejected(frozen):
    _, sources, cfg = frozen
    cfg["jobs"]["LAC-S11-044"]["python_executable_sha256"] = "0" * 64
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="interpreter moved"):
        driver.materialize("LAC-S11-044", sources)
    outside = driver.ROOT.parent / "outside-results"
    outside.mkdir()
    (driver.ROOT / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="output path alias"):
        driver.output("LAC-S11-044")


def test_failing_real_test_cannot_produce_a_passing_receipt(frozen):
    datasets, _, cfg = frozen
    (datasets / "test_owner.py").write_text("def test_failure():\n    assert False, 'named qualification cause'\n")
    with pytest.raises(driver.QualificationFailure, match="test_failure") as caught:
        driver.execute(datasets, cfg["jobs"]["LAC-S11-044"])
    diagnostics = caught.value.diagnostics
    assert diagnostics["failed_cases"][0]["name"] == "test_failure"
    assert "named qualification cause" in diagnostics["failed_cases"][0]["message"]
    assert diagnostics["failed_cases"][0]["status"] == "failure"
    assert diagnostics["returncode"] != 0
    payload = driver.refusal_payload(caught.value)
    assert payload["valid"] is False
    assert "test_failure" in payload["error"]
    assert json.loads(json.dumps(payload))["diagnostics"] == diagnostics


def test_qualification_binds_imports_and_clears_ambient_pytest_flags(frozen, monkeypatch):
    datasets, _, cfg = frozen
    (datasets / "test_owner.py").write_text(
        "import owner\ndef test_exact_pinned_import():\n"
        "    assert owner.PROFILE == 'fictional-source-contract'\n")
    monkeypatch.setenv("PYTHONPATH", str(datasets.parent / "unrelated-import-root"))
    monkeypatch.setenv("PYTEST_ADDOPTS", "--unrecognized-ambient-pytest-flag")
    observed = driver.execute(datasets, cfg["jobs"]["LAC-S11-044"])
    assert observed["returncode"] == 0
    assert observed["environment_policy"]["PYTHONPATH"] == str(datasets.resolve())
    assert observed["environment_policy"]["PYTEST_ADDOPTS"] == ""
    assert observed["environment_policy"]["IPFS_AUTO_INSTALL"] == "false"
    assert observed["environment_policy"]["IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS"] == "0"
    assert observed["environment_policy"]["PYTHONNOUSERSITE"] == "1"


def test_resealed_receipt_cannot_hide_different_execution_environment(frozen):
    _, sources, _ = frozen
    driver.materialize("LAC-S11-044", sources)
    result = driver.read_json(driver.output("LAC-S11-044"))
    result["test_execution"]["environment_policy"]["PYTHONPATH"] = "/unrelated/source"
    result["test_execution_sha256"] = driver.sha(driver.wire(result["test_execution"]))
    with pytest.raises(ValueError, match="environment policy differs"):
        driver.validate(result, "LAC-S11-044", sources)


def test_actual_failure_case_population_and_logs_are_bounded(frozen):
    datasets, _, cfg = frozen
    (datasets / "test_owner.py").write_text(
        "import pytest\n@pytest.mark.parametrize('index', range(7))\n"
        "def test_bounded_named_cause(index):\n"
        "    print('diagnostic-stream-' * 400)\n"
        "    assert False, 'retained cause ' + 'é' * 800\n", encoding="utf-8")
    with pytest.raises(driver.QualificationFailure) as caught:
        driver.execute(datasets, cfg["jobs"]["LAC-S11-044"])
    diagnostic = caught.value.diagnostics
    assert diagnostic["failed_case_count"] == diagnostic["test_count"] == 7
    assert len(diagnostic["failed_cases"]) == driver.MAX_FAILURE_CASES == 5
    for case in diagnostic["failed_cases"]:
        for key in ("class", "name", "message"):
            assert len(case[key].encode("utf-8")) <= driver.MAX_FAILURE_FIELD_BYTES
    for key in ("stdout_tail", "stderr_tail"):
        assert len(diagnostic[key].encode("utf-8")) <= driver.MAX_FAILURE_TAIL_BYTES


def test_skipped_test_is_refused_with_its_named_cause(frozen):
    datasets, _, cfg = frozen
    (datasets / "test_owner.py").write_text(
        "import pytest\ndef test_missing_required_fixture():\n"
        "    pytest.skip('required fixture unavailable')\n")
    with pytest.raises(driver.QualificationFailure) as caught:
        driver.execute(datasets, cfg["jobs"]["LAC-S11-044"])
    case = caught.value.diagnostics["failed_cases"][0]
    assert case["name"] == "test_missing_required_fixture"
    assert case["status"] == "skipped"
    assert "required fixture unavailable" in case["message"]


def test_current_branch_and_prerequisite_cannot_be_relabelled_complete(frozen):
    datasets, sources, cfg = frozen
    cfg["datasets"]["expected_branch"] = "wrong-source-branch"
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="branch differs"):
        driver.bindings("LAC-S11-044", sources)
    cfg["datasets"]["expected_branch"] = driver.git(datasets, "branch", "--show-current")
    prior = Path(cfg["prior_span_reconstruction_completion"]["path"])
    prior.write_text(json.dumps({"status": "incomplete"}))
    cfg["prior_span_reconstruction_completion"] = driver.identity(prior)
    sources.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="prerequisite is not complete"):
        driver.bindings("LAC-S11-044", sources)


def test_changed_training_audit_or_original_plan_blocks_qualification(frozen, tmp_path):
    _, sources, cfg = frozen
    audit = tmp_path / "audit.json"
    audit.write_text('{"scope":"authored diagnostics"}')
    plan = tmp_path / "plan.json"
    plan.write_text('{"original_acceptance":"partial"}')
    cfg["external_evidence"] = [driver.identity(audit)]
    cfg["original_plan"] = driver.identity(plan)
    sources.write_text(json.dumps(cfg))
    driver.materialize("LAC-S11-044", sources)
    audit.write_text('{"scope":"changed"}')
    with pytest.raises(ValueError, match="External evidence moved"):
        driver.validate(driver.read_json(driver.output("LAC-S11-044")), "LAC-S11-044", sources)
    cfg["external_evidence"] = [driver.identity(audit)]
    sources.write_text(json.dumps(cfg))
    plan.write_text('{"original_acceptance":"changed"}')
    with pytest.raises(ValueError, match="Original plan moved"):
        driver.bindings("LAC-S11-044", sources)


def test_resealed_real_input_report_cannot_replace_independent_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    expected = {"rows": 3, "coverage_scope": "fictional-test-population"}
    baseline.write_text(json.dumps(expected))
    job = {"real_input": {"spec": {"kind": "entity_index_census"},
        "baseline": {**driver.identity(baseline), "canonical_json_sha256": driver.sha(driver.wire(expected))}}}
    observed = {"kind": "entity_index_census", "report": expected,
                "report_sha256": driver.sha(driver.wire(expected))}
    driver.validate_real_input(observed, job)
    altered = copy.deepcopy(observed)
    altered["report"]["rows"] = 2
    altered["report_sha256"] = driver.sha(driver.wire(altered["report"]))
    with pytest.raises(ValueError, match="independently pinned baseline"):
        driver.validate_real_input(altered, job)
    with pytest.raises(ValueError, match="Unexpected real-input"):
        driver.validate_real_input(observed, {})
    baseline.write_text(json.dumps(altered["report"]))
    with pytest.raises(ValueError, match="baseline moved"):
        driver.validate_real_input(altered, job)


def test_complete_view_baseline_above_old_cap_is_pinned_and_new_cap_is_finite(tmp_path):
    baseline = tmp_path / "complete-view-baseline.json"
    expected = {"retained_observations": "x" * (4 * 1024 * 1024 + 128)}
    baseline.write_bytes(driver.wire(expected))
    job = {"real_input":{"spec":{"kind":"source_view_inventory"},"baseline":{
        **driver.identity(baseline),"canonical_json_sha256":driver.sha(driver.wire(expected))}}}
    assert baseline.stat().st_size > 4 * 1024 * 1024
    assert driver.expected_real_input(job) == expected
    observed = {"kind":"source_view_inventory","report":driver.pack_real_input_report(expected),
                "report_sha256":driver.sha(driver.wire(expected))}
    driver.validate_real_input(observed, job)
    with baseline.open("wb") as stream:
        stream.truncate(driver.MAX_BYTES + 1)
    job["real_input"]["baseline"].update(driver.identity(baseline))
    with pytest.raises(ValueError, match="exceeds bound"):
        driver.expected_real_input(job)
    with pytest.raises(ValueError, match="read bound"):
        driver.read_json(baseline)


def test_lossless_real_report_envelope_obeys_native_file_gate_and_external_baseline(tmp_path):
    report = {"complete_population":[{"raw":"retained " * 128,"ordinal":n} for n in range(4096)]}
    packed = driver.pack_real_input_report(report)
    assert packed == driver.pack_real_input_report(report)
    assert len(driver.wire(packed)) < driver.MAX_RESULT_BYTES < 1_048_576
    assert driver.unpack_real_input_report(packed) == report
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(driver.wire(report))
    job = {"real_input":{"spec":{"kind":"source_view_inventory"},"baseline":{
        **driver.identity(baseline),"canonical_json_sha256":driver.sha(driver.wire(report))}}}
    observed = {"kind":"source_view_inventory","report":packed,"report_sha256":driver.sha(driver.wire(report))}
    driver.validate_real_input(observed, job)
    report["complete_population"].pop()
    altered = {"kind":"source_view_inventory","report":driver.pack_real_input_report(report),
               "report_sha256":driver.sha(driver.wire(report))}
    with pytest.raises(ValueError, match="independently pinned baseline"):
        driver.validate_real_input(altered, job)


def test_resealed_unrelated_envelope_is_refused_before_decode_or_object_parse(tmp_path, monkeypatch):
    expected = {"physical_rows":20}
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(driver.wire(expected))
    job = {"real_input":{"spec":{"kind":"producer_lineage"},"baseline":{
        **driver.identity(baseline),"canonical_json_sha256":driver.sha(driver.wire(expected))}}}
    observed = {"kind":"producer_lineage","report":driver.pack_real_input_report({"invented":True}),
                "report_sha256":driver.sha(driver.wire(expected))}
    def deny_decode(value):
        pytest.fail("independent full identity must be checked before decoding an unrelated payload")
    monkeypatch.setattr(driver,"unpack_real_input_report",deny_decode)
    with pytest.raises(ValueError,match="independently pinned baseline before decode"):
        driver.validate_real_input(observed,job)


@pytest.mark.parametrize("damage",["blob","compressed_sha","raw_sha","raw_count","expand_bomb",
    "trailing","concatenated","noncanonical","oversized_raw","oversized_encoded","extra"])
def test_envelope_tampering_and_resealed_stream_boundaries_are_refused(damage):
    packed = driver.pack_real_input_report({"retained":"x" * 4096})
    if damage == "blob": packed["data"] = "!" + packed["data"][1:]
    elif damage == "compressed_sha": packed["compressed_sha256"] = "0" * 64
    elif damage == "raw_sha": packed["report_sha256"] = "0" * 64
    elif damage == "raw_count": packed["report_bytes"] += 1
    elif damage == "expand_bomb": packed["report_bytes"] = 2
    elif damage == "oversized_raw": packed["report_bytes"] = driver.MAX_BYTES + 1
    elif damage == "oversized_encoded": packed["encoded_bytes"] = driver.MAX_ENCODED_REPORT_BYTES + 1
    elif damage == "extra": packed["authority"] = True
    else:
        compressed = base64.b64decode(packed["data"])
        if damage == "trailing": compressed += b"trailing"
        elif damage == "concatenated": compressed += zlib.compress(b"{}")
        else:
            raw = b'{"retained": "noncanonical whitespace"}'
            compressed = zlib.compress(raw)
            packed.update(report_bytes=len(raw), report_sha256=driver.sha(raw))
        encoded = base64.b64encode(compressed).decode("ascii")
        packed.update(data=encoded, encoded_bytes=len(encoded),compressed_bytes=len(compressed),
                      compressed_sha256=driver.sha(compressed))
    with pytest.raises((ValueError,zlib.error)):
        driver.unpack_real_input_report(packed)


def test_real_input_executes_in_pinned_root_and_refuses_changed_outcome(frozen):
    datasets, _, cfg = frozen
    package = datasets / "ipfs_datasets_py/logic/autoformal"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text("")
    owner = package / "legal_entity_index_intake.py"
    owner.write_text("def census_parquet(path, **kwargs):\n    return {'fixture_rows': 3}\n")
    raw = datasets / "fictional-index.bin"
    raw.write_bytes(b"fictional fixture, not parquet")
    job = dict(cfg["jobs"]["LAC-S11-044"])
    job["real_input"] = {"spec": {"kind": "entity_index_census", "path": str(raw),
        "file_sha256": driver.identity(raw)["sha256"], "options": {}}}
    expected = driver.run_real_input(datasets, job)
    assert expected == {"fixture_rows": 3}
    baseline = datasets / "baseline.json"
    baseline.write_text(json.dumps(expected))
    job["real_input"]["baseline"] = {**driver.identity(baseline),
        "canonical_json_sha256": driver.sha(driver.wire(expected))}
    assert driver.execute(datasets, job)["real_input_execution"]["report"] == expected
    owner.write_text("def census_parquet(path, **kwargs):\n    return {'fixture_rows': 2}\n")
    with pytest.raises(ValueError, match="independently pinned baseline"):
        driver.execute(datasets, job)


@pytest.mark.parametrize("kind,cap", [("entity_index_census", 16), ("span_cache_sample", 262144)])
def test_real_input_rejects_oversized_file_before_loading_backend(frozen, kind, cap):
    datasets, _, _ = frozen
    raw = datasets / "oversized.bin"
    raw.write_bytes(b"x" * (cap + 1))
    job = {"real_input": {"spec": {"kind": kind, "path": str(raw),
        "file_sha256": driver.identity(raw)["sha256"], "options": {"max_file_bytes": cap}}}}
    with pytest.raises(ValueError, match="initial file byte bound"):
        driver.run_real_input(datasets, job)


def test_real_input_hash_guard_survives_python_optimization(frozen):
    datasets, _, _ = frozen
    raw = datasets / "guarded.bin"
    raw.write_bytes(b"guarded fixture")
    job = {"real_input": {"spec": {"kind": "entity_index_census", "path": str(raw),
        "file_sha256": "0" * 64, "options": {"max_file_bytes": 1024}}}}
    cmd = driver.real_input_command(job)
    run = subprocess.run([cmd[0], "-OO", *cmd[1:]], cwd=datasets,
                         text=True, capture_output=True, timeout=30)
    assert run.returncode != 0 and "Real-input file hash differs" in run.stderr
    assert driver.execution_environment_policy(datasets)["PYTHONOPTIMIZE"] == "0"


@pytest.mark.parametrize("kind", ["primary_graph_files", "primary_graph_reconstruction"])
def test_primary_manifest_is_bounded_and_hash_checked_before_backend_import(frozen, kind):
    datasets, _, _ = frozen
    raw = datasets / "manifest.json"
    raw.write_bytes(b"x" * (1048576 + 1))
    job = {"real_input": {"spec": {"kind": kind, "path": str(raw),
        "file_sha256": driver.identity(raw)["sha256"]}}}
    with pytest.raises(ValueError, match="initial file byte bound"):
        driver.run_real_input(datasets, job)
    raw.write_text("{}")
    job["real_input"]["spec"]["file_sha256"] = "0" * 64
    cmd = driver.real_input_command(job)
    run = subprocess.run([cmd[0], "-OO", *cmd[1:]], cwd=datasets,
                         text=True, capture_output=True, timeout=30)
    assert run.returncode != 0 and "Real-input file hash differs" in run.stderr


@pytest.mark.parametrize("kind", ["primary_graph_files", "primary_graph_reconstruction"])
def test_primary_real_input_executes_selected_backend_with_frozen_pins(frozen, kind):
    datasets, _, _ = frozen
    package = datasets / "ipfs_datasets_py/logic/autoformal"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text("")
    (package / "legal_primary_graph_manifest.py").write_text(
        "from pathlib import Path\n"
        "def deny_unbounded_read(*args, **kwargs):\n"
        "    raise RuntimeError('unbounded path read after backend import')\n"
        "Path.read_bytes = deny_unbounded_read\n"
        "def select_primary_graph_manifest(raw, **kwargs):\n"
        "    if raw != b'{}' or kwargs != {'fixture': 'independently-frozen'}:\n"
        "        raise ValueError('selection binding changed')\n"
        "    return {'selection': 'exact-fixture'}\n"
        "def verify_primary_graph_files(selection, snapshot_root, **kwargs):\n"
        "    if selection != {'selection': 'exact-fixture'} or snapshot_root != 'fixture-snapshot':\n"
        "        raise ValueError('selected population changed')\n"
        "    if kwargs['expected_selection_sha256'] != 'frozen-selection-pin':\n"
        "        raise ValueError('independent selection pin changed')\n"
        "    return {'fixture_files': 3, 'legal_semantic_acceptance': False}\n")
    (package / "legal_primary_graph_reconstruction.py").write_text(
        "from .legal_primary_graph_manifest import verify_primary_graph_files\n"
        "def reconstruct_to_jsonl(selection, snapshot_root, output_path, **kwargs):\n"
        "    result = verify_primary_graph_files(selection, snapshot_root, **kwargs)\n"
        "    if output_path.exists(): raise ValueError('candidate must be new')\n"
        "    output_path.write_bytes(b'fictional-candidate\\n')\n"
        "    return dict(result, fixture_candidate_bytes=20)\n")
    raw = datasets / "manifest.json"
    raw.write_text("{}")
    spec = {"kind": kind, "path": str(raw), "file_sha256": driver.identity(raw)["sha256"],
            "selection_options": {"fixture": "independently-frozen"},
            "expected_selection_sha256": "frozen-selection-pin",
            "snapshot_root": "fixture-snapshot", "blobs_root": "fixture-blobs"}
    job = {"real_input": {"spec": spec}}
    expected = driver.run_real_input(datasets, job)
    assert expected["fixture_files"] == 3 and expected["legal_semantic_acceptance"] is False
    if kind == "primary_graph_reconstruction":
        assert expected["fixture_candidate_bytes"] == 20
    spec["expected_selection_sha256"] = "changed-selection-pin"
    with pytest.raises(ValueError, match="independent selection pin changed"):
        driver.run_real_input(datasets, job)


@pytest.mark.parametrize("kind", ["section_index_census", "span_section_bridge", "source_graph_proposals"])
def test_section_candidate_guard_refuses_oversize_before_import(frozen, kind):
    datasets, _, _ = frozen
    candidate = datasets / "candidate.jsonl"
    candidate.write_bytes(b"x" * 17)
    job = {"real_input": {"spec": {"kind": kind, "path": str(candidate),
        "file_sha256": driver.identity(candidate)["sha256"], "max_file_bytes": 16}}}
    with pytest.raises(ValueError, match="initial file byte bound"):
        driver.run_real_input(datasets, job)


@pytest.mark.parametrize("damage", ["size", "hash", "alias"])
def test_section_reconstruction_report_is_guarded_before_backend_import(frozen, damage):
    datasets, _, _ = frozen
    candidate = datasets / "candidate.jsonl"
    candidate.write_bytes(b"guarded fixture candidate")
    report = datasets / "report.json"
    report.write_text("{}")
    pin = driver.identity(report)
    if damage == "size":
        report.write_bytes(b"x" * (1048576 + 1))
        pin = driver.identity(report)
        expected = "Auxiliary initial file byte bound"
    elif damage == "hash":
        pin["sha256"] = "0" * 64
        expected = "Auxiliary input hash differs"
    else:
        alias = datasets / "aliased-report.json"
        alias.symlink_to(report)
        pin["path"] = str(alias)
        expected = "Auxiliary input alias refused"
    job = {"real_input": {"spec": {"kind": "section_index_census", "path": str(candidate),
        "file_sha256": driver.identity(candidate)["sha256"], "max_file_bytes": 1024,
        "reconstruction_report": pin}}}
    with pytest.raises(ValueError, match=expected):
        driver.run_real_input(datasets, job)


def test_section_real_input_executes_bound_index_and_checks_independent_pin(frozen):
    datasets, _, _ = frozen
    package = datasets / "ipfs_datasets_py/logic/autoformal"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text("")
    (package / "legal_primary_section_index.py").write_text(
        "class PrimarySectionIndex:\n"
        "    def __init__(self, path, **kwargs):\n"
        "        if kwargs['reconstruction_report'] != {'fixture_rows': 3}:\n"
        "            raise ValueError('bound report changed')\n"
        "        if kwargs['expected_reconstruction_sha256'] != 'frozen-report-pin':\n"
        "            raise ValueError('report pin changed')\n"
        "        self.index_sha256 = 'frozen-index-pin'\n"
        "        self.receipt = {'fixture_rows': 3, 'source_semantics_verified': False}\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *args): pass\n")
    candidate = datasets / "candidate.jsonl"
    candidate.write_bytes(b"guarded candidate fixture")
    report = datasets / "report.json"
    report.write_text('{"fixture_rows":3}')
    spec = {"kind": "section_index_census", "path": str(candidate),
        "file_sha256": driver.identity(candidate)["sha256"], "max_file_bytes": 1024,
        "reconstruction_report": driver.identity(report),
        "expected_reconstruction_sha256": "frozen-report-pin", "expected_index_sha256": "frozen-index-pin"}
    job = {"real_input": {"spec": spec}}
    assert driver.run_real_input(datasets, job) == {"fixture_rows": 3, "source_semantics_verified": False}
    spec["expected_index_sha256"] = "wrong-index-pin"
    with pytest.raises(ValueError, match="Independent section index pin differs"):
        driver.run_real_input(datasets, job)


@pytest.mark.parametrize("kind", driver.SOURCE_VIEWS_KINDS)
@pytest.mark.parametrize("damage", ["size", "hash", "alias"])
def test_custody_master_is_guarded_before_backend_import(frozen, kind, damage):
    datasets, _, _ = frozen
    master = datasets / "custody-master.json"
    master.write_text("{}")
    pin = driver.identity(master)
    if damage == "size":
        with master.open("wb") as stream:
            stream.truncate(1048577)
        pin = driver.identity(master)
        expected = "View initial file bound"
    elif damage == "hash":
        pin["sha256"] = "0" * 64
        expected = "Independent view file hash differs"
    else:
        alias = datasets / "alias.json"
        alias.symlink_to(master)
        pin["path"] = str(alias)
        expected = "View input path alias"
    job = {"real_input": {"spec": {"kind": kind, "path": pin["path"],
        "file_bytes": pin["bytes"], "file_sha256": pin["sha256"]}}}
    with pytest.raises(ValueError, match=expected):
        driver.run_real_input(datasets, job)


def custody_fixture(datasets):
    package = datasets / "ipfs_datasets_py/logic/autoformal"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text("")
    (package / "legal_span_provenance.py").write_text("")
    (package / "legal_source_view_inventory.py").write_text(
        "def view_document_population_sha256(inputs): return 'independent-population'\n")
    (package / "legal_official_document_views.py").write_text(
        "def validate_official_document_view(report, raw, **kwargs):\n"
        "    if raw != b'fictional HTML': raise ValueError('bound content changed')\n"
        "    if kwargs['expected_descriptor_sha256'] != 'independent-descriptor':\n"
        "        raise ValueError('bound descriptor changed')\n"
        "    if report != {'fixture': 'complete custody', 'formalized': False}:\n"
        "        raise ValueError('bound full report changed')\n"
        "    return report\n")
    raw = datasets / "official.html"
    raw.write_bytes(b"fictional HTML")
    report = datasets / "view-report.json"
    report.write_bytes(driver.wire({"fixture":"complete custody","formalized":False}))
    body = {"schema": "legal-source-views-population-inputs/v1", "span_files": [],
        "expected_document_population_sha256": "independent-population",
        "official_documents": [{"raw_path": str(raw), "descriptor": {},
            "expected_raw_bytes": raw.stat().st_size, "expected_raw_sha256": driver.identity(raw)["sha256"],
            "expected_descriptor_sha256": "independent-descriptor",
            "view_report_path":str(report),"view_report_bytes":report.stat().st_size,
            "view_report_sha256":driver.identity(report)["sha256"],
            "expected_view_report_sha256":driver.sha(driver.wire(json.loads(report.read_text())))}]}
    return raw, body


@pytest.mark.parametrize("damage", ["size", "hash", "alias"])
def test_custody_raw_capture_guards_precede_document_backend(frozen, damage):
    datasets, _, _ = frozen
    raw, body = custody_fixture(datasets)
    source = body["official_documents"][0]
    if damage == "size":
        with raw.open("wb") as stream:
            stream.truncate(8000001)
        source.update(expected_raw_bytes=raw.stat().st_size,
                      expected_raw_sha256=driver.identity(raw)["sha256"])
        expected = "View initial file bound"
    elif damage == "hash":
        source["expected_raw_sha256"] = "0" * 64
        expected = "Independent view file hash differs"
    else:
        alias = datasets / "aliased-official.html"
        alias.symlink_to(raw)
        source["raw_path"] = str(alias)
        expected = "View input path alias"
    master = datasets / "master.json"
    master.write_bytes(driver.wire(body))
    pin = driver.identity(master)
    job = {"real_input": {"spec": {"kind": "document_views", "path": pin["path"],
        "file_bytes": pin["bytes"], "file_sha256": pin["sha256"]}}}
    with pytest.raises(ValueError, match=expected):
        driver.run_real_input(datasets, job)


def test_custody_profile_executes_captured_bytes_and_population_pin(frozen):
    datasets, _, _ = frozen
    _, body = custody_fixture(datasets)
    master = datasets / "master.json"
    master.write_bytes(driver.wire(body))
    pin = driver.identity(master)
    job = {"real_input": {"spec": {"kind": "document_views", "path": pin["path"],
        "file_bytes": pin["bytes"], "file_sha256": pin["sha256"]}}}
    result = driver.run_real_input(datasets, job)
    assert result["documents"] == [{"fixture": "complete custody", "formalized": False}]
    body["expected_document_population_sha256"] = "resealed-population"
    master.write_bytes(driver.wire(body))
    pin = driver.identity(master)
    job["real_input"]["spec"].update(file_bytes=pin["bytes"], file_sha256=pin["sha256"])
    with pytest.raises(ValueError, match="Independent document population pin differs"):
        driver.run_real_input(datasets, job)


def test_native_supervisor_source_pin_uses_content_identity_across_owned_checkouts(frozen):
    _, sources, cfg = frozen
    controller = sources.parent
    native = controller / "native.py"
    native.write_text("PROTECTION = 'foreign lock custody'\n")
    for args in (["init", "-q"], ["add", "native.py"],
            ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "commit", "-qm", "Freeze native fixture"]):
        subprocess.run(["git", *args], cwd=controller, check=True, capture_output=True)
    cfg["immutable_inputs"].append({"repository": "accelerator", "path": "native.py",
                                   "sha256": driver.identity(native)["sha256"]})
    sources.write_text(json.dumps(cfg))
    first = driver.materialize("LAC-S11-044", sources)
    assert first["valid"] is True
    result = json.loads(driver.output("LAC-S11-044").read_text())
    assert result["bindings"]["accelerator_inputs"] == [{"repository_path": "native.py",
        "sha256": driver.identity(native)["sha256"], "bytes": native.stat().st_size}]
    native.write_text("PROTECTION = 'changed native code'\n")
    with pytest.raises(ValueError, match="Native supervisor source pin differs"):
        driver.materialize("LAC-S11-044", sources)


@pytest.mark.parametrize("damage", ["size", "hash", "alias"])
def test_versioned_view_report_capture_requires_independent_raw_file_pin(frozen, damage):
    datasets, _, _ = frozen
    _, body = custody_fixture(datasets)
    item = body["official_documents"][0]
    report = Path(item["view_report_path"])
    if damage == "size":
        with report.open("wb") as stream:
            stream.truncate(33554433)
        item.update(view_report_bytes=report.stat().st_size,
                    view_report_sha256=driver.identity(report)["sha256"])
        expected = "View initial file bound"
    elif damage == "hash":
        item["view_report_sha256"] = "0" * 64
        expected = "Independent view file hash differs"
    else:
        alias = datasets / "view-alias.json"
        alias.symlink_to(report)
        item["view_report_path"] = str(alias)
        expected = "View input path alias"
    master = datasets / "master.json"
    master.write_bytes(driver.wire(body))
    pin = driver.identity(master)
    job = {"real_input": {"spec": {"kind": "document_views", "path": pin["path"],
        "file_bytes": pin["bytes"], "file_sha256": pin["sha256"]}}}
    with pytest.raises(ValueError, match=expected):
        driver.run_real_input(datasets, job)


def test_producer_lineage_receives_regenerated_complete_ordered_provenance(frozen):
    import pyarrow as pa
    import pyarrow.parquet as pq
    datasets, _, _ = frozen
    package = datasets / "ipfs_datasets_py/logic/autoformal"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text("")
    (package / "legal_span_provenance.py").write_text(
        "import hashlib,json\n"
        "def reconcile_span_provenance(rows, **kwargs):\n"
        "    return {'rows':rows,'record_pins':kwargs['expected_record_sha256s']}\n"
        "def provenance_report_sha256(report):\n"
        "    return hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()\n")
    (package / "legal_producer_lineage.py").write_text(
        "import hashlib,json\n"
        "def reconcile_producer_lineage(reports, **kwargs):\n"
        "    pins=[p for report in reports for p in report['record_pins']]\n"
        "    if pins != kwargs['expected_record_sha256s'] or pins != ['a','b','c']:\n"
        "        raise ValueError('complete ordered record population changed')\n"
        "    if kwargs['witness_manifest'] != {'fixture':'closed witnesses'}:\n"
        "        raise ValueError('witness population changed')\n"
        "    return {'record_pins':pins,'formalized':False}\n"
        "def lineage_report_sha256(report):\n"
        "    return hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()\n")
    files = []
    for number, records in enumerate(([{"text":"first"},{"text":"second"}], [{"text":"third"}])):
        span = datasets / (str(number) + ".parquet")
        pq.write_table(pa.Table.from_pylist(records), span)
        pins = ["a", "b"] if number == 0 else ["c"]
        provenance_report = {"rows": records, "record_pins": pins}
        files.append({"path":str(span),"bytes":span.stat().st_size,
            "file_sha256":driver.identity(span)["sha256"],"expected_rows":len(records),
            "cache_identity":{"fixture":number},"expected_row_sha256s":[],
            "expected_record_sha256s":pins,
            "expected_provenance_report_sha256":driver.sha(driver.wire(provenance_report))})
    witness = datasets / "witness.json"
    witness.write_text('{"fixture":"closed witnesses"}')
    expected = {"record_pins":["a","b","c"],"formalized":False}
    body = {"schema":"legal-source-views-population-inputs/v1","span_files":files,
        "demonstration_catalog":{},"expected_catalog_sha256":"fixture-catalog",
        "expected_catalog_source_pins":[],"producer_witness_manifest":driver.identity(witness),
        "expected_producer_witness_manifest_sha256":driver.sha(driver.wire(json.loads(witness.read_text()))),
        "expected_producer_lineage_sha256":driver.sha(driver.wire(expected))}
    master = datasets / "master.json"
    def job():
        master.write_bytes(driver.wire(body))
        pin = driver.identity(master)
        return {"real_input":{"spec":{"kind":"producer_lineage","path":pin["path"],
            "file_bytes":pin["bytes"],"file_sha256":pin["sha256"]}}}
    result = driver.run_real_input(datasets, job())
    assert result == {"schema":"legal-source-views-real-input/v1","kind":"producer_lineage","report":expected}
    body["span_files"].reverse()
    with pytest.raises(ValueError, match="complete ordered record population changed"):
        driver.run_real_input(datasets, job())
    body["span_files"].reverse()
    body["span_files"][0]["expected_provenance_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Independent provenance report pin differs"):
        driver.run_real_input(datasets, job())


@pytest.mark.parametrize("kind", driver.PARENT_PAGE_KINDS)
@pytest.mark.parametrize("damage", ["size", "hash", "alias"])
def test_parent_page_master_is_bounded_before_backend_import(frozen, kind, damage):
    datasets, _, _ = frozen
    master = datasets / "parent-master.json"
    master.write_text("{}")
    pin = driver.identity(master)
    if damage == "size":
        with master.open("wb") as stream:
            stream.truncate(1048577)
        pin = driver.identity(master)
        expected = "Parent initial file bound"
    elif damage == "hash":
        pin["sha256"] = "0" * 64
        expected = "Independent parent input hash differs"
    else:
        alias = datasets / "parent-master-alias.json"
        alias.symlink_to(master)
        pin["path"] = str(alias)
        expected = "Parent input path alias"
    job = {"real_input":{"spec":{"kind":kind,"path":pin["path"],
        "file_bytes":pin["bytes"],"file_sha256":pin["sha256"]}}}
    with pytest.raises(ValueError,match=expected):
        driver.run_real_input(datasets,job)


def parent_page_fixture(datasets):
    package = datasets / "ipfs_datasets_py/logic/autoformal"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text("")
    (package / "legal_parent_parquet_pages.py").write_text(
        "import hashlib,json\n"
        "def digest(report):\n"
        "    return hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()\n"
        "def preflight_source_alignment(descriptor, **kwargs):\n"
        "    if descriptor != {'fixture':'independently pinned parent'} or kwargs['expected_descriptor_sha256'] != 'frozen-parent-descriptor':\n"
        "        raise ValueError('parent descriptor binding differs')\n"
        "    return {'fixture_pages':3,'authority':False}\n"
        "parent_page_report_sha256=digest\n"
        "def read_selected_parent_rows(descriptor, **kwargs):\n"
        "    report=preflight_source_alignment(descriptor,**kwargs)\n"
        "    if kwargs['preflight_report'] != report or kwargs['expected_preflight_sha256'] != digest(report):\n"
        "        raise ValueError('parent preflight binding differs')\n"
        "    if kwargs['row_ordinals'] != [3,1] or kwargs['expected_row_sha256s'] != ['row3','row1']:\n"
        "        raise ValueError('complete ordered selected parent population differs')\n"
        "    return {'physical_ordinals':[3,1],'original_rows':['complete3','complete1'],'authority':False}\n"
        "parent_row_report_sha256=digest\n")
    preflight = {"fixture_pages":3,"authority":False}
    selected = {"physical_ordinals":[3,1],"original_rows":["complete3","complete1"],"authority":False}
    path = datasets / "preflight-report.json"
    path.write_bytes(driver.wire(preflight))
    body = {"schema":"legal-span-reconstruction-population-inputs/v1",
        "parent_descriptor":{"fixture":"independently pinned parent"},
        "expected_parent_descriptor_sha256":"frozen-parent-descriptor",
        "preflight_report_file":driver.identity(path),"expected_preflight_sha256":driver.sha(driver.wire(preflight)),
        "row_ordinals":[3,1],"expected_row_sha256s":["row3","row1"],
        "expected_selected_rows_sha256":driver.sha(driver.wire(selected))}
    return body,preflight,selected


@pytest.mark.parametrize("kind",["parent_page_preflight","parent_selected_rows"])
def test_parent_profiles_execute_exact_frozen_api_and_lossless_wrapper(frozen,kind):
    datasets,_,_=frozen
    body,preflight,selected=parent_page_fixture(datasets)
    master=datasets/"master.json"
    def job():
        master.write_bytes(driver.wire(body));pin=driver.identity(master)
        return {"real_input":{"spec":{"kind":kind,"path":pin["path"],"file_bytes":pin["bytes"],"file_sha256":pin["sha256"]}}}
    expected=preflight if kind=="parent_page_preflight" else selected
    result=driver.run_real_input(datasets,job())
    assert result=={"schema":"legal-span-reconstruction-real-input/v1","kind":kind,"report":expected}
    assert driver.unpack_real_input_report(driver.pack_real_input_report(result))==result
    body["expected_preflight_sha256"]="0"*64
    with pytest.raises(ValueError,match="Independent FULL parent page preflight pin differs"):
        driver.run_real_input(datasets,job())


@pytest.mark.parametrize("damage",["file_hash","full_pin","row_order","row_pin"])
def test_selected_parent_profile_requires_exact_preflight_and_complete_rows(frozen,damage):
    datasets,_,_=frozen
    body,_,_=parent_page_fixture(datasets)
    if damage=="file_hash":body["preflight_report_file"]["sha256"]="0"*64
    elif damage=="full_pin":body["expected_preflight_sha256"]="0"*64
    elif damage=="row_order":body["row_ordinals"].reverse()
    else:body["expected_selected_rows_sha256"]="0"*64
    master=datasets/"master.json";master.write_bytes(driver.wire(body));pin=driver.identity(master)
    job={"real_input":{"spec":{"kind":"parent_selected_rows","path":pin["path"],"file_bytes":pin["bytes"],"file_sha256":pin["sha256"]}}}
    with pytest.raises(ValueError):driver.run_real_input(datasets,job)


@pytest.mark.parametrize("damage",[None,"provenance_pin","record_order","row_file","row_report",
                                  "page_population_pin","lineage_pin"])
def test_page_lineage_requires_complete_source_rows_and_fresh_parent_regeneration(frozen,damage):
    import pyarrow as pa
    import pyarrow.parquet as pq
    datasets,_,_=frozen
    body,preflight,selected=parent_page_fixture(datasets)
    package=datasets/"ipfs_datasets_py/logic/autoformal"
    (package/"legal_span_provenance.py").write_text(
        "import hashlib,json\n"
        "def reconcile_span_provenance(rows, **kwargs):\n"
        "    return {'rows':rows,'record_pins':kwargs['expected_record_sha256s']}\n"
        "def provenance_report_sha256(report):\n"
        "    return hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()\n")
    (package/"legal_producer_page_lineage.py").write_text(
        "import hashlib,json\n"
        "def reconcile_producer_page_lineage(reports, **kwargs):\n"
        "    pins=[p for report in reports for p in report['record_pins']]\n"
        "    if pins != kwargs['expected_record_sha256s'] or pins != ['a','b','c']:\n"
        "        raise ValueError('complete ordered cache record population changed')\n"
        "    if len(kwargs['parent_page_inputs'])!=1 or kwargs['parent_page_inputs'][0]['row_report']['physical_ordinals']!=[3,1]:\n"
        "        raise ValueError('complete qualified parent population changed')\n"
        "    return {'record_pins':pins,'parent_ordinals':[3,1],'formalized':False}\n"
        "def producer_page_lineage_report_sha256(report):\n"
        "    return hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()\n")
    files=[]
    for number,records in enumerate(([{"text":"first"},{"text":"second"}],[{"text":"third"}])):
        span=datasets/(str(number)+".parquet");pq.write_table(pa.Table.from_pylist(records),span)
        pins=["a","b"] if number==0 else ["c"]
        files.append({"path":str(span),"bytes":span.stat().st_size,"file_sha256":driver.identity(span)["sha256"],
            "expected_rows":len(records),"cache_identity":{"fixture":number},"expected_row_sha256s":[],
            "expected_record_sha256s":pins,"expected_provenance_report_sha256":driver.sha(driver.wire({"rows":records,"record_pins":pins}))})
    selected_path=datasets/"selected-rows.json";selected_path.write_bytes(driver.wire(selected))
    witness=datasets/"witness.json";witness.write_text('{"fixture":"closed witnesses"}')
    parent_inputs=[{"descriptor_sha256":body["expected_parent_descriptor_sha256"],"preflight_report":preflight,
        "expected_preflight_sha256":body["expected_preflight_sha256"],"row_report":selected,
        "expected_row_report_sha256":body["expected_selected_rows_sha256"]}]
    expected={"record_pins":["a","b","c"],"parent_ordinals":[3,1],"formalized":False}
    body.update(span_files=files,demonstration_catalog={},expected_catalog_sha256="fixturecatalog",
        expected_catalog_source_pins=[],selected_rows_report_file=driver.identity(selected_path),
        expected_parent_page_inputs_sha256=driver.sha(driver.wire(parent_inputs)),
        producer_witness_manifest=driver.identity(witness),
        expected_producer_witness_manifest_sha256=driver.sha(driver.wire(json.loads(witness.read_text()))),
        expected_producer_page_lineage_sha256=driver.sha(driver.wire(expected)))
    if damage=="provenance_pin":body["span_files"][0]["expected_provenance_report_sha256"]="0"*64
    elif damage=="record_order":body["span_files"].reverse()
    elif damage=="row_file":body["selected_rows_report_file"]["sha256"]="0"*64
    elif damage=="row_report":
        retained=copy.deepcopy(selected);retained["original_rows"][0]="invented"
        selected_path.write_bytes(driver.wire(retained));body["selected_rows_report_file"]=driver.identity(selected_path)
    elif damage=="page_population_pin":body["expected_parent_page_inputs_sha256"]="0"*64
    elif damage=="lineage_pin":body["expected_producer_page_lineage_sha256"]="0"*64
    master=datasets/"master.json";master.write_bytes(driver.wire(body));pin=driver.identity(master)
    job={"real_input":{"spec":{"kind":"producer_page_lineage","path":pin["path"],"file_bytes":pin["bytes"],"file_sha256":pin["sha256"]}}}
    if damage is None:
        assert driver.run_real_input(datasets,job)=={"schema":"legal-span-reconstruction-real-input/v1","kind":"producer_page_lineage","report":expected}
    else:
        with pytest.raises(ValueError):driver.run_real_input(datasets,job)


@pytest.mark.parametrize("kind",driver.SOURCE_SPAN_KINDS)
@pytest.mark.parametrize("damage",["size","hash","alias"])
def test_source_alignment_master_is_guarded_before_any_owner_import(frozen,kind,damage):
    datasets,_,_=frozen;master=datasets/"span-master.json";master.write_text("{}")
    pin=driver.identity(master)
    if damage=="size":
        with master.open("wb") as stream:stream.truncate(1048577)
        pin=driver.identity(master);expected="Span reconstruction initial file bound"
    elif damage=="hash":pin["sha256"]="0"*64;expected="Independent span reconstruction input hash differs"
    else:
        alias=datasets/"master-alias.json";alias.symlink_to(master);pin["path"]=str(alias)
        expected="Span reconstruction input path alias"
    job={"real_input":{"spec":{"kind":kind,"path":pin["path"],"file_bytes":pin["bytes"],"file_sha256":pin["sha256"]}}}
    with pytest.raises(ValueError,match=expected):driver.run_real_input(datasets,job)


def source_alignment_fixture(datasets):
    package=datasets/"ipfs_datasets_py/logic/autoformal";package.mkdir(parents=True)
    for directory in (package,package.parent,package.parent.parent):(directory/"__init__.py").write_text("")
    (package/"legal_parent_text_views.py").write_text(
        "import hashlib,json\n"
        "def digest(value):\n"
        "    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()\n"
        "def prepare_parent_text_views(selected, **kwargs):\n"
        "    required={'expected_selected_rows_report_sha256','parent_descriptor','expected_parent_descriptor_sha256','preflight_report','expected_preflight_sha256','row_ordinals','expected_row_sha256s'}\n"
        "    if set(kwargs)!=required or kwargs['expected_selected_rows_report_sha256']!=digest(selected):\n"
        "        raise ValueError('complete exact parent-view inputs differ')\n"
        "    if kwargs['row_ordinals']!=[1,3] or kwargs['preflight_report']!={'fixture_pages':3}:\n"
        "        raise ValueError('parent request or page population differs')\n"
        "    return {'selected_rows_report':selected,'raw_texts':['First. Second.'],'authority':False}\n"
        "parent_text_views_report_sha256=digest\n")
    (package/"legal_parent_sentence_candidates.py").write_text(
        "from .legal_parent_text_views import digest,prepare_parent_text_views\n"
        "def prepare_parent_sentence_candidates(report, **kwargs):\n"
        "    if set(kwargs)!={'expected_parent_text_views_report_sha256','parent_text_views_inputs'} or kwargs['expected_parent_text_views_report_sha256']!=digest(report):\n"
        "        raise ValueError('sentence FULL parent pin or exact inputs differ')\n"
        "    if report!=prepare_parent_text_views(report['selected_rows_report'],**kwargs['parent_text_views_inputs']):\n"
        "        raise ValueError('sentence parent regeneration differs')\n"
        "    return {'raw_text':'First. Second.','complete_characters':14,'formalized':False}\n"
        "parent_sentence_candidates_report_sha256=digest\n")
    selected={"physical_rows":[1,3],"complete_original_fields":True}
    preflight={"fixture_pages":3}
    view={"selected_rows_report":selected,"raw_texts":["First. Second."],"authority":False}
    sentence={"raw_text":"First. Second.","complete_characters":14,"formalized":False}
    files={}
    for key,value in (("preflight_report_file",preflight),("selected_rows_report_file",selected),("parent_text_views_report_file",view)):
        path=datasets/(key+".json");path.write_bytes(driver.wire(value));files[key]=driver.identity(path)
    body={"schema":"legal-span-reconstruction-population-inputs/v1",**files,
        "parent_descriptor":{"fixture":"independent parent"},"expected_parent_descriptor_sha256":"a"*64,
        "expected_preflight_sha256":driver.sha(driver.wire(preflight)),
        "expected_selected_rows_report_sha256":driver.sha(driver.wire(selected)),
        "row_ordinals":[1,3],"expected_row_sha256s":["b"*64,"c"*64],
        "expected_parent_text_views_report_sha256":driver.sha(driver.wire(view)),
        "expected_parent_sentence_candidates_report_sha256":driver.sha(driver.wire(sentence))}
    return body,view,sentence


@pytest.mark.parametrize("kind",["parent_text_views","sentence_candidates"])
def test_span_view_and_sentence_profiles_use_exact_current_owner_api(frozen,kind):
    datasets,_,_=frozen;body,view,sentence=source_alignment_fixture(datasets)
    master=datasets/"master.json";master.write_bytes(driver.wire(body));pin=driver.identity(master)
    job={"real_input":{"spec":{"kind":kind,"path":pin["path"],"file_bytes":pin["bytes"],"file_sha256":pin["sha256"]}}}
    expected=view if kind=="parent_text_views" else sentence
    result=driver.run_real_input(datasets,job)
    assert result=={"schema":"legal-span-reconstruction-real-input/v1","kind":kind,"report":expected}
    assert driver.unpack_real_input_report(driver.pack_real_input_report(result))==result


@pytest.mark.parametrize("damage",["selected_file","selected_full_pin","view_file","view_full_pin","rehashed_view","sentence_full_pin"])
def test_source_alignment_does_not_accept_retained_or_self_resealed_report_substitution(frozen,damage):
    datasets,_,_=frozen;body,view,_=source_alignment_fixture(datasets)
    if damage=="selected_file":body["selected_rows_report_file"]["sha256"]="0"*64
    elif damage=="selected_full_pin":body["expected_selected_rows_report_sha256"]="0"*64
    elif damage=="view_file":body["parent_text_views_report_file"]["sha256"]="0"*64
    elif damage=="view_full_pin":body["expected_parent_text_views_report_sha256"]="0"*64
    elif damage=="sentence_full_pin":body["expected_parent_sentence_candidates_report_sha256"]="0"*64
    else:
        view["raw_texts"][0]="invented";path=Path(body["parent_text_views_report_file"]["path"])
        path.write_bytes(driver.wire(view));body["parent_text_views_report_file"]=driver.identity(path)
    master=datasets/"master.json";master.write_bytes(driver.wire(body));pin=driver.identity(master)
    job={"real_input":{"spec":{"kind":"sentence_candidates","path":pin["path"],"file_bytes":pin["bytes"],"file_sha256":pin["sha256"]}}}
    with pytest.raises(ValueError):driver.run_real_input(datasets,job)


@pytest.mark.parametrize("damage",[None,"provenance_pin","provenance_file","record_order","lineage_file",
                                  "lineage_pin","page_population_pin","occurrence_pin"])
def test_occurrence_profile_replays_complete_ordered_cache_and_exact_parent_lineage(frozen,damage):
    import pyarrow as pa
    import pyarrow.parquet as pq
    datasets,_,_=frozen;body,view,_=source_alignment_fixture(datasets)
    package=datasets/"ipfs_datasets_py/logic/autoformal"
    (package/"legal_span_provenance.py").write_text(
        "from .legal_parent_text_views import digest\n"
        "def reconcile_span_provenance(rows, **kwargs):\n"
        "    return {'rows':rows,'record_pins':kwargs['expected_record_sha256s']}\n"
        "provenance_report_sha256=digest\n")
    (package/"legal_producer_page_lineage.py").write_text(
        "from .legal_parent_text_views import digest\n"
        "def validate_producer_page_lineage(report, reports, **kwargs):\n"
        "    required={'expected_report_sha256','expected_provenance_report_sha256s','expected_record_sha256s','witness_manifest','expected_witness_manifest_sha256','parent_page_inputs','expected_parent_page_inputs_sha256'}\n"
        "    if set(kwargs)!=required or kwargs['expected_report_sha256']!=digest(report):\n"
        "        raise ValueError('complete exact v2 lineage inputs differ')\n"
        "    pins=[p for source in reports for p in source['record_pins']]\n"
        "    if report['record_pins']!=pins or kwargs['expected_record_sha256s']!=pins:\n"
        "        raise ValueError('complete ordered lineage cache population differs')\n"
        "    if len(kwargs['parent_page_inputs'])!=1 or kwargs['expected_parent_page_inputs_sha256']!=digest(kwargs['parent_page_inputs']):\n"
        "        raise ValueError('complete qualified page population differs')\n"
        "    return {'valid':True}\n")
    (package/"legal_cache_parent_occurrences.py").write_text(
        "from .legal_parent_text_views import digest,prepare_parent_text_views\n"
        "from .legal_producer_page_lineage import validate_producer_page_lineage\n"
        "def prepare_cache_parent_occurrences(reports,lineage,views,**kwargs):\n"
        "    required={'expected_provenance_report_sha256s','expected_record_sha256s','expected_lineage_report_sha256','lineage_inputs','expected_parent_text_views_report_sha256','parent_text_views_inputs'}\n"
        "    if set(kwargs)!=required:raise ValueError('exact current occurrence API differs')\n"
        "    validate_producer_page_lineage(lineage,reports,expected_report_sha256=kwargs['expected_lineage_report_sha256'],**kwargs['lineage_inputs'])\n"
        "    if kwargs['expected_parent_text_views_report_sha256']!=digest(views) or views!=prepare_parent_text_views(views['selected_rows_report'],**kwargs['parent_text_views_inputs']):\n"
        "        raise ValueError('complete raw parent text replay differs')\n"
        "    return {'record_pins':kwargs['expected_record_sha256s'],'selected_occurrence':None,'formalized':False}\n"
        "cache_parent_occurrences_report_sha256=digest\n")
    files=[]
    for number,records in enumerate(([{"text":"first"},{"text":"second"}],[{"text":"third"}])):
        path=datasets/(str(number)+".parquet");pq.write_table(pa.Table.from_pylist(records),path)
        pins=["a","b"] if number==0 else ["c"];report={"rows":records,"record_pins":pins}
        retained=datasets/(str(number)+"-provenance.json");retained.write_bytes(driver.wire(report))
        files.append({"path":str(path),"bytes":path.stat().st_size,"file_sha256":driver.identity(path)["sha256"],
            "expected_rows":len(records),"cache_identity":{"fixture":number},"expected_row_sha256s":[],
            "expected_record_sha256s":pins,"expected_provenance_report_sha256":driver.sha(driver.wire(report)),
            "provenance_report_file":driver.identity(retained)})
    lineage={"record_pins":["a","b","c"],"authority":False};lineage_path=datasets/"lineage.json";lineage_path.write_bytes(driver.wire(lineage))
    witness={"fixture":"closed witnesses"};witness_path=datasets/"witness.json";witness_path.write_bytes(driver.wire(witness))
    preflight=json.loads(Path(body["preflight_report_file"]["path"]).read_text())
    selected=json.loads(Path(body["selected_rows_report_file"]["path"]).read_text())
    parent_inputs=[{"descriptor_sha256":body["expected_parent_descriptor_sha256"],"preflight_report":preflight,
        "expected_preflight_sha256":body["expected_preflight_sha256"],"row_report":selected,
        "expected_row_report_sha256":body["expected_selected_rows_report_sha256"]}]
    expected={"record_pins":["a","b","c"],"selected_occurrence":None,"formalized":False}
    body.update(span_files=files,demonstration_catalog={},expected_catalog_sha256="fixturecatalog",expected_catalog_source_pins=[],
        producer_witness_manifest=driver.identity(witness_path),expected_producer_witness_manifest_sha256=driver.sha(driver.wire(witness)),
        expected_parent_page_inputs_sha256=driver.sha(driver.wire(parent_inputs)),lineage_report_file=driver.identity(lineage_path),
        expected_lineage_report_sha256=driver.sha(driver.wire(lineage)),expected_cache_parent_occurrences_report_sha256=driver.sha(driver.wire(expected)))
    if damage=="provenance_pin":body["span_files"][0]["expected_provenance_report_sha256"]="0"*64
    elif damage=="provenance_file":body["span_files"][0]["provenance_report_file"]["sha256"]="0"*64
    elif damage=="record_order":body["span_files"].reverse()
    elif damage=="lineage_file":body["lineage_report_file"]["sha256"]="0"*64
    elif damage=="lineage_pin":body["expected_lineage_report_sha256"]="0"*64
    elif damage=="page_population_pin":body["expected_parent_page_inputs_sha256"]="0"*64
    elif damage=="occurrence_pin":body["expected_cache_parent_occurrences_report_sha256"]="0"*64
    master=datasets/"master.json";master.write_bytes(driver.wire(body));pin=driver.identity(master)
    job={"real_input":{"spec":{"kind":"cache_parent_occurrences","path":pin["path"],"file_bytes":pin["bytes"],"file_sha256":pin["sha256"]}}}
    if damage is None:
        assert driver.run_real_input(datasets,job)=={"schema":"legal-span-reconstruction-real-input/v1","kind":"cache_parent_occurrences","report":expected}
    else:
        with pytest.raises(ValueError):driver.run_real_input(datasets,job)
