#!/usr/bin/env python3
"""Qualify complete source alignment, candidate context and review handoffs.

These receipts concern executable source contracts. They do not confer corpus
coverage, reviewed legal meaning, model qualification or Lean kernel authority.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zlib


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legal-autoformalization-source-alignment-qualification/v1"
TASKS = ("LAC-S11-011", "LAC-S11-024", "LAC-S11-044")
# Complete versioned views preserve markup observations and region witnesses;
# their bounded real baselines exceed the old phase7 four-MiB ceiling.
MAX_BYTES = 16 * 1024 * 1024
MAX_RESULT_BYTES = 900 * 1024  # Below unchanged native 1,048,576-byte file gate.
MAX_COMPRESSED_REPORT_BYTES = 625 * 1024
MAX_ENCODED_REPORT_BYTES = 840 * 1024
REPORT_ENVELOPE_SCHEMA = "legal-autoformalization-source-alignment-real-report-envelope/v1"
MAX_FAILURE_CASES = 5
MAX_FAILURE_FIELD_BYTES = 512
MAX_FAILURE_TAIL_BYTES = 1024
MAX_TASKBOARD_BYTES = 128 * 1024
TASKBOARD_PROJECTION_PROFILE = "exact-three-native-todo-to-completed-status-projection/v1"


class QualificationFailure(ValueError):
    """A refused qualification with bounded diagnostics retained by its caller."""

    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        first = next(iter(diagnostics["failed_cases"]), None)
        detail = f": {first['name']}: {first['message']}" if first else ""
        super().__init__(message + detail)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    need(len(raw) <= MAX_BYTES, "JSON exceeds qualification read bound")
    return json.loads(raw)


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
         "Input changed during hashing")
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": after.st_size}


def relative_file(root, value):
    need(isinstance(value, str) and value, "Missing relative source path")
    parsed = PurePosixPath(value)
    need(not parsed.is_absolute() and ".." not in parsed.parts
         and parsed.as_posix() == value and "\\" not in value,
         "Invalid relative source path")
    path = (root / value).resolve(strict=True)
    need(path.is_relative_to(root) and path.is_file(), "Source escaped pinned repository")
    return path


def git(root, *args):
    run = subprocess.run(["git", "-C", str(root), *args], text=True,
                         capture_output=True, timeout=20, check=False,
                         env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    need(run.returncode == 0, "Pinned repository identity unavailable")
    return run.stdout.strip()


def _taskboard_bytes(path, expected=None):
    """Capture a bounded regular file without following aliases or path races."""
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path,
         "Taskboard input alias refused")
    marker = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                           info.st_mtime_ns, info.st_ctime_ns)
    before_path = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_TASKBOARD_BYTES,
             "Taskboard input byte bound exceeded")
        need(marker(before_path) == marker(before), "Taskboard input changed before read")
        raw = stream.read(MAX_TASKBOARD_BYTES + 1)
        need(len(raw) <= MAX_TASKBOARD_BYTES and len(raw) == before.st_size,
             "Taskboard input byte bound exceeded")
        need(marker(before) == marker(os.fstat(stream.fileno())) == marker(path.lstat()),
             "Taskboard input changed during read")
        need(path.resolve(strict=True) == path, "Taskboard input alias appeared during read")
    if expected is not None:
        need(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"],
             "Initial taskboard template pin differs")
    raw.decode("utf-8", errors="strict")
    return raw


def taskboard_binding(sources):
    """Pin the initial board while admitting only its eight native status states.

    Current statuses are operational state and deliberately never enter the
    qualification bindings. Every other byte remains independently pinned.
    """
    pin = sources.get("initial_board_template")
    need(type(pin) is dict and set(pin) == {"path", "sha256", "bytes"},
         "Missing or nonclosed initial taskboard template pin")
    need(type(pin["path"]) is str and type(pin["bytes"]) is int
         and 0 < pin["bytes"] <= MAX_TASKBOARD_BYTES
         and type(pin["sha256"]) is str and len(pin["sha256"]) == 64
         and all(char in "0123456789abcdef" for char in pin["sha256"]),
         "Invalid initial taskboard template pin")
    config_path = ROOT / "config/legal_autoformalization_source_alignment_scheduler.json"
    config = json.loads(_taskboard_bytes(config_path))
    need(type(config) is dict, "Taskboard scheduler must be an object")
    name = config.get("taskboard_path")
    board_path = relative_file(ROOT, name)
    need(board_path == ROOT / name, "Configured taskboard alias refused")
    for row in sources["immutable_inputs"]:
        if row.get("repository") == "accelerator":
            need(relative_file(ROOT, row["path"]) != board_path,
                 "Mutable configured taskboard cannot be an immutable input")
    initial = _taskboard_bytes(Path(pin["path"]), pin)
    lines = initial.splitlines(keepends=True)
    headers, status_lines = [], []
    current_task = None
    for index, line in enumerate(lines):
        if line.startswith(b"## "):
            heading = line[3:].split(None, 1)
            need(bool(heading), "Initial taskboard template has an empty task heading")
            current_task = heading[0].decode("utf-8")
            headers.append(current_task)
        if line.startswith(b"- Status:"):
            need(current_task in TASKS and line == b"- Status: todo\n",
                 "Initial taskboard template must have all three tasks todo")
            status_lines.append((current_task, index))
    need(tuple(headers) == TASKS and tuple(task for task, _ in status_lines) == TASKS,
         "Initial taskboard template exact three-task population differs")
    current = _taskboard_bytes(board_path)
    permitted = False
    for subset in range(8):
        projected = list(lines)
        for offset, (_, index) in enumerate(status_lines):
            if subset & (1 << offset):
                projected[index] = b"- Status: completed\n"
        if current == b"".join(projected):
            permitted = True
            break
    need(permitted, "Current taskboard differs from exact native status-only projection")
    return {"profile": TASKBOARD_PROJECTION_PROFILE, "repository_path": name,
            "initial_board_template": dict(pin), "allowed_states": ["todo", "completed"]}


def bindings(task, sources_path):
    need(task in TASKS, "Unknown source-alignment qualification task")
    sources_path = Path(sources_path).resolve(strict=True)
    need(sources_path.is_relative_to(ROOT), "Sources must be repository owned")
    sources = read_json(sources_path)
    need(sources.get("schema") == "legal-autoformalization-source-alignment-sources/v1",
         "Wrong source manifest schema")
    board = taskboard_binding(sources)
    datasets = Path(sources["datasets"]["root"]).resolve(strict=True)
    commit = git(datasets, "rev-parse", "HEAD")
    need(commit == sources["datasets"]["commit"], "Datasets commit moved")
    if sources["datasets"].get("expected_branch"):
        need(git(datasets, "branch", "--show-current") == sources["datasets"]["expected_branch"],
             "Datasets branch differs")
    need(not git(datasets, "status", "--porcelain=v1", "--untracked-files=no"),
         "Pinned datasets checkout has tracked changes")
    job = sources["jobs"][task]
    tests = job["test_paths"]
    need(isinstance(tests, list) and tests and len(tests) == len(set(tests)),
         "Qualification requires distinct reviewed test files")
    rows = []
    pins = {row["path"]: row["sha256"] for row in sources["immutable_inputs"]
            if row.get("repository", "datasets") == "datasets"}
    for name in dict.fromkeys([*job["code_paths"], *tests]):
        path = relative_file(datasets, name)
        git(datasets, "ls-files", "--error-unmatch", "--", name)
        actual = identity(path)
        need(actual["sha256"] == pins.get(name), "Qualification source pin differs")
        rows.append({**actual, "repository_path": name})
    accelerator_rows = []
    for row in sources["immutable_inputs"]:
        if row.get("repository") != "accelerator":
            continue
        path = relative_file(ROOT, row["path"])
        git(ROOT, "ls-files", "--error-unmatch", "--", row["path"])
        actual = identity(path)
        need(actual["sha256"] == row["sha256"], "Native supervisor source pin differs")
        accelerator_rows.append({"repository_path": row["path"],
            "sha256": actual["sha256"], "bytes": actual["bytes"]})
    prior = sources["prior_span_reconstruction_completion"]
    need(identity(prior["path"])["sha256"] == prior["sha256"], "Parent-pages completion moved")
    prerequisite = read_json(prior["path"])
    need(prerequisite["status"] == "bounded_literal_parent_views_occurrence_inventory_and_sentence_candidates_complete",
         "Parent-pages prerequisite is not complete")
    if "original_plan" in sources:
        plan = sources["original_plan"]
        need(identity(plan["path"])["sha256"] == plan["sha256"], "Original plan moved")
    for row in sources.get("external_evidence", []):
        need(Path(row["path"]).stat().st_size == row["bytes"], "External evidence moved: byte count differs")
        need(identity(row["path"])["sha256"] == row["sha256"], "External evidence moved")
    interpreter = identity(job.get("python_executable", sys.executable))
    if job.get("python_executable_sha256"):
        need(interpreter["sha256"] == job["python_executable_sha256"], "Qualification interpreter moved")
    return sources, datasets, job, {"source_manifest_sha256": identity(sources_path)["sha256"],
        "source_manifest_repository_path": sources_path.relative_to(ROOT).as_posix(),
        "driver_sha256": identity(__file__)["sha256"], "datasets_commit": commit,
        "datasets_tree": git(datasets, "rev-parse", "HEAD^{tree}"), "inputs": rows,
        "prior_span_reconstruction_completion_sha256": prior["sha256"], "accelerator_inputs": accelerator_rows,
        "external_evidence": sources.get("external_evidence", []), "interpreter": interpreter,
        "taskboard": board}


def command(job):
    return [job.get("python_executable", sys.executable), "-B", "-m", "pytest", "-q",
            *job["test_paths"]]


def execution_environment_policy(datasets):
    """Nonsecret, recorded overrides binding imports and pytest invocation."""
    return {"PYTHONPATH": str(Path(datasets).resolve(strict=True)), "PYTEST_ADDOPTS": "", "PYTHONOPTIMIZE": "0", "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "IPFS_AUTO_INSTALL": "false",
            "IPFS_DATASETS_AUTO_INSTALL": "false", "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
            "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": ""}


REAL_INPUT_SCRIPT = '''import hashlib,json,os,stat,sys
from pathlib import Path
def need(ok,message):
    if not ok: raise ValueError(message)
spec=json.loads(sys.argv[1])
path=Path(spec["path"])
need(path.resolve(strict=True)==path,"Real-input alias refused")
need(spec["kind"] in ("entity_index_census","span_cache_sample","primary_graph_files","primary_graph_reconstruction","section_index_census","span_section_bridge","source_graph_proposals"),"Unknown real-input kind")
bridge_kind=spec["kind"] in ("section_index_census","span_section_bridge","source_graph_proposals")
cap=(spec.get("max_file_bytes",1073741824) if bridge_kind else
     1048576 if spec["kind"].startswith("primary_graph_") else
     262144 if spec["kind"]=="span_cache_sample" else spec["options"].get("max_file_bytes",20971520))
need(type(cap) is int and 0<cap<=(1073741824 if bridge_kind else 20971520),"Invalid real-input byte cap")
def snapshot(capture=False):
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(descriptor,"rb") as stream:
        before=os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,"Real-input exceeds initial file byte bound")
        h=hashlib.sha256(); count=0; captured=[]
        for block in iter(lambda:stream.read(1048576),b""):
            count+=len(block); need(count<=cap,"Real-input grew beyond byte bound"); h.update(block)
            if capture: captured.append(block)
        def marker(info):return(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
        need(marker(before)==marker(os.fstat(stream.fileno()))==marker(path.lstat()),"Real-input changed during hashing")
        need(h.hexdigest()==spec["file_sha256"],"Real-input file hash differs")
        return marker(before), b"".join(captured) if capture else None
before,manifest_bytes=snapshot(capture=spec["kind"].startswith("primary_graph_"))
if bridge_kind:
    def auxiliary_bytes(pinned,maximum):
        auxiliary=Path(pinned["path"])
        need(auxiliary.resolve(strict=True)==auxiliary,"Auxiliary input alias refused")
        fd=os.open(auxiliary,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,"rb") as stream:
            initial=os.fstat(stream.fileno())
            need(stat.S_ISREG(initial.st_mode) and initial.st_size==pinned["bytes"] and 0<initial.st_size<=maximum,"Auxiliary initial file byte bound exceeded")
            raw=stream.read(maximum+1)
            need(len(raw)<=maximum and len(raw)==initial.st_size,"Auxiliary grew beyond byte bound")
            marker=lambda i:(i.st_dev,i.st_ino,i.st_size,i.st_mtime_ns,i.st_ctime_ns)
            need(marker(initial)==marker(os.fstat(stream.fileno()))==marker(auxiliary.lstat()),"Auxiliary input changed during read")
            need(hashlib.sha256(raw).hexdigest()==pinned["sha256"],"Auxiliary input hash differs")
            return raw
    report=json.loads(auxiliary_bytes(spec["reconstruction_report"],1048576))
    from ipfs_datasets_py.logic.autoformal import legal_primary_section_index as indexes
    with indexes.PrimarySectionIndex(path,reconstruction_report=report,
            expected_candidate_sha256=spec["file_sha256"],
            expected_reconstruction_sha256=spec["expected_reconstruction_sha256"]) as index:
        need(index.index_sha256==spec["expected_index_sha256"],"Independent section index pin differs")
        if spec["kind"]=="section_index_census": result=index.receipt
        else:
            import pyarrow.parquet as pq
            import pyarrow as pa
            from ipfs_datasets_py.logic.autoformal import legal_span_cache_intake as intake
            from ipfs_datasets_py.logic.autoformal import legal_span_section_bridge as bridges
            from ipfs_datasets_py.logic.autoformal import legal_source_graph_proposals as proposals
            pa.set_cpu_count(1);pa.set_io_thread_count(1)
            observed=[]
            need(type(spec["span_files"]) is list and 0<len(spec["span_files"])<=4,"Bounded span file population required")
            for source in spec["span_files"]:
                import io
                raw=auxiliary_bytes({"path":source["path"],"sha256":source["file_sha256"],"bytes":source["bytes"]},262144)
                table=pq.ParquetFile(io.BytesIO(raw))
                need(type(source["expected_rows"]) is int and 0<source["expected_rows"]<=128,"Span population bound exceeded")
                need(table.metadata.num_rows==source["expected_rows"],"Span footer population differs")
                need(sum(table.metadata.row_group(i).total_byte_size for i in range(table.metadata.num_row_groups))<=4194304,"Span decoded bound exceeded")
                rows=[]
                for batch in table.iter_batches(batch_size=128,use_threads=False):rows.extend(batch.to_pylist())
                need(len(rows)==source["expected_rows"],"Complete span population differs")
                args={"cache_identity":source["cache_identity"],"expected_row_sha256s":source["expected_row_sha256s"],
                    "expected_record_sha256s":source["expected_record_sha256s"],"section_index":index,
                    "expected_section_index_sha256":spec["expected_index_sha256"],"identity_policy":spec["identity_policy"],
                    "expected_identity_policy_sha256":spec["expected_identity_policy_sha256"],
                    "source_bindings":{},"expected_source_bindings_sha256":spec["expected_source_bindings_sha256"]}
                bridge=bridges.reconcile_span_sections(rows,**args)
                need(bridge["bridge_sha256"]==source["expected_bridge_sha256"],"Independent complete bridge pin differs")
                if spec["kind"]=="span_section_bridge": value=bridge
                else:
                    value=proposals.prepare_source_graph_proposals(bridge,rows,expected_bridge_sha256=source["expected_bridge_sha256"],
                        bridge_inputs=args,row_inputs=source["sidecars"],expected_row_inputs_sha256=source["expected_sidecars_sha256"],
                        expected_record_sha256s=source["expected_record_sha256s"])
                    need(value["proposals_sha256"]==source["expected_proposals_sha256"],"Independent complete proposal pin differs")
                observed.append({"label":source["label"],"cache_identity":source["cache_identity"],"report":value})
            result={"schema":"legal-span-section-real-input/v1","kind":spec["kind"],"section_index":index.receipt,"span_files":observed}
elif spec["kind"].startswith("primary_graph_"):
    import tempfile
    from ipfs_datasets_py.logic.autoformal import legal_primary_graph_manifest as primary
    selection=primary.select_primary_graph_manifest(manifest_bytes,**spec["selection_options"])
    options={"expected_selection_sha256":spec["expected_selection_sha256"],
             "blobs_root":spec["blobs_root"],"limits":spec.get("limits")}
    if spec["kind"]=="primary_graph_files":
        result=primary.verify_primary_graph_files(selection,spec["snapshot_root"],**options)
    else:
        from ipfs_datasets_py.logic.autoformal.legal_primary_graph_reconstruction import reconstruct_to_jsonl
        with tempfile.TemporaryDirectory(prefix="legal-primary-candidate-") as temporary:
            result=reconstruct_to_jsonl(selection,spec["snapshot_root"],Path(temporary)/"candidate.jsonl",**options)
elif spec["kind"]=="entity_index_census":
    from ipfs_datasets_py.logic.autoformal.legal_entity_index_intake import census_parquet
    result=census_parquet(path,**spec["options"])
elif spec["kind"]=="span_cache_sample":
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ipfs_datasets_py.logic.autoformal import legal_span_cache_intake as intake
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    need(type(spec["expected_rows"]) is int and 0<spec["expected_rows"]<=128,"Invalid span sample population")
    table=pq.ParquetFile(path)
    need(table.metadata.num_rows==spec["expected_rows"],"Span sample population differs")
    need(sum(table.metadata.row_group(i).total_byte_size for i in range(table.metadata.num_row_groups))<=4194304,"Span sample decoded bound exceeded")
    rows=[]
    for batch in table.iter_batches(batch_size=128,use_threads=False): rows.extend(batch.to_pylist())
    need(len(rows)==spec["expected_rows"],"Span sample decoded population differs")
    identity=spec["cache_identity"]
    result=intake.prepare_span_cache_intake(rows,cache_identity=identity,
        expected_row_sha256s=[intake.row_sha256(row) for row in rows],
        expected_record_sha256s=[intake.record_sha256(row,ordinal=i,cache_identity=identity) for i,row in enumerate(rows)],
        source_bindings={},expected_source_bindings_sha256=intake.source_bindings_sha256({}))
else: raise ValueError("Unknown bounded real-input profile")
need(snapshot()[0]==before,"Real-input identity changed during observation")
print(json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False))
'''


SOURCE_VIEWS_KINDS = ("document_views", "source_view_inventory", "producer_lineage")
SOURCE_VIEWS_INPUT_SCRIPT = '''import hashlib,io,json,os,stat,sys
from pathlib import Path
def need(ok,message):
    if not ok: raise ValueError(message)
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def pinned_bytes(pinned,maximum):
    path=Path(pinned["path"])
    need(path.is_absolute() and path.resolve(strict=True)==path,"View input path alias refused")
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,"rb") as stream:
        before=os.fstat(stream.fileno())
        need(type(pinned["bytes"]) is int and stat.S_ISREG(before.st_mode)
            and 0<before.st_size==pinned["bytes"]<=maximum,"View initial file bound exceeded")
        raw=stream.read(maximum+1)
        need(len(raw)==before.st_size and len(raw)<=maximum,"View input grew beyond bound")
        marker=lambda i:(i.st_dev,i.st_ino,i.st_size,i.st_mtime_ns,i.st_ctime_ns)
        need(marker(before)==marker(os.fstat(stream.fileno()))==marker(path.lstat()),"View input changed during read")
        need(hashlib.sha256(raw).hexdigest()==pinned["sha256"],"Independent view file hash differs")
        return raw
spec=json.loads(sys.argv[1])
need(spec["kind"] in ("document_views","source_view_inventory","producer_lineage"),"Unknown source view profile")
master={"path":spec["path"],"bytes":spec["file_bytes"],"sha256":spec["file_sha256"]}
payload=json.loads(pinned_bytes(master,1048576))
need(payload["schema"]=="legal-source-views-population-inputs/v1","Wrong source views population schema")
from ipfs_datasets_py.logic.autoformal import legal_span_provenance as provenance
if spec["kind"] in ("document_views","source_view_inventory"):
    from ipfs_datasets_py.logic.autoformal import legal_official_document_views as views
    from ipfs_datasets_py.logic.autoformal import legal_source_view_inventory as inventory
    document_inputs=[]
    need(type(payload["official_documents"]) is list and len(payload["official_documents"])<=10,"Official comparison population bound exceeded")
    for item in payload["official_documents"]:
        raw=None if item["raw_path"] is None else pinned_bytes({"path":item["raw_path"],"bytes":item["expected_raw_bytes"],"sha256":item["expected_raw_sha256"]},8000000)
        report=json.loads(pinned_bytes({"path":item["view_report_path"],"bytes":item["view_report_bytes"],"sha256":item["view_report_sha256"]},33554432))
        document_inputs.append({"descriptor":item["descriptor"],"raw_bytes":raw,
            "expected_descriptor_sha256":item["expected_descriptor_sha256"],
            "expected_raw_sha256":item["expected_raw_sha256"],"expected_raw_bytes":item["expected_raw_bytes"],
            "view_report":report,"expected_view_report_sha256":item["expected_view_report_sha256"]})
    need(inventory.view_document_population_sha256(document_inputs)==payload["expected_document_population_sha256"],"Independent document population pin differs")
if spec["kind"]=="document_views":
    documents=[]
    for item in document_inputs:
        arguments={key:item[key] for key in ("descriptor","expected_descriptor_sha256","expected_raw_sha256","expected_raw_bytes")}
        if "limits" in item["view_report"]:arguments["limits"]=item["view_report"]["limits"]
        documents.append(views.validate_official_document_view(item["view_report"],item["raw_bytes"],
            expected_report_sha256=item["expected_view_report_sha256"],**arguments))
    result={"schema":"legal-source-views-real-input/v1","kind":spec["kind"],
        "document_population_sha256":payload["expected_document_population_sha256"],"documents":documents}
else:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    observed=[];provenance_reports=[];provenance_pins=[];record_pins=[]
    need(type(payload["span_files"]) is list and 0<len(payload["span_files"])<=4,"Span file population bound exceeded")
    for source in payload["span_files"]:
        raw=pinned_bytes({"path":source["path"],"bytes":source["bytes"],"sha256":source["file_sha256"]},262144)
        table=pq.ParquetFile(io.BytesIO(raw))
        need(type(source["expected_rows"]) is int and 0<source["expected_rows"]<=128
            and table.metadata.num_rows==source["expected_rows"],"Exact bounded span footer population differs")
        need(sum(table.metadata.row_group(i).total_byte_size for i in range(table.metadata.num_row_groups))<=4194304,"Span decoded bound exceeded")
        rows=[]
        for batch in table.iter_batches(batch_size=128,use_threads=False):rows.extend(batch.to_pylist())
        need(len(rows)==source["expected_rows"],"Complete physical span population differs")
        inputs={"cache_identity":source["cache_identity"],"expected_row_sha256s":source["expected_row_sha256s"],
            "expected_record_sha256s":source["expected_record_sha256s"],"demonstration_catalog":payload["demonstration_catalog"],
            "expected_catalog_sha256":payload["expected_catalog_sha256"],
            "expected_catalog_source_pins":payload["expected_catalog_source_pins"]}
        report=provenance.reconcile_span_provenance(rows,**inputs)
        need(provenance.provenance_report_sha256(report)==source["expected_provenance_report_sha256"],"Independent provenance report pin differs")
        provenance_reports.append(report);provenance_pins.append(source["expected_provenance_report_sha256"])
        record_pins.extend(source["expected_record_sha256s"])
        if spec["kind"]=="source_view_inventory":
            report=inventory.prepare_source_view_inventory(report,rows,provenance_inputs=inputs,
                expected_provenance_report_sha256=source["expected_provenance_report_sha256"],
                document_inputs=document_inputs,expected_document_population_sha256=payload["expected_document_population_sha256"])
            need(inventory.source_view_inventory_sha256(report)==source["expected_view_inventory_sha256"],"Independent versioned view inventory pin differs")
            observed.append({"label":source["label"],"cache_identity":source["cache_identity"],"report":report})
    if spec["kind"]=="source_view_inventory":
        result={"schema":"legal-source-views-real-input/v1","kind":spec["kind"],"span_files":observed}
    else:
        from ipfs_datasets_py.logic.autoformal import legal_producer_lineage as lineage
        witness=json.loads(pinned_bytes(payload["producer_witness_manifest"],4194304))
        need(hashlib.sha256(canonical(witness)).hexdigest()==payload["expected_producer_witness_manifest_sha256"],"Independent FULL producer witness manifest pin differs")
        report=lineage.reconcile_producer_lineage(provenance_reports,
            expected_provenance_report_sha256s=provenance_pins,expected_record_sha256s=record_pins,
            witness_manifest=witness,expected_witness_manifest_sha256=payload["expected_producer_witness_manifest_sha256"])
        need(lineage.lineage_report_sha256(report)==payload["expected_producer_lineage_sha256"],"Independent FULL producer lineage pin differs")
        result={"schema":"legal-source-views-real-input/v1","kind":spec["kind"],"report":report}
pinned_bytes(master,1048576)
print(json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False))
'''


PARENT_PAGE_KINDS = ("parent_page_preflight", "parent_selected_rows", "producer_page_lineage")
PARENT_PAGE_INPUT_SCRIPT = '''import hashlib,io,json,os,stat,sys
from pathlib import Path
def need(ok,message):
    if not ok: raise ValueError(message)
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def pinned_bytes(pinned,maximum):
    path=Path(pinned["path"])
    need(path.is_absolute() and path.resolve(strict=True)==path,"Parent input path alias refused")
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,"rb") as stream:
        before=os.fstat(stream.fileno())
        need(type(pinned["bytes"]) is int and stat.S_ISREG(before.st_mode)
            and 0<before.st_size==pinned["bytes"]<=maximum,"Parent initial file bound exceeded")
        raw=stream.read(maximum+1)
        need(len(raw)==before.st_size and len(raw)<=maximum,"Parent input grew beyond bound")
        marker=lambda i:(i.st_dev,i.st_ino,i.st_size,i.st_mtime_ns,i.st_ctime_ns)
        need(marker(before)==marker(os.fstat(stream.fileno()))==marker(path.lstat()),"Parent input changed during read")
        need(hashlib.sha256(raw).hexdigest()==pinned["sha256"],"Independent parent input hash differs")
        return raw
spec=json.loads(sys.argv[1])
need(spec["kind"] in ("parent_page_preflight","parent_selected_rows","producer_page_lineage"),"Unknown parent page profile")
master={"path":spec["path"],"bytes":spec["file_bytes"],"sha256":spec["file_sha256"]}
payload=json.loads(pinned_bytes(master,1048576))
need(payload["schema"]=="legal-span-reconstruction-population-inputs/v1","Wrong parent page population schema")
from ipfs_datasets_py.logic.autoformal import legal_parent_parquet_pages as pages
arguments={"expected_descriptor_sha256":payload["expected_parent_descriptor_sha256"]}
if "limits" in payload:arguments["limits"]=payload["limits"]
if spec["kind"]=="parent_page_preflight":
    report=pages.preflight_source_alignment(payload["parent_descriptor"],**arguments)
    need(pages.parent_page_report_sha256(report)==payload["expected_preflight_sha256"],"Independent FULL parent page preflight pin differs")
else:
    if spec["kind"]=="producer_page_lineage":
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ipfs_datasets_py.logic.autoformal import legal_span_provenance as provenance
        pa.set_cpu_count(1);pa.set_io_thread_count(1)
        provenance_reports=[];provenance_pins=[];record_pins=[];physical_count=0
        need(type(payload["span_files"]) is list and 0<len(payload["span_files"])<=4,"Span file population bound exceeded")
        for source in payload["span_files"]:
            raw=pinned_bytes({"path":source["path"],"bytes":source["bytes"],"sha256":source["file_sha256"]},262144)
            table=pq.ParquetFile(io.BytesIO(raw))
            need(type(source["expected_rows"]) is int and 0<source["expected_rows"]<=128
                and table.metadata.num_rows==source["expected_rows"],"Exact bounded span footer population differs")
            physical_count+=source["expected_rows"]
            need(physical_count<=128,"Complete physical provenance population bound exceeded")
            need(sum(table.metadata.row_group(i).total_byte_size for i in range(table.metadata.num_row_groups))<=4194304,"Span decoded bound exceeded")
            rows=[]
            for batch in table.iter_batches(batch_size=128,use_threads=False):rows.extend(batch.to_pylist())
            need(len(rows)==source["expected_rows"],"Complete physical span population differs")
            inputs={"cache_identity":source["cache_identity"],"expected_row_sha256s":source["expected_row_sha256s"],
                "expected_record_sha256s":source["expected_record_sha256s"],"demonstration_catalog":payload["demonstration_catalog"],
                "expected_catalog_sha256":payload["expected_catalog_sha256"],
                "expected_catalog_source_pins":payload["expected_catalog_source_pins"]}
            if "start_ordinal" in source:inputs["start_ordinal"]=source["start_ordinal"]
            report=provenance.reconcile_span_provenance(rows,**inputs)
            need(provenance.provenance_report_sha256(report)==source["expected_provenance_report_sha256"],"Independent provenance report pin differs")
            provenance_reports.append(report);provenance_pins.append(source["expected_provenance_report_sha256"])
            record_pins.extend(source["expected_record_sha256s"])
    preflight=json.loads(pinned_bytes(payload["preflight_report_file"],16777216))
    need(pages.parent_page_report_sha256(preflight)==payload["expected_preflight_sha256"],"Independent FULL parent page preflight pin differs")
    selected=pages.read_selected_parent_rows(payload["parent_descriptor"],preflight_report=preflight,
        expected_preflight_sha256=payload["expected_preflight_sha256"],row_ordinals=payload["row_ordinals"],
        expected_row_sha256s=payload["expected_row_sha256s"],**arguments)
    need(pages.parent_row_report_sha256(selected)==payload["expected_selected_rows_sha256"],"Independent FULL selected parent rows pin differs")
    if spec["kind"]=="parent_selected_rows":report=selected
    else:
        retained=json.loads(pinned_bytes(payload["selected_rows_report_file"],16777216))
        need(pages.parent_row_report_sha256(retained)==payload["expected_selected_rows_sha256"],"Independent FULL retained parent row report pin differs")
        need(canonical(retained)==canonical(selected),"Retained parent rows differ from complete current page regeneration")
        parent_inputs=[{"descriptor_sha256":payload["expected_parent_descriptor_sha256"],
            "preflight_report":preflight,"expected_preflight_sha256":payload["expected_preflight_sha256"],
            "row_report":retained,"expected_row_report_sha256":payload["expected_selected_rows_sha256"]}]
        need(hashlib.sha256(canonical(parent_inputs)).hexdigest()==payload["expected_parent_page_inputs_sha256"],"Independent FULL parent page input population pin differs")
        witness=json.loads(pinned_bytes(payload["producer_witness_manifest"],4194304))
        need(hashlib.sha256(canonical(witness)).hexdigest()==payload["expected_producer_witness_manifest_sha256"],"Independent FULL producer witness manifest pin differs")
        from ipfs_datasets_py.logic.autoformal import legal_producer_page_lineage as lineage
        report=lineage.reconcile_producer_page_lineage(provenance_reports,
            expected_provenance_report_sha256s=provenance_pins,expected_record_sha256s=record_pins,
            witness_manifest=witness,expected_witness_manifest_sha256=payload["expected_producer_witness_manifest_sha256"],
            parent_page_inputs=parent_inputs,expected_parent_page_inputs_sha256=payload["expected_parent_page_inputs_sha256"])
        need(lineage.producer_page_lineage_report_sha256(report)==payload["expected_producer_page_lineage_sha256"],"Independent FULL producer page lineage pin differs")
result={"schema":"legal-span-reconstruction-real-input/v1","kind":spec["kind"],"report":report}
pinned_bytes(master,1048576)
print(json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False))
'''


SOURCE_SPAN_KINDS = ("parent_text_views", "cache_parent_occurrences", "sentence_candidates")
SOURCE_SPAN_INPUT_SCRIPT = '''import hashlib,io,json,os,stat,sys
from pathlib import Path
def need(ok,message):
    if not ok: raise ValueError(message)
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def pinned_bytes(pinned,maximum):
    path=Path(pinned["path"])
    need(path.is_absolute() and path.resolve(strict=True)==path,"Span reconstruction input path alias refused")
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,"rb") as stream:
        before=os.fstat(stream.fileno())
        need(type(pinned["bytes"]) is int and stat.S_ISREG(before.st_mode)
            and 0<before.st_size==pinned["bytes"]<=maximum,"Span reconstruction initial file bound exceeded")
        raw=stream.read(maximum+1)
        need(len(raw)==before.st_size and len(raw)<=maximum,"Span reconstruction input grew beyond bound")
        marker=lambda i:(i.st_dev,i.st_ino,i.st_size,i.st_mtime_ns,i.st_ctime_ns)
        need(marker(before)==marker(os.fstat(stream.fileno()))==marker(path.lstat()),"Span reconstruction input changed during read")
        need(hashlib.sha256(raw).hexdigest()==pinned["sha256"],"Independent span reconstruction input hash differs")
        return raw
spec=json.loads(sys.argv[1])
need(spec["kind"] in ("parent_text_views","cache_parent_occurrences","sentence_candidates"),"Unknown span reconstruction profile")
master={"path":spec["path"],"bytes":spec["file_bytes"],"sha256":spec["file_sha256"]}
payload=json.loads(pinned_bytes(master,1048576))
need(payload["schema"]=="legal-span-reconstruction-population-inputs/v1","Wrong span reconstruction population schema")
if spec["kind"]=="cache_parent_occurrences":
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ipfs_datasets_py.logic.autoformal import legal_span_provenance as provenance
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    provenance_reports=[];provenance_pins=[];record_pins=[];physical_count=0
    need(type(payload["span_files"]) is list and 0<len(payload["span_files"])<=4,"Span file population bound exceeded")
    for source in payload["span_files"]:
        raw=pinned_bytes({"path":source["path"],"bytes":source["bytes"],"sha256":source["file_sha256"]},262144)
        table=pq.ParquetFile(io.BytesIO(raw))
        need(type(source["expected_rows"]) is int and 0<source["expected_rows"]<=128
            and table.metadata.num_rows==source["expected_rows"],"Exact bounded span footer population differs")
        physical_count+=source["expected_rows"]
        need(physical_count<=128,"Complete physical provenance population bound exceeded")
        need(sum(table.metadata.row_group(i).total_byte_size for i in range(table.metadata.num_row_groups))<=4194304,"Span decoded bound exceeded")
        rows=[]
        for batch in table.iter_batches(batch_size=128,use_threads=False):rows.extend(batch.to_pylist())
        need(len(rows)==source["expected_rows"],"Complete physical span population differs")
        inputs={"cache_identity":source["cache_identity"],"expected_row_sha256s":source["expected_row_sha256s"],
            "expected_record_sha256s":source["expected_record_sha256s"],"demonstration_catalog":payload["demonstration_catalog"],
            "expected_catalog_sha256":payload["expected_catalog_sha256"],
            "expected_catalog_source_pins":payload["expected_catalog_source_pins"]}
        if "start_ordinal" in source:inputs["start_ordinal"]=source["start_ordinal"]
        checked=provenance.reconcile_span_provenance(rows,**inputs)
        need(provenance.provenance_report_sha256(checked)==source["expected_provenance_report_sha256"],"Independent provenance report pin differs")
        retained=json.loads(pinned_bytes(source["provenance_report_file"],16777216))
        need(canonical(retained)==canonical(checked),"Retained provenance differs from complete current raw-row regeneration")
        provenance_reports.append(checked);provenance_pins.append(source["expected_provenance_report_sha256"])
        record_pins.extend(source["expected_record_sha256s"])
from ipfs_datasets_py.logic.autoformal import legal_parent_text_views as views
preflight=json.loads(pinned_bytes(payload["preflight_report_file"],16777216))
selected=json.loads(pinned_bytes(payload["selected_rows_report_file"],16777216))
view_inputs={"expected_selected_rows_report_sha256":payload["expected_selected_rows_report_sha256"],
    "parent_descriptor":payload["parent_descriptor"],"expected_parent_descriptor_sha256":payload["expected_parent_descriptor_sha256"],
    "preflight_report":preflight,"expected_preflight_sha256":payload["expected_preflight_sha256"],
    "row_ordinals":payload["row_ordinals"],"expected_row_sha256s":payload["expected_row_sha256s"]}
if "page_limits" in payload:view_inputs["page_limits"]=payload["page_limits"]
if "parent_text_view_limits" in payload:view_inputs["limits"]=payload["parent_text_view_limits"]
checked_view=views.prepare_parent_text_views(selected,**view_inputs)
need(views.parent_text_views_report_sha256(checked_view)==payload["expected_parent_text_views_report_sha256"],"Independent FULL parent text views pin differs")
if spec["kind"]=="parent_text_views":report=checked_view
else:
    retained_view=json.loads(pinned_bytes(payload["parent_text_views_report_file"],16777216))
    need(canonical(retained_view)==canonical(checked_view),"Retained parent text views differ from complete fixed-owner regeneration")
    if spec["kind"]=="sentence_candidates":
        from ipfs_datasets_py.logic.autoformal import legal_parent_sentence_candidates as sentences
        options={"limits":payload["sentence_limits"]} if "sentence_limits" in payload else {}
        report=sentences.prepare_parent_sentence_candidates(retained_view,
            expected_parent_text_views_report_sha256=payload["expected_parent_text_views_report_sha256"],
            parent_text_views_inputs=view_inputs,**options)
        need(sentences.parent_sentence_candidates_report_sha256(report)==payload["expected_parent_sentence_candidates_report_sha256"],"Independent FULL parent sentence candidates pin differs")
    else:
        from ipfs_datasets_py.logic.autoformal import legal_producer_page_lineage as lineage
        from ipfs_datasets_py.logic.autoformal import legal_cache_parent_occurrences as occurrences
        witness=json.loads(pinned_bytes(payload["producer_witness_manifest"],4194304))
        need(hashlib.sha256(canonical(witness)).hexdigest()==payload["expected_producer_witness_manifest_sha256"],"Independent FULL producer witness manifest pin differs")
        parent_inputs=[{"descriptor_sha256":payload["expected_parent_descriptor_sha256"],"preflight_report":preflight,
            "expected_preflight_sha256":payload["expected_preflight_sha256"],"row_report":selected,
            "expected_row_report_sha256":payload["expected_selected_rows_report_sha256"]}]
        need(hashlib.sha256(canonical(parent_inputs)).hexdigest()==payload["expected_parent_page_inputs_sha256"],"Independent FULL parent page input population pin differs")
        lineage_inputs={"expected_provenance_report_sha256s":provenance_pins,"expected_record_sha256s":record_pins,
            "witness_manifest":witness,"expected_witness_manifest_sha256":payload["expected_producer_witness_manifest_sha256"],
            "parent_page_inputs":parent_inputs,"expected_parent_page_inputs_sha256":payload["expected_parent_page_inputs_sha256"]}
        lineage_report=json.loads(pinned_bytes(payload["lineage_report_file"],16777216))
        lineage.validate_producer_page_lineage(lineage_report,provenance_reports,
            expected_report_sha256=payload["expected_lineage_report_sha256"],**lineage_inputs)
        options={"limits":payload["occurrence_limits"]} if "occurrence_limits" in payload else {}
        report=occurrences.prepare_cache_parent_occurrences(provenance_reports,lineage_report,retained_view,
            expected_provenance_report_sha256s=provenance_pins,expected_record_sha256s=record_pins,
            expected_lineage_report_sha256=payload["expected_lineage_report_sha256"],lineage_inputs=lineage_inputs,
            expected_parent_text_views_report_sha256=payload["expected_parent_text_views_report_sha256"],
            parent_text_views_inputs=view_inputs,**options)
        need(occurrences.cache_parent_occurrences_report_sha256(report)==payload["expected_cache_parent_occurrences_report_sha256"],"Independent FULL cache-parent occurrence inventory pin differs")
result={"schema":"legal-span-reconstruction-real-input/v1","kind":spec["kind"],"report":report}
pinned_bytes(master,1048576)
print(json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False))
'''


SOURCE_ALIGNMENT_KINDS = ("original_source_alignment", "candidate_context_inventory", "source_review_handoff")
SOURCE_ALIGNMENT_INPUT_SCRIPT = '''import hashlib,json,os,stat,sys
from pathlib import Path
def need(ok,message):
    if not ok: raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def full(value):return hashlib.sha256(canonical(value)).hexdigest()
def pinned_bytes(pin,maximum):
    need(type(pin) is dict and set(pin)=={"path","sha256","bytes"},"Closed raw source-alignment file pin required")
    path=Path(pin["path"])
    need(path.is_absolute() and path.resolve(strict=True)==path,"Source-alignment input alias refused")
    need(type(pin["bytes"]) is int and 0<pin["bytes"]<=maximum,"Source-alignment input byte bound exceeded")
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,"rb") as stream:
        before=os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size==pin["bytes"],"Source-alignment initial file population differs")
        raw=stream.read(maximum+1)
        marker=lambda info:(info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
        need(len(raw)==pin["bytes"] and len(raw)<=maximum,"Source-alignment file grew beyond bound")
        need(marker(before)==marker(os.fstat(stream.fileno()))==marker(path.lstat()) and path.resolve(strict=True)==path,"Source-alignment input changed during capture")
    need(hashlib.sha256(raw).hexdigest()==pin["sha256"],"Source-alignment raw input file pin differs")
    return raw
def pinned_json(pin,maximum=16777216):
    value=json.loads(pinned_bytes(pin,maximum))
    need(type(value) is dict,"Source-alignment pinned report must be an object")
    return value
spec=json.loads(sys.argv[1])
need(type(spec) is dict and set(spec)=={"kind","path","file_bytes","file_sha256"},"Closed source-alignment profile required")
need(spec["kind"] in ("original_source_alignment","candidate_context_inventory","source_review_handoff"),"Unknown source-alignment profile")
master={"path":spec["path"],"sha256":spec["file_sha256"],"bytes":spec["file_bytes"]}
payload=pinned_json(master,1048576)
need(payload["schema"]=="legal-source-alignment-population-inputs/v1","Wrong source-alignment population schema")
from ipfs_datasets_py.logic.autoformal import legal_parent_text_views as parent_views
from ipfs_datasets_py.logic.autoformal import legal_source_view_inventory as document_population
from ipfs_datasets_py.logic.autoformal import legal_parent_source_alignment as alignment
selected=pinned_json(payload["selected_rows_report_file"])
preflight=pinned_json(payload["preflight_report_file"])
parent_report=pinned_json(payload["parent_text_views_report_file"])
view_inputs={"expected_selected_rows_report_sha256":payload["expected_selected_rows_report_sha256"],
    "parent_descriptor":payload["parent_descriptor"],"expected_parent_descriptor_sha256":payload["expected_parent_descriptor_sha256"],
    "preflight_report":preflight,"expected_preflight_sha256":payload["expected_preflight_sha256"],
    "row_ordinals":payload["row_ordinals"],"expected_row_sha256s":payload["expected_row_sha256s"]}
if "page_limits" in payload:view_inputs["page_limits"]=payload["page_limits"]
if "parent_text_view_limits" in payload:view_inputs["limits"]=payload["parent_text_view_limits"]
parent_views.validate_parent_text_views(parent_report,selected,
    expected_report_sha256=payload["expected_parent_text_views_report_sha256"],**view_inputs)
need(type(payload["official_documents"]) is list and 0<len(payload["official_documents"])<=10,"Complete bounded official comparison document population required")
fields={"raw_path","descriptor","expected_descriptor_sha256","expected_raw_sha256","expected_raw_bytes",\
    "view_report_path","view_report_sha256","view_report_bytes","expected_view_report_sha256"}
documents=[];total_raw=0;total_reports=0
for document in payload["official_documents"]:
    need(type(document) is dict and set(document)==fields,"Closed official comparison document file inputs required")
    need(type(document["expected_raw_bytes"]) is int and 0<document["expected_raw_bytes"]<=8388608,"Official raw document byte bound exceeded")
    need(type(document["view_report_bytes"]) is int and 0<document["view_report_bytes"]<=16777216,"Official view report byte bound exceeded")
    total_raw+=document["expected_raw_bytes"];total_reports+=document["view_report_bytes"]
    need(total_raw<=83886080 and total_reports<=16777216,"Complete official comparison input byte bound exceeded before capture")
for document in payload["official_documents"]:
    raw=None if document["raw_path"] is None else pinned_bytes({"path":document["raw_path"],"sha256":document["expected_raw_sha256"],"bytes":document["expected_raw_bytes"]},8388608)
    view=pinned_json({"path":document["view_report_path"],"sha256":document["view_report_sha256"],"bytes":document["view_report_bytes"]})
    need(full(view)==document["expected_view_report_sha256"],"Independent FULL official view report pin differs")
    documents.append({"descriptor":document["descriptor"],"raw_bytes":raw,
        "expected_descriptor_sha256":document["expected_descriptor_sha256"],"expected_raw_sha256":document["expected_raw_sha256"],
        "expected_raw_bytes":document["expected_raw_bytes"],"view_report":view,"expected_view_report_sha256":document["expected_view_report_sha256"]})
need(document_population.view_document_population_sha256(documents)==payload["expected_document_population_sha256"],"Independent FULL complete official document population pin differs")
alignment_inputs={"expected_parent_text_views_report_sha256":payload["expected_parent_text_views_report_sha256"],
    "parent_text_views_inputs":view_inputs,"document_inputs":documents,
    "expected_document_population_sha256":payload["expected_document_population_sha256"]}
if "alignment_limits" in payload:alignment_inputs["limits"]=payload["alignment_limits"]
alignment_report=alignment.prepare_parent_source_alignment(parent_report,**alignment_inputs)
need(alignment.parent_source_alignment_report_sha256(alignment_report)==payload["expected_original_source_alignment_report_sha256"],"Independent FULL source-alignment report pin differs")
need(canonical(alignment_report)==canonical(pinned_json(payload["alignment_report_file"])),"Retained source alignment differs from fixed-owner replay")
if spec["kind"]=="original_source_alignment":report=alignment_report
else:
    from ipfs_datasets_py.logic.autoformal import legal_parent_sentence_candidates as sentences
    from ipfs_datasets_py.logic.autoformal import legal_candidate_context_inventory as contexts
    sentence_report=pinned_json(payload["sentence_candidates_report_file"])
    sentence_inputs={"expected_parent_text_views_report_sha256":payload["expected_parent_text_views_report_sha256"],"parent_text_views_inputs":view_inputs}
    if "sentence_limits" in payload:sentence_inputs["limits"]=payload["sentence_limits"]
    sentences.validate_parent_sentence_candidates(sentence_report,parent_report,
        expected_report_sha256=payload["expected_sentence_candidates_report_sha256"],**sentence_inputs)
    context_arguments={"expected_sentence_candidates_report_sha256":payload["expected_sentence_candidates_report_sha256"],
        "sentence_inputs":sentence_inputs,"expected_alignment_report_sha256":payload["expected_original_source_alignment_report_sha256"],
        "alignment_inputs":alignment_inputs}
    if "context_limits" in payload:context_arguments["limits"]=payload["context_limits"]
    context_report=contexts.prepare_candidate_context_inventory(sentence_report,alignment_report,**context_arguments)
    need(contexts.candidate_context_inventory_report_sha256(context_report)==payload["expected_candidate_context_inventory_report_sha256"],"Independent FULL complete candidate context pin differs")
    need(canonical(context_report)==canonical(pinned_json(payload["context_report_file"])),"Retained candidate context differs from fixed-owner replay")
    if spec["kind"]=="candidate_context_inventory":report=context_report
    else:
        from ipfs_datasets_py.logic.autoformal import legal_source_review_handoff as handoffs
        context_inputs={"sentence_candidates_report":sentence_report,"alignment_report":alignment_report,**context_arguments}
        handoff_arguments={"expected_candidate_context_inventory_report_sha256":payload["expected_candidate_context_inventory_report_sha256"],"context_inputs":context_inputs}
        if "handoff_limits" in payload:handoff_arguments["limits"]=payload["handoff_limits"]
        report=handoffs.prepare_source_review_handoff(context_report,**handoff_arguments)
        need(handoffs.source_review_handoff_report_sha256(report)==payload["expected_source_review_handoff_report_sha256"],"Independent FULL complete source review handoff pin differs")
result={"schema":"legal-source-alignment-real-input/v1","kind":spec["kind"],"report":report}
pinned_bytes(master,1048576)
print(json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False))
'''


def real_input_command(job):
    profile = job.get("real_input")
    if profile is None:
        return None
    spec = profile["spec"]
    need(spec.get("kind") in (*SOURCE_ALIGNMENT_KINDS, *SOURCE_SPAN_KINDS, *PARENT_PAGE_KINDS, *SOURCE_VIEWS_KINDS, "entity_index_census", "span_cache_sample",
                              "primary_graph_files", "primary_graph_reconstruction",
                              "section_index_census", "span_section_bridge", "source_graph_proposals"),
         "Unknown bounded real-input profile")
    script = (SOURCE_ALIGNMENT_INPUT_SCRIPT if spec["kind"] in SOURCE_ALIGNMENT_KINDS else
              SOURCE_SPAN_INPUT_SCRIPT if spec["kind"] in SOURCE_SPAN_KINDS else
              PARENT_PAGE_INPUT_SCRIPT if spec["kind"] in PARENT_PAGE_KINDS else
              SOURCE_VIEWS_INPUT_SCRIPT if spec["kind"] in SOURCE_VIEWS_KINDS else REAL_INPUT_SCRIPT)
    return [job.get("python_executable", sys.executable), "-B", "-c", script,
            wire(spec).decode("utf-8")]


def run_real_input(datasets, job):
    """Recompute a pinned real-input observation in the datasets interpreter."""
    cmd = real_input_command(job)
    need(cmd is not None, "Real-input profile required")
    env = dict(os.environ, **execution_environment_policy(datasets))
    with tempfile.TemporaryDirectory(prefix="legal-span-section-real-input-") as temporary:
        stdout, stderr = Path(temporary) / "stdout", Path(temporary) / "stderr"
        with stdout.open("wb") as out, stderr.open("wb") as err:
            run = subprocess.run(cmd, cwd=datasets, env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, timeout=min(600, int(job.get("real_input_timeout_seconds", 300))),
                check=False)
        need(run.returncode == 0, "Real-input observation failed: " + _failure_tail(stderr))
        report = read_json(stdout)
    return report


def expected_real_input(job):
    baseline = job["real_input"]["baseline"]
    need(type(baseline["bytes"]) is int and 0 < baseline["bytes"] <= MAX_BYTES
         and Path(baseline["path"]).stat().st_size == baseline["bytes"], "Real-input baseline moved or exceeds bound")
    need(identity(baseline["path"])["sha256"] == baseline["sha256"], "Real-input baseline moved")
    expected = read_json(baseline["path"])
    need(sha(wire(expected)) == baseline["canonical_json_sha256"], "Real-input baseline content differs")
    return expected


def pack_real_input_report(report):
    """Retain the entire canonical report below native proposal file limits."""
    raw = wire(report)
    need(0 < len(raw) <= MAX_BYTES, "Complete real-input report exceeds raw byte cap")
    compressed = zlib.compress(raw, level=9)
    need(len(compressed) <= MAX_COMPRESSED_REPORT_BYTES, "Compressed full real-input report exceeds byte cap")
    encoded = base64.b64encode(compressed).decode("ascii")
    need(len(encoded) <= MAX_ENCODED_REPORT_BYTES, "Encoded full real-input report exceeds byte cap")
    return {"schema": REPORT_ENVELOPE_SCHEMA, "encoding": "zlib+base64",
        "report_bytes": len(raw), "report_sha256": sha(raw),
        "compressed_bytes": len(compressed), "compressed_sha256": sha(compressed),
        "encoded_bytes": len(encoded), "data": encoded}


def unpack_real_input_report(envelope):
    """Bound allocation before decode; authentication remains the external baseline."""
    need(type(envelope) is dict and set(envelope) == {"schema", "encoding", "report_bytes",
        "report_sha256", "compressed_bytes", "compressed_sha256", "encoded_bytes", "data"},
        "Closed complete-report envelope required")
    need(envelope["schema"] == REPORT_ENVELOPE_SCHEMA and envelope["encoding"] == "zlib+base64",
         "Wrong complete-report envelope encoding")
    for field, bound in (("report_bytes", MAX_BYTES), ("compressed_bytes", MAX_COMPRESSED_REPORT_BYTES),
                         ("encoded_bytes", MAX_ENCODED_REPORT_BYTES)):
        need(type(envelope[field]) is int and 0 < envelope[field] <= bound,
             "Complete-report envelope " + field + " exceeds finite bound")
    encoded = envelope["data"]
    need(type(encoded) is str and len(encoded) == envelope["encoded_bytes"],
         "Complete-report encoded byte count differs")
    compressed = base64.b64decode(encoded, validate=True)
    need(len(compressed) == envelope["compressed_bytes"] and sha(compressed) == envelope["compressed_sha256"],
         "Complete-report compressed identity differs")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, envelope["report_bytes"] + 1)
    need(len(raw) == envelope["report_bytes"] and decoder.eof and not decoder.unconsumed_tail
         and not decoder.unused_data, "Complete-report expansion size or stream boundary differs")
    need(sha(raw) == envelope["report_sha256"], "Complete-report raw identity differs")
    report = json.loads(raw)
    need(type(report) is dict and wire(report) == raw, "Complete-report canonical object differs")
    return report


def validate_real_input(observed, job):
    profile = job.get("real_input")
    if profile is None:
        need(observed is None, "Unexpected real-input observation")
        return
    need(type(observed) is dict and set(observed) == {"kind", "report", "report_sha256"},
         "Closed real-input observation required")
    expected = expected_real_input(job)
    expected_sha256 = sha(wire(expected))
    need(observed["kind"] == profile["spec"]["kind"] and observed["report_sha256"] == expected_sha256,
         "Real-input observation differs from independently pinned baseline")
    if observed["kind"] in (*SOURCE_ALIGNMENT_KINDS, *SOURCE_SPAN_KINDS, *PARENT_PAGE_KINDS, *SOURCE_VIEWS_KINDS):
        need(type(observed["report"]) is dict and observed["report"].get("report_sha256") == expected_sha256,
             "Complete-report envelope differs from independently pinned baseline before decode")
    report = unpack_real_input_report(observed["report"]) if observed["kind"] in (*SOURCE_ALIGNMENT_KINDS, *SOURCE_SPAN_KINDS, *PARENT_PAGE_KINDS, *SOURCE_VIEWS_KINDS) else observed["report"]
    need(wire(report) == wire(expected),
         "Real-input observation differs from independently pinned baseline")


def _bounded_failure_text(value):
    return str(value or "").encode("utf-8", errors="replace")[:MAX_FAILURE_FIELD_BYTES].decode(
        "utf-8", errors="ignore")


def _failure_tail(path):
    with Path(path).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - MAX_FAILURE_TAIL_BYTES))
        raw = stream.read(MAX_FAILURE_TAIL_BYTES)
    return raw.decode("utf-8", errors="ignore")


def refusal_payload(error):
    payload = {"valid": False, "status": "refused", "error": str(error)}
    if isinstance(error, QualificationFailure):
        payload["diagnostics"] = error.diagnostics
    return payload


def execute(datasets, job):
    environment_policy = execution_environment_policy(datasets)
    env = dict(os.environ, **environment_policy)
    with tempfile.TemporaryDirectory(prefix="legal-span-section-tests-") as temporary:
        junit = Path(temporary) / "junit.xml"
        cases, failed_cases = [], []

        def fail(message, returncode):
            raise QualificationFailure(message, {
                "schema": "legal-autoformalization-qualification-failure/v1",
                "returncode": returncode, "test_count": len(cases),
                "failed_case_count": sum(c["status"] != "passed" for c in cases),
                "failed_cases": failed_cases,
                "stdout_tail": _failure_tail(Path(temporary) / "stdout"),
                "stderr_tail": _failure_tail(Path(temporary) / "stderr"),
                "environment_policy": environment_policy})

        with (Path(temporary) / "stdout").open("wb") as out, \
                (Path(temporary) / "stderr").open("wb") as err:
            try:
                run = subprocess.run([*command(job), "--junitxml", str(junit),
                    "-o", "cache_dir=" + str(Path(temporary) / "cache")],
                    cwd=datasets, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    timeout=min(300, int(job.get("timeout_seconds", 180))), check=False)
            except subprocess.TimeoutExpired:
                out.flush()
                err.flush()
                fail("Qualification tests exceeded the declared timeout", None)
        if not junit.is_file() or junit.stat().st_size > MAX_BYTES:
            fail("Qualification did not produce bounded JUnit evidence", run.returncode)
        try:
            tree = ET.parse(junit)
        except ET.ParseError:
            fail("Qualification produced malformed JUnit evidence", run.returncode)
        for node in tree.iter("testcase"):
            issues = [child for child in node if child.tag in ("failure", "error", "skipped")]
            cases.append({"class": node.get("classname", ""), "name": node.get("name", ""),
                          "status": issues[0].tag if issues else "passed"})
            if issues and len(failed_cases) < MAX_FAILURE_CASES:
                failed_cases.append({"class": _bounded_failure_text(node.get("classname", "")),
                    "name": _bounded_failure_text(node.get("name", "")), "status": issues[0].tag,
                    "message": _bounded_failure_text(issues[0].get("message") or issues[0].text)})
        result = {"command": command(job), "returncode": run.returncode, "cases": cases,
            "ephemeral_options": {"junitxml": "temporary", "cache_dir": "temporary"},
            "environment_policy": environment_policy,
            "test_count": len(cases), "passed_count": sum(c["status"] == "passed" for c in cases),
            "stdout_sha256": identity(Path(temporary) / "stdout")["sha256"],
            "stderr_sha256": identity(Path(temporary) / "stderr")["sha256"]}
        if run.returncode != 0 or not cases or any(c["status"] != "passed" for c in cases):
            fail("Qualification tests failed, errored, or skipped", run.returncode)
        if len(cases) < int(job.get("minimum_tests", 1)):
            fail("Qualification test population shrank", run.returncode)
        if job.get("real_input"):
            report = run_real_input(datasets, job)
            result["real_input_execution"] = {"kind": job["real_input"]["spec"]["kind"],
                "report": pack_real_input_report(report) if job["real_input"]["spec"]["kind"] in (*SOURCE_ALIGNMENT_KINDS, *SOURCE_SPAN_KINDS, *PARENT_PAGE_KINDS, *SOURCE_VIEWS_KINDS) else report,
                "report_sha256": sha(wire(report))}
            validate_real_input(result["real_input_execution"], job)
        return result


def validate(result, task, sources_path):
    need(len(wire(result)) + 1 <= MAX_RESULT_BYTES, "Qualification result exceeds native file safety cap")
    sources, datasets, job, current = bindings(task, sources_path)
    need(result.get("schema") == SCHEMA and result.get("task_id") == task
         and result.get("campaign_id") == sources["campaign_id"], "Wrong qualification identity")
    need(result.get("bindings") == current, "Qualification immutable bindings differ")
    need(result.get("scope") == "source-contract behavioral qualification"
         and result.get("original_task_acceptance") == "partial"
         and result.get("legal_semantic_acceptance") is False
         and result.get("corpus_coverage_complete") is False
         and result.get("provider_dispatched") is False, "Qualification exceeds its acceptance scope")
    observed = result["test_execution"]
    validate_real_input(observed.get("real_input_execution"), job)
    need(observed["command"] == command(job) and observed["returncode"] == 0,
         "Qualification command or outcome differs")
    need(observed.get("ephemeral_options") == {"junitxml": "temporary", "cache_dir": "temporary"},
         "Qualification temporary output policy differs")
    need(observed.get("environment_policy") == execution_environment_policy(datasets),
         "Qualification import/pytest environment policy differs")
    cases = observed["cases"]
    need(isinstance(cases, list) and len(cases) >= int(job.get("minimum_tests", 1))
         and all(c["status"] == "passed" for c in cases), "Incomplete passing test population")
    need(len({(c["class"], c["name"]) for c in cases}) == len(cases), "Duplicate test identities")
    need(observed["test_count"] == len(cases) and observed["passed_count"] == len(cases),
         "Qualification test denominator differs")
    need(result.get("test_execution_sha256") == sha(wire(observed)), "Test execution digest differs")
    return {"valid": True, "task_id": task, "test_count": len(cases),
            "datasets_commit": current["datasets_commit"], "legal_semantic_acceptance": False,
            "original_task_acceptance": "partial", "scope": result["scope"]}


def output(task):
    need(task in TASKS, "Unknown source-alignment qualification task")
    path = ROOT / "data/legal_autoformalization_source_alignment" / task / "result.json"
    need(path.resolve() == path, "Qualification output path alias refused")
    return path


def materialize(task, sources_path):
    path = output(task)
    if path.exists():
        before = identity(path)
        checked = validate(read_json(path), task, sources_path)
        need(identity(path) == before, "Qualification changed during validation")
        return {**checked, "materialized": False}
    sources, datasets, job, frozen = bindings(task, sources_path)
    observed = execute(datasets, job)
    result = {"schema": SCHEMA, "campaign_id": sources["campaign_id"], "task_id": task,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source-contract behavioral qualification", "original_task_acceptance": "partial",
        "legal_semantic_acceptance": False, "corpus_coverage_complete": False,
        "provider_dispatched": False, "bindings": frozen, "test_execution": observed,
        "test_execution_sha256": sha(wire(observed))}
    checked = validate(result, task, sources_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".qualification-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(wire(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        return materialize(task, sources_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**checked, "materialized": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("materialize", "validate"))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--sources", default="config/legal_autoformalization_source_alignment_sources.json")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    sources_path = ROOT / args.sources
    if args.action == "materialize":
        checked = materialize(args.task, sources_path)
    else:
        result = read_json(output(args.task))
        checked = validate(result, args.task, sources_path)
        if args.replay:
            _, datasets, job, _ = bindings(args.task, sources_path)
            replayed = execute(datasets, job)
            need(replayed["cases"] == result["test_execution"]["cases"],
                 "Independent replay test population differs")
            checked["independent_replay_passed"] = True
    print(json.dumps(checked, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, zlib.error, subprocess.TimeoutExpired, ET.ParseError) as error:
        print(json.dumps(refusal_payload(error)), file=sys.stderr)
        raise SystemExit(1)
