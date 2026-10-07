"""Reviewed finite data operations retain signed admission and native replay.

These authored fixtures supply the interpretation explicitly. They do not claim
that learned inference recovered the meaning of arbitrary source instructions.
"""
from copy import deepcopy
import hashlib
import json
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning as symbolic
from ipfs_accelerate_py.agent_supervisor.planning.intent_data_transform import (
    CONTRACT_SCHEMA, CORRESPONDENCE_POLICY, FALSE, SCHEMA, validate_reviewed_data_transform,
)
from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import (
    build_intent_planning_materials,
)
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import (
    build_intent_requirement_contract, validate_intent_requirement_contract,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _data_case(tmp_path, *, mode="rename", public_profile=False):
    """Fresh public fixture plus signed proposal, with no open database owner."""
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
    from ipfs_datasets_py.logic.intent_ir.schema import (
        IntentIRDocument, IntentKind, IntentModality, IntentStatement, NodeGrounding,
        ReviewStatus, SourceRef, SourceSpan, StatementKind,
    )
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    rows = [{"id": "first", "name": "éλ", "enabled": True, "note": None},
            {"id": "second", "name": "Beta", "enabled": False, "note": "present"}]
    expected = [{**row, "label": row["name"]} for row in rows]
    if mode == "rename":
        for row in expected:
            del row["name"]
    input_bytes = b"".join(_wire(row) + b"\n" for row in rows)
    source = (f"Create output.jsonl by {mode} of each input.jsonl record's name field to label. "
              "Preserve record order, count and all unrelated fields. Reject target collisions.\n")
    instruction_path = ".supervisor-instruction.md" if public_profile else "instruction.txt"
    validation_key = "public-structural-smoke" if public_profile else "public-record-correspondence"
    (repository / instruction_path).write_text(source)
    (repository / "input.jsonl").write_bytes(input_bytes)
    check = ("import json\nfrom pathlib import Path\n"
             "rows = [json.loads(line) for line in Path('output.jsonl').read_text().splitlines()]\n"
             f"assert rows == {expected!r}\n"
             "assert all(type(row['enabled']) is bool for row in rows)\n"
             "assert rows[0]['note'] is None and type(rows[1]['note']) is str\n")
    (repository / "public_check.py").write_text(check)
    task_profile = None
    if public_profile:
        from ipfs_accelerate_py.agent_supervisor.runtime.terminal_task_profile import (
            DATA_SCHEMA, PROFILE, SMOKE, instruction_sha256, task_profile_bytes, task_profile_smoke,
        )
        task_profile = {"schema": DATA_SCHEMA, "instruction_sha256": instruction_sha256(source),
            "input_paths": ["input.jsonl", "public_check.py"],
            "data_inputs": [{"path": "input.jsonl", "media_type": "application/x-ndjson"}],
            "outputs": [{"path": "output.jsonl", "effect": "create", "media_type": "application/x-ndjson"}]}
        (repository / PROFILE).write_bytes(task_profile_bytes(task_profile))
        (repository / SMOKE).write_text(task_profile_smoke(task_profile))
    for args in (("init", "-q"), ("add", "."),
            ("-c", "user.name=Authored data qualification", "-c", "user.email=data@example.invalid",
             "commit", "-qm", "Authored finite record fixture")):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    baseline = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    with (repository / ".git" / "info" / "exclude").open("a") as stream:
        stream.write("\n.runtime/\n")
    profile, lifecycle = tmp_path / "profile", tmp_path / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    raw = source.encode()
    digest = hashlib.sha256(raw).hexdigest()
    source_ref = SourceRef(ref_id="public-instruction", source_uri="fixture:finite-data", source_id=digest,
        source_revision=digest, content_sha256=digest, span=SourceSpan(0, len(source)),
        review_status=ReviewStatus.MACHINE_EXTRACTED)
    statement = IntentStatement(statement_id="create-output", kind=StatementKind.GOAL,
        modality=IntentModality.REQUIRED, normalized_text=source.strip(),
        source_ref_ids=(source_ref.ref_id,), predicate="create", arguments=("agent", "output.jsonl"),
        confidence=0, grounding=NodeGrounding.INFERRED, review_status=ReviewStatus.MACHINE_EXTRACTED)
    document = IntentIRDocument(document_id="reviewed-data:" + digest, title="Reviewed finite data operation",
        intent_kind=IntentKind.DECLARATIVE, sources=(source_ref,), statements=(statement,))
    document.validate()
    report = {"schema": "intent-reviewed-source-report@1", "source_sha256": digest,
        "source_bytes": len(raw), "source_characters": len(source),
        "producer": {"name": "authored-finite-data-fixture", "revision": "1"},
        "interpretation_status": "reviewed_candidate",
        "units": [{"unit_id": "unit:data", "start_char": 0, "end_char": len(source),
            "start_byte": 0, "end_byte": len(raw), "text": source, "sha256": digest,
            "disposition": "interpreted_candidate", "reason": "explicit_reviewed_fixture"}],
        "candidates": [{"unit_id": "unit:data", "candidate_intent_ir": document.to_dict()}],
        "proof_authority": False, "execution_authority": False,
        "completion_authority": False, "source_semantics_verified": False}
    report["report_sha256"] = hashlib.sha256(_wire(report)).hexdigest()
    ledger = build_intent_requirement_ledger(source, source_report=report,
        source_identity={"path": instruction_path, "revision": baseline})
    requirement = ledger["requirements"][0]
    output = {"path": "output.jsonl", "effect": "create", "media_type": "application/x-ndjson"}
    native = document.to_dict()["statements"][0]
    operation = {"operation_id": "operation:data", "task_key": "DATA-TASK",
        "matchers": [{"requirement_id": requirement["requirement_id"],
            "native_document_sha256": requirement["native_document_sha256"],
            **{key: native[key] for key in ("statement_id", "predicate", "arguments", "modality")}}],
        "outputs": [output], "validation_keys": [validation_key], "dependency_operation_ids": []}
    operations = {"schema": "intent-symbolic-operation-contract@1", "ledger_sha256": ledger["ledger_sha256"],
        "review_ref": "review:authored-finite-data@1", "interpretation_scope": "administrative_requirement_task_coverage",
        "operations": [operation], "semantic_alignment_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False}
    selector = {"schema": SCHEMA, "review_ref": "review:authored-data-selector@1",
        "operation_id": operation["operation_id"], "input_path": "input.jsonl", "output_path": "output.jsonl",
        "mode": mode, "source_field": "name", "target_field": "label",
        "correspondence_policy": CORRESPONDENCE_POLICY, "validation_key": validation_key, **FALSE}
    contract = build_intent_requirement_contract(source_path=instruction_path, ledger=ledger,
        requirements=[{"requirement_id": requirement["requirement_id"], "outputs": [output],
            "validation_keys": [validation_key], "dependency_requirement_ids": []}],
        source_text=source, symbolic_operations=operations, reviewed_data_transform=selector)
    sources = local._sources(repository, [instruction_path, "input.jsonl", "public_check.py",
        *([PROFILE, SMOKE] if public_profile else [])])
    roots = {"request_cid": local.content_identity({"prompt": source}),
        "scan_cid": local.content_identity({"sources": sources}),
        "program_root": local.content_identity({"git_tree": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"], text=True).strip()})}
    spec = {"task_key": "DATA-TASK", "scope_paths": [*sorted(sources), "output.jsonl"],
        "dependencies": [], "outputs": [output],
        "validations": [{"validation_key": validation_key,
            "argv": ["python3", "-I", "-B", "public_check.py"], "cwd": ".", "expected_exit_codes": [0],
            "policy_cid": local.content_identity(local.LOCAL_POLICY)}],
        "acceptance": [{"criterion_key": "exact-public-record-correspondence",
            "criterion": "The public check preserves ordered records, unrelated values and JSON types.",
            "validation_keys": [validation_key], "evidence_cids": []}]}
    if public_profile:
        from ipfs_accelerate_py.agent_supervisor.runtime.terminal_task_profile import task_profile_spec
        spec = task_profile_spec(task_profile, policy_cid=local.content_identity(local.LOCAL_POLICY))
        # The producer owns the exact administrative task key, not the fixture.
        contract["symbolic_operations"]["operations"][0]["task_key"] = spec["task_key"]
    manifest = local.author_local_benchmark_manifest(repository=repository, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=[spec], planning_roots=roots, intent_requirements=contract)
    proposed = symbolic.build_intent_symbolic_plan(contract, manifest=manifest)
    return dict(repository=repository, manifest=manifest, contract=contract, proposed=proposed,
        profile=profile, lifecycle=lifecycle, baseline_commit=baseline, input_bytes=input_bytes,
        expected_output=expected, public_check_bytes=check.encode(), source=source, task_profile=task_profile)


def test_reviewed_data_selector_survives_signed_native_and_public_replay(tmp_path):
    from benchmarks.agent_supervisor.container_coding.terminal_indexed_preparation import _planning_strategy
    case = _data_case(tmp_path)
    contract, manifest, proposed = (case[key] for key in ("contract", "manifest", "proposed"))
    assert contract["schema"] == CONTRACT_SCHEMA
    assert _planning_strategy(contract) == "intent_symbolic"
    assert proposed["receipt"]["provider_calls"] == proposed["receipt"]["observed_facts_supplied"] == 0
    materials = build_intent_planning_materials(contract, manifest=manifest)
    assert materials.current_facts == () and materials.source_applicability is None
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        task = intent.get_task(materialized["task_cids"][0])
        pending, _, _, _ = local._contract(task["body"], task["task_cid"])
        assert pending["intent_plan"]["schema"] == "supervisor-local-intent-plan@2"
        assert pending["intent_plan"]["symbolic_planning"] == proposed["receipt"]
        reference = intent.get_plan(materialized["plan_id"])["body"]["local_planning_receipt_ref"]
        assert local.load_local_planning_receipt(reference, manifest=manifest) == admission["receipt"]
        assert task["status"] == "ready"
    assert local._verify_intent_requirement_text(manifest["payload"], case["source"]) == contract
    assert not (case["repository"] / "output.jsonl").exists()


@pytest.mark.parametrize("field,value", [
    ("schema", "reviewed-ndjson-data-transform@2"), ("review_ref", ""),
    ("operation_id", "operation:foreign"), ("validation_key", "other"),
    ("mode", "merge"), ("source_field", "nested.name"), ("target_field", "name"),
    ("source_field", "x" * 65), ("source_field", "é"),
    ("input_path", "output.jsonl"), ("output_path", "../outside.jsonl"),
    ("input_path", "input.json"), ("correspondence_policy", "unordered"),
    *((key, True) for key in FALSE), ("caller_checked", True),
])
def test_malformed_or_authority_granting_selector_is_rejected(tmp_path, field, value):
    case = _data_case(tmp_path)
    changed = deepcopy(case["contract"])
    changed["reviewed_data_transform"][field] = value
    with pytest.raises(ValueError):
        validate_intent_requirement_contract(changed, source_text=case["source"])


@pytest.mark.parametrize("mutation", ["missing_input", "input_outside_scope", "output_exists",
    "extra_creation", "changed_validation", "extra_output", "extra_task"])
def test_selector_cannot_escape_independent_manifest_scope(tmp_path, mutation):
    case = _data_case(tmp_path)
    manifest = deepcopy(case["manifest"])
    payload = manifest["payload"]
    task = payload["tasks"][0]
    if mutation == "missing_input":
        del payload["sources"]["input.jsonl"]
    elif mutation == "input_outside_scope":
        task["scope_paths"].remove("input.jsonl")
    elif mutation == "output_exists":
        payload["sources"]["output.jsonl"] = deepcopy(payload["sources"]["input.jsonl"])
    elif mutation == "extra_creation":
        payload["created_outputs"].append("second.jsonl")
    elif mutation == "changed_validation":
        task["validations"][0]["validation_key"] = "other"
    elif mutation == "extra_output":
        task["outputs"].append({"path": "second.jsonl", "effect": "create", "media_type": "application/x-ndjson"})
    else:
        payload["tasks"].append(deepcopy(task))
    with pytest.raises(ValueError):
        build_intent_planning_materials(case["contract"], manifest=manifest)
    with pytest.raises(local.LocalPlanningError):
        local._verify_intent_requirement_text(payload, case["source"])


def test_unsigned_selector_and_header_nomination_are_not_data_authority(tmp_path):
    case = _data_case(tmp_path)
    contract = deepcopy(case["contract"])
    contract["schema"] = "intent-plan-requirement-contract@2"
    with pytest.raises(ValueError, match="exact intent contract"):
        validate_intent_requirement_contract(contract)
    contract.pop("reviewed_data_transform")
    assert validate_intent_requirement_contract(contract)["schema"] == "intent-plan-requirement-contract@2"
    with pytest.raises(ValueError, match="requires explicit v3"):
        build_intent_planning_materials(case["contract"], manifest=case["manifest"],
            source_applicability_nomination={"caller_checked": True})
    with pytest.raises(ValueError, match="differs from signed manifest"):
        build_intent_planning_materials(contract, manifest=case["manifest"])


def test_signed_source_drift_refuses_reviewed_data_admission(tmp_path):
    case = _data_case(tmp_path)
    (case["repository"] / "input.jsonl").write_bytes(b'{"name":"changed"}\n')
    with pytest.raises(local.LocalPlanningError):
        local.admit_local_benchmark_plan(graph=case["proposed"]["graph"], manifest=case["manifest"],
            requirement_bindings=case["proposed"]["requirement_bindings"])


def test_data_profile_keeps_structural_validation_scope_with_signed_selector(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.terminal_source_partition import terminal_profile_partition
    case = _data_case(tmp_path, public_profile=True)
    manifest = case["manifest"]["payload"]
    partition = terminal_profile_partition(repository=case["repository"], manifest=manifest)
    assert partition.program_paths == ("public_check.py",)
    assert ("input.jsonl", "task_data", manifest["sources"]["input.jsonl"]["sha256"]) in partition.support_hashes
    assert manifest["tasks"][0]["validations"][0]["validation_key"] == "public-structural-smoke"
    assert "benchmark correctness remains unverified" in manifest["tasks"][0]["acceptance"][0]["criterion"]
    admission = local.admit_local_benchmark_plan(graph=case["proposed"]["graph"], manifest=case["manifest"],
        requirement_bindings=case["proposed"]["requirement_bindings"])
    assert local.verify_local_benchmark_admission(admission)["receipt"]["completion_authority"] is False
