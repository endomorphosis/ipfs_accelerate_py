"""Source-bound numerical structure for planning; no numerical proof authority.

The datasets library owns the grammar and projections. This consumer persists
its pending obligations through the existing world and metadata lake owners,
then replays the source interpretation before a planner may use the summary.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..proof.formal_verification_contracts import content_identity
from ..semantic_state.program_world_database import ProgramWorldDatabase
from .supervisor_meta_index import SupervisorMetaIndex
from .security_autoencoder_advisor import _bytes, _sha, _read, _write, _inventory, _hydrate_lakes

PROFILE = "python-numerical-structure@1"
SCHEMA = "supervisor-numerical-formalization@1"
OPERATION = "numerical_structure_observation"
AUTHORITY = dict(proof_authority=False, completion_authority=False,
    execution_authority=False, source_semantics_verified=False,
    whole_program_semantics_verified=False, source_obligations_discharged=0,
    training_steps=0, provider_calls=0, model_inference_calls=0)


def validate_profile(value):
    if value is not None and value != PROFILE:
        raise ValueError("unsupported numerical formalization profile")
    return value


def _output(repository, output, *, fresh=False):
    path = Path(output)
    if (not path.is_absolute() or path.resolve() != path
            or path.is_relative_to(repository) or repository.is_relative_to(path)
            or (fresh and path.exists())):
        raise ValueError("canonical external numerical context state required")
    return path


def _sources(repository, source_hashes):
    if type(source_hashes) is not dict or len(source_hashes) > 64:
        raise ValueError("bounded declared numerical sources required")
    sources = {}
    for name, expected in sorted(source_hashes.items()):
        if (type(name) is not str or Path(name).is_absolute()
                or Path(name).as_posix() != name or ".." in Path(name).parts
                or not name.endswith(".py") or any(p in {".runtime", ".git"} for p in Path(name).parts)):
            raise ValueError("canonical declared Python numerical source required")
        raw = _read(repository / name, 1_000_000)
        if _sha(raw) != expected:
            raise ValueError("numerical source differs from signed declaration")
        sources[name] = raw.decode("utf-8")
    return sources


def _producers():
    from ipfs_datasets_py.logic.software_verification import numerical_logic, numerical_source
    return {name: _sha(_read(Path(path).resolve(), 2_000_000)) for name, path in (
        ("consumer", __file__), ("grammar", numerical_source.__file__),
        ("projections", numerical_logic.__file__))}


def _planning_assertion(path, row):
    # The native planner has a six-level context nesting bound. Keep scalar
    # predicate metadata together, but flatten its operand lists and reference
    # full versioned definition rows by identity instead of nesting them here.
    predicate = row["predicate"]
    return dict(path=path, **{key: value for key, value in row.items()
                if key not in {"predicate", "symbol_definitions"}},
        predicate={key: value for key, value in predicate.items()
                   if key not in {"argument_expression_ids", "keyword_names"}},
        argument_expression_ids=list(predicate["argument_expression_ids"]),
        keyword_names=list(predicate["keyword_names"]),
        symbol_definition_ids=[definition["definition_id"] for definition in row["symbol_definitions"]])


def _summary(reports, tree):
    rows, pending, contracts, assertions = [], [], [], []
    contract_count = assertion_count = 0
    for path, report in sorted(reports.items()):
        for row in report["summary"]["per_function"]:
            rows.append(dict(path=path, **row))
        for row in report["pending_obligations"]:
            pending.append(dict(path=path, **{key: row[key] for key in (
                "obligation_id", "kind", "expression_id", "function_ids", "status")}))
        contract_count += len(report.get("compound_contracts", []))
        for row in report["summary"].get("compound_contracts", []):
            contracts.append(dict(path=path, **row))
        assertion_count += len(report.get("assertion_requirements", []))
        for row in report["summary"].get("assertion_requirements", []):
            assertions.append(_planning_assertion(path, row))
    summaries = [r["summary"] for r in reports.values()]
    count = sum(r["functions"] for r in summaries)
    result = dict(schema="terminal-numerical-formalization-planning-summary@1",
        profile=PROFILE, source_tree_id=tree, files=len(reports), functions=count,
        structurally_complete_functions=sum(r["structurally_complete_functions"] for r in summaries),
        partial_or_unsupported_functions=sum(r["partial_or_unsupported_functions"] for r in summaries),
        families_with_artifacts=sorted({f for r in summaries for f in r["families_with_artifacts"]}),
        pending_obligation_count=len(pending), per_function=rows[:8],
        pending_obligations=pending[:8], omitted_function_summaries=count-min(len(rows), 8),
        omitted_obligation_summaries=max(0, len(pending)-8),
        compound_contract_count=contract_count, compound_contracts=contracts[:4],
        omitted_compound_contract_summaries=contract_count-min(len(contracts), 4),
        assertion_requirement_count=assertion_count, assertion_requirements=assertions[:4],
        omitted_assertion_requirement_summaries=assertion_count-min(len(assertions), 4),
        interpretation=("Structural syntax and pending obligations; NumPy calls remain opaque. "
            "Assertions are local requirements if reached with assertions enabled, not callee postconditions. "
            "Read source before editing."),
        autoencoder_coverage_changed=False, **AUTHORITY)
    # Long public names cannot turn a bounded planning constraint into an
    # unbounded prompt. Drop complete rows and account for every omission.
    while len(_bytes(result)) > 8192:
        key = next((name for name in ("pending_obligations", "per_function", "compound_contracts",
                                     "assertion_requirements")
                    if result[name]), None)
        if key is None:
            raise ValueError("numerical planning summary exceeds bound")
        result[key].pop()
        omitted = {"pending_obligations": "omitted_obligation_summaries",
            "per_function": "omitted_function_summaries",
            "compound_contracts": "omitted_compound_contract_summaries",
            "assertion_requirements": "omitted_assertion_requirement_summaries"}[key]
        result[omitted] += 1
    return result


def _catalogs(output, record, world_cid):
    return [("ast", str(output / "reports.json"), "sha256:" + record["reports_sha256"]),
        ("knowledge_graph", str(output / "planning-summary.json"), "sha256:" + record["summary_sha256"]),
        ("world_model", str(output / "world.duckdb"), world_cid)]


def _hydrate(output, record):
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    found = world.records_for_decision(task_id=record["task_id"], operation=OPERATION)
    if found["n"] != 1 or found["records"][0]["payload"] != record:
        raise ValueError("numerical world observation differs")
    world_cid = found["records"][0]["record_cid"]
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    linked = meta.compose_for_subject(subject_kind="tree_id", subject_ref=record["source_tree_id"])
    expected = set(_catalogs(output, record, world_cid))
    if linked["n"] != len(expected) or {(r["catalog_kind"], r["locator_ref"], r["record_ref"])
            for r in linked["linked"]} != expected:
        raise ValueError("numerical metadata links differ")
    return dict(world_record_cid=world_cid, catalog_count=len(expected),
        **_hydrate_lakes(output, record, world_cid))


def _record(repository, source_hashes, producers, index, summary):
    tree = content_identity(dict(schema=SCHEMA, sources=source_hashes))
    return dict(task_id="numerical-observation:" + tree, board="numerical-structure",
        operation=OPERATION, repository=str(repository), source_tree_id=tree,
        source_hashes=source_hashes, producers=producers, reports_sha256=_sha(_bytes(index)),
        summary_sha256=_sha(_bytes(summary)), proposal_only=True, cas_completed=False,
        admitted=False, **AUTHORITY)


def prepare_numerical_context(*, repository, source_hashes, output, profile):
    if validate_profile(profile) is None:
        raise ValueError("numerical context requires explicit profile")
    from ipfs_datasets_py.logic.software_verification.numerical_logic import formalize_numerical_source
    repository = Path(repository).resolve(strict=True)
    output = _output(repository, output, fresh=True)
    sources = _sources(repository, source_hashes)
    producers = _producers()
    reports = {name: formalize_numerical_source(text, path=name) for name, text in sources.items()}
    output.mkdir(parents=True)
    index = {}
    for i, (name, report) in enumerate(reports.items()):
        artifact = f"report-{i:04d}.json"
        raw = _bytes(report)
        if len(raw) > 8_000_000:
            raise ValueError("numerical report exceeds consumer bound")
        index[name] = dict(artifact=artifact, sha256=_write(output / artifact, report))
    _write(output / "reports.json", index)
    tree = content_identity(dict(schema=SCHEMA, sources=source_hashes))
    summary = _summary(reports, tree)
    _write(output / "planning-summary.json", summary)
    record = _record(repository, source_hashes, producers, index, summary)
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    persisted = world.persist(record)
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    for kind, locator, ref in _catalogs(output, record, persisted["record_cid"]):
        catalog = meta.register_catalog(kind=kind, locator_ref=locator,
            repository_id=content_identity({"repository": str(repository)}), tree_id=tree, project=False)
        meta.link_identity(subject_kind="tree_id", subject_ref=tree, catalog_id=catalog["catalog_id"],
            record_kind="pending_numerical_" + kind, record_ref=ref, project=False)
    projected = meta.project_ducklake()
    if persisted["ducklake"]["status"] != "projected" or projected["status"] != "projected":
        raise ValueError("numerical native DuckLake projection unavailable")
    receipt = dict(schema=SCHEMA, profile=profile, repository=str(repository), output=str(output),
        source_hashes=source_hashes, producers=producers, record=record, summary=summary,
        hydration=_hydrate(output, record), files=_inventory(output), **AUTHORITY)
    # Re-read after construction, before making these observations available.
    if _sources(repository, source_hashes) != sources or _producers() != producers:
        raise ValueError("numerical source or producer changed during preparation")
    return {**receipt, "receipt_sha256": _write(output / "receipt.json", receipt)}


def validate_numerical_context(*, repository, expected_receipt):
    from ipfs_datasets_py.logic.software_verification.numerical_logic import validate_numerical_logic_report
    repository = Path(repository).resolve(strict=True)
    receipt = expected_receipt
    if (type(receipt) is not dict or receipt.get("schema") != SCHEMA
            or receipt.get("profile") != PROFILE or receipt.get("repository") != str(repository)
            or any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in AUTHORITY.items())):
        raise ValueError("exact non-authoritative numerical receipt required")
    output = _output(repository, receipt["output"])
    raw = _read(output / "receipt.json")
    if _sha(raw) != receipt.get("receipt_sha256") or json.loads(raw) != {k: v for k, v in receipt.items() if k != "receipt_sha256"}:
        raise ValueError("numerical receipt changed")
    if _inventory(output) != receipt["files"] or _producers() != receipt["producers"]:
        raise ValueError("numerical artifacts or producers changed")
    sources = _sources(repository, receipt["source_hashes"])
    index = json.loads(_read(output / "reports.json"))
    if set(index) != set(sources):
        raise ValueError("numerical reports differ from source population")
    reports = {}
    for i, (name, text) in enumerate(sources.items()):
        ref = index[name]
        if ref.get("artifact") != f"report-{i:04d}.json":
            raise ValueError("numerical report path differs")
        raw = _read(output / ref["artifact"])
        if _sha(raw) != ref.get("sha256"):
            raise ValueError("numerical report digest differs")
        reports[name] = validate_numerical_logic_report(json.loads(raw), text, path=name)
    tree = content_identity(dict(schema=SCHEMA, sources=receipt["source_hashes"]))
    summary = _summary(reports, tree)
    record = _record(repository, receipt["source_hashes"], receipt["producers"], index, summary)
    if (summary != receipt["summary"] or json.loads(_read(output / "planning-summary.json")) != summary
            or record != receipt["record"] or _hydrate(output, record) != receipt["hydration"]):
        raise ValueError("numerical planning/world evidence differs")
    if (_sources(repository, receipt["source_hashes"]) != sources
            or _producers() != receipt["producers"] or _inventory(output) != receipt["files"]
            or _sha(_read(output / "receipt.json")) != receipt["receipt_sha256"]):
        raise ValueError("numerical evidence changed during replay")
    return receipt
