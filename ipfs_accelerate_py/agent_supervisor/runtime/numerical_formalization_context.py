"""Source-bound numerical structure for planning; no numerical proof authority.

The datasets library owns the grammar and projections. This consumer persists
its pending obligations through the existing world and metadata lake owners,
then replays the source interpretation before a planner may use the summary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..proof.formal_verification_contracts import content_identity
from ..semantic_state.program_world_database import ProgramWorldDatabase
from .supervisor_meta_index import SupervisorMetaIndex
from .security_autoencoder_advisor import _bytes, _sha, _read, _write, _inventory, _hydrate_lakes

PROFILE = "python-numerical-structure@1"
SCHEMA = "supervisor-numerical-formalization@1"
OPERATION = "numerical_structure_observation"
RETRIEVAL_SCHEMA = "supervisor-numerical-obligation-retrieval@1"
PAGE_SCHEMA = "supervisor-numerical-obligation-page@1"
MAX_PAGE_BYTES = 8192
MAX_PAGE_ROWS = 8
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
    from ipfs_datasets_py.logic.software_verification import (
        numerical_logic, numerical_source, numerical_obligation_retrieval,
    )
    from ipfs_datasets_py.logic.software_contracts import source_unit_numerical_links
    from ipfs_datasets_py.logic.formalization.autoencoder import source_function_units, source_screening
    from ipfs_datasets_py.logic.formalization.autoencoder.security import (
        source_program_binding_384, security_formula_corpus, security_formalization_evaluation,
    )
    return {name: _sha(_read(Path(path).resolve(), 2_000_000)) for name, path in (
        ("consumer", __file__), ("grammar", numerical_source.__file__),
        ("projections", numerical_logic.__file__),
        ("obligation_retrieval", numerical_obligation_retrieval.__file__),
        ("unit_links", source_unit_numerical_links.__file__),
        ("function_units", source_function_units.__file__),
        ("unit_span_validation", security_formula_corpus.__file__),
        ("unit_span_extraction", security_formalization_evaluation.__file__),
        ("unit_source_screening", source_screening.__file__),
        ("scalar_source_guard", source_program_binding_384.__file__))}


def _function_unit_links(sources, reports):
    from ipfs_datasets_py.logic.formalization.autoencoder.source_function_units import extract_function_units
    from ipfs_datasets_py.logic.software_contracts.source_unit_numerical_links import bind_function_unit_logic

    result = {}
    for path, text in sources.items():
        raw = text.encode("utf-8")
        inventory = extract_function_units(source_bytes=raw, source_sha256=_sha(raw), source_path=path)
        result[path] = bind_function_unit_logic(source_bytes=raw, unit_inventory=inventory,
                                                numerical_report=reports[path])
    return result


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


def _summary(reports, tree, unit_links=None):
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
        pending_obligation_kinds=dict(sorted(Counter(row["kind"] for row in pending).items())),
        obligation_retrieval=dict(schema=RETRIEVAL_SCHEMA,
            api="ipfs_accelerate_py.agent_supervisor.runtime.numerical_formalization_context.retrieve_numerical_obligations",
            isolated_worker_command=("/opt/ipfs-supervisor/venv/bin/python -I "
                "/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/"
                "numerical_obligation_client.py --path <source_path>"),
            isolated_worker_usage=("Run from the allocated workspace while the coding session is active. "
                "Optional: --function-id, --kind, --limit, --max-bytes, --cursor '<next_cursor JSON>'. "
                "Responses include full pending rows; source edits invalidate retrieval."),
            requires_validated_context_receipt=True, max_page_bytes=MAX_PAGE_BYTES,
            max_page_rows=MAX_PAGE_ROWS, source_paths=sorted(reports)[:8],
            omitted_source_paths=max(0, len(reports)-8)),
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
    if unit_links is not None:
        linked = []
        for path, report in sorted(unit_links.items()):
            for link in report["links"]:
                linked.append(dict(path=path, link_report_sha256=report["report_sha256"],
                    **{key: link[key] for key in ("unit_id", "normalized_body_sha256", "function_id",
                        "status", "structural_status", "scalar_guard_status", "scalar_guard_reason",
                        "pending_obligation_count", "compound_contract_count", "assertion_requirement_count")}))
        result.update(source_unit_link_origin="deterministic_source_projection",
            source_unit_count=sum(r["unit_count"] for r in unit_links.values()),
            linked_source_unit_count=sum(r["matched_unit_count"] for r in unit_links.values()),
            unmatched_source_unit_count=sum(r["unmatched_unit_count"] for r in unit_links.values()),
            source_unit_links=linked[:4], omitted_source_unit_link_summaries=max(0, len(linked)-4),
            source_unit_links_establish_learned_coverage=False)
    # Long public names cannot turn a bounded planning constraint into an
    # unbounded prompt. Drop complete rows and account for every omission.
    while len(_bytes(result)) > 8192:
        key = next((name for name in ("pending_obligations", "per_function", "source_unit_links", "compound_contracts",
                                     "assertion_requirements")
                    if result.get(name)), None)
        if key is None:
            retrieval = result["obligation_retrieval"]
            if retrieval["source_paths"]:
                retrieval["source_paths"].pop()
                retrieval["omitted_source_paths"] += 1
                continue
            raise ValueError("numerical planning summary exceeds bound")
        result[key].pop()
        omitted = {"pending_obligations": "omitted_obligation_summaries",
            "per_function": "omitted_function_summaries",
            "source_unit_links": "omitted_source_unit_link_summaries",
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
    if found["n"] != 1 or _bytes(found["records"][0]["payload"]) != _bytes(record):
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
    unit_links = _function_unit_links(sources, reports)
    output.mkdir(parents=True)
    index = {}
    for i, (name, report) in enumerate(reports.items()):
        artifact = f"report-{i:04d}.json"
        raw = _bytes(report)
        if len(raw) > 8_000_000:
            raise ValueError("numerical report exceeds consumer bound")
        links_artifact = f"unit-links-{i:04d}.json"
        index[name] = dict(artifact=artifact, sha256=_write(output / artifact, report),
            unit_links_artifact=links_artifact,
            unit_links_sha256=_write(output / links_artifact, unit_links[name]))
    _write(output / "reports.json", index)
    tree = content_identity(dict(schema=SCHEMA, sources=source_hashes))
    summary = _summary(reports, tree, unit_links)
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
    if (_sha(raw) != receipt.get("receipt_sha256")
            or _bytes(json.loads(raw)) != _bytes({k: v for k, v in receipt.items() if k != "receipt_sha256"})):
        raise ValueError("numerical receipt changed")
    if _bytes(_inventory(output)) != _bytes(receipt["files"]) or _producers() != receipt["producers"]:
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
    unit_links = _function_unit_links(sources, reports)
    for i, (name, links) in enumerate(unit_links.items()):
        ref = index[name]
        if (set(ref) != {"artifact", "sha256", "unit_links_artifact", "unit_links_sha256"}
                or ref.get("unit_links_artifact") != f"unit-links-{i:04d}.json"):
            raise ValueError("numerical unit-link artifact nomination differs")
        raw = _read(output / ref["unit_links_artifact"])
        if _sha(raw) != ref.get("unit_links_sha256") or raw != _bytes(links):
            raise ValueError("numerical unit links differ from exact source replay")
    tree = content_identity(dict(schema=SCHEMA, sources=receipt["source_hashes"]))
    summary = _summary(reports, tree, unit_links)
    record = _record(repository, receipt["source_hashes"], receipt["producers"], index, summary)
    if (_bytes(summary) != _bytes(receipt["summary"])
            or _bytes(json.loads(_read(output / "planning-summary.json"))) != _bytes(summary)
            or _bytes(record) != _bytes(receipt["record"])
            or _bytes(_hydrate(output, record)) != _bytes(receipt["hydration"])):
        raise ValueError("numerical planning/world evidence differs")
    if (_sources(repository, receipt["source_hashes"]) != sources
            or _producers() != receipt["producers"] or _bytes(_inventory(output)) != _bytes(receipt["files"])
            or _sha(_read(output / "receipt.json")) != receipt["receipt_sha256"]):
        raise ValueError("numerical evidence changed during replay")
    return receipt


def _page_bytes(value):
    """Match the datasets pager's canonical UTF-8 byte budget."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")


def retrieve_numerical_obligations(*, repository, expected_receipt, path, cursor=None,
                                   function_id=None, kind=None, limit=8, max_bytes=8192):
    """Read a bounded page of unresolved work from the current, replayed owner.

    The receipt is supplied by the selected context owner, not inferred from a
    model-nominated path. Continuations bind to this exact receipt as well as
    the datasets report/query. No row is promoted to a discharged proof.
    """
    from ipfs_datasets_py.logic.software_verification import numerical_obligation_retrieval as pager

    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_PAGE_BYTES:
        raise ValueError("bounded numerical page byte limit required")
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE_ROWS:
        raise ValueError("bounded numerical page row limit required")
    root = Path(repository).resolve(strict=True)
    receipt = validate_numerical_context(repository=root, expected_receipt=expected_receipt)
    if type(path) is not str or path not in receipt["source_hashes"]:
        raise ValueError("numerical retrieval requires an exact selected source path")
    owner = receipt["receipt_sha256"]
    if cursor is not None:
        if (type(cursor) is not dict or set(cursor) != {"context_receipt_sha256", "cursor"}
                or cursor["context_receipt_sha256"] != owner or cursor["cursor"] is None):
            raise ValueError("numerical continuation differs from selected context receipt")
        cursor = cursor["cursor"]
    output = _output(root, receipt["output"])
    raw = _read(output / "reports.json")
    if _sha(raw) != receipt["files"]["reports.json"]["sha256"]:
        raise ValueError("numerical retrieval index changed after owner replay")
    index = json.loads(raw)
    artifact = f"report-{sorted(receipt['source_hashes']).index(path):04d}.json"
    if index[path]["artifact"] != artifact:
        raise ValueError("numerical retrieval index changed its canonical report locator")
    raw = _read(output / artifact)
    if (_sha(raw) != index[path]["sha256"]
            or _sha(raw) != receipt["files"][artifact]["sha256"]):
        raise ValueError("numerical retrieval report changed after owner replay")
    report = json.loads(raw)
    source = _sources(root, receipt["source_hashes"])[path]

    def wrap(page):
        return dict(page, schema=PAGE_SCHEMA, dataset_schema=page["schema"],
            context_receipt_sha256=owner, source_tree_id=receipt["record"]["source_tree_id"],
            max_page_bytes=max_bytes,
            next_cursor=(dict(context_receipt_sha256=owner, cursor=page["next_cursor"])
                         if page["next_cursor"] is not None else None))

    # Reserve the exact wrapper overhead for a nonterminal cursor. The final
    # page is smaller; row contents and Unicode remain untouched.
    empty = dict(schema=pager.PAGE_SCHEMA, next_cursor={})
    overhead = len(_page_bytes(wrap(empty))) - len(_page_bytes(empty))
    if max_bytes <= overhead:
        raise ValueError("numerical page cannot fit its owner binding")
    page = pager.page_numerical_obligations(report=report, source_text=source, path=path,
        cursor=cursor, function_id=function_id, kind=kind, limit=limit, max_bytes=max_bytes-overhead)
    result = wrap(page)
    if len(_page_bytes(result)) > max_bytes:
        raise ValueError("numerical page exceeds selected byte limit")
    # Fence edits during projection without claiming that a historical receipt
    # remains current after source or producer changes.
    if (_sources(root, receipt["source_hashes"])[path] != source
            or _producers() != receipt["producers"]
            or _bytes(_inventory(output)) != _bytes(receipt["files"])
            or _sha(_read(output / "receipt.json")) != owner):
        raise ValueError("numerical evidence changed during retrieval")
    return result
