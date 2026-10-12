"""Exact, bounded selections from an owner-captured candidate artifact cache.

The caller owns source freshness and the native cache's custody. Parent hashes
bind selections to that cache; they are not signatures or independent evidence
of its provenance. Capsules retain native content identities. Numerical rows
are replayed against the captured source by the datasets-owned pager. Nothing
here executes source, runs a model, opens an index, or promotes a proof.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import PurePosixPath
import re

from .worker_candidate_artifacts import AUTHORITY, MAX_OUTPUT_BYTES, _inputs, _sha, _wire

SCHEMA = "supervisor-worker-candidate-selection@1"
ARTIFACTS = frozenset({"catalog", "capsules", "obligations"})
SELECTION_FIELDS = frozenset({"path", "symbol_id", "function_id", "kind"})
_COMMON = ("workspace_generation", "source_state", "source_hashes", "program_paths",
    "candidate_scope_cid", "semantic_root_cid", "producer_binding_sha256", *AUTHORITY)


def _need(value, message):
    if not value:
        raise ValueError(message)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_selection(artifact, selection):
    """Validate a closed exact query without importing datasets or reading files."""
    _need(type(artifact) is str and artifact in ARTIFACTS, "unsupported candidate selection artifact")
    _need(type(selection) is dict and set(selection) == SELECTION_FIELDS,
        "exact candidate selection fields required")
    for key, value in selection.items():
        _need(value is None or (type(value) is str and value
            and len(value.encode("utf-8")) <= (4096 if key == "path" else 512)
            and not any(ord(char) < 32 or ord(char) == 127 for char in value)),
            "bounded exact candidate selection filter required")
    path = selection["path"]
    if path is not None:
        _need(path != "." and not PurePosixPath(path).is_absolute()
            and str(PurePosixPath(path)) == path and ".." not in PurePosixPath(path).parts
            and ".git" not in PurePosixPath(path).parts and "\\" not in path,
            "canonical candidate selection path required")
    if artifact == "catalog":
        _need(all(selection[key] is None for key in ("symbol_id", "function_id", "kind")),
            "catalog accepts only an optional path")
    elif artifact == "capsules":
        _need(path is not None and selection["function_id"] is selection["kind"] is None,
            "capsules require path and accept only a symbol filter")
    else:
        _need(path is not None and selection["symbol_id"] is None,
            "obligations require path and accept only function and kind filters")
    return {key: selection[key] for key in sorted(SELECTION_FIELDS)}


def _parents(artifacts, sources):
    _need(type(artifacts) is dict and set(artifacts) == {"summary", "capsules", "formal"},
        "complete owner candidate artifacts required")
    _need(all(type(raw) is bytes and raw for raw in artifacts.values())
        and sum(map(len, artifacts.values())) <= MAX_OUTPUT_BYTES,
        "bounded owner candidate artifacts required")
    values = {}
    for key, raw in artifacts.items():
        value = json.loads(raw)
        _need(type(value) is dict and _wire(value) == raw,
            "canonical owner candidate artifact required")
        _need(value.get("schema") == "supervisor-worker-candidate-" + key + "@1",
            "native candidate artifact schema differs")
        values[key] = value
    summary, capsules, formal = (values[key] for key in ("summary", "capsules", "formal"))
    paths = summary["program_paths"]
    _need(type(paths) is list, "canonical candidate program paths required")
    sources = _inputs(sources, tuple(paths), summary["workspace_generation"])
    hashes = {path: _sha(raw) for path, raw in sources.items()}
    _need(summary["source_hashes"] == hashes and summary["source_state"] == "working-candidate",
        "candidate captured source identity differs")
    _need(all(type(summary[key]) is type(expected) and summary[key] == expected
        for key, expected in AUTHORITY.items()), "candidate authority differs")
    for value in values.values():
        _need(all(_wire(value[key]) == _wire(summary[key]) for key in _COMMON),
            "candidate parent identities differ")
    _need(summary["artifacts"] == {key: dict(sha256=_sha(artifacts[key]), bytes=len(artifacts[key]))
        for key in ("capsules", "formal")}, "candidate parent artifact hash differs")
    producers = summary["producer_sources"]
    _need(type(producers) is dict and producers
        and all(type(name) is str and name and _digest(digest) for name, digest in producers.items())
        and summary["producer_binding_sha256"] == _sha(_wire(producers)),
        "candidate producer digest differs")
    from ..semantic_state.wire import cid_for_payload
    scope = dict(schema="supervisor-worker-candidate-scope@1",
        workspace_generation=summary["workspace_generation"], source_hashes=hashes,
        program_paths=paths)
    _need(summary["candidate_scope_cid"] == cid_for_payload(scope), "candidate scope identity differs")
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, validate_cid
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticCapsule
    validate_cid(summary["semantic_root_cid"])
    validate_cid(capsules["capsule_index_cid"])
    _need(type(capsules["capsules"]) is list, "native candidate capsule rows required")
    ids = set()
    for row in capsules["capsules"]:
        native = SemanticCapsule.from_dict(row)
        _need(_wire(native.to_dict()) == _wire(row), "native capsule payload differs")
        _need(native.source_slice_path in paths
            and native.source_cid == cid_for_bytes(sources[native.source_slice_path]),
            "candidate capsule source differs")
        _need(native.stable_symbol_id not in ids, "duplicate candidate capsule symbol")
        ids.add(native.stable_symbol_id)
    reports, frontiers = formal["reports"], formal["frontiers"]
    _need(type(reports) is dict and type(frontiers) is list, "native formal report population required")
    frontier_paths = [row["path"] for row in frontiers]
    _need(len(frontier_paths) == len(set(frontier_paths))
        and set(reports).isdisjoint(frontier_paths)
        and set(reports) | set(frontier_paths) == set(paths), "candidate formal scope differs")
    from ipfs_datasets_py.logic.software_verification.numerical_logic import describe_numerical_logic_profile
    _need(_wire(formal["profile"]) == _wire(describe_numerical_logic_profile()),
        "candidate native logic profile differs")
    for path, report in reports.items():
        _need(type(report) is dict and report["path"] == path
            and report["source_sha256"] == hashes[path]
            and report["report_sha256"] == _sha(_wire({key: value for key, value in report.items()
                if key != "report_sha256"})), "candidate native report identity differs")
        _need(report["unsupported_families"] == formal["profile"]["unsupported_families"],
            "candidate unsupported logic families differ")
        rows = report["pending_obligations"]
        _need(type(rows) is list and all(type(row) is dict and row["status"] == "pending" for row in rows),
            "native pending obligation rows required")
    for row in frontiers:
        _need(row["source_sha256"] == hashes[row["path"]] and row["status"] == "unsupported"
            and type(row["reason"]) is str and row["reason"], "candidate formal frontier differs")
    coverage = summary["coverage"]
    _need(type(coverage) is list and [row["path"] for row in coverage] == list(sources),
        "complete ordered candidate source coverage required")
    capsule_counts = Counter(row["source_slice_path"] for row in capsules["capsules"])
    for row in coverage:
        path = row["path"]
        _need(row["source_sha256"] == hashes[path] and row["source_bytes"] == len(sources[path])
            and row["role"] == ("program" if path in paths else "support")
            and row["capsule_count"] == capsule_counts[path], "candidate source coverage differs")
        report = reports.get(path)
        if report is not None:
            _need(row["formal_report_sha256"] == report["report_sha256"]
                and row["formal_status"] == report["summary"]["structural_status"]
                and row["pending_obligation_count"] == len(report["pending_obligations"])
                and row["families_with_artifacts"] == report["summary"]["families_with_artifacts"],
                "candidate report coverage differs")
        else:
            _need(row["pending_obligation_count"] == 0 and row["families_with_artifacts"] == []
                and row["formal_status"] == ("unsupported" if path in paths else "not_selected_support"),
                "candidate empty formal coverage differs")
            if path in paths:
                _need(row["formal_frontier"] == next(item["reason"] for item in frontiers
                    if item["path"] == path), "candidate formal frontier coverage differs")
    _need(summary["source_files"] == len(sources) and summary["program_files"] == len(paths)
        and summary["support_files"] == len(sources) - len(paths)
        and summary["capsule_count"] == len(capsules["capsules"])
        and summary["formal_report_count"] == len(reports)
        and summary["formal_frontier_count"] == len(frontiers)
        and summary["pending_obligation_count"] == sum(len(r["pending_obligations"]) for r in reports.values()),
        "candidate population counts differ")
    return summary, capsules, formal


def _replay_rows(report, source, path, function_id=None, kind=None):
    from ipfs_datasets_py.logic.software_verification.numerical_obligation_retrieval import (
        MAX_LIMIT, MAX_PAGE_BYTES, page_numerical_obligations)
    cursor, rows = None, []
    while True:
        page = page_numerical_obligations(report=report, source_text=source.decode("utf-8"),
            path=path, cursor=cursor, function_id=function_id, kind=kind,
            limit=MAX_LIMIT, max_bytes=MAX_PAGE_BYTES)
        rows.extend(page["rows"])
        if page["exhausted"]:
            _need(len(rows) == page["matched_count"], "native obligation accounting differs")
            return rows
        _need(page["returned_count"] > 0 and len(rows) <= len(report["pending_obligations"]),
            "native obligation page made no bounded progress")
        cursor = page["next_cursor"]


def _coverage(row, report):
    result = dict(row)
    if report is not None:
        result.update(function_coverage=report["coverage"]["function_coverage"],
            structural_diagnostics=report["structural"]["diagnostics"])
    return result


def _capsule_name(row):
    evidence = [facet.get("version_evidence", {}) for facet in row["metadata"].get("facets", [])]
    return dict(stable_symbol_id=row["stable_symbol_id"], capsule_cid=row["capsule_cid"],
        names=sorted({item["name"] for item in evidence if type(item.get("name")) is str}),
        kinds=sorted({item["_type"] for item in evidence if type(item.get("_type")) is str}))


def _select(*, artifacts, sources, artifact, selection):
    summary, capsules, formal = _parents(artifacts, sources)
    path = selection["path"]
    _need(path is None or path in sources, "candidate selection path outside admitted source scope")
    cover = {row["path"]: row for row in summary["coverage"]}
    all_capsules, reports = capsules["capsules"], formal["reports"]
    rows, total, outside = [], 0, 0
    status = "selected"
    if artifact == "catalog":
        names = [path] if path is not None else sorted(sources)
        for name in names:
            report = reports.get(name)
            if report is not None:
                _replay_rows(report, sources[name], name)
            rows.append(dict(_coverage(cover[name], report),
                capsules=[_capsule_name(row) for row in all_capsules if row["source_slice_path"] == name],
                obligation_kinds=dict(sorted(Counter(row["kind"] for row in
                    (report["pending_obligations"] if report is not None else [])).items()))))
        total, outside = len(rows), len(sources) - len(rows)
    elif artifact == "capsules":
        scoped = [row for row in all_capsules if row["source_slice_path"] == path]
        rows = [row for row in scoped if selection["symbol_id"] is None
            or row["stable_symbol_id"] == selection["symbol_id"]]
        total, outside = len(scoped), len(all_capsules) - len(scoped)
    else:
        report = reports.get(path)
        total = len(report["pending_obligations"]) if report is not None else 0
        outside = summary["pending_obligation_count"] - total
        if report is not None:
            rows = _replay_rows(report, sources[path], path,
                function_id=selection["function_id"], kind=selection["kind"])
    if artifact != "catalog":
        if cover[path]["role"] == "support":
            status = "not_program"
        elif not rows:
            if cover[path]["formal_status"] == "unsupported":
                status = "unsupported"
            elif artifact == "obligations" and total == 0:
                status = "no_pending_obligations"
            else:
                status = "no_matching_rows"
    result = dict(schema=SCHEMA, artifact=artifact, selection=selection, status=status,
        source_state="working-candidate", binding={key: summary[key] for key in
            ("workspace_generation", "candidate_scope_cid", "semantic_root_cid", "producer_binding_sha256")},
        counts=dict(total_count=total, matched_count=len(rows), filtered_out_count=total - len(rows),
            out_of_path_count=outside, returned_count=len(rows), remaining_count=0),
        count_scope="total_count is the selected path population before symbol/function/kind filters; "
            "out_of_path_count is the population in other admitted paths",
        rows=rows, unsupported_families=formal["profile"]["unsupported_families"],
        interpretation="Exact native advisory rows only. Empty or excluded rows do not establish safety. "
            "No source obligations were discharged and no lake build was run.", **AUTHORITY)
    result["binding"].update(source_hashes_sha256=_sha(_wire(summary["source_hashes"])),
        parent_artifact_sha256={key: _sha(raw) for key, raw in sorted(artifacts.items())})
    if artifact != "catalog":
        result["coverage"] = _coverage(cover[path], reports.get(path))
    raw = _wire(result)
    _need(len(raw) <= MAX_OUTPUT_BYTES, "candidate selection output byte bound exceeded")
    return raw


def select_candidate_artifact(*, artifacts: dict[str, bytes], sources: dict[str, bytes],
        artifact: str, selection: dict) -> bytes:
    """Return canonical selected rows; callers must recheck generation afterward."""
    selection = validate_selection(artifact, selection)
    try:
        from .supervisor_meta_index import public_replay_without_metadata
        with public_replay_without_metadata():
            return _select(artifacts=artifacts, sources=sources, artifact=artifact, selection=selection)
    except (KeyError, TypeError, UnicodeError, RecursionError, AttributeError) as exc:
        raise ValueError("malformed candidate parent artifact") from exc


__all__ = ["ARTIFACTS", "SCHEMA", "validate_selection", "select_candidate_artifact"]
