"""Read pinned source and retained metadata, then inspect a complete function bank.

This opt-in benchmark receiver consumes a diagnostic navigation order. It never
executes repository code or calls PlanCreate. Metadata comes from the exact
exports of a previously pinned native readback; no native stores reopen here.
Before/after observations cover declared files, not an atomic checkout snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .terminal_codebase_intent_corpus import _AUTHORITY, _digest, _json, _wire

SCHEMA = "terminal-codebase-source-analysis-inspection@1"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_FILES = 128


class AnalysisReceiverError(ValueError):
    """A declared input changed or does not match the independently pinned bank."""


def _need(condition, reason):
    if condition is not True:
        raise AnalysisReceiverError(reason)


def _load(raw):
    def unique(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    try:
        return _json(json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(AnalysisReceiverError("nonfinite JSON"))))
    except (UnicodeError, RecursionError, TypeError) as error:
        raise AnalysisReceiverError("bounded finite JSON artifact required") from error


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class _ObservedFiles:
    def __init__(self):
        self.values = {}
        self.total = 0

    def read(self, descriptor):
        _need(type(descriptor) is dict and set(descriptor) == {"path", "bytes", "sha256"},
              "exact independently pinned file descriptor required")
        _need(type(descriptor["path"]) is str and type(descriptor["bytes"]) is int
              and 0 <= descriptor["bytes"] <= MAX_FILE_BYTES
              and type(descriptor["sha256"]) is str
              and re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is not None,
              "bounded exact file identity required")
        path = Path(descriptor["path"])
        _need(path.is_absolute() and path.resolve(strict=True) == path,
              "canonical unaliased absolute file required")
        if str(path) in self.values:
            prior = self.values[str(path)]
            _need(_wire(prior["pin"]) == _wire(descriptor), "conflicting file aliases")
            return prior["raw"]
        _need(len(self.values) < MAX_FILES, "declared file population exceeded")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                  and before.st_size == descriptor["bytes"], "independent regular file required")
            raw = stream.read(descriptor["bytes"] + 1)
            after = os.fstat(stream.fileno())
        _need(_signature(before) == _signature(after) == _signature(path.stat())
              and path.resolve(strict=True) == path and len(raw) == descriptor["bytes"]
              and hashlib.sha256(raw).hexdigest() == descriptor["sha256"],
              "declared file changed or differs from external pin")
        self.total += len(raw)
        _need(self.total <= MAX_TOTAL_BYTES, "declared byte population exceeded")
        self.values[str(path)] = {"pin": dict(descriptor), "raw": raw,
                                 "signature": _signature(after)}
        return raw

    def finish(self):
        # Read again, not just stat: mutations between source visitation and the
        # final observation must refuse rather than return a stale success.
        pins = []
        for path, prior in sorted(self.values.items()):
            _need(Path(path).resolve(strict=True) == Path(path)
                  and _signature(Path(path).stat()) == prior["signature"],
                  "declared file changed before final observation")
            checker = _ObservedFiles()
            _need(checker.read(prior["pin"]) == prior["raw"]
                  and checker.values[path]["signature"] == prior["signature"],
                  "declared file changed during final observation")
            pins.append(prior["pin"])
        return pins


def _inspect_candidate(candidate, raw, position):
    """Visit the actual source fragment in order without importing its code."""
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_corpus import _verify_span
    binding = candidate["source_binding"]
    body = candidate["normalized_body"]
    _verify_span(raw, body.encode(), binding)
    fragment = raw[binding["start_byte"]:binding["end_byte"]]
    return {"position": position, "candidate_id": candidate["candidate_id"],
            "codebase_id": candidate["codebase_id"], "path": candidate["path"],
            "symbol": candidate["symbol"], "line": candidate["line"],
            "end_line": candidate["end_line"], "source_sha256": candidate["source_sha256"],
            "ast_sha256": candidate["ast_sha256"],
            "source_fragment_sha256": hashlib.sha256(fragment).hexdigest(),
            "source_fragment_bytes": len(fragment),
            "normalized_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "span_verified": True, "behavioral_facts_emitted": 0}


def inspect_terminal_codebase_analysis(*, frozen_envelope, ranker_result,
        metadata_report, metadata_manifest, metadata_exports, source_files,
        instruction_files, expected_corpus_sha256, expected_checkpoint_sha256,
        expected_training_receipt_sha256, query_binding, enabled=False, method="trained"):
    """Validate all inputs before ordered visits and observe them again afterward.

    All descriptor hashes and query/model identities must be frozen by the
    caller independently of this invocation. No top-k option is supported.
    ``enabled=False`` inspects the same complete bank in its baseline order.
    Successful inspection still abstains from the unresolved planning handoff.
    """
    from . import codebase_ir_metadata as metadata
    from .terminal_codebase_analysis_order import build_terminal_codebase_analysis_order

    observed = _ObservedFiles()
    envelope = _load(observed.read(frozen_envelope))
    _need(set(envelope) == {"schema", "corpus", "original_inputs"}
          and envelope["schema"] == "terminal-intent-relevance-frozen-envelope@1",
          "exact frozen corpus envelope required")
    original = envelope["original_inputs"]
    corpus = envelope["corpus"]
    result = _load(observed.read(ranker_result))
    report = _load(observed.read(metadata_report))
    manifest_raw = observed.read(metadata_manifest)
    manifest = _load(manifest_raw)
    _need("sha256:" + hashlib.sha256(manifest_raw).hexdigest() == report["manifest_sha256"],
          "retained native manifest differs from pinned readback")
    _need(_wire(metadata._report(manifest, report["manifest_sha256"])) == _wire(
        {key: value for key, value in report.items() if key != "fresh_process_readback"}),
        "retained manifest/report association differs")

    _need(type(metadata_exports) is dict and set(metadata_exports) == set(report["exports"]),
          "complete metadata export population required")
    rows = {}
    for family, descriptor in sorted(metadata_exports.items()):
        export = report["exports"][family]
        _need(type(descriptor) is dict and descriptor.get("bytes") == export["bytes"]
              and "sha256:" + str(descriptor.get("sha256")) == export["sha256"]
              and Path(descriptor.get("path", "")) == Path(report["output"]) / export["relative_path"],
              "exact retained metadata export path/identity required")
        raw = observed.read(descriptor)
        rows[family] = [_load(line) for line in raw.splitlines()]
        _need(raw == b"".join(metadata._wire(row) + b"\n" for row in rows[family]),
              "canonical complete metadata export required")

    _need(type(source_files) is list and len(source_files) == len(original["source_records"]),
          "complete live source path bindings required")
    bindings = {}
    for row in source_files:
        _need(type(row) is dict and set(row) == {"codebase_id", "path", "file"},
              "exact source path binding fields required")
        key = (row["codebase_id"], row["path"])
        _need(key not in bindings, "duplicate live source binding")
        bindings[key] = observed.read(row["file"])
    current = []
    for source in original["source_records"]:
        key = (source["codebase_id"], source["path"])
        _need(key in bindings, "missing live source binding")
        raw = bindings[key]
        current.append({**source, "source_text": raw.decode("utf-8"),
                        "source_sha256": hashlib.sha256(raw).hexdigest()})

    _need(type(instruction_files) is list and len(instruction_files) == len(original["query_records"]),
          "complete original instruction path bindings required")
    instructions = {}
    for row in instruction_files:
        _need(type(row) is dict and set(row) == {"query_id", "file"}
              and row["query_id"] not in instructions, "unique instruction path bindings required")
        instructions[row["query_id"]] = observed.read(row["file"])
    for query in original["query_records"]:
        _need(query["query_id"] in instructions
              and instructions[query["query_id"]] == query["instruction_text"].encode(),
              "whole original instruction changed or binding differs")

    order = build_terminal_codebase_analysis_order(corpus_receipt=corpus,
        original_inputs=original, ranker_receipt=result,
        expected_corpus_sha256=expected_corpus_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_training_receipt_sha256=expected_training_receipt_sha256,
        query_binding=query_binding, current_source_records=current,
        metadata_rows=rows, metadata_report=report,
        expected_metadata_report_sha256=_digest(report), enabled=enabled, method=method)
    visits = [_inspect_candidate(candidate,
        bindings[(candidate["codebase_id"], candidate["path"])], position)
        for position, candidate in enumerate(order["ordered_candidates"], 1)]
    pins = observed.finish()
    receipt = {"schema": SCHEMA, "analysis_order": order, "visits": visits,
        "observed_files": pins, "observed_file_bytes": observed.total,
        "ordered_visit_sha256": _digest(visits),
        "visit_population_sha256": _digest(sorted(
            [{key: value for key, value in row.items() if key != "position"} for row in visits],
            key=lambda row: row["candidate_id"])),
        "all_candidates_inspected": len(visits) == order["candidate_count"],
        "source_inspection_consumed_order": True, "files_observed_before_and_after": True,
        "currentness_scope": "declared files at recorded before/after observations",
        "atomic_checkout_snapshot": False, "whole_repository_coverage": False,
        "native_stores_reopened_here": False, "native_SQL_calls_here": 0,
        "new_training_calls": 0, "new_prover_calls": 0, "PlanCreate_calls": 0,
        "planner_tasks": [], "facts": [], "effects": [], "outputs": [],
        "planning_handoff": "abstained_unresolved_original_intent", **_AUTHORITY}
    receipt["receipt_sha256"] = _digest(receipt)
    return _json(receipt)


__all__ = ["AnalysisReceiverError", "inspect_terminal_codebase_analysis"]
