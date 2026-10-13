"""Authored public-instruction selection and byte accounting; no providers.

The selector reads only an instruction and its planner-visible summary. The
callback is supplied by an isolated qualification fixture and must call the
real client. Full rows are compared locally, but exported results retain only
hashes, sizes, and public hint references. No task/proof/token claim follows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(wire(value)).hexdigest()


def select_public_queries(instruction, summary):
    """Select reproducible inspection examples without opening full reports."""
    if type(instruction) is not str or type(summary) is not dict:
        raise ValueError("public instruction and planner-visible summary required")
    if summary.get("profile") != "python-numerical-control-structure@3":
        raise ValueError("explicit numerical control profile required")
    if not all(term in instruction for term in
               ("eigen.py", "eval.py", "np.allclose", "median time")):
        raise ValueError("this authored policy is specific to the public eigenvalue instruction")
    rows = summary.get("pending_obligations", [])
    if type(rows) is not list or len(rows) > 256:
        raise ValueError("bounded visible hint list required")
    cases, missing = [], []

    def choose(label, predicate, motivation):
        candidates = [row for row in rows if type(row) is dict and predicate(row)]
        if not candidates:
            missing.append(label)
            return
        # Preserve the producer's visible hint order; no report/body search.
        row = candidates[0]
        functions = row.get("function_ids")
        if (type(functions) is not list or not functions
                or any(type(value) is not str for value in functions)
                or any(type(row.get(key)) is not str or not row[key]
                       for key in ("path", "kind", "obligation_id"))):
            raise ValueError("selected visible hint lacks exact retrieval references")
        cases.append(dict(scenario=label, motivation=motivation,
            path=row["path"], kind=row["kind"], function_id=sorted(functions)[0],
            obligation_id=row["obligation_id"], hint_sha256=digest(row),
            proof_phase=row.get("proof_phase")))

    for path in ("eigen.py", "eval.py"):
        choose("residual:" + path,
               lambda row, path=path: row.get("path") == path
               and row.get("kind") == "eigenpair_residual",
               "The public instruction explicitly requires the eigenpair residual and names both source files.")
    if not any(row["path"] == "eigen.py" for row in cases):
        choose("target-entrypoint:eigen.py", lambda row: row.get("path") == "eigen.py",
               "The public instruction explicitly names the eigen.py entrypoint; use its first exposed hint when a residual hint is unavailable.")
    for phase in ("activation_entry", "normal_or_continue_backedge",
                  "exit_cause_and_post_state", "whole_activation_progress"):
        choose("loop-proof-phase:" + phase,
               lambda row, phase=phase: row.get("path") == "eval.py"
               and row.get("proof_phase") == phase,
               "The public instruction names the evaluation code and median runtime; this is an authored loop inspection example, not a proved performance requirement.")
    unique = {(row["path"], row["obligation_id"]) for row in cases}
    if len(unique) != len(cases):
        raise ValueError("scenario policy selected duplicate rows")
    return dict(schema="public-numerical-query-selection@1",
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        summary_sha256=digest(summary), summary_bytes=len(wire(summary)),
        selection_inputs=["public instruction", "planner-visible compact hints"],
        full_report_read_for_selection=False, source_body_read_for_selection=False,
        visible_hint_count=len(rows), selected=cases, not_selectable=missing,
        authored_selection=True, task_relevance_proven=False,
        model_calls=0, provider_calls=0, proof_authority=False)


def _query_record(query, kwargs):
    result = query(**kwargs)
    if type(result) is dict and "response" in result:
        response, stdout_bytes = result["response"], result.get("cli_stdout_bytes")
    else:
        response, stdout_bytes = result, None
    if type(response) is not dict or response.get("ok") is not True:
        raise ValueError("qualification query did not return a successful native response")
    params = dict(path=kwargs["path"], function_id=kwargs.get("function_id"),
        kind=kwargs.get("kind"), cursor=kwargs.get("cursor"),
        limit=kwargs.get("limit", 4), max_bytes=kwargs.get("max_bytes", 8192))
    if kwargs.get("obligation_id") is not None:
        params["obligation_id"] = kwargs["obligation_id"]
    # The real client copies these two fixed-width IDs into its response.
    # Reconstruct the exact request bytes, but never export either ID.
    request = dict(schema="supervisor-numerical-obligation-request@1",
        session_id=response["session_id"], request_id=response["request_id"], query=params)
    response_bytes, request_bytes = len(wire(response)), len(wire(request))
    if response_bytes > params["max_bytes"]:
        raise ValueError("complete native response exceeded its selected byte bound")
    if stdout_bytes is not None and (type(stdout_bytes) is not int
                                    or stdout_bytes != response_bytes + 1):
        raise ValueError("actual CLI stdout must match canonical response plus newline")
    page = response["page"]
    if page.get("binding", {}).get("path") != params["path"]:
        raise ValueError("native response belongs to another public source")
    rows = page["rows"]
    if page["returned_count"] != len(rows):
        raise ValueError("native row count differs")
    record = dict(request_sha256=digest(request), response_sha256=digest(response),
        request_bytes=request_bytes, response_bytes=response_bytes,
        framed_request_bytes=request_bytes + 4, framed_response_bytes=response_bytes + 4,
        cli_stdout_bytes=stdout_bytes, returned_rows=len(rows),
        matched_count=page["matched_count"],
        row_sha256=[digest(row) for row in rows])
    return page, record


def _totals(records):
    result = dict(requests=len(records), returned_row_instances=sum(r["returned_rows"] for r in records))
    for key in ("request_bytes", "response_bytes", "framed_request_bytes", "framed_response_bytes"):
        result[key] = sum(row[key] for row in records)
    result["cli_stdout_bytes"] = (sum(row["cli_stdout_bytes"] for row in records)
        if all(row["cli_stdout_bytes"] is not None for row in records) else None)
    result["canonical_request_response_bytes"] = result["request_bytes"] + result["response_bytes"]
    return result


def compare_selected_queries(manifest, query, summary, *, row_observer=None):
    """Compare old filters and exact IDs under identical visible hint input.

    Each scenario runs independently with no caching. Filter paging stops as
    soon as the nominated row occurs, not at exhaustion. An actual callback
    request must retain native response envelopes, not just inner data pages.
    """
    if manifest.get("summary_sha256") != digest(summary):
        raise ValueError("selection summary changed")
    cases, all_filter, all_exact = [], [], []
    for case in manifest["selected"]:
        filtered, chosen, cursor = [], None, None
        for _ in range(64):
            page, record = _query_record(query, dict(path=case["path"],
                function_id=case["function_id"], kind=case["kind"],
                cursor=cursor, limit=4, max_bytes=8192))
            filtered.append(record)
            chosen = next((row for row in page["rows"]
                           if row["obligation_id"] == case["obligation_id"]), None)
            if chosen is not None:
                break
            cursor = page["next_cursor"]
            if cursor is None:
                raise ValueError("visible hint was not found by its existing exact filters")
        if chosen is None:
            raise ValueError("selected row was not reached within the fixture query bound")
        page, record = _query_record(query, dict(path=case["path"],
            obligation_id=case["obligation_id"], limit=4, max_bytes=8192))
        if (len(page["rows"]) != 1 or wire(page["rows"][0]) != wire(chosen)
                or page["next_cursor"] is not None or page["matched_count"] != 1):
            raise ValueError("exact-ID fetch did not preserve the entire native pending row")
        if row_observer is not None:
            row_observer(dict(path=case["path"], obligation_id=case["obligation_id"],
                              row=json.loads(wire(chosen))))
        cases.append(dict(selection=case, full_row_sha256=digest(chosen),
            full_row_bytes=len(wire(chosen)), full_row_equal=True,
            filter_pages=filtered, exact_id_pages=[record],
            existing_filter=_totals(filtered), exact_id=_totals([record]),
            incidental_filter_row_instances=sum(r["returned_rows"] for r in filtered)-1))
        all_filter.extend(filtered)
        all_exact.append(record)
    baseline, treatment = _totals(all_filter), _totals(all_exact)
    for aggregate in (baseline, treatment):
        aggregate["common_summary_bytes"] = manifest["summary_bytes"]
        aggregate["summary_plus_canonical_request_response_bytes"] = (
            manifest["summary_bytes"] + aggregate["canonical_request_response_bytes"])
    return dict(schema="public-numerical-selector-byte-comparison@1",
        selection_sha256=digest(manifest), summary_sha256=digest(summary),
        comparison="same visible hints; existing function/kind filter versus exact-ID full-row selection",
        selected_cases=cases, existing_filter=baseline, exact_id=treatment,
        request_accounting="Exact native request reconstructed from callback arguments and matching response identifiers; IDs omitted from exported records.",
        legacy_summary_discovery_measured=False,
        batch_policy="Independent scenarios with no response cache; no hidden deduplication credit.",
        provider_calls=0, model_calls=0, native_provider_tokens=None,
        official_benchmark_score=None, proof_authority=False,
        source_obligations_discharged=0, token_savings_measured=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = select_public_queries(args.instruction.read_text(), json.loads(args.summary.read_bytes()))
    with args.output.open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
