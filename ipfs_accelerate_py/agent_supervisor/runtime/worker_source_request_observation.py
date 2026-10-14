"""Closed request-shape diagnostics, separate from admission or source effects."""
from __future__ import annotations

AUDIT_SCHEMA = "supervisor-worker-source-session@5"
COUNTS_SCHEMA = "worker-source-request-operation-counts@1"
FIELDS = ("request_operation", "request_source_state", "request_view")
UNKNOWN = ("unknown", "unknown", "unknown")
COMBINATIONS = frozenset({UNKNOWN,
    ("discover", "not_applicable", "not_applicable"),
    ("candidate_context", "working-candidate", "not_applicable"),
    ("edit_prepare", "working-candidate", "compact"),
    ("edit_confirm", "working-candidate", "not_applicable"),
    *(("read", state, view) for state in ("baseline", "working-candidate")
      for view in ("compact", "original"))})


def request_details(request=None):
    """Classify a protocol-validated request before peer/context admission."""
    if request is None:
        selected = UNKNOWN
    else:
        schema, query = request["schema"], request["query"]
        if schema == "supervisor-worker-source-edit-request@1":
            selected = ("edit_" + query["operation"], "working-candidate",
                        "compact" if query["operation"] == "prepare" else "not_applicable")
        elif schema in {"supervisor-worker-candidate-context-request@1",
                         "supervisor-worker-candidate-context-request@2"}:
            selected = ("candidate_context", "working-candidate", "not_applicable")
        elif schema == "supervisor-worker-source-request@3" and query["operation"] == "discover":
            selected = ("discover", "not_applicable", "not_applicable")
        else:
            selected = ("read", query.get("source_state", "baseline"), query["view"])
        if selected not in COMBINATIONS:
            raise ValueError("unsupported validated source request shape")
    return dict(zip(FIELDS, selected))


def validate_record(record):
    """Check tags against existing hash/page evidence, without raw queries."""
    selected = tuple(record.get(field) for field in FIELDS)
    if (any(type(value) is not str for value in selected) or selected not in COMBINATIONS
            or (selected == UNKNOWN) != (record.get("request_sha256") is None)):
        raise ValueError("source request attribution differs from validated envelope")
    if record.get("status") == "ok":
        operation = ("edit_" + record["source_edit_operation"] if record.get("source_edit_page") is True
                     else "candidate_context" if record.get("candidate_context_page") is True
                     else "discover" if record.get("discovery_page") is True else "read")
        if selected[0] != operation:
            raise ValueError("source request attribution differs from successful page")
    return selected


def operation_counts(records):
    """Aggregate after ordinary accounting validates bounded statuses/counts."""
    grouped = {}
    for record in records:
        key = validate_record(record)
        statuses = grouped.setdefault(key, {})
        status = record["status"]
        statuses[status] = statuses.get(status, 0) + 1
    return _counts(grouped)


def combine_operation_counts(values):
    """Combine collector-owned counts; caller keeps historical coverage explicit."""
    grouped = {}
    for value in values:
        for row in value["rows"]:
            key = tuple(row[field] for field in ("operation", "source_state", "view"))
            statuses = grouped.setdefault(key, {})
            for status, count in row["status_counts"].items():
                statuses[status] = statuses.get(status, 0) + count
    return _counts(grouped)


def _counts(grouped):
    return dict(schema=COUNTS_SCHEMA,
        rows=[dict(operation=key[0], source_state=key[1], view=key[2],
                   status_counts=dict(sorted(statuses.items())))
              for key, statuses in sorted(grouped.items())],
        accepted_requests=sum(sum(statuses.values()) for statuses in grouped.values()),
        unparsed_requests=sum(grouped.get(UNKNOWN, {}).values()),
        classification_scope="validated_request_shape_before_peer_and_context_admission",
        raw_query_fields_exported=False, admission_authority=False)
