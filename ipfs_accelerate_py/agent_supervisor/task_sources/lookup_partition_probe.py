"""Measure retained lookup partitions without governing the law.

The live lookup owner keeps every supplied partition, route, empty table,
and unavailable placeholder. A nearest-scope hit stays a lexical observation.
Competing candidates stay ambiguous. The scope owner supplies no accepted
lookup and does not verify a governing use. A mismatch returns None.
"""

from __future__ import annotations

import hashlib
import json

from .dependency_block_resolution import EXPECTED_LOOKUP_PARTITION_PROBE

_TERM = "covered filer"
_DOCUMENT = "doc-A"
_SECTION = "section:near"
_METADATA = {"interpretation_status": "unreviewed_caller_declaration"}
_SCOPE_ASSIGNMENT = '"accepted_lookup": None, "governing_definition_use": None,'
_GOVERNING_REFUSAL = (
    "No approval necessity, cardinality frame, purpose, antecedent, "
    "plural identity, applicability or governing definition use is decided."
)


def measure_lookup_partition_probe() -> dict[str, object] | None:
    """Return the fixed lookup contract, or None when owners disagree."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_LOOKUP_PARTITION_PROBE:
        return None
    return measured


def _wire(value: dict) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _cid(label: str) -> str:
    from ipfs_datasets_py.utils.cid_utils import cid_for_dag_json

    return cid_for_dag_json({"authored_fixture": label})


def _symbols():
    from ipfs_datasets_py.logic.integration.reasoning import legal_ir_symbols

    return legal_ir_symbols


def _lookup():
    from ipfs_datasets_py.logic.autoformal import scoped_definition_resolution

    return scoped_definition_resolution


def _define(builder, symbol_id: str, scope_id: str, document: str = _DOCUMENT) -> None:
    symbols = _symbols()
    builder.add_definition(
        _TERM,
        symbols.LegalIRSymbolKind.DEFINED_TERM,
        symbol_id=symbol_id,
        scope_id=scope_id,
        document_id=document,
        metadata=dict(_METADATA),
    )


def _builder(document: str = _DOCUMENT):
    symbols = _symbols()
    builder = symbols.LegalIRSymbolTableBuilder(
        symbol_table_id="authored-table-" + document
    )
    document_scope = builder.add_document_scope(document)
    return builder, document_scope


def _partition(identity: str, table: dict | None, *, document: str = _DOCUMENT) -> dict:
    raw = None if table is None else _wire(table)
    return {
        "partition_id": identity,
        "release_id": "authored-release-2024",
        "edition": 2024,
        "document_id": document,
        "source_version_cid": _cid("source-" + document),
        "scope_cid": _cid("scope-" + document),
        "applicability_cid": _cid("declared-applicability"),
        "table_raw": raw,
    }


def _registry(*partitions: dict):
    rows = list(partitions)
    expected = [
        {
            "partition_id": row["partition_id"],
            "table_sha256": None if row["table_raw"] is None else _sha(row["table_raw"]),
        }
        for row in rows
    ]
    return _lookup().build_definition_lookup_registry(rows, expected)


def _query(partition_ids: list[str], *, scope_id: str) -> dict:
    symbols = _symbols()
    reference = symbols.LegalIRSymbolReference(
        reference_id="authored-reference-A",
        name=_TERM,
        symbol_kind=symbols.LegalIRSymbolKind.DEFINED_TERM,
        scope_id=scope_id,
        document_id=_DOCUMENT,
        source_node_ids=("authored-use-node",),
        span_ids=("authored-use-span",),
        metadata=dict(_METADATA),
    ).to_dict()
    return {
        "reference": reference,
        "partition_ids": list(partition_ids),
        "source_occurrence_cid": _cid("source-use-occurrence"),
        "interpretation_cid": _cid("unreviewed-interpretation"),
    }


def _resolve(registry: dict, request: dict) -> dict:
    raw = _wire(request)
    return _lookup().resolve_definition_lookup(registry, raw, _sha(raw))


def _require_unverified(result: dict) -> None:
    if result["governing_scope_verified"] is not False:
        raise RuntimeError("nearest lookup verified a governing scope")
    if result["semantic_definition_binding_verified"] is not False:
        raise RuntimeError("lookup verified a semantic binding")
    if result["admitted"] is not False or result["formalized"] is not False:
        raise RuntimeError("lookup admitted a result")
    if result["source_semantics_verified"] is not False:
        raise RuntimeError("lookup verified source semantics")
    if result["exhaustive_search_verified"] is not False:
        raise RuntimeError("lookup claimed an exhaustive search")
    if (
        result["binding_scope"]
        != "complete_supplied_tables_conservative_not_exact_search_trace"
    ):
        raise RuntimeError("lookup binding scope changed")
    if (
        result["registry_binding"]
        != "explicit_query_route_and_complete_supplied_partition_tables"
    ):
        raise RuntimeError("lookup registry binding changed")


def _ids(result: dict) -> list[str]:
    owner_result = result["owner_resolution"]
    if not isinstance(owner_result, dict):
        raise RuntimeError("lookup dropped its owner observation")
    return [str(item) for item in owner_result["symbol_ids"]]


def _path(result: dict) -> list[str]:
    owner_result = result["owner_resolution"]
    if not isinstance(owner_result, dict):
        raise RuntimeError("lookup dropped its scope path")
    return [str(item) for item in owner_result["scope_path"]]


def _scope_owner_refuses_governing_lookup() -> None:
    from ipfs_datasets_py.logic.autoformal import legal_source_norm_scope_v1 as scope
    from pathlib import Path

    authority = scope.AUTHORITY
    if authority.get("governing_use_verified") is not False:
        raise RuntimeError("scope owner verifies a governing use")
    if any(value is not False for value in authority.values()):
        raise RuntimeError("scope owner authority flag is true")
    text = Path(scope.__file__).read_text(encoding="utf-8")
    if text.count(_SCOPE_ASSIGNMENT) != 2:
        raise RuntimeError("scope owner supplies a lookup or governing use")
    if _GOVERNING_REFUSAL not in text:
        raise RuntimeError("scope owner dropped the governing-use refusal")


def _measure() -> dict[str, object]:
    symbols = _symbols()
    _scope_owner_refuses_governing_lookup()
    nearest_builder, document = _builder()
    section = nearest_builder.add_scope(
        _SECTION,
        symbols.LegalIRScopeKind.SECTION,
        parent_scope_id=document.scope_id,
        document_id=_DOCUMENT,
    )
    _define(nearest_builder, "term:far", document.scope_id)
    _define(nearest_builder, "term:near", section.scope_id)
    nearest = _resolve(
        _registry(_partition("local", nearest_builder.to_symbol_table().to_dict())),
        _query(["local"], scope_id=_SECTION),
    )
    _require_unverified(nearest)
    if nearest["status"] != "resolved" or nearest["selected_symbol_id"] != "term:near":
        raise RuntimeError("nearest lookup did not retain the nearer symbol")
    if _ids(nearest) != ["term:near"] or _path(nearest) != [
        _SECTION,
        "document:" + _DOCUMENT,
        "global",
    ]:
        raise RuntimeError("nearest lookup dropped its scope path")

    competing_builder, document = _builder()
    section = competing_builder.add_scope(
        _SECTION,
        symbols.LegalIRScopeKind.SECTION,
        parent_scope_id=document.scope_id,
        document_id=_DOCUMENT,
    )
    _define(competing_builder, "term:first", section.scope_id)
    _define(competing_builder, "term:second", section.scope_id)
    _define(competing_builder, "term:far", document.scope_id)
    competing_table = competing_builder.to_symbol_table().to_dict()
    competing_registry = _registry(_partition("local", competing_table))
    competing = _resolve(competing_registry, _query(["local"], scope_id=_SECTION))
    _require_unverified(competing)
    if (
        competing["status"] != "ambiguous"
        or competing["selected_symbol_id"] is not None
        or _ids(competing) != ["term:first", "term:second"]
    ):
        raise RuntimeError("competing candidates were collapsed")
    if "term:far" not in competing_registry["partitions"][0]["table_json"]:
        raise RuntimeError("farther candidate was dropped from the searched table")

    empty_builder, _document = _builder()
    empty_table = empty_builder.to_symbol_table().to_dict()
    if empty_table["definitions"] != []:
        raise RuntimeError("empty table gained a definition")
    empty = _resolve(
        _registry(_partition("local", empty_table)),
        _query(["local"], scope_id="document:" + _DOCUMENT),
    )
    _require_unverified(empty)
    if (
        empty["status"] != "unresolved"
        or empty["selected_symbol_id"] is not None
        or empty["partition_refs"][0]["availability"] != "available"
        or _ids(empty) != []
    ):
        raise RuntimeError("empty table was not retained as a negative lookup")

    unavailable = _resolve(
        _registry(_partition("local", None)),
        _query(["local"], scope_id="document:" + _DOCUMENT),
    )
    _require_unverified(unavailable)
    if (
        unavailable["status"] != "unavailable"
        or unavailable["selected_symbol_id"] is not None
        or unavailable["owner_resolution"] is not None
        or unavailable["partition_refs"][0]["availability"] != "unavailable"
    ):
        raise RuntimeError("unavailable partition was filled or dropped")

    route_builder, route_document = _builder()
    _define(route_builder, "term:local", route_document.scope_id)
    route = _resolve(
        _registry(
            _partition("local", route_builder.to_symbol_table().to_dict()),
            _partition("external", None, document="doc-B"),
        ),
        _query(["local", "external"], scope_id="document:" + _DOCUMENT),
    )
    _require_unverified(route)
    route_refs = [
        (row["partition_id"], row["availability"]) for row in route["partition_refs"]
    ]
    if route_refs != [("local", "available"), ("external", "unavailable")]:
        raise RuntimeError("searched route dropped a partition")
    if (
        route["status"] != "unavailable"
        or route["selected_symbol_id"] is not None
        or route["owner_resolution"]["status"] != "resolved"
        or _ids(route) != ["term:local"]
    ):
        raise RuntimeError("unavailable route admitted a local selection")

    searched = _registry(
        _partition("searched-empty", empty_table),
        _partition("searched-missing", None),
        _partition("searched-local", route_builder.to_symbol_table().to_dict()),
    )
    searched_rows = [
        (row["partition_id"], row["availability"]) for row in searched["partitions"]
    ]
    if searched_rows != [
        ("searched-empty", "available"),
        ("searched-missing", "unavailable"),
        ("searched-local", "available"),
    ]:
        raise RuntimeError("searched partition population was reduced")
    if len(searched["expected_population"]) != 3:
        raise RuntimeError("expected partition population was reduced")

    return {
        "accepted_lookup_supplied": False,
        "admitted": False,
        "binding_scope": (
            "complete_supplied_tables_conservative_not_exact_search_trace"
        ),
        "competing_candidate_ids": ["term:first", "term:second"],
        "competing_selection_made": False,
        "competing_status": "ambiguous",
        "empty_definition_count": 0,
        "empty_partition_availability": "available",
        "empty_selection_made": False,
        "empty_status": "unresolved",
        "exhaustive_search_verified": False,
        "farther_candidate_retained": True,
        "farther_symbol_selected": False,
        "formalized": False,
        "governing_definition_use_supplied": False,
        "governing_scope_verified": False,
        "governing_use_verified": False,
        "justified_binding_count": 0,
        "nearest_scope_path": [_SECTION, "document:" + _DOCUMENT, "global"],
        "nearest_selected_symbol_id": "term:near",
        "nearest_status": "resolved",
        "registry_binding": (
            "explicit_query_route_and_complete_supplied_partition_tables"
        ),
        "route_availability": ["available", "unavailable"],
        "route_local_owner_status": "resolved",
        "route_local_owner_symbol_ids": ["term:local"],
        "route_partition_ids": ["local", "external"],
        "route_selection_admitted": False,
        "route_status": "unavailable",
        "searched_availability": ["available", "unavailable", "available"],
        "searched_partition_count": 3,
        "searched_partition_ids": [
            "searched-empty",
            "searched-missing",
            "searched-local",
        ],
        "semantic_definition_binding_verified": False,
        "source_semantics_verified": False,
        "unavailable_owner_resolution_present": False,
        "unavailable_partition_availability": "unavailable",
        "unavailable_selection_made": False,
        "unavailable_status": "unavailable",
    }
