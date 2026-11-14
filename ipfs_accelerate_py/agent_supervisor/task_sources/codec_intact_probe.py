"""Measure the current relational codec and one AST round-trip.

The measurement adds no extension and grants no admission. A mismatch with
the recorded codec contract returns None, which leaves the task blocked.
"""

from __future__ import annotations

import hashlib
import json

from .dependency_block_resolution import EXPECTED_CODEC_PROBE


def measure_codec_intact_probe() -> dict[str, object] | None:
    """Return the fixed probe, or None when the live owners disagree."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_CODEC_PROBE:
        return None
    return measured


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import legal_relational_norm_v1 as bridge
    from ipfs_datasets_py.logic.formalization.autoencoder.legal_relational_target_codec_v1 import (
        build_relational_target_codec_v1,
        decode_relational_candidate_v1,
        relational_target_codec_sha256,
    )
    from ipfs_datasets_py.logic.syntax_core import ast, signatures

    codec = build_relational_target_codec_v1()
    digest = relational_target_codec_sha256(codec)
    empty = decode_relational_candidate_v1(
        [], codec=codec, expected_codec_sha256=digest
    )
    zero = decode_relational_candidate_v1(
        [0], codec=codec, expected_codec_sha256=digest
    )
    request = _approval_request(ast, signatures, bridge.REQUEST_SCHEMA)
    report, raw = _prepare(bridge, request)
    replay = bridge.validate_relational_norm_v1(
        report,
        raw,
        expected_request_sha256=hashlib.sha256(raw).hexdigest(),
        expected_report_sha256=bridge.relational_norm_report_sha256(report),
    )
    counts = report["counts"]
    try:
        ast.LogicExtensionNode(
            "fixture:opaque",
            family="tdfol",
            profile="tdfol_default",
            features=("legacy.permission",),
            payload_schema="legacy.deontic/v1",
            payload="opaque-text",
        )
    except ast.AstError as exc:
        opaque_refusal = str(exc)
    else:
        raise RuntimeError("opaque payload was accepted")
    return {
        "vocabulary_size": codec["vocabulary_size"],
        "semantic_action_language": codec["semantic_action_language"],
        "grammar_constrained_generation": codec["grammar_constrained_generation"],
        "source_binding": codec["source_binding"],
        "empty_status": empty["status"],
        "empty_reason": empty["reason"],
        "zero_status": zero["status"],
        "zero_reason": zero["reason"],
        "positive_replay": replay == report,
        "typed_expression_retained": report["typed_expression"] == request["expression"],
        "exact_native_ast_roundtrip": bool(
            report["native_fragment"]["exact_native_ast_roundtrip"]
        ),
        "relational_authority_all_false": not any(report["authority"].values()),
        "admissions": report["admissions"],
        "extensions_added": 0,
        "source_bound": report["source_bound"],
        "native_build_executed": report["native_build_executed"],
        "human_reviews": report["human_reviews"],
        "node_count": counts["nodes"],
        "quantifier_count": counts["quantifiers"],
        "modal_node_count": counts["modal_nodes"],
        "predicate_occurrence_count": counts["predicate_occurrences"],
        "source_binding_refusal": _refusal(
            bridge,
            request,
            lambda item: item.__setitem__(
                "source_binding", {"source": "invented"}
            ),
        ),
        "opaque_payload_refusal": opaque_refusal,
        "reviewed_status_refusal": _refusal(
            bridge,
            request,
            lambda item: item.__setitem__("interpretation_status", "reviewed"),
        ),
    }


class _Builder:
    def __init__(self, ast, object_sort) -> None:
        self._ast = ast
        self._object = object_sort
        self._count = 0

    def _nid(self) -> str:
        self._count += 1
        return "fixture:node:" + str(self._count)

    def atom(self, name: str, *arguments: str):
        return self._ast.mk_predicate(
            self._nid(),
            name,
            [
                self._ast.mk_variable(self._nid(), argument, self._object)
                for argument in arguments
            ],
        )

    def modal(self, letter: str, body):
        kind = {"O": "obligation", "P": "permission", "F": "forbidden"}[letter]
        return self._ast.mk_extension(
            self._nid(),
            family="tdfol",
            profile="tdfol_default",
            features=("legacy." + kind, "legacy.classic_letter"),
            payload_schema="legacy.deontic/v1",
            payload={
                "kind": kind,
                "letter": letter,
                "resolution": "deontic",
                "schema_version": "legacy.deontic/v1",
            },
            children=(body,),
        )

    def binary(self, kind: str, left, right):
        return {
            "and": self._ast.mk_and,
            "or": self._ast.mk_or,
            "implies": self._ast.mk_implies,
            "iff": self._ast.mk_iff,
        }[kind](self._nid(), left, right)

    def quantify(self, body, binders):
        for name, kind in reversed(binders):
            body = (
                self._ast.mk_forall if kind == "forall" else self._ast.mk_exists
            )(self._nid(), [self._ast.Binder(name, self._object)], body)
        return body


def _approval_request(ast, signatures, schema: str) -> dict[str, object]:
    object_sort = signatures.LogicSort("Object")
    builder = _Builder(ast, object_sort)
    action = builder.binary(
        "or",
        builder.atom("Assign", "actor", "recipient"),
        builder.atom("Transfer", "actor", "recipient"),
    )
    approval = builder.atom("Approved", "approver", "actor", "recipient")
    body = builder.binary("implies", builder.modal("P", action), approval)
    declarations = (
        ("Assign", ("actor", "recipient")),
        ("Transfer", ("actor", "recipient")),
        ("Approved", ("approver", "actor", "recipient")),
    )
    root = builder.quantify(
        body,
        (
            ("actor", "forall"),
            ("recipient", "forall"),
            ("approver", "forall"),
        ),
    )
    signature = signatures.LogicSignature(
        "fixture:signature",
        family="tdfol",
        profile="tdfol_default",
        sorts=(object_sort,),
        symbols=tuple(
            signatures.declare_predicate(name, (object_sort,) * len(roles))
            for name, roles in declarations
        ),
    )
    expression = ast.TypedExpression("fixture:expression", root, signature)
    return {
        "schema": schema,
        "interpretation_status": "authored_unreviewed_declaration",
        "expression": expression.to_dict(),
        "argument_roles": [
            {"symbol": name, "roles": list(roles)} for name, roles in declarations
        ],
        "source_binding": None,
    }


def _wire(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _prepare(bridge, request: dict[str, object]):
    raw = _wire(request)
    return bridge.prepare_relational_norm_v1(
        raw, expected_request_sha256=hashlib.sha256(raw).hexdigest()
    ), raw


def _refusal(bridge, request: dict[str, object], mutate) -> str:
    mutated = json.loads(_wire(request))
    mutate(mutated)
    try:
        _prepare(bridge, mutated)
    except ValueError as exc:
        return str(exc)
    raise RuntimeError("unsupported request was accepted")
