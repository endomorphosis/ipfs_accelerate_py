"""Bounded, reversible model views of a closed Python arithmetic subset.

This is a syntax and edit-translation candidate, never executable model code
or a claim of runtime purity, semantic equivalence, proof, or qualification.
The private artifact retains exact native source. The public packet contains
an aliased view and short, hash-bound single-line expression edit handles.
No generic inverse whole-file diff, source writes, or admission are provided.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import keyword
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_bytes,
    cid_for_dag_json,
)

VIEW_SCHEMA = "supervisor-python-code-view@1"
EDIT_SCHEMA = "supervisor-python-code-view-edit@1"
VIEW_REVISION = "python-arithmetic-source-view@1"
MAX_SOURCE_BYTES = 64 * 1024
MAX_TOKENS = 8192
MAX_FUNCTIONS = 16
MAX_AST_NODES = 4096
MAX_AST_DEPTH = 40
MAX_EXPRESSION_BYTES = 8 * 1024
MAX_INTEGER_DIGITS = 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_NODES = 100_000
MAX_ARTIFACT_DEPTH = 48

_NO_AUTHORITY = {
    "source_write_authority": False,
    "execution_authority": False,
    "proof_authority": False,
    "qualification_authority": False,
    "completion_authority": False,
}
_BINDING_FIELDS = {"path", "repository_id", "tree_id", "task_id"}
_VIEW_FIELDS = {
    "schema", "revision", "bindings", "original_source", "source_sha256",
    "source_cid", "compact_source", "compact_sha256", "frontend",
    "functions", "token_map", "slots", "invariants", "authority",
}
_EDIT_FIELDS = {
    "schema", "revision", "bindings", "view_id", "slot_id", "before_source",
    "after_source", "before_sha256", "after_sha256", "before_cid", "after_cid",
    "expression_before_sha256", "model_expression", "native_expression",
    "source_span", "authority",
}
_IGNORED_TOKENS = {
    tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT,
    tokenize.NEWLINE, tokenize.NL,
}
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
_UNARYOPS = (ast.UAdd, ast.USub)


class PythonCodeViewError(ValueError):
    """A closed source-view, identity, grammar, or resource contract failed."""


def _fail(reason: str) -> None:
    raise PythonCodeViewError(reason)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _raw(value: Any, limit: int, label: str) -> bytes:
    if type(value) is not str:
        _fail(f"{label} must be an exact string")
    if len(value) > limit:
        _fail(f"{label} exceeds byte limit")
    try:
        data = value.encode("utf-8")
    except UnicodeError as exc:
        raise PythonCodeViewError(f"{label} is not UTF-8") from exc
    if len(data) > limit:
        _fail(f"{label} exceeds byte limit")
    return data


def _json_string_size(value: str) -> int:
    raw = _raw(value, MAX_ARTIFACT_BYTES, "artifact string")
    size = len(raw) + 2
    for char in value:
        if char in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}:
            size += 1
        elif ord(char) < 32:
            size += 5
        if size > MAX_ARTIFACT_BYTES:
            _fail("artifact exceeds byte limit")
    return size


def _profile(value: Any) -> None:
    pending = [(value, 0)]
    seen = 0
    wire_bytes = 0
    while pending:
        item, depth = pending.pop()
        seen += 1
        if seen > MAX_ARTIFACT_NODES or depth > MAX_ARTIFACT_DEPTH:
            _fail("artifact structure exceeds bounds")
        if type(item) is dict:
            if len(item) > MAX_ARTIFACT_NODES - seen - len(pending):
                _fail("artifact structure exceeds bounds")
            if any(type(key) is not str for key in item):
                _fail("artifact keys must be exact strings")
            wire_bytes += 2 + max(0, len(item) - 1) + len(item)
            for key in item:
                wire_bytes += _json_string_size(key)
                if wire_bytes > MAX_ARTIFACT_BYTES:
                    _fail("artifact exceeds byte limit")
            children = item.values()
        elif type(item) is list:
            if len(item) > MAX_ARTIFACT_NODES - seen - len(pending):
                _fail("artifact structure exceeds bounds")
            wire_bytes += 2 + max(0, len(item) - 1)
            children = item
        elif type(item) is str:
            wire_bytes += _json_string_size(item)
            children = ()
        elif type(item) is int:
            if item.bit_length() > MAX_INTEGER_DIGITS * 4:
                _fail("artifact integer exceeds digit limit")
            text = str(item)
            if len(text.lstrip("-")) > MAX_INTEGER_DIGITS:
                _fail("artifact integer exceeds digit limit")
            wire_bytes += len(text)
            children = ()
        elif type(item) is bool:
            wire_bytes += 4 if item else 5
            children = ()
        elif item is None:
            wire_bytes += 4
            children = ()
        else:
            _fail("artifact contains a non-JSON value")
        if wire_bytes > MAX_ARTIFACT_BYTES:
            _fail("artifact exceeds byte limit")
        if not children:
            continue
        if len(children) > MAX_ARTIFACT_NODES - seen - len(pending):
            _fail("artifact structure exceeds bounds")
        pending.extend((child, depth + 1) for child in children)


def _canonical(value: Any) -> bytes:
    _profile(value)
    try:
        data = canonical_dag_json_bytes(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise PythonCodeViewError("artifact is not canonical JSON") from exc
    if len(data) > MAX_ARTIFACT_BYTES:
        _fail("artifact exceeds byte limit")
    return data


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("artifact has duplicate JSON keys")
        result[key] = value
    return result


def _read_json(text: str) -> dict[str, Any]:
    _raw(text, MAX_ARTIFACT_BYTES, "artifact JSON")
    try:
        value = json.loads(text, object_pairs_hook=_json_pairs)
    except (ValueError, RecursionError) as exc:
        raise PythonCodeViewError("invalid artifact JSON") from exc
    _profile(value)
    if type(value) is not dict:
        _fail("artifact must be an exact dictionary")
    return value


def _bindings(value: Any) -> dict[str, str]:
    if type(value) is not dict or set(value) != _BINDING_FIELDS:
        _fail("invalid source bindings")
    for key, item in value.items():
        _raw(item, 1024 if key == "path" else 512, f"binding {key}")
        if not item or any(char.isspace() for char in item):
            _fail("binding must be a nonempty whitespace-free identifier")
    path = value["path"]
    if path.startswith("/") or "\\" in path or any(
        part in {"", ".", ".."} for part in path.split("/")
    ):
        _fail("path must be a canonical relative path")
    return dict(value)


def _owner_types() -> tuple[Any, Any]:
    # Never silently load the unrelated HACC editable logic install.
    from ipfs_datasets_py.logic.software_contracts import (
        ast_ir,
        content,
        python_frontend,
    )

    expected = Path(__file__).resolve().parents[4] / "ipfs_datasets" / "ipfs_datasets_py"
    for module in (ast_ir, content, python_frontend):
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(expected.resolve()):
            _fail("software-contract frontend is outside the pinned workspace tree")
    return python_frontend.PythonASTExtractor, ast_ir.SourceSpan


def _line_table(source: str) -> tuple[list[str], list[int]]:
    # tokenize columns count characters; AST columns count UTF-8 bytes.
    lines = source.split("\n")
    offsets: list[int] = []
    offset = 0
    for index, line in enumerate(lines):
        offsets.append(offset)
        offset += len(line.encode("utf-8")) + (1 if index < len(lines) - 1 else 0)
    return lines, offsets


def _token_offset(position: tuple[int, int], lines: list[str], offsets: list[int]) -> int:
    row, column = position
    if row == len(lines) + 1 and column == 0:
        return offsets[-1] + len(lines[-1].encode("utf-8"))
    if not 1 <= row <= len(lines) or not 0 <= column <= len(lines[row - 1]):
        _fail("token position is outside source")
    return offsets[row - 1] + len(lines[row - 1][:column].encode("utf-8"))


def _tokens(source: str) -> list[tokenize.TokenInfo]:
    result: list[tokenize.TokenInfo] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            result.append(token)
            if len(result) > MAX_TOKENS:
                _fail("source token limit exceeded")
            if token.type == tokenize.ERRORTOKEN and not token.string.isspace():
                _fail("source contains an invalid token")
            if token.type == tokenize.NAME and not token.string.isascii():
                _fail("only ASCII Python binding names are supported")
            if token.type == tokenize.NUMBER and len(token.string.replace("_", "")) > MAX_INTEGER_DIGITS:
                _fail("integer literal digit limit exceeded")
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise PythonCodeViewError("source tokenization failed") from exc
    return result


def _tree(source: str, *, expression: bool = False) -> ast.AST:
    try:
        tree = ast.parse(source, mode="eval" if expression else "exec", type_comments=True)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise PythonCodeViewError("source syntax is unsupported") from exc
    stack = [(tree, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_AST_DEPTH:
            _fail("source AST limit exceeded")
        for child in ast.iter_child_nodes(node):
            if count + len(stack) >= MAX_AST_NODES:
                _fail("source AST limit exceeded")
            stack.append((child, depth + 1))
    return tree


def _name(name: str) -> None:
    if not name.isascii() or not name.isidentifier() or keyword.iskeyword(name):
        _fail("only ASCII Python binding names are supported")


def _expression(node: ast.AST, available: set[str]) -> None:
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        _name(node.id)
        if node.id not in available:
            _fail("expression references an unavailable binding")
    elif isinstance(node, ast.Constant) and type(node.value) is int:
        return
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARYOPS):
        _expression(node.operand, available)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, _BINOPS):
        _expression(node.left, available)
        _expression(node.right, available)
    else:
        _fail("expression is outside the closed integer arithmetic grammar")


def _docstring(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and type(node.value.value) is str


def _functions(tree: ast.AST) -> list[dict[str, Any]]:
    if not isinstance(tree, ast.Module) or tree.type_ignores:
        _fail("only an ordinary Python module is supported")
    nodes = list(tree.body)
    if nodes and _docstring(nodes[0]):
        nodes.pop(0)
    if not nodes or len(nodes) > MAX_FUNCTIONS:
        _fail("source must contain one to sixteen arithmetic functions")
    names: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, function in enumerate(nodes):
        if not isinstance(function, ast.FunctionDef):
            _fail("only ordinary top-level functions are supported")
        _name(function.name)
        if function.name in names:
            _fail("function names must be unique")
        names.add(function.name)
        args = function.args
        if (
            function.decorator_list or function.returns is not None or function.type_comment
            or getattr(function, "type_params", []) or args.posonlyargs or args.vararg
            or args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults
            or any(arg.annotation is not None or arg.type_comment for arg in args.args)
        ):
            _fail("function interface contains unsupported syntax")
        parameters = [arg.arg for arg in args.args]
        for parameter in parameters:
            _name(parameter)
        if len(set(parameters)) != len(parameters):
            _fail("parameter bindings must be unique")
        available = set(parameters)
        statements = list(function.body)
        if statements and _docstring(statements[0]):
            statements.pop(0)
        if not statements or not isinstance(statements[-1], ast.Return) or statements[-1].value is None:
            _fail("function must end with a value return")
        local_names: list[str] = []
        slots: list[tuple[ast.AST, ast.AST, str, str | None, list[str]]] = []
        binding_order = list(parameters)
        for statement in statements[:-1]:
            if (
                not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name) or statement.type_comment
            ):
                _fail("only fresh single-target local assignments are supported")
            target = statement.targets[0].id
            _name(target)
            if target in available or target in names:
                _fail("local binding must be fresh")
            _expression(statement.value, available)
            slots.append((statement.value, statement, "assignment", target, list(binding_order)))
            available.add(target)
            binding_order.append(target)
            local_names.append(target)
        returned = statements[-1]
        _expression(returned.value, available)
        slots.append((returned.value, returned, "return", None, list(binding_order)))
        result.append({
            "node": function, "function_id": f"f{index}", "parameters": parameters,
            "locals": local_names, "slots": slots,
        })
    return result


def _span(node: ast.AST, offsets: list[int], span_type: Any) -> dict[str, int]:
    return span_type(
        start_byte=offsets[node.lineno - 1] + node.col_offset,
        end_byte=offsets[node.end_lineno - 1] + node.end_col_offset,
        start_line=node.lineno, start_column=node.col_offset,
        end_line=node.end_lineno, end_column=node.end_col_offset,
    ).to_dict()


def _alias_candidate(index: int) -> str:
    result = ""
    while True:
        index, remainder = divmod(index, 26)
        result = chr(97 + remainder) + result
        if index == 0:
            return result
        index -= 1


def _aliases(names: list[str], reserved: set[str]) -> list[dict[str, str]]:
    used = set(reserved)
    counter = 0
    result: list[dict[str, str]] = []
    for original in names:
        while True:
            alias = _alias_candidate(counter)
            counter += 1
            if alias not in used and not keyword.iskeyword(alias):
                break
        if len(alias) >= len(original):
            alias = original
        used.add(alias)
        result.append({"original": original, "alias": alias})
    return result


def _signature(tokens: list[tokenize.TokenInfo]) -> list[list[Any]]:
    return [[token.type, token.string] for token in tokens if token.type not in _IGNORED_TOKENS]


def _pair_separator(left: tokenize.TokenInfo, right: tokenize.TokenInfo, left_text: str, right_text: str) -> str:
    expected = [[left.type, left_text], [right.type, right_text]]
    for separator in ("", " "):
        try:
            if _signature(_tokens(left_text + separator + right_text)) == expected:
                return separator
        except PythonCodeViewError:
            continue
    return " "


def _splice(raw: bytes, edits: list[tuple[int, int, bytes]]) -> bytes:
    edits.sort(key=lambda item: (item[0], item[1]))
    output: list[bytes] = []
    previous = 0
    for start, end, value in edits:
        if start < previous or end < start or end > len(raw):
            _fail("source replacement spans overlap or escape")
        output.extend((raw[previous:start], value))
        previous = end
    output.append(raw[previous:])
    return b"".join(output)


def _inverse_tree(tree: ast.AST, functions: list[dict[str, Any]]) -> str:
    cloned = copy.deepcopy(tree)
    actual_functions = [node for node in cloned.body if isinstance(node, ast.FunctionDef)]
    if len(actual_functions) != len(functions):
        _fail("compact view changed function inventory")
    for function, record in zip(actual_functions, functions, strict=True):
        inverse = {item["alias"]: item["original"] for item in record["aliases"]}
        for node in ast.walk(function):
            if isinstance(node, ast.arg):
                if node.arg not in inverse:
                    _fail("compact view changed parameter inventory")
                node.arg = inverse[node.arg]
            elif isinstance(node, ast.Name):
                if node.id not in inverse:
                    _fail("compact view introduced a binding")
                node.id = inverse[node.id]
    return ast.dump(cloned, include_attributes=False)


def _derive_view(source: str, bindings: dict[str, str]) -> dict[str, Any]:
    raw = _raw(source, MAX_SOURCE_BYTES, "source")
    bindings = _bindings(bindings)
    if "\r" in source.replace("\r\n", ""):
        _fail("standalone carriage-return line endings are unsupported")
    tokens = _tokens(source)
    tree = _tree(source)
    functions = _functions(tree)
    extractor_type, span_type = _owner_types()
    # The owner record has a normalized ID contract; a mediator's exact
    # repository binding can instead be an absolute path. Encode it explicitly
    # and retain both identities, rather than silently normalizing the binding.
    owner_repository_id = "repository:" + cid_for_dag_json({"repository_id": bindings["repository_id"]})
    try:
        record = extractor_type(
            max_source_bytes=MAX_SOURCE_BYTES, max_ast_nodes=MAX_AST_NODES,
            max_ast_depth=MAX_AST_DEPTH,
        ).extract(
            raw, path=bindings["path"], repository_id=owner_repository_id,
            revision=bindings["tree_id"],
        )
    except (TypeError, ValueError) as exc:
        raise PythonCodeViewError("software-contract frontend rejected source bindings") from exc
    if record.diagnostics or record.unsupported or record.imports or record.calls:
        _fail("software-contract frontend did not accept the closed input")
    if record.provenance.source_cid != cid_for_bytes(raw):
        _fail("software-contract frontend source binding mismatch")
    lines, offsets = _line_table(source)
    token_locations: dict[tuple[int, int], tokenize.TokenInfo] = {}
    for token in tokens:
        if token.type == tokenize.NAME:
            key = (_token_offset(token.start, lines, offsets), _token_offset(token.end, lines, offsets))
            token_locations[key] = token
    reserved = {token.string for token in tokens if token.type == tokenize.NAME}
    token_map: list[dict[str, Any]] = []
    function_records: list[dict[str, Any]] = []
    slot_records: list[dict[str, Any]] = []
    replacements: dict[tuple[int, int], str] = {}
    for function in functions:
        node = function["node"]
        node_span = _span(node, offsets, span_type)
        owners = [symbol for symbol in record.symbols if symbol.kind == "function" and symbol.name == node.name]
        if len(owners) != 1 or owners[0].span.to_dict() != node_span:
            _fail("software-contract frontend function source span mismatch")
        owner = owners[0]
        scopes = [scope for scope in record.scopes if scope.owner_symbol_id == owner.symbol_id and scope.kind == "function"]
        if len(scopes) != 1 or owner.signature is None:
            _fail("software-contract frontend function scope mismatch")
        if [parameter.name for parameter in owner.signature.parameters] != function["parameters"]:
            _fail("software-contract frontend interface mismatch")
        names = function["parameters"] + function["locals"]
        alias_records = _aliases(names, reserved)
        aliases = {item["original"]: item["alias"] for item in alias_records}
        for item in alias_records:
            item["kind"] = "parameter" if item["original"] in function["parameters"] else "local"
        function_records.append({
            "function_id": function["function_id"], "name": node.name,
            "parameters": function["parameters"], "aliases": alias_records,
            "owner_symbol_id": owner.symbol_id, "owner_scope_id": scopes[0].scope_id,
            "source_span": node_span,
        })
        for occurrence in ast.walk(node):
            if not isinstance(occurrence, (ast.Name, ast.arg)):
                continue
            name = occurrence.id if isinstance(occurrence, ast.Name) else occurrence.arg
            occurrence_span = _span(occurrence, offsets, span_type)
            key = (occurrence_span["start_byte"], occurrence_span["end_byte"])
            token = token_locations.get(key)
            if token is None or token.string != name or raw[key[0]:key[1]].decode("utf-8") != name:
                _fail("AST binding does not match an exact NAME token")
            if key in replacements:
                _fail("duplicate binding occurrence")
            replacements[key] = aliases[name]
            token_map.append({
                "function_id": function["function_id"], "original": name,
                "alias": aliases[name], "role": "parameter" if isinstance(occurrence, ast.arg) else (
                    "store" if isinstance(occurrence.ctx, ast.Store) else "load"
                ), "source_span": occurrence_span,
            })
        for expression, statement, kind, target, available in function["slots"]:
            expression_span = _span(expression, offsets, span_type)
            original_expression = raw[expression_span["start_byte"]:expression_span["end_byte"]].decode("utf-8")
            statement_span = _span(statement, offsets, span_type)
            statement_text = raw[statement_span["start_byte"]:statement_span["end_byte"]].decode("utf-8")
            editable = not any(char in statement_text for char in ("\r", "\n", "\\"))
            if editable:
                editable = not any(token.type in {tokenize.COMMENT, tokenize.STRING} for token in _tokens(original_expression))
            slot_records.append({
                "slot_id": f"s{len(slot_records)}", "function_id": function["function_id"],
                "kind": kind, "target": target, "source_span": expression_span,
                "original_expression": original_expression,
                "before_sha256": _sha(original_expression.encode("utf-8")),
                "available_aliases": [aliases[name] for name in available],
                "editable": editable,
                "refusal_reason": None if editable else "multiline_or_protected_expression_trivia",
            })
    edits = [(start, end, alias.encode("utf-8")) for (start, end), alias in replacements.items()]
    for left, right in zip(tokens, tokens[1:], strict=False):
        if left.type in _IGNORED_TOKENS or right.type in _IGNORED_TOKENS or left.type in {tokenize.COMMENT, tokenize.STRING} or right.type in {tokenize.COMMENT, tokenize.STRING}:
            continue
        if left.end[0] != right.start[0]:
            continue
        start = _token_offset(left.end, lines, offsets)
        end = _token_offset(right.start, lines, offsets)
        gap = raw[start:end]
        if not gap or any(byte not in (32, 9) for byte in gap):
            continue
        left_key = (_token_offset(left.start, lines, offsets), start)
        right_key = (end, _token_offset(right.end, lines, offsets))
        separator = _pair_separator(left, right, replacements.get(left_key, left.string), replacements.get(right_key, right.string))
        edits.append((start, end, separator.encode("utf-8")))
    compact_source = _splice(raw, edits).decode("utf-8")
    compact_tokens = _tokens(compact_source)
    expected_tokens = []
    for token in tokens:
        if token.type in _IGNORED_TOKENS:
            continue
        key = (_token_offset(token.start, lines, offsets), _token_offset(token.end, lines, offsets))
        expected_tokens.append([token.type, replacements.get(key, token.string)])
    if _signature(compact_tokens) != expected_tokens:
        _fail("compact view changed significant tokens")
    original_ast = ast.dump(tree, include_attributes=False)
    if _inverse_tree(_tree(compact_source), function_records) != original_ast:
        _fail("compact view changed the alpha-normalized AST")
    token_map.sort(key=lambda item: item["source_span"]["start_byte"])
    return {
        "schema": VIEW_SCHEMA, "revision": VIEW_REVISION, "bindings": bindings,
        "original_source": source, "source_sha256": _sha(raw), "source_cid": record.provenance.source_cid,
        "compact_source": compact_source, "compact_sha256": _sha(compact_source.encode("utf-8")),
        "frontend": {
            "record_cid": record.cid, "provenance": record.provenance.to_dict(),
            "capability": record.frontend.to_dict(),
            "repository_identity": {"binding": bindings["repository_id"], "owner_record_id": owner_repository_id},
        },
        "functions": function_records, "token_map": token_map, "slots": slot_records,
        "invariants": {"native_ast_sha256": _sha(original_ast.encode("utf-8")), "significant_token_sha256": _sha(_canonical(expected_tokens))},
        "authority": dict(_NO_AUTHORITY),
    }


def _checked_view(artifact_json: str) -> dict[str, Any]:
    data = _read_json(artifact_json)
    if set(data) != _VIEW_FIELDS or data.get("schema") != VIEW_SCHEMA or data.get("revision") != VIEW_REVISION:
        _fail("unsupported or non-closed source-view artifact")
    expected = _derive_view(data["original_source"], data["bindings"])
    if _canonical(data) != _canonical(expected):
        _fail("source-view artifact identities or maps are forged")
    return data


@dataclass(frozen=True, slots=True)
class PythonCodeView:
    """Immutable, closed private artifact; use model_packet for model input."""

    artifact_json: str

    def __post_init__(self) -> None:
        data = _checked_view(self.artifact_json)
        object.__setattr__(self, "artifact_json", _canonical(data).decode("utf-8"))

    @property
    def view_id(self) -> str:
        return cid_for_dag_json(self.to_dict())

    @property
    def path(self) -> str:
        return self.to_dict()["bindings"]["path"]

    @property
    def compact_source(self) -> str:
        return self.to_dict()["compact_source"]

    @property
    def source_sha256(self) -> str:
        return self.to_dict()["source_sha256"]

    def to_dict(self) -> dict[str, Any]:
        return _read_json(self.artifact_json)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PythonCodeView:
        return cls(_canonical(data).decode("utf-8"))

    @classmethod
    def from_json(cls, text: str) -> PythonCodeView:
        return cls(text)

    def model_packet(self) -> dict[str, Any]:
        data = _checked_view(self.artifact_json)
        names = {item["function_id"]: item["name"] for item in data["functions"]}
        return {
            "schema": "supervisor-python-code-view-model-packet@1", "revision": VIEW_REVISION,
            "view_id": cid_for_dag_json(data), "bindings": data["bindings"],
            "compact_source": data["compact_source"],
            "functions": [{"function_id": item["function_id"], "name": item["name"], "aliases": item["aliases"]} for item in data["functions"]],
            "slots": [{"slot_id": item["slot_id"], "function": names[item["function_id"]], "kind": item["kind"], "before_sha256": item["before_sha256"], "editable": item["editable"], "refusal_reason": item["refusal_reason"]} for item in data["slots"]],
            "authority": dict(_NO_AUTHORITY),
        }


def build_python_code_view(*, source: str, path: str, repository_id: str, tree_id: str, task_id: str) -> PythonCodeView:
    data = _derive_view(source, {"path": path, "repository_id": repository_id, "tree_id": tree_id, "task_id": task_id})
    return PythonCodeView(_canonical(data).decode("utf-8"))


def _current(view: PythonCodeView, current_source: str) -> dict[str, Any]:
    if type(view) is not PythonCodeView:
        _fail("expected an exact PythonCodeView")
    data = _checked_view(view.artifact_json)
    current_raw = _raw(current_source, MAX_SOURCE_BYTES, "current source")
    if _sha(current_raw) != data["source_sha256"] or current_source != data["original_source"]:
        _fail("current source does not match the view preimage")
    return data


def restore_python_code_view(*, view: PythonCodeView, current_source: str) -> str:
    return _current(view, current_source)["original_source"]


def _derive_edit(data: dict[str, Any], slot_id: str, before_sha256: str, expression: str) -> dict[str, Any]:
    if type(slot_id) is not str or type(before_sha256) is not str:
        _fail("slot and preimage identities must be exact strings")
    slots = [item for item in data["slots"] if item["slot_id"] == slot_id]
    if len(slots) != 1 or slots[0]["before_sha256"] != before_sha256:
        _fail("expression slot identity or preimage hash mismatch")
    slot = slots[0]
    if not slot["editable"]:
        _fail("original expression has protected or ambiguous trivia")
    expression_raw = _raw(expression, MAX_EXPRESSION_BYTES, "replacement expression")
    if not expression or any(char in expression for char in ("\r", "\n", "\\")):
        _fail("replacement must be a nonempty single-line expression")
    replacement_tokens = _tokens(expression)
    if any(token.type in {tokenize.COMMENT, tokenize.STRING} for token in replacement_tokens):
        _fail("replacement has protected tokens")
    parsed = _tree(expression, expression=True)
    _expression(parsed.body, set(slot["available_aliases"]))
    function = next(item for item in data["functions"] if item["function_id"] == slot["function_id"])
    inverse = {item["alias"]: item["original"] for item in function["aliases"]}
    lines, offsets = _line_table(expression)
    edits = []
    for token in replacement_tokens:
        if token.type == tokenize.NAME:
            if token.string not in slot["available_aliases"]:
                _fail("replacement references an unavailable alias")
            edits.append((_token_offset(token.start, lines, offsets), _token_offset(token.end, lines, offsets), inverse[token.string].encode("utf-8")))
    native_expression = _splice(expression_raw, edits).decode("utf-8")
    original_raw = data["original_source"].encode("utf-8")
    span = slot["source_span"]
    after_raw = _splice(original_raw, [(span["start_byte"], span["end_byte"], native_expression.encode("utf-8"))])
    after_source = after_raw.decode("utf-8")
    # Recheck the entire native subset and explicit unchanged function interface.
    after_data = _derive_view(after_source, data["bindings"])
    if [(item["name"], item["parameters"]) for item in after_data["functions"]] != [(item["name"], item["parameters"]) for item in data["functions"]]:
        _fail("expression replacement changed a public interface")
    original_tree = _tree(data["original_source"])
    original_functions = _functions(original_tree)
    selected = next(item for item in original_functions if item["function_id"] == slot["function_id"])
    function_slots = [item for item in data["slots"] if item["function_id"] == slot["function_id"]]
    local_index = next(index for index, item in enumerate(function_slots) if item["slot_id"] == slot_id)
    _, statement, kind, _, _ = selected["slots"][local_index]
    native_node = _tree(native_expression, expression=True).body
    if kind == "assignment":
        statement.value = native_node
    else:
        statement.value = native_node
    if ast.dump(original_tree, include_attributes=False) != ast.dump(_tree(after_source), include_attributes=False):
        _fail("replacement escaped the exact native expression slot")
    return {
        "schema": EDIT_SCHEMA, "revision": VIEW_REVISION, "bindings": data["bindings"],
        "view_id": cid_for_dag_json(data), "slot_id": slot_id,
        "before_source": data["original_source"], "after_source": after_source,
        "before_sha256": data["source_sha256"], "after_sha256": _sha(after_raw),
        "before_cid": data["source_cid"], "after_cid": cid_for_bytes(after_raw),
        "expression_before_sha256": before_sha256, "model_expression": expression,
        "native_expression": native_expression, "source_span": span,
        "authority": dict(_NO_AUTHORITY),
    }


@dataclass(frozen=True, slots=True)
class NativeCodeViewEdit:
    """A hash-bound native edit candidate with no write or proof authority."""

    artifact_json: str

    def __post_init__(self) -> None:
        data = _read_json(self.artifact_json)
        if set(data) != _EDIT_FIELDS or data.get("schema") != EDIT_SCHEMA or data.get("revision") != VIEW_REVISION:
            _fail("unsupported or non-closed native edit artifact")
        view = _derive_view(data["before_source"], data["bindings"])
        expected = _derive_edit(view, data["slot_id"], data["expression_before_sha256"], data["model_expression"])
        if _canonical(data) != _canonical(expected):
            _fail("native edit identities or source bytes are forged")
        object.__setattr__(self, "artifact_json", _canonical(data).decode("utf-8"))

    @property
    def edit_id(self) -> str:
        return cid_for_dag_json(self.to_dict())

    @property
    def path(self) -> str:
        return self.to_dict()["bindings"]["path"]

    @property
    def view_id(self) -> str:
        return self.to_dict()["view_id"]

    @property
    def slot_id(self) -> str:
        return self.to_dict()["slot_id"]

    @property
    def before_source(self) -> str:
        return self.to_dict()["before_source"]

    @property
    def after_source(self) -> str:
        return self.to_dict()["after_source"]

    @property
    def before_sha256(self) -> str:
        return self.to_dict()["before_sha256"]

    @property
    def after_sha256(self) -> str:
        return self.to_dict()["after_sha256"]

    def to_dict(self) -> dict[str, Any]:
        return _read_json(self.artifact_json)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NativeCodeViewEdit:
        return cls(_canonical(data).decode("utf-8"))

    @classmethod
    def from_json(cls, text: str) -> NativeCodeViewEdit:
        return cls(text)


def replace_python_expression(*, view: PythonCodeView, slot_id: str, before_sha256: str, expression: str, current_source: str) -> NativeCodeViewEdit:
    data = _current(view, current_source)
    candidate = _derive_edit(data, slot_id, before_sha256, expression)
    return NativeCodeViewEdit(_canonical(candidate).decode("utf-8"))
