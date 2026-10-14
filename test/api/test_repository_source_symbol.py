"""Exact native spans bridge advisory symbols to supported canonical edits."""
import ast
import copy
import hashlib

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.code_source_view import (
    CodeSourceView, SourceViewError,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_symbol import render_indexed_symbol


def indexed(source, name, *, kind="function", path="pkg/__init__.py", ordinal=0):
    data = source.encode("utf-8")
    node = [node for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name][ordinal]
    row = dict(symbol_id="native-index-symbol", path=path, name="untrusted.display.name",
        kind=kind, source_sha256=hashlib.sha256(data).hexdigest(),
        span=dict(schema="ipfs-datasets.software-contracts.semantic-source-span@1", path=path,
            start_line=node.lineno, start_column=node.col_offset,
            end_line=node.end_lineno, end_column=node.end_col_offset))
    return data, row


def render(source, name, *, representation="compact", **options):
    data, row = indexed(source, name, **options)
    return render_indexed_symbol(row["path"], data, row, representation)


def test_native_span_finds_decorated_method_without_module_name_guessing(tmp_path):
    source = ('# module context remains canonical\nimport unseen\n\nclass Example:\n'
        '    @unseen.decorate\n    def café(self, value):\n'
        '        # A deliberately long implementation explanation.\n'
        '        return value + "é"\n\ndef unrelated():\n    return 99\n')
    data, row = indexed(source, "café", kind="method")
    value = render_indexed_symbol(row["path"], data, row, "compact")
    assert value["symbol"] == "Example.café" and value["symbol_id"] == row["symbol_id"]
    assert value["scope"] == "python_definition" and value["edit_supported"] is True
    assert value["representation"] == "compact" and value["encoding"] == "utf-8"
    assert "@unseen.decorate" in value["content"] and "unrelated" not in value["content"]
    assert "implementation explanation" not in value["content"]
    assert value["original_sha256"] == hashlib.sha256(data).hexdigest()
    assert value["proof_authority"] is False
    original = render_indexed_symbol(row["path"], data, row, "original")
    assert original["content"].startswith("@unseen.decorate\ndef café")
    assert "implementation explanation" in original["content"]
    assert "import unseen" not in original["content"]
    assert original["limitations"] == ["surrounding_context_omitted", "definition_dedented"]
    # The returned selector and whole-file hash address the existing editor;
    # importing unseen or running its decorator is never necessary.
    target = tmp_path / row["path"]
    target.parent.mkdir()
    target.write_bytes(data)
    changed = CodeSourceView(tmp_path).replace(row["path"], value["symbol"], value["original_sha256"],
        "@unseen.decorate\ndef café(self, value):\n    return value + 'changed'\n")
    assert changed["changed"] is True
    assert target.read_bytes().startswith(b"# module context remains canonical\nimport unseen\n")
    assert target.read_bytes().endswith(b"def unrelated():\n    return 99\n")


@pytest.mark.parametrize("kind,source,name", [
    ("function", "def call():\n    return 1\n", "call"),
    ("async_function", "async def call():\n    return 1\n", "call"),
    ("method", "class Box:\n    async def call(self):\n        return 1\n", "call"),
    ("property", "class Box:\n    @property\n    def value(self):\n        return 1\n", "value"),
    ("test", "def test_call():\n    assert True\n", "test_call"),
    ("fixture", "@pytest.fixture\ndef setup():\n    return 1\n", "setup"),
    ("class", "class Box:\n    pass\n", "Box"),
    ("dataclass", "@dataclass\nclass Box:\n    size: int\n", "Box"),
    ("typed_dict", "class Box(TypedDict):\n    size: int\n", "Box"),
    ("enum", "class Box(Enum):\n    SMALL = 1\n", "Box"),
])
def test_supported_native_definition_kinds(kind, source, name):
    value = render(source, name, kind=kind)
    assert value["edit_supported"] is True and value["scope"] == "python_definition"


def test_compact_must_be_strictly_smaller_and_original_stays_original():
    value = render("def tiny(): return 1\n", "tiny")
    assert value["representation"] == "original" and value["content"] == "def tiny(): return 1"
    source = "def long():\n    # Explanatory comment to remove in a compact selection.\n    return 1\n"
    assert render(source, "long")["representation"] == "compact"
    original = render(source, "long", representation="original")
    assert original["representation"] == "original" and "Explanatory" in original["content"]


@pytest.mark.parametrize("source,name,kind,code", [
    ("def outer():\n    def inner():\n        return 1\n    return inner\n", "inner", "function", "unsupported_or_missing_symbol"),
    ("if flag:\n    def call():\n        return 1\n", "call", "function", "unsupported_or_missing_symbol"),
    ("class Box:\n    if flag:\n        def call(self):\n            return 1\n", "call", "method", "unsupported_or_missing_symbol"),
    ("def call():\n    return 1\ndef call():\n    return 2\n", "call", "function", "ambiguous_symbol"),
    ("def call():\n    return 1\nif flag:\n    def call():\n        return 2\n", "call", "function", "ambiguous_symbol"),
    ("class Box:\n    def call(self):\n        return 1\nclass Box:\n    pass\n", "call", "method", "ambiguous_symbol"),
    ("class Box:\n    def call(self):\n        return 1\nif flag:\n    class Box:\n        pass\n", "call", "method", "ambiguous_symbol"),
    ("class Box:\n    @property\n    def value(self):\n        return 1\n    @value.setter\n    def value(self, new):\n        pass\n", "value", "property", "ambiguous_symbol"),
    ("def module():\n    return 1\n", "module", "function", "unsupported_or_missing_symbol"),
])
def test_unsupported_or_ambiguous_targets_use_whole_file_fallback(source, name, kind, code):
    with pytest.raises(SourceViewError) as error:
        render(source, name, kind=kind)
    assert error.value.code == code


@pytest.mark.parametrize("change", ["path", "hash", "span_path", "schema", "span_extra", "start", "end", "bool", "kind", "id"])
def test_unbound_or_malformed_native_rows_refused(change):
    data, row = indexed("@decorate\ndef call():\n    return 1\n", "call")
    row = copy.deepcopy(row)
    if change == "path": row["path"] = "other.py"
    elif change == "hash": row["source_sha256"] = "0" * 64
    elif change == "span_path": row["span"]["path"] = "other.py"
    elif change == "schema": row["span"]["schema"] = "other"
    elif change == "span_extra": row["span"]["extra"] = 1
    elif change == "start": row["span"]["start_line"] = 1  # decorator isn't the native def span
    elif change == "end": row["span"]["end_column"] += 1
    elif change == "bool": row["span"]["start_column"] = False
    elif change == "kind": row["kind"] = "class"
    else: row["symbol_id"] = ""
    with pytest.raises(SourceViewError) as error:
        render_indexed_symbol("pkg/__init__.py", data, row, "original")
    assert error.value.code == "indexed_symbol_source_differs"


@pytest.mark.parametrize("kind", ["module", "constant", "variable", "unknown"])
def test_non_definition_native_kinds_are_optional_refusals(kind):
    data, row = indexed("def call():\n    return 1\n", "call", kind=kind)
    with pytest.raises(SourceViewError, match="unsupported_or_missing_symbol"):
        render_indexed_symbol(row["path"], data, row, "original")


def test_assignment_typed_dict_is_not_a_class_edit():
    source = 'Box = TypedDict("Box", {"size": int})\n'
    data = source.encode()
    node = ast.parse(source).body[0]
    row = dict(symbol_id="native-index-symbol", path="types.py", kind="typed_dict",
        source_sha256=hashlib.sha256(data).hexdigest(), span=dict(
            schema="ipfs-datasets.software-contracts.semantic-source-span@1", path="types.py",
            start_line=node.lineno, start_column=node.col_offset,
            end_line=node.end_lineno, end_column=node.end_col_offset))
    with pytest.raises(SourceViewError, match="unsupported_or_missing_symbol"):
        render_indexed_symbol("types.py", data, row, "original")


def test_crlf_bom_and_non_ascii_preserve_original_hash_and_selected_text():
    source = "class Box:\r\n    def café(self):\r\n        return 'é'\r\n"
    data, row = indexed(source, "café", kind="method")
    data = b"\xef\xbb\xbf" + data
    row["source_sha256"] = hashlib.sha256(data).hexdigest()
    value = render_indexed_symbol(row["path"], data, row, "original")
    assert value["content"] == "def café(self):\r\n    return 'é'"
    assert value["original_sha256"] == hashlib.sha256(data).hexdigest()


def test_invalid_target_code_is_never_executed_or_rendered():
    data, row = indexed("def call():\n    return 1\n", "call")
    data = b"def call(value, value):\n    return value\n"
    row["source_sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises(SourceViewError, match="invalid_python"):
        render_indexed_symbol(row["path"], data, row, "original")
