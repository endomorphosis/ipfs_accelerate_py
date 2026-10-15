"""Pure source-view/edit contracts; these tests never execute model source."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring import python_code_view as engine
from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes

SOURCE = (
    "# quantity_total is a semantic name, not a replaceable comment.\n"
    "def calculate_total(quantity_total, tax_amount):\n"
    '    """Keep quantity_total and tax_amount literally in this docstring."""\n'
    "    subtotal_amount = quantity_total + tax_amount * 2\n"
    "    return subtotal_amount // 3\n"
)


def make(source=SOURCE, **kwargs):
    return engine.build_python_code_view(
        source=source, path=kwargs.get("path", "src/calculation.py"),
        repository_id=kwargs.get("repository_id", "repository:fixture"),
        tree_id=kwargs.get("tree_id", "tree:fixture"),
        task_id=kwargs.get("task_id", "task:fixture"),
    )


def aliases(view, index=0):
    return {item["original"]: item["alias"] for item in view.to_dict()["functions"][index]["aliases"]}


def replace(view, source, slot_index, expression, **kwargs):
    slot = view.to_dict()["slots"][slot_index]
    return engine.replace_python_expression(
        view=view, slot_id=kwargs.get("slot_id", slot["slot_id"]),
        before_sha256=kwargs.get("before_sha256", slot["before_sha256"]),
        expression=expression, current_source=source,
    )


def test_aliases_shorten_only_attested_names_and_preserve_literal_tokens():
    view = make()
    names = aliases(view)
    assert names == {"quantity_total": "a", "tax_amount": "b", "subtotal_amount": "c"}
    assert "def calculate_total(a,b):" in view.compact_source
    assert "    c=a+b*2\n    return c//3\n" in view.compact_source
    assert SOURCE.splitlines()[0] in view.compact_source
    assert SOURCE.splitlines()[2] in view.compact_source
    assert len(view.compact_source.encode()) < len(SOURCE.encode())
    assert engine.restore_python_code_view(view=view, current_source=SOURCE) == SOURCE
    native = view.to_dict()
    assert native["source_cid"] == cid_for_bytes(SOURCE.encode())
    assert native["frontend"]["provenance"]["source_cid"] == native["source_cid"]
    assert native["frontend"]["capability"]["frontend_name"] == "cpython-ast"
    assert native["authority"] and not any(native["authority"].values())


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
@pytest.mark.parametrize("final_newline", [False, True])
def test_exact_noop_restores_unicode_comments_docstrings_crlf_and_final_trivia(ending, final_newline):
    source = ending.join([
        '# café λ — descriptive_parameter stays literal',
        '"""Module café λ with descriptive_parameter."""',
        'def compute_result(descriptive_parameter):',
        '    """Unicode λ docstring, with descriptive_parameter."""',
        '    temporary_result = (descriptive_parameter + 0xA)',
        '    return temporary_result % 3  # café λ literal',
    ]) + (ending if final_newline else "")
    view = make(source)
    assert engine.restore_python_code_view(view=view, current_source=source).encode() == source.encode()
    assert "café λ" in view.compact_source
    assert "0xA" in view.compact_source
    assert view.compact_source.endswith(ending) is final_newline
    clone = engine.PythonCodeView.from_json(view.artifact_json)
    assert clone.view_id == view.view_id and clone.to_dict() == view.to_dict()


def test_utf8_spans_are_independently_attested_against_source_slices():
    source = '# é λ\n\ndef compute(descriptive_parameter):\n    """é λ"""; temporary_result = descriptive_parameter + 12\n    return temporary_result\n'
    view = make(source)
    raw = source.encode()
    for occurrence in view.to_dict()["token_map"]:
        span = occurrence["source_span"]
        assert raw[span["start_byte"]:span["end_byte"]].decode() == occurrence["original"]
        prefix = raw[:span["start_byte"]].decode()
        assert span["start_line"] == prefix.count("\n") + 1
        assert span["start_column"] == len(prefix.rsplit("\n", 1)[-1].encode())
    for slot in view.to_dict()["slots"]:
        span = slot["source_span"]
        assert raw[span["start_byte"]:span["end_byte"]].decode() == slot["original_expression"]


def test_aliases_are_bijective_collision_free_and_scoped_per_function():
    source = (
        "def first(a, abc_long):\n    long_value = abc_long + a\n    return long_value\n"
        "def second(b, abc_long):\n    long_value = abc_long + b\n    return long_value\n"
    )
    view = make(source)
    records = view.to_dict()["functions"]
    assert records[0]["function_id"] != records[1]["function_id"]
    assert records[0]["owner_scope_id"] != records[1]["owner_scope_id"]
    all_original = {"a", "b", "abc_long", "long_value", "first", "second"}
    for record in records:
        legend = record["aliases"]
        assert len({item["alias"] for item in legend}) == len(legend)
        for item in legend:
            if item["alias"] != item["original"]:
                assert item["alias"] not in all_original
    assert aliases(view, 0)["a"] == "a"
    assert aliases(view, 1)["b"] == "b"
    assert engine.restore_python_code_view(view=view, current_source=source) == source


def test_public_packet_keeps_semantic_legend_and_omits_private_original_and_spans():
    view = make()
    packet = view.model_packet()
    assert packet["view_id"] == view.view_id
    assert packet["bindings"] == view.to_dict()["bindings"]
    assert packet["compact_source"] == view.compact_source
    assert packet["functions"][0]["aliases"] == view.to_dict()["functions"][0]["aliases"]
    assert set(packet["slots"][0]) == {"slot_id", "function", "kind", "before_sha256", "editable", "refusal_reason"}
    encoded = json.dumps(packet)
    for private in ("original_source", "original_expression", "source_span", "token_map", "source_sha256"):
        assert private not in encoded
    packet["functions"][0]["aliases"][0]["original"] = "changed returned copy"
    assert aliases(view)["quantity_total"] == "a"
    with pytest.raises(FrozenInstanceError):
        view.artifact_json = "{}"


def test_absolute_repository_binding_has_explicit_deterministic_owner_record_id():
    binding = "/home/owner/repository"
    view = make(repository_id=binding)
    data = view.to_dict()
    mapping = data["frontend"]["repository_identity"]
    assert data["bindings"]["repository_id"] == binding
    assert mapping["binding"] == binding
    assert mapping["owner_record_id"].startswith("repository:b")
    assert data["frontend"]["provenance"]["repository_id"] == mapping["owner_record_id"]
    assert make(repository_id=binding).view_id == view.view_id
    assert make(repository_id=binding + "-different").view_id != view.view_id
    assert view.model_packet()["bindings"]["repository_id"] == binding


@pytest.mark.parametrize("slot_index", [0, 1])
def test_longer_expression_edit_expands_aliases_and_preserves_exact_outside_bytes(slot_index):
    source = SOURCE.replace("\n", "\r\n").rstrip("\r\n")
    view = make(source)
    legend = aliases(view)
    replacement = f"({legend['quantity_total']} + {legend['tax_amount']} * 17) % 13"
    edit = replace(view, source, slot_index, replacement)
    slot = view.to_dict()["slots"][slot_index]
    span = slot["source_span"]
    native = "(quantity_total + tax_amount * 17) % 13"
    before_raw = source.encode()
    expected = before_raw[:span["start_byte"]] + native.encode() + before_raw[span["end_byte"]:]
    assert edit.after_source.encode() == expected
    assert edit.before_source == source and edit.path == view.path
    assert edit.view_id == view.view_id and edit.slot_id == slot["slot_id"]
    assert edit.before_sha256 == hashlib.sha256(before_raw).hexdigest()
    assert edit.after_sha256 == hashlib.sha256(expected).hexdigest()
    assert edit.to_dict()["before_cid"] == cid_for_bytes(before_raw)
    assert edit.to_dict()["after_cid"] == cid_for_bytes(expected)
    assert not any(edit.to_dict()["authority"].values())
    assert engine.NativeCodeViewEdit.from_json(edit.artifact_json).edit_id == edit.edit_id


def test_parentheses_and_literal_spellings_survive_slot_translation():
    source = "def compute_total(long_parameter):\n    return ((long_parameter + 0x0f)) # outside\n"
    view = make(source)
    name = aliases(view)["long_parameter"]
    edit = replace(view, source, 0, f"{name} * 0b101_0 - -7")
    assert edit.after_source == "def compute_total(long_parameter):\n    return ((long_parameter * 0b101_0 - -7)) # outside\n"
    assert ast.dump(ast.parse(edit.after_source), include_attributes=False)


@pytest.mark.parametrize("expression", [
    "unknown + 1", "c + 1", "a()", "a.real", "a[0]", "[a]", "a ** 2",
    "a / 2", "True", "1.5", "'literal'", 'f"{a}"', "a if b else 0",
    "(a := 2)", "lambda: a", "a; b", "a # comment", "a\n+b", "a\\\n+b",
    "", " a + 1",
])
def test_edit_refuses_new_self_forward_names_structural_and_unsupported_expressions(expression):
    view = make()
    with pytest.raises(engine.PythonCodeViewError):
        replace(view, SOURCE, 0, expression)
    assert engine.restore_python_code_view(view=view, current_source=SOURCE) == SOURCE


def test_return_slot_can_use_previous_local_but_assignment_cannot_use_itself():
    view = make()
    local_alias = aliases(view)["subtotal_amount"]
    edit = replace(view, SOURCE, 1, f"{local_alias} + 1")
    assert "return subtotal_amount + 1" in edit.after_source
    with pytest.raises(engine.PythonCodeViewError, match="unavailable"):
        replace(view, SOURCE, 0, f"{local_alias} + 1")


@pytest.mark.parametrize("source", [
    "def f(long_name):\n    return (long_name +\n            1)\n",
    "def f(long_name):\n    return (long_name + # keep me\n            1)\n",
    "def f(long_name):\n    return long_name + \\\n        1\n",
])
def test_multiline_or_comment_expression_remains_restorable_but_not_editable(source):
    view = make(source)
    assert not view.to_dict()["slots"][0]["editable"]
    assert engine.restore_python_code_view(view=view, current_source=source) == source
    if "# keep me" in source:
        assert "# keep me" in view.compact_source
    with pytest.raises(engine.PythonCodeViewError, match="trivia"):
        replace(view, source, 0, aliases(view)["long_name"])


@pytest.mark.parametrize("source", [
    "import os\ndef f(x):\n return x\n",
    "class F:\n pass\n",
    "async def f(x):\n return x\n",
    "@decorate\ndef f(x):\n return x\n",
    "def f(x=1):\n return x\n",
    "def f(x: int):\n return x\n",
    "def f(x) -> int:\n return x\n",
    "def f(*args):\n return 1\n",
    "def f(**kwargs):\n return 1\n",
    "def f(*, x):\n return x\n",
    "def f(x, /):\n return x\n",
    "def f(x):\n x = 2\n return x\n",
    "def f(x):\n y = y + 1\n return y\n",
    "def f(x):\n y = z + 1\n z = x\n return y\n",
    "def f(x):\n y = x\n y = y+1\n return y\n",
    "def f(x):\n y = z = x\n return y\n",
    "def f(x):\n y: int = x\n return y\n",
    "def f(x):\n if x:\n  return x\n return 1\n",
    "def f(x):\n def nested():\n  return x\n return x\n",
    "def f(x):\n return abs(x)\n",
    "def f(x):\n return x.value\n",
    "def f(x):\n return x[0]\n",
    "def f(x):\n return [i for i in x]\n",
    "def f(x):\n return True\n",
    "def f(x):\n return 1.0\n",
    "def f(x):\n return x / 2\n",
    "def f(x):\n return x ** 2\n",
    "def f(x):\n return x and 1\n",
    "def f(x):\n return\n",
    "def f(x):\n pass\n",
    "def f(x):\n return x\ndef f(y):\n return y\n",
    "def café(x):\n return x\n",
    "def K(x):\n return x\n",
    "def f(K):\n return K\n",
    "def f(x):\n return x\r",
    "def f(x): # type: ignore\n return x\n",
    "def f(x):\n y = x # type: int\n return y\n",
])
def test_unsupported_native_sources_refuse_without_execution(source):
    with pytest.raises(engine.PythonCodeViewError):
        make(source)


@pytest.mark.parametrize("mutation", [
    lambda data: data.update({"extra": True}),
    lambda data: data.update({"schema": "other@1"}),
    lambda data: data.update({"revision": "other@1"}),
    lambda data: data.update({"compact_source": data["compact_source"].replace("calculate_total", "renamed")}),
    lambda data: data["functions"][0]["parameters"].append("injected"),
    lambda data: data["functions"][0]["aliases"][0].update({"alias": "b"}),
    lambda data: data["functions"][0]["aliases"].pop(),
    lambda data: data["token_map"][0]["source_span"].update({"start_byte": 0}),
    lambda data: data["slots"][0].update({"before_sha256": "0" * 64}),
    lambda data: data["slots"][0]["available_aliases"].append("c"),
    lambda data: data["frontend"].update({"record_cid": "invented"}),
    lambda data: data["authority"].update({"source_write_authority": True}),
])
def test_closed_private_view_rejects_forged_maps_interfaces_identities_and_authority(mutation):
    view = make()
    data = view.to_dict()
    mutation(data)
    with pytest.raises(engine.PythonCodeViewError):
        engine.PythonCodeView.from_dict(data)
    assert engine.restore_python_code_view(view=view, current_source=SOURCE) == SOURCE


def test_stale_source_and_slot_hash_refuse_instead_of_silently_rebasing():
    view = make()
    with pytest.raises(engine.PythonCodeViewError, match="preimage"):
        engine.restore_python_code_view(view=view, current_source=SOURCE + "# unrelated edit\n")
    with pytest.raises(engine.PythonCodeViewError, match="preimage"):
        replace(view, SOURCE, 0, "a + b", before_sha256="0" * 64)
    with pytest.raises(engine.PythonCodeViewError, match="slot"):
        replace(view, SOURCE, 0, "a + b", slot_id="s1000")
    with pytest.raises(engine.PythonCodeViewError, match="preimage"):
        replace(view, SOURCE.replace("* 2", "* 3"), 0, "a + b")


@pytest.mark.parametrize("field,value", [
    ("after_source", "def injected():\n return 1\n"),
    ("view_id", "unbound"), ("after_sha256", "0" * 64),
    ("native_expression", "unknown + 1"), ("slot_id", "s1000"),
])
def test_closed_edit_candidate_rejects_forged_native_source_and_bindings(field, value):
    edit = replace(make(), SOURCE, 0, "a + b + 1")
    data = edit.to_dict()
    data[field] = value
    with pytest.raises(engine.PythonCodeViewError):
        engine.NativeCodeViewEdit.from_dict(data)


@pytest.mark.parametrize("text", ["not JSON", "[]", '{"schema":"x","schema":"y"}', "{" * 1000])
def test_malformed_duplicate_or_wrong_shape_json_is_refused(text):
    with pytest.raises(engine.PythonCodeViewError):
        engine.PythonCodeView.from_json(text)


@pytest.mark.parametrize("kwargs", [
    {"path": "../escape.py"}, {"path": "/absolute.py"}, {"path": "src//bad.py"},
    {"path": "src\\bad.py"}, {"repository_id": ""}, {"tree_id": "two words"},
    {"task_id": None},
])
def test_source_bindings_are_closed_and_canonical(kwargs):
    with pytest.raises(engine.PythonCodeViewError):
        make(**kwargs)


def test_source_function_token_ast_depth_and_integer_digit_limits_are_enforced():
    with pytest.raises(engine.PythonCodeViewError, match="byte limit"):
        make("#" + "é" * engine.MAX_SOURCE_BYTES)
    with pytest.raises(engine.PythonCodeViewError, match="sixteen"):
        make("".join(f"def f{i}(x):\n return x\n" for i in range(engine.MAX_FUNCTIONS + 1)))
    with pytest.raises(engine.PythonCodeViewError, match="token limit"):
        make("def f(x):\n" + "".join(f" v{i} = 1\n" for i in range(2100)) + " return x\n")
    with pytest.raises(engine.PythonCodeViewError, match="AST limit"):
        make("".join(f"def f{i}(x):\n" + "".join(f" v{j} = 1\n" for j in range(80)) + " return x\n" for i in range(16)))
    with pytest.raises(engine.PythonCodeViewError, match="AST limit"):
        make("def f(x):\n return " + "+" * 45 + "x\n")
    with pytest.raises(engine.PythonCodeViewError, match="digit limit"):
        make("def f(x):\n return " + "9" * (engine.MAX_INTEGER_DIGITS + 1) + "\n")


def test_expression_and_result_source_bounds_are_enforced():
    view = make()
    with pytest.raises(engine.PythonCodeViewError, match="byte limit"):
        replace(view, SOURCE, 0, "a" + " " * engine.MAX_EXPRESSION_BYTES)
    source = "#" + "x" * (engine.MAX_SOURCE_BYTES - 100) + "\ndef f(long_parameter):\n return long_parameter\n"
    bounded_view = make(source)
    name = aliases(bounded_view)["long_parameter"]
    with pytest.raises(engine.PythonCodeViewError, match="byte limit"):
        replace(bounded_view, source, 0, name + "+" + " " * 100 + "1")


def test_closed_artifact_structure_and_size_limits_precede_materialization():
    with pytest.raises(engine.PythonCodeViewError, match="byte limit"):
        engine.PythonCodeView.from_json(" " * (engine.MAX_ARTIFACT_BYTES + 1))
    nested = {}
    for _ in range(engine.MAX_ARTIFACT_DEPTH + 1):
        nested = {"nested": nested}
    with pytest.raises(engine.PythonCodeViewError, match="structure"):
        engine.PythonCodeView.from_dict(nested)
    with pytest.raises(engine.PythonCodeViewError, match="structure"):
        engine.PythonCodeView.from_dict({"items": [None] * (engine.MAX_ARTIFACT_NODES + 1)})


@pytest.mark.parametrize("value", [
    {"large": "x" * (engine.MAX_ARTIFACT_BYTES + 1)},
    {"x" * (engine.MAX_ARTIFACT_BYTES + 1): "small"},
    {"large": 1 << (engine.MAX_INTEGER_DIGITS * 4 + 1)},
    {"left": "x" * (engine.MAX_ARTIFACT_BYTES // 2), "right": "x" * (engine.MAX_ARTIFACT_BYTES // 2)},
    {"escaped": "\x00" * (engine.MAX_ARTIFACT_BYTES // 6 + 1)},
])
def test_oversized_strings_keys_integers_and_combined_wire_refuse_before_serialization(value, monkeypatch):
    def forbidden_serializer(_):
        raise AssertionError("oversized artifact reached serialization")

    monkeypatch.setattr(engine, "canonical_dag_json_bytes", forbidden_serializer)
    with pytest.raises(engine.PythonCodeViewError, match="limit"):
        engine.PythonCodeView.from_dict(value)


def test_reordered_json_is_normalized_and_returned_copies_cannot_mutate_artifact():
    view = make()
    clone = engine.PythonCodeView.from_json(json.dumps(view.to_dict(), indent=2, ensure_ascii=False))
    assert clone.artifact_json == view.artifact_json and clone.view_id == view.view_id
    data = view.to_dict()
    data["token_map"].clear()
    assert view.to_dict()["token_map"]
    forged = object.__new__(engine.PythonCodeView)
    object.__setattr__(forged, "artifact_json", '{"schema":"forged"}')
    with pytest.raises(engine.PythonCodeViewError):
        engine.restore_python_code_view(view=forged, current_source=SOURCE)
