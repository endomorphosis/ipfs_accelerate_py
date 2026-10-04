"""Independent source, split and native-fit audits for the public decoder lane."""
from collections import Counter
from copy import deepcopy
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

from ipfs_datasets_py.logic.formalization.autoencoder.security import (
    codebase_autoencoder_transfer as transfer,
    security_formula_decoder as decoder,
    security_formula_grammar as grammar,
    security_formula_grammar_v2 as grammar_v2,
)


PUBLIC_SOURCE_SHA256 = "761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba"
PUBLIC_ROLES = {"_hkey": "train", "_hval": "validation", "html_escape": "test"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _api():
    return importlib.import_module(
        "benchmarks.agent_supervisor.container_coding.terminal_codebase_decoder_training")


@pytest.fixture(scope="module")
def public_source():
    default = (Path(__file__).resolve().parents[4] / "artifacts/terminal_bench_supervisor/"
               "full-integration-20260929/terminal-source-inputs-01/permitted-inputs/bottle.py")
    path = Path(os.environ.get("TERMINAL_CODEBASE_DECODER_PUBLIC_SOURCE", default))
    if not path.is_file():
        pytest.skip("independently captured public Bottle source is unavailable")
    raw = path.read_bytes()
    assert _sha(raw) == PUBLIC_SOURCE_SHA256
    return path, raw


def _independent_functions(raw):
    """Inventory the complete original AST without the production span adapter."""
    found = []

    def visit(node, parents=()):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.append((child, ".".join((*parents, child.name))))
                visit(child, (*parents, child.name))
            elif isinstance(child, ast.ClassDef):
                visit(child, (*parents, child.name))
            else:
                visit(child, parents)

    visit(ast.parse(raw, type_comments=True))
    return sorted(found, key=lambda item: (item[0].lineno, item[0].col_offset, item[1]))


@pytest.fixture
def lexical_fork(tmp_path):
    """An explicitly synthetic portable parent, not an experimental learned head."""
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.modal_autoencoder import (
        ModalAutoencoderTrainingState,
    )
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.modal_autoencoder_checkpoint import (
        serialize_checkpoint,
    )

    state = ModalAutoencoderTrainingState(feature_embedding_weights={
        "token:return": [.03 * (i + 1) for i in range(8)],
        "token:value": [-.02 * (i + 1) for i in range(8)],
        "law-specific:untransferred": [7.] * 8,
    })
    raw = serialize_checkpoint(state, metadata={"fixture": "independent-fit-membership-audit"})
    source = tmp_path / "parent" / "parent.json"
    source.parent.mkdir()
    source.write_bytes(raw)
    fork = transfer.fork_legal_shared_weights(source_checkpoint=source,
        expected_sha256=_sha(raw), output=tmp_path / "fork" / "security-code-initializer")
    return fork, source, raw


def test_actual_native_loss_targets_exclude_validation_and_test_teachers(lexical_fork, tmp_path, monkeypatch):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import modal_autoencoder_cuda as kernel

    samples = [
        {"id": "fit-only-add", "split": "train", "source": "def fit(value):\n    return value + 2\n"},
        {"id": "validation-only-mul", "split": "validation", "source": "def validate(value):\n    return value * 7\n"},
        {"id": "test-only-string", "split": "test", "source": "def withheld(value):\n    return 'heldout'\n"},
    ]
    expected = [grammar.PRODUCTIONS.index(node["teacher_production"])
                for node in grammar.parse_formula_source(samples[0]["source"].encode())["nodes"]]
    observed = []
    native_loss = kernel._loss_chunk

    def audit_loss(state, session, outputs, update_targets, start, stop, total_samples, *args, **kwargs):
        labels = state.family_targets.argmax(dim=1).tolist()
        assert labels == expected
        assert total_samples == len(expected)
        assert update_targets == {"family_logits"}
        observed.append((start, stop, tuple(labels)))
        return native_loss(state, session, outputs, update_targets, start, stop, total_samples, *args, **kwargs)

    monkeypatch.setattr(kernel, "_loss_chunk", audit_loss)
    fork, source, raw = lexical_fork
    checkpoint = decoder.train_security_formula_decoder(samples=samples, weight_transfer=fork,
        output=tmp_path / "decoder", epochs=2,
        training_data_scope="caller_declared_development_controls")
    loaded = decoder.load_security_formula_decoder(checkpoint)
    training = loaded["training"]
    assert len(observed) == training["training_steps"] == len(training["training_losses"]) == 2
    assert len(training["gradient_norms"]) == 2
    assert all(start == 0 and stop == len(expected) for start, stop, _ in observed)
    fitted_classes = {grammar.PRODUCTIONS[index] for index in expected}
    assert "mul" not in fitted_classes and "string" not in fitted_classes
    assert training["metrics"]["test"]["programs"] == 1
    assert training["heldout_used_for_fit"] is False
    assert training["initial_head_sha256"] != training["final_head_sha256"]
    assert source.read_bytes() == raw


@pytest.mark.parametrize("source", [
    "def annotated(value: int) -> int:\n    return value + 1\n",
    "def defaulted(value=1):\n    return value + 1\n",
    "def divide(value, granularity):\n    return ((value + granularity - 1) // granularity) * granularity\n",
    "def stateful(value):\n    value = str(value)\n    return value\n",
])
def test_v2_refuses_original_unsupported_syntax_without_rewriting(source):
    raw = source.encode()
    before = ast.dump(ast.parse(raw), include_attributes=False)
    with pytest.raises(grammar_v2.UnsupportedFormulaSource):
        grammar_v2.parse_formula_source(raw)
    assert ast.dump(ast.parse(raw), include_attributes=False) == before


def test_literal_and_alpha_variants_are_not_independent_split_shapes():
    first = b"def one(value):\n    return value + 1\n"
    second = b"def another(different):\n    return different + 99\n"
    assert first != second and _sha(first) != _sha(second)
    assert grammar.source_shape(first) == grammar.source_shape(second)
    assert grammar_v2.source_shape(first) == grammar_v2.source_shape(second)


@pytest.mark.parametrize("historical_role", ["validation", "test"])
def test_continuation_cannot_fit_an_ancestor_heldout_shape(historical_role):
    from ipfs_datasets_py.logic.formalization.autoencoder.security import (
        security_formula_decoder_continuation as continuation,
    )

    ancestor_source = b"def old(earlier):\n    return earlier + 9\n"
    samples = [
        {"id": "new-fit", "split": "train", "source": "def new(value):\n    return value + 2\n"},
        {"id": "new-validation", "split": "validation", "source": "def validate(value):\n    return value * 7\n"},
        {"id": "new-test", "split": "test", "source": "def withheld(value):\n    return -value\n"},
    ]
    history = {role: [] for role in ("train", "validation", "test")}
    history[historical_role] = [{"id": "ancestor-heldout", "source_sha256": _sha(ancestor_source),
        "program_shape_sha256": grammar_v2.source_shape(ancestor_source)}]
    assert _sha(ancestor_source) != _sha(samples[0]["source"].encode())
    original = deepcopy(history)
    with pytest.raises(ValueError, match="cross-split"):
        continuation.prepare_security_production_samples(samples, historical_splits=history)
    assert history == original


def test_corpus_retains_every_original_function_and_exact_normalization_map(public_source):
    _, raw = public_source
    report = _api().prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py")
    original = _independent_functions(raw)
    assert len(original) == len(report["functions"]) == 358
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    expected = {}
    for node, name in original:
        start_line = min([node.lineno, *(item.lineno for item in node.decorator_list)])
        bounds = (offsets[start_line - 1], offsets[node.end_lineno - 1] + node.end_col_offset)
        expected[(name, *bounds)] = node
    seen = set()
    for row in report["functions"]:
        binding = row["source_binding"]
        start, stop = binding["start_byte"], binding["end_byte"]
        key = (row["qualified_name"], start, stop)
        assert key in expected and key not in seen
        seen.add(key)
        node = expected[key]
        assert row["line"] == node.lineno and row["end_line"] == node.end_lineno
        assert row["source_path"] == "bottle.py" and row["source_sha256"] == PUBLIC_SOURCE_SHA256
        assert row["source_ast_sha256"] == _sha(ast.dump(node, include_attributes=False).encode())
        assert binding["span_sha256"] == _sha(raw[start:stop])
        recovered, normalized_cursor, source_cursor = [], 0, start
        for span in binding["line_byte_map"]:
            a, b = span["source_start_byte"], span["source_end_byte"]
            c, d = span["normalized_start_byte"], span["normalized_end_byte"]
            assert source_cursor <= a <= b <= stop and c == normalized_cursor
            assert all(byte in (9, 32) for byte in raw[source_cursor:a])
            assert d - c == b - a
            recovered.append(raw[a:b])
            source_cursor, normalized_cursor = b, d
        body = b"".join(recovered)
        assert source_cursor == stop and normalized_cursor == len(body)
        assert row["normalized_body_sha256"] == binding["normalized_body_sha256"] == _sha(body)
        assert set(row["eligibility"]) == {"v1", "v2", "typed_v2_logic"}
        for label, parser in (("v1", grammar), ("v2", grammar_v2)):
            eligibility = row["eligibility"][label]
            assert eligibility["status"] in {"supported", "unsupported"}
            if eligibility["status"] == "supported":
                assert ast.dump(ast.parse(body).body[0], include_attributes=False) == ast.dump(node, include_attributes=False)
                parsed = parser.parse_formula_source(body)
                assert eligibility["node_count"] == len(parsed["nodes"])
                assert eligibility["program_shape_sha256"] == parser.source_shape(body)
            else:
                assert eligibility["reason"] and eligibility["node_count"] == 0
        if row["normalized_source"] is not None:
            assert row["normalized_source"].encode() == body
            assert row["teacher_nodes"] == grammar.parse_formula_source(body)["nodes"]
            for teacher in row["teacher_nodes"]:
                a, b = teacher["start_byte"], teacher["end_byte"]
                assert 0 <= a < b <= len(body)
                assert teacher["source_span_sha256"] == _sha(body[a:b])
        else:
            assert row["teacher_nodes"] == [] and row["split"] is None
    assert seen == set(expected)
    assert len({row["unit_id"] for row in report["functions"]}) == 358
    assert report["counts"] == {"functions_observed": 358, "v1_supported": 3,
        "v1_unsupported": 355, "v2_supported": 1, "typed_v2_logic_supported": 0,
        "selected_samples": 3}


def test_roles_and_teacher_membership_are_frozen_before_fit(public_source, monkeypatch):
    _, raw = public_source

    def no_fit(**kwargs):
        pytest.fail("preparing a source corpus must not fit weights")

    monkeypatch.setattr(decoder, "train_security_formula_decoder", no_fit)
    api = _api()
    report = api.prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py")
    selected = [row for row in report["functions"] if row["split"] is not None]
    assert {row["qualified_name"]: row["split"] for row in selected} == PUBLIC_ROLES
    assert report["role_policy"] == PUBLIC_ROLES
    assert report["training_steps"] == 0 and report["training_input_contract_satisfied"] is True
    assert report["lineage"]["checked_before_fit"] is True
    assert {row["split"] for row in report["samples"]} == {"train", "validation", "test"}
    shapes = {row["eligibility"]["v1"]["program_shape_sha256"] for row in selected}
    assert len(shapes) == 3
    fitted = [row for row in selected if row["teacher_targets_used_for_fit"]]
    assert [row["qualified_name"] for row in fitted] == ["_hkey"]
    assert Counter(node["teacher_production"] for row in fitted for node in row["teacher_nodes"]) == {
        "name": 2, "call": 1, "assign": 1, "title": 1, "string": 2, "replace": 1, "return": 1}
    for sample in report["samples"]:
        row = next(row for row in selected if sample["id"] == "terminal-v1:" + row["unit_id"])
        assert sample["source"] == row["normalized_source"]
        assert sample["source_sha256"] == _sha(sample["source"].encode())
        assert row["teacher_targets_used_for_fit"] is (sample["split"] == "train")
    frozen = dict(report)
    recorded_digest = frozen.pop("corpus_sha256")
    assert recorded_digest == _sha(_wire(frozen))
    assert api.prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py") == report
    assert report["proof_authority"] is report["source_semantics_verified"] is False


def test_caller_cannot_mutate_the_predeclared_role_policy(public_source, monkeypatch):
    _, raw = public_source
    api = _api()
    monkeypatch.setattr(api, "ROLE_POLICY", dict(PUBLIC_ROLES))
    first = api.prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py")
    first["role_policy"]["_hkey"] = "test"
    second = api.prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py")
    assert api.ROLE_POLICY == second["role_policy"] == PUBLIC_ROLES
    assert {row["qualified_name"]: row["split"] for row in second["functions"] if row["split"]} == PUBLIC_ROLES


def test_complete_public_source_drift_is_rejected_before_fit(public_source, monkeypatch):
    _, raw = public_source

    def no_fit(**kwargs):
        pytest.fail("changed source must not fit weights")

    monkeypatch.setattr(decoder, "train_security_formula_decoder", no_fit)
    with pytest.raises(ValueError, match="public source identity"):
        _api().prepare_terminal_codebase_decoder_corpus(source_bytes=raw + b"\n# changed\n", source_path="bottle.py")


def test_v2_public_frontier_keeps_exact_parameter_reassignment_and_calls(public_source):
    _, raw = public_source
    report = _api().prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py")
    by_name = {row["qualified_name"]: row for row in report["functions"] if row["split"]}
    for name in ("_hkey", "_hval"):
        row = by_name[name]
        assert row["eligibility"]["v2"]["status"] == "unsupported"
        assert "fresh unannotated local assignment" in row["eligibility"]["v2"]["reason"]
        fn = ast.parse(row["normalized_source"]).body[0]
        assert isinstance(fn.body[0], ast.Assign)
        assert fn.body[0].targets[0].id == fn.args.args[0].arg
    escape = by_name["html_escape"]
    assert escape["eligibility"]["v2"]["status"] == "supported"
    assert escape["eligibility"]["typed_v2_logic"] == {"status": "unsupported", "reason": "unsupported_expression:Call"}


@pytest.mark.parametrize("variant", ["exact_source", "alpha_shape"])
def test_genuine_v1_parent_cannot_reassign_heldout_shape_roles(public_source, lexical_fork, tmp_path, variant):
    _, raw = public_source
    api = _api()
    corpus = api.prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py")
    original = next(sample for sample in corpus["samples"] if sample["split"] == "validation")
    source = original["source"]
    if variant == "alpha_shape":
        source = source.replace("_hval", "renamed").replace("value", "renamed_value").replace("touni", "str")
        assert _sha(source.encode()) != original["source_sha256"]
        assert grammar.source_shape(source.encode()) == grammar.source_shape(original["source"].encode())
    prior_samples = [
        {"id": "ancestor-fit", "split": "train", "source": source},
        {"id": "ancestor-validation", "split": "validation", "source": "def validate(value):\n    return value + 2\n"},
        {"id": "ancestor-test", "split": "test", "source": "def withheld(value):\n    return -value\n"},
    ]
    fork, parent_source, parent_bytes = lexical_fork
    checkpoint = decoder.train_security_formula_decoder(samples=prior_samples, weight_transfer=fork,
        output=tmp_path / "ancestor-decoder", epochs=2,
        training_data_scope="caller_declared_development_controls")
    stored = {path.name: path.read_bytes() for path in Path(checkpoint["output"]).iterdir()}
    with pytest.raises(ValueError, match="historical.*reassignment"):
        api.prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py", parent_checkpoint=checkpoint)
    assert stored == {path.name: path.read_bytes() for path in Path(checkpoint["output"]).iterdir()}
    assert parent_source.read_bytes() == parent_bytes


def test_genuine_same_role_v1_history_is_replayed_and_detached(public_source, lexical_fork, tmp_path):
    _, raw = public_source
    api = _api()
    corpus = api.prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py")
    fork, parent_source, parent_bytes = lexical_fork
    checkpoint = decoder.train_security_formula_decoder(samples=corpus["samples"], weight_transfer=fork,
        output=tmp_path / "ancestor-decoder", epochs=2,
        training_data_scope="caller_declared_development_controls")
    original_descriptor = deepcopy(checkpoint)
    replayed = api.prepare_terminal_codebase_decoder_corpus(source_bytes=raw, source_path="bottle.py", parent_checkpoint=checkpoint)
    assert replayed["training_input_contract_satisfied"] is True
    lineage = replayed["lineage"]
    assert lineage["parent_checkpoint"] == original_descriptor
    assert len(lineage["same_role_matches"]) == lineage["historical_identity_count"] == 6
    assert lineage["cross_role_matches"] == [] and lineage["checked_before_fit"] is True
    assert lineage["parent_head_weights_copied"] is lineage["parent_quality_requalified"] is False
    actual_training = decoder.load_security_formula_decoder(original_descriptor)["training"]
    assert lineage["parent_training_receipt_sha256"] == _sha(_wire(actual_training))
    checkpoint["manifest_sha256"] = "0" * 64
    assert lineage["parent_checkpoint"] == original_descriptor
    assert parent_source.read_bytes() == parent_bytes
