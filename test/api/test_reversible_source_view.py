"""Byte inverse and adversarial controls for read-only source displays."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import reversible_source_view as codec


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _manifest_sha(manifest):
    return _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode())


def _restore(source, result, **overrides):
    kwargs = dict(original=source, view=result.view, manifest=result.manifest,
                  expected_manifest_sha256=result.manifest_sha256)
    kwargs.update(overrides)
    return codec.restore_source_view(**kwargs)


def _assert_partition(source, result):
    cursor, previous, kept = 0, None, []
    for piece in result.manifest["pieces"]:
        assert set(piece) == {"kind", "start", "end"}
        assert piece["start"] == cursor < piece["end"] <= len(source)
        assert piece["kind"] in {"keep", "delete"} and piece["kind"] != previous
        if piece["kind"] == "keep":
            kept.append(source[piece["start"]:piece["end"]])
        cursor, previous = piece["end"], piece["kind"]
    assert cursor == len(source)
    assert b"".join(kept) == result.view
    assert _restore(source, result) == source


def test_crlf_unicode_literals_docstrings_and_default_comments_have_exact_inverse():
    source = (
        '# coding: utf-8\r\n'
        '"""Docstring has spaces  \t\r\n'
        'and a # literal marker.\r\n'
        '"""\r\n'
        'message = "caf\u00e9  \t"   \r\n'
        'payload = """literal  \t\r\n'
        'line  \t"""\r\n'
        'def add(value):  \t\r\n'
        '    return value + 1  \t\r\n'
        '# ordinary explanatory comment stays intact  \t\r\n'
    ).encode()
    result = codec.build_source_view(source)
    assert result.manifest["effective_profile"] == "python-whitespace@1"
    assert result.manifest["fallback_reason"] is None
    assert b"def add(value):\r\n" in result.view
    assert b"# ordinary explanatory comment stays intact  \t\r\n" in result.view
    assert '"caf\u00e9  \t"'.encode() in result.view
    assert b'"""literal  \t\r\nline  \t"""' in result.view
    assert result.view.count(b"\r\n") == source.count(b"\r\n")
    assert result.manifest["candidate_only"] is True
    assert all(value is False for name, value in result.manifest.items()
               if name.endswith(("_authority", "_claimed", "_verified")))
    assert ast.dump(ast.parse(result.view), include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)
    _assert_partition(source, result)


def test_comment_profile_retains_headers_directives_security_contract_and_caller_lines():
    source = (
        b"#!/usr/bin/env python3\n"
        b"# coding: utf-8\n"
        b"# SPDX-License-Identifier: MIT\n"
        b"VALUE = 1\n"
        b"# ordinary omitted description\n"
        b"VALUE = 2\n"
        b"# security: validate external input\n"
        b"VALUE = 3\n"
        b"# caller retained explanation\n"
        b"VALUE = 4\n"
        b"# pragma: no cover\n"
        b"VALUE = 5\n"
        b"# contract: preserve the precondition\n"
        b"VALUE = 6\n"
    )
    result = codec.build_source_view(source, profile="python-comments@1", protected_lines=(9,))
    assert b"ordinary omitted description" not in result.view
    for text in (b"#!/usr/bin/env", b"coding: utf-8", b"SPDX-License", b"security:",
                 b"caller retained explanation", b"pragma:", b"contract:"):
        assert text in result.view
    assert result.view.count(b"\n") == source.count(b"\n")
    wire = json.dumps(result.manifest)
    assert "ordinary omitted description" not in wire
    assert b"ordinary omitted description".hex() not in wire
    _assert_partition(source, result)


def test_inline_comment_offsets_are_utf8_bytes_without_splitting_retained_literal():
    source = 'value = "caf\u00e9 \u03bb"   # ordinary \u00e9clair explanation\r\n'.encode()
    result = codec.build_source_view(source, profile="python-comments@1")
    assert result.view == 'value = "caf\u00e9 \u03bb"   \r\n'.encode()
    deleted = [piece for piece in result.manifest["pieces"] if piece["kind"] == "delete"]
    assert deleted == [{"kind": "delete", "start": source.index(b"# ordinary"), "end": len(source) - 2}]
    for piece in result.manifest["pieces"]:
        source[:piece["start"]].decode("utf-8")
        source[:piece["end"]].decode("utf-8")
    _assert_partition(source, result)


def test_identity_profile_preserves_arbitrary_bytes_without_python_parsing(monkeypatch):
    source = b"\xff\x00\r\nnot valid Python  \t"
    with monkeypatch.context() as context:
        context.setattr(ast, "parse", lambda *a, **k: pytest.fail("identity profile parsed source"))
        result = codec.build_source_view(source, profile="identity@1")
        assert result.view == source
        assert result.manifest["effective_profile"] == "identity@1"
        assert result.manifest["fallback_reason"] is None
        _assert_partition(source, result)


@pytest.mark.parametrize("source,reason", [
    (b"\xff\x80\n", "non_utf8"),
    (b"VALUE =\n", "invalid_python"),
    (b"# coding: latin-1\nVALUE = 1\n", "unsupported_encoding"),
    (b"VALUE = 1\rVALUE = 2\r", "unsupported_newlines"),
    (b"#" * (1024 * 1024 + 1), "source_too_large"),
    (b"VALUE = 1   \n" * 3000, "piece_limit"),
    (b"VALUES = [" + b"0," * 65537 + b"0]\n", "unsupported_tokens"),
], ids=["non-utf8", "malformed", "encoding", "lone-cr", "large", "piece-limit", "token-limit"])
def test_unsupported_or_unbounded_transform_falls_back_to_exact_identity(source, reason, monkeypatch):
    with monkeypatch.context() as context:
        if reason == "unsupported_tokens":
            context.setattr(ast, "parse", lambda *a, **k: pytest.fail("unbounded tokens reached AST parser"))
        result = codec.build_source_view(source)
        assert result.view == source
        assert result.manifest["requested_profile"] == "python-whitespace@1"
        assert result.manifest["effective_profile"] == "identity@1"
        assert result.manifest["fallback_reason"] == reason
        assert result.manifest["pieces"] == [{"kind": "keep", "start": 0, "end": len(source)}]
        _assert_partition(source, result)


@pytest.mark.parametrize("source,profile", [
    (b'value = f"number: {1}"   \n', "python-whitespace@1"),
    (b'value = f"""{(\n    1  # ordinary nested expression comment\n)}"""\n', "python-comments@1"),
], ids=["ordinary-fstring", "nested-expression-comment"])
def test_valid_fstrings_conservatively_fall_back_to_identity(source, profile):
    assert any(isinstance(node, ast.JoinedStr) for node in ast.walk(ast.parse(source)))
    result = codec.build_source_view(source, profile=profile)
    assert result.view == source
    assert result.manifest["requested_profile"] == profile
    assert result.manifest["effective_profile"] == "identity@1"
    assert result.manifest["fallback_reason"] == "unsupported_fstrings"
    _assert_partition(source, result)


def test_empty_source_has_empty_partition_and_exact_inverse():
    result = codec.build_source_view(b"")
    assert result.view == b"" and result.manifest["pieces"] == []
    _assert_partition(b"", result)


def test_edited_compact_view_is_refused_without_an_edit_mapper():
    source = b"VALUE = 1  \n"
    result = codec.build_source_view(source)
    edited = result.view.replace(b"1", b"2")
    assert ast.parse(edited)
    with pytest.raises(codec.SourceViewError):
        _restore(source, result, view=edited)
    assert _restore(source, result) == source


def test_self_consistent_rehashed_alternative_map_cannot_change_profile_recipe():
    source = b"VALUE = 1  \n"
    result = codec.build_source_view(source)
    assert result.view != source
    forged = deepcopy(result.manifest)
    forged.update(view_sha256=_sha(source), view_bytes=len(source),
                  pieces=[{"kind": "keep", "start": 0, "end": len(source)}])
    with pytest.raises(codec.SourceViewError, match="profile replay"):
        _restore(source, result, view=source, manifest=forged, expected_manifest_sha256=_manifest_sha(forged))


def test_rehashed_partition_gap_is_refused_even_with_independently_nominated_new_hash():
    source = b"VALUE = 1  \n"
    result = codec.build_source_view(source)
    forged = deepcopy(result.manifest)
    forged["pieces"][0]["start"] = 1
    with pytest.raises(codec.SourceViewError, match="partition"):
        _restore(source, result, manifest=forged, expected_manifest_sha256=_manifest_sha(forged))


def test_same_length_original_source_change_invalidates_inverse():
    source = b"VALUE = 1  \n"
    result = codec.build_source_view(source)
    with pytest.raises(codec.SourceViewError, match="source/view pins"):
        _restore(source.replace(b"1", b"2"), result)


def test_codec_implementation_copy_drift_invalidates_retained_manifest(tmp_path, monkeypatch):
    source = b"VALUE = 1  \n"
    result = codec.build_source_view(source)
    changed = tmp_path / "isolated_codec.py"
    changed.write_bytes(Path(codec.__file__).read_bytes() + b"\n# authored implementation drift\n")
    monkeypatch.setattr(codec, "__file__", str(changed))
    with pytest.raises(codec.SourceViewError, match="codec binding"):
        _restore(source, result)


def test_rehashed_manifest_cannot_grant_completion_authority():
    source = b"VALUE = 1  \n"
    result = codec.build_source_view(source)
    forged = deepcopy(result.manifest)
    forged["completion_authority"] = True
    with pytest.raises(codec.SourceViewError, match="manifest differs"):
        _restore(source, result, manifest=forged, expected_manifest_sha256=_manifest_sha(forged))


def test_deterministic_results_do_not_share_mutable_manifest_state():
    source = b"VALUE = 1  \n"
    first = codec.build_source_view(source)
    second = codec.build_source_view(source)
    assert first.view == second.view and first.manifest == second.manifest
    assert first.manifest_sha256 == second.manifest_sha256 == _manifest_sha(first.manifest)
    first.manifest["pieces"][0]["end"] = 1
    assert second.manifest["pieces"][0]["end"] != 1
    assert _restore(source, second) == source
