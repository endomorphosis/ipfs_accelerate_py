"""Bounded full-file verification using authored bytes and tiny tokenizers."""

import hashlib
import os
import tracemalloc
from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import local_prompt_tokenizer as local
from test.api.test_local_prompt_tokenizer import MODEL, PROVIDER, make_binding


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("size", [1, 65_535, 65_536, 65_537, 131_079])
def test_verification_authenticates_all_chunk_boundaries_without_capture(tmp_path, monkeypatch, size):
    path = tmp_path / "authored-bytes"
    raw = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    path.write_bytes(raw)
    requests, returned = [], []
    actual = local.os.read

    def observed(fd, count):
        requests.append(count)
        chunk = actual(fd, count)
        returned.append(len(chunk))
        return chunk

    def forbidden(*args, **kwargs):
        raise AssertionError("reverification must not capture a whole file")

    monkeypatch.setattr(local.os, "read", observed)
    monkeypatch.setattr(local.os, "fdopen", forbidden)
    monkeypatch.setattr(local, "_bounded_local_bytes", forbidden)
    assert local._verify_local_file(path, maximum=size, expected_sha256=digest(raw)) is None
    assert sum(returned) == size and returned[-1] == 0
    assert max(requests) <= min(local.VERIFICATION_CHUNK_BYTES, size + 1)
    assert requests[-1] == 1


def test_short_reads_still_hash_the_complete_file(tmp_path, monkeypatch):
    raw = b"authored-full-file" * 13
    path = tmp_path / "short-read-fixture"
    path.write_bytes(raw)
    actual = local.os.read
    returned = []

    def short_read(fd, count):
        chunk = actual(fd, min(count, 7))
        returned.append(len(chunk))
        return chunk

    monkeypatch.setattr(local.os, "read", short_read)
    local._verify_local_file(path, maximum=len(raw), expected_sha256=digest(raw))
    assert sum(returned) == len(raw) and returned[-1] == 0


@pytest.mark.parametrize("position", [0, 65_536, -1])
def test_wrong_hash_at_start_boundary_or_tail_refuses(tmp_path, position):
    raw = b"x" * (local.VERIFICATION_CHUNK_BYTES * 2 + 7)
    path = tmp_path / "changed-bytes"
    altered = bytearray(raw)
    altered[position] = ord("y")
    path.write_bytes(altered)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        local._verify_local_file(path, maximum=len(raw), expected_sha256=digest(raw))


@pytest.mark.parametrize("mutation", ["grow", "truncate", "overwrite"])
def test_mutation_between_chunks_refuses_and_keeps_read_bound(tmp_path, monkeypatch, mutation):
    raw = b"x" * (local.VERIFICATION_CHUNK_BYTES * 2 + 7)
    path = tmp_path / "during-read"
    path.write_bytes(raw)
    actual = local.os.read
    seen = []

    def changing(fd, count):
        chunk = actual(fd, count)
        seen.append(len(chunk))
        if len(seen) == 1:
            if mutation == "grow":
                with path.open("ab") as stream:
                    stream.write(b"y" * local.VERIFICATION_CHUNK_BYTES)
            elif mutation == "truncate":
                with path.open("wb") as stream:
                    stream.truncate(1)
            else:
                with path.open("r+b") as stream:
                    stream.seek(local.VERIFICATION_CHUNK_BYTES + 3)
                    stream.write(b"y")
        return chunk

    monkeypatch.setattr(local.os, "read", changing)
    with pytest.raises(ValueError, match="changed during read"):
        local._verify_local_file(path, maximum=len(raw), expected_sha256=digest(raw))
    assert sum(seen) <= len(raw) + 1


def test_same_length_rewrite_after_eof_refuses_before_return(tmp_path, monkeypatch):
    raw = b"authored-bytes"
    path = tmp_path / "after-eof"
    path.write_bytes(raw)
    actual = local.os.read

    def after_eof(fd, count):
        chunk = actual(fd, count)
        if not chunk:
            path.write_bytes(b"changed-bytes")
        return chunk

    monkeypatch.setattr(local.os, "read", after_eof)
    with pytest.raises(ValueError, match="changed during read"):
        local._verify_local_file(path, maximum=len(raw), expected_sha256=digest(raw))


def test_premature_eof_never_authenticates_a_prefix(tmp_path, monkeypatch):
    raw = b"x" * (local.VERIFICATION_CHUNK_BYTES + 7)
    path = tmp_path / "incomplete-read"
    path.write_bytes(raw)
    actual = local.os.read
    calls = 0

    def premature(fd, count):
        nonlocal calls
        calls += 1
        return actual(fd, count) if calls == 1 else b""

    monkeypatch.setattr(local.os, "read", premature)
    with pytest.raises(ValueError, match="changed during read"):
        local._verify_local_file(path, maximum=len(raw), expected_sha256=digest(raw))


@pytest.mark.parametrize("kind", ["empty", "directory", "fifo", "oversized"])
def test_nonregular_empty_or_oversized_files_refuse_before_read(tmp_path, monkeypatch, kind):
    path = tmp_path / "unacceptable-file"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "empty":
        path.touch()
    else:
        path.write_bytes(b"xx")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid file must refuse before read")

    monkeypatch.setattr(local.os, "read", forbidden)
    with pytest.raises(ValueError, match="bounded nonempty regular file"):
        local._verify_local_file(path, maximum=1, expected_sha256=digest(b"x"))


@pytest.mark.parametrize("kind", ["final", "ancestor"])
def test_streamed_verification_preserves_no_follow_checks(tmp_path, monkeypatch, kind):
    directory = tmp_path / "actual-directory"
    directory.mkdir()
    path = directory / "actual-file"
    path.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(path if kind == "final" else directory, target_is_directory=kind == "ancestor")
    candidate = link if kind == "final" else link / path.name

    def forbidden(*args, **kwargs):
        raise AssertionError("symbolic link must refuse before read")

    monkeypatch.setattr(local.os, "read", forbidden)
    with pytest.raises(ValueError, match="symbolic link"):
        local._verify_local_file(candidate, maximum=1, expected_sha256=digest(b"x"))


def test_verification_memory_is_bounded_by_chunks_not_asset_size(tmp_path):
    raw = b"authored-bytes" * 262_144
    path = tmp_path / "large-authored-file"
    path.write_bytes(raw)
    expected = digest(raw)
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        local._verify_local_file(path, maximum=len(raw), expected_sha256=expected)
        _, peak = tracemalloc.get_traced_memory()
        # Includes directory/file handles and hash state as well as the chunk.
        # An accidental complete-file capture exceeds this independent bound.
        assert peak - baseline < local.VERIFICATION_CHUNK_BYTES * 4
    finally:
        if not was_tracing:
            tracemalloc.stop()


def test_initial_load_captures_json_once_but_counts_only_stream(tmp_path, monkeypatch):
    artifact, expected, _ = make_binding(tmp_path)
    original = local._bounded_local_bytes
    captures = []

    def captured(path, **kwargs):
        captures.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(local, "_bounded_local_bytes", captured)
    loaded = local.load_owner_prompt_tokenizer(
        artifact=artifact, expected_sha256=expected, provider=PROVIDER, model=MODEL)
    assert captures == [artifact.name, "tokenizer.json"]

    def forbidden(*args, **kwargs):
        raise AssertionError("counts must not allocate captured tokenizer JSON")

    monkeypatch.setattr(local, "_bounded_local_bytes", forbidden)
    monkeypatch.setattr(local.os, "fdopen", forbidden)
    assert loaded.count("hello world café 法律") == 4


def test_every_count_keeps_full_checks_on_both_sides_of_actual_encoding(tmp_path, monkeypatch):
    artifact, expected, descriptor = make_binding(tmp_path)
    loaded = local.load_owner_prompt_tokenizer(
        artifact=artifact, expected_sha256=expected, provider=PROVIDER, model=MODEL)
    actual_verify, actual_runtime = local._verify_local_file, local._runtime
    actual_tokenizer = loaded._tokenizer
    calls = []

    def verified(path, **kwargs):
        calls.append(path.name)
        return actual_verify(path, **kwargs)

    def runtime(version):
        calls.append("runtime:" + version)
        return actual_runtime(version)

    class ObservedTokenizer:
        truncation = padding = None

        def encode(self, text, *, add_special_tokens):
            calls.append("encode:" + text)
            return actual_tokenizer.encode(text, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(local, "_verify_local_file", verified)
    monkeypatch.setattr(local, "_runtime", runtime)
    observed = replace(loaded, _tokenizer=ObservedTokenizer())
    assert observed.count("hello world café 法律") == 4
    verification = [artifact.name, "tokenizer.json", "runtime:" + descriptor["tokenizers_version"]]
    assert calls == [*verification, "encode:hello world café 法律", *verification]
    calls.clear()
    assert observed.count("hello world") == 2
    assert calls == [*verification, "encode:hello world", *verification]
