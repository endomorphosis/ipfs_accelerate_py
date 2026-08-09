"""Pure adversarial checks for bounded whole-object Git pack verification."""

from __future__ import annotations

import base64
import hashlib
import os
import struct
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.authority_git_pack import (
    GitPackLimits,
    GitPackVerificationError,
    VerifiedGitPack,
    git_pack_json_size,
    verify_exact_git_pack,
    verify_exact_git_pack_base64,
)


@dataclass(frozen=True)
class _PackFixture:
    repository: Path
    object_format: str
    object_ids: tuple[str, ...]
    pack: bytes
    index: bytes
    limits: GitPackLimits


def _git(
    repository: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        input=input_bytes,
        stdin=subprocess.DEVNULL if input_bytes is None else None,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return completed.stdout


def _hash(object_format: str, value: bytes) -> bytes:
    try:
        digest = hashlib.new(object_format, usedforsecurity=False)
    except TypeError:  # pragma: no cover - compatibility with older Python
        digest = hashlib.new(object_format)
    digest.update(value)
    return digest.digest()


def _pack_objects(
    repository: Path,
    output_root: Path,
    object_ids: tuple[str, ...],
    *,
    label: str,
) -> tuple[bytes, bytes]:
    pack = _git(
        repository,
        "pack-objects",
        "--stdout",
        "--no-revs",
        "--missing=error",
        "--no-write-bitmap-index",
        "--no-reuse-delta",
        "--no-reuse-object",
        "--no-thin",
        "--no-include-tag",
        "--threads=1",
        "--window=0",
        "--depth=0",
        input_bytes="".join(f"{oid}\n" for oid in object_ids).encode("ascii"),
    )
    pack_path = output_root / f"{label}.pack"
    index_path = output_root / f"{label}.idx"
    pack_path.write_bytes(pack)
    _git(
        repository,
        "index-pack",
        "--strict",
        "-o",
        str(index_path),
        str(pack_path),
    )
    return pack, index_path.read_bytes()


def _fixture(tmp_path: Path, object_format: str = "sha1") -> _PackFixture:
    repository = tmp_path / f"repository-{object_format}"
    repository.mkdir()
    _git(repository, "init", "-q", "-b", "main", f"--object-format={object_format}")
    _git(repository, "config", "user.name", "Pack Verifier")
    _git(repository, "config", "user.email", "pack-verifier@example.invalid")
    (repository / "alpha.txt").write_bytes(b"alpha exact bytes\n")
    (repository / "beta.txt").write_bytes(b"beta exact bytes\n")
    _git(repository, "add", "alpha.txt", "beta.txt")
    _git(repository, "commit", "-qm", "seed exact objects")
    object_ids = tuple(
        sorted(
            line.split(maxsplit=1)[0].decode("ascii")
            for line in _git(repository, "rev-list", "--objects", "HEAD").splitlines()
        )
    )
    pack, index = _pack_objects(
        repository,
        tmp_path,
        object_ids,
        label=f"whole-{object_format}",
    )
    limits = GitPackLimits(
        max_pack_bytes=len(pack) + 1024,
        max_index_bytes=len(index) + 1024,
        max_combined_bytes=len(pack) + len(index) + 2048,
        max_json_bytes=git_pack_json_size(len(pack), len(index)) + 2048,
        max_objects=128,
        max_object_bytes=1024 * 1024,
        max_total_inflated_bytes=4 * 1024 * 1024,
    )
    return _PackFixture(
        repository=repository,
        object_format=object_format,
        object_ids=object_ids,
        pack=pack,
        index=index,
        limits=limits,
    )


def _hash_size(object_format: str) -> int:
    return 20 if object_format == "sha1" else 32


def _rehash_pack(pack: bytes, object_format: str) -> bytes:
    hash_size = _hash_size(object_format)
    body = pack[:-hash_size]
    return body + _hash(object_format, body)


def _rehash_index(index: bytes, object_format: str) -> bytes:
    hash_size = _hash_size(object_format)
    body = index[:-hash_size]
    return body + _hash(object_format, body)


def _bind_index_to_pack(
    index: bytes,
    pack: bytes,
    object_format: str,
) -> bytes:
    hash_size = _hash_size(object_format)
    body = bytearray(index[:-hash_size])
    body[-hash_size:] = pack[-hash_size:]
    return bytes(body) + _hash(object_format, bytes(body))


def _index_layout(index: bytes, object_format: str) -> tuple[int, int, int, int]:
    object_count = struct.unpack_from(">I", index, 8 + (255 * 4))[0]
    hash_size = _hash_size(object_format)
    oid_start = 8 + (256 * 4)
    crc_start = oid_start + (object_count * hash_size)
    offset_start = crc_start + (object_count * 4)
    return object_count, oid_start, crc_start, offset_start


def _mutate_index_oid(index: bytes, object_format: str) -> bytes:
    object_count, oid_start, _crc_start, _offset_start = _index_layout(
        index, object_format
    )
    hash_size = _hash_size(object_format)
    oids = [
        index[
            oid_start
            + (position * hash_size) : oid_start
            + ((position + 1) * hash_size)
        ]
        for position in range(object_count)
    ]
    for position, oid in enumerate(oids):
        suffix_bits = (hash_size - 1) * 8
        prefix_low = oid[0] << suffix_bits
        prefix_high = ((oid[0] + 1) << suffix_bits) - 1
        lower = max(
            prefix_low,
            int.from_bytes(oids[position - 1], "big") + 1 if position else prefix_low,
        )
        upper = min(
            prefix_high,
            (
                int.from_bytes(oids[position + 1], "big") - 1
                if position + 1 < len(oids)
                else prefix_high
            ),
        )
        original = int.from_bytes(oid, "big")
        candidate = lower if lower != original else upper
        if lower <= candidate <= upper and candidate != original:
            mutated = bytearray(index)
            start = oid_start + (position * hash_size)
            mutated[start : start + hash_size] = candidate.to_bytes(hash_size, "big")
            return _rehash_index(bytes(mutated), object_format)
    raise AssertionError("fixture unexpectedly has no mutable OID gap")


def _assert_rejected(
    pack: bytes,
    index: bytes,
    fixture: _PackFixture,
    reason: str,
    *,
    limits: GitPackLimits | None = None,
) -> None:
    with pytest.raises(GitPackVerificationError) as caught:
        verify_exact_git_pack(
            pack,
            index,
            object_format=fixture.object_format,
            limits=limits or fixture.limits,
        )
    assert caught.value.reason == reason


@pytest.mark.parametrize("object_format", ("sha1", "sha256"))
def test_generated_whole_pack_and_idx_v2_are_reconstructed_exactly(
    tmp_path: Path,
    object_format: str,
) -> None:
    fixture = _fixture(tmp_path, object_format)

    verified = verify_exact_git_pack(
        fixture.pack,
        fixture.index,
        object_format=object_format,
        limits=fixture.limits,
    )

    assert isinstance(verified, VerifiedGitPack)
    assert tuple(verified) == fixture.object_ids
    assert verified.object_count == len(fixture.object_ids) == 4
    assert verified.pack_version == 2
    assert verified.pack_checksum == fixture.pack[-_hash_size(object_format) :].hex()
    assert verified.index_checksum == fixture.index[-_hash_size(object_format) :].hex()
    assert min(record.offset for record in verified.values()) == 12
    for oid, record in verified.items():
        assert record.object_type in {"blob", "commit", "tree"}
        assert record.type == record.object_type
        assert record.packed_crc == record.packed_crc32
        assert record.raw == _git(
            fixture.repository, "cat-file", record.object_type, oid
        )
        assert (
            _hash(
                object_format,
                f"{record.object_type} {len(record.raw)}\0".encode("ascii")
                + record.raw,
            ).hex()
            == oid
        )
    with pytest.raises(TypeError):
        verified.records["forged"] = next(iter(verified.values()))  # type: ignore[index]


def test_canonical_base64_entrypoint_caps_then_decodes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    pack_base64 = base64.b64encode(fixture.pack).decode("ascii")
    index_base64 = base64.b64encode(fixture.index).decode("ascii")

    verified = verify_exact_git_pack_base64(
        pack_base64,
        index_base64,
        object_format="sha1",
        limits=fixture.limits,
    )

    assert tuple(verified) == fixture.object_ids
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    assert index_base64.endswith("=")
    data_position = -3 if index_base64.endswith("==") else -2
    original = alphabet.index(index_base64[data_position])
    alternate = alphabet[original + 1]
    noncanonical = (
        index_base64[:data_position] + alternate + index_base64[data_position + 1 :]
    )
    assert base64.b64decode(noncanonical, validate=True) == fixture.index
    with pytest.raises(GitPackVerificationError) as caught:
        verify_exact_git_pack_base64(
            pack_base64,
            noncanonical,
            object_format="sha1",
            limits=fixture.limits,
        )
    assert caught.value.reason == "base64_noncanonical"

    with pytest.raises(GitPackVerificationError) as caught:
        verify_exact_git_pack_base64(
            pack_base64,
            index_base64,
            object_format="sha1",
            limits=replace(
                fixture.limits,
                max_combined_bytes=len(fixture.pack) + len(fixture.index) - 1,
            ),
        )
    assert caught.value.reason == "combined_byte_limit"


def test_pack_and_index_trailers_are_independently_verified(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bad_pack = bytearray(fixture.pack)
    bad_pack[-1] ^= 1
    _assert_rejected(bytes(bad_pack), fixture.index, fixture, "pack_checksum_mismatch")

    bad_index = bytearray(fixture.index)
    bad_index[-1] ^= 1
    _assert_rejected(fixture.pack, bytes(bad_index), fixture, "index_checksum_mismatch")

    hash_size = _hash_size(fixture.object_format)
    bad_pack_binding = bytearray(fixture.index)
    bad_pack_binding[-(2 * hash_size)] ^= 1
    bad_pack_binding = bytearray(
        _rehash_index(bytes(bad_pack_binding), fixture.object_format)
    )
    _assert_rejected(
        fixture.pack,
        bytes(bad_pack_binding),
        fixture,
        "index_pack_checksum_mismatch",
    )


def test_idx_oid_and_crc_forgery_fail_after_valid_index_checksum(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _assert_rejected(
        fixture.pack,
        _mutate_index_oid(fixture.index, fixture.object_format),
        fixture,
        "object_oid_mismatch",
    )

    _object_count, _oid_start, crc_start, _offset_start = _index_layout(
        fixture.index, fixture.object_format
    )
    bad_crc = bytearray(fixture.index)
    bad_crc[crc_start] ^= 1
    _assert_rejected(
        fixture.pack,
        _rehash_index(bytes(bad_crc), fixture.object_format),
        fixture,
        "object_crc_mismatch",
    )


def test_idx_offsets_must_partition_every_pack_entry_byte(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    object_count, _oid_start, _crc_start, offset_start = _index_layout(
        fixture.index, fixture.object_format
    )
    bad_offset = bytearray(fixture.index)
    replaced = False
    for position in range(object_count):
        current = struct.unpack_from(">I", bad_offset, offset_start + position * 4)[0]
        if current == 12:
            struct.pack_into(">I", bad_offset, offset_start + position * 4, 13)
            replaced = True
            break
    assert replaced
    _assert_rejected(
        fixture.pack,
        _rehash_index(bytes(bad_offset), fixture.object_format),
        fixture,
        "pack_entry_partition_invalid",
    )

    hash_size = _hash_size(fixture.object_format)
    body_with_gap = fixture.pack[:-hash_size] + b"\0"
    trailing_pack = body_with_gap + _hash(fixture.object_format, body_with_gap)
    trailing_index = _bind_index_to_pack(
        fixture.index, trailing_pack, fixture.object_format
    )
    _assert_rejected(
        trailing_pack,
        trailing_index,
        fixture,
        "pack_object_zlib_trailing_data",
    )


def test_idx_large_offset_references_are_strict_and_canonical(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    object_count, _oid_start, _crc_start, offset_start = _index_layout(
        fixture.index, fixture.object_format
    )
    hash_size = _hash_size(fixture.object_format)
    body = bytearray(fixture.index[:-hash_size])
    original_offset = struct.unpack_from(">I", body, offset_start)[0]
    assert original_offset < 0x80000000
    struct.pack_into(">I", body, offset_start, 0x80000000)
    large_table_start = offset_start + (object_count * 4)
    body[large_table_start:large_table_start] = struct.pack(">Q", original_offset)
    forged = bytes(body) + _hash(fixture.object_format, bytes(body))

    _assert_rejected(
        fixture.pack,
        forged,
        fixture,
        "index_large_offset_noncanonical",
    )


@pytest.mark.parametrize("type_code", (0, 4, 5, 6, 7))
def test_tags_deltas_and_reserved_pack_types_are_rejected(
    tmp_path: Path,
    type_code: int,
) -> None:
    fixture = _fixture(tmp_path)
    verified = verify_exact_git_pack(
        fixture.pack,
        fixture.index,
        object_format=fixture.object_format,
        limits=fixture.limits,
    )
    first_offset = min(record.offset for record in verified.values())
    mutated = bytearray(fixture.pack)
    mutated[first_offset] = (mutated[first_offset] & 0x8F) | (type_code << 4)
    forged_pack = _rehash_pack(bytes(mutated), fixture.object_format)
    forged_index = _bind_index_to_pack(
        fixture.index, forged_pack, fixture.object_format
    )

    _assert_rejected(
        forged_pack,
        forged_index,
        fixture,
        "unsupported_pack_object_type",
    )


def test_real_annotated_tag_object_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _git(fixture.repository, "tag", "-a", "-m", "authority tag", "v1")
    tag_oid = _git(fixture.repository, "rev-parse", "v1^{tag}").decode("ascii").strip()
    tag_pack, tag_index = _pack_objects(
        fixture.repository,
        tmp_path,
        (tag_oid,),
        label="tag-only",
    )
    tag_limits = replace(
        fixture.limits,
        max_pack_bytes=len(tag_pack) + 1024,
        max_index_bytes=len(tag_index) + 1024,
        max_combined_bytes=len(tag_pack) + len(tag_index) + 2048,
        max_json_bytes=git_pack_json_size(len(tag_pack), len(tag_index)) + 2048,
    )

    _assert_rejected(
        tag_pack,
        tag_index,
        replace(fixture, limits=tag_limits),
        "unsupported_pack_object_type",
    )


def test_declared_size_and_expansion_limits_fail_before_authority(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    verified = verify_exact_git_pack(
        fixture.pack,
        fixture.index,
        object_format=fixture.object_format,
        limits=fixture.limits,
    )
    target = next(
        record
        for record in sorted(verified.values(), key=lambda item: item.offset)
        if fixture.pack[record.offset] & 0x0F
    )
    mutated = bytearray(fixture.pack)
    mutated[target.offset] -= 1
    forged_pack = _rehash_pack(bytes(mutated), fixture.object_format)
    forged_index = _bind_index_to_pack(
        fixture.index, forged_pack, fixture.object_format
    )
    _assert_rejected(
        forged_pack,
        forged_index,
        fixture,
        "pack_object_size_mismatch",
    )

    largest_object = max(len(record.raw) for record in verified.values())
    _assert_rejected(
        fixture.pack,
        fixture.index,
        fixture,
        "object_byte_limit",
        limits=replace(fixture.limits, max_object_bytes=largest_object - 1),
    )
    total_inflated = sum(len(record.raw) for record in verified.values())
    _assert_rejected(
        fixture.pack,
        fixture.index,
        fixture,
        "total_inflated_byte_limit",
        limits=replace(
            fixture.limits,
            max_total_inflated_bytes=total_inflated - 1,
        ),
    )


@pytest.mark.parametrize(
    ("limit_name", "reason"),
    (
        ("max_pack_bytes", "pack_byte_limit"),
        ("max_index_bytes", "index_byte_limit"),
        ("max_combined_bytes", "combined_byte_limit"),
        ("max_json_bytes", "json_byte_limit"),
        ("max_objects", "object_count_limit"),
    ),
)
def test_wire_count_combined_and_json_caps_are_hard(
    tmp_path: Path,
    limit_name: str,
    reason: str,
) -> None:
    fixture = _fixture(tmp_path)
    exact_values = {
        "max_pack_bytes": len(fixture.pack),
        "max_index_bytes": len(fixture.index),
        "max_combined_bytes": len(fixture.pack) + len(fixture.index),
        "max_json_bytes": git_pack_json_size(len(fixture.pack), len(fixture.index)),
        "max_objects": len(fixture.object_ids),
    }
    limits = replace(fixture.limits, **{limit_name: exact_values[limit_name] - 1})
    _assert_rejected(
        fixture.pack,
        fixture.index,
        fixture,
        reason,
        limits=limits,
    )


def test_limits_and_wire_types_reject_bools_and_mutable_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    values: dict[str, Any] = {
        "max_pack_bytes": len(fixture.pack),
        "max_index_bytes": len(fixture.index),
        "max_combined_bytes": len(fixture.pack) + len(fixture.index),
        "max_json_bytes": git_pack_json_size(len(fixture.pack), len(fixture.index)),
        "max_objects": len(fixture.object_ids),
        "max_object_bytes": 1024 * 1024,
        "max_total_inflated_bytes": 4 * 1024 * 1024,
    }
    for name in values:
        forged = {**values, name: True}
        with pytest.raises(ValueError, match="positive integer"):
            GitPackLimits(**forged)

    with pytest.raises(GitPackVerificationError) as caught:
        verify_exact_git_pack(
            bytearray(fixture.pack),  # type: ignore[arg-type]
            fixture.index,
            object_format=fixture.object_format,
            limits=fixture.limits,
        )
    assert caught.value.reason == "wire_bytes_required"


def test_pack_v3_is_supported_but_unknown_versions_fail(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    version_three = bytearray(fixture.pack)
    struct.pack_into(">I", version_three, 4, 3)
    pack_v3 = _rehash_pack(bytes(version_three), fixture.object_format)
    index_v3 = _bind_index_to_pack(fixture.index, pack_v3, fixture.object_format)
    assert (
        verify_exact_git_pack(
            pack_v3,
            index_v3,
            object_format=fixture.object_format,
            limits=fixture.limits,
        ).pack_version
        == 3
    )

    unknown = bytearray(fixture.pack)
    struct.pack_into(">I", unknown, 4, 4)
    with pytest.raises(GitPackVerificationError) as caught:
        verify_exact_git_pack(
            bytes(unknown),
            fixture.index,
            object_format=fixture.object_format,
            limits=fixture.limits,
        )
    assert caught.value.reason == "pack_version_invalid"
