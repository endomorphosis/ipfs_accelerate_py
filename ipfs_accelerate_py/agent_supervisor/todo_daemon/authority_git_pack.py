"""Bounded verification for exact, whole-object Git packs.

The verifier in this module is deliberately independent of a Git checkout.
It accepts the complete bytes of one pack and its version-2 index, verifies
both cryptographic trailers, and reconstructs every non-delta object from the
wire bytes.  Only commit, tree, and blob objects are authority-bearing here;
tags, deltas, and reserved pack types fail closed.
"""

from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_PACK_HEADER_SIZE = 12
_INDEX_MAGIC = b"\xfftOc"
_INDEX_VERSION = 2
_INDEX_FANOUT_SIZE = 256 * 4
_JSON_ENVELOPE_EMPTY_SIZE = len(b'{"index_base64":"","pack_base64":""}')
_OBJECT_TYPES = {1: "commit", 2: "tree", 3: "blob"}


class GitPackVerificationError(ValueError):
    """A stable, fail-closed Git pack verification failure."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        message = self.reason if not self.detail else f"{self.reason}: {self.detail}"
        super().__init__(message)


@dataclass(frozen=True)
class GitPackLimits:
    """Hard input and expansion limits for one pack/index pair.

    ``max_json_bytes`` applies to the compact canonical evidence envelope
    ``{"index_base64":"...","pack_base64":"..."}``.  It is checked from
    encoded lengths before base64 decoding and from decoded lengths when the
    raw-bytes API is used.
    """

    max_pack_bytes: int
    max_index_bytes: int
    max_combined_bytes: int
    max_json_bytes: int
    max_objects: int
    max_object_bytes: int
    max_total_inflated_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "max_pack_bytes",
            "max_index_bytes",
            "max_combined_bytes",
            "max_json_bytes",
            "max_objects",
            "max_object_bytes",
            "max_total_inflated_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class GitPackObject:
    """One cryptographically reconstructed whole object from a pack."""

    object_type: str
    raw: bytes = field(repr=False)
    offset: int
    packed_crc32: int

    @property
    def type(self) -> str:
        """Alias matching Git's object-header terminology."""

        return self.object_type

    @property
    def packed_crc(self) -> int:
        """Alias for the unsigned CRC-32 stored in an idx v2 record."""

        return self.packed_crc32


@dataclass(frozen=True)
class VerifiedGitPack(Mapping[str, GitPackObject]):
    """An immutable OID-to-object mapping plus verified wire identities."""

    records: Mapping[str, GitPackObject] = field(repr=False)
    object_format: str
    pack_version: int
    pack_checksum: str
    index_checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))

    def __getitem__(self, oid: str) -> GitPackObject:
        return self.records[oid]

    def __iter__(self) -> Iterator[str]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def object_count(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class _IndexEntry:
    oid: bytes
    crc32: int
    offset: int


@dataclass(frozen=True)
class _ParsedIndex:
    entries: tuple[_IndexEntry, ...]
    pack_checksum: bytes
    index_checksum: bytes


def _digest(object_format: str, value: bytes) -> bytes:
    try:
        digest = hashlib.new(object_format, usedforsecurity=False)
    except TypeError:  # pragma: no cover - compatibility with older Python
        digest = hashlib.new(object_format)
    digest.update(value)
    return digest.digest()


def _hash_size(object_format: Any) -> int:
    if type(object_format) is not str or object_format not in {"sha1", "sha256"}:
        raise GitPackVerificationError("unsupported_object_format")
    return 20 if object_format == "sha1" else 32


def _base64_size(byte_count: int) -> int:
    return 4 * ((byte_count + 2) // 3)


def _base64_decoded_size(value: str) -> int:
    """Return a syntactically possible decoded size without allocating it."""

    if len(value) % 4:
        raise GitPackVerificationError("base64_invalid")
    padding = 2 if value.endswith("==") else 1 if value.endswith("=") else 0
    if (padding and len(value) < 4) or "=" in value[: len(value) - padding]:
        raise GitPackVerificationError("base64_invalid")
    return (len(value) // 4 * 3) - padding


def git_pack_json_size(pack_size: int, index_size: int) -> int:
    """Return the exact compact canonical base64-envelope byte size."""

    for name, value in (("pack_size", pack_size), ("index_size", index_size)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    return (
        _JSON_ENVELOPE_EMPTY_SIZE + _base64_size(pack_size) + _base64_size(index_size)
    )


def _check_raw_input_limits(
    pack_bytes: Any,
    index_bytes: Any,
    limits: GitPackLimits,
) -> tuple[bytes, bytes]:
    if type(limits) is not GitPackLimits:
        raise GitPackVerificationError("invalid_limits")
    if type(pack_bytes) is not bytes or type(index_bytes) is not bytes:
        raise GitPackVerificationError("wire_bytes_required")
    pack_size = len(pack_bytes)
    index_size = len(index_bytes)
    if pack_size > limits.max_pack_bytes:
        raise GitPackVerificationError("pack_byte_limit")
    if index_size > limits.max_index_bytes:
        raise GitPackVerificationError("index_byte_limit")
    if pack_size + index_size > limits.max_combined_bytes:
        raise GitPackVerificationError("combined_byte_limit")
    if git_pack_json_size(pack_size, index_size) > limits.max_json_bytes:
        raise GitPackVerificationError("json_byte_limit")
    return pack_bytes, index_bytes


def _u32(value: bytes, offset: int) -> int:
    return struct.unpack_from(">I", value, offset)[0]


def _parse_index(
    index_bytes: bytes,
    *,
    object_format: str,
    hash_size: int,
    expected_pack_checksum: bytes,
    limits: GitPackLimits,
) -> _ParsedIndex:
    minimum_size = 8 + _INDEX_FANOUT_SIZE + (2 * hash_size)
    if len(index_bytes) < minimum_size:
        raise GitPackVerificationError("index_truncated")
    if index_bytes[:4] != _INDEX_MAGIC:
        raise GitPackVerificationError("index_magic_invalid")
    if _u32(index_bytes, 4) != _INDEX_VERSION:
        raise GitPackVerificationError("index_version_invalid")

    fanout_start = 8
    fanout = tuple(
        _u32(index_bytes, fanout_start + (position * 4)) for position in range(256)
    )
    if any(left > right for left, right in zip(fanout, fanout[1:])):
        raise GitPackVerificationError("index_fanout_invalid")
    object_count = fanout[-1]
    if object_count > limits.max_objects:
        raise GitPackVerificationError("object_count_limit")

    oid_start = fanout_start + _INDEX_FANOUT_SIZE
    oid_end = oid_start + (object_count * hash_size)
    crc_end = oid_end + (object_count * 4)
    offset_end = crc_end + (object_count * 4)
    minimum_exact_size = offset_end + (2 * hash_size)
    if len(index_bytes) < minimum_exact_size:
        raise GitPackVerificationError("index_truncated")

    oids = tuple(
        index_bytes[
            oid_start
            + (position * hash_size) : oid_start
            + ((position + 1) * hash_size)
        ]
        for position in range(object_count)
    )
    if any(left >= right for left, right in zip(oids, oids[1:])):
        raise GitPackVerificationError("index_oids_not_strictly_sorted")
    expected_fanout: list[int] = [0] * 256
    for oid in oids:
        expected_fanout[oid[0]] += 1
    running = 0
    for position, count in enumerate(expected_fanout):
        running += count
        expected_fanout[position] = running
    if tuple(expected_fanout) != fanout:
        raise GitPackVerificationError("index_fanout_invalid")

    crcs = tuple(
        _u32(index_bytes, oid_end + (position * 4)) for position in range(object_count)
    )
    offset_words = tuple(
        _u32(index_bytes, crc_end + (position * 4)) for position in range(object_count)
    )
    large_count = sum(1 for value in offset_words if value & 0x80000000)
    expected_size = minimum_exact_size + (large_count * 8)
    if len(index_bytes) != expected_size:
        raise GitPackVerificationError("index_size_mismatch")

    large_start = offset_end
    large_offsets = tuple(
        struct.unpack_from(">Q", index_bytes, large_start + (position * 8))[0]
        for position in range(large_count)
    )
    pack_checksum_start = large_start + (large_count * 8)
    index_checksum_start = pack_checksum_start + hash_size
    pack_checksum = index_bytes[pack_checksum_start:index_checksum_start]
    index_checksum = index_bytes[index_checksum_start:]
    if _digest(object_format, index_bytes[:index_checksum_start]) != index_checksum:
        raise GitPackVerificationError("index_checksum_mismatch")
    if pack_checksum != expected_pack_checksum:
        raise GitPackVerificationError("index_pack_checksum_mismatch")

    used_large_offsets: list[int] = []
    resolved_offsets: list[int] = []
    for offset_word in offset_words:
        if offset_word & 0x80000000:
            large_index = offset_word & 0x7FFFFFFF
            if large_index >= large_count:
                raise GitPackVerificationError("index_large_offset_invalid")
            used_large_offsets.append(large_index)
            resolved = large_offsets[large_index]
            if resolved < 0x80000000:
                raise GitPackVerificationError("index_large_offset_noncanonical")
        else:
            resolved = offset_word
        resolved_offsets.append(resolved)
    if sorted(used_large_offsets) != list(range(large_count)):
        raise GitPackVerificationError("index_large_offset_invalid")

    entries = tuple(
        _IndexEntry(oid=oid, crc32=crc, offset=offset)
        for oid, crc, offset in zip(oids, crcs, resolved_offsets)
    )
    return _ParsedIndex(
        entries=entries,
        pack_checksum=pack_checksum,
        index_checksum=index_checksum,
    )


def _parse_object_header(
    entry_bytes: bytes,
    *,
    max_object_bytes: int,
) -> tuple[str, int, int]:
    if not entry_bytes:
        raise GitPackVerificationError("pack_entry_truncated")
    first = entry_bytes[0]
    type_code = (first >> 4) & 0x07
    object_type = _OBJECT_TYPES.get(type_code)
    if object_type is None:
        raise GitPackVerificationError("unsupported_pack_object_type", str(type_code))

    declared_size = first & 0x0F
    shift = 4
    position = 1
    current = first
    while current & 0x80:
        if position >= len(entry_bytes) or position >= 10:
            raise GitPackVerificationError("pack_object_header_invalid")
        current = entry_bytes[position]
        declared_size |= (current & 0x7F) << shift
        if declared_size > max_object_bytes:
            raise GitPackVerificationError("object_byte_limit")
        shift += 7
        position += 1
    if declared_size > max_object_bytes:
        raise GitPackVerificationError("object_byte_limit")
    return object_type, declared_size, position


def _inflate_exact(
    compressed: bytes,
    *,
    declared_size: int,
) -> bytes:
    if not compressed:
        raise GitPackVerificationError("pack_object_zlib_truncated")
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(compressed, declared_size + 1)
    except zlib.error as exc:
        raise GitPackVerificationError("pack_object_zlib_invalid") from exc
    if len(raw) != declared_size:
        raise GitPackVerificationError("pack_object_size_mismatch")
    if not inflater.eof:
        raise GitPackVerificationError("pack_object_zlib_truncated")
    if inflater.unconsumed_tail or inflater.unused_data:
        raise GitPackVerificationError("pack_object_zlib_trailing_data")
    return raw


def verify_exact_git_pack(
    pack_bytes: bytes,
    index_bytes: bytes,
    *,
    object_format: str,
    limits: GitPackLimits,
) -> VerifiedGitPack:
    """Verify and reconstruct one exact, non-delta Git pack/index pair.

    The returned object is itself an immutable mapping from lowercase OID hex
    to :class:`GitPackObject`.  Its checksum fields expose the independently
    verified pack and index trailer identities.
    """

    pack_bytes, index_bytes = _check_raw_input_limits(pack_bytes, index_bytes, limits)
    hash_size = _hash_size(object_format)
    if len(pack_bytes) < _PACK_HEADER_SIZE + hash_size:
        raise GitPackVerificationError("pack_truncated")
    if pack_bytes[:4] != b"PACK":
        raise GitPackVerificationError("pack_magic_invalid")
    pack_version = _u32(pack_bytes, 4)
    if pack_version not in {2, 3}:
        raise GitPackVerificationError("pack_version_invalid")
    pack_object_count = _u32(pack_bytes, 8)
    if pack_object_count > limits.max_objects:
        raise GitPackVerificationError("object_count_limit")

    pack_body_end = len(pack_bytes) - hash_size
    pack_checksum = pack_bytes[pack_body_end:]
    if _digest(object_format, pack_bytes[:pack_body_end]) != pack_checksum:
        raise GitPackVerificationError("pack_checksum_mismatch")
    parsed_index = _parse_index(
        index_bytes,
        object_format=object_format,
        hash_size=hash_size,
        expected_pack_checksum=pack_checksum,
        limits=limits,
    )
    if len(parsed_index.entries) != pack_object_count:
        raise GitPackVerificationError("pack_index_object_count_mismatch")

    entries_by_offset: dict[int, _IndexEntry] = {}
    for index_entry in parsed_index.entries:
        if index_entry.offset in entries_by_offset:
            raise GitPackVerificationError("index_duplicate_offset")
        if not _PACK_HEADER_SIZE <= index_entry.offset < pack_body_end:
            raise GitPackVerificationError("index_offset_out_of_range")
        entries_by_offset[index_entry.offset] = index_entry
    physical_offsets = sorted(entries_by_offset)
    if pack_object_count:
        if not physical_offsets or physical_offsets[0] != _PACK_HEADER_SIZE:
            raise GitPackVerificationError("pack_entry_partition_invalid")
    elif pack_body_end != _PACK_HEADER_SIZE:
        raise GitPackVerificationError("pack_entry_partition_invalid")

    records_by_oid: dict[str, GitPackObject] = {}
    total_inflated = 0
    for position, offset in enumerate(physical_offsets):
        end = (
            physical_offsets[position + 1]
            if position + 1 < len(physical_offsets)
            else pack_body_end
        )
        if end <= offset:
            raise GitPackVerificationError("pack_entry_partition_invalid")
        packed_entry = pack_bytes[offset:end]
        object_type, declared_size, header_size = _parse_object_header(
            packed_entry,
            max_object_bytes=limits.max_object_bytes,
        )
        if total_inflated + declared_size > limits.max_total_inflated_bytes:
            raise GitPackVerificationError("total_inflated_byte_limit")
        raw = _inflate_exact(
            packed_entry[header_size:],
            declared_size=declared_size,
        )
        total_inflated += len(raw)
        index_entry = entries_by_offset[offset]
        packed_crc32 = zlib.crc32(packed_entry) & 0xFFFFFFFF
        if packed_crc32 != index_entry.crc32:
            raise GitPackVerificationError("object_crc_mismatch")
        object_header = f"{object_type} {len(raw)}\0".encode("ascii")
        observed_oid = _digest(object_format, object_header + raw)
        if observed_oid != index_entry.oid:
            raise GitPackVerificationError("object_oid_mismatch")
        oid_text = observed_oid.hex()
        records_by_oid[oid_text] = GitPackObject(
            object_type=object_type,
            raw=raw,
            offset=offset,
            packed_crc32=packed_crc32,
        )

    if len(records_by_oid) != pack_object_count:
        raise GitPackVerificationError("pack_object_population_mismatch")
    ordered_records = {
        entry.oid.hex(): records_by_oid[entry.oid.hex()]
        for entry in parsed_index.entries
    }
    return VerifiedGitPack(
        records=ordered_records,
        object_format=object_format,
        pack_version=pack_version,
        pack_checksum=pack_checksum.hex(),
        index_checksum=parsed_index.index_checksum.hex(),
    )


def verify_exact_git_pack_base64(
    pack_base64: str,
    index_base64: str,
    *,
    object_format: str,
    limits: GitPackLimits,
) -> VerifiedGitPack:
    """Decode canonical bounded base64 evidence, then verify its wire bytes."""

    if type(limits) is not GitPackLimits:
        raise GitPackVerificationError("invalid_limits")
    if type(pack_base64) is not str or type(index_base64) is not str:
        raise GitPackVerificationError("base64_text_required")
    if not pack_base64.isascii() or not index_base64.isascii():
        raise GitPackVerificationError("base64_invalid")
    pack_encoded_size = len(pack_base64)
    index_encoded_size = len(index_base64)
    if pack_encoded_size > _base64_size(limits.max_pack_bytes):
        raise GitPackVerificationError("pack_encoded_byte_limit")
    if index_encoded_size > _base64_size(limits.max_index_bytes):
        raise GitPackVerificationError("index_encoded_byte_limit")
    if (
        _JSON_ENVELOPE_EMPTY_SIZE + pack_encoded_size + index_encoded_size
        > limits.max_json_bytes
    ):
        raise GitPackVerificationError("json_byte_limit")
    decoded_pack_size = _base64_decoded_size(pack_base64)
    decoded_index_size = _base64_decoded_size(index_base64)
    if decoded_pack_size > limits.max_pack_bytes:
        raise GitPackVerificationError("pack_byte_limit")
    if decoded_index_size > limits.max_index_bytes:
        raise GitPackVerificationError("index_byte_limit")
    if decoded_pack_size + decoded_index_size > limits.max_combined_bytes:
        raise GitPackVerificationError("combined_byte_limit")
    try:
        pack_ascii = pack_base64.encode("ascii")
        index_ascii = index_base64.encode("ascii")
        decoded_pack = base64.b64decode(pack_ascii, validate=True)
        decoded_index = base64.b64decode(index_ascii, validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise GitPackVerificationError("base64_invalid") from exc
    if (
        base64.b64encode(decoded_pack) != pack_ascii
        or base64.b64encode(decoded_index) != index_ascii
    ):
        raise GitPackVerificationError("base64_noncanonical")
    return verify_exact_git_pack(
        decoded_pack,
        decoded_index,
        object_format=object_format,
        limits=limits,
    )


__all__ = [
    "GitPackLimits",
    "GitPackObject",
    "GitPackVerificationError",
    "VerifiedGitPack",
    "git_pack_json_size",
    "verify_exact_git_pack",
    "verify_exact_git_pack_base64",
]
