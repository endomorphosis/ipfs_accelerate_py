"""Bounded observations of an explicitly admitted native Doctor key set.

The host supplies current source/owner bindings independently of cached receipt
hashes. This adapter creates no storage owner and does not grant replacement or
proof authority. Native gate consultation can write dependency and refusal
bookkeeping in its existing owner; this is not a storage-read-only operation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat

from ..proof.doctor_proof_cache import (
    DoctorCacheAuditReceipt, DoctorCacheDisposition, DoctorCacheLookupResult,
    DoctorCacheStage, DoctorIdentityBinding, DoctorProofCacheGate, DoctorProofCacheKey,
)
from ..proof.formal_verification_cache import FormalVerificationCache
from ..proof.formal_verification_contracts import (
    AssuranceLevel, EvidenceFreshness, ProofReceipt, ScopedProofReceipt, ProofVerdict,
    ResourceBudget, canonical_json,
)
from .supervisor_meta_index import public_replay_without_metadata

SCHEMA = "supervisor-formal-status-census@1"
BOUND_SCHEMA = "supervisor-formal-status-census@2"
MAX_KEYS = 32
MAX_REQUEST_BYTES = 256 * 1024
MAX_KEY_BYTES = 64 * 1024
MAX_RECEIPT_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 16 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_KEY_ID = re.compile(r"doctor-proof-cache-key:sha256:[0-9a-f]{64}\Z")
_AUTHORITIES = (
    "proof_authority", "omission_authority", "execution_authority",
    "completion_authority", "publication_authority", "replacement_authority",
    "complete_run_token_savings_claimed",
)


@dataclass(frozen=True, slots=True)
class FormalStatusCensusBinding:
    """Host-admitted current context, not authenticated by a receipt digest.

All SHA256 values are bare lower-case hex. ``expected_key_ids`` is the host's
independently expected complete population, with explicit empty tuple distinct
from unavailable ``None``. Raw identities are never exported by the census.
"""

    task_sha256: str
    revision_sha256: str
    source_sha256: str
    context_sha256: str
    candidate_tree_id: str
    owner_path_sha256: str | None = None
    expected_key_ids: tuple[str, ...] | None = None
    expected_receipt_ids: tuple[tuple[str, str], ...] | None = None


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_json(value, maximum: int) -> str:
    rendered = canonical_json(value)
    if len(rendered.encode("utf-8")) > maximum:
        raise ValueError("bounded canonical value required")
    return rendered


def _binding_payload(binding: FormalStatusCensusBinding) -> dict:
    if type(binding) is not FormalStatusCensusBinding:
        raise ValueError("native binding required")
    value = {name: getattr(binding, name) for name in binding.__dataclass_fields__}
    # Preserve historical @1 binding bytes and digests when the new lane is absent.
    if binding.expected_receipt_ids is None:
        value.pop("expected_receipt_ids")
    for name in ("task_sha256", "revision_sha256", "source_sha256", "context_sha256"):
        if type(value[name]) is not str or not _DIGEST.fullmatch(value[name]):
            raise ValueError("digest binding required")
    tree = binding.candidate_tree_id
    if (type(tree) is not str or not tree or tree != tree.strip()
            or len(tree.encode("utf-8")) > 4096 or any(ord(c) < 32 for c in tree)):
        raise ValueError("bounded current candidate tree required")
    if binding.owner_path_sha256 is not None and (
            type(binding.owner_path_sha256) is not str
            or not _DIGEST.fullmatch(binding.owner_path_sha256)):
        raise ValueError("owner path digest required")
    expected = binding.expected_key_ids
    if expected is not None and (
            type(expected) is not tuple or len(expected) > MAX_KEYS
            or any(type(item) is not str or not _KEY_ID.fullmatch(item) for item in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError("bounded independently expected key population required")
    pairs = binding.expected_receipt_ids
    if pairs is not None:
        if (type(pairs) is not tuple or len(pairs) > MAX_KEYS
                or any(type(pair) is not tuple or len(pair) != 2
                       or type(pair[0]) is not str or not _KEY_ID.fullmatch(pair[0])
                       or type(pair[1]) is not str or not pair[1] or pair[1] != pair[1].strip()
                       or len(pair[1].encode("utf-8")) > 1024
                       or any(ord(c) < 32 for c in pair[1]) for pair in pairs)
                or len({pair[0] for pair in pairs}) != len(pairs)
                or len({pair[1] for pair in pairs}) != len(pairs)
                or expected is None or {pair[0] for pair in pairs} != set(expected)):
            raise ValueError("bounded independently expected key and receipt population required")
    _bounded_json(value, MAX_REQUEST_BYTES)
    return value


def _keys(keys: tuple[DoctorProofCacheKey, ...], binding: FormalStatusCensusBinding | None):
    if type(keys) is not tuple or len(keys) > MAX_KEYS:
        raise ValueError("bounded immutable native key set required")
    reconstructed, byte_count = [], 0
    for key in keys:
        if type(key) is not DoctorProofCacheKey:
            raise ValueError("exact native key required")
        if (type(key.premises) is not tuple or len(key.premises) > 256
                or type(key.budget) is not ResourceBudget
                or type(key.required_assurance) is not AssuranceLevel):
            raise ValueError("bounded native key components required")
        retained_bytes = 0
        for identity in key.identities:
            if (type(identity) is not DoctorIdentityBinding or type(identity.canonical_bytes) is not bytes
                    or len(identity.canonical_bytes) > MAX_KEY_BYTES // 2):
                raise ValueError("bounded native identity required")
            retained_bytes += len(identity.canonical_bytes)
            if retained_bytes > MAX_KEY_BYTES // 2:
                raise ValueError("aggregate native identity bytes exceed bound")
            for name in ("logical_id", "profile", "cid", "digest", "multibase", "multicodec",
                         "multihash", "domain", "artifact_schema"):
                value = getattr(identity, name)
                if type(value) is not str or len(value.encode("utf-8")) > 4096:
                    raise ValueError("bounded identity fields required")
        body = _bounded_json(key.to_dict(), MAX_KEY_BYTES)
        byte_count += len(body.encode("utf-8"))
        if byte_count > MAX_REQUEST_BYTES:
            raise ValueError("aggregate key bytes exceed bound")
        restored = DoctorProofCacheKey.from_dict(json.loads(body))
        if (type(restored) is not DoctorProofCacheKey or restored.contains_private_material
                or (binding is not None and (
                    restored.candidate_tree.logical_id != binding.candidate_tree_id
                    or restored.tree.logical_id != binding.candidate_tree_id))
                or restored.required_assurance.rank < AssuranceLevel.KERNEL_VERIFIED.rank
                or canonical_json(restored.to_dict()) != body):
            raise ValueError("current complete public native key required")
        reconstructed.append(restored)
    identities = tuple(item.key_id for item in reconstructed)
    if (len(set(identities)) != len(identities) or (binding is not None
            and binding.expected_key_ids is not None
            and set(identities) != set(binding.expected_key_ids))):
        raise ValueError("key population differs from independently expected context")
    return tuple(reconstructed)


def _header_admissions(admissions, keys, binding, gate):
    from .doctor_header_cache_bridge import HeaderProofCacheAdmission

    if (type(admissions) is not tuple or len(admissions) > MAX_KEYS
            or binding is None or binding.expected_receipt_ids is None
            or binding.owner_path_sha256 is None or keys is None or gate is None
            or len(admissions) != len(keys)):
        raise ValueError("complete typed header admission population required")
    expected = dict(binding.expected_receipt_ids)
    by_key = {}
    for admission in admissions:
        if type(admission) is not HeaderProofCacheAdmission:
            raise ValueError("exact native header proof admission required")
        key = admission.key
        if key.key_id not in expected or key.key_id in by_key:
            raise ValueError("header proof admission population differs")
        admission.assert_cache_binding(gate=gate, expected_key_id=key.key_id,
            expected_receipt_id=expected[key.key_id],
            expected_owner_path_sha256=binding.owner_path_sha256)
        by_key[key.key_id] = admission
    if set(by_key) != {key.key_id for key in keys}:
        raise ValueError("header proof admission population differs")
    for key in keys:
        if canonical_json(by_key[key.key_id].key.to_dict()) != canonical_json(key.to_dict()):
            raise ValueError("header proof admission full key differs")
    return by_key


def _result(result, key: DoctorProofCacheKey, stage: DoctorCacheStage):
    if (type(result) is not DoctorCacheLookupResult or type(result.key) is not DoctorProofCacheKey
            or type(result.disposition) is not DoctorCacheDisposition
            or type(result.diagnostic) is not bool
            or canonical_json(result.key.to_dict()) != canonical_json(key.to_dict())
            or result.key.key_id != key.key_id):
        raise ValueError("native exact lookup result required")
    audit = result.audit
    if (type(audit) is not DoctorCacheAuditReceipt or audit.key_id != key.key_id
            or audit.disposition is not result.disposition
            or type(audit.authoritative) is not bool or type(audit.reconstructed) is not bool
            or audit.semantic_root_ids != key.semantic_root_ids):
        raise ValueError("native exact audit required")
    receipt = result.receipt
    if receipt is not None:
        if type(receipt) not in (ProofReceipt, ScopedProofReceipt):
            raise ValueError("native receipt required")
        body = _bounded_json(receipt.to_dict(), MAX_RECEIPT_BYTES)
        receipt = ProofReceipt.from_dict(json.loads(body))
        if receipt.receipt_id != result.receipt.receipt_id:
            raise ValueError("receipt reconstruction changed identity")
    if result.disposition is DoctorCacheDisposition.HIT:
        if (receipt is None or result.diagnostic or not audit.authoritative or not audit.reconstructed
                or audit.stage is not stage or audit.receipt_id != receipt.receipt_id
                or audit.assurance != receipt.authoritative_assurance.value
                or receipt.verdict is not ProofVerdict.PROVED
                or receipt.authoritative_verdict is not ProofVerdict.PROVED
                or receipt.freshness is not EvidenceFreshness.CURRENT
                or not receipt.authoritative_assurance.satisfies(key.required_assurance)
                or DoctorProofCacheGate._receipt_reasons(key, receipt)):
            raise ValueError("native positive receipt binding failed")
    elif audit.authoritative:
        raise ValueError("nonpositive lookup claims authority")
    return receipt


def census_formal_status(*, binding: FormalStatusCensusBinding | None = None,
                         gate: DoctorProofCacheGate | None = None,
                         keys: tuple[DoctorProofCacheKey, ...] | None = None,
                         current_binding_probe: Callable[[], FormalStatusCensusBinding] | None = None,
                         header_proof_admissions: tuple | None = None,
                         require_bound_scoped_receipts: bool = False) -> dict:
    """Observe only the explicit admitted key set; never enumerate/open a store.

Malformed requests are refused before probes or filesystem observation. Missing
owner/key/source bindings remain unmeasured. The host probe brackets native
consultation and render revalidation; path/device/inode checks detect owner
replacement. A successful observation is point-in-time and confers no authority.
    """
    report = {"schema": SCHEMA, "status": "unmeasured",
        "reason_code": "native_owner_binding_unavailable", "observation_current": False,
        "binding_sha256": None, "requested_key_set_sha256": None, "requested_key_count": None,
        "accepted_receipts": None, "lookup_dispositions": None, "render_dispositions": None,
        "recorded_verdict_population": None, "derived_verdict_population": None,
        "verdict_population_status": "not-measured-by-positive-gate-census@1",
        "repeated_status_population": None, "current_binding_probe_calls": 0,
        "gate_lookup_calls": 0, "gate_render_revalidation_calls": 0, "owner_identity_checks": 0,
        "gate_denial_state_checks": 0,
        "native_gate_bookkeeping_may_write_existing_owner": True,
        "storage_read_only": False, "new_storage_owners_created": 0,
        "atomic_owner_snapshot": False, "eligibility_at_return_verified": False,
        "accepted_receipt_count_scope": "per-key-native-lookup-and-render-observations@1",
        "binding_admission": "caller_admitted_host_context@1",
        "receipt_hash_authenticates_source_or_owner": False,
        "scope": "explicit-requested-native-doctor-key-set@1",
        **{name: False for name in _AUTHORITIES}}
    bound_lane = (header_proof_admissions is not None
                  or type(binding) is FormalStatusCensusBinding
                  and binding.expected_receipt_ids is not None)
    if bound_lane:
        report.update(schema=BOUND_SCHEMA, expected_receipt_set_sha256=None,
            header_admission_current_checks=0,
            binding_admission="caller_admitted_header_bridge@1",
            accepted_receipt_count_scope="expected-header-receipts-with-current-bound-admissions@1")

    def finish(status, reason):
        report.update(status=status, reason_code=reason)
        _bounded_json(report, MAX_OUTPUT_BYTES)
        return report

    try:
        if type(require_bound_scoped_receipts) is not bool:
            raise ValueError("explicit scoped receipt binding selection required")
        payload = None if binding is None else _binding_payload(binding)
        if gate is not None and (type(gate) is not DoctorProofCacheGate
                or type(gate.authoritative_cache) is not FormalVerificationCache):
            raise ValueError("exact instantiated native owner required")
        admitted_keys = None if keys is None else _keys(keys, binding)
        if current_binding_probe is not None and not callable(current_binding_probe):
            raise ValueError("host current binding probe required")
        admissions = (_header_admissions(header_proof_admissions, admitted_keys, binding, gate)
                      if bound_lane else None)
        if (binding is None or gate is None or keys is None or current_binding_probe is None
                or binding.owner_path_sha256 is None or binding.expected_key_ids is None):
            return finish("unmeasured", "native_owner_binding_unavailable")
        report["binding_sha256"] = _sha(canonical_json(payload))
        report["requested_key_set_sha256"] = _sha(canonical_json(sorted(binding.expected_key_ids)))
        report["requested_key_count"] = len(admitted_keys)
        if bound_lane:
            report["expected_receipt_set_sha256"] = _sha(canonical_json(
                sorted(binding.expected_receipt_ids)))
    except Exception:
        return finish("refused", "invalid_native_request")

    try:
        native_cache = gate.authoritative_cache
        path = gate.db_path
        if (type(path) is not type(Path()) or not path.is_absolute() or path.suffix != ".duckdb"
                or _sha(str(path)) != binding.owner_path_sha256):
            raise ValueError("admitted native owner path differs")
    except Exception:
        return finish("refused", "native_owner_identity_mismatch")

    def owner_identity():
        report["owner_identity_checks"] += 1
        if (gate.authoritative_cache is not native_cache or gate.db_path != path
                or gate.authoritative_cache.db_path != path
                or path.resolve(strict=True) != path
                or any(part.is_symlink() for part in (path, *path.parents))):
            raise ValueError("native owner path changed")
        stamp = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(stamp.st_mode):
            raise ValueError("native owner is not a file")
        return stamp.st_dev, stamp.st_ino

    try:
        original_owner = owner_identity()
    except Exception:
        return finish("refused", "native_owner_identity_mismatch")

    def current():
        report["current_binding_probe_calls"] += 1
        observed = current_binding_probe()
        if (type(observed) is not FormalStatusCensusBinding
                or canonical_json(_binding_payload(observed)) != canonical_json(payload)):
            raise ValueError("host current context changed")
        if bound_lane:
            rebound = _header_admissions(header_proof_admissions, admitted_keys, binding, gate)
            if any(rebound[key_id] is not admission for key_id, admission in admissions.items()):
                raise ValueError("bound header admission changed")
            for admission in admissions.values():
                report["header_admission_current_checks"] += 1
                admission.assert_current()
        if owner_identity() != original_owner:
            raise ValueError("native owner replaced")

    lookup_counts = {item.value: 0 for item in DoctorCacheDisposition}
    render_counts = {item.value: 0 for item in DoctorCacheDisposition}
    accepted = 0
    try:
        with public_replay_without_metadata():
            current()
            for key in admitted_keys:
                report["gate_lookup_calls"] += 1
                lookup = gate.lookup(key)
                if owner_identity() != original_owner:
                    raise ValueError("native owner replaced")
                first = _result(lookup, key, DoctorCacheStage.LOOKUP)
                if (require_bound_scoped_receipts and not bound_lane
                        and lookup.hit and type(first) is ScopedProofReceipt):
                    raise ValueError("scoped receipt requires its current bound header admission")
                if bound_lane and (not lookup.hit or type(first) is not ScopedProofReceipt
                        or first.receipt_id != dict(binding.expected_receipt_ids)[key.key_id]):
                    raise ValueError("lookup differs from independently expected header receipt")
                current()
                report["gate_render_revalidation_calls"] += 1
                rendered = (admissions[key.key_id].revalidate_for_render() if bound_lane
                            else gate.revalidate_for_render(key))
                if owner_identity() != original_owner:
                    raise ValueError("native owner replaced")
                final = _result(rendered, key, DoctorCacheStage.RENDER)
                if (require_bound_scoped_receipts and not bound_lane
                        and rendered.hit and type(final) is ScopedProofReceipt):
                    raise ValueError("scoped receipt requires its current bound header admission")
                if bound_lane and (not rendered.hit or type(final) is not ScopedProofReceipt
                        or final.receipt_id != dict(binding.expected_receipt_ids)[key.key_id]):
                    raise ValueError("render differs from independently expected header receipt")
                current()
                lookup_counts[lookup.disposition.value] += 1
                render_counts[rendered.disposition.value] += 1
                if lookup.hit and rendered.hit:
                    if first.receipt_id != final.receipt_id:
                        raise ValueError("native receipt changed during census")
                    accepted += 1
            report.update(lookup_dispositions=lookup_counts, render_dispositions=render_counts)
            # A host probe or a later key may have revoked an earlier key. Check
            # visible native denial state for the complete requested set after
            # all callbacks; these existing owner reads may register roots.
            for key in admitted_keys:
                for check in (gate.is_quarantined, gate.is_tombstoned):
                    report["gate_denial_state_checks"] += 1
                    denied = check(key)
                    if type(denied) is not bool or denied:
                        raise ValueError("native key revoked during census")
                    if owner_identity() != original_owner:
                        raise ValueError("native owner replaced")
            if bound_lane:
                # The bound guard reads combined native denial state and checks
                # current source/custody after the final generic consultation.
                current()
        report.update(observation_current=True, accepted_receipts=accepted)
        return finish("measured", "explicit_native_key_set_observed")
    except Exception:
        return finish("refused", "native_observation_refused")


__all__ = ["BOUND_SCHEMA", "FormalStatusCensusBinding", "census_formal_status"]
