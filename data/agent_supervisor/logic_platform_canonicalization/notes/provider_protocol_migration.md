# LPC-050 LogicProviderProtocol@2 Operation-Specific Requests

**Task:** LPC-050 — Add LogicProviderProtocol@2 operation-specific requests  
**Goal:** LPC-G050  
**Depends on:** LPC-021 (catalog drift tests), LPC-032 (success is not proof)  
**Interface:** `LogicProviderProtocol@2`  
**Module:** `ipfs_datasets_py.logic.backends.protocol_v2`  
**Path:** `ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol_v2.py`  
**Protocol version:** `2`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_provider_protocol_v2.py -q`

## Purpose

`LogicProvider@1` (`backends/provider.py`) uses a single envelope with an
unrestricted JSON `payload`. That is intentional for the portable wire leaf,
but it cannot be the new-write path for typed provider work.

LPC-050 adds **LogicProviderProtocol@2**: operation-specific typed request
records that replace free-form payload routing for capability, translation,
prove/check, reconstruct, verify, and attest. Executable operations require
**positive finite bounds** before any provider dispatch.

Provider output remains untrusted until validation or reconstruction
(LPC-032). v1 generic envelopes are **not** silently promoted; LPC-051 owns
the explicit adapter that parses, rejects, or retains them as advisory.

## Generation map

| Generation | Module | Role |
| --- | --- | --- |
| `LogicProvider@1` | `logic/backends/provider.py` | Live portable wire envelope; generic `payload` |
| `BackendRequest@2` | `logic/backends/requests_v2.py` | Provider-selection input; typed family/bounds |
| **`LogicProviderProtocol@2`** | **`logic/backends/protocol_v2.py`** | **Operation-specific typed requests (this task)** |
| v1 adapter (LPC-051) | `logic/backends/protocol_v1_adapter.py` (planned) | Admit/reject/retain v1 generics without bypass |

## Typed request inventory

| Request type | Operation(s) | Executable? | Requires `RequestBounds` | Requires `BackendRequest@2` |
| --- | --- | --- | --- | --- |
| `CapabilityRequestV2` | `capability` | no | no | no |
| `TranslationRequestV2` | `translate` | **yes** | **yes** | **yes** |
| `ProveCheckRequestV2` | `prove`, `check` | **yes** | **yes** | **yes** |
| `ReconstructRequestV2` | `reconstruct` | **yes** | **yes** | **yes** |
| `VerifyRequestV2` | `verify` | **yes** | **yes** | **yes** |
| `AttestRequestV2` | `attest` | **yes** | **yes** | **yes** |

`ProveCheckRequestV2` is one request family with `mode ∈ {prove, check}`.
Both modes are executable.

Closed operation vocabulary (`ProtocolOperationV2` / `PROTOCOL_V2_OPERATIONS`):

```
capability, translate, prove, check, reconstruct, verify, attest
```

Executable set (`EXECUTABLE_OPERATIONS`):

```
translate, prove, check, reconstruct, verify, attest
```

## Bounds contract

Executable requests use `RequestBounds` from `requests_v2`:

| Field | Rule |
| --- | --- |
| `timeout_ms` | positive integer, hard ceiling enforced |
| `max_steps` | positive integer, hard ceiling enforced |
| `max_memory_bytes` | positive integer, hard ceiling enforced |
| `max_output_bytes` | positive integer, hard ceiling enforced |

Missing, zero, negative, incomplete, or non-integer bounds raise
`MissingExecutableBoundsError` (fail closed). Operation bounds may only
**tighten** relative to the admitted `BackendRequest@2` bounds; loosening is
rejected.

Capability is non-executable: discovery/health only. It does not mint proof
authority and does not require execution bounds.

## Admission helpers

| Symbol | Role |
| --- | --- |
| `admit_provider_request_v2` | Discriminated admit of one typed body; rejects free-form `payload` and v1 generic envelopes |
| `ProviderProtocolEnvelopeV2` | Envelope carrying exactly one typed request |
| `require_executable_bounds` | Returns positive finite bounds for executable ops only |
| `is_executable_operation` | Closed executable-set membership |
| `LogicProviderProtocolV2` | Runtime-checkable structural provider surface |
| `v1_operation_for` | Maps @2 ops onto nearest `LogicProviderOperation` (check → prove wire name) |

## What this does **not** do

1. **Does not** replace the @1 wire leaf for existing supervisor facades.
2. **Does not** implement the LPC-051 v1 adapter (parse/reject/retain path).
3. **Does not** promote provider success into proof authority (LPC-032).
4. **Does not** invent a second `BackendRequest` generation; executable ops bind
   the existing `BackendRequest@2`.

## Migration posture

| Write path | Required generation |
| --- | --- |
| New provider operation requests | `LogicProviderProtocol@2` typed request |
| New provider selection inputs | `BackendRequest@2` |
| Legacy @1 envelopes | Explicit adapter only (LPC-051); never silent bypass |
| Capability probes | `CapabilityRequestV2` (non-executable) |

## Validation coverage

`tests/unit/logic/backends/test_provider_protocol_v2.py` asserts:

* interface identity `LogicProviderProtocol@2` and protocol version `2`;
* typed classes for all six operation families;
* executable ops require positive finite bounds;
* missing / non-positive / incomplete bounds fail closed;
* prove and check share `ProveCheckRequestV2` with distinct modes;
* operation bounds cannot exceed admitted `BackendRequest@2` bounds;
* free-form and v1 generic payloads are rejected at admission;
* envelope and per-request dict round-trips preserve operation identity.
