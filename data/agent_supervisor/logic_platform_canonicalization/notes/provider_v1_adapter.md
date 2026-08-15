# LPC-051 Keep v1 Generic Payloads from Bypassing BackendRequest@2

**Task:** LPC-051 — Keep v1 generic payloads from bypassing BackendRequest@2  
**Goal:** LPC-G050  
**Depends on:** LPC-050 (LogicProviderProtocol@2 typed requests)  
**Interface:** `LogicProviderProtocolV1Adapter@1`  
**Module:** `ipfs_datasets_py.logic.backends.protocol_v1_adapter`  
**Path:** `ipfs_datasets_py/ipfs_datasets_py/logic/backends/protocol_v1_adapter.py`  
**Adapter version:** `1.0.0`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_protocol_v1_adapter.py -q`

## Purpose

`LogicProvider@1` (`backends/provider.py`) is the live portable wire leaf. Its
request envelope carries an unrestricted JSON `payload`. That is intentional
for supervisor facades and dual-read hosts, but it must never become a silent
executable path that mints or bypasses `BackendRequest@2`.

LPC-050 added `LogicProviderProtocol@2` with operation-specific typed requests
and rejected free-form / v1 bodies at `admit_provider_request_v2`. LPC-051
owns the **explicit adapter** that dual-reads v1 generics under a closed
three-way disposition:

| Disposition | Meaning | Executable? | Carries `BackendRequest@2`? |
| --- | --- | --- | --- |
| **parsed** | Closed operation identified and elevated to a typed @2 request | only when the elevated op is executable | yes for executable ops |
| **rejected** | Malformed, unknown operation, or free-form bypass attempt | no | no |
| **advisory** | Operation known but not elevatable; retained without authority | **no** | **no** |

New provider writes use v2 only (`admit_new_provider_write`). Provider output
remains untrusted until validation or reconstruction (LPC-032).

## Generation map

| Generation | Module | Role |
| --- | --- | --- |
| `LogicProvider@1` | `logic/backends/provider.py` | Live portable wire envelope; generic `payload` |
| `BackendRequest@2` | `logic/backends/requests_v2.py` | Provider-selection input; typed family/bounds |
| `LogicProviderProtocol@2` | `logic/backends/protocol_v2.py` | Operation-specific typed requests (LPC-050) |
| **`LogicProviderProtocolV1Adapter@1`** | **`logic/backends/protocol_v1_adapter.py`** | **Explicit v1 dual-read (this task)** |

## Closed operation mapping

| LogicProvider@1 operation | LogicProviderProtocol@2 operation | Notes |
| --- | --- | --- |
| `capability` | `capability` | Non-executable; elevates without `BackendRequest@2` |
| `translate` | `translate` | Executable; requires external `BackendRequest@2` + bounds |
| `prove` | `prove` (or `check` via `mode` / adapter `mode=`) | v1 has no distinct check wire name |
| `reconstruct` | `reconstruct` | Executable; requires external `BackendRequest@2` + bounds |
| `verify` | `verify` | Executable; requires external `BackendRequest@2` + bounds |
| `attest` | `attest` | Executable; requires external `BackendRequest@2` + bounds |

Unknown operation names are **rejected**.

## Non-bypass rules (fail closed)

1. **Free-form `payload` never mints `BackendRequest@2`.**  
   Keys such as `backend_request`, `backend_request_v2`, `obligation`,
   `domain_slice`, and `slice` inside the v1 payload raise
   `V1BypassBackendRequestError` → disposition **rejected**.

2. **Executable elevation requires an *external* admitted `BackendRequest@2`.**  
   Supplied as the `backend_request=` argument to
   `adapt_v1_provider_request` / `elevate_v1_to_v2`. Never taken from the
   free-form payload body.

3. **Bounds may only tighten.**  
   When both external bounds and `backend_request.bounds` are present,
   operation bounds cannot exceed the admitted request bounds.

4. **Advisory retention has no executable authority.**  
   `AdvisoryV1Retention.authority_ceiling` is always `"advisory"`,
   `executable` is always `False`, and `backend_request` is always `None`.

5. **New writes reject v1.**  
   `admit_new_provider_write` only admits typed LogicProviderProtocol@2
   bodies. v1 envelopes must go through the explicit adapter.

6. **@2 admission remains the elevation gate.**  
   Successful elevation re-admits through `admit_provider_request_v2` so the
   adapter cannot loosen LPC-050 contracts.

## API surface

| Symbol | Role |
| --- | --- |
| `parse_v1_provider_envelope` | Parse a LogicProvider@1 envelope (fail closed) |
| `classify_v1_operation` | Map envelope / name → closed `ProtocolOperationV2` |
| `adapt_v1_provider_request` | Three-way disposition adapter |
| `elevate_v1_to_v2` | Strict elevation; raises on advisory/rejected |
| `retain_v1_as_advisory` | Explicit non-authoritative retention |
| `reject_v1_backend_request_bypass` | Probe free-form bypass keys |
| `admit_new_provider_write` | New-write gate (v2 only) |
| `V1AdapterResult` | Disposition + optional `request_v2` / `advisory` |
| `AdvisoryV1Retention` | Advisory-only retention record |
| `V1AdapterDisposition` | `parsed` / `rejected` / `advisory` |

## Elevation requirements by operation

| Operation | Elevates when | Otherwise |
| --- | --- | --- |
| `capability` | Always (typed capability fields optional) | — |
| `translate` | External BR@2 + bounds + `source_encoding`/`target_encoding` in payload | **advisory** |
| `prove` / `check` | External BR@2 + bounds | **advisory** (statement optional) |
| `reconstruct` | External BR@2 + bounds + `candidate_digest` | **advisory** |
| `verify` | External BR@2 + bounds + `evidence_digest` | **advisory** |
| `attest` | External BR@2 + bounds + `statement_digest` | **advisory** |

Missing external `BackendRequest@2` for an executable op is **advisory**, not
silent success and not a free-form mint.

## Relationship to LPC-050

| Concern | Owner |
| --- | --- |
| Typed @2 request classes + executable bounds | LPC-050 (`protocol_v2.py`) |
| Reject free-form / v1 bodies at pure @2 admission | LPC-050 (`admit_provider_request_v2`) |
| Explicit parse / reject / advisory dual-read of v1 | **LPC-051 (this module)** |
| New-write path uses v2 only | **LPC-051 (`admit_new_provider_write`)** |
| Provider success ≠ proof authority | LPC-032 |

## What this does **not** do

1. **Does not** replace the @1 wire leaf for existing supervisor facades.
2. **Does not** invent a second `BackendRequest` generation.
3. **Does not** promote advisory retention into proof or satisfiability authority.
4. **Does not** allow free-form payload keys to seed provider selection.
5. **Does not** define typed provider *responses* (LPC-052).

## Migration posture

| Write path | Required generation |
| --- | --- |
| New provider operation requests | `LogicProviderProtocol@2` via `admit_new_provider_write` |
| New provider selection inputs | `BackendRequest@2` |
| Legacy @1 envelopes (dual-read hosts) | Explicit `adapt_v1_provider_request` only |
| Capability probes (new) | `CapabilityRequestV2` |
| Capability probes (legacy dual-read) | Adapter → **parsed** `CapabilityRequestV2` |
| Incomplete executable v1 | Adapter → **advisory** (no execution) |
| Free-form BR@2 mint attempt | Adapter → **rejected** |

## Validation coverage

`tests/unit/logic/backends/test_protocol_v1_adapter.py` asserts:

* interface identity `LogicProviderProtocolV1Adapter@1`;
* v1 envelopes classify into closed operation types;
* unknown / malformed envelopes are rejected;
* free-form payload keys cannot mint or bypass `BackendRequest@2`;
* executable ops without external `BackendRequest@2` are advisory only;
* executable ops with external `BackendRequest@2` + bounds elevate to typed @2;
* advisory retention never reports executable authority or a backend request;
* new writes accept @2 and reject v1 generics;
* pure @2 admission still rejects silent v1 bypass (LPC-050 invariant preserved).
