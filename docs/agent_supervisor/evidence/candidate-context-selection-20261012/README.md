# Selective working-candidate retrieval qualification

The coding worker can discover exact identifiers with a catalog and retrieve
individual native capsules or pending numerical obligations through the existing
owner-controlled source socket. Queries bind the captured working generation,
source scope, producer digest, parent artifacts and exact filters. Source drift
refuses delivery. Excluded rows and unsupported interpretation remain explicit.
The datasets library continues to own the capsule and numerical semantics.

## Validation

The [main regression](final-01/qualification.json) passed 984 cases. Two selector
tests were added during that run; its `source_unchanged` flag therefore correctly
reads false. The [frozen selector rerun](selector-final-02/qualification.json)
passed all 53 selector cases. Production files did not change. There are 986
distinct repository test cases across these runs after normalizing pytest's
different package prefixes; see the [test audit](final-test-review.json).

Both network-disabled Docker qualifications passed with the generated owner and
worker launcher, distinct owner/worker UIDs, real source broker/client, native
public-input indexing and an authored router hook. The owner independently
replayed every complete selected payload. Source edits invalidated the prior
generation; restoration plus a fresh capture rebuilt the catalog. Container,
socket and descriptor cleanup completed. This checks retrieval integration with
an authored worker, not autonomous task performance.

| Complete selected result | Artifact bytes | Response JSON bytes | Pages |
| --- | ---: | ---: | ---: |
| `eigen.py` catalog | 4,595 | 6,331 | 1 |
| One capsule | 5,336 | 7,208 | 1 |
| One obligation | 4,545 | 6,319 | 1 |

Initial discovery plus those two subsets cost **19,858 response JSON bytes** in
each Docker run. The capsule selection returned 1 of 2 capsules in the selected
path and disclosed 6 more in other paths. The obligation selection returned 1 of
16 in the selected path and disclosed 29 more in other paths. Full parent
artifacts were 48,309 capsule bytes and 476,030 formal bytes. Their information
coverage differs from the selections; these ratios are not lossless compression.
The full-artifact interface retrieved only prefixes of those two parent artifacts
in this probe, while each selected result was retrieved completely.

Docker02 also checked the default model-facing CLI projection: the same selected
obligation required 5,520 printed JSON bytes instead of a 6,319-byte raw envelope,
preserving the exact slice, selection, source/artifact pins and nonproof flags.
Even this selected response is larger than the 642-byte source file. No native
provider tokens or task-success advantage was measured.

## Evidence and reproducibility

- [Docker01 result](docker-01/qualification.json) and
  [selection costs](docker-01/results/selection-observation.json): 46.38 seconds,
  53 accepted requests, 136,932 total response JSON bytes.
- [Docker02 result](docker-02/qualification.json) and
  [selection costs](docker-02/results/selection-observation.json): 47.26 seconds,
  54 accepted requests, 143,251 total response JSON bytes, including the extra
  default-display query. The aggregate covers positive, negative, source,
  candidate and repeated reads, not just the three initial selected queries.
- Each run retains its command, source hashes, container controls and exact
  runner snapshots. All 101 source bindings and three runner snapshots were
  checked per run; see the [independent review](protocol-final-review.json).
- [Datasets producer comparison](datasets-producer-equivalence.json) records
  the 24 direct producer modules at the tested datasets checkout and the newer
  parent-repository gitlink. The direct producers match byte for byte; this does
  not qualify unrelated datasets changes or the complete dependency closure.
- [Manifest](manifest.json) binds the retained files. Private source/formula
  response bodies and database state remain outside this published evidence.

The runners record absolute paths for this machine. Reproduction requires the
recorded checkouts, pinned local image, public benchmark input files, fixture
profile and DuckDB extension files at corresponding mounted paths. The standalone
client remains dependency-free. Source and runner snapshots, rather than a
subsequent branch tip, identify what each Docker run used.

## Remaining measurement

These runs made zero provider/model/solver calls and discharged zero source
obligations. They did not execute the official Terminal-Bench verifier, full live
supervisor START, autoencoder inference or `lake build`. A matched native-model
benchmark must still compare source-only, selective-context and full-context
policies at equal task success, counting discovery, retries, initialization,
translation tables, refreshes, provider tokens and elapsed time. Selection is a
way to request less context; relevance and net token savings need that experiment.
