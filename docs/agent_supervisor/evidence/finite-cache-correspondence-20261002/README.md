# Finite source-derived cache correspondence qualification

The selected `python-integer-offset-finite@1` profile passed **104 tests** in
54.71 seconds: 46 new correspondence controls and 58 existing lossless bridge
controls, with no failures or skips. `tests.xml` records individual cases;
`tests.log` removes terminal color and trailing line whitespace. The raw log
digest and that normalization are recorded in `qualification.json`.

The new controls cover all 16 key dimensions, coherently rehashed forged
materials, explicit semantic/operational-bound separation, native source and
head changes, a source change during lowering, missing/corrupt CAS bytes,
executable and producer drift, foreign environments, incomplete domains,
cancellation, deadlines, false authority and positive cache-admission refusal.
A separate process reopens the file-backed DuckDB catalog and immutable CAS
and rederives the same native, execution and bridged key identities.

`reproduce.py` also ran a retained native example over an authored source file:

```python
def increment(n: int) -> int:
    return n + 1
```

The explicit instruction requests exact integer results and `n + 2` on
`[-2, -1, 0, 1, 2]`. The real matcher invokes Python and Lean, yielding five
counterexamples and one bounded type fact. Its head, source CID/hash, domain,
contract, compilation and tool-policy identities exactly join to the new key
materials. The key layer itself performs source observation and lowering only;
it neither executes those tools nor adopts the matcher's evidence authority.

The native observation's complete 11 artifacts plus `result.json` are retained
under `observation/`, including source, trace, process records, generated Lean,
compiled `.olean` and certificate. `matcher.json`, `correspondence.json` and
`request.json` retain the complete joined records; `restart.json` records the
separate replay process. Original absolute paths are historical identity
fields. This archive is not a live owner handle or positive proof-cache entry.

To produce a new qualification with installed native Python and Lean 4.34.1,
make the released accelerate and datasets packages importable and run:

```sh
python reproduce.py --output /absolute/new/qualification-directory
```

The script creates a new authored Git repository, file-backed DuckDB catalog,
CAS and resource scheduler. Its child replay reopens that new persistent state.
It does not reuse the archived process receipts as authority. The retained
example originally ran at
`/home/barberb/lift_coding/artifacts/repository-finite-cache-correspondence-20261002/run-01`.

The source inventory records 37 reviewed producer modules and two qualification
sources. It does not attest all transitive Python/Lean libraries. The native
runtime/caller assumptions and exact model-off declaration remain in the
reports. No model training, learned inference, supervisor daemon run, execution
permission, completion claim or positive proof admission is established here.

The result supplies the owner-derived selected-profile relationship needed by
RPI-004 and the cache-material join for RPI-028's bounded fixture. Generic proof
reuse, arbitrary-language dependency closure, and the rest of those backlog
items' dependency gates require their own qualification. The shared backlog
ledger was not modified by this evidence package.

See [the API and field correspondence](../../finite_cache_correspondence.md).
