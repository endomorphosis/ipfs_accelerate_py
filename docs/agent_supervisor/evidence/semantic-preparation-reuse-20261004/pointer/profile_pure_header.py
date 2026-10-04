"""Body-free profile of deterministic lowering on retained public task source."""
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import sys
import time

from ipfs_datasets_py.logic.security_ir import doctor_header_contracts as contracts
from ipfs_datasets_py.logic.security_ir import code_header_derivation as derivation
from ipfs_datasets_py.logic.ir_core import canonical

mode = sys.argv[1]
assert mode in {"uncached", "cached"}
if mode == "uncached":
    canonical._parse_pointer = canonical._parse_pointer_uncached

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "source384-construction-lifetime-20261004/trial-01/jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__WCih55g/agent/public-output-evidence/before/bottle.py"
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba"
text = raw.decode()
protocol = contracts.WsgiHeaderProtocolContract("public-task-wsgi-header-profile@1")
pins = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (contracts, derivation, canonical)}
analysis = contracts.analyze_http_header_contracts(text, protocol=protocol)
assert analysis.candidate is not None
expected = derivation.derive_header_semantics(source_bytes=raw, source_path="bottle.py", protocol=protocol)
operations = {
    "analyze": lambda: contracts.analyze_http_header_contracts(text, protocol=protocol),
    "verify_candidate": lambda: contracts.verify_header_candidate(text, analysis.candidate),
    "derive": lambda: derivation.derive_header_semantics(source_bytes=raw, source_path="bottle.py", protocol=protocol),
    "validate": lambda: derivation.validate_header_semantics(expected, source_bytes=raw, source_path="bottle.py", protocol=protocol),
}
results = {}
for name, operation in operations.items():
    times = []
    for _ in range(5):
        start = time.perf_counter()
        result = operation()
        times.append(time.perf_counter() - start)
        if name == "verify_candidate": assert result is True
        elif name in {"derive", "validate"}: assert result == expected
        else: assert result.to_dict() == analysis.to_dict()
    profiler = cProfile.Profile()
    profiler.runcall(operation)
    rows = []
    for (path, line, function), (primitive, total, own, cumulative, callers) in pstats.Stats(profiler).stats.items():
        rows.append(dict(path=path, line=line, function=function, primitive_calls=primitive,
                         total_calls=total, own_seconds=own, cumulative_seconds=cumulative))
    rows.sort(key=lambda r: r["cumulative_seconds"], reverse=True)
    results[name] = dict(seconds=times, median_seconds=statistics.median(times), profile=rows[:40])
assert SOURCE.read_bytes() == raw
assert pins == {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (contracts, derivation, canonical)}
record = dict(schema="doctor-pure-source-derivation-profile@1", source_path=str(SOURCE),
    source_sha256=hashlib.sha256(raw).hexdigest(), source_bytes=len(raw), source_and_producer_pins_unchanged=True,
    producer_pins=pins, cold_import_time_excluded=True, bounded_host_process_cpu_count=1,
    solver_calls=0, provider_calls=0, source_executed=False, shared_scheduler_access=False,
    production_qualification=False, results=results, pointer_mode=mode,
    analysis_json_sha256=hashlib.sha256(json.dumps(analysis.to_dict(), sort_keys=True).encode()).hexdigest(),
    derivation_json_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest())
with (OUT / f"result-{mode}.json").open("x") as stream:
    stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps({name: row["median_seconds"] for name, row in results.items()}))
