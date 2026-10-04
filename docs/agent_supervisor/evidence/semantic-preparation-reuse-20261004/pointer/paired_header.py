"""Alternate pure-parser paths, retaining both CPU and host wall durations."""
import hashlib
import json
from pathlib import Path
import statistics
import time

from ipfs_datasets_py.logic.ir_core import canonical
from ipfs_datasets_py.logic.security_ir import code_header_derivation as derivation
from ipfs_datasets_py.logic.security_ir import doctor_header_contracts as contracts

output = Path(__file__).resolve().parent
source = output.parent / "source384-construction-lifetime-20261004/trial-01/jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__WCih55g/agent/public-output-evidence/before/bottle.py"
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba"
protocol = contracts.WsgiHeaderProtocolContract("public-task-wsgi-header-profile@1")
original = canonical._parse_pointer
parsers = {"uncached": canonical._parse_pointer_uncached, "cached": original}
pins = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (canonical, derivation, contracts)}
expected = derivation.derive_header_semantics(source_bytes=raw, source_path="bottle.py", protocol=protocol)
rows = []
try:
    for iteration in range(10):
        for mode in (("uncached", "cached") if iteration % 2 == 0 else ("cached", "uncached")):
            canonical._parse_pointer = parsers[mode]
            wall, cpu = time.perf_counter(), time.process_time()
            actual = derivation.derive_header_semantics(source_bytes=raw, source_path="bottle.py", protocol=protocol)
            row = dict(iteration=iteration, mode=mode, cpu_seconds=time.process_time() - cpu,
                       wall_seconds=time.perf_counter() - wall)
            assert actual == expected
            rows.append(row)
finally:
    canonical._parse_pointer = original
assert source.read_bytes() == raw
assert pins == {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                for m in (canonical, derivation, contracts)}
medians = {mode: {clock: statistics.median(row[clock] for row in rows if row["mode"] == mode)
                 for clock in ("cpu_seconds", "wall_seconds")} for mode in parsers}
record = dict(schema="paired-pure-header-lowering-profile@1", samples=rows, medians=medians,
    alternating_order=True, output_equality_all_samples=True, source_and_producer_pins_unchanged=True,
    source_sha256=hashlib.sha256(raw).hexdigest(), producer_pins=pins,
    derivation_json_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
    source_executed=False, solver_calls=0, provider_calls=0, shared_scheduler_access=False,
    production_qualification=False, scope="one pinned CPU; same-process alternate pure parser only; concurrent host activity uncontrolled")
with (output / "paired-result.json").open("x") as stream:
    stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(medians))
