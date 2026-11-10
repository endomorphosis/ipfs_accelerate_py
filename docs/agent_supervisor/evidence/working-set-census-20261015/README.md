# Working-set census qualification, 2026-10-15

Runtime commit: `5c1ab44ede` on `supervisor-working-set-census-20261015`.
The [behavior and limitations](../../terminal_working_set_census.md) describe
the new router/worker observation mode and the remaining benchmark integration.

`manifest.json` hashes the retained closed metrics, successful JUnit reports,
qualification results and source pins. `qualification-summary.json` records
235 distinct final test cases, all passed. Each final source-pin set matched
the committed runtime/test files, and each run retained unchanged dependency
and installed-distribution pins.

| Final batch | Passed |
| --- | ---: |
| Native indexed public/Doctor flow | 1 |
| Census dispatch and exporter | 28 |
| Component census and translation | 68 |
| Worker routes and existing integration policies | 138 |

`native-measurement.json` joins the indexed fixture's XML/result hashes.
`dispatch-final01-metrics.json` joins its XML, result and source pins. Both
measure complete UTF-8 input bytes; neither measures provider tokens or an
official benchmark reward. Provider output and host UID admission are authored
in these tests. No model service or Docker trial was started.

The earlier `regression-dev01` batch passed 107 tests and is retained separately;
it predates final exporter hardening and worker wiring. Two initial development
failures were fixture mistakes: reusing an existing semantic output directory,
and passing a normalized receipt to an audit that requires the original
container-boundary fields. Corrected final runs pass. Original development
artifacts remain under
`/home/barberb/lift_coding/artifacts/working-set-census-20261015`.

The pinned dependencies are `ipfs_datasets_py` at
`28b9ef1f0b9da51e87b1e68ee565b43681397109` and `ipfs_kit_py` at
`8ac00bb1974b1378e12091d546d8425af34049e0`; imports resolve to the paths recorded
in each result. The virtual environment uses `multiformats==0.3.1.post4` and
records its associated distribution hashes. Tests run with plugin autoload
disabled and isolated seal stores through the retained local `run-tests.py`.
