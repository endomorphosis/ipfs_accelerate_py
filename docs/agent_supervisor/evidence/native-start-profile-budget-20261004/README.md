# Native START profile and driver controls

The final scoped run passed **80/80** tests with no skips in **13.58 seconds**. Its pytest XML and controller exit record are included byte-for-byte; command/environment paths are normalized placeholders and retained original digests identify the exact local records. Four current source/test files match the tested SHA-256 pins.

The extended benchmark profile supplies an explicit START allowance of at most 120 seconds, clamped to remaining work. It refuses when fewer than two seconds remain. Legacy profiles omit the override, STOP remains 20 seconds, and the extended cleanup reserve remains 60 seconds. Startup diagnostics have a closed bounded schema; a mocked driver test confirms diagnostic failure preserves the original START error and still calls STOP and close.

These are configuration and mocked driver-boundary controls, with simulated work clocks. They do not measure an actual native START, sustained health, checkpoint inference, or benchmark reward. Runtime/bootstrap native tests are a separate component.

The first invocation failed during collection because the existing Harbor site-packages path was omitted; its metadata and original XML digest are retained without exporting the private traceback. The corrected 70-case intermediate run passed before ten further diagnostic controls were added. Those 70 cases overlap the final 80 and must not be added together. Raw stdout is excluded throughout.
