# V04 authored container metadata contract cases

This separate source-only suite calls the backend's actual pure
`verify_container` helper against an independently authored literal engineering
contract. This new generation preserves the frozen original and adds an
explicit private cgroup namespace and empty supplementary-group/device-rule
allowances. It contains one positive case and sixteen identity/confinement
cases covering twenty-three independent unsafe mutations. It neither creates
a Docker container nor probes a daemon, opens a
native SDK grant, reads real ledgers or credentials, or manufactures native
acceptance evidence.

The tests import the frozen V04 backend and its ordinary adapter/common source
dependencies through explicit sibling source paths. A genuine finite validation
or engineering owner must pin those sources and run the suite. Actual daemon,
CID, process-birth, cleanup, SDK finalization and one-use-handle qualification
remain the separate live native fixture's responsibility.
