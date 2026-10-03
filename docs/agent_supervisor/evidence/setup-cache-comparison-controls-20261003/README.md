# Setup-cache comparison-control declaration

The first full-trial preparation stopped on the host because the closed
supervisor-kwargs allowlist did not yet recognize `setup_cache_selection`.
The exact traceback and prepare exit are retained. It failed before task/agent
execution; no provider or model call was launched by that preparation.

The additive correction accepts only the existing closed selection schema/policy
and a 64-character lowercase manifest digest, on the full supervisor arm with
the supported common resource profile. Existing resource-limit validation still
runs. Explicit `None`, malformed or extra selection fields, no-index and missing
or incompatible profiles refuse. This pure declaration code performs no archive
or credential reads; deployment owners retain their separate validation.

The declaration schema and serialized shape are unchanged. The existing complete
configuration digest binds every adapter kwarg, including the selected policy and
manifest. Changing or dropping a selection after preparation yields mismatch.
Absent-selection legacy declarations retain their exact digests. Comparison still
means equality of the previously declared common configuration fields only;
unequal adapter setup work and runtime enforcement are not established by it.

**59 actual host-side tests passed in 20.85 seconds**, with no skips. They include
existing comparison controls, actual Harbor config normalization, six fixed
pre-change digest cases and the new shape/profile/tamper controls. The first test
launcher found no pytest in the Harbor virtualenv and collected no tests. Its
failure is retained; the passing launcher supplied existing installed host pytest
paths without installing dependencies or changing source.

Only `benchmark_controls.py` and its new focused test changed. The controls owner
is imported by host benchmark preparation, baseline and comparison code. The host
Harbor adapter also imports their constants/helpers. The container driver,
qualification probe and preparation/deployment owners do not import those host
controllers; all 33 selected runtime source pins are unchanged. Reusing the
qualified container archive therefore does not silently replace runtime owners.

This package contains exact before/final source, raw controls, pre-change golden
declarations, the retained preparation error and source-free scope metadata. It
contains no auth contents, task/verifier source, model weights, database or
runtime archive. No new benchmark reward or token score is claimed here.
