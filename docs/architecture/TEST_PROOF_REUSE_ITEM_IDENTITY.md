# Automatic proof-reuse item identity

`AutomaticItemIdentityAssembler@1` is the fail-closed boundary between a
collected pytest item and `TestExecutionKey@1`.

The assembler needs no test-file registry. It obtains the item path, direct
node selection, parameter values, fixture names, markers, and effect adapters
from pytest; resolves the item against a freshly supplied repository forest;
verifies the current source against the supplied `AnalysisASTIndex`; and mints
path-bound module, class, function, decorator, and aggregate AST CIDs. It then
uses the existing static tracer, identity-component compiler, eligibility
evaluator, and execution-identity compiler.

Session-level dependency injection supplies five capabilities:

1. an exact current `RepositoryForest`;
2. a current complete `AnalysisASTIndex`;
3. exact fixture, conftest, hook, plugin, dependency, environment, interpreter,
   platform, hardware, and capability inputs;
4. current verification, collection, trace, and certificate policies; and
5. current runtime evidence.

No provider is imported or called by importing the module. A missing provider,
optional dependency, malformed inventory, stale AST index, uncontrolled
fixture, unsupported parameter, incomplete trace, or ordinary provider failure
returns a typed `RUN` result and attaches no cache lookup request.

## Collection-time admission limit

Normal pytest fixture values do not exist until setup. Likewise,
`RuntimeTestDependencyTrace@1` is observed while a test executes. Collection
therefore cannot reconstruct a fully authoritative current execution key from
ordinary pytest state.

A cached runtime trace is historical evidence and is not accepted as current.
Before setup, the assembler admits runtime input only as
`CurrentRuntimeTraceEvidence@1`, produced by an injected fresh controlled
preflight and bound to the exact node, repository-forest CID, static-trace CID,
identity-component root, and runtime-completeness-policy CID. This boundary
does not infer that the provider really performed a fresh preflight; deployment
policy must trust and qualify that provider. Without such a provider the test
runs normally.

This means the safe initial integration can automatically assemble and report
collection identity for every test, but most ordinary tests remain
non-reusable until there is a reviewed way to obtain current fixture/runtime
evidence without circularly trusting the prior run. A future design may use
declarative fixture adapters or a qualified isolated preflight. It must not
relabel a prior trace as current or weaken the runtime requirement.

Even a successful assembly only attaches `ProofReuseLookupRequest`. Its action
remains `RUN`; it cannot add a pytest skip marker. The existing local cache
admission and proof verifier remain the only path to an authoritative `SKIP`.
