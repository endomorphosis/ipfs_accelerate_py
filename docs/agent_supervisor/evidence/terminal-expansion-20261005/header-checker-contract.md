# Bounded header-checker compatibility repair

The wider supervisor check `bootstrap-compatibility-final-01` recorded 65 passes and two failures against datasets revision `987cf856b2b902aa68c4587bb492b19b932b5d30`. Both failures were `TypeError`: the public `check_header_semantics` API no longer accepted the deadline, cancellation signal and parent resource lease supplied by supervisor applicability recovery.

The repair is datasets commit `5171a632c6b9f0ecb2939d29d2ad74992cbfeb11`. Its [three focused test modules](datasets-header-qualification.json) passed 86 tests; one optional exact public Bottle fixture was not configured and skipped. All native bounded-Z3 tests ran.

The datasets checkout retained its bounded native runner, but its public checker had lost the integration previously present in commits `ba95d74d2` and `d1c6e830a`. The repair restores that shared-module integration in a separate datasets worktree. It does not remove the supervisor's required execution-profile check or relax source/proof authority.

A leased check uses one decreasing deadline across derivation, admission, version probing, solver queries and cleanup. Native resource failures preserve their original cause and bounded diagnostic; the generic SMT backend cannot turn them into unowned version probes. Solver path and executable bytes are checked before and after each query. Existing optional unleased callers retain their route. These checks concern the finite source-bound header model and do not prove whole-program correctness.

Historical tests and live archives retain their original datasets revision. Fresh compatibility evidence binds the repaired datasets revision separately. The failed compatibility run remains retained and excluded from the green aggregate.

The next compatibility attempt (`bootstrap-compatibility-final-02`) passed 66 cases, including real captured recovery after 45 seconds. Its remaining failure was a stale test expectation: the shared native fixture selects a 30-second STOP limit, while this test expected and recorded 20 seconds. The test now independently asserts the configured 30-second limit; production STOP behavior and separate 20-second unit controls are unchanged. This failed rerun is also retained outside the green aggregate.

The corrected `bootstrap-compatibility-final-03` passed all **67 tests** with unchanged recorded source hashes against datasets `5171a632c6b9f0ecb2939d29d2ad74992cbfeb11`. Its supervisor/test bytes match commit `c92f44d7245e7fffa666e3001a91fb13abfd39c1`. This includes actual native START/STOP with an expired caller replay scope and real captured replay recovery after 45 seconds.
