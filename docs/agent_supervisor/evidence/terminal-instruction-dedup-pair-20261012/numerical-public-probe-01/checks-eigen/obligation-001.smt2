; software-verification semantic SMT compiler 1.0.0
; obligation_id: "numerical:6b7fa0f89f467d7bc0b767dc7d634d6b"
; query_mode: theorem_by_negation
; features: arithmetic,equality
(set-info :smt-lib-version 2.6)
(set-info :source "SoftwareVerificationSMTCompiler@1")
(set-info :obligation "numerical:6b7fa0f89f467d7bc0b767dc7d634d6b")
(set-logic QF_UFLIA)
(declare-const extent Int)
(declare-const index Int)
(assert (not (and (>= extent 0) (and (<= (- extent) index) (< index extent)))))
(check-sat)
