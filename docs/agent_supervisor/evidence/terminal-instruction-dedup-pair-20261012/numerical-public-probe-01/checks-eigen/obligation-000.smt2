; software-verification semantic SMT compiler 1.0.0
; obligation_id: "numerical:a0d0f4836a386f59f45de38ddd0ed3dd"
; query_mode: theorem_by_negation
; features: arithmetic,equality
(set-info :smt-lib-version 2.6)
(set-info :source "SoftwareVerificationSMTCompiler@1")
(set-info :obligation "numerical:a0d0f4836a386f59f45de38ddd0ed3dd")
(set-logic QF_UFLIA)
(declare-const columns Int)
(declare-const rows Int)
(assert (not (and (>= rows 0) (and (>= columns 0) (= rows columns)))))
(check-sat)
