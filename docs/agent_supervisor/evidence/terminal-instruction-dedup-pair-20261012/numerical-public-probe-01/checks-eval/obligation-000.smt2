; software-verification semantic SMT compiler 1.0.0
; obligation_id: "numerical:740c893bb353d639c34b2a273a9f4848"
; query_mode: theorem_by_negation
; features: arithmetic,equality
(set-info :smt-lib-version 2.6)
(set-info :source "SoftwareVerificationSMTCompiler@1")
(set-info :obligation "numerical:740c893bb353d639c34b2a273a9f4848")
(set-logic QF_UFLIA)
(declare-const columns Int)
(declare-const rows Int)
(assert (not (and (>= rows 0) (and (>= columns 0) (= rows columns)))))
(check-sat)
