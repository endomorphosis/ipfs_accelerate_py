; software-verification semantic SMT compiler 1.0.0
; obligation_id: "numerical:a9a3410c3dac8653402811385aac3c23"
; query_mode: theorem_by_negation
; features: arithmetic,equality
(set-info :smt-lib-version 2.6)
(set-info :source "SoftwareVerificationSMTCompiler@1")
(set-info :obligation "numerical:a9a3410c3dac8653402811385aac3c23")
(set-logic QF_UFLIA)
(declare-const extent Int)
(declare-const index Int)
(assert (not (and (>= extent 0) (and (<= (- extent) index) (< index extent)))))
(check-sat)
