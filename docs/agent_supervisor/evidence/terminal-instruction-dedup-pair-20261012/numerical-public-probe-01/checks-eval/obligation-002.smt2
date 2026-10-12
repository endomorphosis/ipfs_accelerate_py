; software-verification semantic SMT compiler 1.0.0
; obligation_id: "numerical:e423320b51a0dfef884af0667ecdac89"
; query_mode: theorem_by_negation
; features: arithmetic,equality
(set-info :smt-lib-version 2.6)
(set-info :source "SoftwareVerificationSMTCompiler@1")
(set-info :obligation "numerical:e423320b51a0dfef884af0667ecdac89")
(set-logic QF_UFLIA)
(declare-const extent Int)
(declare-const index Int)
(assert (not (and (>= extent 0) (and (<= (- extent) index) (< index extent)))))
(check-sat)
