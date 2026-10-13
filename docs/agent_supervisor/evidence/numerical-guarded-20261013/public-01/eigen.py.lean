import Std

namespace NumericalObligations

-- Specifications only: no declaration below asserts a property of Python.
def IndexSafety (extent : Nat) (index : Int) : Prop :=
  -(Int.ofNat extent) ≤ index ∧ index < Int.ofNat extent

def MatrixShape (rows columns : Nat) : Prop := rows = columns

def SelectionMax {α : Type} (le : α → α → Prop)
    (n : Nat) (score : Nat → α) (selected : Nat) : Prop :=
  selected < n ∧ ∀ j, j < n → le (score j) (score selected)

def SelectionMin {α : Type} (le : α → α → Prop)
    (n : Nat) (score : Nat → α) (selected : Nat) : Prop :=
  selected < n ∧ ∀ j, j < n → le (score selected) (score j)

def Ordering {α : Type} (le : α → α → Prop)
    (n : Nat) (score : Nat → α) (order : Nat → Nat) : Prop :=
  (∀ i, i < n → order i < n) ∧
  (∀ i, i < n → ∀ j, j < n → order i = order j → i = j) ∧
  (∀ i, i + 1 < n → le (score (order i)) (score (order (i + 1))))

def finiteSum {α : Type} (zero : α) (add : α → α → α)
    (term : Nat → α) : Nat → α
  | 0 => zero
  | n + 1 => add (finiteSum zero add term n) (term n)

def EigenpairResidual {α : Type} (zero : α) (add mul : α → α → α)
    (n : Nat) (matrix : Nat → Nat → α) (vector : Nat → α) (value : α) : Prop :=
  ∀ i, i < n → finiteSum zero add (fun j => mul (matrix i j) (vector j)) n =
    mul value (vector i)

def NormalizeIndex (extent : Nat) (index : Int) : Int :=
  if index < 0 then index + Int.ofNat extent else index

-- Conditional arithmetic lemmas; their premises are not established for source values.
theorem normalized_index_bounds (extent : Nat) (index : Int)
    (bounds : IndexSafety extent index) :
    0 ≤ NormalizeIndex extent index ∧ NormalizeIndex extent index < Int.ofNat extent := by
  unfold IndexSafety at bounds
  unfold NormalizeIndex
  split <;> omega

theorem last_index_bound (extent : Nat) (nonempty : 0 < extent) : extent - 1 < extent := by
  omega


-- A proposed mathematical contract, not the runtime interpretation of NumPy.
def EigensystemDimensions (rows columns valuesCount vectorRows vectorColumns : Nat) : Prop :=
  0 < rows ∧ columns = rows ∧ valuesCount = rows ∧ vectorRows = rows ∧ vectorColumns = rows

def SameSelectedColumn {α : Type} (values : Nat → α) (vectors : Nat → Nat → α)
    (selected : Nat) (value : α) (vector : Nat → α) : Prop :=
  value = values selected ∧ ∀ row, vector row = vectors row selected

def NonzeroVector {α : Type} (zero : α) (n : Nat) (vector : Nat → α) : Prop :=
  ∃ row, row < n ∧ vector row ≠ zero

def FiniteEigenpair {α : Type} (finite : α → Prop)
    (n : Nat) (value : α) (vector : Nat → α) : Prop :=
  finite value ∧ ∀ row, row < n → finite (vector row)

-- `close` has no chosen tolerance or floating-point semantics. Those remain
-- source interpretation obligations, including the direction of comparison.
def ApproximateEigenpairResidual {α : Type} (zero : α) (add mul : α → α → α)
    (close : α → α → Prop) (n : Nat) (matrix : Nat → Nat → α)
    (vector : Nat → α) (value : α) : Prop :=
  ∀ row, row < n → close
    (finiteSum zero add (fun column => mul (matrix row column) (vector column)) n)
    (mul value (vector row))

def SelectedEigenpair {α β : Type} (zero : α) (add mul : α → α → α)
    (finite : α → Prop) (close : α → α → Prop) (le : β → β → Prop)
    (n : Nat) (matrix : Nat → Nat → α) (score : Nat → β)
    (selected : Nat) (value : α) (vector : Nat → α) : Prop :=
  NonzeroVector zero n vector ∧ FiniteEigenpair finite n value vector ∧
    ApproximateEigenpairResidual zero add mul close n matrix vector value ∧
    SelectionMax le n score selected

-- Both the eigensystem properties and the selection property are premises.
-- This theorem does not prove either premise for any source library call.
theorem selected_column_contract {α β : Type} (zero : α) (add mul : α → α → α)
    (finite : α → Prop) (close : α → α → Prop) (le : β → β → Prop)
    (n : Nat) (matrix vectors : Nat → Nat → α) (values : Nat → α)
    (score : Nat → β) (selected : Nat)
    (choice : SelectionMax le n score selected)
    (columns : ∀ j, j < n → NonzeroVector zero n (fun row => vectors row j) ∧
      FiniteEigenpair finite n (values j) (fun row => vectors row j) ∧
      ApproximateEigenpairResidual zero add mul close n matrix (fun row => vectors row j) (values j)) :
    SelectedEigenpair zero add mul finite close le n matrix score selected
      (values selected) (fun row => vectors row selected) := by
  exact ⟨(columns selected choice.1).1, (columns selected choice.1).2.1,
    (columns selected choice.1).2.2, choice⟩

-- Both propositions are unresolved interpretation parameters.
def GuardedRequirement (evaluationContext requirement : Prop) : Prop := evaluationContext → requirement
def guarded_source_obligation_c42b068b9ae3adc107f18356b65c718e := @GuardedRequirement
def guarded_source_obligation_ad6dff2caa4c4f5789a592e049fd51de := @GuardedRequirement
def guarded_source_obligation_626b6a87225e458b77cd8e6603bc8fac := @GuardedRequirement
def guarded_source_obligation_faeaf0f90a2e291f11a9cc691f74f2e2 := @GuardedRequirement
def guarded_source_obligation_e863efc9521e1bb0d15f2d3bb714712e := @GuardedRequirement
def guarded_source_obligation_ce3c463c43b9dc0bba2d5d7c69fae336 := @GuardedRequirement
def guarded_source_obligation_7e02a3a7388fecc304be028fc982d192 := @GuardedRequirement
def guarded_source_obligation_efcf24286822a2c0ab7a1c103b600195 := @GuardedRequirement
def guarded_source_obligation_bcc36e4f510a27559a2296ed81480189 := @GuardedRequirement
def guarded_source_obligation_7378c27b8fca2074c88e064a78288163 := @GuardedRequirement
def guarded_source_obligation_c5b3e22f1d50aeda0b57dfc9817a391f := @GuardedRequirement
def guarded_source_obligation_d782452e2a67f8579aa1f0a128723772 := @GuardedRequirement
def guarded_source_obligation_2114c52a9d114ee7ee174f6c363e4bdc := @GuardedRequirement
def guarded_source_obligation_a26df754648b70e53125910fc6205571 := @GuardedRequirement
def guarded_source_obligation_bbaddf5503c5d2aaa5af3e3d1068a52e := @GuardedRequirement
def guarded_source_obligation_389a5b86daa933959745de80135fcbc7 := @GuardedRequirement
end NumericalObligations
