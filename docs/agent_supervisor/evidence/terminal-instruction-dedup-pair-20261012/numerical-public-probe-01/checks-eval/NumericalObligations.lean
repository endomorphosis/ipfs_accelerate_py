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


-- Proposed interpretations of source ASSERT expressions. Callable evaluation,
-- Python truthiness, effects and reachability are unresolved parameters.
def AssertionPredicate (predicate : Prop) : Prop := predicate

def CallPredicate {Arguments Result : Type} (call : Arguments → Result)
    (truthy : Result → Prop) (arguments : Arguments) : Prop := truthy (call arguments)

def NegatedCallPredicate {Arguments Result : Type} (call : Arguments → Result)
    (truthy : Result → Prop) (arguments : Arguments) : Prop := ¬ truthy (call arguments)

-- Opposite predicates cannot both hold under the SAME supplied interpretation.
-- This supplies no witness for either predicate or for a Python call.
theorem opposite_call_predicates_incompatible {Arguments Result : Type}
    (call : Arguments → Result) (truthy : Result → Prop) (arguments : Arguments)
    (positive : CallPredicate call truthy arguments)
    (negative : NegatedCallPredicate call truthy arguments) : False := by
  exact negative positive

def source_obligation_740c893bb353d639c34b2a273a9f4848 := @MatrixShape

def source_obligation_a0def67200fcff2cde96ccff30c8b9cf := @EigenpairResidual

def source_obligation_fc55d046dbae42124272fe27104fbe5e := @SelectionMax

def source_obligation_578870438d1215efccce96398413bd88 := @IndexSafety

def source_obligation_e423320b51a0dfef884af0667ecdac89 := @IndexSafety

def source_obligation_4a8bdda45dd33a84e0ab6be8d7d0a0eb := @EigensystemDimensions

def source_obligation_c6ac03131eada3ae0a8e777856eab0a6 := @SameSelectedColumn

def source_obligation_47af77aa3f6121536a61ddf830af03df := @SelectionMax

def source_obligation_3ae391f43d785001574dfcaa31b10242 := @NonzeroVector

def source_obligation_bff23881a0ed72f5c40d84059efcffdb := @FiniteEigenpair

def source_obligation_3f463093c490193d8ee830af32503e9d := @ApproximateEigenpairResidual

def source_obligation_48d865cf8af18b1d565dacae2ec41e71 := @NegatedCallPredicate

def source_obligation_689e9dddb0d9cdd3ce9ed6ac8cc183ba := @NegatedCallPredicate

def source_obligation_322ac9c10d8ba78183a416533f710d67 := @NegatedCallPredicate

def source_obligation_b9e195cd657911b67c1139f7afe5a686 := @CallPredicate

end NumericalObligations
