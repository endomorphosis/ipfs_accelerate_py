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

theorem last_index_bound (extent : Nat)  : extent - 1 < extent := by
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

def source_obligation_a0d0f4836a386f59f45de38ddd0ed3dd := @MatrixShape

def source_obligation_0c42cc95aae03f9b1ed2dab3a560ef82 := @EigenpairResidual

def source_obligation_8d90140bb486c38ae77a53c5693a503c := @SelectionMax

def source_obligation_6b7fa0f89f467d7bc0b767dc7d634d6b := @IndexSafety

def source_obligation_a9a3410c3dac8653402811385aac3c23 := @IndexSafety

def source_obligation_ea119f6f1f25852158ab914d08717967 := @EigensystemDimensions

def source_obligation_662a60ef129d3c485d7e222a45d2f2ae := @SameSelectedColumn

def source_obligation_668f5284b0e6df123d9ce77127f38d0c := @SelectionMax

def source_obligation_de9a05cb0d2b38b7e69b58a8255dd21a := @NonzeroVector

def source_obligation_e3cdf837e14264986b7e86bc4a192868 := @FiniteEigenpair

def source_obligation_3eab9e8001eb4d20e2ebacce9c8c61d8 := @ApproximateEigenpairResidual

end NumericalObligations
