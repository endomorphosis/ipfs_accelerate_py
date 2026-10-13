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
def guarded_source_obligation_5a038dd462b6be503d9522b52123007d := @GuardedRequirement
def guarded_source_obligation_9b4a4e9a2c9b8c87ddab888bb414bef0 := @GuardedRequirement
def guarded_source_obligation_d28d2655f7946ed8ef383a1d881727f7 := @GuardedRequirement
def guarded_source_obligation_ebe0705a3b3306aeddab71509b9adffb := @GuardedRequirement
def guarded_source_obligation_1a5c7882474d1b22e9fe68b621d482db := @GuardedRequirement
def guarded_source_obligation_0023f078671770dc945bd5300df6a1bf := @GuardedRequirement
def guarded_source_obligation_cacd8c3b402c5f6f354c79140b4500bd := @GuardedRequirement
def guarded_source_obligation_ff12556f7a9da07bedb6a52cb0d58e79 := @GuardedRequirement
def guarded_source_obligation_a4a99ce61a5b158af084c7360635158c := @GuardedRequirement
def guarded_source_obligation_2922eed70d698ed294c64dfe1d2481c4 := @GuardedRequirement
def guarded_source_obligation_9192b0566ff016ce4f73405970754044 := @GuardedRequirement
def guarded_source_obligation_d92b3caab44018c2dcabc679f65e7f38 := @GuardedRequirement
def guarded_source_obligation_2bb52a694a36216d7427ae638853fca0 := @GuardedRequirement
def guarded_source_obligation_388a7376504aea28b8978cf0e3fd3140 := @GuardedRequirement
def guarded_source_obligation_9fbecf3ab32e5ad7dcd7927b40caeec3 := @GuardedRequirement
def guarded_source_obligation_b4c7626cd52b1806f4b4210c97c872c2 := @GuardedRequirement
def guarded_source_obligation_0e068762a8602ab282774d6ae3da0e74 := @GuardedRequirement
def guarded_source_obligation_a041773323125fe29c94893ef1adae80 := @GuardedRequirement
def guarded_source_obligation_d3c6d34a7e8d20b0e7e00fd24981f170 := @GuardedRequirement
def guarded_source_obligation_f24a1555eeb2a843dfb770529bf35c2d := @GuardedRequirement
def guarded_source_obligation_202f29114e36c216d5cf7e25eff0cd5c := @GuardedRequirement
def guarded_source_obligation_4117ccb2d65efa8b9f49692c79a8a2bb := @GuardedRequirement
def guarded_source_obligation_d533c2ba99ab2af3df7473625bd6f693 := @GuardedRequirement
def guarded_source_obligation_f72fa30229ebc58f4147e238317a1e0b := @GuardedRequirement
def guarded_source_obligation_71dbda1c6f416f23530e3b8a4174c0b4 := @GuardedRequirement
def guarded_source_obligation_cb26dcb8e92384afb81cbf2dea0ea111 := @GuardedRequirement
def guarded_source_obligation_f777a7c698d2e6901a2b2c80bbe7059a := @GuardedRequirement
def guarded_source_obligation_7aa12bd59aee17ef4a4feeedcd444992 := @GuardedRequirement
def guarded_source_obligation_46305880dc91d42dba59e93038166397 := @GuardedRequirement
def guarded_source_obligation_d04e557c2ac61b877bc4980e1453d4a7 := @GuardedRequirement
def guarded_source_obligation_69ad59dda218981a998c1f98fe4777af := @GuardedRequirement
def guarded_source_obligation_e8ec13f527c5f439224446909a9e3420 := @GuardedRequirement
def guarded_source_obligation_64697e51c7ee2651a9e236a25c70bfce := @GuardedRequirement
def guarded_source_obligation_970ab8c3b3442ad13d7415c8a9ac5cdf := @GuardedRequirement
def guarded_source_obligation_be3e3f3c12c2be2bd7e1df4d31b31b05 := @GuardedRequirement
def guarded_source_obligation_bf4c108a77ec3092e97b962aa1415d4d := @GuardedRequirement
def guarded_source_obligation_719300391ab92b1e0ba8474713ed0f97 := @GuardedRequirement
def guarded_source_obligation_4db21431dfc59f755469016e73d7e2de := @GuardedRequirement
def guarded_source_obligation_2c1350aa69598e0530c49feeecc75656 := @GuardedRequirement
def guarded_source_obligation_38dd9ef5c0db66bbb01974e1fef4d89c := @GuardedRequirement
def guarded_source_obligation_00323efee91e0889c4f71fda709caaaa := @GuardedRequirement
def guarded_source_obligation_0bacc95053f406b4e18b31efd834bad6 := @GuardedRequirement
def guarded_source_obligation_97acbef6f8af4d3f79126e0adf4c4be7 := @GuardedRequirement
def guarded_source_obligation_c8e79569101985119481343eb0e5a6a1 := @GuardedRequirement
def guarded_source_obligation_e2319d97db8a7c1ec5ff5cd861f5ef68 := @GuardedRequirement
def guarded_source_obligation_d3d2d262012a712875e0cc61f7074bfe := @GuardedRequirement
def guarded_source_obligation_168034a8797fe816a6cc3fd067cba60a := @GuardedRequirement
def guarded_source_obligation_9f3fb2f8cb61645f76ace3ff1b6962d4 := @GuardedRequirement
def guarded_source_obligation_9e70f1a22a0a11c3f509071364c10a0d := @GuardedRequirement
def guarded_source_obligation_b8fa75a8b49f48523acac7c81876b6e6 := @GuardedRequirement
def guarded_source_obligation_8243235840646ff10517c5f7b62859ef := @GuardedRequirement
def guarded_source_obligation_ba214eaba3cd7e5e2045302f13fc9376 := @GuardedRequirement
def guarded_source_obligation_4bbce931f801fbf4196652609206f6a9 := @GuardedRequirement
def guarded_source_obligation_5cfd0fd3ab6e057acd7f698354571ef2 := @GuardedRequirement
def guarded_source_obligation_e1fdc3d373787b65a33234cd23194c3d := @GuardedRequirement
def guarded_source_obligation_9563c28aed6c2fc38c74c08fc077ed99 := @GuardedRequirement
def guarded_source_obligation_e0830f23d9f28d901d63e3f991853fa6 := @GuardedRequirement
def guarded_source_obligation_a3eca837bf944b4052aaf9ae09122985 := @GuardedRequirement
def guarded_source_obligation_dc64b57f3964607b63f0113fed89d844 := @GuardedRequirement
end NumericalObligations
