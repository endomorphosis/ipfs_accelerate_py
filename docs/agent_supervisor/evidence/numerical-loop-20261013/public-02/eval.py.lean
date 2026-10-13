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
def guarded_source_obligation_8b604891e2d6512e7aa10b54ec252e9d := @GuardedRequirement
def guarded_source_obligation_39267f90685161be4760d7de524e99bd := @GuardedRequirement
def guarded_source_obligation_438a48626b421491cbd479c9277bf61e := @GuardedRequirement
def guarded_source_obligation_266dfc171777bb8c74a542a773b33d88 := @GuardedRequirement
def guarded_source_obligation_8daf0e0d80d6fe472ac1e32e7a6e3f76 := @GuardedRequirement
def guarded_source_obligation_0b9938e549e6a6975c9d71a6e80bd3a5 := @GuardedRequirement
def guarded_source_obligation_e4660e37c31fd5b0328273df1067707b := @GuardedRequirement
def guarded_source_obligation_935db43693c4723968db9db6cab038e4 := @GuardedRequirement
def guarded_source_obligation_5676a5cdd34b82872a76f70067f5e2e2 := @GuardedRequirement
def guarded_source_obligation_3654f41047dd53626ec90c74a52aa524 := @GuardedRequirement
def guarded_source_obligation_7a1f5f1c792bf9dd22ba422c31fc4d53 := @GuardedRequirement
def guarded_source_obligation_21da9580d511b077bdc250fe48c4eeb4 := @GuardedRequirement
def guarded_source_obligation_0b6e3290727ff78111dff2426c4c6702 := @GuardedRequirement
def guarded_source_obligation_5747d080edd0457724f9e6c70fd55844 := @GuardedRequirement
def guarded_source_obligation_fa93755033ecd2be76e5d3725b60304c := @GuardedRequirement
def guarded_source_obligation_b5cfd97da55b8e8c1e9f2a70c4e4c4cf := @GuardedRequirement
def guarded_source_obligation_5c62b57f62ad6e650dc7f29cfc3a0722 := @GuardedRequirement
def guarded_source_obligation_b14abb2530d47bdc4beb518a72fcc977 := @GuardedRequirement
def guarded_source_obligation_8cd022cd6186bc7ccdc0e31edc58601d := @GuardedRequirement
def guarded_source_obligation_f85400bca90fed3025921ab0745e4e45 := @GuardedRequirement
def guarded_source_obligation_b37f2419aaf196b023793f5f4c2e1665 := @GuardedRequirement
def guarded_source_obligation_ca0f49a26f1fdff23e0d746bd56f3bfb := @GuardedRequirement
def guarded_source_obligation_a7a86e8b0481eca4fc04ac3636c78f4b := @GuardedRequirement
def guarded_source_obligation_eda76cf7de6f36f98f714c4897327a7c := @GuardedRequirement
def guarded_source_obligation_76fa5ed1caf94bed4f980ff6d8922444 := @GuardedRequirement
def guarded_source_obligation_a32daa3690397a86a0d8ff1df90fde02 := @GuardedRequirement
def guarded_source_obligation_61427905e815b9396e0a474a718e6e76 := @GuardedRequirement
def guarded_source_obligation_0245e9251fc957b81bfa0c3a19c4b713 := @GuardedRequirement
def guarded_source_obligation_4cb0bcf9531d5e7a6a90e98797893a3a := @GuardedRequirement
def guarded_source_obligation_8ca07b31353e2ed11123220476fc1fa5 := @GuardedRequirement
def guarded_source_obligation_83753994804154105bc4002cd838c211 := @GuardedRequirement
def guarded_source_obligation_b556c48e737e814944d0ea084d9eac83 := @GuardedRequirement
def guarded_source_obligation_e3747bf6f85419b05461747bf0e7186b := @GuardedRequirement
def guarded_source_obligation_55151c469875f45c74a61e1b8f93235b := @GuardedRequirement
def guarded_source_obligation_0889e9a43c703d345e356c340bd4fd0a := @GuardedRequirement
def guarded_source_obligation_bf7de8ed5606ce41b379a92ed38a961a := @GuardedRequirement
def guarded_source_obligation_d428c6de979dae7c63e61781a694fc5c := @GuardedRequirement
def guarded_source_obligation_d2522a36708178a67d3ed7c153c7a81b := @GuardedRequirement
def guarded_source_obligation_93a65443509f3a621e8acbc8aefd82b2 := @GuardedRequirement
def guarded_source_obligation_d12d170d9c6ab68decd34b2e2c7cf732 := @GuardedRequirement
def guarded_source_obligation_45209efaaf2b2e4ad78ff22967dc9e08 := @GuardedRequirement
def guarded_source_obligation_46138a3593d589fa4cf3b682d3ef76ee := @GuardedRequirement
def guarded_source_obligation_78ea780b799f377b74a031df0e956c60 := @GuardedRequirement
def guarded_source_obligation_740590866aff3621094c1451a23c5fb5 := @GuardedRequirement
def guarded_source_obligation_46752cd55f772326c0442f99ee840d29 := @GuardedRequirement
def guarded_source_obligation_bf052f8eaa70949daa75e4b6aec4fac7 := @GuardedRequirement
def guarded_source_obligation_754edb57981cc0e80d101df2b4dd729c := @GuardedRequirement
def guarded_source_obligation_08d978a935f2791c0babba4a827af1cb := @GuardedRequirement
def guarded_source_obligation_c50cf1f7f539eaa5605b3145e96f9ea0 := @GuardedRequirement
def guarded_source_obligation_66bf4ae3c4793797d7c4cc0b8d5acd1f := @GuardedRequirement
def guarded_source_obligation_f5814f00c3c9643e3754ceec06c2bce4 := @GuardedRequirement
def guarded_source_obligation_08621bf8c0f20c1b4f01385d63abf0f3 := @GuardedRequirement
def guarded_source_obligation_7c30a21ccce8b1cae79c6194eda624be := @GuardedRequirement
def guarded_source_obligation_e7bef72ba4cd18fc1b8645c0166d6be8 := @GuardedRequirement
def guarded_source_obligation_9d1b34a961528b3b4ea3e04f582226b9 := @GuardedRequirement
def guarded_source_obligation_620a5dddb25f86b84a99b75c6a0644e4 := @GuardedRequirement
def guarded_source_obligation_092079e05a71ca54658d2b3cb7856a3b := @GuardedRequirement
def guarded_source_obligation_538630c6e37013af0131e872f6841cee := @GuardedRequirement
def guarded_source_obligation_a36cf11c6b9324d68829f063282c787d := @GuardedRequirement
def guarded_source_obligation_7dbecd0f9930d17f565a696ffd3394fb := @GuardedRequirement
def guarded_source_obligation_77882c16ba9cc61a98c0883a849455a3 := @GuardedRequirement
def guarded_source_obligation_67dd7cd140dd1c0e3cc43dabc24c6803 := @GuardedRequirement
def guarded_source_obligation_c5df1e449e1579960f08c32f0e958b82 := @GuardedRequirement
def guarded_source_obligation_f133f475f0d313872451e2131098c684 := @GuardedRequirement
def guarded_source_obligation_1ee83a96eecca17f60d0b9c17e80e631 := @GuardedRequirement
def guarded_source_obligation_6c32ede85157db096a5d043a71ffa5df := @GuardedRequirement
def guarded_source_obligation_a93b8021251ec1560ee6046525150d93 := @GuardedRequirement
def guarded_source_obligation_3c7cb8b127f6c904359b148cd2d66fc3 := @GuardedRequirement
def guarded_source_obligation_dc3914f0668088feb526e424a1c9d925 := @GuardedRequirement
def guarded_source_obligation_4a372015a19fc7f594322bc2d724d946 := @GuardedRequirement
def guarded_source_obligation_53b2688af7c94ea2308da0ac9567f297 := @GuardedRequirement
def guarded_source_obligation_d456a0caca7df45389f666fd9e83210b := @GuardedRequirement
def guarded_source_obligation_9674d6a9b7901fcb1c7b5c7357820c52 := @GuardedRequirement
def guarded_source_obligation_7d173da995a4b19b1f8d0be4c3566dd9 := @GuardedRequirement
def guarded_source_obligation_bdb808f698d59600adc66fbc9623cbaa := @GuardedRequirement
def guarded_source_obligation_3be0282f7f16761c55f0ff438f8977b2 := @GuardedRequirement
def guarded_source_obligation_877a11ae2456faed5b8ed2f9a8d202b0 := @GuardedRequirement
def guarded_source_obligation_00b642ed2420e4b8aca70543e8d87eaa := @GuardedRequirement
def guarded_source_obligation_e67286f8ca4eeebb4ab8208aa20b36bc := @GuardedRequirement
def guarded_source_obligation_576ee7bfb44afa66aa9bb7335c680932 := @GuardedRequirement
def guarded_source_obligation_4c944000256aa5a127d9f05ec403e2cd := @GuardedRequirement
def guarded_source_obligation_4a0932772f1046be2b7b730b5b9e3345 := @GuardedRequirement
def guarded_source_obligation_b6f9f60f55e0684e36f0256c1dede50d := @GuardedRequirement
def guarded_source_obligation_10090dba0be574b79da9271c623dc2dc := @GuardedRequirement
def guarded_source_obligation_d8a6733b7798bf365e32306bba9d07d9 := @GuardedRequirement
def guarded_source_obligation_8d39e0e0ced089ba185a709f70b44155 := @GuardedRequirement
def guarded_source_obligation_9cf9bc15186e130f631b6ce4b7b3a117 := @GuardedRequirement
def guarded_source_obligation_79f85b312d3d930345169b7ec15696ae := @GuardedRequirement
def guarded_source_obligation_6caa0e81ef8e3bcc4fa3eb8843b052d3 := @GuardedRequirement
end NumericalObligations
