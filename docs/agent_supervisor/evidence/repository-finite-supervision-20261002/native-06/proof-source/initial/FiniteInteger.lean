import Init
-- Recorded table arithmetic only; no Python-origin or universal-runtime theorem.
-- source bafkreig2whzodbawj5uf6jnxwuklau7dypg4y7o76drwhcihfoxilmvtvq
-- domain baguqeera6ja56wvkmi2cff6om2ewexiddz2yvmz7kioz63mn6rlon3pngysq
-- trace baguqeerakwseu5nhnju4pkgub2yblhmdsfsv2626qpdehio554tra3gs7wwa
namespace CodebaseFiniteInteger
set_option autoImplicit false
def inputs : List Int := [(-2 : Int), (-1 : Int), (0 : Int), (1 : Int), (2 : Int)]
def rows : List (Int × Int) := [((-2 : Int), (-1 : Int)), ((-1 : Int), (0 : Int)), ((0 : Int), (1 : Int)), ((1 : Int), (2 : Int)), ((2 : Int), (3 : Int))]
theorem domain_coverage : rows.map Prod.fst = inputs := by decide
theorem recorded_integer_types : (5 : Nat) = inputs.length := by decide
theorem observed_body_offset : rows.all (fun row => decide (row.2 = row.1 + (1 : Int))) = true := by decide
theorem offset_counterexample : (-1 : Int) ≠ (-2 : Int) + (2 : Int) := by decide
end CodebaseFiniteInteger
