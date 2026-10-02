import Init
-- Recorded table arithmetic only; no Python-origin or universal-runtime theorem.
-- source bafkreia3xcqs3wtfgt37k7srqyuls374czxxfpbtzhj7lt6tud4noecfhu
-- domain baguqeera6ja56wvkmi2cff6om2ewexiddz2yvmz7kioz63mn6rlon3pngysq
-- trace baguqeeraswe2uhbsjggkgsrzd5lvzkwugrnzdcartocag3lgaltyrplcb5bq
namespace CodebaseFiniteInteger
set_option autoImplicit false
def inputs : List Int := [(-2 : Int), (-1 : Int), (0 : Int), (1 : Int), (2 : Int)]
def rows : List (Int × Int) := [((-2 : Int), (0 : Int)), ((-1 : Int), (1 : Int)), ((0 : Int), (2 : Int)), ((1 : Int), (3 : Int)), ((2 : Int), (4 : Int))]
theorem domain_coverage : rows.map Prod.fst = inputs := by decide
theorem recorded_integer_types : (5 : Nat) = inputs.length := by decide
theorem observed_body_offset : rows.all (fun row => decide (row.2 = row.1 + (2 : Int))) = true := by decide
theorem offset_clause : rows.all (fun row => decide (row.2 = row.1 + (2 : Int))) = true := by decide
end CodebaseFiniteInteger
