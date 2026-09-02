/**
 * Fail-closed PCPR-042 TypeScript companion for negative and cross-language
 * vectors. Live tsc execution is typed unavailable in the sealed PATH.
 * The runnable second-language encoder is
 * ipfs_accelerate_py/assurance/negative_cross_language_vectors.mjs.
 *
 * Not a freeze, not a remint of PCPR-040/041 identities, not a closed
 * PCPR release, and never writes DuckDB or Quack state.
 */

export const INTERFACE = "NegativeCrossLanguageVectors@1";
export const SCHEMA = "ipfs_accelerate_py/assurance/negative-cross-language-vectors@1";
export const TASK_ID = "PCPR-042";
export const GOAL_ID = "PCPR-G520";
export const COMPATIBILITY_TASK = "PCPR-043";
export const POSITIVE_VECTOR_TASK = "PCPR-041";
export const TYPESCRIPT_COMPILER = "unavailable";
export const JAVASCRIPT_RUNTIME = "node";

export const PINNED_CATALOG_CID =
  "baguqeeraqpptps2gf55wmthv3sm5nzlxqqdjfyg2ggkbl5ovfzuvvvsxap4a";
export const PINNED_VECTOR_DOCUMENT_CID =
  "baguqeeraior5lq3fvefhgwsm3cjjaxozffmdmqmszwxy4ozfb37mnqx6lfca";
export const PINNED_INTENT_CID =
  "baguqeerak5dpgbfqb3y3lzidrzhctptq4czihlmihlrozcysqrwtk57luzya";
export const PINNED_NFC_CID =
  "baguqeeraaaockar3cte5dpud6wuxfbl4k5nvcmg2hz3ex3coajioewdiaonq";

export const NEGATIVE_CATEGORIES = [
  "invalid",
  "stale",
  "unknown",
  "out_of_bound",
  "reordered",
  "reminted",
  "cross_version",
] as const;

export const NEGATIVE_RECIPE_IDS = [
  "invalid.non_mapping",
  "invalid.empty_language",
  "invalid.bool_as_int64",
  "stale.zero_tree",
  "stale.catalog_cid",
  "unknown.extra_field",
  "unknown.extra_field_reordered",
  "out_of_bound.int64",
  "out_of_bound.float",
  "out_of_bound.string",
  "out_of_bound.array",
  "out_of_bound.cidv0",
  "reordered.nfc_key_collision",
  "reminted.schema",
  "reminted.interface",
  "reminted.vector_cid",
  "cross_version.schema_v2",
  "cross_version.interface_v2",
] as const;

export const CROSS_LANGUAGE_IDS = [
  "python_js.intent",
  "python_js.nfc",
  "python_js.key_order",
  "python_js.unknown_identical",
  "python_js.int64_identical",
  "python_js.cross_version_identical",
  "python_js.stale_identical",
] as const;

export type RejectKind =
  | "invalid"
  | "stale"
  | "unknown"
  | "out_of_bound"
  | "reordered"
  | "reminted"
  | "cross_version";

export interface NegativeRecipe {
  readonly id: (typeof NEGATIVE_RECIPE_IDS)[number];
  readonly reject_kind: RejectKind;
}

export const RUNNABLE_JAVASCRIPT_MODULE =
  "ipfs_accelerate_py/assurance/negative_cross_language_vectors.mjs";
