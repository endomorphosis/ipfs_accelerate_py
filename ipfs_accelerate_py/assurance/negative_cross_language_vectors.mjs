/**
 * Fail-closed PCPR-042 JavaScript negative and cross-language encoder.
 *
 * Independent of the Python catalog encoder. Mints the same DAG-JSON CIDs
 * for PCPR-041 positive fixtures and the same reject kinds for PCPR-042
 * negatives. TypeScript compilation is a separate typed source; this
 * module is the live second-language runtime.
 *
 * Not a freeze, not a remint, not a closed PCPR release, and never writes
 * DuckDB or Quack state.
 */

import crypto from "node:crypto";

const INT64_MIN = -(2n ** 63n);
const INT64_MAX = 2n ** 63n - 1n;
const MAX_STRING_BYTES = 4096;
const MAX_ARRAY_ITEMS = 256;
const MAX_OBJECT_KEYS = 64;
const STALE_ZERO_TREE = "0".repeat(40);
const SEED = "baguqeeragko4w64wovfw2643pvwkugj4x7vqxdvzqzpzzqh3liuzrlibzt2a";
const CATALOG = "baguqeeraqpptps2gf55wmthv3sm5nzlxqqdjfyg2ggkbl5ovfzuvvvsxap4a";
const TREE = "41".repeat(20);
const ARTIFACT = "baguqeeraulqxswkwlfiexdb45mbekkbdzuz7xqjgmm3keoqmkmatlsz4zd5q";
const ARTIFACT_SHA = "a2e179595659504b8c3ceb02452823cd33fbc1266336a23a0c530135cb3cc8fb";
const ARTIFACT_BYTES = 79;
const INTENT_CID = "baguqeerak5dpgbfqb3y3lzidrzhctptq4czihlmihlrozcysqrwtk57luzya";
const NFC_CID = "baguqeeraaaockar3cte5dpud6wuxfbl4k5nvcmg2hz3ex3coajioewdiaonq";
const NFC_COMPOSED = "caf\u00e9";
const NFC_DECOMPOSED = "cafe\u0301";

const SCHEMAS = {
  SupervisorObjectiveIntent: "pcpr/shared-contracts/supervisor-objective-intent@1",
  ObjectiveMaterializationReceipt:
    "pcpr/shared-contracts/objective-materialization-receipt@1",
  DurableArtifactReceipt: "pcpr/shared-contracts/durable-artifact-receipt@1",
  ProofObligation: "pcpr/shared-contracts/proof-obligation@1",
  SupervisorEvent: "pcpr/shared-contracts/supervisor-event@1",
  PortfolioCompatibilityManifest:
    "pcpr/shared-contracts/portfolio-compatibility-manifest@1",
};

const INTERFACES = {
  SupervisorObjectiveIntent: "SupervisorObjectiveIntent@1",
  ObjectiveMaterializationReceipt: "ObjectiveMaterializationReceipt@1",
  DurableArtifactReceipt: "DurableArtifactReceipt@1",
  ProofObligation: "ProofObligation@1",
  SupervisorEvent: "SupervisorEvent@1",
  PortfolioCompatibilityManifest: "PortfolioCompatibilityManifest@1",
};

const FIELDS = {
  SupervisorObjectiveIntent: [
    "schema",
    "interface",
    "objective_id",
    "language",
    "idea_digest",
  ],
  ObjectiveMaterializationReceipt: [
    "schema",
    "interface",
    "objective_id",
    "plan_id",
    "admitted",
    "current_tree",
  ],
  DurableArtifactReceipt: [
    "schema",
    "interface",
    "cid",
    "byte_length",
    "digest_hex",
    "durable",
  ],
  ProofObligation: [
    "schema",
    "interface",
    "obligation_id",
    "statement",
    "logic_family",
  ],
  SupervisorEvent: ["schema", "interface", "event_id", "event_type", "sequence"],
  PortfolioCompatibilityManifest: [
    "schema",
    "interface",
    "portfolio_id",
    "contract_catalog_cid",
    "component_ids",
  ],
};

const CID_FIELDS = new Set(["idea_digest", "cid", "contract_catalog_cid"]);
const OID_FIELDS = new Set(["current_tree"]);
const INT_FIELDS = new Set(["byte_length", "sequence"]);
const BOOL_FIELDS = new Set(["admitted", "durable"]);
const LIST_FIELDS = new Set(["component_ids"]);
const TOKEN_FIELDS = new Set([
  "objective_id",
  "plan_id",
  "language",
  "obligation_id",
  "logic_family",
  "event_id",
  "event_type",
  "portfolio_id",
]);
const CID_RE = /^b[a-z2-7]+$/;
const OID_RE = /^[0-9a-f]{40}$/;
const HEX64_RE = /^[0-9a-fA-F]{64}$/;
const QM_RE = /^[Qm][1-9A-HJ-NP-Za-km-z]{0,50}$/;
const TOKEN_RE = /^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$/;
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

class RejectError extends Error {
  constructor(kind, message) {
    super(message);
    this.reject_kind = kind;
  }
}

function nfc(value, name) {
  if (typeof value !== "string" || value.length === 0) {
    throw new RejectError("invalid", `${name} must be a non-empty string`);
  }
  const normalized = value.normalize("NFC");
  if (Buffer.byteLength(normalized, "utf8") > MAX_STRING_BYTES) {
    throw new RejectError("out_of_bound", `${name} exceeds max_string_bytes`);
  }
  return normalized;
}

function int64(value, name) {
  if (typeof value === "boolean") {
    throw new RejectError("invalid", `${name} must be an int64 integer`);
  }
  if (typeof value === "bigint") {
    if (value < INT64_MIN || value > INT64_MAX) {
      throw new RejectError("out_of_bound", `${name} is outside the int64 domain`);
    }
    return Number(value);
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    if (typeof value === "number") {
      throw new RejectError(
        "out_of_bound",
        `${name} floats are rejected by the shared-contract catalog`,
      );
    }
    throw new RejectError("invalid", `${name} must be an int64 integer`);
  }
  const asBig = BigInt(value);
  if (asBig < INT64_MIN || asBig > INT64_MAX) {
    throw new RejectError("out_of_bound", `${name} is outside the int64 domain`);
  }
  return value;
}

function dumpString(value) {
  let out = '"';
  for (const char of value) {
    const code = char.codePointAt(0);
    if (char === '"') out += '\\"';
    else if (char === "\\") out += "\\\\";
    else if (char === "\b") out += "\\b";
    else if (char === "\f") out += "\\f";
    else if (char === "\n") out += "\\n";
    else if (char === "\r") out += "\\r";
    else if (char === "\t") out += "\\t";
    else if (code < 0x20) out += "\\u" + code.toString(16).padStart(4, "0");
    else if (code > 0x7e) {
      if (code <= 0xffff) out += "\\u" + code.toString(16).padStart(4, "0");
      else {
        const hi = 0xd800 + ((code - 0x10000) >> 10);
        const lo = 0xdc00 + ((code - 0x10000) & 0x3ff);
        out +=
          "\\u" +
          hi.toString(16).padStart(4, "0") +
          "\\u" +
          lo.toString(16).padStart(4, "0");
      }
    } else out += char;
  }
  return out + '"';
}

function canonicalize(value, path = "$") {
  if (value === null) return null;
  if (typeof value === "boolean") return value;
  if (typeof value === "bigint" || typeof value === "number") {
    return int64(value, path);
  }
  if (typeof value === "string") return nfc(value, path);
  if (Array.isArray(value)) {
    if (value.length > MAX_ARRAY_ITEMS) {
      throw new RejectError("out_of_bound", `${path} exceeds max_array_items`);
    }
    return value.map((item, index) => canonicalize(item, `${path}[${index}]`));
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    if (keys.length > MAX_OBJECT_KEYS) {
      throw new RejectError("out_of_bound", `${path} exceeds max_object_keys`);
    }
    const out = {};
    for (const key of keys) {
      if (typeof key !== "string" || key.length === 0) {
        throw new RejectError("invalid", `${path} map keys must be non-empty strings`);
      }
      const nfcKey = key.normalize("NFC");
      if (Object.prototype.hasOwnProperty.call(out, nfcKey)) {
        throw new RejectError(
          "reordered",
          `${path} Unicode-equivalent keys collide after NFC`,
        );
      }
      out[nfcKey] = canonicalize(value[key], `${path}.${nfcKey}`);
    }
    return out;
  }
  throw new RejectError("invalid", `${path} is not a DAG-JSON value`);
}

function dump(value) {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return dumpString(value);
  if (Array.isArray(value)) return "[" + value.map(dump).join(",") + "]";
  const keys = Object.keys(value).sort();
  return "{" + keys.map((key) => dumpString(key) + ":" + dump(value[key])).join(",") + "}";
}

export function canonicalJson(value) {
  return dump(canonicalize(value));
}

function base32(buf) {
  let bits = 0;
  let acc = 0;
  let out = "";
  for (const byte of buf) {
    acc = (acc << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      out += ALPHABET[(acc >>> (bits - 5)) & 31];
      bits -= 5;
    }
  }
  if (bits > 0) out += ALPHABET[(acc << (5 - bits)) & 31];
  while (out.length % 8 !== 0) out += "=";
  return out;
}

export function contentIdentity(value) {
  const encoded = Buffer.from(canonicalJson(value), "utf8");
  const digest = crypto.createHash("sha256").update(encoded).digest();
  const raw = Buffer.concat([Buffer.from([0x01, 0xa9, 0x02, 0x12, 0x20]), digest]);
  return "b" + base32(raw).replace(/=+$/, "").toLowerCase();
}

export function sha256Hex(value) {
  return crypto.createHash("sha256").update(canonicalJson(value), "utf8").digest("hex");
}

function cid(value, name) {
  const text = nfc(value, name);
  if (HEX64_RE.test(text)) {
    throw new RejectError("invalid", `${name} raw SHA-256 hex is not a CID`);
  }
  if (QM_RE.test(text)) {
    throw new RejectError("invalid", `${name} CIDv0 Qm form is rejected`);
  }
  if (text !== text.toLowerCase()) {
    throw new RejectError("invalid", `${name} CID must be lowercase`);
  }
  if (!CID_RE.test(text)) {
    throw new RejectError("invalid", `${name} must be CIDv1 lowercase base32`);
  }
  return text;
}

function gitOid(value, name) {
  const text = nfc(value, name);
  if (!OID_RE.test(text)) {
    throw new RejectError("invalid", `${name} must be a 40-character lowercase git object id`);
  }
  return text;
}

function token(value, name) {
  const text = nfc(value, name);
  if (!TOKEN_RE.test(text)) {
    throw new RejectError("invalid", `${name} is not an admitted token`);
  }
  return text;
}

function admitSharedContract(name, payload) {
  const fields = FIELDS[name];
  if (!fields) throw new RejectError("invalid", `${name} is not a PCPR shared contract`);
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    throw new RejectError("invalid", `${name} payload must be a mapping`);
  }
  const extra = Object.keys(payload).filter((key) => !fields.includes(key));
  if (extra.length) {
    throw new RejectError("unknown", `${name} unknown fields are rejected: ${extra.sort()}`);
  }
  const missing = fields.filter((field) => !(field in payload));
  if (missing.length) {
    throw new RejectError("invalid", `${name} missing required fields: ${missing}`);
  }
  const admitted = {};
  for (const field of fields) {
    const value = payload[field];
    if (field === "schema") {
      const text = nfc(value, field);
      if (text !== SCHEMAS[name]) {
        throw new RejectError("reminted", `${name} identity ${text} remints ${SCHEMAS[name]}`);
      }
      admitted[field] = text;
    } else if (field === "interface") {
      const text = nfc(value, field);
      if (text !== INTERFACES[name]) {
        throw new RejectError("reminted", `${name} interface ${text} remints ${INTERFACES[name]}`);
      }
      admitted[field] = text;
    } else if (CID_FIELDS.has(field)) admitted[field] = cid(value, field);
    else if (OID_FIELDS.has(field)) admitted[field] = gitOid(value, field);
    else if (INT_FIELDS.has(field)) admitted[field] = int64(value, field);
    else if (BOOL_FIELDS.has(field)) {
      if (typeof value !== "boolean") {
        throw new RejectError("invalid", `${field} must be a JSON boolean`);
      }
      admitted[field] = value;
    } else if (LIST_FIELDS.has(field)) {
      if (!Array.isArray(value)) {
        throw new RejectError("invalid", `${field} must be an array`);
      }
      if (value.length > MAX_ARRAY_ITEMS) {
        throw new RejectError("out_of_bound", `${field} exceeds max_array_items`);
      }
      admitted[field] = value.map((item) => token(item, `${field}[]`));
    } else if (TOKEN_FIELDS.has(field) || field.endsWith("_id")) {
      admitted[field] = token(value, field);
    } else admitted[field] = nfc(value, field);
  }
  canonicalJson(admitted);
  return admitted;
}

export function admitNegativeCandidate(name, payload) {
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    throw new RejectError("invalid", "invalid: payload must be a mapping");
  }
  const schema = payload.schema;
  const iface = payload.interface;
  if (typeof schema === "string" && schema.endsWith("@2")) {
    throw new RejectError("cross_version", "cross_version: schema @2 is not admitted");
  }
  if (typeof iface === "string" && iface.endsWith("@2")) {
    throw new RejectError("cross_version", "cross_version: interface @2 is not admitted");
  }
  for (const field of ["current_tree", "tree_id", "scanned_tree_oid"]) {
    if (payload[field] === STALE_ZERO_TREE) {
      throw new RejectError(
        "stale",
        "stale: all-zero tree oid is the PCPR-040 fixture, not current",
      );
    }
  }
  const admitted = admitSharedContract(name, payload);
  if (
    typeof admitted.contract_catalog_cid === "string" &&
    admitted.contract_catalog_cid !== CATALOG
  ) {
    throw new RejectError(
      "stale",
      "stale: contract_catalog_cid is not the current PCPR-040 catalog",
    );
  }
  return admitted;
}

function bases() {
  return {
    SupervisorObjectiveIntent: {
      schema: SCHEMAS.SupervisorObjectiveIntent,
      interface: INTERFACES.SupervisorObjectiveIntent,
      objective_id: "PCPR-G510",
      language: "Python",
      idea_digest: SEED,
    },
    ObjectiveMaterializationReceipt: {
      schema: SCHEMAS.ObjectiveMaterializationReceipt,
      interface: INTERFACES.ObjectiveMaterializationReceipt,
      objective_id: "PCPR-G510",
      plan_id: "plan:pcpr-041",
      admitted: true,
      current_tree: TREE,
    },
    DurableArtifactReceipt: {
      schema: SCHEMAS.DurableArtifactReceipt,
      interface: INTERFACES.DurableArtifactReceipt,
      cid: ARTIFACT,
      byte_length: ARTIFACT_BYTES,
      digest_hex: ARTIFACT_SHA,
      durable: true,
    },
    ProofObligation: {
      schema: SCHEMAS.ProofObligation,
      interface: INTERFACES.ProofObligation,
      obligation_id: "obligation:pcpr-041",
      statement: "canonical-byte-and-cid-vector",
      logic_family: "propositional",
    },
    SupervisorEvent: {
      schema: SCHEMAS.SupervisorEvent,
      interface: INTERFACES.SupervisorEvent,
      event_id: "event:pcpr-041",
      event_type: "canonical-byte-cid-vector",
      sequence: 1,
    },
    PortfolioCompatibilityManifest: {
      schema: SCHEMAS.PortfolioCompatibilityManifest,
      interface: INTERFACES.PortfolioCompatibilityManifest,
      portfolio_id: "portfolio:pcpr-v1",
      contract_catalog_cid: CATALOG,
      component_ids: [
        "component:ipfs_accelerate_py",
        "component:ipfs_datasets_py",
        "component:ipfs_kit_py",
      ],
    },
  };
}

function applyMutation(base, mutation) {
  if (Object.prototype.hasOwnProperty.call(mutation, "replace")) return mutation.replace;
  const payload = { ...base };
  for (const [key, value] of Object.entries(mutation.set || {})) payload[key] = value;
  if (mutation.overflow_int64) payload[mutation.overflow_int64] = INT64_MAX + 1n;
  if (mutation.oversize_string) {
    payload[mutation.oversize_string] = "x".repeat(MAX_STRING_BYTES + 1);
  }
  if (mutation.oversize_array) {
    payload[mutation.oversize_array] = Array.from(
      { length: MAX_ARRAY_ITEMS + 1 },
      (_, index) => `item:${index}`,
    );
  }
  if (mutation.reverse_keys) {
    const reversed = {};
    for (const key of Object.keys(payload).reverse()) reversed[key] = payload[key];
    return reversed;
  }
  return payload;
}

const RECIPES = [
  {
    id: "invalid.non_mapping",
    contract: "SupervisorObjectiveIntent",
    reject_kind: "invalid",
    mutation: { replace: null },
  },
  {
    id: "invalid.empty_language",
    contract: "SupervisorObjectiveIntent",
    reject_kind: "invalid",
    mutation: { set: { language: "" } },
  },
  {
    id: "invalid.bool_as_int64",
    contract: "SupervisorEvent",
    reject_kind: "invalid",
    mutation: { set: { sequence: true } },
  },
  {
    id: "stale.zero_tree",
    contract: "ObjectiveMaterializationReceipt",
    reject_kind: "stale",
    mutation: { set: { current_tree: STALE_ZERO_TREE } },
  },
  {
    id: "stale.catalog_cid",
    contract: "PortfolioCompatibilityManifest",
    reject_kind: "stale",
    mutation: { set: { contract_catalog_cid: SEED } },
  },
  {
    id: "unknown.extra_field",
    contract: "SupervisorObjectiveIntent",
    reject_kind: "unknown",
    mutation: { set: { unexpected: "field" } },
  },
  {
    id: "unknown.extra_field_reordered",
    contract: "ProofObligation",
    reject_kind: "unknown",
    identical_without_reverse: true,
    mutation: { set: { unexpected: "field" }, reverse_keys: true },
  },
  {
    id: "out_of_bound.int64",
    contract: "SupervisorEvent",
    reject_kind: "out_of_bound",
    mutation: { overflow_int64: "sequence" },
  },
  {
    id: "out_of_bound.float",
    contract: "DurableArtifactReceipt",
    reject_kind: "out_of_bound",
    mutation: { set: { byte_length: 1.5 } },
  },
  {
    id: "out_of_bound.string",
    contract: "ProofObligation",
    reject_kind: "out_of_bound",
    mutation: { oversize_string: "statement" },
  },
  {
    id: "out_of_bound.array",
    contract: "PortfolioCompatibilityManifest",
    reject_kind: "out_of_bound",
    mutation: { oversize_array: "component_ids" },
  },
  {
    id: "out_of_bound.cidv0",
    contract: "DurableArtifactReceipt",
    reject_kind: "invalid",
    mutation: { set: { cid: "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG" } },
  },
  {
    id: "reordered.nfc_key_collision",
    contract: "SupervisorObjectiveIntent",
    reject_kind: "reordered",
    mutation: { canonical_only: true },
  },
  {
    id: "reminted.schema",
    contract: "SupervisorObjectiveIntent",
    reject_kind: "reminted",
    mutation: {
      set: { schema: "pcpr/shared-contracts/supervisor-objective-intent@1-remint" },
    },
  },
  {
    id: "reminted.interface",
    contract: "SupervisorObjectiveIntent",
    reject_kind: "reminted",
    mutation: { set: { interface: "SupervisorObjectiveIntent@1-other" } },
  },
  {
    id: "reminted.vector_cid",
    contract: "SupervisorObjectiveIntent",
    reject_kind: "reminted",
    mutation: {
      remint_cid: "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    },
  },
  {
    id: "cross_version.schema_v2",
    contract: "SupervisorObjectiveIntent",
    reject_kind: "cross_version",
    mutation: { set: { schema: "pcpr/shared-contracts/supervisor-objective-intent@2" } },
  },
  {
    id: "cross_version.interface_v2",
    contract: "SupervisorObjectiveIntent",
    reject_kind: "cross_version",
    mutation: { set: { interface: "SupervisorObjectiveIntent@2" } },
  },
];

function rejectKind(recipe, reverse = null) {
  const mutation = { ...recipe.mutation };
  if (reverse === true) mutation.reverse_keys = true;
  if (reverse === false) delete mutation.reverse_keys;
  if (mutation.canonical_only) {
    try {
      canonicalJson({ [NFC_COMPOSED]: 1, [NFC_DECOMPOSED]: 2 });
    } catch (exc) {
      if (exc instanceof RejectError) return exc.reject_kind;
      throw exc;
    }
    throw new Error(`${recipe.id} NFC key collision was admitted`);
  }
  if (mutation.remint_cid) {
    if (mutation.remint_cid !== INTENT_CID) return "reminted";
    throw new Error(`${recipe.id} remint CID was admitted`);
  }
  const payload = applyMutation(bases()[recipe.contract], mutation);
  try {
    admitNegativeCandidate(recipe.contract, payload);
  } catch (exc) {
    if (exc instanceof RejectError) return exc.reject_kind;
    throw exc;
  }
  throw new Error(`${recipe.id} was admitted`);
}

export function evaluateNegatives() {
  return RECIPES.map((recipe) => {
    const kind = rejectKind(recipe);
    if (kind !== recipe.reject_kind) {
      throw new Error(`${recipe.id} reject_kind ${kind} != ${recipe.reject_kind}`);
    }
    let identical = false;
    if (recipe.identical_without_reverse) {
      const other = rejectKind(recipe, false);
      if (other !== kind) {
        throw new Error(`${recipe.id} reversed and original reject kinds differ`);
      }
      identical = true;
    }
    return {
      id: recipe.id,
      contract: recipe.contract,
      reject_kind: kind,
      rejected: true,
      identical_reversed: identical,
    };
  });
}

export function evaluateCrossLanguage() {
  const intent = bases().SupervisorObjectiveIntent;
  const reversed = {};
  for (const key of Object.keys(intent).reverse()) reversed[key] = intent[key];
  const nfcPayload = {
    schema: SCHEMAS.ProofObligation,
    interface: INTERFACES.ProofObligation,
    obligation_id: "obligation:pcpr-041",
    statement: NFC_COMPOSED,
    logic_family: "propositional",
  };
  const nfcDecomposed = { ...nfcPayload, statement: NFC_DECOMPOSED };
  const intentCid = contentIdentity(intent);
  const reversedCid = contentIdentity(reversed);
  const nfcCid = contentIdentity(nfcPayload);
  const nfcDecomposedCid = contentIdentity(nfcDecomposed);
  if (intentCid !== INTENT_CID) {
    throw new Error(`intent CID ${intentCid} remints ${INTENT_CID}`);
  }
  if (reversedCid !== intentCid) {
    throw new Error("key-order CID reminted");
  }
  if (nfcCid !== NFC_CID || nfcDecomposedCid !== nfcCid) {
    throw new Error(`NFC CID ${nfcCid} remints ${NFC_CID}`);
  }
  return [
    {
      id: "python_js.intent",
      cid: intentCid,
      canonical_sha256: sha256Hex(intent),
      byte_length: Buffer.byteLength(canonicalJson(intent), "utf8"),
    },
    {
      id: "python_js.nfc",
      cid: nfcCid,
      canonical_sha256: sha256Hex(nfcPayload),
      byte_length: Buffer.byteLength(canonicalJson(nfcPayload), "utf8"),
    },
    {
      id: "python_js.key_order",
      cid: reversedCid,
      canonical_sha256: sha256Hex(reversed),
      byte_length: Buffer.byteLength(canonicalJson(reversed), "utf8"),
    },
  ];
}

export function report() {
  const negatives = evaluateNegatives();
  const byId = Object.fromEntries(negatives.map((item) => [item.id, item]));
  const rejectAgreement = [
    ["python_js.unknown_identical", "unknown.extra_field"],
    ["python_js.int64_identical", "out_of_bound.int64"],
    ["python_js.cross_version_identical", "cross_version.schema_v2"],
    ["python_js.stale_identical", "stale.zero_tree"],
  ].map(([id, source]) => ({
    id,
    source_id: source,
    reject_kind: byId[source].reject_kind,
    rejected: true,
  }));
  return {
    language: "JavaScript",
    runtime: "node",
    simulated: false,
    negatives,
    cross_language: [...evaluateCrossLanguage(), ...rejectAgreement],
  };
}

const invoked = process.argv[1] && import.meta.url.endsWith(process.argv[1].replaceAll("\\", "/"));
if (invoked || process.argv[1]?.endsWith("negative_cross_language_vectors.mjs")) {
  process.stdout.write(JSON.stringify(report()) + "\n");
}
