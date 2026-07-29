#!/usr/bin/env node
/**
 * Deterministic TypeScript-compiler AST fact extraction.
 *
 * Input and output are one JSON document on stdin/stdout. Source bodies are
 * never written to disk or returned. The Python caller supplies the process,
 * byte, file, and memory envelope; this script independently caps stdin.
 */

import crypto from "node:crypto";
import { createRequire } from "node:module";

const PROTOCOL_VERSION = 1;
const PRODUCER_VERSION = "typescript-ast-extractor@1";
const DEFAULT_MAX_INPUT_BYTES = 4 * 1024 * 1024;
const LANGUAGES = new Set(["javascript", "jsx", "typescript", "tsx"]);

function emit(value, status = 0) {
  process.stdout.write(`${JSON.stringify(value)}\n`, () => {
    process.exitCode = status;
  });
}

function failure(code, message, status = 2) {
  emit(
    {
      protocol_version: PROTOCOL_VERSION,
      ok: false,
      error: {
        code,
        type: "PolyglotASTExtractionError",
        message: String(message).slice(0, 1024),
      },
    },
    status,
  );
}

function configuredInputLimit() {
  const parsed = Number.parseInt(
    process.env.POLYGLOT_AST_MAX_INPUT_BYTES || "",
    10,
  );
  return Number.isSafeInteger(parsed) && parsed > 0
    ? parsed
    : DEFAULT_MAX_INPUT_BYTES;
}

async function readRequest() {
  const chunks = [];
  let length = 0;
  const maximum = configuredInputLimit();
  for await (const chunk of process.stdin) {
    length += chunk.length;
    if (length > maximum) {
      throw Object.assign(new Error(`request exceeds ${maximum} bytes`), {
        reasonCode: "file_bytes_exceeded",
      });
    }
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function loadTypeScript() {
  const require = createRequire(import.meta.url);
  const target = process.env.TYPESCRIPT_PATH || "typescript";
  const loaded = require(target);
  if (
    !loaded ||
    typeof loaded.createSourceFile !== "function" ||
    typeof loaded.forEachChild !== "function" ||
    typeof loaded.version !== "string"
  ) {
    throw new Error("loaded module is not the TypeScript compiler API");
  }
  return loaded;
}

function scriptKind(ts, language) {
  return {
    javascript: ts.ScriptKind.JS,
    jsx: ts.ScriptKind.JSX,
    typescript: ts.ScriptKind.TS,
    tsx: ts.ScriptKind.TSX,
  }[language];
}

function fixedFileName(language) {
  return {
    javascript: "source.js",
    jsx: "source.jsx",
    typescript: "source.ts",
    tsx: "source.tsx",
  }[language];
}

function flattenDiagnostic(ts, diagnostic) {
  return ts.flattenDiagnosticMessageText(diagnostic.messageText, " ");
}

function diagnosticText(ts, sourceFile, diagnostic) {
  let location = "";
  if (typeof diagnostic.start === "number") {
    const point = sourceFile.getLineAndCharacterOfPosition(diagnostic.start);
    location = `@${point.line + 1}:${point.character + 1}`;
  }
  const code = Number.isInteger(diagnostic.code) ? `TS${diagnostic.code}` : "TS";
  return `${code}${location}:${flattenDiagnostic(ts, diagnostic)}`;
}

function stableText(value) {
  return String(value || "").replace(/\s+/gu, " ").trim();
}

function nodeText(node, sourceFile) {
  try {
    return stableText(node.getText(sourceFile));
  } catch {
    return "";
  }
}

function nameText(ts, node, sourceFile) {
  if (!node) return "";
  if (ts.isIdentifier(node) || ts.isPrivateIdentifier?.(node)) {
    return String(node.text || node.escapedText || "");
  }
  if (ts.isStringLiteralLike(node) || ts.isNumericLiteral(node)) {
    return String(node.text);
  }
  if (ts.isComputedPropertyName(node)) {
    return `[${nodeText(node.expression, sourceFile)}]`;
  }
  return nodeText(node, sourceFile);
}

function expressionName(ts, node, sourceFile) {
  if (!node) return "";
  if (ts.isIdentifier(node) || node.kind === ts.SyntaxKind.ThisKeyword) {
    return node.kind === ts.SyntaxKind.ThisKeyword ? "this" : String(node.text);
  }
  if (node.kind === ts.SyntaxKind.SuperKeyword) return "super";
  if (node.kind === ts.SyntaxKind.ImportKeyword) return "import";
  if (ts.isPropertyAccessExpression(node)) {
    const parent = expressionName(ts, node.expression, sourceFile);
    const child = nameText(ts, node.name, sourceFile);
    return parent ? `${parent}.${child}` : child;
  }
  if (ts.isElementAccessExpression(node)) {
    const parent = expressionName(ts, node.expression, sourceFile);
    const child = nodeText(node.argumentExpression, sourceFile);
    return `${parent || "<dynamic>"}[${child}]`;
  }
  if (
    ts.isParenthesizedExpression(node) ||
    ts.isNonNullExpression(node) ||
    ts.isAsExpression(node) ||
    ts.isTypeAssertionExpression(node)
  ) {
    return expressionName(ts, node.expression, sourceFile);
  }
  return "";
}

function semanticNode(ts, node) {
  const result = [node.kind];
  if (
    ts.isIdentifier(node) ||
    ts.isPrivateIdentifier?.(node) ||
    ts.isStringLiteralLike(node) ||
    ts.isNumericLiteral(node) ||
    ts.isRegularExpressionLiteral?.(node) ||
    ts.isJsxText?.(node)
  ) {
    result.push(String(node.text ?? node.escapedText ?? ""));
  }
  ts.forEachChild(node, (child) => {
    result.push(semanticNode(ts, child));
  });
  return result;
}

function semanticHash(ts, node) {
  const semantic = JSON.stringify(semanticNode(ts, node));
  return `sha256:${crypto
    .createHash("sha256")
    .update(`typescript-symbol-v1\0${ts.version}\0${semantic}`, "utf8")
    .digest("hex")}`;
}

function sourceLines(sourceFile, node) {
  try {
    const start = sourceFile.getLineAndCharacterOfPosition(
      node.getStart(sourceFile, false),
    );
    const endPosition = Math.max(node.getStart(sourceFile, false), node.getEnd() - 1);
    const end = sourceFile.getLineAndCharacterOfPosition(endPosition);
    return [start.line + 1, end.line + 1];
  } catch {
    return [0, 0];
  }
}

function isDefinition(ts, node) {
  return (
    ts.isClassDeclaration(node) ||
    ts.isClassExpression(node) ||
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isMethodSignature(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isGetAccessorDeclaration(node) ||
    ts.isSetAccessorDeclaration(node) ||
    ts.isInterfaceDeclaration(node) ||
    ts.isTypeAliasDeclaration(node) ||
    ts.isEnumDeclaration(node) ||
    ts.isModuleDeclaration(node)
  );
}

function definitionName(ts, node, sourceFile) {
  if (ts.isConstructorDeclaration(node)) return "constructor";
  const name = nameText(ts, node.name, sourceFile);
  if (name) return name;
  return hasModifier(ts, node, ts.SyntaxKind.DefaultKeyword) ? "default" : "";
}

function opensScope(ts, node) {
  return (
    isDefinition(ts, node) ||
    ts.isArrowFunction(node) ||
    (ts.isVariableDeclaration(node) && Boolean(node.initializer))
  );
}

function hasModifier(ts, node, kind) {
  return Boolean(node.modifiers?.some((modifier) => modifier.kind === kind));
}

function signatureText(ts, node, sourceFile, printer) {
  if (ts.isInterfaceDeclaration(node) || ts.isTypeAliasDeclaration(node)) {
    return stableText(printer.printNode(ts.EmitHint.Unspecified, node, sourceFile));
  }
  const start = node.getStart(sourceFile, false);
  const end = node.body ? node.body.getStart(sourceFile, false) : node.getEnd();
  return stableText(sourceFile.text.slice(start, end)).replace(/\s*\{$/u, "");
}

function importText(ts, node, sourceFile) {
  if (ts.isImportDeclaration(node)) {
    const moduleName = String(node.moduleSpecifier.text || "");
    const clause = node.importClause;
    if (!clause) return `import "${moduleName}"`;
    const parts = [];
    if (clause.isTypeOnly) parts.push("type");
    if (clause.name) parts.push(String(clause.name.text));
    const bindings = clause.namedBindings;
    if (bindings && ts.isNamespaceImport(bindings)) {
      parts.push(`* as ${bindings.name.text}`);
    } else if (bindings && ts.isNamedImports(bindings)) {
      const names = bindings.elements
        .map((item) => {
          const imported = item.propertyName ? `${item.propertyName.text} as ` : "";
          return `${item.isTypeOnly ? "type " : ""}${imported}${item.name.text}`;
        })
        .sort();
      parts.push(`{${names.join(",")}}`);
    }
    return `import ${parts.join(" ")} from "${moduleName}"`;
  }
  if (ts.isImportEqualsDeclaration(node)) {
    return stableText(node.getText(sourceFile));
  }
  if (ts.isExportDeclaration(node) && node.moduleSpecifier) {
    return `export-from "${String(node.moduleSpecifier.text || "")}"`;
  }
  return "";
}

function assignmentOperation(ts, kind) {
  const names = new Map([
    [ts.SyntaxKind.EqualsToken, "assign"],
    [ts.SyntaxKind.PlusEqualsToken, "augassign:Add"],
    [ts.SyntaxKind.MinusEqualsToken, "augassign:Subtract"],
    [ts.SyntaxKind.AsteriskEqualsToken, "augassign:Multiply"],
    [ts.SyntaxKind.SlashEqualsToken, "augassign:Divide"],
    [ts.SyntaxKind.PercentEqualsToken, "augassign:Modulo"],
    [ts.SyntaxKind.AsteriskAsteriskEqualsToken, "augassign:Power"],
    [ts.SyntaxKind.AmpersandEqualsToken, "augassign:BitAnd"],
    [ts.SyntaxKind.BarEqualsToken, "augassign:BitOr"],
    [ts.SyntaxKind.CaretEqualsToken, "augassign:BitXor"],
    [ts.SyntaxKind.LessThanLessThanEqualsToken, "augassign:LeftShift"],
    [ts.SyntaxKind.GreaterThanGreaterThanEqualsToken, "augassign:RightShift"],
    [
      ts.SyntaxKind.GreaterThanGreaterThanGreaterThanEqualsToken,
      "augassign:UnsignedRightShift",
    ],
    [ts.SyntaxKind.AmpersandAmpersandEqualsToken, "augassign:LogicalAnd"],
    [ts.SyntaxKind.BarBarEqualsToken, "augassign:LogicalOr"],
    [ts.SyntaxKind.QuestionQuestionEqualsToken, "augassign:Nullish"],
  ]);
  return names.get(kind) || "";
}

function extractFacts(ts, sourceFile) {
  const qualifiedSymbols = new Set();
  const imports = new Set();
  const calls = new Set();
  const stateTransitions = new Set();
  const interfaces = new Set();
  const symbolHashParts = {};
  const symbolLines = {};
  const scope = [];
  const printer = ts.createPrinter({
    newLine: ts.NewLineKind.LineFeed,
    removeComments: true,
  });

  const owner = () => (scope.length ? scope.join(".") : "<module>");
  const qualify = (name) => [...scope, name].filter(Boolean).join(".");

  function addDefinition(node, name) {
    if (!name) return "";
    const qualified = qualify(name);
    qualifiedSymbols.add(qualified);
    (symbolHashParts[qualified] ||= []).push(semanticHash(ts, node));
    const lines = sourceLines(sourceFile, node);
    const existingLines = symbolLines[qualified];
    symbolLines[qualified] = existingLines
      ? [Math.min(existingLines[0], lines[0]), Math.max(existingLines[1], lines[1])]
      : lines;
    if (
      ts.isInterfaceDeclaration(node) ||
      ts.isTypeAliasDeclaration(node) ||
      ts.isMethodSignature(node)
    ) {
      interfaces.add(`${qualified}:${signatureText(ts, node, sourceFile, printer)}`);
    } else if (
      ts.isFunctionDeclaration(node) ||
      ts.isMethodDeclaration(node) ||
      ts.isConstructorDeclaration(node) ||
      ts.isGetAccessorDeclaration(node) ||
      ts.isSetAccessorDeclaration(node)
    ) {
      const isPrivate =
        hasModifier(ts, node, ts.SyntaxKind.PrivateKeyword) ||
        String(name).startsWith("#");
      if (!isPrivate) {
        interfaces.add(`${qualified}:${signatureText(ts, node, sourceFile, printer)}`);
      }
    }
    return name;
  }

  function visit(node) {
    let scopeName = "";
    if (isDefinition(ts, node)) {
      scopeName = addDefinition(node, definitionName(ts, node, sourceFile));
    } else if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      (scope.length === 0 ||
        ts.isArrowFunction(node.initializer) ||
        ts.isFunctionExpression(node.initializer))
    ) {
      scopeName = addDefinition(node, String(node.name.text));
    }

    const imported = importText(ts, node, sourceFile);
    if (imported) imports.add(imported);

    if (ts.isCallExpression(node) || ts.isNewExpression(node)) {
      const callee = expressionName(ts, node.expression, sourceFile) || "<dynamic>";
      calls.add(`${owner()}->${callee}`);
      if (
        ts.isCallExpression(node) &&
        callee === "require" &&
        node.arguments.length === 1 &&
        ts.isStringLiteralLike(node.arguments[0])
      ) {
        imports.add(`require:"${node.arguments[0].text}"`);
      }
      if (
        ts.isCallExpression(node) &&
        callee === "import" &&
        node.arguments.length === 1 &&
        ts.isStringLiteralLike(node.arguments[0])
      ) {
        imports.add(`import:"${node.arguments[0].text}"`);
      }
      const effectName = callee.split(".").at(-1).toLowerCase();
      if (
        new Set([
          "transition",
          "transitionto",
          "setstate",
          "setstatus",
          "changestate",
          "updatestate",
        ]).has(effectName)
      ) {
        const args = (node.arguments || [])
          .map((argument) => nodeText(argument, sourceFile))
          .join(",");
        stateTransitions.add(`${owner()}:${callee}:call(${args})`);
      }
    }

    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.initializer
    ) {
      stateTransitions.add(
        `${owner()}:${node.name.text}:declare:${nodeText(node.initializer, sourceFile)}`,
      );
    } else if (
      (ts.isPropertyDeclaration(node) || ts.isPropertySignature(node)) &&
      node.initializer
    ) {
      stateTransitions.add(
        `${owner()}:${nameText(ts, node.name, sourceFile)}:initialize:${nodeText(
          node.initializer,
          sourceFile,
        )}`,
      );
    } else if (ts.isBinaryExpression(node)) {
      const operation = assignmentOperation(ts, node.operatorToken.kind);
      if (operation) {
        const target = nodeText(node.left, sourceFile);
        const value = nodeText(node.right, sourceFile);
        if (target) stateTransitions.add(`${owner()}:${target}:${operation}:${value}`);
      }
    } else if (
      (ts.isPrefixUnaryExpression(node) || ts.isPostfixUnaryExpression(node)) &&
      (node.operator === ts.SyntaxKind.PlusPlusToken ||
        node.operator === ts.SyntaxKind.MinusMinusToken)
    ) {
      const operation =
        node.operator === ts.SyntaxKind.PlusPlusToken ? "increment" : "decrement";
      stateTransitions.add(
        `${owner()}:${nodeText(node.operand, sourceFile)}:${operation}`,
      );
    } else if (
      ts.isDeleteExpression?.(node) ||
      (ts.isPrefixUnaryExpression(node) &&
        node.operator === ts.SyntaxKind.DeleteKeyword)
    ) {
      stateTransitions.add(
        `${owner()}:${nodeText(node.expression || node.operand, sourceFile)}:delete`,
      );
    }

    if (scopeName && opensScope(ts, node)) scope.push(scopeName);
    ts.forEachChild(node, visit);
    if (scopeName && opensScope(ts, node)) scope.pop();
  }

  visit(sourceFile);
  const symbolHashes = Object.fromEntries(
    Object.entries(symbolHashParts).map(([name, values]) => {
      const ordered = [...values].sort();
      if (ordered.length === 1) return [name, ordered[0]];
      return [
        name,
        `sha256:${crypto
          .createHash("sha256")
          .update(`typescript-overloads-v1\0${ts.version}\0${ordered.join("\0")}`)
          .digest("hex")}`,
      ];
    }),
  );
  const byKey = ([left], [right]) => (left < right ? -1 : left > right ? 1 : 0);
  return {
    qualified_symbols: [...qualifiedSymbols].sort(),
    imports: [...imports].sort(),
    calls: [...calls].sort(),
    state_transitions: [...stateTransitions].sort(),
    interfaces: [...interfaces].sort(),
    symbol_hashes: Object.fromEntries(
      Object.entries(symbolHashes).sort(byKey),
    ),
    symbol_lines: Object.fromEntries(
      Object.entries(symbolLines).sort(byKey),
    ),
  };
}

let request;
try {
  request = await readRequest();
} catch (error) {
  failure(error.reasonCode || "protocol_error", "invalid bounded JSON request");
  process.exitCode = 2;
}

if (request !== undefined) {
  if (
    !request ||
    request.protocol_version !== PROTOCOL_VERSION ||
    typeof request.source !== "string" ||
    typeof request.language !== "string" ||
    typeof request.source_sha256 !== "string"
  ) {
    failure("protocol_error", "request does not match extractor protocol");
  } else if (!LANGUAGES.has(request.language)) {
    failure("unsupported_language", `unsupported language ${request.language}`);
  } else {
    let ts;
    try {
      ts = loadTypeScript();
    } catch {
      failure("compiler_unavailable", "the local TypeScript compiler API is unavailable");
    }
    if (ts) {
      try {
        const sourceFile = ts.createSourceFile(
          fixedFileName(request.language),
          request.source,
          ts.ScriptTarget.Latest,
          true,
          scriptKind(ts, request.language),
        );
        const diagnostics = [...(sourceFile.parseDiagnostics || [])].sort(
          (left, right) =>
            (left.start ?? -1) - (right.start ?? -1) ||
            (left.code ?? -1) - (right.code ?? -1),
        );
        const parseError = diagnostics.length
          ? `typescript_parse_error:${diagnostics
              .slice(0, 20)
              .map((item) => diagnosticText(ts, sourceFile, item))
              .join("|")}`
          : "";
        emit({
          protocol_version: PROTOCOL_VERSION,
          ok: true,
          producer: "typescript-compiler-api",
          producer_version: PRODUCER_VERSION,
          compiler: { name: "typescript", version: ts.version },
          language: request.language,
          source_sha256: request.source_sha256,
          parse_error: parseError,
          facts: parseError
            ? {
                qualified_symbols: [],
                imports: [],
                calls: [],
                state_transitions: [],
                interfaces: [],
                symbol_hashes: {},
                symbol_lines: {},
              }
            : extractFacts(ts, sourceFile),
        });
      } catch {
        failure("process_failed", "TypeScript compiler AST traversal failed");
      }
    }
  }
}
