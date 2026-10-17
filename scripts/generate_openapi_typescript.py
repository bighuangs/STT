"""Generate dependency-free TypeScript clients from the two FastAPI applications.

The generated files are intentionally checked in.  Browser pages can consume the
compiled JavaScript runtime immediately, while a future component migration gets
the same operation and schema contract without maintaining a second DTO model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "frontend" / "src" / "generated"
OPENAPI_DIR = ROOT / "frontend" / "openapi"
HTTP_METHODS = ("get", "post", "put", "patch", "delete")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _identifier(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_$]", "_", value)
    if not candidate or candidate[0].isdigit():
        candidate = f"_{candidate}"
    return candidate


def _literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _union(types: list[str]) -> str:
    unique = list(dict.fromkeys(types))
    return " | ".join(unique) or "never"


def _schema_type(schema: Mapping[str, Any] | None) -> str:
    if not schema:
        return "unknown"
    reference = schema.get("$ref")
    if isinstance(reference, str):
        return _identifier(reference.rsplit("/", 1)[-1])
    if "const" in schema:
        return _literal(schema["const"])
    if isinstance(schema.get("enum"), list):
        return _union([_literal(item) for item in schema["enum"]])
    if "anyOf" in schema:
        return _union([_schema_type(item) for item in schema["anyOf"]])
    if "oneOf" in schema:
        return _union([_schema_type(item) for item in schema["oneOf"]])
    if "allOf" in schema:
        return " & ".join(_schema_type(item) for item in schema["allOf"])
    schema_type = schema.get("type")
    if schema_type == "string" and (
        schema.get("format") == "binary"
        or schema.get("contentMediaType") == "application/octet-stream"
    ):
        return "File"
    if schema_type == "array":
        return f"Array<{_schema_type(schema.get('items'))}>"
    if schema_type == "object" or "properties" in schema:
        inline = _inline_object(schema)
        if isinstance(schema.get("additionalProperties"), Mapping):
            # An index signature supports recursive JSON aliases; wrapping the
            # same self-reference in the built-in Record<> triggers TS2456.
            additional = f"{{[key: string]: {_schema_type(schema['additionalProperties'])}}}"
            return additional if not schema.get("properties") else f"{inline} & {additional}"
        if schema.get("additionalProperties") is True:
            return "Record<string, unknown>" if not schema.get("properties") else f"{inline} & Record<string, unknown>"
        return inline
    return {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "null": "null",
    }.get(str(schema_type), "unknown") + (" | null" if schema.get("nullable") else "")


def _inline_object(schema: Mapping[str, Any]) -> str:
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if not properties:
        return "Record<string, unknown>"
    fields = []
    # The checked-in OpenAPI snapshots are JSON with sorted object keys.  Sort
    # here as well so generating from a live FastAPI dict and checking from the
    # frozen JSON document produce byte-identical TypeScript.
    for name, child in sorted(properties.items()):
        optional = "" if name in required else "?"
        fields.append(f"{_literal(name)}{optional}: {_schema_type(child)}")
    return "{ " + "; ".join(fields) + " }"


def _resolve_component(
    document: Mapping[str, Any],
    value: Mapping[str, Any],
    component_kind: str,
) -> Mapping[str, Any]:
    reference = value.get("$ref")
    prefix = f"#/components/{component_kind}/"
    if not isinstance(reference, str) or not reference.startswith(prefix):
        return value
    name = reference.removeprefix(prefix)
    resolved = document.get("components", {}).get(component_kind, {}).get(name)
    if not isinstance(resolved, Mapping):
        raise ValueError(f"unresolved OpenAPI reference: {reference}")
    return resolved


def _parameter_schema(parameter: Mapping[str, Any]) -> Mapping[str, Any] | None:
    schema = parameter.get("schema")
    if isinstance(schema, Mapping):
        return schema
    for media in (parameter.get("content") or {}).values():
        if isinstance(media, Mapping) and isinstance(media.get("schema"), Mapping):
            return media["schema"]
    return None


def _object_type(fields: list[str]) -> str:
    return "{ " + "; ".join(fields) + " }" if fields else "Record<string, never>"


def _parameters(
    document: Mapping[str, Any],
    operation: Mapping[str, Any],
    path_item: Mapping[str, Any],
) -> tuple[str, str, str, bool]:
    path_fields: list[str] = []
    query_fields: list[str] = []
    header_fields: list[str] = []
    raw_required_headers = operation.get("x-stt-required-headers") or []
    if not isinstance(raw_required_headers, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_required_headers
    ):
        raise ValueError("x-stt-required-headers must be a list of non-empty names")
    required_headers = {item.strip().lower() for item in raw_required_headers}
    parameters: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in [*(path_item.get("parameters") or []), *(operation.get("parameters") or [])]:
        if not isinstance(raw, Mapping):
            continue
        parameter = _resolve_component(document, raw, "parameters")
        location = str(parameter.get("in") or "")
        name = str(parameter.get("name") or "")
        parameters[(location, name.lower() if location == "header" else name)] = parameter
    declared_headers: set[str] = set()
    if_match_required = "if-match" in required_headers
    for parameter in parameters.values():
        parameter_name = str(parameter.get("name"))
        header_name = parameter_name.lower()
        required = bool(parameter.get("required")) or (
            parameter.get("in") == "header" and header_name in required_headers
        )
        field = _literal(parameter_name)
        optional = "" if required else "?"
        declaration = f"{field}{optional}: {_schema_type(_parameter_schema(parameter))}"
        if parameter.get("in") == "path":
            path_fields.append(declaration)
        elif parameter.get("in") == "query":
            query_fields.append(declaration)
        elif parameter.get("in") == "header":
            declared_headers.add(header_name)
            if header_name == "if-match":
                if_match_required = required
            else:
                header_fields.append(declaration)
    missing_required_headers = required_headers - declared_headers
    if missing_required_headers:
        names = ", ".join(sorted(missing_required_headers))
        raise ValueError(f"required OpenAPI header parameters are undeclared: {names}")
    return (
        _object_type(path_fields),
        _object_type(query_fields),
        _object_type(header_fields),
        if_match_required,
    )


def _request_type(
    document: Mapping[str, Any], operation: Mapping[str, Any]
) -> tuple[str, str, bool]:
    raw_request_body = operation.get("requestBody") or {}
    request_body = (
        _resolve_component(document, raw_request_body, "requestBodies")
        if isinstance(raw_request_body, Mapping)
        else {}
    )
    content = request_body.get("content") or {}
    for media_type in ("application/json", "multipart/form-data", "application/x-www-form-urlencoded"):
        if media_type in content:
            return (
                _schema_type(content[media_type].get("schema")),
                media_type,
                bool(request_body.get("required")),
            )
    return "undefined", "", False


def _response_type(document: Mapping[str, Any], operation: Mapping[str, Any]) -> str:
    responses = operation.get("responses") or {}
    success_statuses = sorted(
        (str(status) for status in responses if str(status).startswith("2")),
        key=lambda value: (value == "default", value),
    )
    for status in [*success_statuses, "default"]:
        raw_response = responses.get(status)
        response = (
            _resolve_component(document, raw_response, "responses")
            if isinstance(raw_response, Mapping)
            else None
        )
        if not isinstance(response, Mapping):
            continue
        if status in {"204", "205"}:
            return "undefined"
        content = response.get("content") or {}
        for media_type in ("application/json", "text/event-stream", "text/plain"):
            if media_type in content:
                if media_type == "text/event-stream":
                    return "string"
                return _schema_type(content[media_type].get("schema"))
        for media_type, media in content.items():
            if not isinstance(media, Mapping):
                continue
            media_schema = media.get("schema")
            if (
                str(media_type).startswith(("image/", "audio/", "video/"))
                or media_type in {"application/octet-stream", "application/pdf"}
                or (
                    isinstance(media_schema, Mapping)
                    and media_schema.get("type") == "string"
                    and media_schema.get("format") == "binary"
                )
            ):
                return "Blob"
            if str(media_type).startswith("text/"):
                return "string"
            return _schema_type(media_schema)
        return "undefined"
    return "unknown"


def _render_schema(name: str, schema: Mapping[str, Any]) -> str:
    identifier = _identifier(name)
    if schema.get("type") == "object" or "properties" in schema:
        return f"export type {identifier} = {_inline_object(schema)};"
    return f"export type {identifier} = {_schema_type(schema)};"


def generate(document: Mapping[str, Any], service_name: str) -> str:
    schema_lines = [
        _render_schema(name, schema)
        for name, schema in sorted((document.get("components", {}).get("schemas") or {}).items())
    ]
    operations: list[tuple[str, str, str, str, str]] = []
    seen: set[str] = set()
    for path, path_item in sorted((document.get("paths") or {}).items()):
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, Mapping):
                continue
            operation_id = _identifier(str(operation.get("operationId") or f"{method}_{path}"))
            if operation_id in seen:
                raise ValueError(f"duplicate operationId: {operation_id}")
            seen.add(operation_id)
            path_type, query_type, header_type, if_match_required = _parameters(
                document, operation, path_item
            )
            body_type, content_type, body_required = _request_type(document, operation)
            response_type = _response_type(document, operation)
            definition = (
                "{ "
                f"method: {_literal(method.upper())}; path: {_literal(path)}; "
                f"pathParams: {path_type}; query: {query_type}; "
                f"headerParams: {header_type}; body: {body_type}; "
                f"bodyRequired: {str(body_required).lower()}; "
                f"ifMatchRequired: {str(if_match_required).lower()}; "
                f"contentType: {_literal(content_type)}; response: {response_type}"
                " }"
            )
            operations.append((operation_id, definition, method.upper(), path, content_type))
    operation_lines = [f"  {name}: {definition};" for name, definition, _, _, _ in operations]
    descriptor_lines = [
        f"  {name}: {{ method: {_literal(method)}, path: {_literal(path)}, "
        f"contentType: {_literal(content_type)} }},"
        for name, _, method, path, content_type in operations
    ]
    return "\n".join(
        [
            "/* eslint-disable */",
            "// Generated by scripts/generate_openapi_typescript.py. DO NOT EDIT.",
            'import {createOpenApiClient} from "../runtime/openapi-client";',
            f"export const serviceName = {_literal(service_name)} as const;",
            "",
            *schema_lines,
            "",
            "export interface Operations {",
            *operation_lines,
            "}",
            "",
            "export const operationDescriptors = {",
            *descriptor_lines,
            "} as const;",
            "",
            "export const createClient = (baseUrl: string) =>",
            "  createOpenApiClient<Operations>(baseUrl, operationDescriptors);",
            "",
        ]
    )


def _load_apps() -> dict[str, Any]:
    from app.api.http.import_server import app as import_app
    from app.api.http.query_server import app as query_app

    return {"import-api": import_app, "query-api": query_app}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail when generated files differ")
    parser.add_argument(
        "--check-apps",
        action="store_true",
        help="also load both FastAPI applications and compare their live OpenAPI documents",
    )
    parser.add_argument(
        "--from-snapshots",
        action="store_true",
        help="regenerate TypeScript from the checked-in OpenAPI JSON without loading apps",
    )
    args = parser.parse_args()
    if args.check_apps and args.from_snapshots:
        parser.error("--check-apps and --from-snapshots cannot be combined")
    if args.check_apps and not args.check:
        parser.error("--check-apps requires --check")
    if not args.check:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        OPENAPI_DIR.mkdir(parents=True, exist_ok=True)
    changed: list[Path] = []
    if (args.check and not args.check_apps) or args.from_snapshots:
        missing = [
            OPENAPI_DIR / f"{name}.json"
            for name in ("import-api", "query-api")
            if not (OPENAPI_DIR / f"{name}.json").exists()
        ]
        if missing:
            names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
            raise SystemExit(
                "OpenAPI snapshots are missing: "
                f"{names}. Run scripts/generate_openapi_typescript.py first."
            )
        documents = {
            name: json.loads((OPENAPI_DIR / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("import-api", "query-api")
        }
    else:
        documents = {name: app.openapi() for name, app in _load_apps().items()}
    for service_name, document in documents.items():
        spec_target = OPENAPI_DIR / f"{service_name}.json"
        expected_spec = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        current_spec = spec_target.read_text(encoding="utf-8") if spec_target.exists() else None
        if current_spec != expected_spec:
            changed.append(spec_target)
            if not args.check:
                spec_target.write_text(expected_spec, encoding="utf-8")
        target = GENERATED_DIR / f"{service_name}.ts"
        expected = generate(document, service_name)
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != expected:
            changed.append(target)
            if not args.check:
                target.write_text(expected, encoding="utf-8")
    if args.check and changed:
        names = ", ".join(str(path.relative_to(ROOT)) for path in changed)
        raise SystemExit(f"OpenAPI TypeScript clients are stale: {names}")


if __name__ == "__main__":
    main()
