export type OperationShape = {
  method: string;
  path: string;
  pathParams: Record<string, unknown>;
  query: Record<string, unknown>;
  headerParams?: Record<string, unknown>;
  body: unknown;
  bodyRequired?: boolean;
  ifMatchRequired?: boolean;
  contentType: string;
  response: unknown;
};

export type OperationMap = Record<string, OperationShape>;

type RequiredKeys<Value> = {
  [Key in keyof Value]-?: Record<string, never> extends Pick<Value, Key> ? never : Key;
}[keyof Value];

type HasNoFields<Value> = Value extends Record<string, never> ? true : false;

type HeaderParams<Operation extends OperationShape> = Operation extends {
  headerParams: infer Value extends Record<string, unknown>;
} ? Value : Record<string, never>;

type PathArguments<Operation extends OperationShape> = HasNoFields<Operation["pathParams"]> extends true
  ? {path?: never}
  : {path: Operation["pathParams"]};

type QueryArguments<Operation extends OperationShape> = HasNoFields<Operation["query"]> extends true
  ? {query?: never}
  : RequiredKeys<Operation["query"]> extends never
  ? {query?: Operation["query"]}
  : {query: Operation["query"]};

type HeaderArguments<Operation extends OperationShape> = HasNoFields<HeaderParams<Operation>> extends true
  ? {header?: never}
  : RequiredKeys<HeaderParams<Operation>> extends never
  ? {header?: HeaderParams<Operation>}
  : {header: HeaderParams<Operation>};

type BodyValue<Operation extends OperationShape> = Operation["body"];

type BodyArguments<Operation extends OperationShape> = Operation["body"] extends undefined
  ? {body?: never}
  : Operation extends {bodyRequired: true}
    ? {body: BodyValue<Operation>}
    : {body?: BodyValue<Operation>};

type IfMatchArguments<Operation extends OperationShape> = Operation extends {ifMatchRequired: true}
  ? {ifMatch: string}
  : {ifMatch?: string};

export type RequestArgs<Operation extends OperationShape> = PathArguments<Operation>
  & QueryArguments<Operation>
  & HeaderArguments<Operation>
  & BodyArguments<Operation>
  & IfMatchArguments<Operation>
  & {
    headers?: HeadersInit;
    signal?: AbortSignal;
  };

type RequestArguments<Operation extends OperationShape> = RequiredKeys<RequestArgs<Operation>> extends never
  ? [args?: RequestArgs<Operation>]
  : [args: RequestArgs<Operation>];

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export class ApiProblem extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;
  readonly details: unknown;
  readonly retryAfter: string | null;

  constructor(
    status: number,
    payload: unknown,
    requestId: string | null,
    retryAfter: string | null = null,
  ) {
    const record = recordValue(payload);
    const error = recordValue(record.error);
    const problem = Object.keys(error).length ? error : record;
    const message = stringValue(problem.message) ?? `HTTP ${status}`;
    super(message);
    this.name = "ApiProblem";
    this.status = status;
    this.code = stringValue(problem.code) ?? "HTTP_ERROR";
    this.requestId = stringValue(requestId)
      ?? stringValue(record.request_id)
      ?? stringValue(problem.request_id);
    this.details = problem.details ?? null;
    this.retryAfter = stringValue(retryAfter);
  }
}

export class ApiContractError extends Error {
  readonly requestId: string | null;

  constructor(message: string, requestId: string | null) {
    super(message);
    this.name = "ApiContractError";
    this.requestId = requestId;
  }
}

export type OperationDescriptor = {
  method: string;
  path: string;
  contentType: string;
};

function appendPrimitive(
  target: URLSearchParams,
  name: string,
  raw: unknown,
): void {
  if (raw === undefined || raw === null) return;
  if (typeof raw === "number" && !Number.isFinite(raw)) {
    throw new TypeError(`Query parameter ${name} must be a finite number`);
  }
  if (!["string", "number", "boolean"].includes(typeof raw)) {
    throw new TypeError(`Query parameter ${name} must be a primitive or primitive array`);
  }
  target.append(name, String(raw));
}

function appendQuery(url: URL, query: Record<string, unknown>): void {
  for (const [name, raw] of Object.entries(query)) {
    if (Array.isArray(raw)) {
      for (const value of raw) appendPrimitive(url.searchParams, name, value);
    } else {
      appendPrimitive(url.searchParams, name, raw);
    }
  }
}

function appendFormValue(form: FormData, name: string, raw: unknown): void {
  if (raw === undefined || raw === null) return;
  if (Array.isArray(raw)) {
    for (const value of raw) appendFormValue(form, name, value);
    return;
  }
  if (raw instanceof Blob) {
    form.append(name, raw);
    return;
  }
  if (typeof raw === "object") {
    form.append(name, JSON.stringify(raw));
    return;
  }
  form.append(name, String(raw));
}

function multipartBody(value: unknown): FormData {
  if (value instanceof FormData) return value;
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("Multipart request body must be FormData or an object");
  }
  const form = new FormData();
  for (const [name, raw] of Object.entries(value)) appendFormValue(form, name, raw);
  return form;
}

function urlEncodedBody(value: unknown): URLSearchParams {
  if (value instanceof URLSearchParams) return value;
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("Form request body must be URLSearchParams or an object");
  }
  const form = new URLSearchParams();
  for (const [name, raw] of Object.entries(value)) {
    if (Array.isArray(raw)) {
      for (const item of raw) appendPrimitive(form, name, item);
    } else {
      appendPrimitive(form, name, raw);
    }
  }
  return form;
}

function requestBody(
  descriptor: OperationDescriptor,
  value: unknown,
  headers: Headers,
): BodyInit | undefined {
  if (value === undefined) return undefined;
  if (descriptor.contentType === "multipart/form-data") {
    headers.delete("Content-Type");
    return multipartBody(value);
  }
  if (descriptor.contentType === "application/x-www-form-urlencoded") {
    if (!headers.has("Content-Type")) {
      headers.set("Content-Type", "application/x-www-form-urlencoded;charset=UTF-8");
    }
    return urlEncodedBody(value);
  }
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", descriptor.contentType || "application/json");
  }
  return headers.get("Content-Type")?.includes("json")
    ? JSON.stringify(value)
    : String(value);
}

async function responsePayload(
  response: Response,
  descriptor: OperationDescriptor,
): Promise<unknown> {
  if (response.status === 204 || response.status === 205 || descriptor.method === "HEAD") {
    return undefined;
  }
  const responseType = response.headers.get("content-type") ?? "";
  const textual = !responseType
    || responseType.includes("json")
    || responseType.startsWith("text/")
    || responseType.includes("xml")
    || responseType.includes("javascript")
    || responseType.includes("x-www-form-urlencoded");
  if (!textual) return response.blob();
  const text = await response.text();
  if (!text) return undefined;
  if (!responseType.includes("json")) return text;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    if (!response.ok) return undefined;
    throw new ApiContractError(
      "API returned an invalid JSON response",
      response.headers.get("x-request-id"),
    );
  }
}

export function createOpenApiClient<
  Operations extends {[Key in keyof Operations]: OperationShape},
>(
  baseUrl: string,
  descriptors: { [Key in keyof Operations]: OperationDescriptor },
) {
  return async function request<Key extends keyof Operations>(
    operationId: Key,
    ...requestArguments: RequestArguments<Operations[Key]>
  ): Promise<Operations[Key]["response"]> {
    const args = (requestArguments[0] ?? {}) as RequestArgs<Operations[Key]>;
    const descriptor = descriptors[operationId];
    let path = descriptor.path;
    for (const [name, value] of Object.entries(args.path ?? {})) {
      if (value === undefined || value === null) {
        throw new TypeError(`Missing path parameter ${name} for ${String(operationId)}`);
      }
      path = path.replace(`{${name}}`, encodeURIComponent(String(value)));
    }
    if (/\{[^}]+\}/.test(path)) {
      throw new TypeError(`Missing path parameter for ${String(operationId)}`);
    }
    const url = new URL(path, baseUrl);
    appendQuery(url, args.query ?? {});
    const headers = new Headers(args.headers);
    if (!headers.has("Accept")) headers.set("Accept", "application/json");
    for (const [name, value] of Object.entries(args.header ?? {})) {
      if (value !== undefined && value !== null) headers.set(name, String(value));
    }
    if (args.ifMatch !== undefined) {
      if (!args.ifMatch.trim()) throw new TypeError("If-Match cannot be empty");
      headers.set("If-Match", args.ifMatch);
    }
    const body = requestBody(descriptor, args.body, headers);
    if (body !== undefined && ["GET", "HEAD"].includes(descriptor.method)) {
      throw new TypeError(`${descriptor.method} requests cannot include a body`);
    }
    const requestInit: RequestInit = {
      method: descriptor.method,
      headers,
      credentials: "include",
    };
    if (body !== undefined) requestInit.body = body;
    if (args.signal !== undefined) requestInit.signal = args.signal;
    const response = await fetch(url, requestInit);
    const payload = await responsePayload(response, descriptor);
    if (!response.ok) {
      throw new ApiProblem(
        response.status,
        payload,
        response.headers.get("x-request-id"),
        response.headers.get("retry-after"),
      );
    }
    return payload as Operations[Key]["response"];
  };
}
