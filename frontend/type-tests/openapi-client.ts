import {createOpenApiClient} from "../src/runtime/openapi-client";

interface ContractOperations {
  uploadVersion: {
    method: "POST";
    path: "/documents/{document_id}/versions";
    pathParams: {document_id: string};
    query: {validate?: boolean};
    headerParams: {"Idempotency-Key": string};
    body: {file: File; parser_config?: string};
    bodyRequired: true;
    ifMatchRequired: true;
    contentType: "multipart/form-data";
    response: {job_id: string};
  };
  listDocuments: {
    method: "GET";
    path: "/documents";
    pathParams: Record<string, never>;
    query: {kb_id: string; cursor?: string};
    headerParams: Record<string, never>;
    body: undefined;
    bodyRequired: false;
    ifMatchRequired: false;
    contentType: "";
    response: {items: Array<{id: string}>};
  };
  health: {
    method: "GET";
    path: "/health";
    pathParams: Record<string, never>;
    query: Record<string, never>;
    headerParams: Record<string, never>;
    body: undefined;
    bodyRequired: false;
    ifMatchRequired: false;
    contentType: "";
    response: undefined;
  };
}

const client = createOpenApiClient<ContractOperations>("http://127.0.0.1:8000", {
  uploadVersion: {
    method: "POST",
    path: "/documents/{document_id}/versions",
    contentType: "multipart/form-data",
  },
  listDocuments: {method: "GET", path: "/documents", contentType: ""},
  health: {method: "GET", path: "/health", contentType: ""},
});

void client("uploadVersion", {
  path: {document_id: "doc_1"},
  header: {"Idempotency-Key": "request_1"},
  body: {file: new File(["content"], "manual.pdf"), parser_config: "{}"},
  ifMatch: '"revision-1"',
  signal: new AbortController().signal,
});
void client("listDocuments", {query: {kb_id: "kb_1"}});
void client("health");

// @ts-expect-error required multipart body cannot be omitted
void client("uploadVersion", {
  path: {document_id: "doc_1"},
  header: {"Idempotency-Key": "request_1"},
  ifMatch: '"revision-1"',
});
// @ts-expect-error required If-Match cannot be omitted
void client("uploadVersion", {
  path: {document_id: "doc_1"},
  header: {"Idempotency-Key": "request_1"},
  body: {file: new File([], "manual.pdf")},
});
// @ts-expect-error required path parameters cannot be omitted
void client("uploadVersion", {
  header: {"Idempotency-Key": "request_1"},
  body: {file: new File([], "manual.pdf")},
  ifMatch: '"revision-1"',
});
// @ts-expect-error required OpenAPI header parameters cannot be omitted
void client("uploadVersion", {
  path: {document_id: "doc_1"},
  body: {file: new File([], "manual.pdf")},
  ifMatch: '"revision-1"',
});
// @ts-expect-error kb_id is a required query parameter
void client("listDocuments");
void client("uploadVersion", {
  path: {document_id: "doc_1"},
  header: {"Idempotency-Key": "request_1"},
  // @ts-expect-error multipart binary fields retain the browser File contract
  body: {file: new Blob()},
  ifMatch: '"revision-1"',
});
