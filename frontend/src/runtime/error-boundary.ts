import {ApiContractError, ApiProblem} from "./openapi-client";

export type WorkbenchError = Readonly<{
  title: string;
  message: string;
  requestId: string | null;
  retryable: boolean;
  kind: "cancelled" | "unauthenticated" | "forbidden" | "conflict"
    | "not-found" | "validation" | "rate-limited" | "network"
    | "server" | "contract" | "unknown";
}>;

function workbenchError(
  kind: WorkbenchError["kind"],
  title: string,
  message: string,
  options: {requestId?: string | null; retryable?: boolean} = {},
): WorkbenchError {
  return {
    title,
    message,
    requestId: options.requestId ?? null,
    retryable: options.retryable ?? false,
    kind,
  };
}

export function toWorkbenchError(error: unknown): WorkbenchError {
  if (error instanceof DOMException && error.name === "AbortError") {
    return workbenchError(
      "cancelled",
      "请求已取消",
      "资源上下文已变化，旧响应不会覆盖当前页面。",
    );
  }
  if (error instanceof ApiContractError) {
    return workbenchError(
      "contract",
      "服务响应格式异常",
      "服务返回的数据不符合当前页面契约，请稍后重试并保留请求 ID。",
      {requestId: error.requestId, retryable: true},
    );
  }
  if (error instanceof ApiProblem) {
    const request = {requestId: error.requestId};
    if (error.status === 401) {
      return workbenchError(
        "unauthenticated",
        "登录已失效",
        "请重新登录后继续操作。",
        request,
      );
    }
    if (error.status === 403) {
      return workbenchError(
        "forbidden",
        "没有访问权限",
        "当前账号无权访问该租户或资源。",
        request,
      );
    }
    if (error.status === 404) {
      return workbenchError(
        "not-found",
        "资源不存在或已不可访问",
        "资源可能已删除、已切换租户，或不在当前授权范围内。",
        request,
      );
    }
    if (error.status === 409 || error.status === 412 || error.status === 428) {
      return workbenchError(
        "conflict",
        "资源已发生变化",
        "请刷新当前资源后重新确认操作，旧版本不会覆盖新版本。",
        {...request, retryable: true},
      );
    }
    if (error.status === 400 || error.status === 422) {
      return workbenchError(
        "validation",
        "请求内容不符合要求",
        "请检查页面输入和资源状态后重试。",
        request,
      );
    }
    if (error.status === 429) {
      return workbenchError(
        "rate-limited",
        "请求过于频繁",
        "请稍后重试；页面不会自动重复提交写操作。",
        {...request, retryable: true},
      );
    }
    if (error.status >= 500) {
      return workbenchError(
        "server",
        "服务暂时不可用",
        "服务未能完成请求，请稍后重试并保留请求 ID。",
        {...request, retryable: true},
      );
    }
    return workbenchError(
      "unknown",
      "服务请求未完成",
      "请求未能完成，请检查当前资源状态并保留请求 ID。",
      request,
    );
  }
  if (error instanceof TypeError) {
    return workbenchError(
      "network",
      "请求无法发送",
      "请检查网络连接和页面输入后重试。",
      {retryable: true},
    );
  }
  return workbenchError(
    "unknown",
    "发生未知错误",
    "页面未能完成操作；为避免泄露内部信息，详细异常仅记录在受控日志中。",
  );
}
