export type ResourceScope = Readonly<{
  tenantId: string;
  userId: string;
  kbId?: string;
  resourceType?: string;
  resourceId?: string;
}>;

export type QueryKeyValue = null | boolean | number | string
  | readonly QueryKeyValue[]
  | Readonly<{[key: string]: QueryKeyValue}>;

type CacheEntry<Value> = {
  hasValue: boolean;
  value?: Value;
  promise?: Promise<Value>;
  controller?: AbortController;
  expiresAt: number;
  generation: number;
  scopeFingerprint: string;
  queryParts: readonly string[];
};

function normalizedKey(value: QueryKeyValue): QueryKeyValue {
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new TypeError("Query cache keys cannot contain non-finite numbers");
  }
  if (Array.isArray(value)) return value.map(normalizedKey);
  if (value && typeof value === "object") {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError("Query cache keys must contain only plain objects");
    }
    const record = value as Readonly<{[key: string]: QueryKeyValue}>;
    return Object.fromEntries(
      Object.keys(record)
        .sort()
        .map((key) => [key, normalizedKey(record[key] as QueryKeyValue)]),
    );
  }
  return value;
}

function canonicalKey(value: QueryKeyValue): string {
  return JSON.stringify(normalizedKey(value));
}

function assertScope(scope: ResourceScope): void {
  if (!scope.tenantId.trim() || !scope.userId.trim()) {
    throw new TypeError("Query cache scope requires non-empty tenantId and userId");
  }
}

function boundaryKey(scope: ResourceScope): string {
  return scopeKey(scope);
}

function scopeKey(scope: ResourceScope): string {
  assertScope(scope);
  return canonicalKey([
    scope.tenantId,
    scope.userId,
    scope.kbId ?? null,
    scope.resourceType ?? null,
    scope.resourceId ?? null,
  ]);
}

function staleResponse(): DOMException {
  return new DOMException("Stale resource response", "AbortError");
}

export class ScopedQueryCache {
  #entries = new Map<string, CacheEntry<unknown>>();
  #generation = 0;
  #boundary: string | null = null;

  generation(): number {
    return this.#generation;
  }

  #reset(reason: string): number {
    this.#generation += 1;
    for (const entry of this.#entries.values()) entry.controller?.abort(reason);
    this.#entries.clear();
    return this.#generation;
  }

  #ensureBoundary(scope: ResourceScope): void {
    const next = boundaryKey(scope);
    if (this.#boundary === null) {
      this.#boundary = next;
      return;
    }
    if (this.#boundary !== next) {
      this.#reset("tenant, user, knowledge-base, or resource scope changed");
      this.#boundary = next;
    }
  }

  changeScope(nextScope?: ResourceScope): number {
    const nextBoundary = nextScope ? boundaryKey(nextScope) : null;
    const generation = this.#reset("resource scope changed");
    this.#boundary = nextBoundary;
    return generation;
  }

  async query<Value>(options: {
    scope: ResourceScope;
    key: readonly QueryKeyValue[];
    ttlMs?: number;
    load: (signal: AbortSignal) => Promise<Value>;
  }): Promise<Value> {
    const ttlMs = options.ttlMs ?? 15_000;
    if (!Number.isFinite(ttlMs) || ttlMs < 0) {
      throw new TypeError("Query cache ttlMs must be a non-negative finite number");
    }
    const scopeFingerprint = scopeKey(options.scope);
    const queryParts = options.key.map(canonicalKey);
    this.#ensureBoundary(options.scope);
    const key = canonicalKey([scopeFingerprint, ...queryParts]);
    const now = Date.now();
    const existing = this.#entries.get(key) as CacheEntry<Value> | undefined;
    if (existing?.hasValue && existing.expiresAt > now) return existing.value as Value;
    if (existing?.promise) return existing.promise;
    const generation = this.#generation;
    const controller = new AbortController();
    const entry: CacheEntry<Value> = {
      hasValue: false,
      controller,
      expiresAt: 0,
      generation,
      scopeFingerprint,
      queryParts,
    };
    const promise = Promise.resolve()
      .then(() => options.load(controller.signal))
      .then((value) => {
        if (
          generation !== this.#generation
          || controller.signal.aborted
          || this.#entries.get(key) !== entry
        ) {
          throw staleResponse();
        }
        entry.hasValue = true;
        entry.value = value;
        delete entry.promise;
        delete entry.controller;
        entry.expiresAt = Date.now() + ttlMs;
        return value;
      })
      .catch((error: unknown) => {
        if (this.#entries.get(key) === entry) this.#entries.delete(key);
        throw error;
      });
    entry.promise = promise;
    this.#entries.set(key, entry);
    return promise;
  }

  invalidate(scope: ResourceScope, prefix: readonly QueryKeyValue[] = []): void {
    const expectedScope = scopeKey(scope);
    const expectedParts = prefix.map(canonicalKey);
    for (const [key, entry] of this.#entries) {
      if (entry.scopeFingerprint !== expectedScope) continue;
      if (!expectedParts.every((part, index) => entry.queryParts[index] === part)) continue;
      entry.controller?.abort("query invalidated");
      this.#entries.delete(key);
    }
  }
}
