(() => {
  'use strict';

  const ERROR_MESSAGES = {
    UNAUTHENTICATED: '登录已失效，请重新登录。',
    FORBIDDEN: '当前账号没有执行此操作的权限。',
    SESSION_KB_CONFLICT: '已为新的知识库切换到独立会话，请重试。',
    RETRIEVAL_SCOPE_DENIED: '该知识库当前不可查询，可能未授权或尚未发布。',
    INDEX_NOT_READY: '当前知识库还没有可查询的已发布索引。请等待文档处理完成，确认索引已发布后再试。',
    STATE_CONFLICT: '资源状态已变化，请刷新后再试。',
    NOT_FOUND: '目标资源已不存在或不可见。',
    DEPENDENCY_UNAVAILABLE: '依赖服务暂不可用，请稍后重试。',
    VALIDATION_ERROR: '输入内容不符合要求，请检查后重试。',
  };

  const STATUS_META = {
    ready: ['可用', 'success'],
    active: ['已发布', 'success'],
    succeeded: ['已完成', 'success'],
    completed: ['已完成', 'success'],
    healthy: ['正常', 'success'],
    processing: ['处理中', 'info'],
    running: ['运行中', 'info'],
    leased: ['已领取', 'info'],
    queued: ['等待处理', 'warning'],
    pending: ['等待处理', 'warning'],
    staging: ['待发布', 'warning'],
    empty: ['暂无内容', 'warning'],
    degraded: ['部分可用', 'warning'],
    blocked: ['不可用', 'danger'],
    failed: ['失败', 'danger'],
    cancelled: ['已取消', 'danger'],
    disabled: ['已停用', 'danger'],
    offline: ['离线', 'danger'],
    stale: ['心跳过期', 'danger'],
  };
  let refreshPromise = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>'"]/g, character => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;',
    })[character]);
  }

  function statusMeta(status) {
    const normalized = String(status || 'unknown').toLowerCase();
    const [label, tone] = STATUS_META[normalized] || [normalized || '未知', 'neutral'];
    return {status: normalized, label, tone};
  }

  function statusBadge(status, overrideLabel = '') {
    const meta = statusMeta(status);
    return `<span class="stt-status" data-tone="${escapeHtml(meta.tone)}">${escapeHtml(overrideLabel || meta.label)}</span>`;
  }

  function normalizeError(payload, status = 0) {
    const error = payload?.error || {};
    const code = String(error.code || payload?.code || (status === 401 ? 'UNAUTHENTICATED' : ''));
    const safeMessage = ERROR_MESSAGES[code]
      || error.message
      || payload?.detail
      || (status ? `请求失败（HTTP ${status}）` : '请求失败，请稍后重试。');
    return {
      code: code || 'UNKNOWN_ERROR',
      message: safeMessage,
      requestId: payload?.request_id || '',
      status,
      retryable: Boolean(error.retryable),
    };
  }

  function notifyAuthenticationExpired() {
    window.dispatchEvent(new CustomEvent('stt:auth-expired'));
  }

  async function refreshSessionOnce() {
    if (!refreshPromise) {
      refreshPromise = (async () => {
        try {
          const response = await fetch('/api/v1/auth/refresh', {
            method: 'POST',
            credentials: 'same-origin',
            headers: {'Content-Type': 'application/json'},
            body: '{}',
          });
          return response.ok;
        } catch (_) {
          return false;
        }
      })();
    }
    const activeRefresh = refreshPromise;
    try {
      return await activeRefresh;
    } finally {
      if (refreshPromise === activeRefresh) refreshPromise = null;
    }
  }

  async function apiJson(path, options = {}) {
    const {authRefresh = true, ...fetchOptions} = options;
    const headers = new Headers(options.headers || {});
    if (
      typeof fetchOptions.body === 'string'
      && fetchOptions.body.trim().length > 0
      && !headers.has('Content-Type')
    ) {
      headers.set('Content-Type', 'application/json');
    }
    const requestOptions = {
      ...fetchOptions,
      credentials: 'same-origin',
      headers,
    };
    let response = await fetch(path, requestOptions);
    if (
      response.status === 401
      && authRefresh
      && !String(path).startsWith('/api/v1/auth/')
    ) {
      const refreshed = await refreshSessionOnce();
      if (refreshed) response = await fetch(path, requestOptions);
      if (response.status === 401) notifyAuthenticationExpired();
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const normalized = normalizeError(payload, response.status);
      const error = new Error(normalized.message);
      Object.assign(error, normalized);
      throw error;
    }
    return payload;
  }

  function ensureToastRegion() {
    let region = byId('sttToastRegion');
    if (region) return region;
    region = document.createElement('div');
    region.id = 'sttToastRegion';
    region.className = 'stt-toast-region';
    region.setAttribute('role', 'status');
    region.setAttribute('aria-live', 'polite');
    region.setAttribute('aria-atomic', 'true');
    document.body.appendChild(region);
    return region;
  }

  function toast(message, tone = 'info', timeoutMs = 4200) {
    const region = ensureToastRegion();
    const item = document.createElement('div');
    item.className = 'stt-toast';
    item.dataset.tone = tone;
    item.textContent = message;
    region.appendChild(item);
    window.setTimeout(() => item.remove(), timeoutMs);
  }

  function announce(message) {
    let live = byId('sttLiveRegion');
    if (!live) {
      live = document.createElement('div');
      live.id = 'sttLiveRegion';
      live.className = 'stt-sr-only';
      live.setAttribute('aria-live', 'polite');
      live.setAttribute('aria-atomic', 'true');
      document.body.appendChild(live);
    }
    live.textContent = '';
    window.setTimeout(() => { live.textContent = message; }, 10);
  }

  function setBusy(element, busy, label = '') {
    if (!element) return;
    if (busy) {
      element.dataset.originalLabel = element.textContent;
      element.disabled = true;
      element.setAttribute('aria-busy', 'true');
      if (label) element.textContent = label;
    } else {
      element.disabled = false;
      element.removeAttribute('aria-busy');
      if (element.dataset.originalLabel) {
        element.textContent = element.dataset.originalLabel;
        delete element.dataset.originalLabel;
      }
    }
  }

  function ensureConfirmDialog() {
    let dialog = byId('sttConfirmDialog');
    if (dialog) return dialog;
    dialog = document.createElement('dialog');
    dialog.id = 'sttConfirmDialog';
    dialog.className = 'stt-dialog';
    const form = document.createElement('form');
    form.method = 'dialog';
    const header = document.createElement('header');
    const title = document.createElement('h2');
    title.id = 'sttConfirmTitle';
    title.textContent = '确认操作';
    header.appendChild(title);
    const message = document.createElement('div');
    message.id = 'sttConfirmMessage';
    message.className = 'stt-dialog-body';
    const footer = document.createElement('footer');
    const cancel = document.createElement('button');
    cancel.className = 'stt-button';
    cancel.value = 'cancel';
    cancel.textContent = '取消';
    const accept = document.createElement('button');
    accept.id = 'sttConfirmAccept';
    accept.className = 'stt-button primary';
    accept.value = 'confirm';
    accept.textContent = '确认';
    footer.append(cancel, accept);
    form.append(header, message, footer);
    dialog.appendChild(form);
    document.body.appendChild(dialog);
    return dialog;
  }

  async function confirmAction({
    title = '确认操作',
    message = '该操作会更改当前资源状态。',
    confirmLabel = '确认',
    danger = false,
  } = {}) {
    const dialog = ensureConfirmDialog();
    byId('sttConfirmTitle').textContent = title;
    byId('sttConfirmMessage').textContent = message;
    const accept = byId('sttConfirmAccept');
    accept.textContent = confirmLabel;
    accept.className = `stt-button ${danger ? 'danger' : 'primary'}`;
    if (typeof dialog.showModal !== 'function') {
      return window.confirm(`${title}\n\n${message}`);
    }
    return new Promise(resolve => {
      const onClose = () => {
        dialog.removeEventListener('close', onClose);
        resolve(dialog.returnValue === 'confirm');
      };
      dialog.addEventListener('close', onClose);
      dialog.showModal();
    });
  }

  function hasPermission(bootstrap, action) {
    const permissions = new Set(bootstrap?.permissions || []);
    return permissions.has('*') || permissions.has(action);
  }

  function knowledgeBaseStorageKey(bootstrap, namespace = 'workspace') {
    const identity = bootstrap?.identity || {};
    return `stt.ui.kb.${namespace}.${identity.tenant_id || 'unknown'}.${identity.user_id || 'unknown'}`;
  }

  function populateKnowledgeBaseSelect(select, bootstrap, {
    namespace = 'workspace',
    allowBlocked = true,
  } = {}) {
    const rows = Array.isArray(bootstrap?.knowledge_bases) ? bootstrap.knowledge_bases : [];
    const visible = allowBlocked ? rows : rows.filter(item => item.readiness === 'ready');
    const key = knowledgeBaseStorageKey(bootstrap, namespace);
    const previous = localStorage.getItem(key) || select.value;
    const options = visible.map(item => {
      const suffix = item.readiness === 'ready' ? '' : ` · ${item.readiness_message}`;
      const option = document.createElement('option');
      option.value = String(item.id || '');
      option.textContent = `${item.name || '未命名知识库'}${suffix}`;
      return option;
    });
    select.replaceChildren(...options);
    if (visible.some(item => item.id === previous)) select.value = previous;
    if (select.value) localStorage.setItem(key, select.value);
    select.disabled = visible.length === 0;
    return visible.find(item => item.id === select.value) || null;
  }

  function rememberKnowledgeBase(select, bootstrap, namespace = 'workspace') {
    if (!select?.value) return;
    localStorage.setItem(knowledgeBaseStorageKey(bootstrap, namespace), select.value);
  }

  function selectedKnowledgeBase(select, bootstrap) {
    return (bootstrap?.knowledge_bases || []).find(item => item.id === select?.value) || null;
  }

  function formatDate(value) {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).format(date);
  }

  function shortId(value) {
    const text = String(value || '');
    if (text.length <= 18) return text || '—';
    return `${text.slice(0, 10)}…${text.slice(-6)}`;
  }

  async function bootstrap() {
    return apiJson('/api/v1/ui/bootstrap');
  }

  async function login({email, password, tenantId}) {
    return apiJson('/api/v1/auth/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        email: String(email || '').trim(),
        password: String(password || ''),
        tenant_id: String(tenantId || '').trim() || null,
      }),
    });
  }

  async function logout() {
    // logout 是服务端幂等公开端点；即使 access cookie 已过期，也必须让
    // 服务端撤销 refresh credential 并回写两个过期的 HttpOnly cookie。
    return apiJson('/api/v1/auth/logout', {
      method: 'POST',
      authRefresh: false,
    });
  }

  window.STTUI = Object.freeze({
    announce,
    apiJson,
    bootstrap,
    byId,
    confirmAction,
    escapeHtml,
    formatDate,
    hasPermission,
    knowledgeBaseStorageKey,
    login,
    logout,
    normalizeError,
    populateKnowledgeBaseSelect,
    rememberKnowledgeBase,
    selectedKnowledgeBase,
    setBusy,
    shortId,
    statusBadge,
    statusMeta,
    toast,
  });
})();
