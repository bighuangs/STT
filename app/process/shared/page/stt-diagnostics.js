(() => {
  'use strict';

  const page = document.body.dataset.sttDiagnostic;
  if (!page || !window.STTUI) return;

  const shell = document.querySelector('[data-stt-protected], #diagnosticMain');
  const kbSelect = document.getElementById(
    page === 'evaluation' ? 'kbIdInput' : 'kbId'
  );
  let bootstrapState = null;
  let sessionEpoch = 0;
  let workspacePromise = null;

  function createGate() {
    const gate = document.createElement('section');
    gate.id = 'diagnosticLoginGate';
    gate.className = 'stt-login-gate stt-card';
    gate.setAttribute('aria-labelledby', 'diagnosticLoginTitle');

    const header = document.createElement('div');
    header.className = 'stt-card-header';
    const copy = document.createElement('div');
    const eyebrow = document.createElement('div');
    eyebrow.className = 'stt-eyebrow';
    eyebrow.textContent = '管理员验证中心';
    const title = document.createElement('h2');
    title.id = 'diagnosticLoginTitle';
    title.textContent = '登录后查看验证与诊断工具';
    const description = document.createElement('p');
    description.textContent = '此页面面向具备相应 P2、P3 或评测权限的运维与研发人员；日常知识库问答请使用查询工作台。';
    copy.append(eyebrow, title, description);
    header.append(copy);

    const body = document.createElement('div');
    body.className = 'stt-card-body stt-login-form';
    const fields = [
      ['diagnosticEmail', 'email', '邮箱', 'name@example.com'],
      ['diagnosticPassword', 'password', '密码', '请输入密码'],
      ['diagnosticTenant', 'text', '租户（可选）', '首次登录可留空'],
    ];
    for (const [id, type, labelText, placeholder] of fields) {
      const wrapper = document.createElement('label');
      wrapper.className = 'stt-field';
      wrapper.htmlFor = id;
      const label = document.createElement('span');
      label.textContent = labelText;
      const input = document.createElement('input');
      input.id = id;
      input.type = type;
      input.className = 'stt-input';
      input.placeholder = placeholder;
      input.autocomplete = type === 'password' ? 'current-password' : (type === 'email' ? 'username' : 'organization');
      wrapper.append(label, input);
      body.append(wrapper);
    }
    const error = document.createElement('div');
    error.id = 'diagnosticLoginError';
    error.className = 'stt-login-error';
    error.setAttribute('role', 'alert');
    const actions = document.createElement('div');
    actions.className = 'stt-login-actions';
    const login = document.createElement('button');
    login.id = 'diagnosticLoginButton';
    login.type = 'button';
    login.className = 'stt-button primary';
    login.textContent = '登录';
    const query = document.createElement('a');
    query.className = 'stt-button quiet';
    query.href = '/html';
    query.textContent = '返回查询工作台';
    actions.append(login, query);
    body.append(error, actions);
    gate.append(header, body);
    document.body.prepend(gate);
    return gate;
  }

  const gate = createGate();

  function legacyAuthControls(hidden) {
    const auth = document.querySelector('.auth, .auth-bar');
    if (!auth) return;
    for (const input of auth.querySelectorAll('input')) input.hidden = hidden;
    const login = auth.querySelector('#loginBtn, #loginButton');
    if (login) login.hidden = hidden;
  }

  function configureTechnicalDetails() {
    const raw = document.getElementById('raw');
    if (raw && raw.parentElement?.tagName !== 'DETAILS') {
      const details = document.createElement('details');
      details.className = 'stt-disclosure';
      const summary = document.createElement('summary');
      summary.textContent = '技术详情与原始响应';
      raw.parentElement.insertBefore(details, raw);
      details.append(summary, raw);
    }
    for (const button of document.querySelectorAll('button')) {
      if (!button.hasAttribute('type')) button.type = 'button';
    }
    for (const control of document.querySelectorAll('input:not([type="hidden"]), select, textarea')) {
      if (control.hasAttribute('aria-label') || control.hasAttribute('aria-labelledby')) continue;
      if (control.id && document.querySelector(`label[for="${CSS.escape(control.id)}"]`)) continue;
      control.setAttribute('aria-label', control.name || control.id || '输入项');
    }
  }

  function updateKnowledgeBaseContext() {
    if (!kbSelect || !bootstrapState) return;
    const rows = Array.isArray(bootstrapState.knowledge_bases)
      ? bootstrapState.knowledge_bases
      : [];
    const storageKey = STTUI.knowledgeBaseStorageKey(bootstrapState, `diagnostic-${page}`);
    const queryKb = new URLSearchParams(location.search).get('kb_id');
    const legacyKey = page === 'evaluation' ? 'stt.eval.kbId' : '';
    const preferred = queryKb
      || (legacyKey ? localStorage.getItem(legacyKey) : '')
      || localStorage.getItem(storageKey)
      || kbSelect.value;
    kbSelect.replaceChildren();
    for (const item of rows) {
      const option = document.createElement('option');
      option.value = String(item.id || '');
      option.textContent = `${item.name || item.id}${
        item.readiness === 'ready' ? '' : ` · ${item.readiness_message || item.readiness || '不可用'}`
      }`;
      option.disabled = !item.id;
      kbSelect.appendChild(option);
    }
    if (rows.some(item => item.id === preferred)) kbSelect.value = preferred;
    kbSelect.disabled = rows.length === 0;
    const selected = rows.find(item => item.id === kbSelect.value) || null;
    if (selected) {
      localStorage.setItem(storageKey, selected.id);
      if (legacyKey) localStorage.setItem(legacyKey, selected.id);
    }
    kbSelect.classList.add('stt-select');
    kbSelect.setAttribute('aria-label', '选择授权知识库');
    if (!selected) {
      STTUI.toast('当前账号没有可用于该验证页的知识库。', 'danger');
    }
  }

  async function showWorkspace(prefetched = null) {
    if (workspacePromise) return workspacePromise;
    const epoch = ++sessionEpoch;
    const currentPromise = (async () => {
      const nextBootstrap = prefetched || await STTUI.bootstrap();
      if (epoch !== sessionEpoch) return;
      bootstrapState = nextBootstrap;
      updateKnowledgeBaseContext();
      const identity = document.getElementById('identity');
      if (identity) {
        const me = bootstrapState.identity || {};
        identity.textContent = `${(me.roles || []).join('、') || '已登录'} · ${
          me.tenant_name || me.tenant_id || '当前租户'
        }`;
        identity.className = identity.className.includes('pill') ? 'pill success' : 'badge good';
      }
      const pageController = window.STTDiagnosticPage;
      if (pageController && typeof pageController.onAuthenticated === 'function') {
        await pageController.onAuthenticated(bootstrapState);
      }
      if (epoch !== sessionEpoch) return;
      gate.hidden = true;
      if (shell) shell.hidden = false;
      legacyAuthControls(true);
    })();
    workspacePromise = currentPromise;
    try {
      await currentPromise;
    } finally {
      if (workspacePromise === currentPromise) workspacePromise = null;
    }
  }

  async function showLogin() {
    sessionEpoch += 1;
    workspacePromise = null;
    bootstrapState = null;
    gate.hidden = false;
    if (shell) shell.hidden = true;
    legacyAuthControls(true);
    const pageController = window.STTDiagnosticPage;
    if (pageController && typeof pageController.onSignedOut === 'function') {
      await pageController.onSignedOut();
    }
    document.getElementById('diagnosticEmail')?.focus();
  }

  document.getElementById('diagnosticLoginButton').addEventListener('click', async event => {
    const button = event.currentTarget;
    const error = document.getElementById('diagnosticLoginError');
    sessionEpoch += 1;
    workspacePromise = null;
    STTUI.setBusy(button, true, '正在登录…');
    error.textContent = '';
    try {
      const loginResult = await STTUI.login({
        email: document.getElementById('diagnosticEmail').value,
        password: document.getElementById('diagnosticPassword').value,
        tenantId: document.getElementById('diagnosticTenant').value,
      });
      document.getElementById('diagnosticPassword').value = '';
      await showWorkspace(loginResult?.bootstrap || null);
    } catch (caught) {
      document.getElementById('diagnosticPassword').value = '';
      error.textContent = caught.message;
    } finally {
      STTUI.setBusy(button, false);
    }
  });

  document.getElementById('diagnosticPassword').addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      document.getElementById('diagnosticLoginButton').click();
    }
  });

  const logout = document.getElementById('logoutBtn') || document.getElementById('logoutButton');
  if (logout) {
    logout.addEventListener('click', async () => {
      await showLogin();
      const error = document.getElementById('diagnosticLoginError');
      error.textContent = '正在撤销服务端会话…';
      try {
        await STTUI.logout();
        error.textContent = '已安全退出。';
      } catch (caught) {
        error.textContent = `当前页面已退出，但服务端会话撤销失败：${caught.message}。网络恢复后请刷新页面并再次退出。`;
      }
    });
  }

  kbSelect?.addEventListener('change', () => {
    if (!bootstrapState) return;
    STTUI.rememberKnowledgeBase(kbSelect, bootstrapState, `diagnostic-${page}`);
    if (page === 'evaluation') {
      localStorage.setItem('stt.eval.kbId', kbSelect.value);
    }
  });
  window.addEventListener('stt:auth-expired', () => {
    showLogin().catch(() => {});
  });

  configureTechnicalDetails();
  if (shell) shell.hidden = true;
  const initialSessionEpoch = sessionEpoch;
  STTUI.bootstrap()
    .then(prefetched => {
      if (sessionEpoch === initialSessionEpoch) return showWorkspace(prefetched);
      return undefined;
    })
    .catch(() => {
      if (sessionEpoch === initialSessionEpoch) return showLogin();
      return undefined;
    });
})();
