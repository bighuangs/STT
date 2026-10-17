from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
from xml.etree import ElementTree

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAGES = (
    ROOT / "app/process/import_/page/import.html",
    ROOT / "app/process/query/page/chat.html",
    ROOT / "app/process/query/page/evaluation.html",
    ROOT / "app/process/query/page/p2.html",
    ROOT / "app/process/query/page/p3.html",
)
PRIMARY_PAGES = PAGES[:2]
SHARED_JS = (
    ROOT / "app/process/shared/page/stt-console.js",
    ROOT / "app/process/shared/page/stt-diagnostics.js",
)
LUCIDE_VENDOR_ROOT = ROOT / "app/process/shared/page/vendor/lucide"
LUCIDE_NAV_ICONS = {
    "message-circle": "知识问答",
    "database": "知识库运维",
    "chart-no-axes-combined": "能力评测",
    "shield-check": "系统验证",
}


class _SemanticPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.label_for: set[str] = set()
        self.controls: list[tuple[str, dict[str, str | None], bool]] = []
        self.buttons: list[dict[str, str | None]] = []
        self.h1_count = 0
        self.main_count = 0
        self.nav_count = 0
        self.skip_link_count = 0
        self.live_region_count = 0
        self._label_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(str(values["id"]))
        if tag == "label":
            self._label_depth += 1
            if values.get("for"):
                self.label_for.add(str(values["for"]))
        if tag in {"input", "select", "textarea"}:
            self.controls.append((tag, values, self._label_depth > 0))
        if tag == "button":
            self.buttons.append(values)
        if tag == "h1":
            self.h1_count += 1
        if tag == "main":
            self.main_count += 1
        if tag == "nav":
            self.nav_count += 1
        if tag == "a" and "stt-skip-link" in str(values.get("class") or "").split():
            self.skip_link_count += 1
        if values.get("role") in {"status", "alert"} or values.get("aria-live"):
            self.live_region_count += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "label":
            self._label_depth -= 1


def _parse(path: Path) -> tuple[str, _SemanticPageParser]:
    html = path.read_text(encoding="utf-8")
    parser = _SemanticPageParser()
    parser.feed(html)
    return html, parser


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.name)
def test_workbench_pages_have_semantic_and_keyboard_baseline(path: Path) -> None:
    html, page = _parse(path)
    assert '<html lang="zh-CN">' in html
    assert 'name="viewport"' in html
    assert page.main_count >= 1
    assert page.nav_count >= 1
    assert page.h1_count >= 1
    assert page.skip_link_count == 1
    assert page.live_region_count >= 1

    duplicates = [key for key, count in Counter(page.ids).items() if count > 1]
    assert duplicates == []

    missing_button_type = [
        item.get("id") or item.get("class") or "<anonymous>"
        for item in page.buttons
        if not item.get("type")
    ]
    assert missing_button_type == []

    missing_labels: list[str] = []
    for tag, attributes, nested_in_label in page.controls:
        if (
            str(attributes.get("type") or "").lower() == "hidden"
            or "hidden" in attributes
        ):
            continue
        control_id = str(attributes.get("id") or "")
        labelled = (
            nested_in_label
            or bool(attributes.get("aria-label"))
            or bool(attributes.get("aria-labelledby"))
            or bool(control_id and control_id in page.label_for)
        )
        if not labelled:
            missing_labels.append(control_id or f"<{tag}>")
    assert missing_labels == []


@pytest.mark.parametrize("path", PAGES, ids=lambda path: path.name)
def test_login_is_a_real_gate_and_pages_do_not_publish_bootstrap_credentials(
    path: Path,
) -> None:
    html = path.read_text(encoding="utf-8")
    assert "Stt-Admin-ChangeMe" not in html
    assert 'value="admin@stt.local"' not in html
    assert "stt-console.css" in html
    assert "stt-console.js" in html
    assert re.search(
        r'(?:id="(?:workspace|appShell)"[^>]*hidden|data-stt-protected[^>]*hidden|class="shell"[^>]*hidden)',
        html,
    )
    assert not re.search(
        r'(?:localStorage|sessionStorage)\.setItem\([^,\n]*(?:token|api.?key)',
        html,
        flags=re.IGNORECASE,
    )


@pytest.mark.parametrize("path", PRIMARY_PAGES, ids=lambda path: path.name)
def test_primary_business_pages_use_authorized_kb_context_and_safe_dom(path: Path) -> None:
    html = path.read_text(encoding="utf-8")
    assert "/api/v1/ui/bootstrap" in html or "STTUI.bootstrap()" in html
    assert "<select" in html
    assert "innerHTML" not in html
    assert "insertAdjacentHTML" not in html
    assert "outerHTML" not in html
    assert "STTUI.confirmAction" in html


def test_query_workspace_uses_durable_query_run_and_scoped_sessions() -> None:
    html = PAGES[1].read_text(encoding="utf-8")
    assert "stt.query.sessions.v1" in html
    assert "JSON.stringify([currentPrincipal.tenant_id, currentPrincipal.user_id, kbId])" in html
    assert "/api/v1/query-runs" in html
    assert "Last-Event-ID" in html
    assert "SESSION_KB_CONFLICT" in html
    assert "activateSessionForCurrentKb({forceNew: true})" in html


def test_query_workspace_rejects_stale_async_results_and_deduplicates_events() -> None:
    html = PAGES[1].read_text(encoding="utf-8")
    assert "generation: runtime.generation" in html
    assert "context.generation !== runtime.generation" in html
    assert "requestControllers: new Set()" in html
    assert "abortRuntimeRequests()" in html
    assert "runtime.currentRun?.query_run_id !== runId" in html
    assert "runtime.botMessage !== botMessage" in html
    assert "incomingRevision < currentRevision" in html
    assert "sequence <= runtime.lastSeq || runtime.eventBuffer.has(sequence)" in html
    assert "runtime.eventBuffer.has(runtime.lastSeq + 1)" in html
    assert "runtime.eventSource !== source" in html
    assert "!isRuntimeContextCurrent(context)" in html
    assert "if (!isRuntimeContextCurrent(runContext)) return;" in html
    logout_block = html[
        html.index("byId('btnLogout').addEventListener")
        : html.index("function resetRuntimeState")
    ]
    assert logout_block.index("resetRuntimeState()") < logout_block.index(
        "await STTUI.logout()"
    )


def test_query_workspace_recovers_terminal_results_and_reuses_uncertain_submit_key() -> None:
    html = PAGES[1].read_text(encoding="utf-8")
    assert 'id="btnRetryResult"' in html
    assert "RESULT_RETRY_DELAYS_MS" in html
    assert "QUERY_RESULT_NOT_READY" in html
    assert "runtime.resultRecoveryAvailable = true" in html
    assert "运行已完成，正在读取持久结果与引用" in html
    assert "stt.query.pending.v1" in html
    assert "pendingSubmission(text)" in html
    assert "clearPendingSubmission(pending.storageKey, key)" in html


def test_query_workspace_lists_and_loads_all_current_user_conversations() -> None:
    html = PAGES[1].read_text(encoding="utf-8")
    assert 'id="conversationHistorySelect"' in html
    assert 'id="conversationHistoryList"' in html
    assert 'id="btnRefreshHistory"' in html
    assert 'id="btnHistoryNew"' in html
    assert "我的全部对话" in html
    assert (
        "/api/v1/conversations?limit=200&offset=${offset}&include_archived=true"
        in html
    )
    assert "if (items.length < 200) break;" in html
    assert "/messages`" in html
    assert "/query-runs`" in html
    assert "item?.status !== 'committed'" in html
    assert "item.role !== 'assistant'" in html
    assert "beginConversationHistoryGeneration" in html
    assert "abortConversationHistoryRequests()" in html
    assert "isConversationHistoryContextCurrent(historyContext)" in html
    assert "该历史会话已归档，只能查看" in html
    assert "旧对话不会删除" in html


def test_query_workspace_history_selection_only_persists_server_identity() -> None:
    html = PAGES[1].read_text(encoding="utf-8")
    block = html[
        html.index("function rememberSelectedConversation")
        : html.index("function abortConversationHistoryRequests")
    ]
    assert "currentPrincipal.tenant_id" in html
    assert "currentPrincipal.user_id" in html
    assert "conversation_id: normalized" in block
    assert "title:" not in block
    assert "content:" not in block
    assert "answer:" not in block


def test_query_workspace_renders_only_same_origin_evidence_images_with_recovery() -> None:
    html = PAGES[1].read_text(encoding="utf-8")
    assert "function sameOriginEvidenceImageUrl" in html
    assert "url.origin !== window.location.origin" in html
    assert (
        "^\\/api\\/v1\\/evidence\\/[a-zA-Z0-9_-]+\\/images\\/[a-zA-Z0-9_-]+$"
        in html
    )
    assert "source?.images" in html
    assert "evidence?.image_urls" in html
    assert "正在检查引用证据" in html
    assert "图片加载失败：资产可能已归档" in html
    assert "retry.addEventListener('click'" in html
    assert "打开原图" in html
    assert "image.loading = 'lazy'" in html
    image_card = html[
        html.index("function appendEvidenceImageCard")
        : html.index("async function discoverEvidenceImages")
    ]
    assert "image.hidden" not in image_card
    assert ".answer-image-frame.is-loaded img { min-height: 0; opacity: 1; }" in html
    assert "image.remove()" not in html
    assert "asset_refs" not in html


def test_import_workspace_scopes_async_facts_and_reuses_uncertain_upload_key() -> None:
    html = PAGES[0].read_text(encoding="utf-8")
    assert "workspaceRevision" in html
    assert "beginWorkspaceRevision()" in html
    assert "isCurrentWorkspace(context)" in html
    assert "requestController()" in html
    assert "signal: controller.signal" in html
    assert "clearSelectedResourceFacts()" in html
    assert "stt.ui.pending-upload.v1" in html
    assert "uploadIdempotency(file, config, kbId)" in html
    assert "clearUploadIdempotency(idempotency.scope)" in html
    assert "/api/v1/ingestion-workers" not in html
    assert "/api/v1/cleanup/orphans" not in html
    assert "job.scope || {}" in html


def test_import_job_updates_refresh_worker_summary_with_task_cards() -> None:
    html = PAGES[0].read_text(encoding="utf-8")
    replace_job = html[
        html.index("function replaceJob(job)") : html.index("function pollStatus")
    ]
    assert "renderWorkers();" in replace_job
    assert replace_job.index("renderWorkers();") < replace_job.index("renderActiveJobs();")


def test_import_job_manual_refresh_uses_workspace_context() -> None:
    html = PAGES[0].read_text(encoding="utf-8")
    bindings = html[html.index("function bindEvents()") : html.index("async function init()")]
    assert (
        "$('refreshJobsButton').addEventListener('click', () => "
        "refreshJobs(workspaceContext()));"
    ) in bindings
    assert "addEventListener('click', refreshJobs)" not in bindings


def test_index_coverage_prefers_logical_document_counts_over_manifest_rows() -> None:
    html = PAGES[0].read_text(encoding="utf-8")
    assert "冻结覆盖 / 清单校验" in html
    render_indexes = html[
        html.index("function renderIndexVersions") : html.index("function renderCleanupJobs")
    ]
    assert "index.document_coverage" in render_indexes
    assert "coverage.bound_documents" in render_indexes
    assert "coverage.covered_documents" in render_indexes
    assert "coverage.expected_documents" in render_indexes
    assert "coverage.validated_documents" in render_indexes
    assert "冻结覆盖 ${boundDocuments}/${expectedDocuments} · 清单校验 ${validatedDocuments}/${expectedDocuments}" in render_indexes


def test_import_workspace_reconciles_kb_readiness_after_job_terminal_state() -> None:
    html = PAGES[0].read_text(encoding="utf-8")
    reconciliation = html[
        html.index("function reconcileSelectedKbSummary()")
        : html.index("function renderIdentity()")
    ]
    assert "summary.ready_document_count = queryableCount" in reconciliation
    assert "activeJobStatuses.has(item.latest_job?.status)" in reconciliation
    assert "summary.readiness = 'ready'" in reconciliation
    assert "summary.readiness_message = processingCount" in reconciliation
    assert "selectedOption.textContent" in reconciliation
    refresh = html[
        html.index("async function refreshSelectedKnowledgeBase")
        : html.index("async function refreshJobs")
    ]
    assert "const summary = reconcileSelectedKbSummary()" in refresh


def test_shared_auth_refresh_is_singleflight_and_logout_is_server_confirmed() -> None:
    js = SHARED_JS[0].read_text(encoding="utf-8")
    assert "let refreshPromise = null" in js
    assert "async function refreshSessionOnce()" in js
    assert "const activeRefresh = refreshPromise" in js
    assert "new CustomEvent('stt:auth-expired')" in js
    logout = js[js.index("async function logout()") : js.index("window.STTUI")]
    assert "authRefresh: false" in logout
    assert "catch" not in logout


def test_shared_json_client_sets_content_type_for_serialized_request_bodies() -> None:
    js = SHARED_JS[0].read_text(encoding="utf-8")
    api_json = js[js.index("async function apiJson") : js.index("function ensureToastRegion")]
    assert "typeof fetchOptions.body === 'string'" in api_json
    assert "!headers.has('Content-Type')" in api_json
    assert "headers.set('Content-Type', 'application/json')" in api_json


def test_evaluation_workspace_unwraps_shared_resource_envelopes() -> None:
    html = PAGES[2].read_text(encoding="utf-8")
    api_json = html[
        html.index("async function apiJson")
        : html.index("function showNotice")
    ]
    assert "unwrap(await window.STTUI.apiJson" in api_json
    assert "return unwrap(body)" in api_json


def test_evaluation_run_form_adapts_to_primary_column_width() -> None:
    html = PAGES[2].read_text(encoding="utf-8")
    run_form_css = html[
        html.index("    .run-form {")
        : html.index("    .table-wrap {")
    ]

    # The desktop workspace reserves a fixed context rail, so viewport breakpoints
    # cannot tell how much width remains in the primary column.  Auto-fit must
    # wrap all five controls before they can overflow beneath the sticky rail.
    assert "repeat(auto-fit, minmax(min(100%, 220px), 1fr))" in run_form_css
    assert "minmax(170px, .7fr)" not in run_form_css
    assert 'id="runBaselineButton"' in html
    assert 'id="runCandidateButton"' in html


def test_all_workbenches_hide_local_state_before_server_logout_finishes() -> None:
    import_html = PAGES[0].read_text(encoding="utf-8")
    import_logout = import_html[
        import_html.index("async function logout()")
        : import_html.index("async function createKnowledgeBase")
    ]
    assert import_logout.index("showAuthGate(") < import_logout.index(
        "await STTUI.logout()"
    )

    diagnostic_js = SHARED_JS[1].read_text(encoding="utf-8")
    diagnostic_logout = diagnostic_js[
        diagnostic_js.index("logout.addEventListener")
        : diagnostic_js.index("kbSelect?.addEventListener")
    ]
    assert diagnostic_logout.index("await showLogin()") < diagnostic_logout.index(
        "await STTUI.logout()"
    )


def test_shared_scripts_and_all_inline_scripts_parse_with_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")

    for path in SHARED_JS:
        result = subprocess.run(
            [node, "--check", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    pattern = re.compile(r"<script(?:\s[^>]*)?>(.*?)</script>", re.DOTALL | re.IGNORECASE)
    for path in PAGES:
        html = path.read_text(encoding="utf-8")
        for index, source in enumerate(pattern.findall(html)):
            if not source.strip():
                continue
            result = subprocess.run(
                [node, "--check", "-"],
                cwd=ROOT,
                input=source,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, f"{path.name} script {index}: {result.stderr}"


def test_shared_style_has_responsive_and_reduced_motion_guards() -> None:
    css = (ROOT / "app/process/shared/page/stt-console.css").read_text(
        encoding="utf-8"
    )
    assert "@media (max-width: 1000px)" in css
    assert "@media (max-width: 680px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_option_two_workspace_shell_is_consistent_across_pages() -> None:
    for path in PAGES:
        html = path.read_text(encoding="utf-8")
        assert 'class="stt-topbar' in html
        assert 'class="stt-nav"' in html
        assert 'aria-current="page"' in html
        assert "知识问答" in html
        assert "知识库运维" in html
        assert "能力评测" in html
        assert "系统验证" in html

    import_html = PAGES[0].read_text(encoding="utf-8")
    assert import_html.count('<aside class="import-context-rail"') == 1
    for context_id in (
        "contextDocumentName",
        "contextDocumentStatus",
        "contextDocumentVersion",
        "contextDocumentChunks",
        "contextDocumentParser",
        "contextDocumentTemplate",
        "healthContextPanel",
        "healthContextRows",
    ):
        assert f'id="{context_id}"' in import_html

    query_html = PAGES[1].read_text(encoding="utf-8")
    assert query_html.count('<aside class="stt-context-rail"') == 1
    assert re.search(r'id="adminSidebar"[^>]*hidden', query_html)


def test_import_dependency_summary_uses_health_ready_contract() -> None:
    import_html = PAGES[0].read_text(encoding="utf-8")

    assert "const dependenciesReady = payload.ready === true;" in import_html
    assert "payload.status === 'healthy'" not in import_html
    assert "healthContextMeta').textContent = dependenciesReady" in import_html


def test_frontend_asset_versions_are_atomic_and_local() -> None:
    versions: set[str] = set()
    for path in PAGES:
        html = path.read_text(encoding="utf-8")
        assert "https://cdn" not in html
        assert "data:image/" not in html
        page_versions = re.findall(
            r'/ui-assets/(?:stt-console\.(?:css|js)|stt-diagnostics\.js)\?v=([^"\']+)',
            html,
        )
        assert page_versions
        versions.update(page_versions)
    assert versions == {"20260807.1"}


def test_shared_style_exposes_option_two_desktop_and_mobile_contracts() -> None:
    css = (ROOT / "app/process/shared/page/stt-console.css").read_text(
        encoding="utf-8"
    )
    assert "--stt-sidebar-width: 248px" in css
    assert "--stt-context-rail-width: 360px" in css
    assert "@media (min-width: 1180px)" in css
    assert "position: fixed" in css
    assert ".stt-page-grid" in css
    assert "@media (max-width: 760px)" in css


def test_primary_navigation_uses_local_decorative_lucide_icons() -> None:
    for path in PAGES:
        html = path.read_text(encoding="utf-8")
        navigation = re.search(
            r'<nav class="stt-nav"[^>]*>(.*?)</nav>',
            html,
            flags=re.DOTALL,
        )
        assert navigation is not None
        markup = navigation.group(1)
        assert markup.count('class="stt-nav-icon"') == len(LUCIDE_NAV_ICONS)
        assert markup.count('aria-hidden="true"') == len(LUCIDE_NAV_ICONS)
        assert markup.count('focusable="false"') == len(LUCIDE_NAV_ICONS)
        for icon_id, visible_label in LUCIDE_NAV_ICONS.items():
            assert (
                f'href="/ui-assets/lucide-nav.svg#{icon_id}"' in markup
            )
            assert f"<span>{visible_label}</span>" in markup


def test_lucide_sprite_is_pinned_valid_and_licensed() -> None:
    sprite_path = LUCIDE_VENDOR_ROOT / "lucide-nav.svg"
    license_path = LUCIDE_VENDOR_ROOT / "LICENSE"
    source_path = LUCIDE_VENDOR_ROOT / "SOURCE.md"

    root = ElementTree.parse(sprite_path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    symbols = {
        node.attrib["id"]: node
        for node in root.findall("svg:symbol", namespace)
    }
    assert set(symbols) == set(LUCIDE_NAV_ICONS)
    for symbol in symbols.values():
        assert symbol.attrib["viewBox"] == "0 0 24 24"
        assert symbol.attrib["fill"] == "none"
        assert symbol.attrib["stroke"] == "currentColor"
        assert symbol.attrib["stroke-width"] == "2"
        assert symbol.attrib["stroke-linecap"] == "round"
        assert symbol.attrib["stroke-linejoin"] == "round"

    license_text = license_path.read_text(encoding="utf-8")
    assert "ISC License" in license_text
    assert "Copyright (c) 2026 Lucide Icons and Contributors" in license_text
    assert "The MIT License (MIT) (for the icons listed above)" in license_text
    assert "Copyright (c) 2013-present Cole Bemis" in license_text

    source_text = source_path.read_text(encoding="utf-8")
    assert "0f8d48b266e7af1e2bab49ff12ab6ec0795ed204" in source_text
    assert "https://github.com/lucide-icons/lucide" in source_text


def test_shared_navigation_icon_style_keeps_icons_small_and_inherited() -> None:
    css = (ROOT / "app/process/shared/page/stt-console.css").read_text(
        encoding="utf-8"
    )
    icon_style = css[
        css.index(".stt-nav-icon {") : css.index(".stt-nav a:hover")
    ]
    assert "width: 18px" in icon_style
    assert "height: 18px" in icon_style
    assert "color: currentColor" in icon_style
    assert "pointer-events: none" in icon_style
