from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPORT_PAGE = ROOT / "app/process/import_/page/import.html"
QUERY_PAGE = ROOT / "app/process/query/page/chat.html"


def _html() -> str:
    return IMPORT_PAGE.read_text(encoding="utf-8")


def test_import_workbench_has_deep_linked_server_8000_views() -> None:
    html = _html()

    for view, section_id in (
        ("knowledge-bases", "knowledgeBaseSection"),
        ("documents", "documentSection"),
        ("tasks", "taskSection"),
        ("preview", "previewSection"),
        ("recycle-bin", "cleanupPanel"),
    ):
        assert f'data-workbench-view-link="{view}"' in html
        assert re.search(
            rf'id="{section_id}"[^>]*data-workbench-view="{view}"', html
        )

    assert "const WORKSPACE_VIEWS = new Set(" in html
    assert "new URLSearchParams(location.search)" in html
    assert "history[method]" in html
    assert "'pushState'" in html
    assert "'replaceState'" in html
    assert "window.addEventListener('popstate'" in html
    assert "url.searchParams.set('kb', kbId)" in html
    assert "url.searchParams.set('document', state.selectedDocumentId)" in html

    create = html[
        html.index("async function createKnowledgeBase") : html.index(
            "async function deleteKnowledgeBase"
        )
    ]
    selected = create.index("await loadKnowledgeBases(payload.data.id);")
    cleared = create.index("state.selectedDocumentId = '';", selected)
    routed = create.index("writeWorkspaceRoute({replace: true});", cleared)
    refreshed = create.index("await loadWorkbenchKnowledgeBases({reset: true});", routed)
    assert selected < cleared < routed < refreshed


def test_import_workbench_uses_real_workbench_api_adapter() -> None:
    html = _html()
    adapter = html[
        html.index("const workbenchApi = {") : html.index(
            "function normalizeWorkspaceView"
        )
    ]

    for compatibility_endpoint in (
        "/api/v1/ui/bootstrap",
        "/api/v1/knowledge-bases/",
        "/api/v1/ingestion-jobs",
        "/api/v1/cleanup-jobs",
        "/api/v1/parsing/preview",
        "/versions",
    ):
        assert compatibility_endpoint in adapter

    for endpoint in (
        "/api/v1/workbench/knowledge-bases",
        "/api/v1/workbench/documents:batch-action",
        "/api/v1/workbench/documents/",
        "/api/v1/workbench/document-versions/",
        "/diff",
        "/source",
        "/blocks",
        "/chunks",
        "/api/v1/workbench/jobs",
        "/chunk-drafts",
        ":split",
        "/chunks:merge",
        ":publish",
    ):
        assert endpoint in adapter

    assert "'If-Match': etag" in adapter
    assert "function recoverWorkbenchPrecondition" in adapter
    assert "[412, 428]" in adapter
    assert "不会自行拼接不存在的分页、编辑或恢复能力" not in html
    assert "已发布文档的逐页 PDF、坐标框和切片编辑" not in html
    assert "软删除资源会保留审计与恢复窗口" not in html
    assert "当前版本不提供软删除恢复入口" in html


def test_import_workbench_lists_are_filterable_cursor_paged_and_race_guarded() -> None:
    html = _html()

    for element_id in (
        "wbKbSearch",
        "wbKbStatus",
        "wbKbSort",
        "wbDocumentSearch",
        "wbDocumentStatus",
        "wbDocumentMime",
        "wbJobStatus",
        "wbKbPrevious",
        "wbKbNext",
        "wbDocumentPrevious",
        "wbDocumentNext",
        "wbJobPrevious",
        "wbJobNext",
    ):
        assert f'id="{element_id}"' in html

    assert "function beginWorkbenchRequest(channel)" in html
    assert "function isCurrentWorkbenchRequest(channel, token)" in html
    assert "function workbenchPagePrevious" in html
    assert "function workbenchPageNext" in html
    assert "function invalidateWorkbenchPage" in html
    assert "筛选条件已变化，请重新查询" in html
    assert "cursor: page.cursor" in html
    assert "next_cursor" in html
    assert "mime_type: $('wbDocumentMime').value.trim()" in html
    assert "status: $('wbJobStatus').value" in html


def test_import_workbench_document_tree_diff_batch_and_job_aggregates_are_real() -> None:
    html = _html()

    for element_id in (
        "wbDocumentRows",
        "wbDocumentBatchAction",
        "wbDocumentBatchResult",
        "wbDocumentDetail",
        "wbVersionTree",
        "wbDiffLeft",
        "wbDiffRight",
        "wbJobList",
        "wbJobDetail",
    ):
        assert f'id="{element_id}"' in html

    assert "items: selected.map(item => ({document_id: item.id, etag: item.etag}))" in html
    assert "renderWorkbenchBatchResult(result)" in html
    assert "version.parse_artifacts" in html
    assert "version.chunking_runs" in html
    assert "diff.hunks?.items" in html
    assert "detail.events" in html
    assert "detail.checkpoints" in html
    assert "detail.manifest" in html
    assert "detail.artifacts" in html
    assert "detail.warnings" in html


def test_import_workbench_persistent_source_chunks_and_draft_editor_are_wired() -> None:
    html = _html()

    for element_id in (
        "wbPreviewVersion",
        "wbSourceFrame",
        "wbContentMode",
        "wbContentSearch",
        "wbContentType",
        "wbContentPage",
        "wbContentResults",
        "wbContentContextPanel",
        "wbContentContextPage",
        "wbContentContextBbox",
        "wbDraftChunkRows",
        "wbDraftPrevious",
        "wbDraftNext",
        "wbDraftReplace",
        "wbDraftSplit",
        "wbDraftMerge",
        "wbDraftDelete",
        "wbDraftPublish",
    ):
        assert f'id="{element_id}"' in html

    assert "mode === 'blocks'" in html
    assert "page_no: $('wbContentPage').value" in html
    assert "workbenchLocation(item)" in html
    assert "if (Array.isArray(value)) candidates.push" in html
    assert "coordinate_space" in html
    assert "#page=${encodeURIComponent(locationValue.page)}" in html
    assert "base_chunking_run_id" in html
    assert "draft.etag" in html
    assert "stt.workbench.chunk-draft.v1:${tenant}:${user}:${versionId}" in html
    assert "draft?.kb_id !== $('kbSelect').value" in html
    assert "wbReplaceDraftChunk" in html
    assert "wbSplitDraftChunk" in html
    assert "wbMergeDraftChunks" in html
    assert "wbDeleteDraftChunk" in html
    assert "wbPublishDraft" in html
    assert "published_job_id" in html


def test_upload_batch_freezes_kb_config_and_version_target() -> None:
    html = _html()
    freeze = html[
        html.index("function freezeUploadBatch") : html.index(
            "function isCurrentUploadBatch"
        )
    ]
    upload = html[
        html.index("async function uploadFile") : html.index(
            "async function handleFiles"
        )
    ]
    handle = html[
        html.index("async function handleFiles") : html.index(
            "async function documentAction"
        )
    ]

    assert "kbId," in freeze
    assert "config: Object.freeze({...config})" in freeze
    assert "generation: state.uploadBatchGeneration" in freeze
    assert "documentId: String(documentId || '')" in freeze
    assert "const {kbId, documentId, config} = batch;" in upload
    assert "const kbId = $('kbSelect').value;" not in upload
    assert "workbenchApi.uploadVersion(documentId, options)" in upload
    assert "workbenchApi.uploadDocument(kbId, options)" in upload
    assert "if (!isCurrentUploadBatch(batch))" in handle
    assert "for (const file of batch.files)" in handle
    assert "state.uploadBatchGeneration += 1;" in html


def test_pending_upload_is_removed_before_stale_workspace_exit() -> None:
    html = _html()
    upload = html[
        html.index("async function uploadFile") : html.index(
            "async function handleFiles"
        )
    ]

    removed = upload.index("state.pendingUploads.delete(pendingId);")
    detached = upload.index("card.remove();", removed)
    stale_exit = upload.index("if (!isCurrentUploadBatch(batch)) return;", detached)
    assert removed < detached < stale_exit
    assert "if (error.name === 'AbortError')" in upload
    assert "state.pendingUploads.delete(pendingId);" in upload


def test_document_rows_are_selectable_and_versions_are_not_conflated() -> None:
    html = _html()
    render = html[
        html.index("function renderDocumentContext") : html.index(
            "function renderIndexVersions"
        )
    ]

    assert 'id="contextDocumentCandidateVersion"' in html
    assert 'id="contextDocumentLatestJob"' in html
    assert "state.selectedDocumentId" in render
    assert "data-document-select" in html
    assert "selectButton.dataset.documentSelect" in render
    assert "row.dataset.selected" in render
    assert "activeVersion" in render
    assert "candidateVersion" in render
    assert "activeConfig" in render
    assert "candidateConfig" in render
    assert "候选版本处理失败；活动版本未受影响" in render
    assert "openVersionUpload(documentRow.id)" in render


def test_task_events_have_error_state_and_quiet_live_region() -> None:
    html = _html()
    loader = html[
        html.index("async function loadJobEvents") : html.index(
            "function replaceJob"
        )
    ]

    assert 'id="fileList" class="import-task-list" aria-live="off"' in html
    assert 'id="taskStatusAnnouncement"' in html
    assert 'aria-atomic="true"' in html
    assert "eventErrors: new Map()" in html
    assert "state.eventErrors.set(jobId, error)" in loader
    assert "state.eventErrors.delete(jobId)" in loader
    assert "阶段事件读取失败" in html
    assert "重试读取事件" in html
    assert "announceJobTransition" in html


def test_mobile_context_and_focus_management_are_reachable() -> None:
    html = _html()

    assert 'id="heroContextButton"' in html
    assert 'aria-controls="workspaceContextRail"' in html
    assert 'id="workspaceContextRail"' in html
    assert 'id="contextBackButton"' in html
    assert ".import-context-shortcut { display: inline-flex; }" in html
    assert ".import-context-back { display: inline-flex;" in html
    assert "function focusContextRail()" in html
    assert "function focusWorkspaceHeading()" in html
    assert "workspaceTitle').focus" in html
    assert "button?.focus()" in html


def test_preview_technical_payload_does_not_dump_storage_fields() -> None:
    html = _html()
    preview = html[
        html.index("function previewDisplayPayload") : html.index(
            "async function previewSelectedFile"
        )
    ]

    assert "JSON.stringify(previewDisplayPayload(payload)" in preview
    assert "JSON.stringify(payload, null, 2)" not in preview
    assert "object_key" not in preview
    assert "artifact_uri" not in preview
    assert "内部存储地址" in preview


def test_import_inline_script_is_valid_javascript() -> None:
    node = shutil.which("node")
    if node is None:
        return

    scripts = re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", _html())
    inline_scripts = [script for script in scripts if script.strip()]
    assert inline_scripts
    for script in inline_scripts:
        result = subprocess.run(
            [node, "--check", "-"],
            input=script,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_query_workbench_has_real_deep_links_and_structured_ab_preview() -> None:
    html = QUERY_PAGE.read_text(encoding="utf-8")
    restore = html[
        html.index("async function restoreWorkbenchConversation") : html.index(
            "function selectKnowledgeBaseForConversation"
        )
    ]

    assert 'href="/workbench/chat"' in html
    assert 'href="/workbench/retrieval"' in html
    assert "location.pathname.endsWith('/retrieval')" in html
    assert "new URLSearchParams(location.search)" in restore
    assert re.search(r"\.get\(\s*['\"]conversation['\"]\s*\)", restore)
    assert "stt_preview_page=" in _html()
    assert "const explicitKb = String(params.get('kb') || '');" in restore
    assert "if (explicitKb && !validDeepLink)" in restore
    assert "preferredConversationId: ''" in restore
    assert 'id="retrievalProfileSelect"' in html
    assert 'id="retrievalProfileBSelect"' in html
    assert 'id="btnCompareProfiles"' in html
    assert "Promise.allSettled" in html
    for stage in ("route/raw", "normalized*", "RRF", "rerank", "cutoff"):
        assert stage in html
    assert "normalized 为本页面" in html
    assert "error.code === 'INDEX_NOT_READY' ? '索引未就绪' : '失败'" in html


def test_shared_ui_explains_how_to_recover_from_unpublished_index() -> None:
    shared_ui = (ROOT / "app" / "process" / "shared" / "page" / "stt-console.js").read_text(
        encoding="utf-8"
    )

    assert "INDEX_NOT_READY" in shared_ui
    assert "等待文档处理完成" in shared_ui
    assert "确认索引已发布后再试" in shared_ui


def test_query_history_feedback_and_citation_modal_use_existing_safe_apis() -> None:
    html = QUERY_PAGE.read_text(encoding="utf-8")

    assert 'id="conversationSearch"' in html
    assert 'id="conversationStatusFilter"' in html
    assert "mutateSelectedConversation('archive')" in html
    assert "mutateSelectedConversation('restore')" in html
    assert "expected_revision" in html
    assert "'/api/v1/feedback'" in html
    assert 'id="evidenceDialog"' in html
    assert "source-inline" in html
    assert "#page=${page}&view=FitH" in html
    assert "evidenceDialogReturnFocus" in html
    assert "target.focus()" in html
    assert "object_key" not in html
    assert "artifact_uri" not in html


def test_query_inline_script_is_valid_javascript() -> None:
    node = shutil.which("node")
    if node is None:
        return

    html = QUERY_PAGE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", html)
    inline_scripts = [script for script in scripts if script.strip()]
    assert inline_scripts
    for script in inline_scripts:
        result = subprocess.run(
            [node, "--check", "-"],
            input=script,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
