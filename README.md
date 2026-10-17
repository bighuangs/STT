# STT · 掌柜智库

面向文档知识库的 RAG 应用，覆盖资料导入、版本化索引、混合检索、带证据的问答、历史会话及质量评测。后端采用 Python、FastAPI、SQLAlchemy 和 LangGraph；管理与对话页面由 FastAPI 提供，TypeScript 接口契约由 OpenAPI 生成。

## 核心能力

- **知识库与版本管理**：知识库、文档、文档版本、索引版本及发布绑定分别建模；支持候选索引校验、显式发布、回滚与清理。
- **持久导入**：API 持久入队，独立 Worker 执行；提供租约、心跳、幂等、有限重试、检查点、取消及批次进度。
- **解析与证据**：多格式解析适配器与切块模板，保存来源块、页码、坐标、表格和图片身份；具体格式能力取决于启用的解析器。
- **检索与生成**：问题处理、可配置稠密/稀疏混合检索、融合、重排、上下文预算、证据充分性检查、拒答及引用校验。
- **可靠对话**：Conversation / Turn / QueryRun 身份、持久事件、SSE 续传、取消、恢复、幂等和用户历史会话目录。
- **隔离与运维**：认证、多租户 RBAC、知识库授权求交、受控外发、鉴权原文与图片访问、配置脱敏、健康检查与可选 Phoenix/OpenTelemetry。
- **独立评测**：版本化数据集、检索与生成指标、完整案例配对比较及质量门禁。

高级检索、连接器、工具、长期记忆及多模态能力具有独立配置和授权边界。使用前需准备相应服务、模型和凭据；离线自检不代表外部服务已经接通。

## 架构与目录

```text
浏览器 → Import API :8000 → SQL 持久任务 → Ingestion Worker
                                      ↓
                           解析 / 切块 / 向量化 / 索引校验
                                      ↓
                         Milvus + MinIO + 控制库发布绑定

浏览器 → Query API :8001 → 授权范围 / 活动索引 → 检索 / 融合 / 重排
                                      ↓
                           上下文 / 生成 / 引用与证据校验
                                      ↓
                             持久 QueryRun / SSE / 历史
```

| 目录 | 职责 |
| --- | --- |
| `app/api/` | HTTP 接口与公开请求/响应模型 |
| `app/application/`、`app/domain/` | 应用用例、端口、领域模型与策略 |
| `app/bootstrap/`、`app/runtime/` | 服务装配、图编译与运行时集成 |
| `app/agent_system/` | 智能体相关行为与编排 |
| `app/infra/`、`app/infrastructure/` | SQL、向量库、对象存储、模型及观测适配 |
| `app/process/`、`app/rag/` | 导入/查询图、RAG 能力及内置页面 |
| `app/security/`、`app/shared/`、`app/foundation/` | 安全、配置与共享基础能力 |
| `app/worker/`、`alembic/` | 导入 Worker 与控制库迁移 |
| `evals/` | 数据集、配置、grader、harness 与评测平台适配 |
| `frontend/` | OpenAPI 快照、生成 Client 与浏览器运行时契约 |
| `scripts/`、`test/`、`tests/` | 工程工具及单元、接口、集成、架构与契约测试 |

项目正在进行模块化收敛，部分旧入口仍然共存。详细边界见 [架构说明](docs/architecture/refactor-2026-08-14.md)。

## 环境要求

- Python **3.11～3.13** 与 `uv`；Node.js/npm 仅用于前端契约生成与校验。
- Milvus 向量服务、MinIO 对象存储、MongoDB 兼容历史服务，以及 SQL 控制库（本地默认 SQLite）。
- OpenAI 兼容的 LLM/Vision 服务；按解析路径配置 MinerU 服务。
- Embedding 与 Reranker 模型文件，默认使用 BGE-M3 和 BGE reranker，具体路径与设备见 `.env.example`。

依赖可由已有 Docker 环境或独立服务提供。仓库不包含服务数据卷和模型权重；远程模型、Vision、解析等服务可能计费。

## 安装与配置

```powershell
git clone https://github.com/bighuangs/STT.git
cd STT
uv sync --locked
Copy-Item .env.example .env
```

编辑 `.env`，至少检查：

| 配置 | 用途 |
| --- | --- |
| `AUTH_TOKEN_SECRET` | 随机认证密钥，替换示例值 |
| `AUTH_BOOTSTRAP_ADMIN_EMAIL`、`AUTH_BOOTSTRAP_ADMIN_PASSWORD` | 首次启动管理员邮箱与强密码 |
| `QUERY_CHECKPOINT_ENCRYPTION_KEY` | 独立随机 32 字节密钥的 URL-safe Base64，用于可靠查询检查点 |
| `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`LLM_DEFAULT_MODEL` | 模型端点、密钥与模型名 |
| `BGE_M3_PATH`、`BGE_RERANKER_LARGE` | 本机模型目录 |
| `MILVUS_URL`、`MONGO_URL` | 向量库与兼容历史服务 |
| `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY` | 对象存储地址与凭据 |
| `MINERU_BASE_URL`、`MINERU_API_TOKEN` | PDF 解析服务 |
| `CONTROL_DATABASE_URL` | 默认 `sqlite:///./volumes/control/stt-control.db` |

分别生成认证随机值和检查点密钥，再填入对应字段：

```powershell
uv run python -c "import secrets; print(secrets.token_urlsafe(48))"
uv run python -c "import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
```

这两项必须使用不同的值。`.env` 已被 Git 忽略；`.env.example` 只包含示例和占位符。生产环境还需覆盖开发凭据、配置 CORS、数据外发策略及共享存储。

## 启动与使用

先执行迁移：

```powershell
uv run alembic upgrade head
```

分别在三个终端启动：

```powershell
# 终端 1：导入与知识库管理 API
uv run uvicorn app.api.http.import_server:app --host 127.0.0.1 --port 8000

# 终端 2：持久导入 Worker（必须）
uv run python -m app.worker.ingestion_worker

# 终端 3：查询 API
uv run uvicorn app.api.http.query_server:app --host 127.0.0.1 --port 8001
```

| 页面 | 地址 |
| --- | --- |
| 知识库与文档运维 | <http://127.0.0.1:8000/html> |
| 知识问答与历史 | <http://127.0.0.1:8001/html> |
| 检索工作台 | <http://127.0.0.1:8001/workbench/retrieval> |
| 评测控制台 | <http://127.0.0.1:8001/eval/html> |
| 高级能力 / 可靠性 | <http://127.0.0.1:8001/p2/html> / <http://127.0.0.1:8001/p3/html> |
| API 文档 | 两个服务各自的 `/docs` |

首次登录使用 `.env` 中的 bootstrap 管理员邮箱与密码，默认租户为 `tenant_default`。已有数据库不会因修改 `.env` 自动重置密码。

创建知识库并上传自己有权使用的资料，观察任务与 Worker；任务完成后核对索引校验及发布状态，再在问答页选择同一知识库。引用、原文与图片对应固定文档版本；历史会话从左侧目录恢复。

`/health/live` 表示进程存活，`/health/ready` 检查当前角色的必需依赖并返回 200/503；依赖详情需要相应管理权限。

## 开发与验证

```powershell
# 离线测试需要完整依赖；不要向测试环境提供生产凭据
uv run pytest test tests -m "not external" --maxfail=1
uv run pytest tests/architecture -q
uv run python -m evals.harness --project-root .

# 静态检查、秘密扫描与链接校验
uv run ruff check app evals test tests scripts --select E9,F63,F7,F82
uv run python scripts/secret_scan.py
uv run python scripts/check_markdown_links.py

# 前端契约
uv run python scripts/generate_openapi_typescript.py --check
npm --prefix frontend ci --ignore-scripts
npm --prefix frontend run typecheck
```

路由变化后，在已配置环境中重新生成客户端并检查差异，详见 [前端契约说明](frontend/README.md)。真实模型、服务或手册的外部验证需要另行准备环境和授权资料。

## 发布范围与限制

公开仓库保留主代码、迁移、测试、前端契约和必要工程文档；不包含本机 `.env`、密钥、数据库、模型、日志、原始 PDF、解析输出和本地运行产物。示例评测集中的文档身份与摘要用于测试，原始资料需另行准备；第三方模型、服务和资料遵守各自许可。

单份手册验收与离线自检不能证明生产容量、灾备或所有格式均已覆盖；部署前应验证目标服务、租户权限、恢复能力、模型质量和资源需求。
