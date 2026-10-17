# 公开源码快照说明

发布日期：2026-10-17。此目录是从本地当前工作区筛选得到的独立源码快照，使用全新 Git 历史，原开发仓库与 Gitee 远端保留原状。

## 内容范围

包含应用源码与 Prompt、SQL 迁移、前端契约与第三方图标许可、工程脚本、测试及评测所需的轻量配置/数据集。公开仓库不携带本机凭据、原始手册、课程资料、模型、数据库、日志、解析输出、截图、临时目录或大体积评测运行记录。

部分历史评测注册表和研究脚本仍引用未公开的运行制品。需要重放历史实验时，应另行准备对应制品与合法资料；离线注册表校验不能证明这些外部制品齐备。

## 本次实际验证

- 依赖锁：`uv lock --check --offline` 通过。
- Python 源码语法检查及 Ruff `E9,F63,F7,F82` 通过。
- OpenAPI 生成文件一致性及 TypeScript 类型检查通过。
- 秘密扫描、候选文件与本地真实秘密比对、排除内容扫描通过。
- 公开 Markdown 的相对链接检查通过。
- 秘密扫描正反例与前端契约测试：28 passed。
- 加入架构守卫的组合：38 passed、1 failed。失败用例为 `test_application_outward_dependencies_do_not_exceed_baseline`，当前代码已有以下反向依赖超出历史债务基线：
  - `app.application.query_answering.builders → app.rag.query.context_pack_service`
  - `app.application.retrieval_service → app.infra.persistence.sql.database`
  - `app.application.retrieval_service → app.infra.persistence.sql.repositories`

发布准备没有放宽该守卫、改写架构债务基线或宣称全量测试通过。上述问题需要独立的架构修复；GitHub CI 中的架构检查可能据此报告失败。

本次仅执行工程与发布检查，未重新调用付费模型，也未修改运行数据库、知识库索引或部署服务。
