# STT Workbench 前端契约

该目录保存 TODO-12 工作台的类型契约和无框架运行时。当前生产页面仍由 FastAPI
提供原生 HTML/JavaScript，以避免一次性迁移影响已经验证的导入与查询主链；所有新增
组件工程均以两个 FastAPI OpenAPI 文档作为唯一 DTO 来源。

## 生成与校验

在项目虚拟环境中执行：

```powershell
cd STT
uv run python scripts/generate_openapi_typescript.py
cd frontend
npm ci --ignore-scripts
npm run typecheck
npm run check:generated
```

常规 `--check` 使用仓库内冻结的 OpenAPI 快照，速度快且不触发模型初始化；发布前使用
`--check-apps` 同时装配两个 FastAPI 应用，检查快照是否与真实路由发生漂移：

```powershell
cd STT
uv run python scripts/generate_openapi_typescript.py --check --check-apps
```

## 安全与并发约束

- Client 固定使用同源 Cookie，不在浏览器持久化或注入 Bearer Token。
- 可变资源写入通过 `If-Match`，服务端以 `428/412` 拒绝缺失或陈旧版本。需要保留
  自定义 428 的路由使用 operation-level `x-stt-required-headers: [If-Match]`；生成器会
  将其提升为必填 TypeScript 调用参数，而不会根据 HTTP 方法猜测。
- 缓存键必须包含 tenant、user、KB 和资源身份；身份、KB 或视图切换会中止旧请求。
- API 错误保留服务端 request ID，但不会把对象存储键、本地路径或内部异常暴露给页面。
- OpenAPI 快照和生成文件都需要提交；`frontend/node_modules` 不进入版本库。
