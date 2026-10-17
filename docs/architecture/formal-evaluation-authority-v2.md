# 正式评测运行权威 v2

## 执行入口

新的 HAK180 正式运行必须使用
`evals/configs/hak180_online_v1_3_authority_v3.json`。既有
`hak180_online_v1_1.json`、`v1_2.json`、`v1_3.json` 是已发布的 v2
资产，只用于历史读取和重放，不得原地改写或用于创建新的正式运行。

只读校验：

```powershell
python scripts/run_formal_evaluation.py `
  --config evals/configs/hak180_online_v1_3_authority_v3.json `
  --project-root . `
  --kb-id <kb-id> `
  --user-id <user-id>
```

加 `--execute --allow-paid-model-calls` 后，CLI 不再提前执行一次 preflight。
`EvaluationService.create_run` 执行唯一权威 preflight，repository 再使用实际选择的
case、run ID、tenant、KB、baseline 和完整 runtime snapshot 完成
`stt.eval.formal-run.v2`。CLI 的执行结果只读取耐久 run，不复用调用前临时值。

## 耐久绑定

`config_snapshot.formal_evaluation` 包含调用方的 `formal-intent.v1` 与服务端生成的
`run_authority`。自哈希绑定配置、dataset registry/记录、scenario、grader、失败策略、
实际有序 case 集合、retrieval profile、vector read set、evaluation security、代码与配置
版本、模型、运行 flags、preflight、baseline contract 以及完整 runtime snapshot。
历史 `formal-run.v1` 可以读取，但不能用于创建新的正式运行。

paired run 只允许 `allowed_intervention_dimensions` 明确声明的变化。当前 HAK180 v3
资产只允许 `code_revision`；profile、模型、安全、向量读集或运行 flags 的未声明变化
均失败关闭。

## 预算与用量语义

`planning_admission_budgets` 只限制每个 case 的 QueryPlan 准入：计划 token 上限、
最大 route 数和 planner cost units。它不是实际 provider token、USD 或 RetrievalPort
调用次数的硬限制。

当前 contract 必须显式声明：

- provider token 用量：`unavailable`
- provider USD 成本：`unavailable`
- retrieval 调用次数：`unavailable`
- 实际用量硬限制：`not_enforced`

运行权威另行冻结已有真实执行点的 route timeout、生成输出 token cap、provider 请求
timeout、单请求最大 attempts 与 retry delay；这些字段不能被解释为全图 token/USD 账本。
