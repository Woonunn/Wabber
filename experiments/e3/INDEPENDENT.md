# 213 个独立请求

本协议保留历史 24 个成功 Capture，各使用一次；新增 189 个请求独立执行真实 LLM Capture。新增数量为 DataSciBench 48、TravelPlanner 47、ClaudeCodeReview 47、ScientificComputation 47，最终数量为 54/53/53/53。十分钟的 213 个原始 Copilot 到达 ID 和时间保持不变，Workload 分配按新组成预先确定。

`results/e3-independent/plan.json` 保存请求定义和输入。2026-11-04 已完成189条新增请求的本地 Docker + 官方 DeepSeek-V4.1-Flash 实际采集，其中182条通过业务验证、7条失败。加上保留的24条，`capture-index.json` 记录213个独立 live request_id；报告见 `CAPTURE_REPORT.md`，验证审计见 `local-capture/live-audit.json`。四种Workload各一条本地Session冻结回放样例均通过，未新增模型API调用。未通过业务验证的请求仍不能封存为成功的正式 Replay 输入，完整213请求集尚未完成严格Replay封存，也未完成新增请求的四云验证和实际费用平衡检验。

## 请求与输入

- TravelPlanner：从固定官方验证集选择另外 47 个任务，保留官方原 Prompt、约束和验证器；不与旧六个正式任务及原 Profiling 行重叠。
- ScientificComputation：生成另外 47 个数值问题，各有独立矩阵、响应向量、输入摘要和独立 solve/residual 验证，保留真实状态复用需求。
- DataSciBench：保留原六套数据和完整业务要求，新增 48 个有不同沟通、证据解释要求的派生请求。
- ClaudeCodeReview：保留现有真实项目及缺陷 Fixture，新增 47 个有不同沟通、证据解释要求的派生请求，继续使用原缺陷与测试验证。

派生 Prompt 明确标注为人工构造请求，不声称是真实用户日志。DataSci 和 Review 当前仍复用各自六套数据或 PR；213 个独立请求并不代表 213 个互不相关的业务问题。它们能引入独立模型执行变化，但对新数据集、新 PR 的泛化仍需另行验证。失败尝试全部保留，复用现有固定三次上限、首个验证通过结果的采集规则；不按费用或路径形态筛选。

Descriptor 从当前请求的 Prompt、文件信息与工具能力产生。文件公开元数据中的 rows/columns/features 可成为到达特征。不同请求允许落入相同资源类别，描述符中不加入任务 ID、Prompt/文件哈希或 Capture 编号。任务 ID 只传给执行与业务验证，路由决策接口收到的 task 参数为 None。

## 从空模型启动

`profile.initialize` 检查四 Workload × 四 Provider 的预检覆盖和正确性，创建空模型，不导入预检 Profile 或成本。`run.build_router` 每次直接构造新模型，忽略历史模型路径，移除费率配置中已有的性能校准参数，重置探索 epoch；探索状态随新 Router 创建。公开价格、平台配置、工具能力、并发和超时继续保留。

正式运行期间，完成的请求首先更新执行画像。`feedback.NativeCostFeedback` 使用现有 CloudWatch/SLS 采集、E3 关联与 ATSuite 计价，每 30 秒尝试补齐已完成请求的原生费用。只有所选 Provider 的观测可回灌，available_ns 使用实际获得数据的时间；迟到修订也保留版本。缺失费用保持未知。AgentCore 用量必须覆盖成功 StopRuntimeSession 回执，不能只覆盖最后一次 Tool。

AgentCore 原生用量可能晚于整个十分钟请求窗口到达，因此本轮在线决策可能无法及时利用其原生成本反馈。这是结果需要报告的限制，不能通过导入预检价格消除。

## 准备、采集与封存

```bash
PYTHONPATH=. .venv/bin/python -m experiments.e3.independent prepare \
  --source results/e3-stateful/trace-balanced \
  --matrix results/e3-stateful/final/cost-matrix.json \
  --output results/e3-independent

PYTHONPATH=. .venv/bin/python -m experiments.e3.independent capture \
  --directory results/e3-independent --configs LOCAL_CAPTURE_CONFIGS.json --concurrency 2

PYTHONPATH=. .venv/bin/python -m experiments.e3.independent seal \
  --directory results/e3-independent --output results/e3-independent/trace
```

现有目录已完成 prepare，不重复运行覆盖。`LOCAL_CAPTURE_CONFIGS.json` 按四个 Workload 名称映射到现有本地 Tool 配置，例如 `{"TravelPlanner":{"tool":{...}}, ...}`；请求自己的模型配置来自封存的 agent.json。启动现有 Scientific Tool 服务时，将 `ATSUITE_SCIENTIFIC_PROBLEMS` 指向准备目录中的 `runtime-inputs/scientific`，其文件与已封存原始输入使用硬链接共享只读内容。DataSci 继续在现有 Docker sandbox 执行生成程序。

成功 Capture 不会重复调用 API；存在未完成或失败的采集目录时停止该请求，保留原记录供检查。全部 Capture 到齐后，seal 检查 213 个独立 live request_id、独立目录、对应 Prompt 与输入，以及每个任务恰好分配一次。单纯复制目录无法通过。自然生成相同决策路径允许保留。

每条请求的 Replay 使用其 Capture 中实际模型 seed/参数，避免重试后被全局配置覆盖。`captures/approved_trace.json` 合并 DataSci 各独立 Capture 的精确程序摘要，沿用现有 frozen_policy 接口，不放宽云端程序准入。`runtime-manifests` 导出 Travel/Scientific 汇总输入清单，供现有部署打包器使用。新输入和获准程序需要打入新镜像后再做云端预检。

实际 Agent 配置也保留 `planner_instruction` / `validation_contract` 文本覆盖，用于明确官方房型名称和原始 Prompt 引文等通用规则。Capture 与 Replay 都读取同一份已记录配置；原始 Prompt、输入和独立验证器不因此变化。运行错误修复及模型参数调整前的尝试单独归档，计数和 token 审计必须保留这些历史，不能把每轮三次上限误报为整个请求只执行过三次。

新协议修改了执行源码，所以旧协议的源文件封存校验会拒绝使用当前源码重新执行旧实验。旧数据和结果保留；新实验在 Capture 完成后封存当前源码，不修改旧 seal 来绕过检查。

## 费用平衡

准备时先用历史 24 个任务的 Ali FC/MCP 费用确定任务族配比，再固定请求和输入。现有计划的两侧估算均约 ¥1.79855。该值是历史任务族的规划估算，不是新增请求的实测费用，也不意味着 AWS 已平衡。

“差不多”的预先判据为同一云厂商下两种模式总费用差额 / 较高总费用不超过 5%。Ali 使用 CNY，AWS 使用 USD，分别检查；不混币种，不以缺失费用记零。复用 ATSuite 原生用量计价，不增加一套费率公式。

```bash
PYTHONPATH=. .venv/bin/python -m experiments.e3.independent balance \
  --directory results/e3-independent --prices NEW_NATIVE_EVIDENCE/request-usage.jsonl
```

价格记录必须匹配本次每个 arrival 的具体请求和 Workload；旧 213 请求的费用不能冒充新请求费用。完整配对价格到齐后才返回平衡结论。若实际超过 5%，如实标记未达到目标，保留请求流与结果，不自动删除、替换或重新加权已捕获任务。
