# E3：跨 Workload 请求路由

最新两组真实测试的性能、计价、路由分布和问题分析见 [TWO_ARM.md](TWO_ARM.md)。

新一轮大状态、长计算任务集的设计、配对验证与 ¥30 开销范围见 [STATEFUL.md](STATEFUL.md)。
平衡后的 213 请求四平台重测、AWS 环境校准与 Lambda 额度等待状态见 [BALANCED_FOUR_PLATFORM.md](BALANCED_FOUR_PLATFORM.md)。
按用户要求，两侧额外成本平衡后的新请求流位于 `results/e3-stateful/trace-balanced`，
保留原 213 个到达时间与四类请求数量，任务频次、成本平衡误差和设计依据见其中的 `balance.json`。

正式实验仅比较两组：

1. `fixed`：213 条请求全部固定在预先选定的一个 Provider（当前为阿里 FC）。
2. `exploration`：OnlineRouter 根据到达时画像、Provider 成本预测与完成反馈选池，并进行在线探索。

两组共享原始 10 分钟到达记录、任务、Prompt、Capture、执行环境和原生弹性策略。不存在 Static 或独立纯 Online 实验选项，也不设置总预算、探索预算或请求费用上限。并发准入、执行超时和 Agent 最大轮数仍保留；不管理云端实例数量，不迁移运行中的 Session。

四种 Workload 各有独立 LangGraph Agent 和业务验证，共享 TracingAdapter、Tool Runtime、Session 绑定与反馈。请求只选择一次 Provider，全部 Tool 调用固定在该池。MCP Session 内保留状态；FaaS 每次调用恢复/提交对象存储快照。到达描述只使用原 Prompt、输入元数据、工具能力和 Workload 类型，不使用未来 DAG 或 task_id 匹配。

费用统一使用 `atsuite.analysis.api.price_metrics` 和 `atsuite.analysis.pricing`。E3 evidence 只负责把已关联原生 Request/Session ID 的测量转换成 ATSuite 的 ProviderMetric/StateObservation，不维护第二套计价公式。Lambda 使用原生计费时长、内存、请求与临时盘；AgentCore 使用原生 CPU-hours/Memory GB-hours；阿里 Session 使用原生完整生命周期，避免只统计 Tool 活跃时间。状态费用沿用 ATSuite，另行披露请求内 Trace 状态观测范围。结果是原生用量计价，不声称为已结算发票；CUR 和日账单不再是实验前置条件。

初始化模型仅使用预先完成的 24 任务×四平台预检。正式基线结果不回灌初始化。`profile.initialize` 保存初始 Workload/Provider 模型及来源；每组从相同封存环境开始，Online 仅在请求完成后更新模型。两组顺序运行，避免争用同一执行池。

容量观察为只读：阿里原生 Active Session 与本地未释放预约共同进入准入。当前实验在任务验证结束后主动释放 Session；释放失败时保留预约，并由固定 idle 策略兜底。AgentCore 原生用量日志有交付延迟，计价须等待终止后的用量记录。观测失败时对应池停止新增准入，不调整实例、配额或弹性策略。

```bash
.venv/bin/python -m experiments.e3.run --sequence results/e3-two-arm/trace \
  --config results/e3-two-arm/initial-final/config.json --strategy fixed \
  --output results/e3-two-arm/fixed
.venv/bin/python -m experiments.e3.run --sequence results/e3-two-arm/trace \
  --config results/e3-two-arm/initial-final/config.json --strategy exploration \
  --output results/e3-two-arm/exploration
```

输入选取、官方 Travel 检查、真实 Review 项目与 Capture 谱系见 [TEN_MINUTE.md](TEN_MINUTE.md)。历史四平台测试保留于 [REAL_TEST.md](REAL_TEST.md)，旧 Pilot 保留于 [PROFILING.md](PROFILING.md)。新的双组协议只更新客户端/计价源码封存，原始 213 条逻辑请求与 Capture 字节不变。

正常路由中的经验/置信度不足仅作为诊断和探索信息，不排除仍具备可比较费用预测的廉价候选；容量、工具能力和执行截止时间仍是准入条件。超过历史负载范围本身不构成切换到更贵 Provider 的理由。原双组实验中 11 个错误决策已完成离线回归，见 TWO_ARM.md；没有将该回归描述为新云端收益。
