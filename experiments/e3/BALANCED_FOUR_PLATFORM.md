# 平衡请求流四平台重测

本轮用于测量四个固定 Provider，确认 FC/Session MCP 与 Lambda/AgentCore 各自是否存在任务成本交叉。后续路由实验仍只有 fixed 和 online+exploration 两组。

输入固定为 `results/e3-stateful/trace-balanced`：原始十分钟 Copilot 到达流，213 个请求、24 个任务，DataSciBench/TravelPlanner/ClaudeCodeReview/ScientificComputation 分别为 64/51/53/45 个请求。Manifest SHA-256 为 `c0249606b0909810c003ec6076da7480e5a86a3b6ed31af74a7a5794ac8e0c2f`。任务频次在本轮 AWS 执行前已确定；不根据本轮结果再次调权。

原 Prompt、输入、Tool 程序、LLM 决策与真实 idle 固定。AWS 的 ARM DataSci 数值与序列化输出存在环境差异，因此使用 ATSuite 已有环境参考封存机制：先独立业务验证，核对程序/决策/idle 未变，再严格 Replay。AgentCore 六种 DataSci 任务已通过这些检查；不将跨环境结果声称为字节完全相同的 Capture。

运行配置、原生资源 read-back、镜像摘要、预检、请求记录与计价证据均保存在 `results/e3-balanced-four-platform`。费用复用 ATSuite 的原生用量计价；不依赖 CUR，也不将这些结果声称为已结算发票。FaaS 每次恢复与提交对象存储状态，MCP 在 Session 内保留状态并在任务验证结束后立即释放。资源和原生弹性策略在各组派发前固定，不主动扩缩实例。AgentCore 资源由平台管理，未声称其与 FaaS 的 4 GiB 配置相同。

FC、Session MCP、AgentCore 的正式组均已完成 213/213，全部成功。完整原生用量计价分别为 ¥1.76753675、¥1.78326245 和 $0.52294396；24 种任务中 18 种 FC 更便宜，6 种 MCP 更便宜。实测阿里总费用未精确相等，这是预先按 Profiling 平衡后实际执行的结果，予以保留。AgentCore 的延迟日志已覆盖全部 213 次成功 Session 终止，未用缺失日志外推费用。

Lambda 当前账户内存上限为 3008 MiB，3072 阶矩阵预检出现原生 `Runtime.OutOfMemory`。用户选择等待至少 4096 MiB 额度再执行，所以正式组派发数为零，AWS 成本交叉尚未验证。恢复步骤见 `results/e3-balanced-four-platform/WORK_STATE.md`；不得以失败费用或缩小任务替代正式组。

最新费用、延迟和任务矩阵见 [报告](../../results/e3-balanced-four-platform/report/REPORT.md)、[结构化汇总](../../results/e3-balanced-four-platform/report/summary.json) 与 [24 任务矩阵](../../results/e3-balanced-four-platform/report/task-matrix.csv)。失败预检、校准及临时镜像传输的原生记录单独保存，不混入 213 请求基线费用。
