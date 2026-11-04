# E3 十分钟四平台真实测试

当前实验协议已收敛为 Fixed（单一 Provider）与 Online＋Exploration 两组，不含 Static、独立纯 Online 或财务预算。计价统一复用 ATSuite；本文中的旧策略计划、账单等待状态和测量记录保留为历史证据，现行入口见 [README.md](README.md)。

2026-11-03 12:45 UTC 账单刷新：AWS CUR 已交付 345 行，但 Usage 最新结束于 07:00 UTC，与本轮 09:36–09:52 UTC 的正式流重叠行数为 0。阿里两组执行池的当前日累计税前费用为 FC ¥0.10725、Session MCP ¥0.76065；这些是混有其他实验的部分日账单，不是本轮 213 请求总成本。AgentCore 原生用量现已匹配 213/213 个 Session、75,809 个样本，观测到 1.986509 vCPU·h、81.364600 GB·h；用量日志不能替代最终费用归属。[最新账单与用量汇总](../../results/e3-realtest/cost-refresh-20261103T1245/summary.json)、[刷新后的逐请求证据](../../results/e3-realtest/cost-refresh-20261103T1245/evidence/report.json)。

路由验证准备度：已重新通过 34 项路由/E3 本地检查。云端三策略尚未启动；当前 Fixed 执行配置不包含在线所需的初始模型、路由/预算/执行边界、limits、Static 映射和负载设置。当前 Router 的 occupied 仅统计 router.active，需要接入原生保留 Session 占用，避免请求完成后仍占云端配额却被视为可用。补齐后使用同一 213 请求分别运行 Static、Online、Online+Exploration，沿用相同评估器缓存、调度、Replay 和超时；初始模型与 Static 映射只使用预检/既定 Profiling 数据，不用完整正式基线结果进行事后挑选。[准备度检查记录](../../results/e3-realtest/routing-readiness-tests-20261103T1245.log)。费用未归属不妨碍功能、成功率和延迟验证，成本节约结论继续保持未知。

以下为运行结束时的原始测试结果与当时的账单/用量快照；其后交付的数据以上述刷新记录为准。

已完成 24 个任务 × 4 平台预检（96/96 成功），以及同一封存 Trace 的四组 Fixed 基线（每组 213 条，共 852 次尝试、811 次成功）。成本最优 Provider 是否随 Workload 变化仍未知；厂商费用尚不能完整归属。

正式运行窗口：2026-11-03T09:36:56.220158+00:00 至 2026-11-03T09:52:07.055345+00:00。请求发送窗口为原始 600 秒，剩余时间用于排空真实长请求。

Manifest SHA-256：`0a5ddfcc5773bc5248b4677f94181e9df238284977dad12c617264b93ef15b9b`。213 条原始 Copilot 到达记录、Prompt、24 个模板及 Capture 未修改；DataSciBench/TravelPlanner/ClaudeCodeReview/ScientificComputation 分别为 64/51/53/45 条。1,519 个封存文件的哈希验证通过。

| Fixed Provider | 成功/尝试 | 成功率 | 全部尝试均值 s | p95 s | p99 s | 成功请求均值 s |
|---|---:|---:|---:|---:|---:|---:|
| AWS Lambda | 213/213 | 100.00% | 49.582 | 111.910 | 279.042 | 49.582 |
| AWS AgentCore | 213/213 | 100.00% | 53.987 | 122.566 | 281.064 | 53.987 |
| Aliyun FC | 213/213 | 100.00% | 49.003 | 112.074 | 276.967 | 49.003 |
| Aliyun Session MCP | 172/213 | 80.75% | 49.909 | 127.773 | 293.217 | 61.492 |

延迟包含实际提交后排队、Session 初始化、原始 Replay LLM idle、真实云端 Tool、独立业务验证及客户端关闭。全部尝试包含失败；阿里 MCP 的快速失败会拉低均值，成功样本的任务组成也与其他平台不同。不能把失败请求解释为延迟收益。

以下为各 Workload 的成功请求平均延迟（秒），括号内为成功/尝试数。

| Workload | AWS Lambda | AWS AgentCore | Aliyun FC | Aliyun Session MCP |
|---|---:|---:|---:|---:|
| DataSciBench | 10.948 (64/64) | 17.153 (64/64) | 10.996 (64/64) | 15.082 (64/64) |
| TravelPlanner | 57.068 (51/51) | 61.740 (51/51) | 55.926 (51/51) | 83.616 (36/51) |
| ClaudeCodeReview | 113.454 (53/53) | 116.971 (53/53) | 112.541 (53/53) | 136.862 (39/53) |
| ScientificComputation | 20.818 (45/45) | 23.406 (45/45) | 20.376 (45/45) | 38.293 (33/45) |

四组调度器均提交 213 条、time_scale=1、无丢弃；实际到达窗口为 600000.338–600001.083 ms。单进程并行四流存在可测调度迟到：各组平均 59–186 ms、最大约 1.007 s，另存完整原始/实际到达表。延迟从实际提交计时，未将这段提交迟到隐去为零。每条请求的 Tool 均执行于固定 Provider；原生 Session 不跨请求共享，也没有运行中跨 Provider 迁移。

阿里 Session MCP 的 41 次失败全部发生在新 Session 初始化，均为 HTTP 429：TravelPlanner 15、Review 14、Science 12、DataSci 0。限流期间原生 API 实测 E3 池有 64 个 Active Session、64 个实例，配置 reservedConcurrency=64；DataSci 使用独立执行池。当前 MCPClient.close 只关闭 HTTP 客户端，不发送原生 Session 终止，因此请求结束后仍按空闲策略占用资源。配额与保留 Session 叠加是有观测支持的解释；本次未保存 429 响应体，无法确认更细的厂商错误码。尾部核验已将 172 个成功 Session 全部匹配为 Expired。

此行为与[阿里 Session 隔离生命周期](https://www.alibabacloud.com/help/en/functioncompute/session-isolation-configuration)、[会话资源回收规则](https://www.alibabacloud.com/help/en/functioncompute/cookie-affinity-feature-1)一致。MCP Session 的资源生命周期不能用单次 Tool duration 替代。

所有策略使用固定原生弹性：Lambda 3008 MiB、并发额度 64；阿里 E3 两池 2 vCPU/3072 MiB，DataSci 两池 2 vCPU/3008 MiB，minInstances=0、reservedConcurrency=64；Session idle=300 s、最大生命周期=900 s。AgentCore CPU/内存由平台管理，未填造固定资源规格。镜像、资源、生命周期策略在运行前后读取核对一致，阿里弹性/隔离/配额另行核对；未启用主动扩缩容或容量搜索。

本轮使用真实 LangGraph Agent 和真实远端 Tool，严格重放已录制的 LLM 决策与 idle，不再次调用 LLM API。Travel 每条结果仍运行未修改的官方检查器，官方模块在独立进程中缓存，加载时间单独记录；Review 使用三个固定真实上游项目和注入缺陷 PR，Science 使用独立数值参考与状态读回，DataSci 使用原有业务验证。预检每模板每平台一次，正式流每策略一次；不是多轮独立重复试验。

实际费用目前全部保留 null。Lambda 的 1,758 次 Tool 调用、FC 的 1,758 次、阿里 MCP 的 1,380 次成功 Tool 调用均已按云端 Request ID 匹配原生日志；这些只证明观测到的调用用量，不是完整费用。初始化失败、Session 保留、对象存储与伸缩相关费用仍需厂商账单归属；FaaS 对象保留/清理费用没有被当成零。AgentCore 本轮已交付 5,334 条可匹配 USAGE 样本，覆盖 80/213 个原生 Session；其余 133 个及最终生命周期覆盖仍未知，缺失不记为零。

AWS 已配置私有、小时级、含资源 ID 的 CUR 导出，当前尚无交付文件；Cost Explorer 当前日快照标记为 estimated 且为账户级汇总，资源级接口尚未启用。阿里当前日账单尚未提供这批执行的可归属费用。旧 Pilot、预检与正式流共用服务，不能把账户或资源整日账单全部分配给正式流。[AgentCore 原生日志可能延迟](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-runtime-metrics.html)，[CUR 首次交付也有延迟](https://docs.aws.amazon.com/cur/latest/userguide/what-is-cur.html)。本轮不报告最佳固定 Provider、成本交叉或节约百分比。

结果证据：

- [24×4 预检](../../results/e3-realtest/preflight/summary.json)、[逐次尝试](../../results/e3-realtest/preflight/attempts.json)
- [正式四策略统计](../../results/e3-realtest/formal/summary.json)、[调度结果](../../results/e3-realtest/formal/scheduler-results.json)、[完整性核验](../../results/e3-realtest/integrity.json)
- [Workload×Provider 矩阵](../../results/e3-realtest/formal-evidence-tail/workload-matrix.csv)、[24任务×Provider 矩阵](../../results/e3-realtest/formal-evidence-tail/task-matrix.csv)
- [逐请求 Profile 与原生用量关联](../../results/e3-realtest/formal-evidence-tail/request-usage.jsonl)、[用量及账单快照](../../results/e3-realtest/usage-formal-tail/)、[Session 尾部核验](../../results/e3-realtest/ali-session-final-join.json)
- [固定策略执行入口](fixed_baselines.py)、[用量关联入口](evidence.py)

下一步先将原生 Active Session 与保留占用纳入路由准入，再用相同逻辑请求运行 Static Workload-Aware、Online Routing、Online+Exploration；若修改 SDK 生命周期或原生策略，须新建协议版本并重跑四组基线。本轮未运行这三种路由策略，未验证 Wabber 相对最佳固定 Provider 的成本收益。六项新增检查通过；Ruff 在执行边界捕获远端任意异常的两处保留 BLE001 例外。
