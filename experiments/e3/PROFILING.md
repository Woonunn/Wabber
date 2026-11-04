# E3 四平台配对 Profiling 实测

当前实验协议已收敛为 Fixed（单一 Provider）与 Online＋Exploration 两组，不含 Static、独立纯 Online 或财务预算。计价统一复用 ATSuite；本文中的旧策略计划、账单等待状态和测量记录保留为历史证据，现行入口见 [README.md](README.md)。

最终 cohort：`results/e3-cloud/pilot-v3`。四 Workload × 四平台 × 三次重复，48/48 通过。实际费用尚不能完整归属，所有成本单元保留 `null`，不能据此报告最佳固定 Provider 或节约率。

## 固定设计与证据

任务及 Capture 在云端测量前确定于 `results/e3-cloud/pilot-tasks.json`，最终封存于 `pilot-sequence-v4/manifest.json`。使用原始输入、相同逻辑请求、实际 Tool 执行和固定 Capture 的模型决策及 LLM idle；Replay 不调用付费 LLM API。Travel 使用 easy 的有效 Capture；medium/hard 原始验证失败另存，未被包装为成功请求。Review Pilot 使用 ItsDangerous，小型数值 Pilot 使用 normal-8-1，DataSci 使用 bcb1011 的既有校准环境参考及原始 live 谱系。任务目录与原始 Trace/Profile 可从 attempts.json 逐项追溯。

每轮将同一任务发送到四个平台并行执行，平台内顺序重复三次。没有显式预热、强制冷启动或主动实例调整；先前兼容性 cohort 可能留下自然温实例，因此本轮不是独立冷启动实验。原生策略固定，MCP Session 在请求内保留状态，FaaS 每次调用恢复/提交状态。运行窗口为 2026-11-03T07:16:05.077642+00:00 至 2026-11-03T07:24:56.566139+00:00。

## 总延迟矩阵

单位秒，每格为三次 Harness 总耗时的均值（最小–最大），包括 Session 初始化、Replay idle、Tool、独立业务验证及清理。所有单元成功率均为 3/3。

| Workload | AWS Lambda | AWS AgentCore | Aliyun FC | Aliyun Session MCP |
|---|---:|---:|---:|---:|
| DataSciBench | 8.289 (7.054–10.464) | 14.859 (12.938–16.023) | 8.297 (6.810–11.162) | 11.863 (11.552–12.042) |
| TravelPlanner | 78.816 (75.256–84.715) | 76.689 (75.890–77.921) | 75.635 (75.032–76.412) | 92.493 (91.303–93.873) |
| ClaudeCodeReview | 24.698 (24.568–24.916) | 26.835 (26.389–27.631) | 24.328 (24.200–24.400) | 42.422 (42.188–42.687) |
| ScientificComputation | 9.374 (9.336–9.417) | 10.482 (10.318–10.711) | 8.861 (8.825–8.880) | 27.270 (27.182–27.405) |

完整统计（包括算法计算的 p95/p99）在 `pilot-v3/matrix.json`，逐项实测和资源指标在 `pilot-v3/attempts.json`、`matrix.csv`、`measured-metrics.json`。每格只有三次，p95/p99 不能作为稳定尾延迟结论。

## 初始化与执行范围

下表均为三次均值，单位秒。初始化取 run_start 到 tool_session_open 的实际事件间隔；Agent 时间取既有 Profile，覆盖 Graph 与独立验证。Tool 时间是 RPC 累加，遇并行调用会重叠，不能直接与 LLM idle 相加。Runtime 时间是 SDK 内工具墙钟时间累加。

| Workload | Provider | Session 初始化 | Agent 时间 | Tool RPC 累加 | Runtime Tool 累加 |
|---|---|---:|---:|---:|---:|
| DataSciBench | aws_lambda | 0.010 | 8.090 | 4.724 | 3.794 |
| DataSciBench | aws_agentcore | 0.174 | 14.525 | 11.198 | 8.922 |
| DataSciBench | ali_fc | 0.012 | 8.089 | 4.749 | 3.617 |
| DataSciBench | ali_agentrun | 3.240 | 8.514 | 5.183 | 4.827 |
| TravelPlanner | aws_lambda | 0.002 | 78.693 | 10.612 | 0.536 |
| TravelPlanner | aws_agentcore | 0.043 | 76.545 | 9.027 | 0.513 |
| TravelPlanner | ali_fc | 0.010 | 75.495 | 4.245 | 0.464 |
| TravelPlanner | ali_agentrun | 18.417 | 73.994 | 2.800 | 0.755 |
| ClaudeCodeReview | aws_lambda | 0.001 | 24.364 | 2.309 | 0.764 |
| ClaudeCodeReview | aws_agentcore | 0.048 | 26.532 | 5.778 | 0.133 |
| ClaudeCodeReview | ali_fc | 0.002 | 24.004 | 1.748 | 0.309 |
| ClaudeCodeReview | ali_agentrun | 18.516 | 23.678 | 1.385 | 0.293 |
| ScientificComputation | aws_lambda | 0.001 | 9.235 | 1.251 | 0.594 |
| ScientificComputation | aws_agentcore | 0.043 | 10.325 | 2.328 | 0.004 |
| ScientificComputation | ali_fc | 0.005 | 8.718 | 0.717 | 0.114 |
| ScientificComputation | ali_agentrun | 18.774 | 8.420 | 0.431 | 0.064 |

Aliyun Session MCP 的新 Workload 约 18 秒额外开销主要出现在 tool_session_open 之前，不能归为清理或矩阵/Git 计算。实际事件显示其科学请求最后一个 Tool 到请求封存仅约几十毫秒。Session 内 Tool RPC 与 Agent 执行可比 FaaS 更快，但请求总耗时受初始化影响。Travel 的官方评估子进程也占据总延迟，因此这不是纯云端执行延迟表。

## LLM、工作流与状态

| Workload | 固定 LLM idle 秒 | Agent 调用 Tool 数 | 执行特征 |
|---|---:|---:|---|
| DataSciBench | 3.292608 | 4 | 原 PLAN/PREPARE/ANALYSIS/MERGE/REVIEW；保留原环境参考 |
| TravelPlanner | 52.803384 | 5 | 一次真实批次查询 5 个 Tool，然后生成行程并执行官方独立验证 |
| ClaudeCodeReview | 22.551636 | 13 | 13 次文件/Git 检查；项目绑定由 Harness 提前完成 |
| ScientificComputation | 7.931845 | 5 | 创建两个张量、求逆、乘积、读回，并独立解线性方程/检查残差 |

Review 的项目绑定调用属于执行 Trace，但不计入上表模型调用的 13 个 Tool。Travel 批次保留真实重叠调用；Profile 的 parallel_workflow_peak 描述 Graph 节点重叠，不能用它把批次 Tool DAG 改写成串行。

| Workload | Lambda 状态传输字节 | AgentCore | Ali FC | Ali MCP |
|---|---:|---:|---:|---:|
| DataSciBench | 248716 | 122880 | 248716 | 122880 |
| TravelPlanner | 0 | 0 | 0 | 0 |
| ClaudeCodeReview | 24661 | 0 | 24661 | 0 |
| ScientificComputation | 33148 | 0 | 33164 | 0 |

新 Workload 的 MCP 状态传输为 0，表示 SDK 未向对象存储同步驻留状态；不表示其内存状态为 0。Travel Pilot 的五次查询本身无 Notebook 写入，不能将它当作 Notebook 状态写回覆盖。新 Workload 的逻辑 state_bytes/read/write 尚缺完整观测，保持 null。DataSci 的逻辑读/写为 15/12 次。SDK CPU 计数只覆盖调用线程，遗漏原生线程/子进程，tool_cpu_s 保持 null；measured-metrics.json 保留可用 subtotal，不拿它作为完整 billable CPU。

## 部署与原生策略

三个新 Workload 共用 E3Tools 镜像；DataSci 复用既有 E2 部署。共享镜像会加载 Travel 数据库，因此其内存/冷启动开销影响 Review 和科学请求。本次测量适用于该固定部署，不能当成每种 Workload 单独精简依赖后的最优性能。四个新镜像各 521 个 SDK/Tool/数据库/Git 文件经过哈希核对，凭据不进入打包数据；审计为 `fixed-image-audit.json`，固定镜像为 `fixed-images.json`。AgentCore 使用 ARM，其他镜像 AMD64。

| 执行池 | 内存 | CPU / 并发 | 固定原生策略 |
|---|---|---|---|
| AWS Lambda | 3008 MB | CPU 随内存；每实例并发 1，保留请求配额 64 | 磁盘 2048 MB，Tool timeout 240 秒 |
| AWS AgentCore | 平台管理，实际值未知 | 平台管理，不能把请求的 CPU/内存当实配 | Session idle 300 秒 / max lifetime 900 秒 |
| Aliyun FC | E3Tools 3072 MB；DataSci 3008 MB | 2 CPU；每实例并发 1，保留请求配额 64 | 磁盘 10240 MB，min_instances 0，timeout 240 秒 |
| Aliyun Session MCP | E3Tools 3072 MB；DataSci 3008 MB | 2 CPU；Session 独占，每实例 1 Session | idle 60 秒 / TTL 900 秒，min_instances 0 |

原生读回在 `deployed-*.json`；既有 DataSci 的本轮资源/镜像重新读回在 `datasci-deployments/deployed-*.json`。Ali Session 使用现有 FC Session-exclusive MCP adapter，不代表新增 AgentRun API 后端。所有策略后续必须复用同一组配置；Wabber 不调整实例数量。

## 实际用量与成本缺口

| Workload | AWS Lambda 成本 | AWS AgentCore 成本 | Ali FC 成本 | Ali MCP 成本 |
|---|---|---|---|---|
| DataSciBench | 未归属 | 未归属 | 未归属 | 未归属 |
| TravelPlanner | 未归属 | 未归属 | 未归属 | 未归属 |
| ClaudeCodeReview | 未归属 | 未归属 | 未归属 | 未归属 |
| ScientificComputation | 未归属 | 未归属 | 未归属 | 未归属 |

新池 `evidence-v3/receipts.json` 保存 Lambda 392 条、AgentCore 0 条、Ali FC 299 条、Ali MCP 359 条用量日志；DataSci 对应 `evidence-datasci-v3` 为 80/0/53/79 条。日志窗口含初始化和实例事件，行数不是请求数，必须按原生 request/session ID 关联后才能分配费用。AgentCore 空日志不能解释为零用量或零费用。Lambda REPORT 含真实 billed duration 和 Max Memory Used，Ali 请求日志含 duration/memory usage，但这些本身不是覆盖全部保留/存储费用的账单。

AWS 返回账户级估算日汇总；Ali 返回账户级实际日账单，目前没有本批 E3 runtime 可直接匹配的归属行。没有用价格表投影或模拟费用填补缺口，也未把日账单整体计入本次 Pilot。完整 execution、Session retention、storage 和平台伸缩成本尚不可得；保留成本、最便宜 Provider、费用优势交叉和 Wabber 节约率均未知。

## 兼容性 cohort 与解释

| Cohort | 成功数 / 48 | 保留的失败原因 |
|---|---:|---|
| pilot-v1 | 21 | MCP DataFrame 输出未统一；容器 Git ownership/UTC 格式；数值 BLAS 舍入差异 |
| pilot-v2 | 45 | 三次 Travel 本地独立 VERIFY 超时，不能记作后端 Tool 失败 |
| pilot-v3 | 48 | 最终同逻辑 Capture 全部通过 |

前两轮结果、日志和源码封存另存，没有删去失败后拼凑一个成功 cohort。共享 SDK 统一 DataFrame records 输出；Git 仅信任当前 Fixture 并统一时间；科学结果允许严格数值容差，仍独立验算。随后将同步 Session 清理移到线程并关闭评估遥测，最终 cohort 通过；这不能单独证明此前超时的唯一原因。

就本轮总延迟而言，Ali FC 在三个新 Workload 均值最低；DataSci Lambda 与 Ali FC 只差约 0.008 秒，样本不足以形成可靠优势交叉。Agent 执行与含初始化的总时间具有不同排序，说明路由目标和状态保留政策影响选择。Pilot 尚未给出成本侧跨 Workload 的优势交叉证据，不以此改写任务或增加人工计算。

运行结束后四个 E3 服务关闭 SDK 外部状态 bootstrap，保持镜像和原生弹性配置；读回见 `post-pilot-environment.json`。该更新发生在最终测量之后，不重写测量时的部署凭据。原生 Session TTL 与对象存储仍可能产生后续保留费用。

后续正式实验仍需扩大预先封存的任务覆盖、获得完整可归属账单，并分别运行七个策略；当前报告不提供长期收益结论。
