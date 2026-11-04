# E3 两组云端测试结果

规则修订说明：下列云端结果来自修订前的置信度门槛。后续已移除正常比价的可靠性过滤：低经验但费用预测可比较的候选仍参与目标排序，成本单目标不要求未使用的延迟预测。修订在原 11 个误选决策快照中全部选回 FC；这是决策回归，不是新的云端成本或高负载结果。[回归证据](../../results/e3-confidence-fix/decision-regression.json)。原实验使用的 router.py 已另行保留，原测量未修改。

正式测试只比较 Fixed Aliyun FC 与 Online＋Exploration。两组各执行相同 213 条、600 秒原始到达流，并正常排空长请求；未运行 Static 或独立纯 Online，没有财务预算限制。

| 方案 | 成功/请求 | 平均延迟 s | p95 s | p99 s | ATSuite 计价 CNY |
|---|---:|---:|---:|---:|---:|
| 固定 Aliyun FC | 213/213 | 42.285 | 107.658 | 270.056 | 0.096721 |
| Online＋Exploration | 213/213 | 44.730 | 111.507 | 272.434 | 1.821061 |

按统一计价口径，Online＋Exploration 相比固定 FC 的费用增加 1.724340 CNY（1782.80%）。

总共 29 次主动探索；这些请求已观测计价合计 1.153782 CNY。这是探索请求的实际计价子集，不是探索的因果增量成本，也没有伪造未执行 Provider 的反事实费用。

| Workload | FC | Lambda | AgentCore | Session MCP |
|---|---:|---:|---:|---:|
| DataSciBench | 51 | 4 | 3 | 6 |
| TravelPlanner | 40 | 5 | 3 | 3 |
| ClaudeCodeReview | 46 | 3 | 2 | 2 |
| ScientificComputation | 38 | 2 | 1 | 4 |

路由记录发现 11 次非探索的替代后端选择：AgentCore 1、Session MCP 6、Lambda 4。示例中 FC 的预测费用更低且准入可行，却因预测置信度不足被排除，算法从更昂贵的可靠预测中选择。这是当前置信度门槛的实际失误，费用上升不能全部归因于主动探索。详见[选择诊断](../../results/e3-two-arm/selection-diagnostics.json)。

预检的 24 个模板按相同 ATSuite 口径均为 FC 最便宜，四类 Workload 的平均费用也均为 FC 最低。该任务池缺少明显的 Provider 成本优势交叉。LLM idle 与固定 300 秒 Session 保留期会持续产生耦合池费用，短 Tool 计算在 FaaS 上很便宜；十分钟请求窗口也难以摊销探索成本。结果不支持为了节约百分比事后更换任务或增加无意义计算。

实现与计价：

- E3 入口只保留 fixed/exploration；同一请求只选择一次池。两组采用相同 LangGraph、Capture、官方检查器缓存、并发 64 和原生弹性策略。
- 模型初始化只使用之前 96 次配对预检，未使用正式基线结果，未按相同 task_id 查表。完成反馈更新画像与 Provider 性能统计；原生成本标签在日志到齐后归集，运行时并非所有 Provider 都具备即时完整计费反馈。
- 阿里只读 Active Session 观测与本地未释放预约共同参与准入。AgentCore 运行时缺少可用的原生 Session 列表接口，当前单所有者运行池使用原生 idle 策略的保守本地保留预约；不调整任何实例数量、配额或伸缩参数。
- 所有公式由 atsuite.analysis.api.price_metrics 调用既有 pricing.py 生成；E3 仅转换数据。Lambda 使用原生计费时长/内存/请求/临时盘，AgentCore 使用原生 CPU-hours/Memory GB-hours，阿里 Session 使用原生完整生命周期。AgentCore 尾部未到齐时不以零替代。
- 上述费用是实际原生用量按 ATSuite 单价计算，含已观测运行/保留费用及请求内 Trace 状态估算。请求结束后对象长期保留、未观测存储 API、日志/镜像平台费用、厂商免费额度、折扣与税费不包含在该口径中；不是已结算发票或账户全部基础设施费用。跨币种沿用封存 ECB 参考，仅供比较。

完整性：新协议与旧封存流的 213 条原始到达、任务和 Capture 不变；两组的 source Manifest 一致，实际到达不缩放，原生资源/镜像/生命周期在前后读取核验一致。首轮启动曾因压缩决策文件读取问题中断，12 条已完成开发请求保留在 aborted-startup，不混入正式两组；压缩文件归属回归测试已补充。只有单轮实验，延迟差异不能当作稳定显著性结论。

证据入口：

- [两组费用与性能对比](../../results/e3-two-arm/comparison.json)、[请求/配置完整性](../../results/e3-two-arm/integrity.json)
- [预检成本矩阵](../../results/e3-two-arm/preflight-cost-matrix.json)、[初始化来源](../../results/e3-two-arm/initial-final/initialization.json)
- [Fixed 结果](../../results/e3-two-arm/fixed/result.json)、[探索组结果](../../results/e3-two-arm/exploration/result.json)、[原始路由与反馈](../../results/e3-two-arm/exploration/routing.jsonl)
- [固定组计价](../../results/e3-two-arm/fixed-pricing-latest.json)、[探索组计价](../../results/e3-two-arm/exploration-pricing-latest.json)、[完整原生用量快照](../../results/e3-two-arm/metering/)
- [源协议](../../results/e3-two-arm/protocol.json)、[新增启动/归属检查](../../results/e3-two-arm/selection-checks.log)、[本地验证](../../results/e3-two-arm/completed-checks.log)
