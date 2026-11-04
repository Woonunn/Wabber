# E3 状态与计算覆盖补充

本轮是用户指定的任务集设计与收益可行性诊断，云端验证总上限 ¥30。
原 24 个任务、213 条原始到达和已有结果保留。新的任务集只有在业务验证、严格跨部署 Replay
和原生用量计价完成后才封存；不能把按预检设计的任务集包装成未参与选型的泛化测试集。

## 任务来源

- DataSciBench：复用原始 `human_7`、`human_12`、`human_142`、`csv_excel_34`、
  `csv_excel_29`、`human_15` 的 Prompt、真实输入和成功 Agent Capture。
  包含真实交易数据挖掘、模型与聚类中间结果，以及原 Capture 已有的反馈修改。
  不扩增原始数据，不增加重复训练。
- ScientificComputation：补充固定种子的 2560、3072 阶稠密多响应逆问题，每种规模两个种子。
  SVD 因子留在 Tensor Workspace 中，并用于求解 16 组响应；使用独立 LU 求解与原方程残差验证。
  输入矩阵是公开标明的合成数值问题，并非外部实测科学数据。保留两个原有轻量任务。
- ClaudeCodeReview：在现有真实项目 Fixture 上补充 SymPy 1.13.3，固定上游提交
  `c870d17c2638f5061800e344b72ff80086a0b41d`。
  注入可复现的多项式系数/重数、矩阵特征值重数/符号回归。
  Agent 实际运行原生多项式或矩阵测试，要求 base 通过、head 失败；独立检查缺陷位置与已完成测试记录。
  这些是注入回归，不是历史上游 PR。保留两个原有轻量 Review。
- TravelPlanner：保留原六个官方验证集任务和原 Capture。

## 实现与复现

复用既有 LangGraph Agent、TracingAdapter、ToolClient 和 SDK 状态管理。
新增工具仅提供存储 SVD 因子的求解复用，以及批准范围内的 Git ref 测试执行与结果读取。
大型输入通过不可变 NPZ 文件提供，数组不会塞进 Prompt 或工具返回值。
MCP 在 Session 内保留状态；FaaS 每次调用恢复对象存储快照并同步更新。

```bash
git clone --depth 1 --branch 1.13.3 https://github.com/sympy/sympy.git results/e3-sympy-polys-upstream
ln -s e3-sympy-polys-upstream results/e3-sympy-matrices-upstream
.venv/bin/python -m experiments.e3.stateful \
  --output results/e3-stateful/scientific --sizes 2560 3072
.venv/bin/python -m experiments.e3.review_projects --sources results \
  --output results/e3-stateful/review --expanded --compute
```

完整原始证据、失败尝试、候选声明、运行脚本和操作者费用预约位于 `results/e3-stateful/`。
Review 准备和测试使用固定依赖：pytest 8.3.5、SymPy 1.13.3、mpmath 1.3.0、Hypothesis 6.131.9。
`scientific_factorization` 和 `review_test_execution` 原生事件记录纯计算/测试时间；
结果中不混入时间字段，因此不会为了跨平台 Replay 而比较运行耗时。

批量工具调用只有在访问类型明确为只读或无状态时才并发；缺少访问声明时串行执行，
避免并发提交状态写操作后让 FaaS 在分布式锁上产生不必要的计费等待。

非思考模式的原生 Tool Capture 强制选择工具，最终答案通过 `submit_result` 提交，
避免模型把函数调用包裹成自由文本后产生错误 JSON。思考模式沿用自动选择，兼容服务端限制。

## 云端协议与费用

只验证阿里 FC 与 ATSuite 已有的 Session MCP 后端（配置名 `ali_agentrun`，实际为 FC 的
SESSION_EXCLUSIVE/MCP_STREAMABLE 适配器）。不增加云平台或实例扩缩容控制。
两种形式使用相同 CPU、内存、磁盘与执行超时。

已有 DataSciBench 专用池保持 2 CPU、4 GiB、10 GiB 磁盘、240 秒 Tool 超时、
Session idle 60 秒/最大 900 秒。新 E3 专用池也使用 2 CPU、4 GiB、10 GiB、240 秒，
但保持前一轮 E3 的 Session idle 300 秒/最大 900 秒作为故障兜底。
按用户最新修正，正常任务完成独立验证后立即调用原生清理接口，保留关闭请求 ID 与响应证据；
不等待 idle timeout。清理发生前的真实 LLM 等待仍计入 Session 内存时间。
策略在配对执行前固定并读取原生配置。与原 3 GiB E3 以及不同 Workload 池之间的差异必须披露。

费用复用 `atsuite.analysis.api.price_metrics`，按原生调用 ID、Session ID 关联用量，
Session 计入初始化到实际清理完成的完整生命周期。不得用仅 Tool 活跃时间代替驻留费用。
Session 空闲期不收 CPU 费用，内存仍收费，10 GiB 临时磁盘免费。
原生 MCP 流式响应的 `durationMs` 在本次 SVD 中只有 3.22 ms，但完整的
`invokeFunctionLatencyMs` 为 17,588.37 ms，且与运行时实际计算一致。
因此 CPU 活跃区间使用原生完整请求起止时间的并集，避免首包时长漏计和并发区间重复计费。
明细保留于 `BILLING_OBSERVATION.md` 与原始 SLS 文件；不把这些用量计价声称为结算账单。
原生 DELETE 后 Session 会从列表移除。关闭回执因此保留 Session ID、原生请求 ID、状态码与时间，
费用优先使用原生初始化到原生 DELETE 完成的区间重建生命周期，包含初始化且没有假定的保留尾段。
操作者预约只限制本轮 ¥30 验证开销，不重新给路由器加入财务预算。
LLM Capture 的实际缓存命中/未命中 token 和失败调用也单独留存，Replay 不调用 LLM API。

## 最终配对证据

24 个任务各完成一次 FC 和立即清理 Session 的 MCP 执行，48/48 通过独立业务验证和严格 Replay。
每对使用同一个 Capture，原有 DataSci 真实校验失败后修改的过程也保留。
24 个 MCP 请求均有成功关闭回执，原生用量关联初始化到 DELETE 完成的时间；最终原生 Active Session 为零。
矩阵位于 `results/e3-stateful/final/cost-matrix.{json,csv}`，完整请求费用明细和关闭证明在同目录。

| Workload | FC 更便宜的任务数 | MCP 更便宜的任务数 |
| --- | ---: | ---: |
| DataSciBench | 4 | 2 |
| TravelPlanner | 6 | 0 |
| ClaudeCodeReview | 6 | 0 |
| ScientificComputation | 2 | 4 |

大型科学任务 MCP 单次约 ¥0.0070–0.0089，FC 约 ¥0.0228–0.0343，降幅约 66.9%–75.1%。
例如 3072 阶 seed1103：FC ¥0.032225 / 148.92 秒，MCP ¥0.008910 / 47.46 秒。
FC 最大快照为 636,249,051 字节，累计状态传输约 3.39 GB；收益主要来自避免反复序列化、恢复和同步大型因子。
不能把 FC 含状态同步的 140.73 秒 Tool 时间称为纯矩阵计算：原生 SVD 日志的纯计算为 11.61 秒（FC）、
13.04 秒（MCP），同一 Capture 的 LLM 等待为 4.38 秒。全部八个重任务的 16 次纯计算证据在 `final/pure-compute.json`。

四个重 Review 实际云端测试计算约 33–36 秒（MCP），LLM 等待约 9–12 秒；它们仍是 FC 更便宜。
Session 初始化、实际 LLM 等待中的内存和关闭成本会抵消状态复用收益，长计算不自动意味着 MCP 获胜。
DataSci Adult 的实际快照约 64 MiB，MCP 约省 18.3%；Retail 约 158.6 MiB、累计状态传输约 1.74 GB，
本次 MCP 只低 0.3%，不能据单次测量宣称稳定优势。

Review 旧 Capture 含文件读取锁超时，实际云端读取成功导致严格重放失败。
四个任务使用修正状态调度后的新本地真实 Capture（`review-captures-v6`），任务、Prompt、输入与已知缺陷不变；
差异记录在 `capture-revision.json`。旧失败、启动失败、等待超时和中断批次全部归档并计入费用范围，
不混入最终成功比较。旧版错误计价和只计首包时长的临时报价已作废。

48 次最终执行的用量计价合计约 ¥0.4123，不是本轮全部开销。
操作者账本另包含旧批次、失败、初始验证和 LLM/存储/日志预留，保守预约合计约 ¥10.31，低于 ¥30；
这仍是基于原生用量和明确余量的费用上界，不能称为已结算账单。

新的 `results/e3-stateful/trace` 继续使用原 Copilot 连续 10 分钟、213 个请求、seed9；
四类请求数量仍为 64/51/53/45，原始到达 ID、时间精度和突发间隔不变，每类覆盖六个任务。
原有 24 任务和 213 请求文件保留。新任务集参与了显式设计和 Profiling，不能作为未参与选型的泛化测试集。

115 项相关测试通过。新增 AWS 镜像仅在本地准备，没有新增 AWS Workload 调用；
本轮仅验证阿里两后端，没有重新运行四平台或正式 213 请求路由对比，不能据此声称 Wabber 已取得长期收益。

## 按两侧额外成本平衡任务频次

用户要求在正式执行前调整 213 个请求的任务分布，使选 FC 时在 MCP 优势任务上多花的钱，
与选 MCP 时在 FC 优势任务上多花的钱相等。设每任务请求数为 `n_i`，两种部署的已测成本为
`f_i`、`m_i`，目标是 `sum(n_i * max(f_i-m_i, 0)) = sum(n_i * max(m_i-f_i, 0))`。
这等价于两种固定部署的预计总成本相等，不要求两类请求数相等。

`results/e3-stateful/trace-balanced` 是单独封存的新版本；原 `trace` 与早期原始 Trace 保留。
任务、输入、Prompt、Capture、213 个到达 ID 与全部时间字段不变，每类仍覆盖六个任务，
请求数仍为 DataSci64、Travel51、Review53、Scientific45。按 seed9 只更换七个请求的任务，
产生 161 个 FC 更便宜任务请求、52 个 MCP 更便宜任务请求。

两种固定部署预计总成本分别为 FC ¥1.784058299574、MCP ¥1.784058298150，
两侧额外成本差约 ¥0.000000001424，低于预先固定的 ¥0.00000001 容差。
整数请求数不保证数学上的精确相等，因此保存完整 Decimal 费用与实际残差，不伪造零误差。
完整每任务频次、更换的七个请求及输入成本矩阵哈希见 `balance.json`；这些费用只用于离线设计，
不会进入请求到达时的路由上下文。

```bash
.venv/bin/python -m experiments.e3.balance \
  --sequence results/e3-stateful/trace \
  --matrix results/e3-stateful/final/cost-matrix.json \
  --output results/e3-stateful/trace-balanced
```

这是按已有阿里 FC/Session MCP Profiling 成本设计的平衡实验，不是自然任务频次或独立测试集；
也不表示 AWS 两个后端成本已平衡。真实并发运行的初始化、容量、费用和延迟仍须云端测量。
本次只生成请求流，没有发起新云端调用，也没有修改任务以扩大成本差异。
