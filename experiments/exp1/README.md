# DataSciBench 实验准备

已完成 Phase 0 和 Task 1.1 Prompt 导入；尚未开始输入数据准备、LangGraph、DeepSeek、到达时序或正式 E1 实验。
本地隔离测试的数值输入用于检查基础设施正确性，不代表真实 DataSciBench 工作负载。

## Tool Response 和执行记录

`RuntimeAdapter.invoke()` 返回 `InvocationResult`（SDK 中的 `ToolResponse`），保留真实输出类型，以及
`status`、`error`、`request_id`、`run_id`、`provider_request_id`、`client_elapsed_ms`、
`execution_elapsed_ms` 和 `provider_metadata`。无法测得的服务端耗时为 `null`。
MCP `isError` 和 JSON-RPC 错误按错误返回，超时保持 `timeout`。
旧 FaaS `invoke()` 的 `(request_id, is_stateful)` 和旧 MCP `invoke()` 的 request ID 返回约定保留；完整响应使用 `invoke_result()`。

Replay 使用唯一 run/request ID；JSONL 事件包含 UTC 时间、单调时钟、输出、错误和 traceback。
每次 Replay 的结果文件在成功或失败时均保存，不替换服务进程 stdout。
缺失指标使用 `null`；Excel 保留空值并单列 coverage。费用公式仅标为基于观测的估算，不当作账单。

## MCP：Session 内存驻留

已确认的部署前提是：每个 MCP Session 拥有独立执行环境，Session 生命周期内保留内存。
`ATSUITE_RUNTIME=mcp` 必须在 SDK 导入前设置，各 MCP Dockerfile 均声明此值；本轮不构建或推送镜像。
`ATSUITE_MCP_STATELESS` 只涉及 HTTP/MCP 传输，不改变应用 Session 的内存语义。

正常调用直接使用驻留的注册对象及稳定 Workspace：

- 不初始化对象存储，不读取/写入本地或云快照。
- 不逐调用复制、恢复、重置对象，也不在调用结束提交状态。
- 不使用跨 Session 文件锁、云锁或历史全局对象兼容锁。
- `session_scoped=True` 在 MCP 返回原对象；FaaS 才使用调用级代理。
- 一个执行环境绑定一个逻辑 UID；不同 UID 必须路由到各自环境，误路由会报错。

旧 `__atsuite_state_snapshot` 参数仍支持显式的首次初始化，以兼容现有 Invoker 的初始状态注入。
仅在首次有状态调用前应用一次；运行中的 Session 再次注入会报错，不恢复普通调用。

当前 MCP handler 使用 `asyncio.to_thread()`，允许同 Session 内 Tool 并行。因此在 SDK 内按状态域使用进程内读写锁：

| 模式 | 同一 Session、同一 domain 的并发规则 | 状态提交 |
| --- | --- | --- |
| r | 允许多个读取并行；与冲突写入互斥 | 无 |
| w / rw | 独占，序列化冲突写入 | 原地更新驻留状态，无快照提交 |
| stateless | 不申请状态锁 | 无 |

不同 domain 可以并行；Tool 必须正确声明实际访问的状态域及读写模式。
锁保证调度规则，不提供任意 Python 对象的事务回滚。普通 Python 异常不会自动重置 Session。

## FaaS：恢复与持久化

FaaS 保留逐调用对象副本及状态恢复，成功的 w/rw 调用提交快照；r 不提交，stateless 不恢复或提交。
稳定 Workspace 允许后续调用使用已保存的路径。失败调用不提交快照。

快照继续兼容 `service/uid.json`，因此同一快照里的写入保守地按 service + uid 串行化，防止不同 domain 合并时丢失更新。
同实例使用文件锁；多实例使用对象存储条件创建的写锁，提交采用原子覆盖，不先清空再追加。
云存储初始化失败时有状态调用明确报错，不退化为仅进程内保存。

云写锁不自动过期。写入进程异常终止可能留下 `service/uid.lock`；确认原写入已停止后才能清理。
`ATSUITE_STATE_LOCK_TIMEOUT` 默认 30 秒。
MCP 和 FaaS 的一致性指支持的数据与成功调用的可观察结果、文件连续性及访问规则，不要求相同存储机制或通用事务回滚。

## DataSci 执行方式及必要性

MCP 环境中的 `datasci_workspace` 保留原对象，不再把进程级对象视作跨用户共享缺陷。
数值计算仍在 ATSuite 托管 Tool 内完成，Agent Harness 不执行这些代码。

| 环境 | Python 执行方式 | 解决的问题 |
| --- | --- | --- |
| MCP | 每个 Session 保留一个 Python worker 与完整 namespace | 持续保留函数、自定义对象、RNG、文件句柄等原生内存状态；避免代码改变服务进程 cwd/stdout，并允许终止超时计算 |
| FaaS | 每次调用创建子进程，恢复/导出既有可序列化变量状态 | 允许同进程处理不同请求时隔离 cwd/stdout，并将成功状态持久化 |

MCP 后续调用不重新创建 worker，不通过 JSON 重建 namespace。仅首次可从显式初始状态建立 namespace。
输出通过独立 RPC fd 返回；stdout/stderr（包括原生库输出）只在 Tool 子进程中捕获，不重定向服务进程的流。
变量、工作目录及文件保持连续；普通异常保留 Session 的原生 Python 状态，并返回错误与已产生的输出。

`ATSUITE_DATASCI_TIMEOUT` 默认 240 秒。MCP 硬超时会终止 worker 并保留已产生输出；后续调用明确报错，必须新建 Session，不悄悄重新初始化内存。
FaaS 超时不提交持久化状态。数据与依赖应预先准备，Tool 不内置业务 API 调用。
此子进程用于执行和超时隔离，不是恶意代码安全沙箱。

## 本地测试及证据边界

在 `wlh-190:/home/wlh/Wabber` 执行：

```bash
ATSUITE_TEST_ARTIFACT_DIR=results/phase0/<唯一标识> \
  uv run --group aws --group gcp python -m unittest discover -s tests -q
~/.nvm/versions/node/v22.4.0/bin/node --test web/trace_viewer/metadata.test.mjs
```

测试清楚区分：

- `MCPSameProcessAccessTests`：模拟同一个 Session 的进程内锁、对象身份、无恢复/提交及首次初始化规则，不作为生产隔离证据。
- `MCPSeparateEnvironmentTests`：本地创建 16 个独立 MCP 服务进程，各自有独立 worker、Workspace、UID 和真实 HTTP endpoint，验证连续状态与不串状态；这是本地进程环境验证，不冒充云平台隔离验证。
- FaaS：同一真实 HTTP 服务的 16 个 UID，以及独立实例目录与共享条件存储模型下的写入冲突测试。后者是本地存储模型，不是云存储验收。

结果与撤销/保留清单见 `PHASE0_VALIDATION.md`。本轮只在 wlh-190 测试，不部署云资源，不采集账单。


## Phase 1：真实输入与离线参考 Profiling

`tasks/catalog.json` 现保留 **172 条 Prompt**：固定上游 Commit `84ef3d4d94d7362a5149cf14a73dc168fc4f2f33` 共 222 条；13 条需审核、45 条上游缺文件，重叠 3 条，合计标记 55 条。保留未触发标记的 167 条，加回已经补齐数据且通过断网 Docker 验证的 `human_12`、`human_142`、`csv_excel_34`、`human_7`、`csv_excel_29`；其余 50 条排除。`human_15` 本来就在 167 条中，六个实测候选全部保留。

导入器默认应用此筛选规则，记录排除 ID/原因；五项例外同时核对上游 Commit 和 Prompt SHA256，原文变化后不会沿用旧验证结论。仅解析 Git 中的标准 JSON，保留 Commit、Git blob 和 SHA256；不执行上游脚本。172 条是筛选后的候选数量，其中 6 条已完成 Profiling，166 条未完成运行验证。

`tasks/datasets.lock.json` 固定真实输入的来源、许可说明、字节数、SHA256 和必要格式转换。下载发生在准备阶段。再次准备时验证已有文件；缺失文件从同一来源重建并核对锁定哈希，上游内容变更会失败。原始数据位于 `benchmarks/DataSciBench/data/`，不纳入 Git。零售 XLSX 从固定 Commit 的公开原始文件镜像取得，SHA256 与镜像元数据一致，Git blob SHA1 也与另一独立归档一致；UCI 原始来源和 CC BY 4.0 归属保留在锁文件。

本次实测任务、耗时及证据索引见 [PHASE1_VALIDATION.md](PHASE1_VALIDATION.md)。

`tasks/manifest.json` 保留 6 个已在 wlh-190 Docker 中通过完整参考执行和输出校验的任务。自动提取结果及 missing_from_upstream 保留为来源诊断；本地是否齐备以 readiness.inputs 的哈希核验为准。其余 166 条仍在 catalog 中待验证。`csv_excel_29` 在取得 Hugging Face 数据集访问权限后补回，其他被排除任务不会自动恢复。准备脚本只处理 catalog 中的任务。

```bash
uv run python -m experiments.exp1.data.import_prompts \
  --repo results/upstream/DataSciBench \
  --revision 84ef3d4d94d7362a5149cf14a73dc168fc4f2f33
uv run python -m experiments.exp1.data.prepare_datasets

docker build -t wabber-phase1-reference:20261025-cpu-lstm \
  -f experiments/exp1/profile/Dockerfile experiments/exp1/profile
uv run python -m experiments.exp1.profile.profile_tasks
# 用上一步实际输出的目录替换下列路径；只有三次全部成功才发布活动清单。
uv run python -m experiments.exp1.profile.profile_tasks --publish results/phase1/<batch-id>
uv run python -m unittest discover -s tests -q
```

参考环境基于 wlh-190 已有的 DataSci 镜像，仅在本机构建补充依赖。每次测量固定最终 image ID、2 CPU 配额、4 GiB 内存、单线程数值库、seed=0、无 GPU；容器设置 `--network none`、只读根文件系统、只读输入与唯一输出目录。镜像构建阶段可以安装依赖，测量中不下载数据或安装依赖。镜像标识、软件包版本和参考代码哈希写入结果；相同标签重建后 image ID 变化，应视作新环境。

人工编写的参考实现位于 `profile/reference_tasks.py`，由现有 `datasci_run_code.execute_from_dict` 调入 ATSuite Tool 的 Python worker 执行全部数值计算。这是对真实任务可行性与资源需求的 Profiling，**不是 LLM Agent 轨迹，也不保证模型解题成功率**。

每个任务独立运行三次，记录：

- 以 `perf_counter` 测量各实际步骤耗时（包含步骤自身文件 I/O）；CPU user+system 时间、单核和配额归一化利用率。
- 10ms 采样各步骤 RSS；Linux `ru_maxrss` 记录整个计算进程峰值，包含 Python、库、校验和仪器开销。不是容器 cgroup 峰值；ATSuite 父进程与文件缓存不在该 RSS 中。
- 真实中间状态以 pickle protocol 5 序列化的字节数、哈希和单独计时的序列化开销；这是参考解法的共享状态大小，不能充当 SDK JSON 快照大小、FaaS 传输成本或账单。另记工作目录文件大小。
- 每步 UTC 时间戳、唯一 run/request ID、状态、失败栈和超时/OOM 诊断；缺失记录保留 null，失败不发布清单。
- 原 Prompt 对应的文件和数值校验：Adult 官方留出集指标、模型重载重现预测、聚类统计；Gini 两种独立公式；信用卡/零售全部 itemset support 与 rule confidence 直接对交易重算，以及销量、收入和聚类一致性；PNG 解码、PDF 解析和报告内容检查。

数据清洗与分析参数显式记录，不添加合成行、无意义循环或等待。信用卡的 k=2..8 重复拟合用于 Prompt 要求的 Elbow 选 k，拟合结果被复用。零售的频繁单项预筛选遵循 Apriori 的 downward closure，不以任意 top-N 截断替代完整交易集。

分类使用三次计算总耗时中位数：<5 秒为轻量，5–15 秒为中等，>=15 秒为重计算；峰值 RSS >=512 MiB 另标内存较高。清单还给出最重要步骤的实测耗时，5–45 秒是选择偏好，不给轻量原任务填充计算。子任务 DAG 表示基于不可变输入可独立处理的工作；当前测量按序执行，未声称测得并行加速。宿主机页面缓存与其他用户负载未隔离，因此耗时不是专用硬件上的保证值。

数据分批准备时，可将同一环境下、任务不重叠的完整批次合并发布，禁止混合环境或重复任务：

```bash
uv run python -m experiments.exp1.analysis.aggregate \
  results/phase1/<batch-a> results/phase1/<batch-b>
```

所有原始证据位于 `results/phase1/<batch-id>/`：输入校验后产生独立容器、事件 JSONL、真实输出、输出哈希、资源记录、错误和环境信息。该目录不纳入 Git；活动清单保留证据路径。未部署 AWS 或任何云资源，未推送镜像或生成模拟账单。


csv_excel_29 的官方输入位于 `zd21/DataSciBench` 数据集固定修订 `f15d3e4043498febc26d7bcb7562b7e344b514f2` 下的 `DataSciBench-data/csv_excel_29/data.csv`。需要先在 Hugging Face 登录并获得该数据集自身访问权限；仅登录账号可能仍返回 403。令牌只在数据准备阶段使用，不写入镜像、清单或容器环境。原文件共 200 行，data_source_type=3（上游人工编写数据），不称作真实生产时间序列。

CPU 镜像固定 `torch==2.5.1+cpu`，安装只在构建阶段进行。LSTM 参考代码独立放在 `profile/lstm_reference.py`，通过现有 ATSuite Tool 执行；其他任务不加载 PyTorch。使用 1–5 月 151 天的 Value1，按时间留出最后 28 天选择 epoch，再用全部 151 天重训；递归预测 6 月 1 日至 7 月 19 日的 49 天。检查真实权重更新、损失下降、模型重载复现、独立 NumPy 门公式、精确输出日期及 PNG。另用独立断网容器改写全部未来 Value1 和未使用列，验证模型及预测不变。该任务当前约 1.2 秒，归为轻量任务；报告中保留真实预测误差及末值基线，不以格式校验代替预测质量比较。

PyTorch CPU 安装方式参照官方说明：https://pytorch.org/get-started/previous-versions/ 。加入 PyTorch 后六项在同一个新镜像 ID 中重新测量，不将不同镜像环境的耗时直接合并。


## Phase 2：LangGraph Agent

已完成六项候选真实任务的 DeepSeek + LangGraph Capture/Replay（仅 wlh-190 Docker）。使用方法见 [Agent README](../../benchmarks/DataSciBench/langgraph_agent/README.md)，数值、并行/共享状态证据和限制见 [PHASE2_VALIDATION.md](PHASE2_VALIDATION.md)。新增三项的验证见 [CAPTURE_SIX_VALIDATION.md](CAPTURE_SIX_VALIDATION.md)。完整 Capture 索引位于 benchmarks/DataSciBench/trace/captured/manifest.json。

## Phase 3：Copilot 抽样 Session 到达流量

已完成整周下载/清洗、固定窗口选择与原速 Open-loop 轻量验证。复现见 [PHASE3_README.md](PHASE3_README.md)，实际计数、到达误差、丢失率及限制见 [PHASE3_VALIDATION.md](PHASE3_VALIDATION.md)。不将抽样到达率当作完整产品 QPS。

默认流量窗口已改为 10 分钟，并支持 15/60 分钟比较；独立绘图脚本为 analysis/plot_windows.py，选择依据与放大图索引见 [WINDOW_SELECTION.md](WINDOW_SELECTION.md)。保持 time_scale=1.0，缩短的是连续观测区间。

图表按类别保存在 figure/traffic_windows/（流量窗口图及其分箱、哈希记录）和 figure/task_profiles/（六个候选任务的资源特征图）。任务图使用现有 profile，不重新执行任务；运行 uv run python -m experiments.exp1.analysis.plot_task_profiles 生成 PNG/PDF。两类绘图脚本均保留已有输出，重绘时需指定新的 --output 路径。

四个固定 provider 与 Wabber 的请求序列累计美元成本图见 [绘图说明](analysis/CUMULATIVE_COST.md)。逐请求数据模板位于 `data/cumulative_cost.template.csv`，待实测数据填入后生成 PDF/SVG/PNG。

Phase 5 task-composition preparation, local Docker replay and paired stage analysis: [PHASE5_README.md](PHASE5_README.md).
