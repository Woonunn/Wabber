# LangGraph Data-Analysis Agent

入口是 `experiments.exp1.capture.capture_agent_traces`。使用真实 `StateGraph` 和 `Send`；没有 MetaGPT Harness、预构建 ReAct Agent或预先编写的任务执行计划。

流程为 PLAN → PREPARE_WORKSPACE → DISPATCH → 动态 ANALYSIS 分支 → MERGE → 下一批就绪任务或 REVIEW；REVIEW 的实际模型决策可以完成、失败或进入 REPLAN。计划校验失败也可以进入 REPLAN。每个步骤必须引用原始 Prompt 的文字依据，DAG 必须无环，输出路径不得冲突，重复计算不能作为并行理由。`tool_results` 使用检查重复键的 State Reducer。

`state.py` 保存 request/session/task ID、原始 Prompt、消息、Workspace 引用、输入哈希、计划、Tool 的结构化结果、产物引用、迭代、状态和错误。DataFrame、数组和模型保留在 MCP Tool Runtime 的驻留内存中，或 FaaS 的可恢复状态文件中，不进入 LangGraph State。图表几何证据只保留小规模数值，较长曲线不放进 State。

## 本地运行

先按 Phase 1 README 准备输入及 `wabber-phase1-reference:20261025-cpu-lstm` 镜像。在项目 `.env` 配置 `DEEPSEEK_API_KEY`；不提交密钥文件。

```bash
docker build -t wabber-agent-sandbox:resident -f benchmarks/DataSciBench/nodes/AgentTools/Dockerfile benchmarks/DataSciBench/nodes
export WABBER_SANDBOX_IMAGE="$(docker image inspect wabber-agent-sandbox:resident --format '{{.Id}}')"
export WABBER_DATA_ROOT="$PWD/benchmarks/DataSciBench/data"
export ATSUITE_SESSION_ROOT="$PWD/results/phase2/tool-state"
export ATSUITE_STATE_SERVICE=datasci-agent
export PROVIDER=""
uv run python -m experiments.exp1.capture.local_tool_service --family faas --port 9200
```

在另一个终端执行：

```bash
uv run python -m experiments.exp1.capture.capture_agent_traces --task human_12
uv run python -m experiments.exp1.capture.capture_agent_traces \
  --task human_12 --replay results/phase2/<capture-id>
```

默认配置是 `config/langgraph_experiment.json`。`--output` 必须为不存在的目录，防止覆盖旧执行。结果包括压缩后的完整事件 JSONL、最终控制状态和统一 Trace；ATSuite ATG 按需导出。正式接受的六份真实 Capture 位于 `trace/captured/manifest.json` 索引内；范围为 manifest 中的六个已准备候选，不代表所有目录任务均可执行。

## Tool Runtime 与隔离

原有 `datasci_run_code` 接口保留。它与 AgentTools 共用 `DataSci/kernel.py` 中的 Python 执行与输出捕获入口；旧 FaaS 入口继续流式输出，保留超时前诊断。AgentTools 提供 prepare_workspace、execute_analysis、read_workspace、merge_artifacts、validate_result。

- prepare/merge 为共享状态域的 rw 操作，受 ATSuite 的写入锁保护；execute/read/validate 对已提交状态为 r，不提交 SDK 共享状态快照。
- 每个并行分析拥有独立输出目录、状态输出和不可变分支对象。各分支读取同一代状态，MERGE 检查 generation 和冲突，将 Python 状态按 step ID 分区合并后原子发布新代。共享状态不会由并行分支直接改写。
- MCP 每个 Session 启动一个计算容器和常驻 Python 内核，canonical state 始终在该内核的内存中；每次只读分析从常驻内核 fork，继承相同内存视图，分支对继承对象的修改不影响 canonical state。MERGE 只导入分支增量，并在产物发布后确认内存提交；失败则取消。没有逐调用 canonical 快照恢复、序列化或对象存储提交。独立输出与分支增量是并行通信，不是恢复整个 Session。
- FaaS 每次计算启动独立执行环境，从 SDK 恢复的文件加载 Python 状态；MERGE 将新状态序列化，由 SDK 在写入调用结束时持久化，保留既有多实例写入锁。r 调用只恢复，不提交 canonical state。两者的存储机制各自保留，generation、访问规则和可观察计算结果一致。
- 结果记录 canonical restore/save 是否发生、实际耗时/字节数和分支导出开销；未发生的耗时为 null。SDK 同步时间和 Tool RPC 总时间仍保留。MCP prepare 单独记录 Session 初始化时间，其后复用相同 container_name / resident_kernel_pid。
- 本地 FaaS 风格 RPC 服务可以承载多个 UID；本地 MCP 服务每个进程仅代表一个 Session 环境，不能把不同 UID 指向同一进程来模拟生产多 Session。MCP 客户端配置 endpoint 使用服务 base URL，不重复附加 `/mcp`。
- 计算容器无外部网络、根文件系统只读、非 root、移除 capabilities、no-new-privileges；限制 2 CPU、4 GiB、128 PID、每文件 512 MiB、每分支总输出 1 GiB及超时。只挂载本 Session 的受控输入/状态和执行目录；没有 Docker socket、仓库或凭据挂载。MCP 分支另用 Landlock 限制写入自己的输出与 /tmp，并禁止向父内核发信号或改写命令队列。要求 Linux Landlock ABI >= 6，随仓库固定的 Moby seccomp profile 允许相关系统调用，其余仍按默认拒绝策略。缺少隔离能力时调用失败，不降级。
- MCP 的 2 CPU / 4 GiB 是整个 Session（含并行分支）的预算；FaaS 是每次调用的预算。当前本地执行方式用于检查生命周期，正式策略比较必须同时记录并匹配各自资源配置。MCP 容器在 Session 生命周期内保留，包括 LLM 等待期；Session 退出时释放。分支超时终止分支，已提交内存仍可用；内核退出则必须新建 Session，不自动恢复空状态。
- 输入和依赖在测量前准备。原始输入及依赖产物以只读视图出现在工作目录；新输出写到私有 output_dir。显式跨步骤访问可使用 artifact_path 或 shared[step_id]。state_out 只导出真正需要继续使用的 Python 值。
- pickle/model 反序列化及文件内容校验都在沙箱中执行，前端不反序列化不可信 Python 对象；拒绝路径逃逸、用户输出符号链接、特殊文件、冲突输出和超额数据。运行器创建的只读输入别名须保持原目标，不能用于写入。

`WABBER_CODE_TIMEOUT` 配置执行时限。可信 RPC 前端需要 Docker daemon；生成代码的容器不获得它。当前启动器只在 wlh-190 验证，不可直接用于没有 Docker daemon 的云 FaaS。部署这些 Provider 前需要原生隔离执行器；本轮没有部署云平台。Runtime Adapter 的 FaaS/MCP 接口和原有部署解析器保持兼容，`config/langgraph_atg.json` 提供 ATG 路由配置，不能把配置解析成功当作云端部署验收。

## 模型、Capture 和 Replay

模型由 `ModelClient` 抽象统一管理决策、重试、记录和关闭。`DeepSeekClient`、`OpenAIClient` 和 `OpenAICompatibleClient` 分别构造供应商支持的参数，通过 `model.provider` 选择；不填 provider 时仍使用 DeepSeek。

要接入托管 GPT，在现有运行配置的 `model` 对象中设置（保留原有 Tool 配置）：

```json
{"provider":"openai","name":"gpt-5.4-mini","seed":0,"reasoning_effort":"low","max_tokens":16000,"retries":2}
```

密钥放在项目 `.env` 的 `OPENAI_API_KEY`。该客户端使用 Chat Completions，并将配置中的 `max_tokens` 转成 `max_completion_tokens`；启用推理时不发送 temperature/top_p。配置 `reasoning_effort: "none"` 后可传 temperature/top_p。GPT-6 Astra、GPT-6.1 Sol 等要求 Responses API 才能使用工具的模型不在此 Chat Completions 客户端的工具调用支持范围内。

`gpt-oss-20b` 是开放权重模型，不能通过 `api.openai.com` 调用。已有自部署或第三方 Chat Completions 服务时，使用：

```json
{"provider":"openai_compatible","name":"openai/gpt-oss-20b","base_url":"https://YOUR_INFERENCE_HOST/v1","api_key_env":"GPT_OSS_API_KEY","seed":0,"temperature":0,"top_p":1,"max_tokens":16000,"retries":2}
```

将 `GPT_OSS_API_KEY` 写入 `.env`，并替换 `base_url` 和 `name` 为供应商的实际值；例如 Ollama 使用 `http://127.0.0.1:11434/v1`、`gpt-oss:20b`，无鉴权端点也需设置 SDK 所需的非空占位密钥。服务必须支持 JSON object 输出和原生 function tools。修改的是 `--config` 指定的 JSON 文件，无需修改图或 Tool 实现。不要把密钥写进 JSON 配置或 URL，因为配置会被写入 Capture。

### OpenRouter：gpt-oss-20b

服务器项目 `.env` 使用 `OPENROUTER_API_KEY`，密钥不进入 Git。已提供完整配置 `benchmarks/DataSciBench/config/langgraph_openrouter_gpt_oss_20b.json`，其中模型为 `openai/gpt-oss-20b`、地址为 `https://openrouter.ai/api/v1`，复用 `OpenAICompatibleClient`。默认使用 low reasoning，输出上限 16000 tokens。

固定 CoreWeave 上游并关闭自动回退（`model.extra_body.provider`）：本次实测自动路由至 Groq 时，`submit_result` 的答案字段变成空对象；CoreWeave 已通过真实 JSON、工具执行、最终答案和 Capture/Replay 检查。切换上游后应重新验证这些行为。

在原有本地 Tool 服务运行后执行：

```bash
uv run python -m experiments.exp1.capture.capture_agent_traces \
  --task human_12 \
  --config benchmarks/DataSciBench/config/langgraph_openrouter_gpt_oss_20b.json
```

其它 workload 入口也读取同一个 `model` 配置对象，可直接使用该对象切换模型。原有 DeepSeek 配置继续可用。

### OpenRouter：Qwen3.6-35B-A3B

使用 `benchmarks/DataSciBench/config/langgraph_openrouter_qwen3_6_35b_a3b.json`，模型 ID 为 `qwen/qwen3.6-35b-a3b`，复用 `.env` 中的 `OPENROUTER_API_KEY` 和现有 `OpenAICompatibleClient`。配置固定 PrimeIntellect 上游并关闭自动回退，输出上限 16000 tokens，保留模型默认推理设置。

在原有本地 Tool 服务运行后执行：

```bash
uv run python -m experiments.exp1.capture.capture_agent_traces --task human_12 --config benchmarks/DataSciBench/config/langgraph_openrouter_qwen3_6_35b_a3b.json
```

保存完整请求/响应、原始 Prompt、JSON 决策、Tool 参数/结果、模型标识、Response ID、Token usage、采集 UTC 时间、单调时钟耗时、temperature/top-p/max_tokens、错误、重试与限流状态。返回模型名原样保存，不推断未提供的权重版本。数值随机过程使用 seed；外部模型的可复现性依靠完整 Capture 决策，不能保证再次请求 API 会生成同一响应。

Replay 仍运行相同 LangGraph：LLM 节点使用捕获的决策和延迟等待，不再调用模型 API；Tool 节点调用目标 endpoint，重新进行真实计算和计时。Provider endpoint、物理 UID、Request ID与实际耗时可以不同，业务参数及逻辑分支引用必须相同。检查原始用户消息、每个 Tool 的参数、状态、Python 状态签名和实际文件签名。CSV 规范化到 8 位小数、PNG 使用解码像素、PDF 使用文本/图像；其它文件采用 SHA256，不能忽略不兼容输出。

新出现的失败或不一致立即中止 Replay。Capture 已记录且成功恢复的同一种结构化失败可以重现，随后只能走已记录的 REVIEW/REPLAN；失败分支不会解锁其依赖。FaaS 使用结构化错误响应，MCP 设置 isError；错误输出和诊断保留。ATG 保留依赖和并行边，已通过原 Trace 解析及 FaaS/MCP 路由解析。要求一致性验证的实验使用上述 LangGraph Replay，不依赖旧 Invoker 自动实现本层校验。

## 评测范围

当前 Agent 的独立检查覆盖 human_12、human_15、csv_excel_34、human_142、human_7、csv_excel_29。六个已 profiling 候选与官方 21 项指标的对应关系、额外契约和缺口见 [EVALUATION.md](../tasks/EVALUATION.md) 与 [evaluation_alignment.json](../tasks/evaluation_alignment.json)。这些是自定义 Prompt 检查，不能当作官方 TMC/CR 得分；尤其未完成官方 GT 的准确率/轮廓系数比较和 VLM 评分。最终控制状态中的 evaluation_basis 保留此区分。新增三项的真实模型决策、执行与跨运行时重放见 experiments/exp1/CAPTURE_SIX_VALIDATION.md。

## 检查

```bash
uv run python -m unittest discover -s tests -q
uv run python -m experiments.exp1.data.validate_evaluation_alignment
uv run python -m experiments.exp1.tests.check_agent_lifecycle --image "$WABBER_SANDBOX_IMAGE"
uv run python -m experiments.exp1.tests.check_resident_kernel
uv run python -m experiments.exp1.tests.check_agent_replays --image "$WABBER_SANDBOX_IMAGE"
WABBER_SANDBOX_IMAGE="$WABBER_SANDBOX_IMAGE" PROVIDER="" ATSUITE_RUNTIME=function \
  uv run python -m experiments.exp1.tests.check_agent_sandbox
```

单元测试中的模型和 Tool doubles 只用于控制流检查，不作为真实执行证据。真实验收及失败诊断见 `experiments/exp1/PHASE2_VALIDATION.md`。

新增 Capture 的 Docker 自检使用 `experiments/exp1/tests/check_extended_validators.py`（镜像内 /opt/wabber 导入路径）。Torch checkpoint 使用 weights_only 加载并按数值签名比较，实际 optimizer hook 记录步数和前后权重哈希；这些是执行证据，不是官方 MAE 评分。

## Harness TracingAdapter

Capture/Replay 入口默认启用 LangGraph TracingAdapter。完成后仅默认保存
`trace.json.gz`、`events.jsonl.gz` 和 `result.json.gz`，历史明文 Capture 仍可直接 Replay。
按需生成派生文件：

```bash
uv run python -m atsuite.tracing.export results/tracing-capture atg
uv run python -m atsuite.tracing.export results/tracing-capture spans
```

统一 schema、独立接入示例、状态传输边界和 rho 的缺失值规则见
[Tracing 说明](../../../atsuite/tracing/README.md)；本轮验证见
[TRACING_VALIDATION.md](../../../experiments/exp1/TRACING_VALIDATION.md)。

Adapter 已支持嵌套 StateGraph、原生 interrupt/static breakpoint 以及 checkpoint 续跑。
使用 `TraceCollector.load` 可以加载之前的观测并关联恢复后的执行；接入方式见上述 Tracing 说明。
当前 Capture/Replay CLI 仍按原流程运行，未新增业务 checkpoint 或 Tool Session 的持久化机制。
