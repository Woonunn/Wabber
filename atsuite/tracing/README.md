# Harness tracing

`TracingAdapter` 只负责将 Harness 的真实调用和因果关系送入每请求一个的 `TraceCollector`。
三个入口共享 `TracingAdapter` 抽象类与 `TraceCollector`：`LangGraphTracingAdapter`、
复用它的 `LangChainTracingAdapter`（LangChain v1 `create_agent`），以及 `OpenAIAgentsTracingAdapter`。
验证版本为 LangGraph **1.0.5**、LangChain **1.2.1**、langchain-core **1.2.6**、OpenAI Agents SDK **0.20.0**（`uv.lock`）。
Tool 的固定依赖仍使用 `benchmarks/DataSciBench/nodes/AgentTools/requirements.txt`，Docker 环境仍按原 README 固定镜像 ID。

```python
from pathlib import Path
from atsuite.tracing import TraceCollector
from atsuite.tracing.langgraph import LangGraphTracingAdapter

collector = TraceCollector(request_id="request-1", session_id="session-1", directory=Path("results/my-trace"))
adapter = LangGraphTracingAdapter(collector)
async for state in adapter.astream(compiled_graph, inputs, {"max_concurrency": 3}):
    pass  # state 与原 graph.astream(..., stream_mode="values") 相同
print(collector.snapshot()["summary"])
```

## 原生 LangChain / OpenAI Agents SDK

业务 Tool 使用框架本身的声明方式即可，不需要调用 Collector、记录 Trace 或采集 CPU/RSS。
Wabber 的接入点是运行 Agent 的位置；两个适配器都保留原生循环和返回值：

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from atsuite.tracing.langchain import LangChainTracingAdapter

@tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right

agent = create_agent(model, tools=[add])
result = await LangChainTracingAdapter(collector).ainvoke(
    agent, {"messages": [{"role": "user", "content": "Calculate 2 + 3"}]}
)
```

```python
from agents import Agent, function_tool
from atsuite.tracing.openai_agents import OpenAIAgentsTracingAdapter

@function_tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right

agent = Agent(name="Calculator", model=model, tools=[add])
result = await OpenAIAgentsTracingAdapter(collector).ainvoke(agent, "Calculate 2 + 3")
print(result.final_output)  # 原生 RunResult
```

每个请求新建 Collector/Adapter。LangChain 的 `config`、`context` 和 checkpoint 续跑沿用 LangGraph 接口；
OpenAI 的 `run_config`、`hooks`、`context`、`session`、`max_turns` 透传给 `Runner.run`，用户 hooks 仍执行。
OpenAI `astream` 产生一个最终 `RunResult`，不声称是 SDK 流式事件接口。
Wabber 只额外注册一个按请求上下文分发的 SDK trace processor，不替换已有 exporter；
应先配置 SDK exporter，再创建 Adapter，且必须启用 SDK tracing。仅用 DeepSeek 做本地采集时，可在创建 Adapter 前
显式调用 SDK 的 `set_trace_processors([])`，避免 SDK 默认 exporter 尝试上传到 OpenAI。
OpenAI 当前不合并跨次 Runner 调用的历史 Trace；SDK Session 由应用持有，各次运行使用新 Collector。
`trace_include_sensitive_data=False` 同时关闭 Wabber 的模型/工具业务内容记录，保留拓扑、ID 和资源观测。

LangChain 内部默认关闭部分路由回调。Adapter 通过实际 `tool_call_id`、工具结果、graph namespace 和 superstep
补齐隐藏路由的依赖，支持工具批次并发、汇合及原生 checkpoint 恢复；无法建立来源证据时仍输出 diagnostics。
OpenAI 工具依赖其生成调用的模型实例，保留并行 sibling，不根据完成时间推断串行边。

## 通过 Wabber Runtime 执行业务 Tool

已有 SDK Tool 注册函数只保留业务实现和正常的 Tool/状态声明。`RuntimeToolset` 自动创建框架原生代理，
隐藏 `uid`，管理 Runtime Session、校验参数并调用相同的 MCP/FaaS 接口：

```python
from atsuite.runtime import create_runtime_adapter
from atsuite.tracing.tools import RuntimeToolset

runtime = create_runtime_adapter("local", "faas")
runtime.connect({"targets": {"calculator": "http://127.0.0.1:9200"}})
async with RuntimeToolset(runtime, "calculator") as tools:
    # registered_add 是 Tool 服务已注册的业务函数 / UnifiedAgentTool。
    agent = create_agent(model, tools=tools.langchain_tools([registered_add]))
    result = await LangChainTracingAdapter(collector).ainvoke(agent, inputs)
    # OpenAI 使用 tools.openai_tools([registered_add])，再通过对应 Adapter 运行。
```

Runtime 调用自动继承 Wabber request ID；响应中的状态传输、worker CPU/RSS、错误和物理调用 ID
自动关联到原生 Tool span，不额外生成重复的代理 span。MCP 保留 Session 状态，FaaS 沿用 SDK 每次恢复、写调用提交的机制。
业务代码不需要了解这些观测逻辑。

普通进程内 Tool 自动记录 harness 进程的 CPU 时间区间和 RSS。它们包含同进程其他线程/并发调用，
CPU 不是可加总的独占 Tool 用量，峰值 RSS 是进程生命周期高水位。Runtime 提供实测值时优先使用 Runtime，
隔离 worker 提供实测值时优先使用 worker；各层证据分别保留，不重复相加。没有把 RPC 客户端用量冒充远端用量。

`atsuite.tracing.topology.tool_topology(trace)` 投影出只包含 Tool 实例的因果 DAG，去除模型/控制节点并约简传递边；
保留失败和重试实例。完整 Trace 与 ATG 导出继续保留模型及控制节点。

三种 Harness 的真实 DeepSeek V4.1 Flash + 本地 Docker 同任务验证入口：

```bash
uv run python -m experiments.exp1.capture.compare_harnesses \
  --output results/harness-comparison \
  --image sha256:<包含现有 DataSci sandbox_worker 的固定镜像 ID>
uv run python -m unittest tests.test_harnesses tests.test_tracing tests.test_tracing_continuation -q
```

验证使用原始 `human_15` Prompt，模型自行生成代码，复用已有数值/样式校验器；三个入口分别真实请求模型、
执行 Docker 并验证三张图表。输出原始响应、Trace、ATG、Tool DAG、图表、校验结果以及 `comparison.svg/png/json`。
LangGraph 使用既有 PLAN/DISPATCH/REVIEW 图，另两个使用各自原生 agent loop；一次任务的差异不代表框架普遍性能优劣。

编译图可以共享；每个独立请求使用新 Adapter/Collector 和独立目录，同一请求的 checkpoint 续跑复用原 Collector。
已有 callbacks 会保留，同一个 Adapter 的续跑须串行执行。
原生 LangChain model/tool 回调自动捕获。直接调用其他 SDK 不会自动生成这些回调；DataSciBench
通过现有 `TraceRecorder.emit` 桥接 DeepSeek 和 `ToolClient`，不更改 Agent 节点业务流程。

## 数据与依赖

- 默认仅保存 `trace.json.gz`：gzip level 6 压缩的统一 Trace，保留 request/session、全部 spans、诊断、恢复记录及 summary。
- DataSciBench 同时保存 `events.jsonl.gz`、`result.json.gz`，内容字段和 Replay 校验规则保持不变。
  事件日志在运行中逐条写入 `events.jsonl`，结束后原子压缩，成功后才移除明文日志；中途退出或压缩失败时保留原日志。
- 历史明文 Capture/Trace 仍可读取；同名压缩文件与明文文件并存时优先读取压缩文件，损坏时直接报错。
- `spans.jsonl` 和 `trace.atg.json` 改为按需导出。后者仍可用 `atsuite_sdk.workflow.Trace.from_file` 读取：

```bash
uv run python -m atsuite.tracing.export results/my-trace spans
uv run python -m atsuite.tracing.export results/my-trace atg
```

分析脚本可用 `atsuite.tracing.storage.read_text(Path("results/my-trace/trace.json"))` 透明读取新旧格式。
新运行不自动生成派生格式，也不自动迁移或删除历史文件；按需导出的明文文件由调用方管理。
压缩只改变日志存储，不改变 Runtime 实测状态字节、LLM 等待间隔或 rho 的计算。

Span 的 `parent_id` 表示调用包含关系，`dependencies` 表示前置执行实例；循环的每次执行都有独立 ID。
时间使用进程单调时钟纳秒，`duration_s` 使用秒。LLM 输入/响应、Tool 输入/结果、错误、重试、usage、
Runtime request ID/Session UID、state lifecycle 和可观测 worker CPU/RSS 保存在 span 中。
`resources=null` 表示没有测量。原始 JSONL 另外保留 UTC 时间与完整 API 决策证据。

Workflow 边来自编译 StateGraph 的静态边、实际执行的条件路由返回值、Command/Send、
callback 中的 superstep/task path。Send 使用 LangGraph 的确定性 task path 次序对应真实发送者，
普通边匹配前一 superstep 的实际执行实例，barrier 边保留跨 superstep 的待汇合来源并在消费后清空。
同节点内已结束调用形成后续调用的执行前置关系；重叠调用保留并行。
ATG 用零耗时控制节点表达包含关系，LLM/Tool 时间只计一次。

支持 LangGraph 能发现的嵌套 StateGraph：通过 `get_subgraphs(recurse=True)` 取得图定义，按实际 checkpoint namespace
隔离同名节点、条件路由、superstep 与 Send 来源；`parent_id` 保留实际子图包含关系。
子图中的 `Command.PARENT` 按返回父图的控制流记录，不当作业务异常。
`parallel_workflow_peak` 统计叶子 Workflow 节点的并发峰值，避免把父图容器和内部节点重复算成并行。
当前仍未实现 deferred 节点及运行时无法发现的动态子图的完整依赖追踪。
遇到无法重建的来源会留下 diagnostics，并停止发布完整性与 rho 结论；不要把这种 Trace 当作完整 ATG。
回调只有控制依赖证据，不声称追踪 Python 对象内部的数据血缘。

## 嵌套子图与断点恢复

按原生 LangGraph 方式将编译子图放进父图节点，再为最外层图配置 checkpointer。Adapter 不创建业务状态数据库，
也不改变原生暂停、重试和 pending writes 的执行语义。

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# child_builder / parent_builder 是正常的 StateGraph；父图也可以包含多层、并行子图。
parent_builder.add_node("agent", child_builder.compile())
graph = parent_builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "request-1"}}

# 动态 interrupt() 或 compile(interrupt_before=[...], interrupt_after=[...]) 会结束当前采集段。
async for state in adapter.astream(graph, initial_state, config):
    pass

# 原生动态中断用 Command(resume=...)；静态断点和异常重试用 None。
async for state in adapter.astream(graph, Command(resume="approved"), config):
    pass
```

在中断位置，最后的 session span 为 `interrupted`，`summary.complete/successful` 为 false，rho 为 null。
恢复会追加新的 session 段及实际节点执行实例；重跑节点通过 `metadata.retry_of` 和依赖边关联上一次尝试。
之前成功的 sibling/pending write 复用原 span，不凭 checkpoint 伪造新的 LLM/Tool 调用。
中断节点在恢复时从头执行，其前置 LLM/Tool 若真正重跑，仍按实际次数累计；Adapter 不提供业务副作用去重。

跨 Adapter 或进程续跑时加载原 Trace：

```python
collector = TraceCollector.load(Path("results/my-trace"))
adapter = LangGraphTracingAdapter(collector)
# graph 的 checkpointer 必须由应用恢复；继续使用原 thread_id 和对应的 checkpoint。
async for state in adapter.astream(graph, None, config):
    pass
```

`TraceCollector.load` 仅恢复观测 JSON，不反序列化/恢复 Agent 业务对象。
`InMemorySaver` 只适用于同进程；跨进程必须由应用提供持久化或恢复后的 checkpointer，Tool Runtime Session/UID 也由应用保持。
`trace.json.gz` 原子替换，保存 thread、各层 checkpoint ID、pending task 与 resume/interrupt/error 控制记录。
续跑前核对这些记录；错误 thread、旧 checkpoint、未经采集的中间 resume 不会静默并入当前 Trace。
需要回退/分叉到其他历史 checkpoint 时，应使用匹配的历史 Trace 副本和独立输出目录。

如果直接从已有 checkpoint 开始而没有历史 Trace，会保留缺失历史诊断，停止发布完整 workload 结论。
若进程被强制终止，Trace 与业务 checkpoint 可能无法同时提交；两者不匹配时必须补齐对应的观测记录，不能声称完整恢复。
`workflow_duration_s` 为各执行段耗时之和，不包含两段之间的人为等待；LLM/Tool 统计包含所有真实尝试。
并发峰值按执行段分别计算，不跨进程拼接单调时钟。

## 状态传输与 rho(G)

Harness 不估算状态大小。Runtime 在实际 I/O 成功后按 payload/bundle 的字节数累加：

| Runtime `state_sync` 字段 | 测量边界 |
| --- | --- |
| `load_bytes` / `save_bytes` | SDK snapshot 对象读取/提交；含 JSON/base64 编码后的实际对象内容 |
| `branch_read_bytes` / `branch_write_bytes` | AgentTools 不可变 branch bundle 的读取/发布；含状态、声明产物和归档开销 |

MCP 不执行 canonical snapshot 恢复/提交，对应计数为实测 0；branch 通信仍单独计数。
FaaS 每次调用恢复，写调用提交。重复读取同一对象重复累计；不使用文件最终大小代替累计传输。
传输失败且无法确定实际量时该方向保留 null，已有旧 Runtime 未返回计数也保留 null。
这里只计对象 payload，不计 TLS/HTTP 头、重传、锁对象等协议/协调流量。
内核 `canonical_restore_bytes`、`canonical_save_bytes`、`branch_export_bytes` 是另一个序列化层的诊断，
保留原值但不再次加入总量，避免把同一份状态的序列化与存储传输重复计数。

`state_read_bytes=load_bytes+branch_read_bytes`，写方向同理。
未启用 checkpoint 时，`S_G` 是所有 Tool 读写量之和；`D_G` 是所有 LLM span 的 `duration_s` 之和。
启用 checkpoint 后，session span 的状态读写字段预留给 checkpointer 存储 I/O，目前无法实测，保持 null。
这部分不能用 checkpoint JSON 的长度或 Tool Runtime 字节数代替，因此即使执行 DAG 完整，
`state_measurements_complete` 仍为 false，完整 `state_transfer_bytes` 和 rho 为 null。
`observed_state_transfer_bytes` 继续累计各段中已实测的 Tool I/O，包含重跑发生的真实重复传输。
DataSciBench 沿用原 Replay 的完整模型调用等待间隔，包含显式重试/退避；并行 LLM 间隔相加，不做时间并集。
`rho_kb_s=S_G/1000/D_G`（十进制 KB/s），只在运行成功、Trace 完整、所有 Tool 计数已知且 D_G>0 时提供。
缺失时 `state_transfer_bytes` 和 rho 为 null；`observed_state_transfer_bytes` 仅为已观测部分，不能冒充完整 S_G。
`tool_elapsed_s` 是客户端 Tool RPC 耗时之和，包含传输/状态同步；worker 执行耗时和资源信息在 Runtime 原始证据中。

Analyzer 的 RunRecorder 是既有 ATG Replay 的观测接口，依赖整数节点和 Provider 配置；本轮不重构该接口。
新 Collector 负责 Harness 调用实例，复用现有 ToolResponse/SDK 观测、ATG schema 和 Trace 解析器。
后续 Harness 只需实现统一 span/依赖采集，不需要复用 LangGraph 回调类。

## DataSciBench 示例

按 [Agent README](../../benchmarks/DataSciBench/langgraph_agent/README.md) 准备输入、固定镜像并启动 Tool RPC，
在仓库 `.env` 设置 `DEEPSEEK_API_KEY` 后，原入口默认启用 Adapter：

```bash
uv run python -m experiments.exp1.capture.capture_agent_traces \
  --task csv_excel_34 --output results/tracing-capture
uv run python -m experiments.exp1.capture.capture_agent_traces \
  --task csv_excel_34 --replay results/tracing-capture --output results/tracing-replay
uv run python -m unittest tests.test_tracing tests.test_tracing_continuation tests.test_langgraph_agent -q
```

本轮实测结果见 [验证记录](../../experiments/exp1/TRACING_VALIDATION.md)。
未加入在线学习、路由、扩缩容、数据库或新的分布式采集服务。
