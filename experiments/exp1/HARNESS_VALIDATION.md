# 三种 Harness 的同任务本地验证

2026-11-04 使用真实 DeepSeek API 和本地 Docker 完成 DataSciBench `human_15`。
没有部署或调用云平台的 Tool Runtime。完整运行证据保存在
`results/harness-comparison-20261104-final/`；较早的试跑保留在 `results/harness-comparison-20261104/`。

## 接口与业务代码

`TracingAdapter` 提供统一 `ainvoke` / `astream`、请求上下文、调用关联及资源观测。
`LangGraphTracingAdapter` 实现图/路由/checkpoint 解析；`LangChainTracingAdapter` 继承它，
为 `create_agent` 隐藏回调的路由补充实际 tool-call/result ID 证据；
`OpenAIAgentsTracingAdapter` 继承同一个抽象类，通过 SDK 原生 tracing 和 lifecycle hooks 采集。

普通原生 Tool 只写业务函数与框架正常的 Tool 声明，不调用观测接口。
部署到 Wabber Runtime 的 Tool 使用现有 SDK 注册方式；`RuntimeToolset` 根据业务签名生成框架代理，
自动处理 Session、隐藏 UID、关联 Runtime 响应、保留工具错误，并在取消时等待有界 RPC 收尾后释放 Session。
已有 LangGraph benchmark 的模型事件桥接继续保留，不要求业务 Tool 新增记录代码。

## 固定输入与环境

- 原任务要求：按年汇总销售额的红色/黑边柱状图、按类别的 dash-dot 折线图、按类别着色且点大小正比于销售额的散点图。
- 原始 Prompt SHA256：`b0d7b1d7d473dcac930c1a3f5231572c787d4ae70e417dc2622cbbb74990018a`。
- 三次运行使用同一份已准备的 `sales.csv` 引用、同一工具实现、同一固定 Docker 镜像。
- Docker 镜像：`sha256:58d925a5af9a0974132bbabf3d37bf88a00f7e34108212923d3c323894013a39`。
- 模型：`deepseek-flash`；真实 `/models` 响应标识名称为 `DeepSeek-V4.1-Flash`，原响应保存在 `models.json`。
- LangGraph 1.0.5、LangChain 1.2.1、langchain-core 1.2.6、OpenAI Agents SDK 0.20.0、OpenAI Python SDK 2.54.0。
- 三次运行均为真实 Capture；没有用冻结模型决策替代推理，也没有预先提供图表代码。

LangGraph 使用已有 PLAN/DISPATCH/REVIEW 编排。LangChain 和 OpenAI 使用各自原生 agent loop，
共享相同的 workspace 工具说明和原始用户 Prompt，由模型自行决定代码、工具分组与调用。

## 结果

| Harness | 独立校验 | 模型调用 | Tool 调用 | 请求内 Tool 并发峰值 | Workflow 执行耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| LangGraph | 通过 | 4 | 15 | 3 | 41.44 s |
| LangChain create_agent | 通过 | 6 | 7 | 3 | 21.48 s |
| OpenAI Agents SDK | 通过 | 5 | 4 | 1 | 18.32 s |

三者均实际生成并校验 `bar_chart.png`、`line_chart.png`、`scatter_chart.png`。
现有独立检查确认年度销售额为 4800 / 7800，并检查类别总额、柱颜色/边框、折线样式、散点数值/大小/颜色及透明度。
校验结果、产物、原始模型响应、统一 Trace、完整 ATG 和 Tool DAG 均分别保留。

- LangGraph：准备数据、提交，然后三个图表分支并发。第一次散点图校验报
  `Incorrect point sizes or category colors`，Agent 重新规划并重算后通过。
  失败校验实例和第二轮分支保留在 Trace/拓扑中，未过滤失败来缩短图。
- LangChain：准备、读取 workspace，然后三个图表分支并发，汇合后验证通过。
- OpenAI：模型选择在一次 `execute_analysis` 中生成全部三张图，随后提交、验证。
  这次 Capture 的串行形状来自模型选择；SDK 两工具并发及其后续汇合另有离线测试覆盖。

![实际 Tool 拓扑](../../results/harness-comparison-20261104-final/comparison.png)

[矢量图](../../results/harness-comparison-20261104-final/comparison.svg) /
[结构化对比](../../results/harness-comparison-20261104-final/comparison.json)。
图中省略模型/控制节点，边来自调用因果关系的 Tool 投影；红色节点表示失败实例。
耗时是这几次实际运行的记录，包含模型决策、Docker 启动与错误恢复等影响，不作为框架普遍性能排名。

## 回归与边界

75 项相关测试完成，74 项通过，1 项因宿主虚拟环境缺少数值依赖而跳过。
真实 Docker 图表任务由上述三次运行单独验证。命令：

```bash
uv run python -m unittest \
  tests.test_harnesses tests.test_tracing tests.test_tracing_continuation \
  tests.test_runtime_refactor tests.test_resource_confidence tests.test_logical_state \
  tests.test_workload tests.test_langgraph_agent -q

uv run python -m experiments.exp1.capture.compare_harnesses \
  --output results/harness-comparison-new \
  --image sha256:58d925a5af9a0974132bbabf3d37bf88a00f7e34108212923d3c323894013a39
```

新测试覆盖普通业务函数、并发与汇合、Runtime CPU/状态元数据自动关联、用户回调、
模型/业务错误、并发请求隔离、LangChain checkpoint 续跑、SDK 嵌套 Runner、敏感内容关闭及 RPC 取消收尾。
新增文件通过 Ruff 导入/未使用符号检查，工作区 diff 通过空白检查。

Runtime/worker 的实际资源证据优先于客户端 fallback。普通进程内 Tool 的 CPU 区间包含同进程并发工作，
RSS 是进程高水位，不能当作 Tool 独占内存，也不能将多个重叠区间简单加总。
OpenAI SDK Session 透传给应用持有，但目前每次 Runner 调用使用独立 Collector，不合并历史 Trace。
此次没有验证云部署，也没有宣称新的两个 harness 已实现原 LangGraph benchmark 的严格 Capture/Replay 校验。
