# Context and execution prediction

生产链路为：执行前上下文 → 工具集合与 Harness 硬过滤 → Embedding 召回 → 执行相关性重排 → 联合 G/S/D 情景 → 当前平台模型 → 路由。没有关键词分类、意图标签距离、全局历史或资源相近回退。

## 身份与上下文

```python
from atsuite.workload import WorkloadModel, describe

context = describe(
    prompt=prompt,
    files=[{"name": "input.csv", "sha256": input_digest, "bytes": input_size,
            "schema": column_schema, "usage": "not yet determined"}],
    tools=[{"name": "python", "definition": exported_tool_schema,
            "implementation": {"id": "package/python", "revision": deployment_digest}}],
    harness={"id": "langgraph/my-agent", "revision": harness_digest},
    model={"name": model_name, "revision": immutable_model_revision},
    control={"max_iterations": 20, "max_parallel": 3},
    session={"messages": visible_messages, "objects": visible_object_inventory},
)
```

`revision` 必须来自可追溯、不可变的代码/镜像/模型版本或制品摘要。不要使用 endpoint、工具名称、工具数量、workload 名称或可变的 `latest` 标签代替实现身份。工具的完整定义也参与身份哈希；改变实现、schema、Harness 任一项即隔离。相同集合不受工具顺序影响。身份是可信部署配置，不从 Prompt 生成。

`describe_request(task, config, workload)` 从现有 LangGraph runner 的实际输入构建上下文，读取 `config.prediction_identity.tools` 与 `.harness`。非 DataSci 的工具定义必须与实际 `task.tool_definitions` 一致。没有部署锁定身份的配置仍可执行受约束默认方案，但不可检索历史。部署和 Harness 应在构建/发布时产出这些身份，不能仅根据当前 checkout 猜测远端代码版本。

文件原始元数据与完整 Prompt 一同进入模型；未知用途/大小不当作已发生的读取/零。文件内容未提供时，不声称知道文件内容。`model`、`control`、`session` 为执行前可见数据，禁止把实际计划、执行结果或未来状态写进这些字段。当前默认新请求的会话为空；有会话的集成应显式传入可见消息与对象。

## 模型配置与降级

现有 SDK 支持的兼容服务可以提供 Embedding 和 JSON 重排。部署选择实际可用、版本固定的模型，并填写对应账单币种和每百万 token 费率：

```python
retrieval = {
    "base_url": service_url,
    "api_key_env": "CONTEXT_MODEL_API_KEY",
    "embedding_model": pinned_embedding_model,
    "rerank_model": pinned_reranker,
    "revision": serving_artifact_digest,
    "currency": "USD",
    "rates": {"embedding": embedding_rate, "input": input_rate, "output": output_rate},
}
model = WorkloadModel(retrieval_config=retrieval)
```

费率与模型版本由部署配置提供，不内置当前供应商报价；API key 仅引用环境变量，不保存到快照。测试不调用该服务。两个模型需由同一服务提供，或由调用方注入 `ContextRetriever(backend)`；backend 提供 `embed(texts)`、`rerank(query, candidates)`、`revision`、`currency` 和累计 `cost`。

百炼北京配置见 `experiments/exp1/configs/retrieval-dashscope.json`。在 `usa` 上，密钥位于 `/root/Wabber/.env`（权限 600），使用方式：

```python
import json
from pathlib import Path
from dotenv import load_dotenv
from atsuite.workload import WorkloadModel

load_dotenv(".env")
retrieval = json.loads(Path("experiments/exp1/configs/retrieval-dashscope.json").read_text())
model = WorkloadModel(retrieval_config=retrieval)
```

现有实验配置的 `retrieval` 字段接收同一个对象。此配置使用业务空间专属域名，`text-embedding-v4` 每批最多 10 条；`rerank_api: "dashscope"` 经 `rerank_base_url` 调用原生排序接口，保留完整上下文与已有行为，用执行相关性指令排序，验证结果索引、分数和 token 用量。默认聊天 JSON 重排接口仍兼容旧配置。

原生 `qwen3-rerank` 返回相关性分数，不能产生原聊天模型的逐项 `usable` 判断或解释。因此原生适配器将候选交给现有兼容性、规模与数量限制，并按相关性排序；分数不作为预测正确率，也不引入未经校准的拒绝阈值。需要独立的行为可用性判定时应使用聊天重排后端。此配置不会自动打开 `enable_context_routing`。

费率核对日期为 2026-11-04，北京两个模型均按每百万输入 token 0.5 CNY 统计，排序无输出费；免费额度、优惠和实际账单结算不在该估算中。来源：[阿里云模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)。模型名是托管服务别名，`revision` 标记本次配置与费率版本，不代表供应商不可变权重摘要；严谨对比实验仍需供应商版本保证。

硅基流动免费模型配置见 `experiments/exp1/configs/retrieval-siliconflow.json`，使用上述代码时替换配置文件路径即可；E3 回放可传 `--retrieval experiments/exp1/configs/retrieval-siliconflow.json`。`usa` 上的 `SILICONFLOW_API_KEY` 已配置在同一个 `.env`（权限 600），无需逐个申请模型。此配置使用 `https://api.siliconflow.cn/v1`、`BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3`；`rerank_api: "siliconflow"` 调用原生 `/rerank`，验证完整且不重复的索引、有限的 0..1 分数和 `meta.tokens.input_tokens`。

按 2026-11-05 核对的[官方价格](https://www.siliconflow.cn/pricing)，这两个不带 `Pro/` 前缀的模型调用免费，配置中的三个费率均为 0；`Pro/` 版本收费，不会自动切换。[官方限流说明](https://docs.siliconflow.cn/docs/userguide/faqs/rate-limit-and-upgradation)要求实名认证，免费模型有固定账户级限流，超限会返回 429。模型权重分别采用 [MIT](https://huggingface.co/BAAI/bge-m3) 和 [Apache-2.0](https://huggingface.co/BAAI/bge-reranker-v2-m3) 许可；自行部署仍需承担算力费用。

BGE 重排只返回相关性分数，不支持行为判断指令，也不产生逐项可用性解释；与上述原生适配一样，分数仅用于排序，不代表行为可用性或预测正确率。托管模型名不是不可变权重版本，严谨实验仍需版本保证。此配置不自动更改现有实验配置或启用路由。

重排效率：召回仍取前 40 条，现有规模门槛在重排调用之前排除必定不能采用的候选；不从第 41 条补位，纯 Embedding 基线保持原行为。BGE 原生请求仅在完整工具/Harness 身份硬过滤通过后省略双方重复的 tool_definitions 与 harness，保留工具名、完整 Prompt、文件信息、模型、控制策略、会话及历史行为。原始上下文、兼容性检查和 Embedding 输入仍保留全量字段；直接传入不完整或不同身份的上下文时不精简。请求格式与检索算法版本更新，旧校准不会当成新格式的证据。没有跨不同查询复用重排分数。精简会改变分词和相关性分数，不能保证与全量文本同序；整体预测质量仍需独立回放验证。

当原生重排后端（BGE 或百炼）通过硬过滤和规模检查的候选不超过最终样本上限（默认 20）时，跳过远程重排，保留全部候选；原生适配本来就接受所有候选，且下游对所选样本等权统计，排序不会改变该样本集合。证据 reason 显式标注跳过，utility 为占位值 0，不冒充模型相关性分数。超过上限时仍调用重排选取前 20；聊天后端仍逐项判断 usable，纯 Embedding 基线行为保持不变。版本更新会隔离旧检索校准。

过滤在模型调用与缓存之前进行。缓存键包含模型版本、兼容身份与完整上下文。召回上限 40，重排后的联合情景上限 20；数量不足不放宽过滤。重排显式比较否定、条件、文件用途、规模、模型与控制策略；其 utility 仅排序，不是预测正确率。规模越出历史支持时冷启动，不虚构线性缩放关系。模型失败、无兼容样本或不适合参照的历史都进入冷启动。服务出错时已知费用是下界，`cost_complete=false`；预算启用时不会把未知费用当零。

未配置语义服务时，只接受身份完整且执行前上下文完全相同的历史；这是明确的保守模式，不是 Embedding 替代品。此模式无法识别改写句，输出会明确标识 `exact_context`。模型和控制策略不完整、输入大小未知、样本不足或残差未校准，均保留低置信标记。

## 观测、联合预测与反馈

每份 profile 保留冻结的 `descriptor` 和执行后的 `observations`：工具参数、依赖/时间区间、资源、逻辑状态访问、物理同步、LLM 调用与重试元数据。原始 `trace.json` 也保存执行前上下文。缺失测量保持 `None`。

现有统计口径不变：D 为累计 LLM span 时间（包括已记录重试），不是区间并集；原始起止时间保留重叠证据。S 的逻辑访问量与实际 `state_transfer_bytes` 分开。MCP Session 继续保留内存状态；FaaS 每次 Tool 执行继续恢复/持久化对象存储状态，本次不改变该生命周期。

预测是兼容、相关历史的条件经验情景分布。`joint_profiles` 与 `scenario_weights` 保留每条执行的 G/S/D 组合；平台评估逐条映射，不将独立均值拼成假执行，也不复制历史 provider。平台原先替换成目标 provider 原生历史 profile 的逻辑已删除；跨平台计时仍需要共享 reference 校准。workflow 汇总描述工具次数、依赖、并行、因果重访和分支；不声称已生成精确未来 DAG。单个近邻只能提供低支持证据，不能成为确定预测。

`begin()` 或 Router 在执行前冻结预测；`update()`/`complete()` 只用之后到达的真实观测产生残差。所有历史和残差要求 `available_ns < arrival`，同时间戳也不可用。完成时不能替换冻结上下文。残差按兼容身份、预测器版本、模型、控制策略隔离；至少 10 个此前的独立预测残差才形成 90% 名义绝对误差区间。模型相似度不参与校准。覆盖率依赖残差可交换性，漂移时没有保证；需用独立回放/在线数据检验。

新记录会自动参与后续兼容检索，新的工具集合/Harness 独立冷启动。`validation.supervision_pairs()` 导出只含此前兼容历史的训练对，以真实执行距离作为监督标签。它是训练数据出口；本次没有训练新的 Embedding/重排权重，部署替换权重时必须更新 `revision`，旧残差不可继续充当校准证据。

## 旁路与启用

Router 默认 `enable_context_routing=False`：实际执行使用经验证的默认方案，预测和误差仍写回；不会依据新预测触发探索或扩缩容。`shadow=True` 可计算候选选择，不占执行容量，完成时必须提供实际执行池。

经过独立旁路验证后，在现有 `routing` 配置中显式设置 `enable_context_routing: true`。需求或平台费用/时延未校准时仍用受约束默认方案；显式配置的有界探索可积累冷启动数据。容量、工具、内存、截止时间和预算检查始终执行，不存在执行中迁移。JointController 使用同一隔离预测，未启用时不应用新预测驱动的扩缩容。

预测器输出包含模型调用费、检索延迟与完整决策延迟。Router 将模型费用加入候选目标和实际预测费用账本，有预算的实际决策也计入该预算；影子请求的推理费单列，影子请求不预留执行预算。平台评估时间计入执行前等待和截止时间。JointController 预测费用按请求到达率计入 horizon 成本。缺失费用/币种换算保留不确定性，不用于虚构便宜方案。

## 旧数据迁移与旧基线

旧算法基线保存在提交 `0ab1067657a5ef0861c98a899a93834273f9ec4f`，生产代码不再携带旧实现。历史冻结报告仍为历史结果，不代表当前算法。

v1 快照可读，但旧标签不会成为检索上下文。只有原始 Prompt、文件与当时可见状态，以及可验证工具/Harness 身份能重建的记录可以迁移。不能从最终计划/结果猜测到达时上下文。迁移 sidecar 格式为：

```json
{"request-id": {"context": "完整 describe() 返回字典", "visible_ns": 123,
                "evidence": ["原始到达记录与部署锁定制品路径/摘要"]}}
```

```sh
uv run python -m atsuite.workload.migrate old.json --arrivals raw-arrivals.json --output migrated.json
```

输出必须是新文件；标签数据保留审计，计入 `quarantined`，永不跨组回退。历史初始 prior 同样需要完整身份和明确 `available_ns`、`completed_ns`；不接受无时间来源的 prior。

## 验证与验收边界

```sh
uv run python -m unittest tests.test_context_prediction -q
uv run python -m atsuite.workload.validation records.json --config validation.json \
  --baseline frozen-old-predictions.json --output report.json --supervision training-pairs.json
```

`records.json` 是 request observation 列表，一条逻辑请求一行，含原始 arrival context、profile、实际 pool、measurements、availability；字段与 `ProviderModel.update()` 相同。可选 `outcomes` 按候选 pool 给出相同逻辑请求在匹配执行中的 `{cost, latency_s, status}`，费用统一为配置 `currency`，成本需包含失败尝试。评估器不会把未执行候选的实际结果编造出来。

配置包含 `pools`、`tariffs`、`execution_bounds`、`default_pool`、`currency`，可选 `retrieval`、`fx`、`latency_weight`、`held_tasks`。固定历史按时间回放，未来/同时间反馈不可见；未见任务的全部反馈被隔离。

报告分别列出：上下文方案、纯 Embedding 最近邻、使用真实需求的参考上限；旧基线只读取冻结预测，不执行旧分类器。未配置语义模型时，两种检索评估都退化为精确上下文模式，不能据此比较语义能力。报告提供行为误差、区间覆盖、支持量与冷启动轨迹。仅在全部候选有真实匹配结果时比较最佳固定部署与逐请求 oracle，并计算含完整预测开销的回放成本、延迟、失败和路由损失。失败率单独报告，不隐含折算成未声明的金钱惩罚。真实需求参考与匹配结果 oracle 是不同的上限。

`online_routing_results` 始终为 null：固定轨迹不证明真实在线收益。上线验收仍需已锁定版本的语义服务、独立时间序列/未见任务数据、当前平台计价校准与真实在线实验。合成测试只证明隔离、因果与接线契约，不证明语义泛化或云收益。
