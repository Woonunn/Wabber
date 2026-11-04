> 历史实验归档：其中的意图标签/资源先验算法已移除。当前实现与复现入口见 [Context prediction](../../atsuite/workload/README.md)。这些旧结果仅用于离线基线比较。

# Stage 7：语义请求族与受预算约束的主动探索

基于 `737abd9`（2026-11-01 获取的 origin/main）增量实现；没有新增运行依赖。
沿用现有 Trace、解析费率、ProviderModel、Session 准入、JointController 与云实验门禁。
本阶段不预测完整 DAG，不迁移运行中的 Session，不引入 SLO 硬约束。

## 通用机制修正（后续更新）

以下修正不使用任务 ID、DataSciBench 特定算法或新的领域词条；下文原始数值表保留为修正前的历史证据。

- **未知与先验选择分开**：未知计算意图可以使用同 Tools、相近文件规模/显式资源 hints 的 `resource_prior`，但始终 unknown、低置信度。沿用 0.5 的对数资源距离阈值；没有邻近证据时使用显式初始 prior，没有该 prior 则不输出伪精确资源均值。不同文件编码只可提供低置信度先验，不因此认定计算族相同。
- **保留完整意图组合**：已知族比较整个标签集合，`primary` 仅保留作兼容性元数据/历史分组，不再以固定优先级掩盖其他计算操作。未知组合可以借资源先验，不能自动提升为已知族。原有有限词表没有增加 Apriori 或其他 benchmark 特判，也没有声称具备通用自然语言理解。
- **统一样本预算**：已知族、资源先验、全局和显式初始 prior 均最多保留 20 个完整联合样本；同距离优先近期反馈。Router/JointController 不再进行第二次截断。所有原始历史仍保存，限制只影响本次预测样本。
- **缺失数据保持缺失**：未知文件大小不等于 0；缺失文件规模保持低置信度。保留显式 rows/columns/features/cpu/memory/state hints，但不新增某种文件格式专用解析器、不从 Prompt 或文件大小猜测状态/内存，也不改变累计逻辑状态访问量的定义。
- **消除“前 20 次”冻结**：Stage 7 留出 CLI 默认使用确定性哈希优先级的因果蓄水池采样，最多保留每 Provider 20 个已可见尝试（可用既有 history_limit 配置修改）。失败仍是尝试，延迟修订不增加一次抽样；未来结果和测试族结果均不能进入训练。新旧描述器使用同一 seed 和相同训练请求。底层 `evaluate()` 的 `first` 默认保留兼容性，旧实验可显式复现。
- **评估标签独立**：配置可直接提供 `descriptors` 与覆盖数据的 `family_labels`，无需加载 DataSciBench 面板。标签只控制训练/测试划分，不传给匹配器；任意文字标签采用哈希文件名保存结果。没有独立标签时，报告标注 `descriptor_derived_diagnostic`，不能将 unknown 比例解释为语义准确率。虚拟影子 Tools 来自声明的描述器，仍不是云端能力验证证据。

通用评估配置示意（费率与资源必须使用现有验证输入）：

```python
config = {
    "descriptors": {request_type: describe(prompt=prompt, files=files, tools=tools, hints=declared_hints)},
    "family_labels": {request_type: independently_assigned_family},
    "tariffs": validated_tariffs,
    "fx": dated_fx,
    "history_limit": 20,
    "history_sampling": "reservoir",
    "sampling_seed": 7,
}
```

新 [通用回归测试](../../tests/test_workload_generalization.py) 覆盖视频转码、文档检索、压缩、复合操作、输入形状与缺失值、未知族路由置信度、未来观测隔离、快照往返、采样次序/修订与通用 CLI。
这些是领域无关机制的合成契约测试，不是其他业务的真实云端性能证明。
全量测试：278 tests，OK，1 skipped（宿主缺少 sklearn/joblib）；原有 Docker/AWS 文档工作区修改保留。

### 通用修正后的历史回归边界

在相同已可见候选历史上，用 seed=7、每 Provider 最多 20 次的蓄水池对比；逐请求核对确认 legacy/semantic 的训练请求 ID 完全一致。
这次采样策略与上方原始 first-arrival 实验不同，不能把两轮总 MAE 直接相减来宣称改进。
以下同时报告预测覆盖率，以及两种描述器**都输出延迟预测的相同请求**上的 MAE：

| 留出族 | legacy 有预测/总数 | semantic 有预测/总数 | 共同请求数 | legacy 共同样本 MAE 秒 | semantic 共同样本 MAE 秒 |
|---|---:|---:|---:|---:|---:|
| training | 384/388 | 256/388 | 256 | 29.103 | 29.280 |
| visualization | 371/376 | 120/376 | 120 | 22.803 | 28.822 |

资源邻域外的请求缺失预测，仍可由经验证的默认池有界执行。共同样本上仍存在退化，因此本轮不能声称新描述器提高了真实资源预测或 Provider 选择质量。
这些修改验证的是跨 workload 的证据选择、缺失值、因果性与评估独立性；不为了当前留出集的误差增加领域词条、改权重或调整邻域阈值。
[本轮历史报告](../../results/stage7/generic-fixes-holdout/report.json) 保留采样配置、实现摘要、每次训练/预测样本 ID 与覆盖率字段；原有 first-arrival 报告没有覆盖。
尚未具备其他业务的真实 Trace/云端数据来证明跨业务预测收益。

## 描述器与未知请求

`describe()` 增加版本化的 `semantic`：计算主意图与多标签意图，覆盖统计、数据处理、训练、迭代、可视化以及状态复用。
规则使用可审计的 DataSciBench 中英文词汇，不训练语义模型，不读取当前请求的执行结果。
文件大小/类型、Tools 以及 `cpu_s/memory_mb/state_bytes/rows/columns/features` 输入 hints 保留。
原始实现按主意图分族；最新匹配与回退行为以上方“通用机制修正”为准。
Prompt 长度保留在审计记录中，不再参与新描述器的匹配或族标识。

已有快照继续可读：旧描述器之间使用旧距离，新旧描述器不会凭长度自动匹配。
迁移历史数据时，应从原始 Prompt 重新生成描述器，不能从完成后的 ATG 生成到达时特征。
WorkloadModel 保留计算、LLM idle、状态、调用次数、依赖/分支等经验联合分布；
没有相似族时，最新实现先尝试低置信度资源先验，再使用显式初始 prior；标记 `unknown_request=true`、`low_confidence=true`。
没有描述器的全局 prior 和显式初始 prior 也不表示未知请求已属于某个已知族。
ProviderModel 的目标池近邻也使用语义过滤，避免在平台预测阶段重新混入不同计算意图。

## 探索策略与证据

在现有 routing 配置中增加：

```json
{
  "budget": 10,
  "exploration": {
    "strategy": "contextual",
    "epsilon": 0.1,
    "budget": 1,
    "seed": 7,
    "min_samples": 3,
    "window_s": 3600,
    "epoch_ns": 0,
    "optimism": 0.5
  }
}
```

金额仅示范配置结构，不是获准的实验预算。实际数值必须来自原有预算机制。
`execution_bounds`、经过验证的 Tools/环境、各池真实 Session 并发上限、超时、内存声明和 dated FX 仍为准入前提。
默认不启用探索；`random` 提供同预算、同约束的随机探索基线。

- 先选择可靠预测的成本—延迟最优池，或有验证执行边界的默认池。
- 上下文探索只考虑完整解析成本和延迟已知、同币种且有可能改善目标的候选。
  乐观目标为预测目标乘以 `1 - optimism / sqrt(1 + n)`，`n` 是相似请求族在该执行环境中已可见的近期独立尝试数。
  多个候选按 `1/sqrt(1+n)` 分配探索概率；持续失败的池不被当成证据空白继续奖励。
- 实际 epsilon 为 `epsilon/sqrt(1 + total_attempts/min_samples)`。优势明确后，劣势池可以得到零探索概率；没有强制多 Provider 流量份额。
- 近期窗口到期、显式新 epoch，或价格/部署配置变化，会允许重新收集证据。预算消耗不会因此清零。
- 请求 ID 与 seed 决定可复现的随机数；重复决策返回原始记录。
  记录所有候选预测、不可行原因、概率分布、被选池的概率、随机数、模型配置摘要和当时可用的证据数。
- 成功与失败的实际完成都反馈到绑定池；严格使用 `available_ns < decision_ns`。
  不会摄入未被选择池的反事实执行结果。真实在线调用按事件时间顺序运行。

每个探索请求预留**整个执行 cap**，不是预期额外成本。并发预留同时扣减总预算和探索预算。
完成时有可归属费用则使用它，否则继续按 cap 保守计入；日志明确区分 `execution_cap`、可归属执行费用和模拟计量估算。
模拟器按生成的执行用量结算，保持独立的完整基础设施成本时间线，模拟金额不标记为实际账单。
`exploration_report()` 提供选择比例、探索次数/成功数、按来源分类的执行费用和容量预留。
单个请求的真实反事实收益保持 unknown；收益通过相同输入、不同独立运行的策略比较获得。

## 联合容量

可选的 `control.capacity_exploration`：

```json
{
  "lease_s": 60,
  "max_cost": {"ali_agentrun": 0.1},
  "evidence": ["经校验的预热、保留、冷却、缩容、控制操作全生命周期费用上限"]
}
```

这里只表示配置形式。费用上限必须是预算币种，并覆盖租期、在途请求排空、Session 尾部和恢复；执行费用另行预留。
一次只给一个有潜在收益的空闲池申请已校准的最小正容量，先预留生命周期 cap，再发起变更。
真实容量读回完成后才允许探索。租期结束关闭该探测池的新增探测、排空并恢复原容量；只有确认恢复后才结算 cap。
失败或不确定的缩容不释放预留。同一池在证据窗口内不重复申请探测租期。
已可用的池无需探测租期即可参与受限探索。模型认为长期劣势的池不保留探测容量。

`routing` 对比策略保持初始容量，在所有读回可用的池之间比较请求目标；
它不会再被空闲期控制周期的旧权重困住。`static` 仍固定流量，`joint` 继续联合优化容量与长期成本。
请求 cap 和探测容量 cap 不替代原有整场实验预算；普通基础设施保留和共享费用仍由整场预留覆盖。

## 可复现实验入口

```bash
# 12 个原始 Prompt，来源版本与逐 Prompt 哈希；不调用 LLM/云服务
.venv/bin/python -m experiments.exp1.runner.stage7 panel --output results/stage7/panel

# 隐藏快池执行历史，仅允许被选择池的反馈参与学习
.venv/bin/python -m experiments.exp1.runner.stage7 smoke --seed 7 --output results/stage7/smoke-seed7

# E1-B 整个计算意图族留出，同时跑 legacy/semantic 描述器
# config 含 tariffs（费率字典）、fx 和 history_limit；也可 --rows 指向已提取的 observation 数组
.venv/bin/python -m experiments.exp1.runner.stage7 holdout \
  --config results/stage7/holdout-config.json \
  --e1b results/phase5/cloud-fixed-typical-envref-v3/formal \
  --output results/stage7/e1b-family-holdout

# 从已有经过审核的 joint config 和封存 workload 生成独立配置与输入哈希
.venv/bin/python -m experiments.exp1.runner.stage7 plan \
  --config CONTROL.json --workload SEALED_WORKLOAD --output PLAN_DIR
```

策略包括 `fixed_aws_lambda`、`fixed_aws_agentcore`、`fixed_ali_fc`、`fixed_ali_agentrun`（Aliyun FC Session MCP）、
`cost_aware`、`random_exploration`、`exploration_aware`、`full_wabber`。
每个策略从相同模型快照独立开始，使用同一封存 Prompt/数据、请求相对到达时序、Harness 与 Tool 版本。
生成 plan 只产生文件；真实运行仍使用 `experiments.exp1.runner.run_joint --policy STRATEGY`，
并要求匹配该策略配置/实现摘要的独立模拟、部署校验、计算正确性参考、Pilot 和预算预留。
固定策略保留配置中真实存在的所有资源费用；不能将没有卸载的其他 Provider 基础设施费用直接删除。

```bash
# 将完整运行的请求/容量日志与可归属费用证据汇总
.venv/bin/python -m experiments.exp1.runner.stage7 collect \
  --run RUN_DIR --config CONTROL.json --charges CHARGES.json --output REPORT_DIR

# cost_aware 为参照，仅比较相同封存 workload；保留成功率，不用低完成率伪造节省
.venv/bin/python -m experiments.exp1.runner.stage7 compare \
  --reports BASELINE_REPORT.json CANDIDATE_REPORT.json --output COMPARISON_DIR
```

`CHARGES.json` 沿用 `atsuite.routing.cost.total_cost` 的排他费用行：
`provider/resource/charge_id/start_ns/end_ns/component/amount/currency/basis/evidence/attribution/complete`。
Stage 7 要求每个实际配置资源完整覆盖运行窗口的 `execution/storage/retention/scaling/shared` 五项，
零费用也需显式证据；shared 包括必要控制费用。重复账单行去重，重叠费用拒绝，缺项或缺窗口返回未知总额。
`compare` 不混算预测与实际账单，也不会把独立 Provider Replay 的影子选择统计当作新混合运行。

## 本地验证结果与限制

新增测试覆盖意图/形状分离、未知族、旧快照、整族留出、未来结果不可见、并发预算、失败反馈、价格变更、
探索转利用、劣势池零概率、容量生命周期预留/恢复、策略隔离以及费用证据完整性。

seed=7、同一 60 请求序列的合成机制结果（全部 60 次成功）：

| 策略 | 快池选择次数 | 探索次数 | 合成基础设施 USD | p95 秒 |
|---|---:|---:|---:|---:|
| 无探索 | 0 | 0 | 0.03501401 | 4.460 |
| 随机探索 | 40 | 14 | 0.03445627 | 4.422 |
| 上下文探索 | 47 | 7 | 0.03435907 | 4.344 |

这些数值只证明受控合成场景中的学习与利用链路可运行，不是云端收益保证。
费用包含模拟的两个池保留；不把较快的请求耗时直接当作整场成本降低。

全量 `unittest discover -s tests -q`：266 tests，OK（1 skipped：宿主缺少 sklearn/joblib）。
新增 16 个 Stage 7 测试；后续针对实验身份、反馈描述器与容量恢复的相关回归也通过。Ruff F 检查与 `git diff --check` 通过。
[合成结果与概率日志](../../results/stage7/smoke-seed7/report.json) 保留实现摘要；没有将合成数据写入真实 Provider 历史。

### 真实 E1-B 的整族留出诊断

复用既有 `cloud-fixed-typical-envref-v3/formal` 的 764 次独立 Provider 请求；按声明的计算主意图整族留出，
每 Provider 最多 20 个非留出训练请求，测试族结果从不更新模型。两种描述器使用同一事件时序和相同 held-out 集合。
所有影子决策仍回退 AWS Lambda；这不是本轮真实混合路由的运行结果。

| 留出族 | 请求数 | 描述器 | 延迟 MAE 秒 | Tool 耗时 MAE 秒 | LLM idle MAE 秒 | 状态量 MAE 字节 |
|---|---:|---|---:|---:|---:|---:|
| training | 388 | legacy | 15.445 | 10.976 | 10.957 | 58,422,076 |
| training | 388 | semantic | 12.629 | 8.941 | 8.341 | 149,534,514 |
| visualization | 376 | legacy | 56.640 | 53.624 | 9.146 | 406,368,884 |
| visualization | 376 | semantic | 59.332 | 55.649 | 4.872 | 497,553,709 |

新描述器将 764 个整族留出请求全部识别为 unknown；旧描述器只有 7 个没有资源族匹配。
训练类延迟/计算误差降低，但可视化类和状态量误差变大，说明 unknown 的全局先验仍不够准确。
这批数据支持“避免错误的已知族置信度”，**不支持新描述器全面提高资源预测，也尚未证明真实选池收益**。
应扩充相似意图、不同规模的真实 ATG，而不是根据本次留出结果调规则后继续把它当独立测试。
[完整留出结果与来源哈希](../../results/stage7/e1b-family-holdout/report.json)；费用误差仍是历史预测诊断，不是混合账单。

尚未进行新的真实云端混合运行或实际账单核对。12 个原始 Prompt 中只有原有 6 个已有准备好的数据与验证流程；
其余 6 个 BCB Prompt 用于语义面板，必须先接入现有输入准备与输出校验才能执行云实验。
规则会遗漏否定语义、隐含计算和领域外意图，返回 unknown 是预期行为；需要后续扩充真实 ATG 数据验证误差。
精确历史、决策与日志仍保存在单进程内存/本地文件中，不实现分布式服务或跨进程恢复控制器状态。
