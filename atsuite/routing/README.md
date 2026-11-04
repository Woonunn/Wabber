# Wabber context-based routing

请求预测、严格工具/Harness 隔离、模型配置、迁移、旁路与验收流程见
[Context prediction](../workload/README.md)。生产链路已经删除人工意图标签、关键词正则和全局/资源先验回退。

`OnlineRouter` 默认只记录新预测，实际执行走受约束默认方案；验证后显式设置
`enable_context_routing=True`。`shadow=True` 计算候选决策且不预留执行容量。
兼容身份或原始上下文缺失的旧历史不会参与预测。所有路由保留容量、工具、内存、超时与预算约束。

新检索的费用、耗时和误差校准参与评估。历史 provider 不是最优标签；G/S/D 联合情景交给当前平台模型定价与计时。

## Cost and latency

`ProviderModel(..., tariffs={provider: card})` accepts dated, region-specific tariff
cards. `pool["billing"]` overrides a provider card for a particular capacity level.
The example is [stage6-tariffs.json](../../experiments/exp1/configs/stage6-tariffs.json).
It is a declared first-tier on-demand scenario, not an account quote. Supply actual
rates and explicit shared-service/floor/scaling charges for your deployment.
AgentCore cards distinguish runtime versions and require measured/calibrated actual
resource use; requested CPU/memory are not treated as allocations.

For each predicted joint resource sample, the model computes:

* Execution: billable CPU-seconds, GB-seconds, disk GB-seconds and invocation count.
  FaaS durations round **each call** to the tariff quantum. Resident resources use
  Session billing. Active CPU and allocated CPU have different formulas.
* State: snapshot reads/writes, physical transfer volume, GB-seconds of retained
  objects and applicable network transfer. Logical state accesses are not storage
  RPC counts. Without physical counters, snapshot size and per-call RPCs are
  explicitly approximate and need calibration. MCP does not reload state per call.
* Retention: billable memory/disk during LLM waits and the declared Session tail.
* Pool total: marginal request costs plus the pool floor and shared-service cost
  **once per horizon**, scaling events, and old/new capacity overlap. A floor can
  explicitly include compute, in which case that compute is excluded from marginal
  request charges. Existing Sessions are charged separately from future arrivals.

`analytical_cost` is the optimization input and also populates `cost` when a
tariff is configured. Without tariffs, the older `cost` field remains a legacy
historical-scope diagnostic (also exposed as `historical_cost`); neither new
Router nor JointController ranks candidates from historical costs. Missing measurements remain
visible. Analytical completeness means the configured formula has inputs, **not**
that its assumptions or rates constitute a complete vendor invoice.

Timing separates total Tool work from the union of occupied Tool/LLM intervals.
The former prices FaaS invocations; the latter estimates elapsed time and resident
memory retention. Historical overlap is an empirical prior, not a predicted DAG.
Native Session samples already include observed waiting; queue samples are not
added a second time. It mixes distributions, never adds component p95s.

Speed ratios require an explicit shared `reference_id` on profiling inputs; a
tariff's declared `tool_scale` must name that same reference. Ordinary completion
profiles are target-native observations, not common benchmarks. A foreign native profile is never replaced by a target-provider historical execution.
Without an explicit shared reference its estimate remains uncalibrated. Physical state-transfer counters are reused only for the exact source
execution environment. Other modes use their own logical-state transport estimate.

Confidence checks both resource/load coverage and usable calibration evidence.
Three successful rows with missing runtime counters do not establish calibration;
missing features and extrapolation remain visible and cause fallback. Existing
snapshots without reference provenance cannot silently become speed benchmarks.
Callers supplying `reference_profile=` must now include its shared `reference_id`.
JointController excludes unreliable forecasts from scaling plans too; a fresh,
validated default remains available for bounded requests when no plan is trusted.

`predict_cost(..., measured=True)` reprices completed usage in its actual
`execution_pool`; it does not apply prediction speed/init/sync multipliers again.
This is a tariff-based usage estimate, not an invoice. Simulation accounting and
native E1-B reference costs use this path. An E1-A reference Capture is not the
actual target execution, so only separately recorded attributable vendor compute
charges are used as its cost reference.

Router minimizes `cost_weight * marginal_cost + latency_weight * mean_latency_s`.
JointController minimizes `cost_weight * pool_horizon_cost / horizon_s +
latency_weight * mean_latency_s`, including queue/scaling wait estimates. The two
objectives have explicitly different cost units. Weights are **not silently
normalized**. Different currencies require a dated FX reference. `pareto` lists
nondominated feasible pool/plan alternatives; changing the weights selects among
them. Frontier estimates are not evidence of measured savings.

## Joint control and compatibility

Existing adapters, polling, read-back, cooldown, capacity levels, Session drain,
failed-teardown holds, lifetime retention and budget-reserved cloud execution are
retained. Scaling targets never count as ready until read back. A default fallback
under JointController needs fresh READY capacity and cannot bypass pending scaling
or draining. Offered load is grouped by the full arrival context, including rejected
arrivals; late arrival metadata updates that family without double-counting.

Stage 5 `risk` configuration is replaced by `routing` options above. Per-task
`level.cost.per_request` is no longer used for capacity optimization: supply the
level's `pool.billing` card (or a provider tariff). Public cloud deployment levels
still require independent capacity evidence. The legacy `forecast_cost` helper is
retained only for old accounting callers/tests. [JOINT_CONTROL.md](JOINT_CONTROL.md)
is the archived Stage 5 experiment contract; its SLO and historical pricing rules
are superseded here. The old `routing.evaluate` remains an archived SLO diagnostic;
use `workload.validation` for the current context-prediction evaluation.

The LangGraph runner accepts `prompt=` for a new Prompt over a supported DataSciBench
dataset/validator. It uses live planning and disallows replaying a different Prompt
against an old Capture. `task_id` still identifies that audited dataset and result
validator in DataSciBench; the Router does not require it. Frozen-trace cloud
images still only accept their sealed Tool code. New generated programs require
the existing validated Docker confinement, not weakening the frozen-code policy.

## Reproduce validation

```sh
uv run python -m unittest tests.test_stage6 tests.test_routing \
  tests.test_joint_control tests.test_workload tests.test_provider_model -q
uv run python -m atsuite.routing.generalization \
  --e1b results/phase5/cloud-fixed-typical-envref-v3/formal \
  --output results/stage6/e1b-new
uv run python -m atsuite.routing.generalization \
  --e1a results/phase4/e1a-formal-useast-seed11-v1 \
  --output results/stage6/e1a-new
uv run python -m experiments.exp1.runner.stage6_simulation \
  --output results/stage6/simulation-new
uv run python -m experiments.exp1.runner.run_stage6_local \
  --image YOUR_VALIDATED_SANDBOX_IMAGE --output results/stage6/prompts-new
```

The last command makes live DeepSeek API calls (two tasks, at most two planning
iterations each), with real local Tool execution and no cloud deployment. It pins
the supplied existing Docker image digest before execution. Outputs must be new
paths. The historical experiments use independent task holdouts and 0/3/20 initial
samples per provider, separating frozen versus online feedback. They report
repriced measured demand separately from attributable compute estimates and leave
actual invoice error unknown when absent. E1-A scores every tenth arrival in a
fixed arrival/provider/request-ID order while ingesting all eligible evidence;
sampling never depends on completion times or results. The simulator's independent source IDs
check mechanics; its synthetic data do not establish generalization or cloud gains.
See [Stage 6 validation](../../experiments/exp1/STAGE6_VALIDATION.md) for results,
limitations and artifact paths.
