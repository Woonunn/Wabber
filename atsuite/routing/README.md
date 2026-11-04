# Wabber Stage 7: semantic workloads and exploration

`OnlineRouter` predicts from information available at admission and binds the whole
Agent Session to one execution pool. It has no SLO, p95, risk-bound or same-task
sample-count admission gate. p50/p95/p99/mean remain descriptive statistics.
MCP retains in-process state; FaaS restores/persists state at each Tool invocation.

## Admission and feedback

```python
from atsuite.workload import WorkloadModel, describe
from atsuite.provider_model import ProviderModel
from atsuite.routing import OnlineRouter

router = OnlineRouter(
    WorkloadModel(initial_profiles=independent_profiling_profiles),
    ProviderModel(tariffs=dated_tariff_cards),
    pools, verified_session_limits,
    cost_weight=1, latency_weight=0.001,
    default_pool="verified-default",
    execution_bounds=validated_bounds,
    budget=run_budget, fx=dated_currency_reference,
)
descriptor = describe(prompt=prompt, files=input_metadata, tools=available_tools)
decision = router.decide(request_id, load_rps=current_load, descriptor=descriptor)
# Dispatch only if selected is not None, using the bound pool and execution_bounds.
# Complete failed attempts too, after the existing executor has drained pending RPCs.
router.complete(request_id, trace)
```

Stage 7 configuration, validation and fair comparison commands are documented in
[STAGE7_VALIDATION](../../experiments/exp1/STAGE7_VALIDATION.md).

`task` remains an optional second argument for experiment bookkeeping. It never
selects training samples or determines admission. `files` contains `{name, bytes}`;
`hints` optionally contains declared `cpu_s`, `memory_mb`, `state_bytes`, `rows`,
`columns`, and `features`.
Descriptors retain lengths, file types/sizes and Tool capabilities, and add a
versioned computational intent signature extracted from the Prompt. Known-family
matching requires the observed combination of intent tags, compatible resource
scale, and the same Tools/encoding. A priority-selected `primary` label cannot hide
a new secondary operation. Legacy snapshots remain readable.

An unseen intent can use a `resource_prior` drawn from nearby declared sizes and
hints with the same Tools, even across file encodings. This prior remains
`unknown_request=true` and low confidence. Missing file sizes are not zero-byte
inputs. Without comparable resource evidence, only an explicitly supplied initial
prior is available; otherwise prediction metrics are missing. No file-format or
benchmark-specific memory/state estimator is used.

Every prediction retains at most 20 joint profiles, including global/initial
priors; equal-distance peers prefer recent available observations. Router and
JointController consume this same sample without truncating it again. Compute,
state, LLM time and call-structure correlations are preserved; state access volume
is not renamed or imputed as peak memory. Missing input sizes keep confidence low.

Normal routing ranks feasible candidates with complete, comparable cost forecasts
by the configured cost/latency objective even when history or model confidence is
low. Confidence remains visible in diagnostics and exploration; it is not an
admission constraint. Cost-only routing does not require an unused latency
forecast. Unknown/exhausted capacity, unsupported Tools and actual execution
deadlines still prevent dispatch. Missing cost estimates or incomparable currencies
still use the validated fallback rather than fabricated zero costs.

Optional budgeted contextual epsilon-greedy explores feasible, analytically
promising pools and records the full probability distribution. Only chosen-pool
feedback trains the online model. Unknown resource priors do not bypass validated
default execution, capacity, timeout or budget constraints.

The `stage7 holdout` evaluator also accepts a configuration containing `descriptors`
and independent `family_labels` for any workload, without loading the DataSciBench
panel. Its default seeded reservoir retains at most `history_limit` already-visible
attempts per provider, instead of freezing the first arrivals. Legacy/semantic
variants use identical selected attempts; selected IDs are recorded. For historical
reproduction, set `history_sampling` to `first`. The `evaluate()` API keeps `first`
as its backward-compatible default. Reservoir evaluation keeps held-out families
frozen (`adaptive=False`); normal online routing is unaffected. Derived-label
holdouts are explicitly labeled diagnostics, not semantic accuracy measurements.

Each `execution_bounds[pool]` has `timeout_s`, `max_cost`, `currency`, `tools` and
`evidence`. Evidence must cover the actual supported execution environment and the
worst-case cost of bounded execution, including remote cleanup/retention. These are
trusted deployment configuration, not bounds inferred from p95 or supplied by a
Prompt. Every admitted request atomically reserves its cap. Completion consumes
that cap unless the caller supplies authoritative attributable `actual_cost` in
the router's budget currency. Shared infrastructure is reserved by the existing
campaign budget ledger, separately from request caps. An empirical cost range is
never a hard spending limit.

Unknown capacity, exhausted concurrency, unsupported Tools, elapsed execution
deadlines and exhausted budgets still prevent dispatch. A validated default cannot
override those checks. Repeated request IDs return frozen decisions; migration and
automatic retry are forbidden. `shadow=True` reserves no capacity or budget.
Standalone callers must enforce the returned execution bounds in their executor.
The LangGraph hook enforces Session deadlines and retains existing Tool timeouts,
iteration limits, sandbox restrictions and pending-RPC cleanup. Cleanup can outlast
the application deadline; it must finish before capacity is released.

`complete()` attaches the frozen Descriptor to the Workload profile and updates
platform timing calibration from complete Tool runtime counters. Native profiles
retain `execution_pool` (provider, environment and resources). Availability is
strictly `< decision_ns`, including delayed provider evidence revisions. No result
from the current request is used before dispatch.

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
profiles are target-native observations, not common benchmarks. A foreign native
profile uses descriptor-matched observations from the target pool when available;
without those or a shared reference, its estimate remains uncalibrated and routing
falls back. Physical state-transfer counters are reused only for the exact source
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
or draining. Offered load is grouped by resource family, including rejected
arrivals; late arrival metadata updates that family without double-counting.

Stage 5 `risk` configuration is replaced by `routing` options above. Per-task
`level.cost.per_request` is no longer used for capacity optimization: supply the
level's `pool.billing` card (or a provider tariff). Public cloud deployment levels
still require independent capacity evidence. The legacy `forecast_cost` helper is
retained only for old accounting callers/tests. [JOINT_CONTROL.md](JOINT_CONTROL.md)
is the archived Stage 5 experiment contract; its SLO and historical pricing rules
are superseded here. The old `routing.evaluate` remains an archived SLO diagnostic;
use `routing.generalization` for Stage 6 evaluation.

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
