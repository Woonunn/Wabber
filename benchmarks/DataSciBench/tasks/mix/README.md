# DataSciBench-Mix

本目录封存 **100 个不同原始 Prompt** 的真实 LangGraph Agent 执行和第二次真实计算的环境参考。原始 Prompt 来自 catalog 锁定的 DataSciBench commit；没有改写 Prompt、增加无意义计算或跳过测试。输入来自该版本的 BCB 评测 fixture，不是新造的生产数据。此集合不包含旧的六个大数据任务。

- `manifest.json` / `seal.json`：Prompt、输入文件、测试来源、实际输出、Capture 和环境参考的哈希。
- `audit.json`：全部 172 条候选的审查结果与失败原因。最终 100 条通过，50 条参考预检失败，9 条 Agent 验证失败，2 条引用未封存的 `/tmp` 输入，5 条尚无适配，6 条保留原有独立执行链路。
- `bcb_cases.json.gz`：锁定上游的输入代码、测试和参考实现；只供隔离预检使用。
- `runtime_checks.json.gz`：运行镜像使用的测试，**不包含参考实现**。
- `evidence.tar.gz`：1,206 个原始输入、Trace、执行结果和实际返回产物，保持仓库内原路径；不需要重新调用 LLM 才能复跑。
- `validation.json`：本地验证与独立模拟检查的摘要。模拟结果不属于真实云端成本证据。

## 恢复并检查证据

在仓库根目录执行。pickle 只由隔离 Tool 读取，宿主机不反序列化它。

```sh
uv run python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('benchmarks/DataSciBench/tasks/mix')
manifest = json.loads((root / 'manifest.json').read_text())
actual = hashlib.sha256((root / 'evidence.tar.gz').read_bytes()).hexdigest()
if actual != manifest['bundle']['sha256']:
    raise ValueError('Evidence archive checksum mismatch')
PY
tar -xzf benchmarks/DataSciBench/tasks/mix/evidence.tar.gz
docker build -t wabber-datascibench-mix:reproducible -f experiments/exp1/profile/Dockerfile.mix .
```

镜像基础 digest、Python 和全部 Python 包均固定。工具容器仍禁网、只读根目录、限制 CPU/内存/PID/文件大小并执行超时；FaaS 每次恢复/保存状态，MCP 继续使用原有驻留内核。BCB 验证在私有临时目录执行，不能删除只读输入。图表结果比较数值与渲染像素，原始 pickle 的 SHA256 另外保留；云端 `bcb61` 仅对已核对真实调用时间的时间注释文字进行规范化，`bcb823` 用真实调用时间点独立复算耗时统计，详见实验目录的 `comparison-policy.json`；测试错误的比较只排除堆栈位置和耗时，保留失败测试及断言。

## 准备新证据

```sh
uv run python -m experiments.exp1.data.prepare_mix \
  --repo results/upstream/DataSciBench --output results/new-mix-import
uv run python -m experiments.exp1.runner.mix preflight \
  --bundle results/new-mix-import/bcb_cases.json.gz \
  --image "$(docker image inspect wabber-datascibench-mix:reproducible --format '{{.Id}}')" \
  --output results/new-mix-preflight
```

参考实现通过不算 Agent 验收。`mix capture` 使用原 LangGraph 和真实模型；`mix seal` 逐项检查真实模型事件、原始 Prompt、输入、函数调用、独立测试和 Trace，至少 100 个不同 Prompt 才开放正式序列封存。`mix references` 复用已捕获的模型决策、重新执行真实 Tool 并校验；`mix reference-seal` 保留原始 Capture，另外封存环境参考。已有输出目录不会被覆盖。

本地 Tool 服务沿用 `experiments.exp1.capture.local_tool_service`。设置 `WABBER_SANDBOX_IMAGE` 为镜像 ID、`WABBER_DATA_ROOT` 为数据绝对路径、`ATSUITE_SESSION_ROOT` 为新的状态目录、`PROVIDER=''`，再启动 FaaS 服务。配置文件的 Tool endpoint 必须对应这个进程。

```sh
uv run python -m experiments.exp1.runner.mix capture \
  --manifest results/new-mix-preflight/manifest.json --config local-config.json \
  --output results/new-mix-captures --concurrency 4
```

## 到达序列与在线检查

```sh
uv run python -m experiments.exp1.data.prepare_mix_arrivals \
  --mix benchmarks/DataSciBench/tasks/mix --traffic results/phase3/traffic-10min-v1 \
  --output results/new-e2-cold --roles typical --experiment E2-A
uv run python -m experiments.exp1.data.prepare_mix_arrivals \
  --mix benchmarks/DataSciBench/tasks/mix --traffic results/phase3/traffic-10min-v1 \
  --output results/new-e2-stable --roles typical typical typical typical --composition stable
uv run python -m experiments.exp1.data.prepare_mix_arrivals \
  --mix benchmarks/DataSciBench/tasks/mix --traffic results/phase3/traffic-10min-v1 \
  --output results/new-e2-shift --roles typical typical high high low low --composition shift
```

序列保留每个 Copilot 窗口的原始间隔，明确记录窗口拼接，封存种子、任务分配和首次 Prompt/首次语义族标志。变化组成按到达前的语义族分组调整为 4:1 权重，不使用完成后的 DAG 或时长。

当前用户选定的稳定组成版本为 4 个 typical 窗口：40 分钟、764 个请求、100 个不同 Prompt，四个平台各执行一次共 3,056 次。这里是同一个已选择的 191 请求 typical 窗口重复拼接四次，各段任务分配使用固定种子独立打乱；不是四段独立采集的流量。序列位于 `results/stage8/e2-stable-4typical-v1`，云端格式位于 `results/stage8/cloud-stable-4typical-workload-v1`。用户恢复后已完成四个平台共 3,056 次正式执行，全部通过；旧的 1,444 请求序列仅保留为历史记录。

`stage8_online local` 通过原 open-loop scheduler 和严格 Replay 执行真实时序，并从空模型开始更新 Workload/Provider 模型。它只允许本地 FaaS；`local-config.json` 还需要 `local_pool`，准确填写 `provider: local`、镜像环境 ID、CPU/内存和并发上限。该入口不是多云实验入口。

```sh
uv run python -m experiments.exp1.runner.stage8_online local \
  --sequence results/new-e2-stable --config local-config.json \
  --output results/new-e2-local --duration-ms 30000 --concurrency 4
uv run python -m experiments.exp1.runner.stage8_online smoke --output results/new-cold-smoke
uv run python -m experiments.exp1.runner.stage8_online summarize \
  --events controller.jsonl --output online-summary.json
```

省略 `--duration-ms` 可执行完整封存序列。`smoke` 是另外的合成服务模型测试，不会填充或替代 DataSciBench-Mix 的真实时长。四平台云端适配、各 100 个任务的环境参考校准、严格 Pilot 和各 764 次正式固定平台运行已完成；完整账单对照与在线路由收敛实验仍未完成，详见 `experiments/exp1/STAGE8_VALIDATION.md`。
