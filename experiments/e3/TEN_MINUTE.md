# E3 正式任务池与十分钟 Trace

当前实验协议已收敛为 Fixed（单一 Provider）与 Online＋Exploration 两组，不含 Static、独立纯 Online 或财务预算。计价统一复用 ATSuite；本文中的旧策略计划、账单等待状态和测量记录保留为历史证据，现行入口见 [README.md](README.md)。

本轮设计固定于 `ten_minute.plan.json`。已封存 213 条逻辑请求、24 个不同任务模板；18 个新增模板完成本轮真实 Capture 和严格本地 Replay，6 个复用已验证的 DataSciBench 环境参考及原始 live 谱系。没有把开发 Capture 或配对 Pilot 的重复次数当作正式请求数量。

## 预先声明的任务与到达流

使用 Copilot 原始时序首个连续窗口 `[2026-06-01T00:00:00Z, 00:10:00Z)`，来源为 `results/phase3/arrivals-v1/arrivals.csv`。保留全部 213 条记录、原始 arrival_id、时间戳、突发和初始/末尾 idle，不抽样、不缩放、不合成到达时间。最后一次到达在 594.574 秒，窗口长度 600 秒；请求执行可以在窗口结束后正常排空。

混合分配种子为 9，四类权重均为 1。不是强行调整为每类精确 25%：实际为 DataSciBench 64、TravelPlanner 51、ClaudeCodeReview 53、ScientificComputation 45 条。每类六个不同任务，共 24 个模板，每个模板被使用 4–16 次。请求数为 213，独立任务数为 24；同一模板复用已冻结的真实 LLM Capture，所有部署策略消费完全相同的逻辑请求。

| Workload | 正式任务 ID | 与本轮 E3 Pilot 的区分 |
|---|---|---|
| DataSciBench | bcb1024、bcb1035、bcb1037、bcb1043、bcb105、bcb1061 | 不使用 Pilot 的 bcb1011；复用已验证环境参考及原始 live 谱系 |
| TravelPlanner | validation-1、2、61、62、121、122 | 每个难度取 Pilot 后紧接的两题；0/60/120 为开发验证，不进入正式池 |
| ClaudeCodeReview | requests-301/302、itsdangerous-303/304、click-305/306 | 原 201/202/203 为 Pilot；正式变更覆盖第二类缺陷及双文件组合 |
| ScientificComputation | normal-{16-1,32-4,96-8}-seed-{1001,1002} | 不使用 Pilot 的 normal-8-1；固定六份输入，不增加无用计算 |

这里按任务 ID 分开 E3 Pilot 与正式池，不声称项目、缺陷家族或此前 Stage 8 的 DataSci 学习数据完全隔离。Review 的组合 PR 会包含同项目的已知缺陷类别；不是跨项目泛化或历史真实 PR 基准。所有 Review 变更明确标为实验注入，单文件一个提交或双文件两个提交，不能称为大 PR 覆盖。

## Capture、状态与独立验证

原始 Travel 用户 Prompt、官方数据库和评估代码不变。Agent 增加通用行程连续性、预算、住宿容量/入住夜数和约束检查提示，验证反馈不返回模型。官方验证集与 evaluator 的 revision/hash 沿用 `travel_source.lock.json`。失败尝试完整保留，不根据云端表现替换任务。

Review 在固定 Requests、ItsDangerous、Click 上游代码上构造可复现 Git 分支，新增流剩余长度、压缩载荷解码、范围端点比较缺陷，以及与已有缺陷的双文件组合。独立回归逐条确认 base 通过、head 失败。评分还检查引入缺陷位置与解释，不把人工 oracle 放进可读仓库。

科学任务使用 16/32/96 维正定矩阵及 1/4/8 个 RHS。新增输入加载 Tool 只从固定问题目录读取 A/B，拒绝路径穿越、越界符号链接及非有限矩阵。复用原求逆、矩阵乘积和张量状态；可返回已存储结果的名称/形状，最终验证实际读回完整解，并独立计算 solve 与残差。没有用摘要替代计算，也不靠 LLM 抄写大数组来增加耗时。

含状态写入的原生 Tool 批次依照返回顺序执行，避免同批读取观察到部分写入；不含已声明写访问的批次继续并发，旧 Capture 缺少访问元数据时保留既有行为。新科学任务的 tool_access 来自已有 SDK Tool 定义。最初科学 cohort 曾发现并发读写 Replay 分歧，已保留该失败并修复执行顺序，重新采集整个六任务 cohort，不能只筛掉出问题的那一题。

旧 Pro 开发 cohort 和最终 Flash Travel cohort 分开记录；同一版本不向模型暴露验证反馈。Capture 每任务每版本最多三次，选择第一个独立验证成功的执行；不把验证反馈拼入下一次模型消息。API 错误、业务失败和兼容性失败都保留。DataSci 原环境参考不是新增 live LLM 调用，必须保留其原始 Capture 谱系。混合种子 9 与 DataSci 原执行种子 0 是不同概念；冻结 Harness 配置保留原执行种子。

## 复现入口

先按 README 准备官方 Travel 数据、独立 evaluator 环境及三个固定上游项目，然后运行：

```bash
python -m experiments.e3.travel_prepare --prepared results/e3-travel-v2 \
  --indices 0 60 120 1 2 61 62 121 122 --output results/e3-ten-minute/travel
python -m experiments.e3.review_projects --sources results --expanded \
  --output results/e3-ten-minute/review
python -m experiments.e3.prepare --scientific-only \
  --output results/e3-ten-minute/scientific-v2
```

使用各 Workload 本地 Tool 服务和 `experiments.e3.corpus` 按声明任务列表运行 Capture（Travel 最终使用 Flash、16 轮上限；Review/科学使用 Pro、32 轮上限）；模型凭据仅从环境读取。科学服务设置 ATSUITE_SCIENTIFIC_PROBLEMS 指向输入目录，Review 设置 ATSUITE_REVIEW_PROJECTS 指向项目目录。Replay 沿用 capture_agent_traces 的 `--replay`，并重新真实执行 Tool，不重新请求 LLM。

用独立目录准备运行包装，保留原 Pilot 数据和部署：

```bash
python -m experiments.e3.deployment --travel results/e3-ten-minute/travel \
  --review results/e3-ten-minute/review \
  --scientific results/e3-ten-minute/scientific-v2/ScientificComputation.json \
  --capture results/e3-ten-minute/scientific-captures-v2/normal-16-1-seed-1001/attempt-0 \
  --benchmark results/e3-ten-minute/runtime/E3Tools
python -m experiments.e3.sequence --plan experiments/e3/ten_minute.plan.json \
  --receipts results/e3-ten-minute/travel-captures-v2/receipts.json \
             results/e3-ten-minute/review-captures/receipts.json \
             results/e3-ten-minute/scientific-captures-v2/receipts.json \
  --output results/e3-ten-minute/trace
```

封存包括原始时序、计划、任务输入、所有选定 Capture、原始 DataSci 谱系、官方 evaluator、SDK/Agent/路由源码、依赖锁和新的打包配置。任何变化都应触发校验失败。所有策略共享 manifest/seal，按 requests.csv 可直接逐条检查分配。

此步骤准备 Trace 与执行环境，不构建/部署新的四平台镜像，不运行七种策略。新的科学加载 Tool 和 Review 分支需要新的云端打包；旧 Pilot 镜像不具备这些新增任务。DataSci 的六个任务也需要在各镜像中复核输入资产。实际镜像审计、费用归属和七策略实测属于后续阶段，不能用本轮本地结果代替。

## 本轮实际验收与产物

- `results/e3-ten-minute/trace/manifest.json`、`seal.json`：正式逻辑请求与完整文件哈希；`requests.csv` 为逐请求可读索引，`summary.json` 记录数量和分布。
- `results/e3-ten-minute/trace-agents.json`：从实际 Capture 提取的 Harness 配置，保留 DataSci 原执行种子 0；`trace-environment.json` 记录本地版本。
- `results/e3-ten-minute/validation.json`、`corpus.csv`：24 个模板的来源、实际 Tool 调用/LLM idle、验证与 Replay 依据。
- `results/e3-ten-minute/E3-10m.tar.gz`、`E3-10m.sha256`：按仓库相对路径保存的封存包及校验和。恢复时在 Wabber checkout 中解压；当前 manifest 使用 `/root/Wabber` 绝对路径，换目录需重新封存，不能直接修改旧 seal。
- `results/e3-ten-minute/runtime/E3Tools`：新增任务的不可变部署包装；这里只生成输入与配置，尚未构建或部署镜像。

六个正式 Travel 任务全部通过未修改的官方 evaluator，并完成严格本地 Replay。其最终 Capture 共 7 次尝试：validation-122 首次失败、第二次通过，其余首次通过。此前 Pro 开发尝试仍保留：validation-60 三次均被闭环检查拒绝，validation-120 存在预算失败；无直飞时反复搜索中转的执行已中断并保留结果。随后增加通用地面交通策略，以统一 Flash 配置对原六道正式题重新采集，未换题、未更改数据库或降低验证条件；不据此宣称原始任务全集成功率为 100%。

六个 Review 任务完成真实 Capture 和严格本地 Replay，共 7 次尝试；ItsDangerous 组合任务首次遇到输出解析错误，下一次完成。三项目回归证明在 `review/regressions.json`，单文件一提交、双文件两提交。六个科学任务在修正状态批次顺序后全部首次通过 Capture 和本地 Replay；早期 cohort 中的一次 Replay 竞争失败仍保留。独立 MCP Session Replay 另外验证了 96 维科学任务和 ItsDangerous Review，驻留状态与 FaaS 恢复/提交语义均按原 SDK 执行。

`regression.log` 记录主回归 170 项：169 通过、1 跳过；状态批次修复后的 `final-tests.log` 记录 32 项针对性检查全部通过，封存检查另见 `seal-tests-final.log`。新逻辑覆盖原始到达精度/边界、失败留存、首个有效 Capture 选择、不可变问题输入路径、摘要结果的实际存储与独立数学验证、状态批次顺序及配置/设计文件篡改拒绝。部署包装也已在临时目录中实际准备并核对六份科学输入和固定数据库。

十分钟仅定义请求到达窗口。封存时尚未执行完整云端调度；后续已完成 24 模板×四平台预检及四组 213 请求 Fixed 基线，执行结果与限制见 [REAL_TEST.md](REAL_TEST.md)。三种路由策略和本批可归属完整费用仍未完成；旧 Pilot 保留于 PROFILING.md。
