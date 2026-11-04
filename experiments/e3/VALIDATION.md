# Stage 9 验收记录

当前实验协议已收敛为 Fixed（单一 Provider）与 Online＋Exploration 两组，不含 Static、独立纯 Online 或财务预算。计价统一复用 ATSuite；本文中的旧策略计划、账单等待状态和测量记录保留为历史证据，现行入口见 [README.md](README.md)。

本记录区分官方业务验证、本地 Agent 执行、云端配对 Pilot 和长期实验。Stage 9 尚未完成整体验收；本轮补齐官方评估接入、三个真实 Review 项目及四平台小规模实测。

所有结果路径相对于仓库根目录。`results/` 被 gitignore 忽略，当前工作区可以直接检查；跨机器需按 README 重建。四平台具体测量和解释见 [PROFILING.md](PROFILING.md)。

## TravelPlanner 官方业务验证

使用官方验证集按难度出现的首个原始任务 0/60/120，保持原 Prompt、数据库与约束。来源及校验和在 `travel_source.lock.json`，官方代码固定为 `e52c87f4ac348a3410c46dc3553c519db5ec5e23`。评估在独立环境 `results/e3-eval-env` 中执行未修改的 commonsense/hard constraint 检查；先核验代码和数据，关闭遥测。未指定的 hard constraint 按不适用处理。模型收到官方 planner instruction 和通用序列化要求，不收到独立验证反馈。

| 原始任务 | 实际 Agent 证据 | 官方结果 |
|---|---|---|
| validation-0 / easy | `results/e3-travel-v2/pro-validation-0` | 输出格式、所有适用 commonsense/hard constraint 检查通过 |
| validation-60 / medium | `results/e3-travel-v2/pro-validation-60` | 失败：官方要求闭环行程，当前计划没有返回出发城市 |
| validation-120 / hard | `results/e3-travel-v2/pro-validation-120` | 失败：重复早餐餐厅及超出预算；其他适用检查通过 |

这三次使用 DeepSeek Pro 的真实原生 Tool Calling 和 thinking；另外保留 Flash 的成功 easy Capture `batch-validation-0` 及先前失败尝试。没有将 medium/hard 的失败 Capture 纳入成功 Pilot catalog，也未根据云端结果更换 Pilot 任务。easy 的同一成功 Capture 已在四个平台重新执行 Tool 并核对结果。当前证据证明官方评估和一个有效旅行请求可以完成，不代表三个难度的规划成功率已达标。

## 多真实 Review 项目

`review_projects.py` 使用 Requests、ItsDangerous、Click 的固定真实上游代码，注入可复现的回归 PR；这些是实验构造的变更，不是历史上游 PR。完整 Git 仓库、文件工具和 Git 工具实际执行，独立回归确认 base 通过、head 失败。私有缺陷与测试元数据放在可审查仓库之外。

| 项目 / PR | 固定上游 revision | 独立缺陷 | 成功 live Capture |
|---|---|---|---|
| Requests / 201 | `0e322af87745eff34caffe4df68456ebc20d9068` | 合并设置时错误保留空值并移除非空值 | `results/e3-review-projects/bounded-requests-201` |
| ItsDangerous / 202 | `096c8d42545d3b68ea21a4f890fb2b2d8979c0bd` | token 年龄比较反向，过期签名被接受 | `results/e3-review-projects/final-itsdangerous-202` |
| Click / 203 | `934813e4d421071a1b3db3973c02fe2721359a6e` | ANSI 转义被误算作终端显示宽度 | `results/e3-review-projects/bounded-click-203` |

三者分别在 `results/e3-review-projects/replay-{requests-201,itsdangerous-202,click-203}` 完成严格本地 Replay，汇总为 `replay-summary.json`。四平台 Pilot 固定使用 ItsDangerous Capture。验证核对缺陷行、解释关键概念、无额外发现及独立回归证据；不能完整评价自然语言审查质量。

项目选择由 Harness 在请求开始时写入独立状态，模型不能更换项目、UID 或注入 SDK 快照。文件读取排除越界符号链接，文本搜索跳过二进制；Git 时间统一 ISO 格式，容器 Git 只信任被选 Fixture 目录。当前三个 PR 均为小变更，尚未覆盖大 PR、复杂跨文件缺陷和真实历史 PR 分布。

## 四 Workload / 四平台

Pilot 任务在 `results/e3-cloud/pilot-tasks.json` 预先固定：DataSciBench bcb1011、TravelPlanner validation-0、ItsDangerous 202、ScientificComputation normal-8-1。每任务每平台三次执行，固定 Capture、输入、真实 Tool DAG 与 LLM idle；每次生成新的请求状态。既有 DataSci 环境参考保留原始 live Capture 谱系及原有校准验证。

三个新 Workload 复用原 Tool，打包为 E3Tools 共享执行池；DataSci 使用既有四平台部署。镜像审计核对四个镜像中各 521 个打包文件，包含 SDK、Tool、数据库、Git 对象及引用；AgentCore ARM 镜像具有实际可执行的 Git。证据在 `results/e3-cloud/fixed-image-audit.json`，实际资源和原生策略在 `deployed-*.json`。FaaS 每次 Tool 恢复/提交状态，MCP 使用独立 Session 驻留状态，没有跨 Provider 迁移或主动实例扩缩容。

ScientificComputation 的跨平台 BLAS 舍入误差以 rel_tol=1e-10、abs_tol=1e-12 比较；结构、文本和标志仍严格核对，最终结果另经独立线性求解、残差和存储读回验证。其他新 Workload 使用严格业务结果比较。

云端修复前后的 cohort 独立保留，最终成功率、延迟、资源及费用缺口见 PROFILING.md。没有把旧失败覆盖为成功，也没有用价格模型填充实际费用。

## 自动检查

`results/e3-cloud/final-checks-v3.log` 记录 145 项回归检查：144 项通过、1 项跳过，同时包含 MCP residency 的 7 项及 16 个独立进程的驻留检查。收尾新增原生环境配置检查后，E3 共 19 项；E3 与 E1-A 的 42 项针对性回归全部通过，见 `post-pilot-checks-final.log`。E3 覆盖原生批次、项目状态隔离、输入边界、并发清理、数值误差边界及不完整账单拒绝收益计算。旧 E1-A Replay 测试手工绕过构造器，已补齐新增的输入比较器初始化。

检查沿用 `tests.test_e3`、LangGraph、routing、tracing、workload、logical_state、runtime_refactor、stage8、standard_cost、joint_control 和 mcp_residency 等既有 unittest 模块。新模块经过 Ruff F/I、格式与 compileall 检查，补丁经过 `git diff --check`。主环境由 `uv sync --frozen --group aws` 恢复，官方评估依赖单独锁定于 `evaluation.requirements.lock`。

## 尚未达到的整体验收项

1. 原开发题 validation-60/120 的有效规划结果及更广任务覆盖；本轮另选的六道正式题（含 medium/hard）已通过，见 TEN_MINUTE.md。
2. Review 的大变更、多文件和历史 PR 覆盖；所有候选任务逐项跨云验证。
3. 可归属到本次执行、覆盖原生伸缩、冷启动、Session 保留和对象存储的完整实际费用。账户日账单不等于本实验账单。
4. 正式封存的长期 Copilot 混合流及七策略独立运行、在线反馈与账户级预算集成。当前 Pilot catalog 不代表该长期实验已执行。
5. 最佳固定 Provider 的完整成本比较及 Wabber 收益归因；费用缺失时保持 total/savings 为 null。

保留失败、费用未知及缺少优势交叉的结果，不能因实验结果不理想重写正式任务。

## 十分钟正式 Trace 补充

按用户要求，新增池与十分钟流见 [TEN_MINUTE.md](TEN_MINUTE.md)。正式池固定为每类 6 个不同任务，共 24 个模板；原始 Copilot 首个连续十分钟的 213 条请求完整保留，分配为 DataSci 64、Travel 51、Review 53、Scientific 45 条。

18 个新增模板已完成本轮真实 Capture、独立验证和严格本地 Replay；6 个 DataSci 模板复用此前经验证的环境参考和原始 live 谱系。本轮另有两个独立 MCP Session Replay。最终封存在 `results/e3-ten-minute/trace`，未运行整条正式流、七策略实验或新增任务的云端部署，不能把这份 Trace 当成长期成本收益结果。
