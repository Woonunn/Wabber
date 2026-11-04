# 请求序列累计成本图

`plot_cumulative_cost.py` 绘制四个固定 provider 基线与 Wabber 动态路由的累计成本。
样式参考 ATSuite EuroSys Submission 的 Figure 11：Aliyun 浅橙 `#FACEB1`、
AWS 浅蓝 `#ACC4E8`、细黑边框和浅色网格。虚线/圆点表示 FaaS，实线/方点表示
保留 Session 的平台；Wabber 使用 Figure 9 中的深红 `#B2182B`、加粗实线和菱形标记。
字体优先使用论文的 Arial；未安装时依次回退到 Liberation Sans、DejaVu Sans。
默认尺寸为单栏 3.5 × 2.8 英寸，无图内标题，适合由论文 caption 解释实验设置。

复制 `experiments/exp1/data/cumulative_cost.template.csv`，在表头后填入真实数据。
模板只含表头，不包含模拟结果。

| 列 | 含义 |
| --- | --- |
| `request_index` | 从 1 开始连续递增的请求序号，按实验的逻辑请求顺序排列 |
| `ali_fc_usd` | 固定使用 Aliyun FC 时，该请求新增的美元成本 |
| `ali_agentrun_usd` | 固定使用 Aliyun AgentRun 时，该请求新增的美元成本 |
| `aws_lambda_usd` | 固定使用 AWS Lambda 时，该请求新增的美元成本 |
| `aws_agentcore_usd` | 固定使用 AWS AgentCore 时，该请求新增的美元成本 |
| `wabber_usd` | Wabber 实际路由执行该请求新增的美元成本 |

每行必须对应五种策略下的**同一个请求**，五列均填**单次增量成本**。
脚本计算 `C(k) = sum(cost[i] for i in 1..k)`，并加入 `(0, 0)` 起点；
使用完整序列连线，不平滑、不下采样数据，也不根据结果调整曲线次序或成本。
零成本允许填 `0`；空值、负值、NaN、无穷大、请求序号缺失或重复会报错。
不要将已经累计的成本再次填入模板。

所有输入必须已统一为 USD。人民币成本需先按实验明确记录的汇率换算；
脚本不查询价格或汇率。各策略应采用一致的费用口径，包含实验要计入的计算、
状态同步/存储、请求、空闲 Session 和路由开销；Wabber 列填实际策略总开销，
不能填四个基线逐请求的最小值。
并发执行也按同一逻辑请求序号配对，而非各 provider 的完成顺序。
Session 尾部/共享费用需在上游按明确规则分配，确保每列求和与对应实验总成本一致。
因此曲线表示前 k 个逻辑请求的已归属成本，不是第 k 个请求到达时刻的实时账单。

在仓库根目录执行（Matplotlib 已在现有 `dev` 依赖组中）：

```bash
uv run --group dev python -m experiments.exp1.analysis.plot_cumulative_cost \
  --input results/cumulative_cost.csv \
  --output experiments/exp1/figure/cumulative_cost/cumulative_cost
```

输出同名 `.pdf`、`.svg`、`.png`；PDF 为矢量图并嵌入 TrueType 字体，PNG 为 400 DPI。
若任一目标文件已存在，脚本拒绝覆盖；重绘时指定新的输出文件名。
数据尚未填入时运行会明确报错，不生成空结果图。

检查解析、累计、输入校验及三种格式的渲染：

```bash
uv run --group dev python -m unittest tests.test_cumulative_cost_plot -v
```
