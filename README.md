# 基于深度学习的股票收益预测与指数增强策略

量化笔试项目:使用 1000 只股票 2019-2023 的日度行情数据,构建 GRU 模型预测未来 5 个交易日收益,并实现周度调仓的指数增强策略回测。

研究报告(含标签定义、特征工程、模型设计、回测结果与实盘风险说明)见 [report.md](report.md)。

## 目录结构与文件说明

```
├── README.md                # 本文件
├── requirements.txt         # 依赖包及版本
├── report.md                # 研究报告(含实盘风险说明)
├── Makefile                 # 常用命令入口(make all / train / final / plots / package)
├── Quote.parquet            # 原始数据:1000 只股票 2019-2023 日度行情(随仓库提交)
├── src/
│   ├── data_loader.py       # 数据加载、复权因子计算、可交易标志、周最后交易日
│   ├── features.py          # 39 个特征与标签构造、逐日截面标准化、特征缓存
│   ├── model.py             # GRU(主模型)/ LSTM / MLP 模型定义
│   ├── train.py             # 滚动训练(--roll-months 参数化)+ 验证集 RankIC 早停
│   ├── backtest.py          # 周度调仓回测引擎(--buffer / --neutral / --smooth /
│   │                        #   --skip / --rebal-weeks / --self-check)
│   ├── plots.py             # IC 时序图与分组收益图
│   └── utils.py             # 截面预处理、IC 统计、最大回撤等工具函数
├── notebooks/
│   ├── eda.py               # 数据质量体检(停牌语义 / 复权 / 涨跌停 / VWAP)
│   ├── feature_ablation.py  # 特征组消融(Ridge 线性代理,报告 8.2 节)
│   ├── tune_smooth.py       # 信号平滑窗口验证集调参(最终版 smooth=40 的来源,报告 8.4 节)
│   └── tune_skip.py         # 分位剔除验证集调参(报告 8.3 节负结果实验)
└── results/                 # 训练与回测产物(见下表)
```

**results/ 文件说明**(`<tag>` 为策略配置变体,对照表见后):

| 文件 | 内容 |
|---|---|
| predictions.csv | 测试集(2023)每日全股票预测值,回测默认输入(内容同 predictions_gru.csv) |
| predictions_{gru,lstm,mlp}.csv | 三个模型各自的测试集预测值 |
| daily_rank_ic_{gru,lstm,mlp}.csv | 各模型测试集日度 RankIC 序列 |
| model_ic_comparison.csv | 模型 IC 对比汇总(报告 8.1 节表格) |
| feature_ablation.csv | 特征组消融结果(报告 8.2 节表格) |
| tune_smooth_val.csv | 平滑窗口验证集调参表(报告 8.4 节) |
| tune_skip_val.csv | 分位剔除验证集调参表(报告 8.3 节) |
| backtest_results`<_tag>`.csv | 各配置回测指标(年化超额 / IR / 回撤 / 换手 / 胜率) |
| nav_daily`<_tag>`.csv | 各配置组合与基准的日度收益序列 |
| equity_curve`<_tag>`.png | 各配置净值曲线图(组合 / 基准 / 超额) |
| ic_timeseries.png | 三模型日度 RankIC 移动平均与累计图(报告 8.1 节) |
| group_returns_gru.png | GRU 预测值分组收益图(报告 8.3 节) |
| cache/ | 特征面板缓存,不入库,可由 features.py 重新生成 |

**回测配置 tag 对照**:

| tag | 配置 | 报告章节 |
|---|---|---|
| (无后缀) | 基础版:周度调仓 Top200 等权 | 7 |
| buffer | + 换手缓冲(排名 350 内不卖出) | 7 |
| neutral | + 行业中性化 | 7 |
| buffer_neutral | 缓冲 + 中性 | 7 |
| **smooth40** | **缓冲 + 中性 + 40 日信号平滑(最终版)** | 7 / 8.4 |
| biweekly / monthly | 双周 / 月度调仓(频率实验) | 8.4 |
| skip100 | 剔除预测最高前 100 名(负结果留档) | 8.3 |
| roll3 | 季度滚动训练(敏感性分析) | 6 |

## 运行方式

**一键复现最终结果**(训练 GRU → 最终版回测 → 图表;GPU 约 15 分钟,CPU 约 2 小时):

```bash
pip install -r requirements.txt
make all      # 产出 results/backtest_results_smooth40.csv(年化超额 +3.9%, IR 1.08)与净值曲线
```

分步运行:

```bash
# 1. 数据体检(可选)
python notebooks/eda.py

# 2. 训练(gru / lstm / mlp / all;首次运行会构建特征缓存,约 2-3 分钟)
python src/train.py --model gru
python src/train.py --model gru --roll-months 3   # 季度滚动(默认 6=半年,12=不滚动)

# 3. 回测(默认读 results/predictions.csv)
python src/backtest.py
python src/backtest.py --self-check                          # 回测引擎自洽性检验
python src/backtest.py --buffer 350 --tag buffer             # 换手缓冲(加分项)
python src/backtest.py --neutral --tag neutral               # 行业中性化(加分项)
python src/backtest.py --buffer 350 --neutral --smooth 40 --tag smooth40  # 最终版
python src/plots.py                                          # IC 时序 + 分组收益图

# 4. 可选:特征消融 / 组合规则验证集调参
python notebooks/feature_ablation.py
python notebooks/tune_smooth.py     # 信号平滑窗口调参(最终版参数来源)
python notebooks/tune_skip.py
```

CPU 可运行;GPU 下(`--device cuda`,或默认 auto 自动检测)单模型两折约 2 分钟。

## 核心设计要点

- **标签**:`adjVWAP(T+6) / adjVWAP(T+1) - 1`,逐日截面标准化;T 日收盘出信号、T+1 日 VWAP 成交,标签与回测执行口径完全自洽
- **防泄露**:所有特征仅用 T 日及以前数据;截面预处理逐日进行;训练/验证/测试严格按时间切分,训练集尾部设 6 个交易日 embargo
- **滚动训练**:2023 年按半年滚动重训(扩张窗口),满足滚动训练/滚动验证要求
- **回测自洽检验**:全持仓 + 零成本时组合净值与基准严格一致(`--self-check`)
- **特征亮点**:从复权因子反推分红事件,构造股息率与送转强度因子(报告 4.3 节)
- **最终策略**:周度调仓 Top200 等权 + 换手缓冲(350)+ 行业中性化 + 40 日信号平滑,
  2023 年样本外年化超额 +3.9%、IR 1.08;组合参数仅用 2022 验证集调优(报告 8.4 节)
