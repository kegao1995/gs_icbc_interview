# 基于深度学习的股票收益预测与指数增强策略

量化笔试项目:使用 1000 只股票 2019-2023 的日度行情数据,构建 GRU 模型预测未来 5 个交易日收益,并实现周度调仓的指数增强策略回测。

研究报告(含标签定义、特征工程、模型设计、回测结果与实盘风险说明)见 [report.md](report.md)。

## 目录结构

```
├── README.md                # 本文件
├── requirements.txt         # 依赖包及版本
├── report.md                # 研究报告
├── Quote.parquet            # 原始数据(随仓库提交,clone 后可直接运行)
├── src/
│   ├── data_loader.py       # 数据加载、复权因子、可交易标志
│   ├── features.py          # 特征工程与标签构造(39 个特征,含复权因子反推的分红/送转因子)
│   ├── model.py             # GRU / LSTM / MLP 模型定义
│   ├── train.py             # 滚动训练 + 早停(同时保存验证期预测)
│   ├── backtest.py          # 周度调仓回测与评价(buffer / neutral / skip / self-check)
│   ├── plots.py             # IC 时序图、分组收益图
│   └── utils.py             # 截面预处理、IC、回撤等工具函数
├── notebooks/
│   ├── eda.py               # 探索性数据分析(数据体检)
│   ├── feature_ablation.py  # 特征组消融(Ridge 代理)
│   └── tune_skip.py         # 组合规则验证集调参(报告中的负结果实验)
└── results/
    ├── predictions.csv      # 测试集(2023)每日全股票预测值(GRU)
    ├── backtest_results.csv # 回测指标(另有 _buffer/_neutral/_buffer_neutral/_skip100 变体)
    ├── equity_curve.png     # 净值曲线(组合/基准/超额)
    └── ...                  # 模型对比、IC 序列、消融、调参表等
```

## 运行方式

```bash
pip install -r requirements.txt

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
