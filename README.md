# 基于深度学习的股票收益预测与指数增强策略

量化笔试项目:使用 1000 只股票 2019-2023 的日度行情数据,构建 GRU 模型预测未来 5 个交易日收益,并实现周度调仓的指数增强策略回测。

研究报告(含标签定义、特征工程、模型设计、回测结果与实盘风险说明)见 [report.md](report.md)。

## 目录结构

```
├── README.md                # 本文件
├── requirements.txt         # 依赖包及版本
├── report.md                # 研究报告
├── 量化候选人笔试项目文件/    # 原始数据(Quote.parquet)
├── src/
│   ├── data_loader.py       # 数据加载、复权因子、可交易标志
│   ├── features.py          # 特征工程与标签构造(29 个量价特征)
│   ├── model.py             # GRU / LSTM / MLP 模型定义
│   ├── train.py             # 滚动训练 + 早停
│   ├── backtest.py          # 周度调仓回测与评价
│   └── utils.py             # 截面预处理、IC、回撤等工具函数
├── notebooks/
│   └── eda.py               # 探索性数据分析(数据体检)
└── results/
    ├── predictions.csv      # 测试集(2023)每日全股票预测值(GRU)
    ├── backtest_results.csv # 回测指标
    ├── equity_curve.png     # 净值曲线(组合/基准/超额)
    └── ...                  # 模型对比、IC 序列、消融实验等
```

## 运行方式

```bash
pip install -r requirements.txt

# 1. 数据体检(可选)
python notebooks/eda.py

# 2. 训练(gru / lstm / mlp / all;首次运行会构建特征缓存,约 2-3 分钟)
python src/train.py --model gru

# 3. 回测(默认读 results/predictions.csv)
python src/backtest.py
python src/backtest.py --self-check          # 回测引擎自洽性检验
python src/backtest.py --buffer 350 --tag buffer   # 换手缓冲版本(加分项)
python src/backtest.py --neutral --tag neutral     # 行业中性化版本(加分项)
```

CPU 即可运行(无需 GPU);GRU 单折训练约 30-60 分钟。

## 核心设计要点

- **标签**:`adjVWAP(T+6) / adjVWAP(T+1) - 1`,逐日截面标准化;T 日收盘出信号、T+1 日 VWAP 成交,标签与回测执行口径完全自洽
- **防泄露**:所有特征仅用 T 日及以前数据;截面预处理逐日进行;训练/验证/测试严格按时间切分,训练集尾部设 6 个交易日 embargo
- **滚动训练**:2023 年按半年滚动重训(扩张窗口),满足滚动训练/滚动验证要求
- **回测自洽检验**:全持仓 + 零成本时组合净值与基准严格一致(`--self-check`)
