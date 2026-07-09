# 交接清单(提交前删除本文件)

办公电脑 CPU 太弱,训练中途终止。代码已全部完成并验证,回家机器上按顺序执行:

## 1. 环境

```bash
pip install -r requirements.txt
# 有 NVIDIA 显卡的话 torch 装 CUDA 版更快(train.py 目前用 CPU,
# 如需 GPU 把 train_fold 的 device 参数改为 "cuda" 即可)
```

数据:`量化候选人笔试项目文件/Quote.parquet` 已随仓库提交,clone 下来即可直接跑。

## 2. 训练(核心,必跑)

```bash
python src/train.py --model gru     # 主模型,CPU 约 1-1.5h,好机器更快
python src/train.py --model lstm    # 加分项对比,可选
python src/train.py --model mlp     # 加分项对比,很快,建议跑
```

产出:`results/predictions_{model}.csv`、`results/daily_rank_ic_{model}.csv`、`results/model_ic_comparison.csv`(自动汇总磁盘上所有已训模型)。

参考:办公电脑上 GRU 验证集 RankIC——fold1 峰值 0.0478、fold2 峰值 0.0543,均在 epoch 1 出现(早停会自动选中)。如果家里跑出来量级差不多,说明一切正常。

## 3. 回测与图表

```bash
cp results/predictions_gru.csv results/predictions.csv   # 指定主模型预测
python src/backtest.py                                   # 主回测
python src/backtest.py --self-check                      # 引擎自洽检验(应输出 0)
python src/backtest.py --buffer 350 --tag buffer         # 加分项:换手缓冲
python src/backtest.py --neutral --tag neutral           # 加分项:行业中性化
python src/plots.py                                      # IC 时序图 + 分组收益图
```

## 4. 报告收尾

`report.md` 已写好框架和方法论,搜索"占位"填入 4 处实际数字/图表:

1. 第 1 节:核心结论(测试集 RankIC/ICIR、年化超额、IR、超额回撤、换手、月度胜率)
2. 第 7 节:回测结果表(backtest_results.csv)+ 净值曲线图
3. 第 8 节:模型对比表(model_ic_comparison.csv)、buffer/neutral 对比、两张图;特征消融结果已在 `results/feature_ablation.csv`(波动组单独 RankIC 5.0% 最强、动量组 2023 年负贡献,可直接引用)
4. 第 9 节:换手率处把"约 25 倍(占位)"换成实际值

## 5. 提交

删除本文件,核对提交清单(README/requirements/src/notebooks/results/report),打包 zip 或推 GitHub(数据与 cache 已被 .gitignore 排除,体积无忧)。
