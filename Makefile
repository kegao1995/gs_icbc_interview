# 量化笔试项目 Makefile
#
# 快速开始:
#   make install   安装依赖
#   make all       一键复现最终结果:训练 GRU -> 最终版回测 -> 图表
#                  (GPU 约 15 分钟 / CPU 约 2 小时;产出 results/backtest_results_smooth40.csv)
#
# 变量:PYTHON 可指定解释器,如 make train PYTHON=/path/to/python
PYTHON ?= python

.PHONY: all install eda train train-all backtest backtest-all self-check \
        buffer neutral final tune-smooth tune-skip ablation plots \
        package clean clean-all

# ---------------------------------------------------------------- 主流水线

# 完整流水线:训练主模型 -> 最终版回测(缓冲+行业中性+40日信号平滑) -> 图表
all:
	@echo "========== [1/3] 训练 GRU 主模型 =========="
	$(MAKE) train
	@echo "========== [2/3] 最终版回测 =========="
	$(MAKE) final
	@echo "========== [3/3] 生成图表 =========="
	$(MAKE) plots
	@echo "========== 全部完成:最终指标见 results/backtest_results_smooth40.csv,净值曲线见 results/equity_curve_smooth40.png =========="

install:
	$(PYTHON) -m pip install -r requirements.txt

# 数据质量体检(可选;停牌语义/复权/涨跌停/VWAP 检查)
eda:
	$(PYTHON) notebooks/eda.py

# ---------------------------------------------------------------- 训练

# 训练主模型(GRU,两折半年滚动),并将其预测设为回测默认输入。
# 注意:torch 2.13 + CUDA 在 Windows 上训练完成后、进程收尾瞬间可能
# 崩溃(0xC0000409),此时全部产物已正确落盘。故此处不以退出码判定成败,
# 而是训练前删除产物、训练后验证产物已重新生成(test -s),更为可靠。
train:
	@echo ">>> 清理旧预测产物(成败以产物是否重新生成判定,防止陈旧文件干扰)"
	rm -f results/predictions_gru.csv results/predictions.csv
	@echo ">>> GRU 两折滚动训练:首次运行先构建特征缓存约 3 分钟;训练 GPU 约 5 分钟、CPU 约 2 小时"
	@echo ">>> (Windows+CUDA 下训练结束瞬间可能报 0xC0000409,系 cuDNN 收尾竞态,产物已落盘,可忽略)"
	-$(PYTHON) src/train.py --model gru
	@echo ">>> 校验预测文件已由本次训练生成"
	test -s results/predictions_gru.csv
	@echo ">>> 将 GRU 预测设为回测默认输入 results/predictions.csv"
	cp results/predictions_gru.csv results/predictions.csv

# 训练全部模型(GRU/LSTM/MLP,报告 8.1 节模型对比);成败判定同上
train-all:
	rm -f results/predictions_gru.csv results/predictions_lstm.csv \
		results/predictions_mlp.csv results/predictions.csv
	-$(PYTHON) src/train.py --model all
	test -s results/predictions_gru.csv
	test -s results/predictions_lstm.csv
	test -s results/predictions_mlp.csv
	cp results/predictions_gru.csv results/predictions.csv

# ---------------------------------------------------------------- 回测

# 基础版:周度调仓 Top200 等权(题目基准口径)
backtest:
	$(PYTHON) src/backtest.py

# 回测引擎自洽性检验:全持仓+零成本时组合与基准偏差应为 0
self-check:
	$(PYTHON) src/backtest.py --self-check

# 加分项单项:换手缓冲(排名 350 内不卖出)/ 行业中性化
buffer:
	$(PYTHON) src/backtest.py --buffer 350 --tag buffer

neutral:
	$(PYTHON) src/backtest.py --neutral --tag neutral

# 最终版:缓冲 + 行业中性 + 40 日信号平滑(报告第 7 节主结果,超额 +3.89%)
final:
	@echo ">>> 最终版回测:周度调仓 Top200 等权 + 换手缓冲350 + 行业中性 + 40日信号平滑,双边千三成本(约 4 分钟)"
	$(PYTHON) src/backtest.py --buffer 350 --neutral --smooth 40 --tag smooth40

# 报告涉及的全部回测变体(第 7 节表格 + 8.3/8.4 节实验)
backtest-all: backtest buffer neutral final
	$(PYTHON) src/backtest.py --buffer 350 --neutral --tag buffer_neutral
	$(PYTHON) src/backtest.py --buffer 350 --neutral --rebal-weeks 2 --tag biweekly
	$(PYTHON) src/backtest.py --buffer 350 --neutral --rebal-weeks 4 --tag monthly
	$(PYTHON) src/backtest.py --buffer 350 --neutral --skip 100 --tag skip100

# ---------------------------------------------------------------- 分析与图表

# 信号平滑窗口验证集调参(最终版 smooth=40 的来源,报告 8.4 节;需先 train)
tune-smooth:
	$(PYTHON) notebooks/tune_smooth.py

# 分位剔除验证集调参(报告 8.3 节负结果实验;需先 train)
tune-skip:
	$(PYTHON) notebooks/tune_skip.py

# 特征组消融(Ridge 线性代理,报告 8.2 节)
ablation:
	$(PYTHON) notebooks/feature_ablation.py

# IC 时序图 / 分组收益图 / 特征重要性图(需先 train 与 ablation)
plots:
	@echo ">>> 生成图表:IC 时序、预测分组收益、特征重要性(约 1 分钟)"
	$(PYTHON) src/plots.py

# ---------------------------------------------------------------- 打包与清理

# 打包提交 zip:代码 + 报告 + 全部结果 + 原始数据(自包含,解压即可运行)
# 排除特征缓存与编译产物(--exclude 必须写在路径之前才生效)。
# 注意:zip 格式需要 bsdtar(Windows 10+ 系统自带,macOS 默认即是);
# Git Bash 的 GNU tar 不支持 zip,只会生成改名的 tar 包,故优先取系统 bsdtar。
TAR ?= $(shell [ -x /c/Windows/System32/tar.exe ] && echo /c/Windows/System32/tar.exe || echo tar)

package:
	$(TAR) -a --exclude "results/cache" --exclude "__pycache__" \
		-cf submission.zip README.md requirements.txt report.md Makefile \
		Quote.parquet src notebooks results

# 清理缓存与编译产物(不动 results 下的交付物)
clean:
	rm -rf results/cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f src/*.pyc notebooks/*.pyc

# 彻底清理:连训练/回测产出与打包文件一并删除(重跑用)
clean-all: clean
	rm -rf results
	rm -f submission.zip
