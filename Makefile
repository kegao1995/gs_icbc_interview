# 量化笔试项目 Makefile
#
# 快速开始:
#   make install   安装依赖
#   make all       一键复现最终结果:训练 GRU -> 最终版回测 -> 图表
#                  (GPU 约 15 分钟 / CPU 约 2 小时;产出 results/backtest_results_smooth40.csv)
#
# 变量:PYTHON 可指定解释器,如 make train PYTHON=/path/to/python
PYTHON ?= python

# 步骤说明打印:echo 输出裸字节,中文在 GBK 代码页的终端下会乱码;
# 改用 python print,消息写成 \u 转义(命令行纯 ASCII,任何代码页转手
# 都不会损坏;Python 经 Windows 宽字符 API 输出,中文在任意终端正确)。
# 每条 SAY 上方的顶格注释即消息中文明文。
SAY = @$(PYTHON) -c

.PHONY: all install eda train train-all backtest backtest-all self-check \
        buffer neutral final tune-smooth tune-skip ablation plots \
        package clean clean-all

# ---------------------------------------------------------------- 主流水线

# 完整流水线:训练主模型 -> 最终版回测(缓冲+行业中性+40日信号平滑) -> 图表
all:
# ========== [1/3] 训练 GRU 主模型 ==========
	$(SAY) "print('========== [1/3] \u8bad\u7ec3 GRU \u4e3b\u6a21\u578b ==========')"
	$(MAKE) train
# ========== [2/3] 最终版回测 ==========
	$(SAY) "print('========== [2/3] \u6700\u7ec8\u7248\u56de\u6d4b ==========')"
	$(MAKE) final
# ========== [3/3] 生成图表 ==========
	$(SAY) "print('========== [3/3] \u751f\u6210\u56fe\u8868 ==========')"
	$(MAKE) plots
# ========== 全部完成:最终指标 results/backtest_results_smooth40.csv | 净值曲线 results/equity_curve_smooth40.png ==========
	$(SAY) "print('========== \u5168\u90e8\u5b8c\u6210:\u6700\u7ec8\u6307\u6807 results/backtest_results_smooth40.csv | \u51c0\u503c\u66f2\u7ebf results/equity_curve_smooth40.png ==========')"

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
# >>> 清理旧预测产物(成败以产物是否重新生成判定,防止陈旧文件干扰)
	$(SAY) "print('>>> \u6e05\u7406\u65e7\u9884\u6d4b\u4ea7\u7269(\u6210\u8d25\u4ee5\u4ea7\u7269\u662f\u5426\u91cd\u65b0\u751f\u6210\u5224\u5b9a,\u9632\u6b62\u9648\u65e7\u6587\u4ef6\u5e72\u6270)')"
	rm -f results/predictions_gru.csv results/predictions.csv
# >>> GRU 两折滚动训练:首次运行先构建特征缓存约 3 分钟;GPU 约 5 分钟 / CPU 约 2 小时
# >>> (Windows+CUDA 下训练结束瞬间可能报 0xC0000409:cuDNN 收尾竞态,产物已落盘,可忽略)
	$(SAY) "print('>>> GRU \u4e24\u6298\u6eda\u52a8\u8bad\u7ec3:\u9996\u6b21\u8fd0\u884c\u5148\u6784\u5efa\u7279\u5f81\u7f13\u5b58\u7ea6 3 \u5206\u949f;GPU \u7ea6 5 \u5206\u949f / CPU \u7ea6 2 \u5c0f\u65f6\n>>> (Windows+CUDA \u4e0b\u8bad\u7ec3\u7ed3\u675f\u77ac\u95f4\u53ef\u80fd\u62a5 0xC0000409:cuDNN \u6536\u5c3e\u7ade\u6001,\u4ea7\u7269\u5df2\u843d\u76d8,\u53ef\u5ffd\u7565)')"
	-$(PYTHON) src/train.py --model gru
# >>> 校验预测文件已由本次训练生成
	$(SAY) "print('>>> \u6821\u9a8c\u9884\u6d4b\u6587\u4ef6\u5df2\u7531\u672c\u6b21\u8bad\u7ec3\u751f\u6210')"
	test -s results/predictions_gru.csv
# >>> 将 GRU 预测设为回测默认输入 results/predictions.csv
	$(SAY) "print('>>> \u5c06 GRU \u9884\u6d4b\u8bbe\u4e3a\u56de\u6d4b\u9ed8\u8ba4\u8f93\u5165 results/predictions.csv')"
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
# >>> 最终版回测:周度调仓 Top200 等权 + 换手缓冲350 + 行业中性 + 40日信号平滑,双边千三成本(约 4 分钟)
	$(SAY) "print('>>> \u6700\u7ec8\u7248\u56de\u6d4b:\u5468\u5ea6\u8c03\u4ed3 Top200 \u7b49\u6743 + \u6362\u624b\u7f13\u51b2350 + \u884c\u4e1a\u4e2d\u6027 + 40\u65e5\u4fe1\u53f7\u5e73\u6ed1,\u53cc\u8fb9\u5343\u4e09\u6210\u672c(\u7ea6 4 \u5206\u949f)')"
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
# >>> 生成图表:IC 时序、预测分组收益、特征重要性(约 1 分钟)
	$(SAY) "print('>>> \u751f\u6210\u56fe\u8868:IC \u65f6\u5e8f\u3001\u9884\u6d4b\u5206\u7ec4\u6536\u76ca\u3001\u7279\u5f81\u91cd\u8981\u6027(\u7ea6 1 \u5206\u949f)')"
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
