# 量化笔试项目 Makefile
# 用法: make install / make train / make backtest / make plots / make clean
PYTHON ?= python

.PHONY: all install eda train train-all backtest self-check buffer neutral \
        plots ablation package clean clean-all

# 完整流水线:训练主模型 -> 回测 -> 图表
all: train backtest plots

install:
	$(PYTHON) -m pip install -r requirements.txt

eda:
	$(PYTHON) notebooks/eda.py

# 训练主模型(GRU)并将其预测设为回测输入
train:
	$(PYTHON) src/train.py --model gru
	cp results/predictions_gru.csv results/predictions.csv

# 训练全部模型(GRU/LSTM/MLP,用于模型对比加分项)
train-all:
	$(PYTHON) src/train.py --model all
	cp results/predictions_gru.csv results/predictions.csv

backtest:
	$(PYTHON) src/backtest.py

self-check:
	$(PYTHON) src/backtest.py --self-check

# 加分项:换手缓冲 / 行业中性化
buffer:
	$(PYTHON) src/backtest.py --buffer 350 --tag buffer

neutral:
	$(PYTHON) src/backtest.py --neutral --tag neutral

plots:
	$(PYTHON) src/plots.py

ablation:
	$(PYTHON) notebooks/feature_ablation.py

# 打包提交(排除原始数据、缓存、交接文件;Windows 10+ 自带 bsdtar 支持 zip)
package:
	tar -a -cf submission.zip README.md requirements.txt report.md Makefile \
		src notebooks results --exclude "results/cache" --exclude "__pycache__"

# 清理缓存与编译产物(不动 results 下的交付物)
clean:
	rm -rf results/cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f src/*.pyc notebooks/*.pyc

# 彻底清理:连训练/回测产出一并删除
clean-all: clean
	rm -rf results
	rm -f submission.zip
