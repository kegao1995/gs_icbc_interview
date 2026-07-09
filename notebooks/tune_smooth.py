# -*- coding: utf-8 -*-
"""信号平滑窗口调参:在 2022 验证集上网格搜索 smooth(预测值滚动均值窗口)

动机:调仓频率实验显示 5 日预测信号在数周内衰减缓慢(信号自相关高),
周度重排大量消耗在排名边缘的无效换手上。将排序分数替换为过去 N 个
交易日预测值的滚动均值,可在保持周度调仓(题目要求)的同时把换手
压至低频水平,且滚动均值对日度预测做了时间维度的集成,自身具有降噪作用。

平滑窗口仅用验证集预测调优,测试集(2023)只做最终一次评估。

运行: python notebooks/tune_smooth.py   (需先 train.py 生成 predictions_val_gru.csv)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_quote  # noqa: E402
from backtest import (prepare_market, neutralize_by_industry, simulate,
                      evaluate)  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BUFFER = 350
SMOOTH_GRID = [1, 5, 10, 20, 40, 60, 80]


def main():
    quote = load_quote()
    mkt = prepare_market(quote)
    pred_df = pd.read_csv(os.path.join(RESULTS_DIR, "predictions_val_gru.csv"),
                          parse_dates=["TradingDay"])
    # 两折验证期重叠部分保留 fold2 预测(训练窗口更长)
    pred_df = pred_df.drop_duplicates(subset=["TradingDay", "StockID"], keep="last")
    pred_n = neutralize_by_industry(pred_df, quote)

    bench_ret, _ = simulate(pred_df.assign(pred=0.0), mkt,
                            n_top=len(mkt["stocks"]), cost_per_side=0.0)

    rows = {}
    for smooth in SMOOTH_GRID:
        port_ret, turnovers = simulate(pred_n, mkt, buffer=BUFFER, smooth=smooth)
        m = evaluate(port_ret, bench_ret.reindex(port_ret.index), turnovers)
        rows[smooth] = m
        print(f"smooth={smooth:3d}  超额={m['年化超额收益']:+.4f}  IR={m['信息比率']:+.3f}  "
              f"换手={m['年化单边换手率']:.1f}  胜率={m['月度胜率']:.2f}")

    df = pd.DataFrame(rows).T
    df.index.name = "smooth"
    df.to_csv(os.path.join(RESULTS_DIR, "tune_smooth_val.csv"), encoding="utf-8-sig")
    print("saved ->", os.path.join(RESULTS_DIR, "tune_smooth_val.csv"))


if __name__ == "__main__":
    main()
