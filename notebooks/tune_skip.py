# -*- coding: utf-8 -*-
"""组合构建规则调参:在 2022 验证集上网格搜索 skip(剔除预测最高的前 N 名)

动机:分组收益诊断显示预测最高分位的股票短期均值回归(刚暴涨的高波股),
纯多头组合买入该分位反而拖累收益。skip 参数只用验证集预测调优,
测试集(2023)仅做最终一次评估,避免测试集数据窥探。

注意:验证集预测来自以该验证集早停的模型,存在轻微乐观偏差,报告中说明。

运行: python notebooks/tune_skip.py   (需先 train.py 生成 predictions_val_gru.csv)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_quote  # noqa: E402
from backtest import (prepare_market, neutralize_by_industry, simulate,
                      evaluate)  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BUFFER = 350          # 换手缓冲,已单独验证有效,固定
SKIP_GRID = [0, 25, 50, 100, 150, 200]


def main():
    quote = load_quote()
    mkt = prepare_market(quote)
    pred_df = pd.read_csv(os.path.join(RESULTS_DIR, "predictions_val_gru.csv"),
                          parse_dates=["TradingDay"])
    # fold1 验证期(2022 全年)与 fold2 验证期(2022H2)重叠,
    # 重叠日保留 fold2 的预测(训练窗口更长,与 2023H2 所用模型一致)
    pred_df = pred_df.drop_duplicates(subset=["TradingDay", "StockID"], keep="last")
    pred_n = neutralize_by_industry(pred_df, quote)

    bench_ret, _ = simulate(pred_df.assign(pred=0.0), mkt,
                            n_top=len(mkt["stocks"]), cost_per_side=0.0)

    rows = {}
    for skip in SKIP_GRID:
        port_ret, turnovers = simulate(pred_n, mkt, buffer=BUFFER, skip=skip)
        m = evaluate(port_ret, bench_ret.reindex(port_ret.index), turnovers)
        rows[skip] = m
        print(f"skip={skip:4d}  超额={m['年化超额收益']:+.4f}  IR={m['信息比率']:+.3f}  "
              f"换手={m['年化单边换手率']:.1f}  胜率={m['月度胜率']:.2f}")

    df = pd.DataFrame(rows).T
    df.index.name = "skip"
    df.to_csv(os.path.join(RESULTS_DIR, "tune_skip_val.csv"), encoding="utf-8-sig")
    print("\nsaved ->", os.path.join(RESULTS_DIR, "tune_skip_val.csv"))


if __name__ == "__main__":
    main()
