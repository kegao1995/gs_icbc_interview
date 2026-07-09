# -*- coding: utf-8 -*-
"""探索性数据分析:数据质量体检

运行: python notebooks/eda.py
结论摘要见 report.md 第 2 节。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_quote, get_week_last_days  # noqa: E402


def main():
    df = load_quote()
    print("=" * 60)
    print("基本信息")
    print("=" * 60)
    print("行数:", len(df), "| 股票数:", df.StockID.nunique(), "| 交易日数:", df.TradingDay.nunique())
    print("日期范围:", df.TradingDay.min().date(), "->", df.TradingDay.max().date())
    per_stock = df.groupby("StockID").size()
    print("每只股票行数 min/max:", per_stock.min(), per_stock.max(), "(平衡面板,无中途上市/退市)")
    print("缺失值总数:", df[["PrevClosePrice", "OpenPrice", "HighPrice", "LowPrice",
                          "ClosePrice", "Volume", "Amount"]].isna().sum().sum())
    print("行业数:", df.IndustryName.nunique())
    print(df.IndustryName.value_counts().head())

    print("\n" + "=" * 60)
    print("停牌语义")
    print("=" * 60)
    susp = df[~df.is_trade]
    print("停牌行数:", len(susp), "| 涉及股票:", susp.StockID.nunique())
    print("停牌日 Close==PrevClose 比例:", (susp.ClosePrice == susp.PrevClosePrice).mean())
    print("停牌日 Volume==0 且 Amount==0 比例:",
          ((susp.Volume == 0) & (susp.Amount == 0)).mean())

    print("\n" + "=" * 60)
    print("复权因子")
    print("=" * 60)
    prev_close = df.groupby("StockID")["ClosePrice"].shift(1)
    single = prev_close / df.PrevClosePrice
    n_event = (np.abs(single - 1) > 1e-6).sum()
    print("除权除息事件行数:", n_event)
    print("单期因子范围:", round(single.min(), 4), "~", round(single.max(), 4))
    print("累计复权因子范围:", round(df.adj.min(), 4), "~", round(df.adj.max(), 4))
    # 抽查:复权收益应在除权日保持连续(不出现 -30% 的假跳空)
    df["adj_ret"] = df.groupby("StockID")["adj_close"].pct_change()
    event_rows = df[np.abs(single - 1) > 0.05]
    print("除权日复权收益分位数(应与正常日同量级):")
    print(event_rows.adj_ret.describe().round(4))

    print("\n" + "=" * 60)
    print("收益分布与涨跌停")
    print("=" * 60)
    trade = df[df.is_trade]
    q = trade.ret1.quantile([0, 0.001, 0.5, 0.999, 1.0])
    print("日收益分位数:", q.round(4).to_dict())
    print("|ret|≈10% 的行数(主板涨跌停):",
          ((trade.ret1.abs() > 0.099) & (trade.ret1.abs() < 0.101)).sum())
    print("|ret|≈20% 的行数(创业板/科创板):",
          ((trade.ret1.abs() > 0.198) & (trade.ret1.abs() < 0.202)).sum())
    print("=> 存在极端值(max=%.2f),特征与标签均需截面去极值" % trade.ret1.max())

    print("\n" + "=" * 60)
    print("VWAP 合理性")
    print("=" * 60)
    vwap = trade.Amount / trade.Volume
    print("VWAP 落在 [Low, High](1%% 容差)比例: %.4f"
          % ((vwap >= trade.LowPrice * 0.99) & (vwap <= trade.HighPrice * 1.01)).mean())

    print("\n" + "=" * 60)
    print("周度调仓日")
    print("=" * 60)
    wl = get_week_last_days(df)
    print("周最后交易日数量:", len(wl), "| 2023 年内:", (wl >= "2023-01-01").sum())


if __name__ == "__main__":
    main()
