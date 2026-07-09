# -*- coding: utf-8 -*-
"""数据加载与复权处理

数据语义(EDA 确认):
- 停牌日:Open/High/Low = 0,Volume = Amount = 0,ClosePrice = PrevClosePrice(价格前推)
- 除权除息日:PrevClosePrice != 前一日 ClosePrice,按数据字典公式计算复权因子
- VWAP = Amount / Volume(万元/万股 = 元/股),100% 落在当日 [Low, High] 区间内
"""
import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Quote.parquet")


def load_quote(path: str = DATA_PATH) -> pd.DataFrame:
    """读取行情数据,计算复权因子、复权价、可交易标志。

    返回按 (StockID, TradingDay) 排序的长表,新增列:
      is_trade   当日是否正常交易(非停牌)
      adj        当日复权因子(历史单期因子累乘)
      adj_close / adj_open / adj_high / adj_low / adj_vwap  复权价(停牌日 vwap 为 NaN)
      ret1       当日涨跌幅 Close/PrevClose - 1(停牌日为 0)
    """
    df = pd.read_parquet(path)
    df = df.sort_values(["StockID", "TradingDay"]).reset_index(drop=True)

    df["is_trade"] = df["OpenPrice"] > 0

    # 复权因子:单期因子 = 前一日 Close / 当日 PrevClose,历史累乘
    prev_close = df.groupby("StockID")["ClosePrice"].shift(1)
    single = (prev_close / df["PrevClosePrice"]).fillna(1.0)
    df["adj"] = single.groupby(df["StockID"]).cumprod()

    for c, src in [("adj_close", "ClosePrice"), ("adj_open", "OpenPrice"),
                   ("adj_high", "HighPrice"), ("adj_low", "LowPrice")]:
        df[c] = df[src] * df["adj"]

    # VWAP:停牌日 Volume=0,置为 NaN
    vwap = np.where(df["Volume"] > 0, df["Amount"] / df["Volume"], np.nan)
    df["adj_vwap"] = vwap * df["adj"]

    df["ret1"] = df["ClosePrice"] / df["PrevClosePrice"] - 1.0
    return df


def get_trading_days(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df["TradingDay"].unique()))


def get_week_last_days(df: pd.DataFrame) -> pd.DatetimeIndex:
    """每周最后一个交易日(信号产生日)"""
    days = pd.Series(get_trading_days(df))
    iso = days.dt.isocalendar()
    key = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    return pd.DatetimeIndex(days.groupby(key.values).max().sort_values().values)


if __name__ == "__main__":
    df = load_quote()
    print(df.shape)
    print(df[["StockID", "TradingDay", "adj", "adj_close", "adj_vwap", "ret1", "is_trade"]].tail())
    wl = get_week_last_days(df)
    print("week-last days:", len(wl), wl[:5].tolist())
