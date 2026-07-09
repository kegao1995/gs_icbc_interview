# -*- coding: utf-8 -*-
"""特征工程与标签构造

实现方式:把长表 pivot 成 (交易日 x 股票) 的宽矩阵,所有滚动特征在宽矩阵上向量化计算。

因果性保证:
- 所有特征仅使用 T 日及之前的数据(rolling 窗口右端对齐当日);
- 截面预处理(去极值/标准化)逐日进行,仅用当日截面,不引入时序泄露;
- 标签使用 T+1 ~ T+6 的复权 VWAP,仅用于训练目标,不参与特征。

停牌处理:
- 停牌日 ret1 = 0(价格前推),量额为 0;
- 日内类特征(隔夜/日内收益、振幅、区间位置、VWAP 偏离)在停牌日为 NaN,
  截面标准化后统一填 0(即视为截面均值)。
"""
import os

import numpy as np
import pandas as pd

from data_loader import load_quote, DATA_PATH

SEQ_LEN = 40          # 输入序列长度(约两个月)
LABEL_HORIZON = 5     # 预测未来 5 个交易日
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cache")


# ---------------------------------------------------------------- 截面预处理

def cs_standardize_matrix(m: np.ndarray, n_mad: float = 5.0) -> np.ndarray:
    """逐行(逐日)MAD 去极值 + z-score,NaN 保留。m: [n_days, n_stocks]"""
    m = m.astype(np.float64, copy=True)
    med = np.nanmedian(m, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(m - med), axis=1, keepdims=True)
    scale = 1.4826 * mad
    scale[scale == 0] = np.inf  # 截面无离散度时不做截断
    m = np.clip(m, med - n_mad * scale, med + n_mad * scale)
    mu = np.nanmean(m, axis=1, keepdims=True)
    sd = np.nanstd(m, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (m - mu) / sd


# ---------------------------------------------------------------- 特征计算

def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="TradingDay", columns="StockID", values=col)


def build_feature_panel(df: pd.DataFrame = None):
    """构造特征面板。

    返回 dict:
      X:        np.float32 [n_days, n_stocks, n_features],截面标准化后,NaN 已填 0
      y:        np.float32 [n_days, n_stocks],截面标准化标签,无效处为 NaN
      y_raw:    np.float32 [n_days, n_stocks],原始区间收益(评估 IC 用),无效处为 NaN
      tradable: bool [n_days, n_stocks],当日是否正常交易
      days:     DatetimeIndex,stocks: ndarray,feat_names: list[str]
    """
    if df is None:
        df = load_quote()

    close = _pivot(df, "adj_close")
    open_ = _pivot(df, "adj_open")
    high = _pivot(df, "adj_high")
    low = _pivot(df, "adj_low")
    vwap = _pivot(df, "adj_vwap")
    prev_close = close.shift(1)
    volume = _pivot(df, "Volume")
    amount = _pivot(df, "Amount")
    ret1 = _pivot(df, "ret1")
    tradable = _pivot(df, "is_trade").astype(bool)

    # 停牌日 O/H/L 为 0,复权后仍为 0,统一置 NaN
    open_ = open_.where(tradable)
    high = high.where(tradable)
    low = low.where(tradable)
    vol_nan = volume.where(tradable)
    amt_nan = amount.where(tradable)

    feats = {}

    # --- 收益 / 动量 / 反转 ---
    feats["ret1"] = ret1
    for k in (5, 10, 20, 60):
        feats[f"ret{k}"] = close / close.shift(k) - 1
    feats["overnight"] = open_ / prev_close - 1
    feats["intraday"] = close.where(tradable) / open_ - 1

    # --- 波动 ---
    for k in (5, 20, 60):
        feats[f"vol{k}"] = ret1.rolling(k, min_periods=k).std()
    feats["range_hl"] = (high - low) / prev_close
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()]).groupby(level=0).max()
    feats["atr14"] = (tr / prev_close).rolling(14, min_periods=14).mean()

    # --- 价格位置 ---
    rng = (high - low).replace(0, np.nan)
    feats["pos_hl"] = ((close.where(tradable) - low) / rng).fillna(0.5).where(tradable)
    feats["close_vwap"] = close / vwap - 1
    for k in (5, 20, 60):
        feats[f"ma{k}_dev"] = close / close.rolling(k, min_periods=k).mean() - 1
    feats["ma5_ma20"] = (close.rolling(5, min_periods=5).mean()
                         / close.rolling(20, min_periods=20).mean() - 1)
    feats["high60"] = close / close.rolling(60, min_periods=60).max() - 1
    feats["low60"] = close / close.rolling(60, min_periods=60).min() - 1

    # --- 量额 / 流动性 ---
    feats["log_amt"] = np.log1p(amt_nan)
    amt_ma20 = amt_nan.rolling(20, min_periods=10).mean()
    feats["amt_ratio20"] = amt_nan / amt_ma20
    vol_ma5 = vol_nan.rolling(5, min_periods=3).mean()
    vol_ma20 = vol_nan.rolling(20, min_periods=10).mean()
    feats["turn_ratio5_20"] = vol_ma5 / vol_ma20
    feats["vol_cv20"] = (vol_nan.rolling(20, min_periods=10).std()
                         / vol_ma20)
    dvol = vol_nan.pct_change()
    feats["corr_pv20"] = ret1.rolling(20, min_periods=15).corr(dvol)
    feats["amihud20"] = np.log1p((ret1.abs() / amt_nan.replace(0, np.nan))
                                 .rolling(20, min_periods=10).mean() * 1e6)
    feats["susp_frac20"] = (~tradable).rolling(20, min_periods=20).mean()

    # --- 行业相对强弱 ---
    ind = df.groupby("StockID")["IndustryName"].first().reindex(close.columns)
    for k in (5, 20):
        r = feats[f"ret{k}"]
        ind_med = r.T.groupby(ind.values).transform("median").T
        feats[f"ind_rel_ret{k}"] = r - ind_med

    feat_names = list(feats.keys())
    days = close.index
    stocks = close.columns.to_numpy()

    # --- 逐日截面标准化,NaN 填 0 ---
    X = np.empty((len(days), len(stocks), len(feat_names)), dtype=np.float32)
    for j, name in enumerate(feat_names):
        m = cs_standardize_matrix(feats[name].to_numpy())
        X[:, :, j] = np.nan_to_num(m, nan=0.0).astype(np.float32)

    # --- 标签:adjVWAP_{T+6} / adjVWAP_{T+1} - 1 ---
    # T+1 或 T+6 停牌时 adj_vwap 为 NaN,标签自动无效(该样本不可交易,剔除)
    y_raw = (vwap.shift(-(LABEL_HORIZON + 1)) / vwap.shift(-1) - 1).to_numpy()
    y = cs_standardize_matrix(y_raw).astype(np.float32)

    return {
        "X": X,
        "y": y,
        "y_raw": y_raw.astype(np.float32),
        "tradable": tradable.to_numpy(),
        "days": days,
        "stocks": stocks,
        "feat_names": feat_names,
    }


def load_feature_panel(cache: bool = True):
    """带缓存的特征面板加载"""
    cache_file = os.path.join(CACHE_DIR, "feature_panel.npz")
    if cache and os.path.exists(cache_file):
        z = np.load(cache_file, allow_pickle=True)
        return {
            "X": z["X"], "y": z["y"], "y_raw": z["y_raw"], "tradable": z["tradable"],
            "days": pd.DatetimeIndex(z["days"]), "stocks": z["stocks"],
            "feat_names": list(z["feat_names"]),
        }
    panel = build_feature_panel()
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_file, X=panel["X"], y=panel["y"], y_raw=panel["y_raw"],
                            tradable=panel["tradable"], days=panel["days"].to_numpy(),
                            stocks=panel["stocks"], feat_names=np.array(panel["feat_names"]))
    return panel


if __name__ == "__main__":
    panel = load_feature_panel(cache=False)
    X, y = panel["X"], panel["y"]
    print("X:", X.shape, "y:", y.shape, "features:", len(panel["feat_names"]))
    print(panel["feat_names"])
    print("label valid ratio:", np.isfinite(y).mean().round(4))
    print("X abs mean (should be O(1)):", np.abs(X).mean().round(4))
