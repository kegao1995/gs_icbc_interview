# -*- coding: utf-8 -*-
"""通用工具函数:随机种子、截面预处理、IC 计算等"""
import random

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 244
WEEKS_PER_YEAR = 52


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def mad_winsorize(x: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """MAD 去极值(单个截面)"""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return x
    lo, hi = med - n_mad * 1.4826 * mad, med + n_mad * 1.4826 * mad
    return x.clip(lo, hi)


def zscore(x: pd.Series) -> pd.Series:
    """截面标准化(单个截面)"""
    std = x.std()
    if std == 0 or np.isnan(std):
        return x * 0.0
    return (x - x.mean()) / std


def cs_standardize(df: pd.DataFrame, cols, by: str = "TradingDay", n_mad: float = 5.0) -> pd.DataFrame:
    """逐日截面去极值 + 标准化。仅使用当日截面信息,不存在时序泄露。"""
    out = df.copy()
    g = out.groupby(by, sort=False)
    for c in cols:
        out[c] = g[c].transform(lambda s: zscore(mad_winsorize(s, n_mad)))
    return out


def daily_rank_ic(df: pd.DataFrame, pred_col: str = "pred", label_col: str = "label_raw",
                  by: str = "TradingDay") -> pd.Series:
    """逐日截面 Spearman RankIC 序列"""
    def _ic(g):
        if g[pred_col].notna().sum() < 30:
            return np.nan
        return g[pred_col].corr(g[label_col], method="spearman")
    return df.groupby(by).apply(_ic, include_groups=False)


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    return {
        "IC_mean": ic.mean(),
        "IC_std": ic.std(),
        "ICIR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "IC_positive_ratio": (ic > 0).mean(),
        "n_days": len(ic),
    }


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤(输入净值序列,返回正数)"""
    peak = nav.cummax()
    return ((peak - nav) / peak).max()
