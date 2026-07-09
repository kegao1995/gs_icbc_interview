# -*- coding: utf-8 -*-
"""特征组消融实验(加分项)

用线性代理模型(Ridge)快速对比不同特征集的样本外 RankIC。
线性模型训练秒级完成,适合做特征集层面的快速消融;
GRU 层面的结构对比见 train.py --model all 的输出。

运行: python notebooks/feature_ablation.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import load_feature_panel  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

GROUPS = {
    "收益/动量": ["ret1", "ret5", "ret10", "ret20", "ret60", "overnight", "intraday",
                 "ind_rel_ret5", "ind_rel_ret20"],
    "波动": ["vol5", "vol20", "vol60", "range_hl", "atr14"],
    "价格位置": ["pos_hl", "close_vwap", "ma5_dev", "ma20_dev", "ma60_dev",
               "ma5_ma20", "high60", "low60"],
    "量额/流动性": ["log_amt", "amt_ratio20", "turn_ratio5_20", "vol_cv20",
                 "corr_pv20", "amihud20", "susp_frac20"],
    "资金流": ["mfi14", "cmf20", "net_flow5", "net_flow20"],
    "技术指标": ["rsi14", "macd_hist", "boll_z20", "stoch14"],
    "分红/送转": ["div_yield_244", "split_int_244"],
}


def rank_ic_by_day(pred, y_raw, day_idx):
    ics = []
    for k, t in enumerate(day_idx):
        v = np.isfinite(y_raw[t]) & np.isfinite(pred[k])
        if v.sum() < 100:
            continue
        pr = pd.Series(pred[k][v]).rank()
        yr = pd.Series(y_raw[t][v]).rank()
        ics.append(np.corrcoef(pr, yr)[0, 1])
    return np.array(ics)


def run_subset(panel, cols_idx, tr_mask, te_idx):
    X, y, y_raw = panel["X"][:, :, cols_idx], panel["y"], panel["y_raw"]
    Xd, yd = X[tr_mask], y[tr_mask]
    v = np.isfinite(yd)
    m = Ridge(alpha=1.0).fit(Xd[v], yd[v])
    pred = np.stack([m.predict(X[t]) for t in te_idx])
    ics = rank_ic_by_day(pred, y_raw, te_idx)
    return ics.mean(), ics.mean() / ics.std()


def main():
    panel = load_feature_panel()
    days, names = panel["days"], panel["feat_names"]
    tr = (days >= "2019-04-01") & (days <= "2021-12-31")
    te = np.where(days >= "2023-01-01")[0]

    name_idx = {n: i for i, n in enumerate(names)}
    rows = {}
    all_idx = list(range(len(names)))
    rows["全部特征"] = run_subset(panel, all_idx, tr, te)
    for gname, cols in GROUPS.items():
        idx = [name_idx[c] for c in cols]
        rows[f"仅 {gname}"] = run_subset(panel, idx, tr, te)
        rest = [i for i in all_idx if i not in idx]
        rows[f"剔除 {gname}"] = run_subset(panel, rest, tr, te)

    df = pd.DataFrame(rows, index=["RankIC", "日度ICIR"]).T.round(4)
    print(df)
    df.to_csv(os.path.join(RESULTS_DIR, "feature_ablation.csv"), encoding="utf-8-sig")
    print("saved ->", os.path.join(RESULTS_DIR, "feature_ablation.csv"))


if __name__ == "__main__":
    main()
