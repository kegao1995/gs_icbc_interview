# -*- coding: utf-8 -*-
"""结果可视化(加分项):IC 时序、分组收益单调性、模型对比

用法: python src/plots.py   (需先运行 train.py 生成预测文件)
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import load_feature_panel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_ic_timeseries(models=("gru", "lstm", "mlp")):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for m in models:
        f = os.path.join(RESULTS_DIR, f"daily_rank_ic_{m}.csv")
        if not os.path.exists(f):
            continue
        ic = pd.read_csv(f, index_col=0, parse_dates=True)["rank_ic"]
        axes[0].plot(ic.index, ic.rolling(20).mean(), lw=1.3,
                     label=f"{m.upper()}(20日均值)")
        axes[1].plot(ic.index, ic.cumsum(), lw=1.3, label=m.upper())
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_title("测试集日度 RankIC(20 日移动平均)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("累计 RankIC")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ic_timeseries.png"), dpi=150)
    plt.close(fig)


def plot_group_returns(model="gru", n_groups=10):
    """按预测值分十组的未来 5 日平均收益(单调性检验)"""
    panel = load_feature_panel()
    days, stocks, y_raw = panel["days"], panel["stocks"], panel["y_raw"]
    pred = pd.read_csv(os.path.join(RESULTS_DIR, f"predictions_{model}.csv"),
                       parse_dates=["TradingDay"])
    pw = pred.pivot(index="TradingDay", columns="StockID", values="pred")

    sums = np.zeros(n_groups)
    cnt = 0
    for d in pw.index:
        t = days.get_loc(d)
        p = pw.loc[d].reindex(stocks).to_numpy()
        v = np.isfinite(y_raw[t]) & np.isfinite(p)
        if v.sum() < 300:
            continue
        q = pd.qcut(pd.Series(p[v]).rank(method="first"), n_groups, labels=False)
        g = pd.Series(y_raw[t][v]).groupby(q).mean()
        sums += g.to_numpy()
        cnt += 1
    avg = sums / cnt * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, n_groups + 1), avg, color=["#c44" if x < 0 else "#2a7" for x in avg])
    ax.set_xlabel("预测值分组(1=最低, 10=最高)")
    ax.set_ylabel("未来 5 日平均收益(%)")
    ax.set_title(f"{model.upper()} 预测值分组收益单调性(2023 测试集)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"group_returns_{model}.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance():
    """特征组重要性:消融实验中各组单独的 RankIC(需先跑 feature_ablation.py)"""
    f = os.path.join(RESULTS_DIR, "feature_ablation.csv")
    if not os.path.exists(f):
        print("skip feature_importance: run notebooks/feature_ablation.py first")
        return
    df = pd.read_csv(f, index_col=0, encoding="utf-8-sig")
    solo = df[df.index.str.startswith("仅 ")].copy()
    solo.index = solo.index.str.replace("仅 ", "", regex=False)
    solo = solo.sort_values("RankIC")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#c44" if v < 0 else "#2a7" for v in solo["RankIC"]]
    ax.barh(solo.index, solo["RankIC"], color=colors)
    ax.axvline(df.loc["全部特征", "RankIC"], color="gray", ls="--", lw=1,
               label=f"全部特征({df.loc['全部特征', 'RankIC']:.3f})")
    ax.set_xlabel("单独 RankIC(Ridge,2023 测试集)")
    ax.set_title("特征组重要性(分组消融)")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_ic_timeseries()
    plot_group_returns("gru")
    plot_feature_importance()
    print("saved plots to", RESULTS_DIR)
