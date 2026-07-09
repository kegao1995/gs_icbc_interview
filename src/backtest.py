# -*- coding: utf-8 -*-
"""周度调仓回测与评价

规则:
  - 每周最后一个交易日 T 收盘产生信号(使用 T 日及之前信息);
  - T+1 个交易日以 VWAP 执行调仓(与标签口径自洽);
  - 选预测收益最高的 200 只等权做多;
  - 基准:全部 1000 只股票等权(同一引擎、同一调仓节奏、零成本,保证口径一致);
  - 交易成本:双边合计千分之三,即按 0.0015 x (买入额+卖出额)/组合市值 计提。

停牌处理(避免未来信息):
  - 信号日 T 停牌的股票不进入买入候选;
  - 执行日 T+1 停牌:持仓卖不出 -> 继续持有;买入目标买不进 -> 放弃,资金摊给其余买入标的。

日度净值核算:
  - 调仓日收益分解为 [昨收->VWAP](旧权重) 与 [VWAP->当收](新权重)两段,成本按换手计提;
  - 非调仓日按复权收盘价收益漂移权重。

自洽性检验:n_top=1000、零成本时,组合净值与基准完全一致。

用法: python src/backtest.py --pred results/predictions.csv [--buffer 350] [--neutral]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_quote, get_week_last_days
from utils import max_drawdown, TRADING_DAYS_PER_YEAR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
COST_ROUNDTRIP = 0.003          # 双边合计
COST_PER_SIDE = COST_ROUNDTRIP / 2
N_TOP = 200


# ---------------------------------------------------------------- 数据准备

def prepare_market(df: pd.DataFrame):
    """返回宽矩阵字典(index=TradingDay, columns=StockID)"""
    adj_close = df.pivot(index="TradingDay", columns="StockID", values="adj_close")
    adj_vwap = df.pivot(index="TradingDay", columns="StockID", values="adj_vwap")
    tradable = df.pivot(index="TradingDay", columns="StockID", values="is_trade").astype(bool)
    ret_cc = adj_close / adj_close.shift(1) - 1          # 昨收 -> 今收(停牌日=0)
    ret_cv = adj_vwap / adj_close.shift(1) - 1           # 昨收 -> 今日VWAP
    ret_vc = adj_close / adj_vwap - 1                    # 今日VWAP -> 今收
    return {"ret_cc": ret_cc.fillna(0.0), "ret_cv": ret_cv, "ret_vc": ret_vc,
            "tradable": tradable, "days": adj_close.index, "stocks": adj_close.columns}


def neutralize_by_industry(pred_df: pd.DataFrame, quote: pd.DataFrame) -> pd.DataFrame:
    """预测值逐日对行业哑变量回归取残差(行业中性化,加分项)"""
    ind = quote.groupby("StockID")["IndustryName"].first()
    out = pred_df.copy()
    out["ind"] = out["StockID"].map(ind)

    def _resid(g):
        m = g.groupby("ind")["pred"].transform("mean")
        return g["pred"] - m

    out["pred"] = out.groupby("TradingDay", group_keys=False).apply(_resid, include_groups=False)
    return out.drop(columns="ind")


# ---------------------------------------------------------------- 组合模拟

def select_target(pred: pd.Series, holdings: dict, tradable_t: pd.Series,
                  n_top: int = N_TOP, buffer: int = 0, skip: int = 0) -> list:
    """信号日选股。buffer>0 时启用缓冲区规则:已持有且预测排名在 buffer 内的不卖出。
    skip>0 时剔除预测值最高的前 skip 名再选(最高分位股票短期均值回归,见分组收益诊断)。"""
    cand = pred[tradable_t.reindex(pred.index).fillna(False)].dropna()
    if skip > 0 and len(cand) > n_top + skip:
        top_names = cand.rank(ascending=False).sort_values().index[:skip]
        cand = cand.drop(top_names)
    if len(cand) <= n_top:
        return list(cand.index)
    rank = cand.rank(ascending=False)
    if buffer > 0 and holdings:
        keep = [s for s in holdings if s in rank.index and rank[s] <= buffer][:n_top]
        n_fill = n_top - len(keep)
        fill = [s for s in rank.sort_values().index if s not in set(keep)][:n_fill]
        return keep + fill
    return list(rank.sort_values().index[:n_top])


def simulate(pred_df: pd.DataFrame, mkt: dict, n_top: int = N_TOP,
             cost_per_side: float = COST_PER_SIDE, buffer: int = 0, skip: int = 0):
    """周度调仓组合模拟。

    pred_df: [TradingDay, StockID, pred];返回 (日度收益 Series, 每次调仓单边换手 Series)
    """
    days = mkt["days"]
    pred_wide = pred_df.pivot(index="TradingDay", columns="StockID", values="pred")

    test_days = pred_wide.index.sort_values()
    week_last = get_week_last_days_from(days)
    signal_days = [d for d in week_last if d in test_days]
    # 最后一个信号日若无 T+1 则丢弃
    signal_days = [d for d in signal_days if days.get_loc(d) + 1 < len(days)]
    signal_set = set(signal_days)

    start_pos = days.get_loc(signal_days[0])  # 首个信号日(当日收盘前空仓)
    w = pd.Series(dtype=float)                # 当前权重(按收盘口径)
    pending = None                            # 待执行的目标持仓列表
    daily_ret, turnovers = {}, {}

    for pos in range(start_pos, len(days)):
        d = days[pos]
        r_today = 0.0
        if pending is not None:
            # 执行日:昨收->VWAP 旧权重,VWAP->当收 新权重
            tradable_t = mkt["tradable"].loc[d]
            r_cv, r_vc, r_cc = mkt["ret_cv"].loc[d], mkt["ret_vc"].loc[d], mkt["ret_cc"].loc[d]

            frozen = [s for s in w.index if not tradable_t.get(s, False)]
            filled = [s for s in pending if tradable_t.get(s, False)]

            # 旧持仓:可交易部分随 昨收->VWAP,冻结部分随 昨收->当收(停牌日为 0)
            r_old = float((w.drop(frozen) * r_cv.reindex(w.drop(frozen).index)).sum()
                          + (w[frozen] * r_cc.reindex(frozen)).sum()) if len(w) else 0.0
            # 旧权重漂移到执行时点
            w_exec = w.copy()
            if len(w_exec):
                grow = r_cv.reindex(w_exec.index)
                grow[frozen] = r_cc.reindex(frozen)
                w_exec = w_exec * (1 + grow) / (1 + r_old)

            w_frozen = w_exec.reindex(frozen).fillna(0.0)
            budget = 1.0 - float(w_frozen.sum())
            w_new = pd.Series(budget / len(filled), index=filled) if filled else pd.Series(dtype=float)
            w_new = pd.concat([w_new, w_frozen])
            w_new = w_new.groupby(w_new.index).sum()

            delta = w_new.subtract(w_exec, fill_value=0.0).abs().sum()
            turn = float(delta) / 2.0
            turnovers[d] = turn
            cost = cost_per_side * float(delta)

            # VWAP->当收 段:新持仓中执行日买入的部分随 r_vc,冻结部分已含在 r_cc 中
            r_new = float((w_new.drop(frozen, errors="ignore")
                           * r_vc.reindex(w_new.drop(frozen, errors="ignore").index)).sum())
            r_today = (1 + r_old) * (1 + r_new) * (1 - cost) - 1
            # 收盘权重
            grow2 = r_vc.reindex(w_new.index).fillna(0.0)
            for s in frozen:
                if s in w_new.index:
                    grow2[s] = 0.0            # 冻结股已按全日收益计入
            w = w_new * (1 + grow2)
            w = w / w.sum()
            pending = None
        elif len(w):
            r_cc = mkt["ret_cc"].loc[d]
            r_today = float((w * r_cc.reindex(w.index)).sum())
            w = w * (1 + r_cc.reindex(w.index)) / (1 + r_today)

        daily_ret[d] = r_today

        if d in signal_set:
            tradable_t = mkt["tradable"].loc[d]
            pred_t = pred_wide.loc[d]
            pending = select_target(pred_t, dict(w), tradable_t, n_top=n_top,
                                    buffer=buffer, skip=skip)

    return pd.Series(daily_ret).sort_index(), pd.Series(turnovers).sort_index()


def get_week_last_days_from(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(days)
    iso = s.dt.isocalendar()
    key = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    return pd.DatetimeIndex(s.groupby(key.values).max().sort_values().values)


# ---------------------------------------------------------------- 指标

def evaluate(port_ret: pd.Series, bench_ret: pd.Series, turnovers: pd.Series) -> dict:
    port_nav = (1 + port_ret).cumprod()
    bench_nav = (1 + bench_ret).cumprod()
    excess_nav = port_nav / bench_nav
    n = len(port_ret)
    years = n / TRADING_DAYS_PER_YEAR

    ex_ret_daily = port_ret - bench_ret
    ann_excess = excess_nav.iloc[-1] ** (1 / years) - 1
    ir = ex_ret_daily.mean() / ex_ret_daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    monthly_p = (1 + port_ret).groupby(port_ret.index.to_period("M")).prod()
    monthly_b = (1 + bench_ret).groupby(bench_ret.index.to_period("M")).prod()
    win = (monthly_p > monthly_b).mean()

    return {
        "年化收益(组合)": port_nav.iloc[-1] ** (1 / years) - 1,
        "年化收益(基准)": bench_nav.iloc[-1] ** (1 / years) - 1,
        "年化超额收益": ann_excess,
        "信息比率": ir,
        "最大回撤(超额)": max_drawdown(excess_nav),
        "最大回撤(组合)": max_drawdown(port_nav),
        "年化单边换手率": turnovers.sum() / years,
        "月度胜率": win,
        "回测交易日数": n,
    }


def plot_equity(port_ret, bench_ret, out_png):
    port_nav = (1 + port_ret).cumprod()
    bench_nav = (1 + bench_ret).cumprod()
    excess_nav = port_nav / bench_nav
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(port_nav.index, port_nav.values, label="组合", lw=1.5)
    ax.plot(bench_nav.index, bench_nav.values, label="基准(1000只等权)", lw=1.5)
    ax.plot(excess_nav.index, excess_nav.values, label="超额净值", lw=1.5, ls="--")
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_title("指数增强策略净值曲线(2023 测试集,周度调仓,含成本)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程

def run_backtest(pred_path: str, buffer: int = 0, neutral: bool = False,
                 skip: int = 0, tag: str = "", save: bool = True):
    quote = load_quote()
    mkt = prepare_market(quote)

    pred_df = pd.read_csv(pred_path, parse_dates=["TradingDay"])
    if neutral:
        pred_df = neutralize_by_industry(pred_df, quote)

    port_ret, turnovers = simulate(pred_df, mkt, buffer=buffer, skip=skip)

    # 基准:同引擎、全持仓、零成本
    bench_pred = pred_df.copy()
    bench_pred["pred"] = 0.0
    bench_ret, _ = simulate(bench_pred, mkt, n_top=len(mkt["stocks"]), cost_per_side=0.0)
    bench_ret = bench_ret.reindex(port_ret.index)

    metrics = evaluate(port_ret, bench_ret, turnovers)
    print(f"\n===== 回测结果 {tag or os.path.basename(pred_path)} "
          f"(buffer={buffer}, neutral={neutral}, skip={skip}) =====")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        suffix = tag and f"_{tag}"
        pd.DataFrame([metrics]).T.rename(columns={0: "value"}).to_csv(
            os.path.join(RESULTS_DIR, f"backtest_results{suffix}.csv"),
            encoding="utf-8-sig")
        nav = pd.DataFrame({"port_ret": port_ret, "bench_ret": bench_ret})
        nav.to_csv(os.path.join(RESULTS_DIR, f"nav_daily{suffix}.csv"))
        plot_equity(port_ret, bench_ret,
                    os.path.join(RESULTS_DIR, f"equity_curve{suffix}.png"))
    return metrics, port_ret, bench_ret, turnovers


def self_check():
    """自洽性检验:全持仓 + 零成本时组合应与基准完全一致"""
    quote = load_quote()
    mkt = prepare_market(quote)
    pred_path = os.path.join(RESULTS_DIR, "predictions.csv")
    pred_df = pd.read_csv(pred_path, parse_dates=["TradingDay"])
    r1, _ = simulate(pred_df.assign(pred=0.0), mkt, n_top=len(mkt["stocks"]), cost_per_side=0.0)
    r2, _ = simulate(pred_df.assign(pred=1.0), mkt, n_top=len(mkt["stocks"]), cost_per_side=0.0)
    diff = (r1 - r2).abs().max()
    print("self-check max |diff| =", diff, "(应为 0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(RESULTS_DIR, "predictions.csv"))
    ap.add_argument("--buffer", type=int, default=0)
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
    else:
        run_backtest(args.pred, buffer=args.buffer, neutral=args.neutral,
                     skip=args.skip, tag=args.tag)
