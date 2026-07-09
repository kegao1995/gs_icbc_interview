# -*- coding: utf-8 -*-
"""滚动训练脚本

数据切分(严格时序,不随机打乱;滚动步长参数化,--roll-months,默认 6 个月):
  fold 1: 训练 2019-04-01 ~ 2021-12-31 | 验证 2022 全年       | 预测 2023-01 ~ 2023-06
  fold 2: 训练 2019-04-01 ~ 2022-06-30 | 验证 2022-07 ~ 2022-12 | 预测 2023-07 ~ 2023-12
(--roll-months 3 为季度滚动四折,12 退化为题目基础切分;验证集始终限定在 2022 年内)
训练集尾部设 6 个交易日 embargo(标签窗口跨入验证期的样本剔除),验证集尾部同理。

训练细节:
  - 损失: MSE(标签为逐日截面标准化后的未来 5 日收益)
  - 优化器: Adam(lr=1e-3, weight_decay=1e-5),batch=4096,max_epochs=30
  - 早停: 监控验证集日度 RankIC 均值,patience=5,回滚最优权重
  - 训练样本仅在训练时段内打乱批次顺序(时序切分本身不打乱)

用法: python src/train.py --model gru   (可选 gru / lstm / mlp / all)
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from features import load_feature_panel, SEQ_LEN, LABEL_HORIZON
from model import build_model
from utils import set_seed, ic_summary

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
EMBARGO = LABEL_HORIZON + 1  # 标签用到 T+6,故留 6 个交易日隔离带

# 基础切分(题目给定):训练 2019-2021 | 验证 2022 | 测试 2023
TRAIN_START = "2019-04-01"          # 前 60 个交易日为特征窗口预热,不作样本
BASE_TRAIN_END = pd.Timestamp("2021-12-31")
VAL_START0 = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2022-12-31")
TEST_START0 = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2023-12-29")


def make_folds(roll_months: int = 6) -> list:
    """按滚动步长(月)生成训练折:每 roll_months 个月扩张一次训练窗口、
    验证起点相应后移、预测下一段测试区间。

    验证集始终限定在 2022 年内(2023 仅作最终测试,不参与任何调参/早停);
    roll_months=6 为默认两折,12 退化为题目给定的基础切分,3 为季度滚动。
    """
    folds = []
    for i in range(int(np.ceil(12 / roll_months))):
        off = pd.DateOffset(months=i * roll_months)
        val_start = VAL_START0 + off
        test_start = TEST_START0 + off
        test_end = min(TEST_START0 + pd.DateOffset(months=(i + 1) * roll_months)
                       - pd.Timedelta(days=1), TEST_END)
        if val_start > VAL_END or test_start > TEST_END:
            break
        fmt = "%Y-%m-%d"
        folds.append({
            "train": (TRAIN_START, (BASE_TRAIN_END + off).strftime(fmt)),
            "val": (val_start.strftime(fmt), VAL_END.strftime(fmt)),
            "test": (test_start.strftime(fmt), test_end.strftime(fmt)),
        })
    return folds


FOLDS = make_folds(6)


# ---------------------------------------------------------------- 采样与批量

def day_range_idx(days: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return np.where((days >= start) & (days <= end))[0]


def sample_index(panel, day_idx: np.ndarray, embargo: int = 0) -> np.ndarray:
    """生成 (t, i) 训练样本索引:t 有完整序列窗口、信号日正常交易、标签有效"""
    if embargo > 0:
        day_idx = day_idx[:-embargo] if len(day_idx) > embargo else day_idx[:0]
    y, tradable = panel["y"], panel["tradable"]
    ts, is_ = [], []
    for t in day_idx:
        if t < SEQ_LEN - 1:
            continue
        ok = np.isfinite(y[t]) & tradable[t]
        ii = np.where(ok)[0]
        ts.append(np.full(len(ii), t)), is_.append(ii)
    if not ts:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([np.concatenate(ts), np.concatenate(is_)], axis=1)


def gather_batch(X: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    """向量化抽取序列窗口。idx: [B,2] 的 (t,i) -> [B, SEQ_LEN, F]"""
    t, i = idx[:, 0], idx[:, 1]
    offs = np.arange(SEQ_LEN - 1, -1, -1)
    xb = X[t[:, None] - offs[None, :], i[:, None], :]
    return torch.from_numpy(xb)


@torch.no_grad()
def predict_days(model, panel, day_idx: np.ndarray, device="cpu") -> np.ndarray:
    """逐日预测全部股票,返回 [n_days_selected, n_stocks]"""
    model.eval()
    X = panel["X"]
    n_stocks = X.shape[1]
    out = np.full((len(day_idx), n_stocks), np.nan, dtype=np.float32)
    all_i = np.arange(n_stocks)
    for k, t in enumerate(day_idx):
        if t < SEQ_LEN - 1:
            continue
        idx = np.stack([np.full(n_stocks, t), all_i], axis=1)
        xb = gather_batch(X, idx).to(device)
        out[k] = model(xb).cpu().numpy()
    return out


def val_rank_ic(pred: np.ndarray, y_raw: np.ndarray, day_idx: np.ndarray) -> float:
    """验证集日度 RankIC 均值(与原始标签算,避免依赖标签标准化)"""
    ics = []
    for k, t in enumerate(day_idx):
        v = np.isfinite(y_raw[t]) & np.isfinite(pred[k])
        if v.sum() < 100:
            continue
        pr = pd.Series(pred[k][v]).rank()
        yr = pd.Series(y_raw[t][v]).rank()
        ics.append(np.corrcoef(pr, yr)[0, 1])
    return float(np.mean(ics))


# ---------------------------------------------------------------- 训练

def train_fold(model_name, panel, fold, seed=42, max_epochs=30, patience=5,
               batch_size=4096, lr=1e-3, weight_decay=1e-5, device="cpu", log=print):
    set_seed(seed)
    days, X, y, y_raw = panel["days"], panel["X"], panel["y"], panel["y_raw"]

    tr_idx = sample_index(panel, day_range_idx(days, *fold["train"]), embargo=EMBARGO)
    val_days = day_range_idx(days, *fold["val"])
    val_days_eval = val_days[:-EMBARGO]  # 验证集尾部标签不完整,剔除
    test_days = day_range_idx(days, *fold["test"])
    log(f"  train samples={len(tr_idx)}, val days={len(val_days_eval)}, test days={len(test_days)}")

    model = build_model(model_name, X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_ic, best_state, bad = -np.inf, None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(tr_idx))
        t0, ep_loss, nb = time.time(), 0.0, 0
        for s in range(0, len(order), batch_size):
            b = tr_idx[order[s:s + batch_size]]
            xb = gather_batch(X, b).to(device)
            yb = torch.from_numpy(y[b[:, 0], b[:, 1]]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        pred_val = predict_days(model, panel, val_days_eval, device)
        ic = val_rank_ic(pred_val, y_raw, val_days_eval)
        log(f"  epoch {epoch:2d}  loss={ep_loss / nb:.4f}  val RankIC={ic:.4f}  ({time.time() - t0:.0f}s)")
        if ic > best_ic:
            best_ic, best_state, bad = ic, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop at epoch {epoch} (best val RankIC={best_ic:.4f})")
                break
    model.load_state_dict(best_state)

    pred_test = predict_days(model, panel, test_days, device)
    # 最优权重下的验证期预测(用于组合构建规则在验证集上调参)
    pred_val_best = predict_days(model, panel, val_days_eval, device)
    return model, test_days, pred_test, best_ic, val_days_eval, pred_val_best


def run(model_name: str, seed: int = 42, device: str = "cpu", folds: list = None):
    panel = load_feature_panel()
    days, stocks, y_raw = panel["days"], panel["stocks"], panel["y_raw"]
    if folds is None:
        folds = FOLDS

    rows, rows_val, ics_all = [], [], []
    print(f"===== model: {model_name} (device={device}, folds={len(folds)}) =====")
    for f_i, fold in enumerate(folds, 1):
        print(f"fold {f_i}: train {fold['train']}  val {fold['val']}  test {fold['test']}")
        _, test_days, pred, best_val_ic, val_days, pred_val = train_fold(
            model_name, panel, fold, seed=seed, device=device)
        for k, t in enumerate(test_days):
            rows.append(pd.DataFrame({
                "TradingDay": days[t], "StockID": stocks, "pred": pred[k]}))
        for k, t in enumerate(val_days):
            rows_val.append(pd.DataFrame({
                "TradingDay": days[t], "StockID": stocks, "pred": pred_val[k]}))
        # 测试期日度 IC(标签完整的日子)
        for k, t in enumerate(test_days):
            v = np.isfinite(y_raw[t]) & np.isfinite(pred[k])
            if v.sum() < 100:
                continue
            pr = pd.Series(pred[k][v]).rank()
            yr = pd.Series(y_raw[t][v]).rank()
            ics_all.append((days[t], np.corrcoef(pr, yr)[0, 1]))

    pred_df = pd.concat(rows, ignore_index=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f"predictions_{model_name}.csv")
    pred_df.to_csv(out_csv, index=False)
    pd.concat(rows_val, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, f"predictions_val_{model_name}.csv"), index=False)

    ic_s = pd.Series(dict(ics_all)).sort_index()
    ic_s.to_csv(os.path.join(RESULTS_DIR, f"daily_rank_ic_{model_name}.csv"),
                header=["rank_ic"])
    summ = ic_summary(ic_s)
    print(f"[{model_name}] test RankIC mean={summ['IC_mean']:.4f}  "
          f"ICIR={summ['ICIR']:.3f}  IC>0 ratio={summ['IC_positive_ratio']:.2f}")
    print("saved:", out_csv)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru", choices=["gru", "lstm", "mlp", "all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--roll-months", type=int, default=6,
                    help="滚动更新步长(月),6=半年两折(默认),3=季度,12=不滚动")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    folds = make_folds(args.roll_months)
    torch.set_num_threads(os.cpu_count())
    names = ["gru", "lstm", "mlp"] if args.model == "all" else [args.model]
    for n in names:
        run(n, seed=args.seed, device=device, folds=folds)

    # 汇总磁盘上已有的全部模型结果(支持分多次/多机器训练)
    summary = {}
    for n in ["gru", "lstm", "mlp"]:
        f = os.path.join(RESULTS_DIR, f"daily_rank_ic_{n}.csv")
        if os.path.exists(f):
            ic_s = pd.read_csv(f, index_col=0)["rank_ic"]
            summary[n] = ic_summary(ic_s)
    df = pd.DataFrame(summary).T
    df.to_csv(os.path.join(RESULTS_DIR, "model_ic_comparison.csv"))
    print("\n", df.round(4))
