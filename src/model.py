# -*- coding: utf-8 -*-
"""模型定义

输入张量: [batch, seq_len=40, n_features=29]
输出:     [batch],预测截面标准化后的未来 5 日收益

- GRUNet (主模型): 2 层 GRU(hidden=64) + 全连接头,取最后时刻隐状态
- LSTMNet(对比):  同结构,GRU 换 LSTM
- MLPNet (对比):  仅用 T 日截面特征的浅层网络,作为"时序信息是否有用"的对照
"""
import torch
import torch.nn as nn


class RNNBase(nn.Module):
    def __init__(self, rnn_cls, n_features: int, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = rnn_cls(input_size=n_features, hidden_size=hidden,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):                      # x: [B, T, F]
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GRUNet(RNNBase):
    def __init__(self, n_features, **kw):
        super().__init__(nn.GRU, n_features, **kw)


class LSTMNet(RNNBase):
    def __init__(self, n_features, **kw):
        super().__init__(nn.LSTM, n_features, **kw)


class MLPNet(nn.Module):
    """仅使用序列最后一天(T 日)的截面特征"""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):                      # x: [B, T, F]
        return self.net(x[:, -1, :]).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return GRUNet(n_features)
    if name == "lstm":
        return LSTMNet(n_features)
    if name == "mlp":
        return MLPNet(n_features)
    raise ValueError(f"unknown model: {name}")
