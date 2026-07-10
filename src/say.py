# -*- coding: utf-8 -*-
"""Makefile 步骤说明打印器:python src/say.py <key>

中文消息集中于本文件(UTF-8 源码,直接可读可改),经 Python 的
Windows 宽字符控制台 API 输出,任意终端代码页下显示均正常;
Makefile 侧仅出现 ASCII 键名,保持配方可读。
(echo 输出裸字节,中文在 GBK 代码页终端下会乱码,故不用 echo。)
"""
import sys

MSGS = {
    "step1": "========== [1/3] 训练 GRU 主模型 ==========",
    "step2": "========== [2/3] 最终版回测 ==========",
    "step3": "========== [3/3] 生成图表 ==========",
    "done": ("========== 全部完成:最终指标 results/backtest_results_smooth40.csv"
             " | 净值曲线 results/equity_curve_smooth40.png =========="),
    "train-clean": ">>> 清理旧预测文件(防止旧文件干扰)",
    "train-run": (">>> GRU 两折滚动训练:首次运行先构建特征缓存约 3 分钟;"
                  "GPU 约 5 分钟 / CPU 约 2 小时\n"),
    "train-check": ">>> 校验预测文件已由本次训练生成",
    "train-set": ">>> 将 GRU 预测设为回测默认输入 results/predictions.csv",
    "final": (">>> 最终版回测:周度调仓 Top200 等权 + 换手缓冲350 + 行业中性"
              " + 40日信号平滑,一字板约束默认启用,双边千三成本"),
    "plots": ">>> 生成图表:IC 时序、预测分组收益、特征重要性",
}

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    print(MSGS.get(key, key))
