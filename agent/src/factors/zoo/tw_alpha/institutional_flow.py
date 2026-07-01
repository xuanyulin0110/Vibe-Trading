
# ============================================================
# 中文名称: 台股三大法人籌碼流向動能
# 简要说明: rank(ts_mean(外資買賣超 + 投信買賣超, 10))，衡量近兩週外資與投信合計淨買超力道。
# 典型用途: 做多近期三大法人（外資+投信）持續買超的標的，做空持續遭賣超的標的。
# ============================================================
"""TW institutional-flow momentum (三大法人籌碼流向).

Formula: rank(ts_mean(institutional_foreign_net + institutional_trust_net, 10))

外資 (foreign) and 投信 (domestic trust) net buying/selling (股數, shares) is
the TWSE chip-flow signal retail traders watch most closely -- unlike 自營商
(dealer), foreign/trust flow is widely read as a "smart money" indicator.
Summing the two and taking a 10-trading-day (~2 week) mean smooths daily
noise while staying responsive to a real accumulation/distribution trend.

Source data: finlab institutional_investors_trading_summary via
FinlabFundamentalProvider (see backtest/loaders/finlab_fundamentals.py),
merged onto the price panel point-in-time-safe (same-day disclosure, no
look-ahead) by src.tools.alpha_bench_tool._load_tw_panel.
"""

from __future__ import annotations

import pandas as pd

from src.factors.base import rank, ts_mean

ALPHA_ID = "tw_alpha_institutional_flow"

__alpha_meta__ = {
    'id': 'tw_alpha_institutional_flow',
    'nickname': '三大法人籌碼流向動能',
    'theme': ['sentiment'],
    'formula_latex': 'rank(ts\\_mean(foreign\\_net + trust\\_net, 10))',
    'columns_required': [],
    'extras_required': ['institutional_foreign_net', 'institutional_trust_net'],
    'requires_sector': False,
    'universe': ['equity_tw'],
    'frequency': ['1D'],
    'decay_horizon': 10,
    'min_warmup_bars': 10,
    'notes': '外資+投信合計買賣超股數的10日均值排名；外陸資及自營商買賣超不計入，避免混入避險性質的自營商倉位。',
}


def compute(panel: dict) -> pd.DataFrame:
    """Compute the alpha on the panel and return a wide DataFrame."""
    flow = panel["institutional_foreign_net"] + panel["institutional_trust_net"]
    return rank(ts_mean(flow, 10))
