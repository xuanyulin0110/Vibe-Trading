
# ============================================================
# 中文名称: 台股融資使用率槓桿風險
# 简要说明: -rank(delta(融資使用率, 5))，融資使用率5日內快速攀升的標的視為槓桿風險升高、給予較低分數；急速上升常伴隨散戶追價、部位脆弱。
# 典型用途: 做多近期去槓桿（融資使用率下降）的標的，做空融資使用率快速攀升、斷頭風險升高的標的。
# ============================================================
"""TW margin-usage leverage-risk factor (融資使用率槓桿風險).

Formula: -rank(delta(margin_margin_usage_rate, 5))

融資使用率 (margin_usage_rate) is TWSE's published ratio of a stock's used
margin-buy quota (融資餘額) against its total approved margin quota -- a
direct, per-stock leverage gauge unique to markets with a formal margin-quota
system (not simply "% owned via margin", but capacity utilisation). A fast
5-trading-day rise signals retail chasing on borrowed money -- historically a
fragile setup prone to forced margin-call liquidation on any pullback. This
factor scores *rising* leverage low (bearish/risk) and *falling* leverage
high (deleveraging, comparatively safer), i.e. a contrarian leverage-risk
signal rather than a momentum-following one.

Source data: finlab margin_transactions via FinlabFundamentalProvider,
merged point-in-time-safe (same-day disclosure) by
src.tools.alpha_bench_tool._load_tw_panel.
"""

from __future__ import annotations

import pandas as pd

from src.factors.base import delta, rank

ALPHA_ID = "tw_alpha_margin_leverage"

__alpha_meta__ = {
    'id': 'tw_alpha_margin_leverage',
    'nickname': '融資使用率槓桿風險',
    'theme': ['leverage', 'reversal'],
    'formula_latex': '-rank(delta(margin\\_usage\\_rate, 5))',
    'columns_required': [],
    'extras_required': ['margin_margin_usage_rate'],
    'requires_sector': False,
    'universe': ['equity_tw'],
    'frequency': ['1D'],
    'decay_horizon': 5,
    'min_warmup_bars': 6,
    'notes': '融資使用率5日變動的反向排名；不看槓桿的絕對水準（不同股票核定額度不同，無法跨標的比較水準），只看變動速度。',
}


def compute(panel: dict) -> pd.DataFrame:
    """Compute the alpha on the panel and return a wide DataFrame."""
    usage = panel["margin_margin_usage_rate"]
    return -rank(delta(usage, 5))
