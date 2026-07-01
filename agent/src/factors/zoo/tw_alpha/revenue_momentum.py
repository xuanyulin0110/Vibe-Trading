
# ============================================================
# 中文名称: 台股月營收年增率動能
# 简要说明: rank(monthly_revenue_revenue_yoy_pct)，當月營收年增率的橫截面排名，衡量基本面成長動能。
# 典型用途: 做多月營收年增率領先同業的標的，做空年增率落後或衰退的標的。
# ============================================================
"""TW monthly-revenue growth momentum (月營收年增率動能).

Formula: rank(monthly_revenue_revenue_yoy_pct)

Taiwan-listed companies are unique among major markets in disclosing
consolidated revenue *monthly* (by the 10th of the following month per TWSE
rule), well ahead of quarterly earnings -- 營收年增率 (revenue YoY growth) is
the earliest fundamental growth signal available and a long-documented TW
factor (roughly analogous to SUE/PEAD-style fundamental momentum in US
equity research, but on a monthly rather than quarterly cadence). A plain
cross-sectional rank of the latest disclosed YoY figure is used rather than
a rolling window: the figure itself already IS a year-over-year comparison,
so no additional smoothing is needed, and using the raw level (not its
change) keeps the factor a straightforward "who's growing fastest right now"
signal rather than an acceleration/deceleration one.

Source data: finlab monthly_revenue via FinlabFundamentalProvider, merged
point-in-time-safe (actual announcement date, not period-end) by
src.tools.alpha_bench_tool._load_tw_panel -- see finlab_fundamentals.py's
module docstring for why this table's date index is already PIT-safe.
"""

from __future__ import annotations

import pandas as pd

from src.factors.base import rank

ALPHA_ID = "tw_alpha_revenue_momentum"

__alpha_meta__ = {
    'id': 'tw_alpha_revenue_momentum',
    'nickname': '月營收年增率動能',
    'theme': ['growth', 'momentum'],
    'formula_latex': 'rank(revenue\\_yoy\\_pct)',
    'columns_required': [],
    'extras_required': ['monthly_revenue_revenue_yoy_pct'],
    'requires_sector': False,
    'universe': ['equity_tw'],
    'frequency': ['1D'],
    'decay_horizon': 20,
    'min_warmup_bars': 1,
    'notes': '月營收每月僅更新一次，訊號在下次公告前維持不變（PIT前值延續），decay_horizon以約一個月的交易日數設定。',
}


def compute(panel: dict) -> pd.DataFrame:
    """Compute the alpha on the panel and return a wide DataFrame."""
    return rank(panel["monthly_revenue_revenue_yoy_pct"])
