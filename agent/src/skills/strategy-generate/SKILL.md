---
name: strategy-generate
description: Create, modify, and optimize quantitative trading strategies, then backtest and evaluate them.
category: strategy
---

## Workflow

1. **Requirements parsing**: parse user intent, extract instrument codes, time range, and strategy logic, then write `config.json`
2. **Strategy design**: think through the 5 questions of data / signal / position sizing / backtest / validation
3. **Strategy coding**: write `code/signal_engine.py` (following the `SignalEngine` contract)
4. **Syntax check**: `bash("python -c \"import ast; ast.parse(open('code/signal_engine.py').read()); print('OK')\"")`
5. **Run backtest**: call the `backtest` tool (built into the engine; no need to write `run_backtest.py`)
6. **Evaluate results**: read `artifacts/metrics.csv` and judge by the review criteria
7. **Iterative fixing**: if results are poor, modify with `edit_file` → run `backtest` → re-evaluate

**You only need to write `signal_engine.py` and `config.json`. The `backtest` tool automatically handles data loading and backtest execution.**

## Requirements Parsing

Extract the following from the user's description:
- **Instrument codes**: process them according to the normalization rules below
- **Time range**: if the user does not specify dates, default to **10 years back from today** (for example, if today is `2026-03-18`, then `start_date=2016-03-18`, `end_date=2026-03-18`)
- **Strategy logic**: entry / exit conditions and indicator parameters

**If critical information is missing, you must ask the user instead of guessing:**
- Instrument not specified → ask which instrument they want to backtest (offer several popular suggestions)
- Strategy description is vague (for example, "help me build a strategy") → provide 2-3 strategy directions for the user to choose from
- Mixed markets but not clearly specified → confirm the data source

**Write `config.json` first, then write code.** `config.json` must be placed in the root of `run_dir`.

## Strategy Design

Before writing code, think through these 5 questions:

1. **Data requirements**: what fields are needed (basic OHLCV only, daily valuation fields such as `pe/pb/roe`, or statement fields such as `income_total_revenue` / `fina_indicator_roe`?), data frequency (daily), and market (which determines the data source)
2. **Signal logic**: what are the entry conditions? What are the exit conditions? Direction (long / short / long-short)? Are there filters (volume, trend confirmation, and so on)?
3. **Position management**: equal-weight allocation or scaling in/out? Risk control (stop-loss, maximum position)? In portfolio strategies, once top N names are selected, each weight = 1/N
4. **Backtest parameters**: time range, initial capital (default 1,000,000), commission (default 0.1%)
5. **Validation checklist**: signal consistency (no NaN signals), position check (normalized to prevent leverage), and completeness of generated artifacts

There is no need to output a JSON design document. Express these design decisions directly in code.

## `SignalEngine` Contract

```python
class SignalEngine:
    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Args:
            data_map: code -> DataFrame (columns: open, high, low, close, volume, DatetimeIndex)
                     If config.extra_fields is specified (China A-shares only), pe, pb, roe, and similar
                     daily_basic columns will also be present.
                     If config.fundamental_fields is specified, PIT-safe statement/chip columns will also
                     be present, prefixed by table name -- e.g. income_total_revenue, fina_indicator_roe
                     for China A-shares, or institutional_foreign_net, fundamental_features_roe for
                     Taiwan equities/futures (see "Market Detection and Data Sources" below for the
                     full per-market table list).
        Returns:
            code -> signal Series, value range [-1.0, 1.0]
            1.0 = fully long, 0.5 = half position, 0.0 = flat, -1.0 = fully short
            Portfolio strategy: selected stocks split weights equally (for example top 10 -> each 0.1)
            Legacy integer signals {-1, 0, 1} remain compatible (treated as -100% / 0% / 100%)
        """
```

**Hard constraints:**
- The signal `Series` index must align exactly with the input `DataFrame` index
- Include all required imports (`numpy`, `pandas`, and so on)
- Do not hardcode dates or stock codes (read them from `config.json`)
- Do not include an `if __name__ == "__main__"` block
- Pure pandas / numpy implementation, with no external signal libraries
- Output plain Python code, not Markdown fences

## Quality Checklist

Self-check after writing `signal_engine.py`:
- [ ] All imports are included (`numpy`, `pandas`, `typing`, and so on)
- [ ] No undefined variables
- [ ] Signal logic is consistent with the strategy description
- [ ] Boundary handling: for empty data or insufficient history before the lookback window, use `fillna(0)` or skip
- [ ] Portfolio strategy: once N stocks are selected, each weight = 1/N (for example top 10 → each 0.1), unselected names = 0
- [ ] Signal values stay within `[-1.0, 1.0]`

## Instrument Code Normalization

- 6-digit China A-share codes → automatically append suffix: codes starting with `600/601/603` → `.SH`, all others → `.SZ`
- US stocks: uppercase letters + `.US`, such as `AAPL.US` (`yfinance` converts automatically)
- Hong Kong stocks: digits + `.HK`, such as `700.HK` (`yfinance` converts automatically)
- Cryptocurrencies: `BTC-USDT` format (OKX spot pairs, **must use the hyphen `-`, not slash `/`**)
  - The user may write `BTC/USDT`, but `config.json` must use `"BTC-USDT"`

## Cryptocurrency Notes

- **Code format**: must be `XXX-USDT` (uppercase + hyphen), such as `BTC-USDT` and `ETH-USDT`
- **source**: must be set to `"okx"`
- **extra_fields**: must be `null` (OKX does not support fundamentals)
- **Data format**: `DataLoader` has already normalized the output to match China A-shares exactly: `open, high, low, close, volume` + `DatetimeIndex`
- **No special handling needed in strategy code**: `signal_engine.py` should be written the same way as for China A-shares; do not add extra data conversion for OKX

## Market Detection and Data Sources

| Pattern | Market | source | Extra Fields |
|------|------|--------|----------|
| `^\d{6}\.(SZ\|SH\|BJ)$` | China A-shares | tushare | `extra_fields`: pe, pb, pe_ttm, ps_ttm, dv_ttm, total_mv, circ_mv, roe; `fundamental_fields`: income/balancesheet/cashflow/fina_indicator |
| `^[A-Z]+\.US$` | US stocks | yfinance | - |
| `^\d{3,5}\.HK$` | Hong Kong stocks | yfinance | - |
| `^[A-Z]+-USDT$` | Cryptocurrency | okx | - |
| `^\d{4,6}\.TW$` | Taiwan equities | finlab (primary) / shioaji | `fundamental_fields` only (see below); no `extra_fields` |
| `^\w+\.TWF$` | Taiwan index futures (TXF/MXF/TMF) | shioaji_futures | `fundamental_fields` only, table `futures_institutional` |

**`extra_fields` selection logic**: only China A-shares (`tushare`) support daily valuation fields. If the strategy needs `PE/PB/ROE` and similar daily_basic fields, specify them in `config.json.extra_fields` and `DataLoader` will retrieve them automatically. Hong Kong stocks, US stocks, crypto, and **Taiwan equities/futures do not support `extra_fields`** — always use `null` for these markets, even if the strategy wants ROE or similar fields (use `fundamental_fields` instead).

**`fundamental_fields` selection logic**: use this for financial-statement/chip-data pre-filters. Table names and the underlying provider are market-specific:

- **China A-shares**: queries `income`, `balancesheet`, `cashflow`, and/or `fina_indicator` through the Tushare fundamental provider, merged into daily bars only after their announcement/disclosure date. Output columns: `income_total_revenue`, `income_n_income`, `balancesheet_total_hldr_eqy_exc_min_int`, `fina_indicator_roe`.
- **Taiwan equities/futures**: routes to `FinlabFundamentalProvider` instead (`agent/backtest/loaders/finlab_fundamentals.py`) whenever any code in the backtest is a TW equity or `.TWF` future — no separate config needed to select it, `_detect_market()` does this automatically. Available tables (call `list_tables()`/`describe_table()` for the authoritative list, this may drift):

  | table | fields |
  |---|---|
  | `institutional` (三大法人) | foreign_net, foreign_ex_dealer_net, trust_net, dealer_self_net, dealer_hedge_net |
  | `margin` (融資融券) | margin_balance, margin_buy, margin_sell, margin_usage_rate, short_balance, short_buy, short_sell |
  | `monthly_revenue` (月營收) | revenue, revenue_yoy_pct, revenue_mom_pct |
  | `rotc_monthly_revenue` (興櫃月營收) | same as `monthly_revenue` |
  | `financial_statement` (財報科目, quarter-indexed, PIT-resolved via disclosure-date table) | total_assets, total_liabilities, total_equity, revenue, gross_profit, operating_income, net_income, eps, operating_cash_flow |
  | `fundamental_features` (財務比率, quarter-indexed, PIT-resolved) | roe, roa, gross_margin, operating_margin, net_margin, revenue_growth, current_ratio, debt_ratio, free_cash_flow |
  | `foreign_shareholding` (外資持股) | shares_held, holding_pct, remaining_investable_pct |
  | `director_shareholding` (董監持股不足, sparse) | director_insufficient_shares, supervisor_insufficient_shares |
  | `futures_institutional` (期貨三大法人, TXF/MXF/TMF only) | foreign_net_oi, trust_net_oi, dealer_net_oi, foreign_net_volume, trust_net_volume, dealer_net_volume |

  Output columns are prefixed by table name, e.g. `institutional_foreign_net`, `monthly_revenue_revenue_yoy_pct`, `fundamental_features_roe`. Example for a single-stock TW strategy:
  ```json
  "fundamental_fields": {
    "institutional": ["foreign_net"],
    "monthly_revenue": ["revenue_yoy_pct"],
    "fundamental_features": ["roe"]
  }
  ```
  `financial_statement`/`fundamental_features` are indexed by quarter (`'2025-Q1'`) internally, but the provider resolves each quarter to its real per-stock disclosure date before merging — TSMC's 2025-Q1 statement wasn't public until 2025-05-15, so a naive quarter-end merge would leak ~45 days of look-ahead; this is already handled, no extra work needed in `signal_engine.py`.

## `config.json` Format

```json
{
  "source": "auto",
  "codes": ["000001.SZ"],
  "start_date": "2016-03-18",
  "end_date": "2026-03-18",
  "interval": "1D",
  "initial_cash": 1000000,
  "commission": 0.001,
  "extra_fields": null,
  "fundamental_fields": null,
  "optimizer": null,
  "optimizer_params": {},
  "engine": "daily",
  "validation": null
}
```

- `source`: `"auto"` (recommended, auto-select by code format) / `"tushare"` / `"yfinance"` / `"okx"` / `"akshare"` / `"ccxt"`
  - `"auto"` supports mixed instruments. For example, `["000001.SZ", "BTC-USDT"]` will be automatically routed to `tushare` and `okx`
  - Futures codes (e.g. `"IF2406.CFFEX"`, `"ESZ4"`) and forex pairs (e.g. `"EUR/USD"`) are also auto-routed
- `interval`: candlestick interval, default `"1D"`. Supported values: `"1m"` / `"5m"` / `"15m"` / `"30m"` / `"1H"` / `"4H"` / `"1D"`
  - The annualization factor for minute backtests is inferred automatically from `source` (252 trading days for China A-shares, 365 calendar days for crypto)
  - Minute backtests can be very data-heavy. Recommended limits are no more than 30 days for `1m`, or 1 year for `1H`
- `extra_fields`: China A-shares can use values such as `["pe", "pb", "roe"]`; other markets (including Taiwan) should use `null`
- `fundamental_fields`: optional statement/chip pre-filter fields, table name -> field list. China A-shares: `{"income": ["total_revenue", "n_income"], "fina_indicator": ["roe"]}`. Taiwan equities/futures: `{"institutional": ["foreign_net"], "monthly_revenue": ["revenue_yoy_pct"], "fundamental_features": ["roe"]}` (see "Market Detection and Data Sources" for the full TW table list). `null` unless the strategy needs financial-statement/chip pre-filtering.
- `optimizer`: optional, one of `"equal_volatility"` / `"risk_parity"` / `"mean_variance"` / `"max_diversification"` / `null` (equal-weight by default)
- `optimizer_params`: optimizer parameters, such as `{"lookback": 60}`. `mean_variance` additionally supports `{"risk_free": 0.0}`
- `engine`: backtest engine, default `"daily"`. For options strategies, set `"options"` (requires `OptionsSignalEngine`)
- `initial_cash`: default 1,000,000
- `commission`: default 0.1%
- `validation`: optional statistical validation after backtest completes. Omit to skip. Example:
  ```json
  "validation": {
    "monte_carlo": {"n_simulations": 1000},
    "bootstrap": {"n_bootstrap": 1000, "confidence": 0.95},
    "walk_forward": {"n_windows": 5}
  }
  ```
  - `monte_carlo`: permutation test — shuffles trade order to compute p-value (is Sharpe significantly better than random?)
  - `bootstrap`: resamples daily returns to compute Sharpe 95% confidence interval
  - `walk_forward`: splits equity curve into N windows, checks performance consistency
  - Each key is optional — include only the validations you want
  - Can also run standalone on past results: `python -m backtest.validation <run_dir>`

## Review Criteria

### Hard Gates (any failure → `passed=false`)

1. `artifacts/metrics.csv` exists and is non-empty
2. `artifacts/equity.csv` exists and is non-empty
3. `exit_code == 0` (backtest exits normally)
4. The `equity` column in `equity.csv` contains no `NaN` values
5. `trade_count > 0` (zero trades = signal bug)

### Scoring Rules

- Successful backtest + complete artifacts + at least 1 trade → `score ≥ 60` → **passed**
- Poor return / low Sharpe alone should not push the score below 60; they are optimization suggestions only
- `score ≥ 60` = `passed=true`

### Bug Categories (reduce the score)

1. **Zero trades** (`trade_count=0`): signal-logic bug, conditions may be too strict
2. **Late first trade** (first trade > 2 years after backtest start): data-filtering bug or overly long lookback window
3. **Capital utilization < 50%**: position-management bug, portfolio is flat most of the time
4. **Open position at the end** (positions still open when backtest ends): exit-signal timing bug

### `action_items` Format

If improvements are needed after evaluation, write `action_items`:
- Format: `"Change X from A to B"` or `"Add X logic in signal_engine.py"`
- Must be specific down to parameter values, file names, and function names
- At least 2 items
- Examples:
  - `"Change short MA from 5 to 10 days to reduce whipsaw signals"`
  - `"Add stop-loss: force close when loss exceeds 5%"`
  - `"Add volume filter in signal_engine.py: only trigger buy on high volume"`

## Cross-Market Strategies

When the user requests a backtest with codes from **different markets** (e.g. `["000001.SZ", "BTC-USDT"]`):
- Set `source: "auto"` in `config.json`
- The `CompositeEngine` handles calendar alignment, shared capital, and per-market rules automatically
- Use volatility-adjusted weights so high-vol assets (crypto) don't dominate the risk budget
- See the [cross-market-strategy](../cross-market-strategy/SKILL.md) skill for per-market parameters, vol-adjustment, and example code

## Supporting Files

- [examples.md](examples.md) — example call sequence
