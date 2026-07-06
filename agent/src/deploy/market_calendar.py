"""TWSE/TAIFEX trading-day, session, and settlement-date arithmetic.

Deliberately independent of ``src/live/runtime/triggers.py`` (the LLM
framework's US-centric market specs) -- this module is the deploy runtime's
single source of truth for Taiwan market time. Everything computes in
``Asia/Taipei`` explicitly; nothing may read the container's system timezone
(containers default to UTC).

Session facts (verified against real TXFR1 minute data and official TAIFEX
docs earlier in this project -- see ``_shioaji_kbars._assign_taifex_trading_day``):

* TWSE equities: 09:00-13:30, day session only.
* TAIFEX futures day session: 08:45-13:45.
* TAIFEX futures night session: 15:00 -> next calendar day 05:00 (crosses
  midnight; attributed to the NEXT trading day by the exchange's clearing
  convention -- attribution matters to bar data, which ``_shioaji_kbars``
  already handles; here we only care about "is the market open now").

The holiday table is best-effort and hand-maintained (same trade-off as the
LLM framework's ``_US_EQUITY_HOLIDAYS``): a missed holiday just fires a tick
that the broker rejects/no-ops -- wasteful, not unsafe. Verify against
https://www.twse.com.tw/holidaySchedule when extending.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")

# Markets this runtime schedules.
TW_EQUITY = "tw_equity"
TW_FUTURES = "tw_futures"

EQUITY_OPEN = dt.time(9, 0)
EQUITY_CLOSE = dt.time(13, 30)
FUTURES_DAY_OPEN = dt.time(8, 45)
FUTURES_DAY_CLOSE = dt.time(13, 45)
FUTURES_NIGHT_OPEN = dt.time(15, 0)
FUTURES_NIGHT_CLOSE = dt.time(5, 0)  # next calendar day

#: Best-effort Taiwan market closures (TWSE & TAIFEX share these). Weekends
#: are handled separately. Extend year by year; wrong entries only waste a
#: tick, they never place a bad order.
_HOLIDAYS: frozenset[dt.date] = frozenset(
    {
        # 2026
        dt.date(2026, 1, 1),
        # Lunar New Year window (best-effort; includes eve and market-closure tail)
        dt.date(2026, 2, 12),
        dt.date(2026, 2, 13),
        dt.date(2026, 2, 16),
        dt.date(2026, 2, 17),
        dt.date(2026, 2, 18),
        dt.date(2026, 2, 19),
        dt.date(2026, 2, 20),
        dt.date(2026, 2, 27),  # 228 Peace Memorial Day (observed)
        dt.date(2026, 4, 3),   # Children's Day (observed)
        dt.date(2026, 4, 6),   # Tomb Sweeping Day (observed)
        dt.date(2026, 5, 1),   # Labor Day
        dt.date(2026, 6, 19),  # Dragon Boat Festival
        dt.date(2026, 9, 25),  # Mid-Autumn Festival
        dt.date(2026, 10, 9),  # National Day (observed)
    }
)


def now_taipei() -> dt.datetime:
    """Timezone-aware current time in Asia/Taipei."""
    return dt.datetime.now(TAIPEI)


def is_trading_day(day: dt.date) -> bool:
    """Weekday and not a known holiday."""
    return day.weekday() < 5 and day not in _HOLIDAYS


def prev_trading_day(day: dt.date) -> dt.date:
    d = day - dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def next_trading_day(day: dt.date) -> dt.date:
    d = day + dt.timedelta(days=1)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def session_open_now(market: str, now: dt.datetime, *, include_night: bool = False) -> bool:
    """Whether ``market`` is in-session at ``now`` (tz-aware, any zone).

    Night session (futures only, ``include_night=True``) crosses midnight:
    the 15:00-24:00 leg requires *today* to be a trading day (a session that
    starts trading must have opened on a trading day), while the 00:00-05:00
    leg requires *yesterday* to have been a trading day (it is the tail of
    the session that opened yesterday 15:00).
    """
    local = now.astimezone(TAIPEI)
    day, t = local.date(), local.time()

    if market == TW_EQUITY:
        return is_trading_day(day) and EQUITY_OPEN <= t < EQUITY_CLOSE

    if market == TW_FUTURES:
        if is_trading_day(day) and FUTURES_DAY_OPEN <= t < FUTURES_DAY_CLOSE:
            return True
        if include_night:
            if is_trading_day(day) and t >= FUTURES_NIGHT_OPEN:
                return True
            if t < FUTURES_NIGHT_CLOSE and is_trading_day(day - dt.timedelta(days=1)):
                return True
        return False

    raise ValueError(f"unknown market {market!r}")


def day_session_close(market: str) -> dt.time:
    """Day-session close time -- the completeness cutoff for 1D bars."""
    return EQUITY_CLOSE if market == TW_EQUITY else FUTURES_DAY_CLOSE


def day_session_open(market: str) -> dt.time:
    return EQUITY_OPEN if market == TW_EQUITY else FUTURES_DAY_OPEN


def taifex_settlement_date(year: int, month: int) -> dt.date:
    """TAIFEX index-futures final settlement day: the 3rd Wednesday.

    If that Wednesday is a holiday TAIFEX shifts settlement to the next
    business day -- mirrored here so rollover logic lands on the true final
    trading day.
    """
    first = dt.date(year, month, 1)
    offset = (2 - first.weekday()) % 7  # Wednesday == weekday 2
    third_wed = first + dt.timedelta(days=offset + 14)
    while not is_trading_day(third_wed):
        third_wed += dt.timedelta(days=1)
    return third_wed


def is_settlement_day(day: dt.date) -> bool:
    return day == taifex_settlement_date(day.year, day.month)
