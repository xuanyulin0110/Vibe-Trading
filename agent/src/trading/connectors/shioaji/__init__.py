"""Shioaji (SinoPac, 永豐金證券) trading connector.

Read-only Taiwan equity/futures account and market access in this phase
(account/positions/quote/history). Order placement is deferred to a later
phase behind the mandate gate, matching the Tiger connector's layering.

Unlike Tiger's paper-vs-live split (two different account-number formats),
Shioaji uses ONE account with a ``simulation`` boolean toggle for which
trading environment a session talks to; market data is the same real feed
in both modes (see ``references/PREPARE.md`` in the bundled Shioaji skill).
"""
