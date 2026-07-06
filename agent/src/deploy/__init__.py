"""Deterministic strategy deployment runtime (no LLM in the order path).

Runs a backtested run's ``signal_engine.py`` against fresh market data on a
per-bar schedule and converges the broker position to the signal's target --
the deterministic counterpart to (and fully isolated from) the LLM-in-the-loop
live framework under ``src/live``, which this package must never import.
"""
