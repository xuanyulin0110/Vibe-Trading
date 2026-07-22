"""Regression tests for the MCP ``factor_analysis`` contract.

Upstream issue #635: the MCP wrapper forwarded ``codes``/``factor_name``/…
while the registered ``FactorAnalysisTool`` required ``factor_csv``/
``return_csv``/``output_dir`` — every MCP call died on ``KeyError:
'factor_csv'`` before any analysis ran. Upstream fixed the mismatch by making
the wrapper mirror the CSV contract.

This fork resolved the same crash the other way (2026-07): remote MCP clients
(Claude Code over Streamable HTTP, committee sub-agents) cannot stage CSV
files on the server, so the wrapper is a self-contained codes-based tool that
routes through ``run_factor_analysis_by_codes`` (which fetches prices and
fundamentals itself). These tests pin *that* contract so a future upstream
merge can't silently swap back the server-side-file API.
"""

from __future__ import annotations

import inspect

import mcp_server

# fastmcp wraps the tool; reach the raw callable.
_fa = getattr(mcp_server.factor_analysis, "fn", None) or getattr(
    mcp_server.factor_analysis, "__wrapped__", mcp_server.factor_analysis
)


def test_wrapper_signature_is_codes_based() -> None:
    """Drift guard: the remote-client-usable contract, not the CSV one."""
    sig = inspect.signature(_fa)
    assert set(sig.parameters) == {
        "codes", "factor_name", "start_date", "end_date",
        "source", "top_n", "bottom_n",
    }
    required = {
        name for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
    }
    assert required == {"codes", "factor_name", "start_date", "end_date"}
    # The CSV contract needs server-local files; remote MCP clients can't
    # provide those. Its keys must never reappear here.
    assert not {"factor_csv", "return_csv", "output_dir"} & set(sig.parameters)


def test_wrapper_forwards_to_codes_implementation(monkeypatch) -> None:
    """The wrapper must forward every argument to the codes-based runner."""
    calls: list[dict] = []

    def _record(**kwargs) -> str:
        calls.append(kwargs)
        return '{"status": "ok"}'

    import src.tools.factor_analysis_by_codes as impl

    monkeypatch.setattr(impl, "run_factor_analysis_by_codes", _record)

    _fa(
        codes=["2330.TW", "2317.TW"],
        factor_name="roe",
        start_date="2025-01-01",
        end_date="2025-06-30",
        source="auto",
        top_n=1,
        bottom_n=1,
    )

    assert calls == [{
        "codes": ["2330.TW", "2317.TW"],
        "factor_name": "roe",
        "start_date": "2025-01-01",
        "end_date": "2025-06-30",
        "source": "auto",
        "top_n": 1,
        "bottom_n": 1,
    }]
