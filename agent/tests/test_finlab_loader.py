from __future__ import annotations

import pytest

from backtest.loaders.finlab_loader import DataLoader


class TestLazyLogin:
    def test_init_does_not_log_in(self) -> None:
        """Constructing a loader must not touch finlab -- login is deferred to fetch()."""
        loader = DataLoader()
        assert loader._logged_in is False

    def test_ensure_logged_in_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        class _FakeFinlab:
            @staticmethod
            def login(token: str) -> None:
                calls.append(token)

        monkeypatch.setenv("FINLAB_API_TOKEN", "real-token")
        monkeypatch.setitem(__import__("sys").modules, "finlab", _FakeFinlab)

        loader = DataLoader()
        loader._ensure_logged_in()
        loader._ensure_logged_in()

        assert calls == ["real-token"]


class TestEmptyOrPlaceholderToken:
    @pytest.mark.parametrize("token", ["", "your-finlab-token"])
    def test_ensure_logged_in_raises_instead_of_calling_finlab(
        self, monkeypatch: pytest.MonkeyPatch, token: str,
    ) -> None:
        """A placeholder/empty token must never reach finlab.login() -- finlab's SDK
        falls back to an interactive browser-auth flow that prints to stdout and
        blocks, which is fatal for a headless MCP subprocess."""
        monkeypatch.setenv("FINLAB_API_TOKEN", token)

        loader = DataLoader()
        with pytest.raises(RuntimeError, match="FINLAB_API_TOKEN is not configured"):
            loader._ensure_logged_in()
        assert loader._logged_in is False

    def test_fetch_raises_before_any_network_call_with_no_token(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("FINLAB_API_TOKEN", raising=False)

        loader = DataLoader()
        with pytest.raises(RuntimeError, match="FINLAB_API_TOKEN is not configured"):
            loader.fetch(["2330.TW"], "2024-01-01", "2024-01-31")
