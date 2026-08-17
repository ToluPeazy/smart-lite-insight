"""Tests for the src/serve.py CLI entry point."""

from unittest.mock import patch

import pytest

from src.serve import main, reload_enabled


class TestReloadEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("SMARTLITE_RELOAD", raising=False)

        assert reload_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_enabled_by_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("SMARTLITE_RELOAD", value)

        assert reload_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_disabled_by_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("SMARTLITE_RELOAD", value)

        assert reload_enabled() is False


class TestMain:
    def test_runs_without_reload_by_default(self, monkeypatch):
        monkeypatch.delenv("SMARTLITE_RELOAD", raising=False)

        with patch("uvicorn.run") as run:
            main()

        assert run.call_args.kwargs["reload"] is False
        assert run.call_args.kwargs["port"] == 8000

    def test_reload_opt_in(self, monkeypatch):
        monkeypatch.setenv("SMARTLITE_RELOAD", "1")

        with patch("uvicorn.run") as run:
            main()

        assert run.call_args.kwargs["reload"] is True
