"""Tests for src/agent.py — tool surface of the in-process LLM agent.

The agent is reachable from the unauthenticated dashboard, so its exposed
tools are a security boundary: they must stay read-only. These tests build an
Agent without touching __init__ (which requires a live Ollama) and exercise
the tool dispatch directly.
"""

import json

import pytest

import src.agent as agent_module
from src.agent import TOOL_DISPATCH, TOOLS, Agent


@pytest.fixture
def offline_agent():
    """An Agent instance without the Ollama connection check in __init__."""
    a = Agent.__new__(Agent)
    a.base_url = "http://localhost:11434"
    a.model = "llama3.1:8b"
    a.conversation = []
    a.tool_log = []
    return a


def tool_names() -> set[str]:
    return {t["function"]["name"] for t in TOOLS}


class TestToolSurfaceIsReadOnly:
    def test_retrain_tool_not_advertised(self):
        assert "retrain_model" not in tool_names()

    def test_retrain_tool_not_dispatchable(self):
        assert "retrain_model" not in TOOL_DISPATCH

    def test_retrain_implementation_removed(self):
        assert not hasattr(agent_module, "tool_retrain_model")

    def test_every_advertised_tool_has_an_implementation(self):
        assert tool_names() == set(TOOL_DISPATCH)

    def test_advertised_tools_are_the_expected_read_only_set(self):
        assert tool_names() == {
            "get_timeseries",
            "get_anomalies",
            "get_statistics",
            "get_model_info",
            "get_date_range",
        }


class TestRetrainRequestIsDeclined:
    def test_retrain_call_is_rejected_as_unknown_tool(self, offline_agent):
        result = json.loads(
            offline_agent._execute_tool("retrain_model", {"confirm": True})
        )

        assert result == {"error": "Unknown tool: retrain_model"}

    def test_rejected_call_does_not_trigger_training(self, offline_agent, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("train_pipeline must not be reachable via the agent")

        monkeypatch.setattr("src.train.train_pipeline", fail)

        offline_agent._execute_tool("retrain_model", {"confirm": True})

    def test_rejected_call_is_logged(self, offline_agent):
        offline_agent._execute_tool("retrain_model", {"confirm": True})

        assert len(offline_agent.tool_log) == 1
        assert offline_agent.tool_log[0]["tool"] == "retrain_model"

    def test_no_pending_retrain_state(self, offline_agent):
        """The 'type yes to confirm' flow is gone along with the tool."""
        assert not hasattr(offline_agent, "_retrain_pending")


class TestArgumentFiltering:
    def test_unexpected_arguments_are_dropped(self, offline_agent, monkeypatch):
        seen = {}

        def fake_date_range(**kwargs):
            seen.update(kwargs)
            return {"ok": True}

        monkeypatch.setitem(TOOL_DISPATCH, "get_date_range", fake_date_range)

        offline_agent._execute_tool("get_date_range", {"db_path": "/etc/passwd"})

        assert seen == {}
