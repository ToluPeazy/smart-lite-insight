"""LLM Agent for Smart-Lite Insight.

A local LLM agent (Ollama) that can query energy data, explain anomalies,
and manage the ML pipeline using tool-calling patterns.

Usage:
    # As a module
    from src.agent import Agent
    agent = Agent()
    response = agent.chat("What was my peak usage this week?")

    # CLI
    python -m src.agent
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

import requests
from loguru import logger

from src.train import DEFAULT_DB_PATH

# ── Configuration ──

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are an energy monitoring assistant for Smart-Lite Insight, a household energy anomaly detection system running on a Raspberry Pi 5.

You have access to tools that query real energy data from a SQLite database. Use them to answer questions about energy consumption, anomalies, and patterns.

Rules:
1. ALWAYS use tools to get data before answering questions about energy usage. Never guess or make up numbers.
2. When asked about anomalies, use the get_anomalies tool.
3. When asked about consumption or usage, use the get_timeseries tool.
4. When asked about the model, use the get_model_info tool.
5. Keep responses concise and actionable.
6. If a tool returns an error, explain it clearly to the user.
7. Format numbers nicely (e.g., "4.21 kW" not "4.216148").
8. When discussing time periods, be specific about dates and times.

The data comes from the UCI Individual Household Electric Power Consumption dataset — a real French household measured at 1-minute intervals from 2006 to 2010."""

# ── Tool Definitions ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_timeseries",
            "description": "Get energy consumption data for a time period. Returns timestamps and power readings. Use this when asked about usage, consumption, patterns, or trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Start datetime (ISO 8601, e.g. '2007-01-15T00:00:00')",
                    },
                    "end": {
                        "type": "string",
                        "description": "End datetime (ISO 8601, e.g. '2007-01-15T23:59:59')",
                    },
                    "resample": {
                        "type": "string",
                        "description": "Resample interval: '1h', '15min', '1D'. Default '1h' for readability.",
                        "default": "1h",
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomalies",
            "description": "Find anomalous energy readings in a time period. Returns the most severe anomalies sorted by score. Use this when asked about unusual usage, spikes, or problems.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Start datetime (ISO 8601)",
                    },
                    "end": {
                        "type": "string",
                        "description": "End datetime (ISO 8601)",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top anomalies to return. Default 10.",
                        "default": 10,
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statistics",
            "description": "Get summary statistics for energy consumption over a period. Returns mean, max, min, total consumption, and reading count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Start datetime (ISO 8601)",
                    },
                    "end": {
                        "type": "string",
                        "description": "End datetime (ISO 8601)",
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Get information about the currently trained anomaly detection model, including version, training date, number of features, and anomaly rate.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date_range",
            "description": "Get the available date range in the database. Use this when the user asks about what data is available or when you need to know valid date ranges.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrain_model",
            "description": "Retrain the anomaly detection model. This is a WRITE operation that requires explicit user confirmation. Only call this if the user has clearly asked to retrain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to proceed. Ask the user to confirm first.",
                    },
                },
                "required": ["confirm"],
            },
        },
    },
]


# ── Tool Implementations ──


def tool_get_timeseries(start: str, end: str, resample: str = "1h") -> dict:
    """Fetch time-series data from SQLite."""
    try:
        import pandas as pd

        conn = sqlite3.connect(DEFAULT_DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT timestamp, global_active_power_kw, voltage_v,
                   global_intensity_a, sub_metering_1_wh,
                   sub_metering_2_wh, sub_metering_3_wh
            FROM readings
            WHERE site_id = 'home-01' AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            conn,
            params=(start, end),
            parse_dates=["timestamp"],
        )
        conn.close()

        if df.empty:
            return {"error": f"No data found between {start} and {end}"}

        df = df.set_index("timestamp")

        if resample:
            df = df.resample(resample).mean().dropna()

        # Limit output size for LLM context
        if len(df) > 48:
            df = df.head(48)
            truncated = True
        else:
            truncated = False

        records = []
        for ts, row in df.iterrows():
            records.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "power_kw": round(float(row["global_active_power_kw"]), 3),
                "voltage_v": round(float(row["voltage_v"]), 1),
            })

        return {
            "readings": records,
            "count": len(records),
            "truncated": truncated,
            "resample": resample,
        }

    except Exception as e:
        return {"error": str(e)}


def tool_get_anomalies(start: str, end: str, top_n: int = 10) -> dict:
    """Find anomalies in a time range using the trained model."""
    try:
        import pandas as pd

        from src.detect import AnomalyDetector
        from src.features import build_feature_matrix

        conn = sqlite3.connect(DEFAULT_DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT timestamp, global_active_power_kw, global_reactive_power_kw,
                   voltage_v, global_intensity_a,
                   sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
            FROM readings
            WHERE site_id = 'home-01' AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            conn,
            params=(start, end),
            parse_dates=["timestamp"],
        )
        conn.close()

        if df.empty:
            return {"error": f"No data found between {start} and {end}"}

        df = df.set_index("timestamp")
        df_features = build_feature_matrix(df, drop_na=True)

        if df_features.empty:
            return {"error": "Not enough data for feature engineering (need 24h+)"}

        detector = AnomalyDetector()
        scored = detector.score_dataframe(df_features)
        anomalies = detector.get_anomalies(scored, top_n=top_n)

        total_anomalies = scored["is_anomaly"].sum()

        results = []
        for ts, row in anomalies.iterrows():
            results.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "power_kw": round(float(row["global_active_power_kw"]), 3),
                "voltage_v": round(float(row["voltage_v"]), 1),
                "anomaly_score": round(float(row["anomaly_score"]), 4),
            })

        return {
            "anomalies": results,
            "total_anomalies_in_range": int(total_anomalies),
            "total_readings_in_range": len(scored),
            "anomaly_rate": round(float(total_anomalies / len(scored) * 100), 2),
        }

    except FileNotFoundError:
        return {"error": "No trained model found. Run 'python -m src.train' first."}
    except Exception as e:
        return {"error": str(e)}


def tool_get_statistics(start: str, end: str) -> dict:
    """Get summary statistics for a time period."""
    try:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as readings,
                ROUND(AVG(global_active_power_kw), 3) as avg_power,
                ROUND(MAX(global_active_power_kw), 3) as max_power,
                ROUND(MIN(global_active_power_kw), 3) as min_power,
                ROUND(AVG(voltage_v), 1) as avg_voltage,
                ROUND(MIN(voltage_v), 1) as min_voltage,
                ROUND(MAX(voltage_v), 1) as max_voltage,
                ROUND(AVG(global_intensity_a), 1) as avg_current
            FROM readings
            WHERE site_id = 'home-01' AND timestamp BETWEEN ? AND ?
            """,
            (start, end),
        )
        row = cursor.fetchone()
        conn.close()

        if row[0] == 0:
            return {"error": f"No data found between {start} and {end}"}

        # Estimate total kWh (each reading is 1 minute)
        total_kwh = round(row[1] * row[0] / 60, 2)

        return {
            "period": {"start": start, "end": end},
            "readings": row[0],
            "power_kw": {
                "mean": row[1],
                "max": row[2],
                "min": row[3],
            },
            "voltage_v": {
                "mean": row[4],
                "min": row[5],
                "max": row[6],
            },
            "avg_current_a": row[7],
            "estimated_total_kwh": total_kwh,
        }

    except Exception as e:
        return {"error": str(e)}


def tool_get_model_info() -> dict:
    """Get model registry information."""
    try:
        from pathlib import Path

        registry_path = Path("models") / "registry.json"
        if not registry_path.is_file():
            return {"error": "No model registry found. Train a model first."}

        with open(registry_path) as f:
            registry = json.load(f)

        if not registry.get("models"):
            return {"error": "No models in registry."}

        latest = registry["models"][-1]
        return {
            "model_name": latest["model_name"],
            "version": latest["version"],
            "training_date": latest["training_date"],
            "n_training_samples": latest["n_training_samples"],
            "anomaly_rate": f"{latest['anomaly_rate']:.2%}",
            "n_features": len(latest["feature_names"]),
            "total_models_in_registry": len(registry["models"]),
        }

    except Exception as e:
        return {"error": str(e)}


def tool_get_date_range() -> dict:
    """Get available date range in the database."""
    try:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        cursor = conn.execute(
            """
            SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
            FROM readings WHERE site_id = 'home-01'
            """
        )
        row = cursor.fetchone()
        conn.close()

        return {
            "earliest": row[0],
            "latest": row[1],
            "total_readings": row[2],
        }

    except Exception as e:
        return {"error": str(e)}


def tool_retrain_model(confirm: bool = False) -> dict:
    """Retrain the anomaly detection model."""
    if not confirm:
        return {
            "status": "blocked",
            "message": "Retraining requires explicit confirmation. Ask the user to confirm.",
        }

    try:
        from src.train import train_pipeline

        results = train_pipeline(compare=False)
        return {
            "status": "success",
            "message": "Model retrained successfully.",
            "version": results.get("version", "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}


# Tool dispatch
TOOL_DISPATCH = {
    "get_timeseries": tool_get_timeseries,
    "get_anomalies": tool_get_anomalies,
    "get_statistics": tool_get_statistics,
    "get_model_info": tool_get_model_info,
    "get_date_range": tool_get_date_range,
    "retrain_model": tool_retrain_model,
}


# ── Agent ──


class Agent:
    """LLM Agent with tool-calling for energy data queries."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.conversation: list[dict] = []
        self.tool_log: list[dict] = []

        # Verify Ollama is reachable
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model not in models:
                logger.warning(f"Model '{self.model}' not found. Available: {models}")
            else:
                logger.info(f"Connected to Ollama. Model: {self.model}")
        except requests.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise ConnectionError(
                f"Ollama not reachable at {self.base_url}. "
                "Start it with 'ollama serve' or check if it's running."
            )

    def _call_ollama(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Make a chat completion request to Ollama."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1024,
            },
        }

        if tools:
            payload["tools"] = tools

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a string."""
        logger.info(f"Tool call: {tool_name}({arguments})")

        func = TOOL_DISPATCH.get(tool_name)
        if func is None:
            result = {"error": f"Unknown tool: {tool_name}"}
        else:
            try:
                result = func(**arguments)
            except Exception as e:
                result = {"error": f"Tool execution failed: {e}"}

        # Log the tool call
        self.tool_log.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "arguments": arguments,
            "result_summary": str(result)[:200],
        })

        return json.dumps(result, indent=2)

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, handling tool calls."""
        # Build messages with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation)
        messages.append({"role": "user", "content": user_message})

        # Store user message
        self.conversation.append({"role": "user", "content": user_message})

        # First LLM call (may trigger tool use)
        response = self._call_ollama(messages, tools=TOOLS)
        msg = response.get("message", {})

        # Handle tool calls (up to 3 rounds)
        rounds = 0
        while msg.get("tool_calls") and rounds < 3:
            rounds += 1

            # Add assistant's tool-call message
            messages.append(msg)

            # Execute each tool call
            for tool_call in msg["tool_calls"]:
                func = tool_call["function"]
                tool_name = func["name"]
                arguments = func.get("arguments", {})

                result = self._execute_tool(tool_name, arguments)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": result,
                })

            # Call LLM again with tool results
            response = self._call_ollama(messages, tools=TOOLS)
            msg = response.get("message", {})

        # Extract final text response
        content = msg.get("content", "I wasn't able to generate a response.")

        # Store assistant response
        self.conversation.append({"role": "assistant", "content": content})

        return content

    def reset(self):
        """Clear conversation history."""
        self.conversation = []

    def get_tool_log(self) -> list[dict]:
        """Return the audit log of all tool calls."""
        return self.tool_log


# ── CLI ──


def main():
    """Interactive CLI chat with the agent."""
    print("=" * 60)
    print("  Smart-Lite Insight — Energy AI Assistant")
    print("  Type 'quit' to exit, 'reset' to clear history")
    print("  Type 'log' to see tool call history")
    print("=" * 60)
    print()

    try:
        agent = Agent()
    except ConnectionError as e:
        print(f"Error: {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("Conversation history cleared.\n")
            continue

        if user_input.lower() == "log":
            log = agent.get_tool_log()
            if not log:
                print("No tool calls yet.\n")
            else:
                for entry in log:
                    print(f"  [{entry['timestamp'][:19]}] {entry['tool']}({entry['arguments']})")
                    print(f"    → {entry['result_summary'][:100]}")
                print()
            continue

        print("Thinking...", end="", flush=True)
        try:
            response = agent.chat(user_input)
            print(f"\r{'':12}\r", end="")  # Clear "Thinking..."
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"\rError: {e}\n")


if __name__ == "__main__":
    main()
