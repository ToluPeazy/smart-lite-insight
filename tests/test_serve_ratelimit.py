"""Tests for src/serve.py rate limiting — key function and the heavy endpoints.

Each test uses its own `CF-Connecting-IP` so it gets its own limiter bucket and
tests can't exhaust each other's quota within the same minute.
"""

import os
import sqlite3
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.requests import Request

from src.serve import app, client_identifier

os.environ["SMARTLITE_API_KEY"] = "test-key-for-ci"

client = TestClient(app, headers={"X-API-Key": "test-key-for-ci"})


def make_request(headers: dict | None = None, client_host: str = "10.0.0.1") -> Request:
    """Build a minimal Starlette Request for the key function."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/timeseries",
            "headers": [
                (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
            ],
            "client": (client_host, 54321),
        }
    )


def empty_db_connection():
    """A connection with no `readings` table, so queries fail fast with 503."""
    return sqlite3.connect(":memory:", check_same_thread=False)


class TestClientIdentifier:
    def test_prefers_cf_connecting_ip(self):
        request = make_request(
            {"CF-Connecting-IP": "203.0.113.7"}, client_host="10.0.0.1"
        )

        assert client_identifier(request) == "203.0.113.7"

    def test_falls_back_to_remote_address(self):
        assert client_identifier(make_request(client_host="10.0.0.1")) == "10.0.0.1"

    def test_strips_whitespace(self):
        request = make_request({"CF-Connecting-IP": " 203.0.113.7 "})

        assert client_identifier(request) == "203.0.113.7"

    def test_empty_header_falls_back(self):
        request = make_request({"CF-Connecting-IP": ""}, client_host="10.0.0.2")

        assert client_identifier(request) == "10.0.0.2"

    def test_different_cf_ips_are_different_keys(self):
        first = make_request({"CF-Connecting-IP": "203.0.113.7"})
        second = make_request({"CF-Connecting-IP": "203.0.113.8"})

        assert client_identifier(first) != client_identifier(second)


class TestTimeSeriesRateLimit:
    def test_returns_429_over_the_limit(self):
        headers = {"CF-Connecting-IP": "198.51.100.10"}

        with patch("src.serve.get_db_connection", side_effect=empty_db_connection):
            statuses = [
                client.get(
                    "/timeseries", params={"hours": 1}, headers=headers
                ).status_code
                for _ in range(21)
            ]

        assert 429 not in statuses[:20]
        assert statuses[-1] == 429

    def test_limit_is_per_client_ip(self):
        """A second caller is unaffected by the first caller's exhausted quota."""
        with patch("src.serve.get_db_connection", side_effect=empty_db_connection):
            for _ in range(21):
                client.get(
                    "/timeseries",
                    params={"hours": 1},
                    headers={"CF-Connecting-IP": "198.51.100.11"},
                )

            other = client.get(
                "/timeseries",
                params={"hours": 1},
                headers={"CF-Connecting-IP": "198.51.100.12"},
            )

        assert other.status_code != 429


class TestAnomaliesRateLimit:
    def test_returns_429_over_the_limit(self):
        headers = {"CF-Connecting-IP": "198.51.100.20"}

        with patch("src.serve.get_db_connection", side_effect=empty_db_connection):
            statuses = [
                client.get(
                    "/anomalies", params={"hours": 1}, headers=headers
                ).status_code
                for _ in range(11)
            ]

        assert 429 not in statuses[:10]
        assert statuses[-1] == 429


class TestScoreRateLimitStillApplies:
    def test_returns_429_over_the_limit(self):
        headers = {"CF-Connecting-IP": "198.51.100.30"}
        # A schema-valid body: request validation runs before the limiter, so a
        # malformed payload is rejected without consuming quota.
        payload = {
            "readings": [
                {
                    "timestamp": "2024-01-15T19:30:00",
                    "global_active_power_kw": 4.216,
                    "global_reactive_power_kw": 0.418,
                    "voltage_v": 234.84,
                    "global_intensity_a": 18.4,
                    "sub_metering_1_wh": 0.0,
                    "sub_metering_2_wh": 1.0,
                    "sub_metering_3_wh": 17.0,
                }
            ]
        }

        statuses = [
            client.post("/anomaly/score", json=payload, headers=headers).status_code
            for _ in range(31)
        ]

        assert 429 not in statuses[:30]
        assert statuses[-1] == 429
