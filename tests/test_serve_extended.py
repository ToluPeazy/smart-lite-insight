"""Additional tests for src/serve.py — covering previously untested paths."""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serve import ReadingInput, app, get_db_connection, readings_to_dataframe

# client = TestClient(app)

os.environ["SMARTLITE_API_KEY"] = "test-key-for-ci"

client = TestClient(app, headers={"X-API-Key": "test-key-for-ci"})

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_reading(offset_minutes: int = 0) -> dict:
    ts = datetime(2024, 1, 15, 19, 0, 0) + timedelta(minutes=offset_minutes)
    return {
        "timestamp": ts.isoformat(),
        "global_active_power_kw": 4.216,
        "global_reactive_power_kw": 0.418,
        "voltage_v": 234.84,
        "global_intensity_a": 18.4,
        "sub_metering_1_wh": 0.0,
        "sub_metering_2_wh": 1.0,
        "sub_metering_3_wh": 17.0,
    }


def make_readings(n: int) -> list[dict]:
    return [make_reading(i) for i in range(n)]


# ── get_db_connection ─────────────────────────────────────────────────────────


class TestGetDbConnection:
    def test_raises_503_on_bad_path(self):
        from fastapi import HTTPException

        with (
            pytest.raises(HTTPException) as exc,
            patch("src.serve.sqlite3.connect", side_effect=sqlite3.Error("fail")),
        ):
            get_db_connection("/nonexistent/path/db.db")
        assert exc.value.status_code == 503


# ── readings_to_dataframe ─────────────────────────────────────────────────────


class TestReadingsToDataframeExtra:
    def test_index_is_sorted(self):
        readings = [
            ReadingInput(**make_reading(2)),
            ReadingInput(**make_reading(0)),
            ReadingInput(**make_reading(1)),
        ]
        df = readings_to_dataframe(readings)
        assert df.index.is_monotonic_increasing

    def test_correct_columns(self):
        readings = [ReadingInput(**make_reading(0))]
        df = readings_to_dataframe(readings)
        assert "global_active_power_kw" in df.columns
        assert "voltage_v" in df.columns


# ── /anomaly/score — scoring path ─────────────────────────────────────────────


class TestScoreEndpointWithDetector:
    def test_returns_422_when_features_empty_after_engineering(self):
        """Score endpoint returns 422 when feature matrix is empty post-engineering."""
        mock_detector = MagicMock()

        with (
            patch("src.serve.detector", mock_detector),
            patch("src.serve.build_feature_matrix", return_value=pd.DataFrame()),
        ):
            response = client.post(
                "/anomaly/score", json={"readings": make_readings(5)}
            )
        assert response.status_code == 422

    def test_returns_422_when_feature_engineering_raises(self):
        mock_detector = MagicMock()

        with (
            patch("src.serve.detector", mock_detector),
            patch("src.serve.build_feature_matrix", side_effect=ValueError("bad data")),
        ):
            response = client.post(
                "/anomaly/score", json={"readings": make_readings(5)}
            )
        assert response.status_code == 422

    def test_returns_scored_results(self):
        mock_detector = MagicMock()
        ts = pd.Timestamp("2024-01-15 19:00:00")
        scored_df = pd.DataFrame(
            {
                "anomaly_score": [-0.1],
                "is_anomaly": [False],
                "global_active_power_kw": [4.216],
            },
            index=pd.DatetimeIndex([ts]),
        )

        feature_df = pd.DataFrame(
            {"global_active_power_kw": [4.216]},
            index=pd.DatetimeIndex([ts]),
        )

        mock_detector.score_dataframe.return_value = scored_df

        with (
            patch("src.serve.detector", mock_detector),
            patch("src.serve.build_feature_matrix", return_value=feature_df),
        ):
            response = client.post(
                "/anomaly/score", json={"readings": make_readings(5)}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert "anomaly_count" in data
        assert "anomaly_rate" in data


# ── /timeseries — with start/end params ──────────────────────────────────────


class TestTimeSeriesWithStartEnd:
    def _make_db(self, tmp_path: Path) -> str:
        db_path = str(tmp_path / "energy.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                site_id TEXT NOT NULL DEFAULT 'home-01',
                device_id TEXT NOT NULL DEFAULT 'meter-main',
                global_active_power_kw REAL,
                global_reactive_power_kw REAL,
                voltage_v REAL,
                global_intensity_a REAL,
                sub_metering_1_wh REAL,
                sub_metering_2_wh REAL,
                sub_metering_3_wh REAL,
                ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        for i in range(5):
            ts = datetime(2024, 1, 15, 10, i, 0).isoformat()
            conn.execute(
                "INSERT INTO readings (timestamp, site_id, device_id, "
                "global_active_power_kw, global_reactive_power_kw, voltage_v, "
                "global_intensity_a, sub_metering_1_wh, sub_metering_2_wh, "
                "sub_metering_3_wh) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts, "home-01", "meter-main", 4.2, 0.4, 234.0, 18.0, 0.0, 1.0, 17.0),
            )
        conn.commit()
        conn.close()
        return db_path

    def test_returns_data_with_start_end(self, tmp_path):
        db_path = self._make_db(tmp_path)

        with patch(
            "src.serve.get_db_connection",
            return_value=sqlite3.connect(db_path, check_same_thread=False),
        ):
            response = client.get(
                "/timeseries",
                params={
                    "start": "2024-01-15T10:00:00",
                    "end": "2024-01-15T10:10:00",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_points"] == 5

    # def test_returns_data_with_hours_param(self, tmp_path):
    #    db_path = self._make_db(tmp_path)

    #    with patch("src.serve.DEFAULT_DB_PATH", db_path):
    #        response = client.get("/timeseries", params={"hours": 24})

    #    assert response.status_code == 200

    # def test_returns_404_with_no_matching_data(self, tmp_path):
    #    db_path = self._make_db(tmp_path)

    #    with patch("src.serve.DEFAULT_DB_PATH", db_path):
    #        response = client.get(
    #            "/timeseries",
    #            params={
    #                "start": "2020-01-01T00:00:00",
    #                "end": "2020-01-01T01:00:00",
    #            },
    #        )

    #    assert response.status_code == 404


# ── /anomalies endpoint ───────────────────────────────────────────────────────


class TestAnomaliesEndpointWithData:
    def _make_db(self, tmp_path: Path) -> str:
        db_path = str(tmp_path / "energy.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                site_id TEXT NOT NULL DEFAULT 'home-01',
                device_id TEXT NOT NULL DEFAULT 'meter-main',
                global_active_power_kw REAL,
                global_reactive_power_kw REAL,
                voltage_v REAL,
                global_intensity_a REAL,
                sub_metering_1_wh REAL,
                sub_metering_2_wh REAL,
                sub_metering_3_wh REAL,
                ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        for i in range(10):
            ts = datetime(2024, 1, 15, 10, i, 0).isoformat()
            conn.execute(
                "INSERT INTO readings (timestamp, site_id, device_id, "
                "global_active_power_kw, global_reactive_power_kw, voltage_v, "
                "global_intensity_a, sub_metering_1_wh, sub_metering_2_wh, "
                "sub_metering_3_wh) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts, "home-01", "meter-main", 4.2, 0.4, 234.0, 18.0, 0.0, 1.0, 17.0),
            )
        conn.commit()
        conn.close()
        return db_path

    def test_returns_404_when_no_data(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE readings (
            id INTEGER PRIMARY KEY, timestamp TEXT, site_id TEXT,
            device_id TEXT, global_active_power_kw REAL,
            global_reactive_power_kw REAL, voltage_v REAL,
            global_intensity_a REAL, sub_metering_1_wh REAL,
            sub_metering_2_wh REAL, sub_metering_3_wh REAL,
            ingested_at TEXT DEFAULT (datetime('now')))""")
        conn.commit()
        conn.close()

        mock_detector = MagicMock()
        with (
            patch("src.serve.detector", mock_detector),
            patch(
                "src.serve.get_db_connection",
                return_value=sqlite3.connect(db_path, check_same_thread=False),
            ),
        ):
            response = client.get("/anomalies", params={"hours": 24})

        assert response.status_code == 404

    # def test_returns_anomalies_with_mocked_detector(self, tmp_path):
    #    db_path = self._make_db(tmp_path)
    #    ts = pd.Timestamp("2024-01-15 10:00:00")

    #    scored_df = pd.DataFrame(
    #        {
    #            "anomaly_score": [-0.3],
    #            "is_anomaly": [True],
    #            "global_active_power_kw": [4.2],
    #            "voltage_v": [234.0],
    #        },
    #        index=pd.DatetimeIndex([ts]),
    #    )

    #    mock_detector = MagicMock()
    #    mock_detector.score_dataframe.return_value = scored_df
    #    mock_detector.get_anomalies.return_value = scored_df

    #    feature_df = pd.DataFrame(
    #        {"global_active_power_kw": [4.2]}, index=pd.DatetimeIndex([ts])
    #    )

    #    with (
    #        patch("src.serve.detector", mock_detector),
    #        patch("src.serve.DEFAULT_DB_PATH", db_path),
    #        patch("src.serve.build_feature_matrix", return_value=feature_df),
    #    ):
    #        response = client.get("/anomalies", params={"hours": 24})

    #    assert response.status_code == 200
    #    data = response.json()
    #    assert "anomalies" in data
    #    assert "total_found" in data

    def test_returns_anomalies_with_start_end(self, tmp_path):
        db_path = self._make_db(tmp_path)
        ts = pd.Timestamp("2024-01-15 10:00:00")

        scored_df = pd.DataFrame(
            {
                "anomaly_score": [-0.3],
                "is_anomaly": [True],
                "global_active_power_kw": [4.2],
                "voltage_v": [234.0],
            },
            index=pd.DatetimeIndex([ts]),
        )

        mock_detector = MagicMock()
        mock_detector.score_dataframe.return_value = scored_df
        mock_detector.get_anomalies.return_value = scored_df

        feature_df = pd.DataFrame(
            {"global_active_power_kw": [4.2]}, index=pd.DatetimeIndex([ts])
        )

        with (
            patch("src.serve.detector", mock_detector),
            patch(
                "src.serve.get_db_connection",
                return_value=sqlite3.connect(db_path, check_same_thread=False),
            ),
            patch("src.serve.build_feature_matrix", return_value=feature_df),
        ):
            response = client.get(
                "/anomalies",
                params={"start": "2024-01-15T10:00:00", "end": "2024-01-15T10:10:00"},
            )

        assert response.status_code == 200
