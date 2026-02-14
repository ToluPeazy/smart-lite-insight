"""Tests for seed/replayer.py"""

from datetime import datetime

import pytest

from seed.replayer import (
    generate_dataset,
    generate_reading,
    get_load_multiplier,
    replay_bulk,
)


class TestLoadProfile:
    """Tests for daily load pattern generation."""

    def test_night_lower_than_evening(self):
        """Power should be lower at 3am than at 7pm."""
        night = datetime(2024, 1, 15, 3, 0)  # Tuesday 3am
        evening = datetime(2024, 1, 15, 19, 0)  # Tuesday 7pm
        assert get_load_multiplier(night) < get_load_multiplier(evening)

    def test_weekend_morning_lower_than_weekday(self):
        """Weekend mornings start slower than weekday mornings."""
        weekday_7am = datetime(2024, 1, 15, 7, 0)  # Tuesday
        weekend_7am = datetime(2024, 1, 13, 7, 0)  # Saturday
        assert get_load_multiplier(weekend_7am) < get_load_multiplier(weekday_7am)

    def test_multiplier_always_positive(self):
        """Multiplier should never be negative."""
        dt = datetime(2024, 1, 15, 0, 0)
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                t = dt.replace(hour=hour, minute=minute)
                assert get_load_multiplier(t) >= 0


class TestGenerateReading:
    """Tests for single reading generation."""

    def test_returns_expected_structure(self):
        dt = datetime(2024, 1, 15, 12, 0)
        reading = generate_reading(dt, anomaly_rate=0.0)

        assert "timestamp" in reading
        assert "metrics" in reading
        assert reading["timestamp"] == dt

        metrics = reading["metrics"]
        expected_keys = {
            "global_active_power",
            "global_reactive_power",
            "voltage",
            "global_intensity",
            "sub_metering_1",
            "sub_metering_2",
            "sub_metering_3",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_are_non_negative(self):
        dt = datetime(2024, 1, 15, 12, 0)
        for _ in range(100):
            reading = generate_reading(dt, anomaly_rate=0.0)
            for key, value in reading["metrics"].items():
                assert value >= 0, f"{key} was negative: {value}"

    def test_voltage_reasonable_range(self):
        dt = datetime(2024, 1, 15, 12, 0)
        voltages = []
        for _ in range(200):
            reading = generate_reading(dt, anomaly_rate=0.0)
            voltages.append(reading["metrics"]["voltage"])

        avg = sum(voltages) / len(voltages)
        assert 220 < avg < 240, f"Average voltage {avg} outside expected range"


class TestGenerateDataset:
    """Tests for full dataset generation."""

    def test_correct_row_count(self):
        readings = generate_dataset(days=1, seed=42)
        assert len(readings) == 1440  # 24 hours * 60 minutes

    def test_seven_days(self):
        readings = generate_dataset(days=7, seed=42)
        assert len(readings) == 7 * 1440

    def test_reproducible_with_seed(self):
        r1 = generate_dataset(days=1, seed=42)
        r2 = generate_dataset(days=1, seed=42)
        assert r1[0]["metrics"]["global_active_power"] == r2[0]["metrics"]["global_active_power"]
        assert r1[-1]["metrics"]["voltage"] == r2[-1]["metrics"]["voltage"]

    def test_different_seeds_differ(self):
        r1 = generate_dataset(days=1, seed=42)
        r2 = generate_dataset(days=1, seed=99)
        # Very unlikely to be identical
        powers1 = [r["metrics"]["global_active_power"] for r in r1[:10]]
        powers2 = [r["metrics"]["global_active_power"] for r in r2[:10]]
        assert powers1 != powers2

    def test_timestamps_are_sequential(self):
        readings = generate_dataset(days=1, seed=42)
        for i in range(1, len(readings)):
            diff = readings[i]["timestamp"] - readings[i - 1]["timestamp"]
            assert diff.total_seconds() == 60


class TestReplayBulk:
    """Tests for bulk insertion of synthetic data."""

    def test_inserts_into_db(self, tmp_db):
        import sqlite3

        stats = replay_bulk(days=1, db_path=tmp_db, seed=42)

        assert stats["readings"] == 1440
        assert stats["days"] == 1

        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM readings")
        assert cursor.fetchone()[0] == 1440

        # Check site/device IDs
        cursor = conn.execute("SELECT DISTINCT site_id FROM readings")
        assert cursor.fetchone()[0] == "home-synth"
        conn.close()
