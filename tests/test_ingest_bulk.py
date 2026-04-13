"""Tests for bulk_load and live_ingest in src/ingest.py."""

import sqlite3
from pathlib import Path

import pytest

from src.ingest import (
    bulk_load,
    live_ingest,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

UCI_HEADER = "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"

VALID_ROW = "16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000\n"
MISSING_ROW = "16/12/2006;17:25:00;?;?;?;?;?;?;?\n"
INVALID_ROW = "16/12/2006;17:26:00;NOT_A_NUMBER;0.418;234.840;18.400;0.0;1.0;17.0\n"


def make_uci_file(tmp_path: Path, rows: list[str]) -> Path:
    """Write a minimal UCI-format file and return its path."""
    raw = tmp_path / "household_power_consumption.txt"
    raw.write_text(UCI_HEADER + "".join(rows), encoding="utf-8")
    return raw


# ── bulk_load ────────────────────────────────────────────────────────────────


class TestBulkLoad:
    def test_inserts_valid_rows(self, tmp_path):
        raw = make_uci_file(tmp_path, [VALID_ROW] * 3)
        db = str(tmp_path / "energy.db")

        stats = bulk_load(raw_path=str(raw), db_path=db)

        assert stats["inserted"] == 3
        assert stats["total_lines"] == 3
        assert stats["skipped_missing"] == 0
        assert stats["skipped_invalid"] == 0

    def test_skips_missing_rows(self, tmp_path):
        raw = make_uci_file(tmp_path, [VALID_ROW, MISSING_ROW, MISSING_ROW])
        db = str(tmp_path / "energy.db")

        stats = bulk_load(raw_path=str(raw), db_path=db)

        assert stats["inserted"] == 1
        assert stats["skipped_missing"] == 2

    def test_skips_invalid_rows(self, tmp_path):
        raw = make_uci_file(tmp_path, [VALID_ROW, INVALID_ROW])
        db = str(tmp_path / "energy.db")

        stats = bulk_load(raw_path=str(raw), db_path=db)

        assert stats["inserted"] == 1
        assert stats["skipped_invalid"] == 1

    def test_returns_stats_dict(self, tmp_path):
        raw = make_uci_file(tmp_path, [VALID_ROW])
        db = str(tmp_path / "energy.db")

        stats = bulk_load(raw_path=str(raw), db_path=db)

        assert set(stats.keys()) == {
            "total_lines",
            "inserted",
            "skipped_missing",
            "skipped_invalid",
            "warnings",
        }

    def test_rows_stored_in_db(self, tmp_path):
        raw = make_uci_file(tmp_path, [VALID_ROW] * 5)
        db = str(tmp_path / "energy.db")

        bulk_load(raw_path=str(raw), db_path=db)

        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        conn.close()
        assert count == 5

    def test_site_and_device_id_stored(self, tmp_path):
        raw = make_uci_file(tmp_path, [VALID_ROW])
        db = str(tmp_path / "energy.db")

        bulk_load(
            raw_path=str(raw),
            db_path=db,
            site_id="test-site",
            device_id="test-device",
        )

        conn = sqlite3.connect(db)
        row = conn.execute("SELECT site_id, device_id FROM readings LIMIT 1").fetchone()
        conn.close()
        assert row == ("test-site", "test-device")

    def test_exits_when_file_missing(self, tmp_path):
        db = str(tmp_path / "energy.db")

        with pytest.raises(SystemExit):
            bulk_load(raw_path=str(tmp_path / "nonexistent.txt"), db_path=db)

    def test_bulk_load_exceeds_batch_size(self, tmp_path):
        """Ensure batch flushing works correctly for > BATCH_SIZE rows."""
        raw = make_uci_file(tmp_path, [VALID_ROW] * 6001)
        db = str(tmp_path / "energy.db")

        stats = bulk_load(raw_path=str(raw), db_path=db)

        assert stats["inserted"] == 6001

    def test_empty_file(self, tmp_path):
        raw = make_uci_file(tmp_path, [])
        db = str(tmp_path / "energy.db")

        stats = bulk_load(raw_path=str(raw), db_path=db)

        assert stats["inserted"] == 0
        assert stats["total_lines"] == 0


# ── live_ingest ───────────────────────────────────────────────────────────────


class TestLiveIngest:
    def test_inserts_up_to_max_rows(self, tmp_path):
        raw = make_uci_file(tmp_path, [VALID_ROW] * 10)
        db = str(tmp_path / "energy.db")

        # interval_sec=0 so it doesn't actually sleep in CI
        live_ingest(
            raw_path=str(raw),
            db_path=db,
            interval_sec=0,
            max_rows=3,
        )

        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        conn.close()
        assert count == 3

    def test_skips_missing_and_invalid_rows(self, tmp_path):
        rows = [MISSING_ROW, INVALID_ROW, VALID_ROW, VALID_ROW]
        raw = make_uci_file(tmp_path, rows)
        db = str(tmp_path / "energy.db")

        live_ingest(raw_path=str(raw), db_path=db, interval_sec=0, max_rows=5)

        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        conn.close()
        assert count == 2

    def test_exits_when_file_missing(self, tmp_path):
        db = str(tmp_path / "energy.db")

        with pytest.raises(SystemExit):
            live_ingest(
                raw_path=str(tmp_path / "nonexistent.txt"),
                db_path=db,
                interval_sec=0,
            )
