"""Tests for src/ingest.py"""

import sqlite3

from src.ingest import init_db, insert_batch, insert_reading
from src.validate import parse_uci_row


class TestInitDb:
    """Tests for database initialisation."""

    def test_creates_database_file(self, tmp_db):
        conn = init_db(tmp_db)
        assert conn is not None
        conn.close()

    def test_creates_readings_table(self, tmp_db):
        conn = init_db(tmp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='readings'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_creates_indices(self, tmp_db):
        conn = init_db(tmp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        index_names = [row[0] for row in cursor.fetchall()]
        assert "idx_readings_timestamp" in index_names
        assert "idx_readings_site_device" in index_names
        conn.close()

    def test_idempotent(self, tmp_db):
        """Calling init_db twice should not raise."""
        conn1 = init_db(tmp_db)
        conn1.close()
        conn2 = init_db(tmp_db)
        conn2.close()


class TestInsertReading:
    """Tests for single row insertion."""

    def test_insert_and_retrieve(self, tmp_db, sample_raw_row):
        conn = init_db(tmp_db)
        parsed = parse_uci_row(sample_raw_row)

        insert_reading(conn, parsed, "home-01", "meter-main")
        conn.commit()

        cursor = conn.execute("SELECT COUNT(*) FROM readings")
        assert cursor.fetchone()[0] == 1

        cursor = conn.execute(
            "SELECT global_active_power_kw, voltage_v, site_id FROM readings"
        )
        row = cursor.fetchone()
        assert abs(row[0] - 4.216) < 0.001
        assert abs(row[1] - 234.84) < 0.01
        assert row[2] == "home-01"

        conn.close()


class TestInsertBatch:
    """Tests for batch insertion."""

    def test_batch_insert_multiple_rows(self, tmp_db):
        conn = init_db(tmp_db)

        lines = [
            "16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000",
            "16/12/2006;17:25:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000",
            "16/12/2006;17:26:00;5.374;0.498;233.290;23.000;0.000;2.000;17.000",
        ]
        parsed_rows = [parse_uci_row(line) for line in lines]

        insert_batch(conn, parsed_rows, "home-01", "meter-main")

        cursor = conn.execute("SELECT COUNT(*) FROM readings")
        assert cursor.fetchone()[0] == 3

        # Check ordering is preserved
        cursor = conn.execute(
            "SELECT global_active_power_kw FROM readings ORDER BY id"
        )
        powers = [row[0] for row in cursor.fetchall()]
        assert abs(powers[0] - 4.216) < 0.001
        assert abs(powers[1] - 5.360) < 0.001
        assert abs(powers[2] - 5.374) < 0.001

        conn.close()

    def test_empty_batch(self, tmp_db):
        """Inserting an empty batch should not raise."""
        conn = init_db(tmp_db)
        insert_batch(conn, [], "home-01", "meter-main")
        cursor = conn.execute("SELECT COUNT(*) FROM readings")
        assert cursor.fetchone()[0] == 0
        conn.close()
