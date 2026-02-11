"""Data ingestion for Smart-Lite Insight.

Loads the UCI Individual Household Electric Power Consumption dataset
into a local SQLite database. Supports both bulk loading (full dataset)
and simulated live ingestion (drip-feed mode via APScheduler).

Usage:
    # Bulk load the entire dataset
    python -m src.ingest

    # Simulated live ingestion (1 row/sec by default)
    python -m src.ingest --live
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.validate import ValidationError, parse_uci_row, validate_metrics

load_dotenv()

# ── Defaults ──
DEFAULT_RAW_PATH = "data/raw/household_power_consumption.txt"
DEFAULT_DB_PATH = "data/processed/energy.db"
DEFAULT_SITE_ID = "home-01"
DEFAULT_DEVICE_ID = "meter-main"
BATCH_SIZE = 5000


# ── Database Setup ──


def init_db(db_path: str) -> sqlite3.Connection:
    """Create the SQLite database and readings table if they don't exist.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        An open sqlite3.Connection.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS readings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            site_id         TEXT    NOT NULL DEFAULT 'home-01',
            device_id       TEXT    NOT NULL DEFAULT 'meter-main',
            global_active_power_kw      REAL,
            global_reactive_power_kw    REAL,
            voltage_v                   REAL,
            global_intensity_a          REAL,
            sub_metering_1_wh           REAL,
            sub_metering_2_wh           REAL,
            sub_metering_3_wh           REAL,
            ingested_at     TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_readings_timestamp
        ON readings (timestamp)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_readings_site_device
        ON readings (site_id, device_id)
        """
    )
    conn.commit()
    return conn


# ── Insertion ──


def insert_reading(conn: sqlite3.Connection, parsed: dict, site_id: str, device_id: str) -> None:
    """Insert a single parsed reading into the database.

    Args:
        conn: Open SQLite connection.
        parsed: Dict from parse_uci_row() with 'timestamp' and 'metrics'.
        site_id: Site identifier.
        device_id: Device identifier.
    """
    m = parsed["metrics"]
    conn.execute(
        """
        INSERT INTO readings (
            timestamp, site_id, device_id,
            global_active_power_kw, global_reactive_power_kw,
            voltage_v, global_intensity_a,
            sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parsed["timestamp"].isoformat(),
            site_id,
            device_id,
            m["global_active_power"],
            m["global_reactive_power"],
            m["voltage"],
            m["global_intensity"],
            m["sub_metering_1"],
            m["sub_metering_2"],
            m["sub_metering_3"],
        ),
    )


def insert_batch(conn: sqlite3.Connection, rows: list[dict], site_id: str, device_id: str) -> None:
    """Insert a batch of parsed readings.

    Args:
        conn: Open SQLite connection.
        rows: List of parsed dicts from parse_uci_row().
        site_id: Site identifier.
        device_id: Device identifier.
    """
    data = []
    for parsed in rows:
        m = parsed["metrics"]
        data.append((
            parsed["timestamp"].isoformat(),
            site_id,
            device_id,
            m["global_active_power"],
            m["global_reactive_power"],
            m["voltage"],
            m["global_intensity"],
            m["sub_metering_1"],
            m["sub_metering_2"],
            m["sub_metering_3"],
        ))

    conn.executemany(
        """
        INSERT INTO readings (
            timestamp, site_id, device_id,
            global_active_power_kw, global_reactive_power_kw,
            voltage_v, global_intensity_a,
            sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        data,
    )
    conn.commit()


# ── Bulk Load ──


def bulk_load(
    raw_path: str = DEFAULT_RAW_PATH,
    db_path: str = DEFAULT_DB_PATH,
    site_id: str = DEFAULT_SITE_ID,
    device_id: str = DEFAULT_DEVICE_ID,
) -> dict:
    """Load the entire UCI dataset into SQLite.

    Args:
        raw_path: Path to household_power_consumption.txt
        db_path: Path to SQLite database.
        site_id: Site identifier for all readings.
        device_id: Device identifier for all readings.

    Returns:
        Dict with load statistics.
    """
    raw = Path(raw_path)
    if not raw.is_file():
        logger.error(f"Dataset not found: {raw_path}")
        logger.info(
            "Download from: https://archive.ics.uci.edu/dataset/235/"
            "individual+household+electric+power+consumption"
        )
        sys.exit(1)

    conn = init_db(db_path)

    stats = {
        "total_lines": 0,
        "inserted": 0,
        "skipped_missing": 0,
        "skipped_invalid": 0,
        "warnings": 0,
    }

    batch = []
    start_time = time.time()

    logger.info(f"Loading dataset from {raw_path} into {db_path}")

    with open(raw, "r", encoding="utf-8") as f:
        # Skip header line
        header = f.readline()
        logger.debug(f"Header: {header.strip()}")

        for line_num, line in enumerate(f, start=2):
            stats["total_lines"] += 1

            try:
                parsed = parse_uci_row(line)
            except ValidationError as e:
                stats["skipped_invalid"] += 1
                if stats["skipped_invalid"] <= 10:
                    logger.warning(f"Line {line_num}: {e}")
                continue

            if parsed is None:
                stats["skipped_missing"] += 1
                continue

            # Validate bounds (log warnings but still insert)
            bound_warnings = validate_metrics(parsed["metrics"])
            if bound_warnings:
                stats["warnings"] += len(bound_warnings)

            batch.append(parsed)

            if len(batch) >= BATCH_SIZE:
                insert_batch(conn, batch, site_id, device_id)
                stats["inserted"] += len(batch)
                batch.clear()

                if stats["inserted"] % 100_000 == 0:
                    elapsed = time.time() - start_time
                    rate = stats["inserted"] / elapsed
                    logger.info(
                        f"  {stats['inserted']:,} rows inserted "
                        f"({rate:,.0f} rows/sec)"
                    )

    # Insert remaining rows
    if batch:
        insert_batch(conn, batch, site_id, device_id)
        stats["inserted"] += len(batch)

    conn.close()
    elapsed = time.time() - start_time

    logger.info("─" * 50)
    logger.info(f"Bulk load complete in {elapsed:.1f}s")
    logger.info(f"  Total lines:     {stats['total_lines']:,}")
    logger.info(f"  Inserted:        {stats['inserted']:,}")
    logger.info(f"  Skipped (missing): {stats['skipped_missing']:,}")
    logger.info(f"  Skipped (invalid): {stats['skipped_invalid']:,}")
    logger.info(f"  Bound warnings:  {stats['warnings']:,}")
    logger.info("─" * 50)

    return stats


# ── Live Simulation ──


def live_ingest(
    raw_path: str = DEFAULT_RAW_PATH,
    db_path: str = DEFAULT_DB_PATH,
    interval_sec: float = 1.0,
    site_id: str = DEFAULT_SITE_ID,
    device_id: str = DEFAULT_DEVICE_ID,
    max_rows: int | None = None,
) -> None:
    """Simulate live data ingestion by drip-feeding rows.

    Reads the UCI dataset line by line and inserts one row at a time
    with a configurable delay, simulating real-time sensor data arrival.

    Args:
        raw_path: Path to household_power_consumption.txt
        db_path: Path to SQLite database.
        interval_sec: Seconds between each row insertion.
        site_id: Site identifier.
        device_id: Device identifier.
        max_rows: Stop after this many rows (None = run until file ends).
    """
    raw = Path(raw_path)
    if not raw.is_file():
        logger.error(f"Dataset not found: {raw_path}")
        sys.exit(1)

    conn = init_db(db_path)
    inserted = 0

    logger.info(f"Starting live ingestion (interval={interval_sec}s)")
    logger.info("Press Ctrl+C to stop")

    try:
        with open(raw, "r", encoding="utf-8") as f:
            f.readline()  # Skip header

            for line in f:
                try:
                    parsed = parse_uci_row(line)
                except ValidationError:
                    continue

                if parsed is None:
                    continue

                insert_reading(conn, parsed, site_id, device_id)
                conn.commit()
                inserted += 1

                if inserted % 100 == 0:
                    logger.info(f"  Live ingested: {inserted} rows")

                if max_rows and inserted >= max_rows:
                    logger.info(f"Reached max_rows limit ({max_rows})")
                    break

                time.sleep(interval_sec)

    except KeyboardInterrupt:
        logger.info("Live ingestion stopped by user")
    finally:
        conn.close()
        logger.info(f"Total live ingested: {inserted} rows")


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(
        description="Ingest UCI energy dataset into SQLite"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Simulate live ingestion (drip-feed mode)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between rows in live mode (default: 1.0)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after N rows in live mode",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=DEFAULT_RAW_PATH,
        help=f"Path to raw dataset (default: {DEFAULT_RAW_PATH})",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    if args.live:
        live_ingest(
            raw_path=args.raw_path,
            db_path=args.db_path,
            interval_sec=args.interval,
            max_rows=args.max_rows,
        )
    else:
        bulk_load(raw_path=args.raw_path, db_path=args.db_path)


if __name__ == "__main__":
    main()
