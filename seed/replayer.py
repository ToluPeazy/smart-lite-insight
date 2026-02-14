"""Synthetic energy data replayer for Smart-Lite Insight.

Generates realistic household energy consumption data and inserts it
into SQLite, so anyone can clone the repo and demo the project without
needing the real UCI dataset.

The generated data models:
- Daily consumption patterns (low at night, peaks at morning/evening)
- Weekday vs weekend differences
- Random noise and natural variation
- Occasional anomalies (spikes, sustained high usage)

Usage:
    # Generate 7 days of data (default)
    python -m seed.replayer

    # Custom duration and options
    python -m seed.replayer --days 14 --anomaly-rate 0.02

    # Drip-feed mode (simulate real-time arrival)
    python -m seed.replayer --live --interval 0.5
"""

import argparse
import math
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from src.ingest import DEFAULT_DB_PATH, init_db, insert_batch, insert_reading

# ── Configuration ──

DEFAULT_DAYS = 7
DEFAULT_SITE_ID = "home-synth"
DEFAULT_DEVICE_ID = "meter-synth"
DEFAULT_ANOMALY_RATE = 0.01  # 1% of readings are anomalous


# ── Daily Load Profile ──
# Hourly multipliers modelling a typical household pattern.
# Index = hour of day (0-23). Values are relative power multipliers.
WEEKDAY_PROFILE = [
    0.15, 0.12, 0.10, 0.10, 0.10, 0.12,  # 00-05: overnight baseline
    0.25, 0.55, 0.70, 0.50, 0.35, 0.30,  # 06-11: morning peak
    0.35, 0.40, 0.35, 0.30, 0.35, 0.55,  # 12-17: afternoon
    0.80, 0.90, 0.75, 0.55, 0.35, 0.20,  # 18-23: evening peak
]

WEEKEND_PROFILE = [
    0.15, 0.12, 0.10, 0.10, 0.10, 0.10,  # 00-05: overnight
    0.15, 0.20, 0.35, 0.55, 0.60, 0.55,  # 06-11: later morning rise
    0.50, 0.45, 0.50, 0.55, 0.60, 0.65,  # 12-17: more active afternoon
    0.80, 0.85, 0.70, 0.50, 0.35, 0.20,  # 18-23: evening peak
]


def get_load_multiplier(dt: datetime) -> float:
    """Get the load profile multiplier for a given timestamp.

    Interpolates between hourly values for smooth transitions and
    selects weekday vs weekend profile.

    Args:
        dt: Timestamp to get multiplier for.

    Returns:
        Float multiplier (roughly 0.1 to 0.9).
    """
    profile = WEEKEND_PROFILE if dt.weekday() >= 5 else WEEKDAY_PROFILE

    hour = dt.hour
    minute_frac = dt.minute / 60.0
    next_hour = (hour + 1) % 24

    # Linear interpolation between hours
    return profile[hour] + (profile[next_hour] - profile[hour]) * minute_frac


def generate_reading(dt: datetime, anomaly_rate: float = DEFAULT_ANOMALY_RATE) -> dict:
    """Generate a single synthetic energy reading.

    Args:
        dt: Timestamp for this reading.
        anomaly_rate: Probability of generating an anomalous reading.

    Returns:
        Dict matching the format expected by insert_reading().
    """
    multiplier = get_load_multiplier(dt)
    is_anomaly = random.random() < anomaly_rate

    if is_anomaly:
        # Anomaly: sudden spike (2-5x normal) or near-zero dropout
        if random.random() < 0.7:
            # Spike
            anomaly_factor = random.uniform(2.0, 5.0)
            multiplier *= anomaly_factor
        else:
            # Dropout
            multiplier *= random.uniform(0.0, 0.05)

    # Base active power: typical EU household 0.3-8 kW
    base_power = 4.5
    noise = random.gauss(0, 0.15)
    global_active_power = max(0.0, base_power * multiplier + noise)

    # Reactive power: typically 5-15% of active power
    reactive_ratio = random.uniform(0.05, 0.15)
    global_reactive_power = global_active_power * reactive_ratio

    # Voltage: EU nominal 230V with small fluctuations
    voltage = random.gauss(230.0, 3.0)

    # Current intensity: P = V * I (approximately)
    global_intensity = (global_active_power * 1000) / voltage if voltage > 0 else 0.0

    # Sub-meterings: divide active power among 3 sub-meters + remainder
    total_sub = global_active_power * 1000 / 60  # Convert kW to Wh/minute
    sub1_share = random.uniform(0.05, 0.25)  # Kitchen
    sub2_share = random.uniform(0.05, 0.20)  # Laundry
    sub3_share = random.uniform(0.10, 0.35)  # Water heater + AC

    sub_metering_1 = max(0.0, total_sub * sub1_share + random.gauss(0, 0.5))
    sub_metering_2 = max(0.0, total_sub * sub2_share + random.gauss(0, 0.5))
    sub_metering_3 = max(0.0, total_sub * sub3_share + random.gauss(0, 0.5))

    return {
        "timestamp": dt,
        "metrics": {
            "global_active_power": round(global_active_power, 3),
            "global_reactive_power": round(global_reactive_power, 3),
            "voltage": round(voltage, 2),
            "global_intensity": round(global_intensity, 1),
            "sub_metering_1": round(sub_metering_1, 1),
            "sub_metering_2": round(sub_metering_2, 1),
            "sub_metering_3": round(sub_metering_3, 1),
        },
    }


def generate_dataset(
    days: int = DEFAULT_DAYS,
    start_date: datetime | None = None,
    anomaly_rate: float = DEFAULT_ANOMALY_RATE,
    seed: int | None = 42,
) -> list[dict]:
    """Generate a full synthetic dataset.

    Args:
        days: Number of days to generate.
        start_date: Starting timestamp (defaults to `days` ago from now).
        anomaly_rate: Probability of anomalous readings.
        seed: Random seed for reproducibility (None for random).

    Returns:
        List of parsed reading dicts (1-minute intervals).
    """
    if seed is not None:
        random.seed(seed)

    if start_date is None:
        start_date = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=days)

    total_minutes = days * 24 * 60
    readings = []

    for i in range(total_minutes):
        dt = start_date + timedelta(minutes=i)
        reading = generate_reading(dt, anomaly_rate)
        readings.append(reading)

    return readings


def replay_bulk(
    days: int = DEFAULT_DAYS,
    db_path: str = DEFAULT_DB_PATH,
    site_id: str = DEFAULT_SITE_ID,
    device_id: str = DEFAULT_DEVICE_ID,
    anomaly_rate: float = DEFAULT_ANOMALY_RATE,
    seed: int | None = 42,
) -> dict:
    """Generate and bulk-insert synthetic data.

    Args:
        days: Number of days to generate.
        db_path: Path to SQLite database.
        site_id: Site identifier for synthetic readings.
        device_id: Device identifier for synthetic readings.
        anomaly_rate: Probability of anomalous readings.
        seed: Random seed for reproducibility.

    Returns:
        Dict with generation statistics.
    """
    conn = init_db(db_path)

    logger.info(f"Generating {days} days of synthetic data (seed={seed})")
    readings = generate_dataset(days=days, anomaly_rate=anomaly_rate, seed=seed)

    logger.info(f"Inserting {len(readings):,} readings into {db_path}")

    # Insert in batches
    batch_size = 5000
    for i in range(0, len(readings), batch_size):
        batch = readings[i : i + batch_size]
        insert_batch(conn, batch, site_id, device_id)

    conn.close()

    expected_anomalies = int(len(readings) * anomaly_rate)
    logger.info("─" * 50)
    logger.info(f"Synthetic replay complete")
    logger.info(f"  Days:               {days}")
    logger.info(f"  Readings generated: {len(readings):,}")
    logger.info(f"  Expected anomalies: ~{expected_anomalies}")
    logger.info(f"  Site/Device:        {site_id}/{device_id}")
    logger.info(f"  Database:           {db_path}")
    logger.info("─" * 50)

    return {
        "days": days,
        "readings": len(readings),
        "expected_anomalies": expected_anomalies,
    }


def replay_live(
    days: int = DEFAULT_DAYS,
    db_path: str = DEFAULT_DB_PATH,
    interval_sec: float = 1.0,
    site_id: str = DEFAULT_SITE_ID,
    device_id: str = DEFAULT_DEVICE_ID,
    anomaly_rate: float = DEFAULT_ANOMALY_RATE,
    max_rows: int | None = None,
) -> None:
    """Generate and drip-feed synthetic data one row at a time.

    Args:
        days: Number of days of data to generate.
        db_path: Path to SQLite database.
        interval_sec: Seconds between each insertion.
        site_id: Site identifier.
        device_id: Device identifier.
        anomaly_rate: Probability of anomalous readings.
        max_rows: Stop after this many rows (None = run full dataset).
    """
    conn = init_db(db_path)

    logger.info(f"Starting live synthetic replay (interval={interval_sec}s)")
    logger.info("Press Ctrl+C to stop")

    readings = generate_dataset(days=days, anomaly_rate=anomaly_rate)
    inserted = 0

    try:
        for reading in readings:
            insert_reading(conn, reading, site_id, device_id)
            conn.commit()
            inserted += 1

            if inserted % 100 == 0:
                logger.info(
                    f"  Live replay: {inserted:,} / {len(readings):,} "
                    f"({100 * inserted / len(readings):.1f}%)"
                )

            if max_rows and inserted >= max_rows:
                logger.info(f"Reached max_rows limit ({max_rows})")
                break

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        logger.info("Live replay stopped by user")
    finally:
        conn.close()
        logger.info(f"Total live replayed: {inserted:,} rows")


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic energy data for Smart-Lite Insight"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of days to generate (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=DEFAULT_ANOMALY_RATE,
        help=f"Anomaly probability per reading (default: {DEFAULT_ANOMALY_RATE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Drip-feed mode (one row at a time)",
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
    args = parser.parse_args()

    if args.live:
        replay_live(
            days=args.days,
            db_path=args.db_path,
            interval_sec=args.interval,
            anomaly_rate=args.anomaly_rate,
            max_rows=args.max_rows,
        )
    else:
        replay_bulk(
            days=args.days,
            db_path=args.db_path,
            anomaly_rate=args.anomaly_rate,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
