"""Base-load estimator for Smart-Lite Insight.

Produces a half-hourly inflexible base-load profile B_t from historical
readings, in the units an external MILP-style scheduler (e.g. EcoHome)
expects: mean power (kW) and energy (kWh) per half-hourly slot, aligned
to the Agile settlement period.

The estimator excludes the flexible appliances a downstream scheduler
would control (washing machine, dishwasher) so the base load and the
scheduled flexible load don't double-count energy.

Usage:
    from src.baseload import compute_base_load

    profile = compute_base_load(start="2007-01-01", end="2007-12-31")

    # Or via CLI:
    python -m src.baseload --start 2007-01-01 --end 2007-12-31
"""

import argparse
import json
import sqlite3

import pandas as pd
from loguru import logger

from src.train import DEFAULT_DB_PATH

SCHEMA_VERSION = "baseload-1.0"
N_SLOTS = 48
SLOT_MINUTES = 30
DEFAULT_QUANTILE = 0.25
METHODS = ("submeter_subtraction", "quantile")

# Sub-metering columns are Wh consumed over a 1-minute reading interval.
# Average power over that minute: W = Wh * 60. kW = Wh * 60 / 1000.
WH_PER_MIN_TO_KW = 60 / 1000


def _load_readings(
    db_path: str,
    site_id: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Load raw readings from SQLite for the given site and date range."""
    conn = sqlite3.connect(db_path)
    try:
        if start and end:
            query = """
                SELECT timestamp, global_active_power_kw,
                       sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
                FROM readings
                WHERE site_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            params = (site_id, start, end)
        else:
            query = """
                SELECT timestamp, global_active_power_kw,
                       sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
                FROM readings
                WHERE site_id = ?
                ORDER BY timestamp
            """
            params = (site_id,)
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp"])
    finally:
        conn.close()

    if df.empty:
        raise ValueError(
            f"No readings found for site_id='{site_id}' in the given range"
        )

    return df.set_index("timestamp").sort_index()


def _slot_index(index: pd.DatetimeIndex) -> pd.Index:
    """Map each timestamp to its half-hourly slot number (0-47) within the day."""
    return (index.hour * 60 + index.minute) // SLOT_MINUTES


def _submeter_subtraction_base_load(df: pd.DataFrame) -> pd.Series:
    """Base load = total active power minus the flexible-appliance channels.

    Subtracts sub_metering_1 (kitchen/dishwasher channel) and sub_metering_2
    (laundry channel) from global active power, leaving sub_metering_3
    (water heater/AC) plus unmetered load as the always-on baseline.

    UCI's sub-metering channels bundle appliances together (sub_metering_2
    also covers the fridge and lighting alongside the washing machine and
    dryer, and sub_metering_1 bundles the dishwasher with the oven and
    microwave), so this subtracts the whole flexible-appliance channel
    rather than isolating a single appliance. That's a stated assumption,
    not a precise per-appliance split — surfacing it here so a downstream
    consumer of this profile can flag disagreement early.
    """
    flexible_kw = (df["sub_metering_1_wh"] + df["sub_metering_2_wh"]) * WH_PER_MIN_TO_KW
    base_kw = df["global_active_power_kw"] - flexible_kw
    return base_kw.clip(lower=0)


def _quantile_base_load(
    df: pd.DataFrame, slots: pd.Index, quantile: float
) -> pd.Series:
    """Fallback: per-slot low quantile of total active power.

    Used when sub-metering channels aren't available, e.g. on hardware
    that only exposes aggregate power.
    """
    return df["global_active_power_kw"].groupby(slots).quantile(quantile)


def compute_base_load(
    db_path: str = DEFAULT_DB_PATH,
    site_id: str = "home-01",
    start: str | None = None,
    end: str | None = None,
    method: str = "submeter_subtraction",
    quantile: float = DEFAULT_QUANTILE,
) -> dict:
    """Compute a 48-slot half-hourly base-load profile.

    Args:
        db_path: Path to the SQLite database.
        site_id: Site to compute the profile for.
        start: Inclusive start timestamp/date (ISO 8601). None = earliest available.
        end: Inclusive end timestamp/date (ISO 8601). None = latest available.
        method: "submeter_subtraction" (default) or "quantile".
        quantile: Quantile used by the "quantile" method (default p25).

    Returns:
        A dict matching the baseload-1.0 schema (see docs/schemas/baseload_v1.json):
        schema_version, site_id, method, source_range, and 48 slots each
        carrying mean power (kW) and energy (kWh) for that half hour.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Must be one of {METHODS}")

    df = _load_readings(db_path, site_id, start, end)
    slots = _slot_index(df.index)

    if method == "submeter_subtraction":
        base_kw_per_reading = _submeter_subtraction_base_load(df)
        base_kw_by_slot = base_kw_per_reading.groupby(slots).mean()
    else:
        base_kw_by_slot = _quantile_base_load(df, slots, quantile)

    base_kw_by_slot = base_kw_by_slot.reindex(range(N_SLOTS))
    if base_kw_by_slot.isna().any():
        missing = base_kw_by_slot[base_kw_by_slot.isna()].index.tolist()
        raise ValueError(
            f"No data for slot(s) {missing} in the given range; "
            "cannot build a full 48-slot profile"
        )

    slots_out = []
    for slot in range(N_SLOTS):
        hour, minute = divmod(slot * SLOT_MINUTES, 60)
        base_kw = round(float(base_kw_by_slot.loc[slot]), 4)
        slots_out.append(
            {
                "slot": slot,
                "start": f"{hour:02d}:{minute:02d}",
                "base_load_kw": base_kw,
                "base_load_kwh": round(base_kw * 0.5, 4),
            }
        )

    profile = {
        "schema_version": SCHEMA_VERSION,
        "site_id": site_id,
        "method": method,
        "source_range": {
            "start": start or df.index.min().date().isoformat(),
            "end": end or df.index.max().date().isoformat(),
        },
        "slots": slots_out,
    }

    logger.info(
        f"Base-load profile computed: site='{site_id}' method='{method}' "
        f"range={profile['source_range']} "
        f"mean_kw={sum(s['base_load_kw'] for s in slots_out) / N_SLOTS:.3f}"
    )

    return profile


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Compute a base-load profile.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite DB path")
    parser.add_argument("--site-id", default="home-01", help="Site identifier")
    parser.add_argument("--start", default=None, help="Start date/timestamp (ISO 8601)")
    parser.add_argument("--end", default=None, help="End date/timestamp (ISO 8601)")
    parser.add_argument(
        "--method",
        default="submeter_subtraction",
        choices=METHODS,
        help="Base-load estimation method",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=DEFAULT_QUANTILE,
        help="Quantile used by the 'quantile' method",
    )
    parser.add_argument(
        "--output", default=None, help="Write JSON to this file instead of stdout"
    )
    args = parser.parse_args()

    profile = compute_base_load(
        db_path=args.db_path,
        site_id=args.site_id,
        start=args.start,
        end=args.end,
        method=args.method,
        quantile=args.quantile,
    )

    output = json.dumps(profile, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        logger.info(f"Wrote base-load profile to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
