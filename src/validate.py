"""Schema validation for energy telemetry readings.

Validates raw UCI dataset rows and parsed telemetry dicts against
the versioned schema defined in docs/schemas/telemetry_v1.json.
"""

from datetime import datetime

# Column names matching the UCI dataset (semicolon-separated)
UCI_COLUMNS = [
    "date",
    "time",
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]

# Expected number of fields in a raw UCI row
UCI_FIELD_COUNT = 9

# Reasonable physical bounds for validation
BOUNDS = {
    "global_active_power": (0.0, 15.0),    # kW — typical household max ~11kW
    "global_reactive_power": (0.0, 5.0),    # kW
    "voltage": (100.0, 300.0),              # V — EU nominal 230V ± wide margin
    "global_intensity": (0.0, 80.0),        # A
    "sub_metering_1": (0.0, 100.0),         # Wh per minute
    "sub_metering_2": (0.0, 100.0),         # Wh per minute
    "sub_metering_3": (0.0, 100.0),         # Wh per minute
}


class ValidationError(Exception):
    """Raised when a telemetry reading fails validation."""

    pass


def parse_uci_row(raw_line: str) -> dict | None:
    """Parse a single semicolon-separated UCI dataset row.

    Args:
        raw_line: A raw line from household_power_consumption.txt

    Returns:
        Parsed dict with typed values, or None if the row has missing data ('?').
    """
    parts = raw_line.strip().split(";")

    if len(parts) != UCI_FIELD_COUNT:
        raise ValidationError(
            f"Expected {UCI_FIELD_COUNT} fields, got {len(parts)}"
        )

    # Check for missing values (represented as '?' in the UCI dataset)
    if "?" in parts:
        return None

    date_str, time_str = parts[0], parts[1]

    try:
        timestamp = datetime.strptime(
            f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S"
        )
    except ValueError as e:
        raise ValidationError(f"Invalid date/time: {date_str} {time_str}") from e

    try:
        metrics = {
            "global_active_power": float(parts[2]),
            "global_reactive_power": float(parts[3]),
            "voltage": float(parts[4]),
            "global_intensity": float(parts[5]),
            "sub_metering_1": float(parts[6]),
            "sub_metering_2": float(parts[7]),
            "sub_metering_3": float(parts[8]),
        }
    except ValueError as e:
        raise ValidationError(f"Non-numeric metric value: {e}") from e

    return {
        "timestamp": timestamp,
        "metrics": metrics,
    }


def validate_metrics(metrics: dict) -> list[str]:
    """Check metric values against physical bounds.

    Args:
        metrics: Dict of metric_name -> float value.

    Returns:
        List of warning strings (empty if all values are within bounds).
    """
    warnings = []
    for field, (low, high) in BOUNDS.items():
        value = metrics.get(field)
        if value is not None and not (low <= value <= high):
            warnings.append(
                f"{field}={value} out of bounds [{low}, {high}]"
            )
    return warnings


def validate_row(raw_line: str) -> tuple[dict | None, list[str]]:
    """Parse and validate a single UCI row.

    Args:
        raw_line: Raw semicolon-separated line.

    Returns:
        Tuple of (parsed_dict_or_None, list_of_warnings).
        Returns (None, []) for rows with missing data.

    Raises:
        ValidationError: If the row is structurally invalid.
    """
    parsed = parse_uci_row(raw_line)
    if parsed is None:
        return None, []

    warnings = validate_metrics(parsed["metrics"])
    return parsed, warnings
