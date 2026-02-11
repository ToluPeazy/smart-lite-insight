"""Tests for src/validate.py"""

import pytest

from src.validate import (
    ValidationError,
    parse_uci_row,
    validate_metrics,
    validate_row,
)


class TestParseUciRow:
    """Tests for parsing raw UCI dataset lines."""

    def test_valid_row(self, sample_raw_row):
        result = parse_uci_row(sample_raw_row)
        assert result is not None
        assert result["timestamp"].year == 2006
        assert result["timestamp"].month == 12
        assert result["timestamp"].day == 16
        assert result["timestamp"].hour == 17
        assert result["timestamp"].minute == 24
        assert result["metrics"]["global_active_power"] == pytest.approx(4.216)
        assert result["metrics"]["voltage"] == pytest.approx(234.84)

    def test_missing_values_returns_none(self):
        line = "16/12/2006;17:24:00;?;0.418;234.840;18.400;0.000;1.000;17.000"
        assert parse_uci_row(line) is None

    def test_all_missing_returns_none(self):
        line = "16/12/2006;17:25:00;?;?;?;?;?;?;?"
        assert parse_uci_row(line) is None

    def test_wrong_field_count_raises(self):
        line = "16/12/2006;17:24:00;4.216;0.418"
        with pytest.raises(ValidationError, match="Expected 9 fields"):
            parse_uci_row(line)

    def test_invalid_date_raises(self):
        line = "99/99/9999;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000"
        with pytest.raises(ValidationError, match="Invalid date/time"):
            parse_uci_row(line)

    def test_non_numeric_metric_raises(self):
        line = "16/12/2006;17:24:00;abc;0.418;234.840;18.400;0.000;1.000;17.000"
        with pytest.raises(ValidationError, match="Non-numeric"):
            parse_uci_row(line)

    def test_strips_whitespace(self):
        line = "16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000\n"
        result = parse_uci_row(line)
        assert result is not None

    def test_zero_values_are_valid(self):
        line = "16/12/2006;17:24:00;0.000;0.000;234.840;0.000;0.000;0.000;0.000"
        result = parse_uci_row(line)
        assert result is not None
        assert result["metrics"]["global_active_power"] == 0.0


class TestValidateMetrics:
    """Tests for physical bounds validation."""

    def test_valid_metrics_no_warnings(self):
        metrics = {
            "global_active_power": 4.0,
            "global_reactive_power": 0.4,
            "voltage": 234.0,
            "global_intensity": 18.0,
            "sub_metering_1": 0.0,
            "sub_metering_2": 1.0,
            "sub_metering_3": 17.0,
        }
        assert validate_metrics(metrics) == []

    def test_out_of_bounds_voltage_warns(self):
        metrics = {
            "global_active_power": 4.0,
            "voltage": 500.0,  # Way too high
        }
        warnings = validate_metrics(metrics)
        assert len(warnings) == 1
        assert "voltage" in warnings[0]

    def test_negative_power_warns(self):
        metrics = {"global_active_power": -1.0}
        warnings = validate_metrics(metrics)
        assert len(warnings) == 1
        assert "global_active_power" in warnings[0]

    def test_boundary_values_pass(self):
        metrics = {
            "global_active_power": 0.0,
            "voltage": 100.0,
        }
        assert validate_metrics(metrics) == []


class TestValidateRow:
    """Tests for the combined parse + validate function."""

    def test_valid_row_returns_parsed_and_no_warnings(self, sample_raw_row):
        parsed, warnings = validate_row(sample_raw_row)
        assert parsed is not None
        assert warnings == []

    def test_missing_row_returns_none(self):
        line = "16/12/2006;17:24:00;?;0.418;234.840;18.400;0.000;1.000;17.000"
        parsed, warnings = validate_row(line)
        assert parsed is None
        assert warnings == []
