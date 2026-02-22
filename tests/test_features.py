"""Tests for src/features.py"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_lag_features,
    add_rate_of_change,
    add_rolling_features,
    add_submetering_ratios,
    add_temporal_features,
    build_feature_matrix,
)


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame mimicking raw readings."""
    n = 200  # 200 minutes of data
    idx = pd.date_range("2024-01-15 08:00", periods=n, freq="min")
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "global_active_power_kw": rng.uniform(0.5, 5.0, n),
            "global_reactive_power_kw": rng.uniform(0.05, 0.5, n),
            "voltage_v": rng.normal(230, 3, n),
            "global_intensity_a": rng.uniform(2, 22, n),
            "sub_metering_1_wh": rng.uniform(0, 10, n),
            "sub_metering_2_wh": rng.uniform(0, 8, n),
            "sub_metering_3_wh": rng.uniform(0, 15, n),
        },
        index=idx,
    )


class TestTemporalFeatures:
    def test_adds_expected_columns(self, sample_df):
        result = add_temporal_features(sample_df)
        expected = [
            "hour", "day_of_week", "month", "is_weekend",
            "time_of_use", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "month_sin", "month_cos",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_hour_range(self, sample_df):
        result = add_temporal_features(sample_df)
        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23

    def test_weekend_flag(self, sample_df):
        result = add_temporal_features(sample_df)
        # Jan 15, 2024 is a Monday → is_weekend should be 0
        assert result["is_weekend"].iloc[0] == 0

    def test_cyclical_range(self, sample_df):
        result = add_temporal_features(sample_df)
        assert result["hour_sin"].min() >= -1.0
        assert result["hour_sin"].max() <= 1.0
        assert result["hour_cos"].min() >= -1.0
        assert result["hour_cos"].max() <= 1.0

    def test_time_of_use_categories(self, sample_df):
        result = add_temporal_features(sample_df)
        valid_categories = {
            "off_peak", "morning_peak", "daytime",
            "evening_peak", "late_evening",
        }
        actual = set(result["time_of_use"].unique())
        assert actual.issubset(valid_categories)

    def test_does_not_modify_original(self, sample_df):
        original_cols = set(sample_df.columns)
        add_temporal_features(sample_df)
        assert set(sample_df.columns) == original_cols


class TestLagFeatures:
    def test_default_lags(self, sample_df):
        result = add_lag_features(sample_df)
        for lag in [1, 5, 15, 60]:
            col = f"global_active_power_kw_lag_{lag}m"
            assert col in result.columns

    def test_custom_lags(self, sample_df):
        result = add_lag_features(sample_df, lags=[2, 10])
        assert "global_active_power_kw_lag_2m" in result.columns
        assert "global_active_power_kw_lag_10m" in result.columns

    def test_lag_values_correct(self, sample_df):
        result = add_lag_features(sample_df, lags=[1])
        # Row 1's lag should equal row 0's value
        assert result["global_active_power_kw_lag_1m"].iloc[1] == pytest.approx(
            sample_df["global_active_power_kw"].iloc[0]
        )

    def test_lag_introduces_nans(self, sample_df):
        result = add_lag_features(sample_df, lags=[5])
        assert result["global_active_power_kw_lag_5m"].iloc[:5].isna().all()
        assert result["global_active_power_kw_lag_5m"].iloc[5:].notna().all()


class TestRollingFeatures:
    def test_default_windows(self, sample_df):
        result = add_rolling_features(sample_df)
        for label in ["1h", "6h", "24h"]:
            for stat in ["mean", "std", "min", "max"]:
                col = f"global_active_power_kw_roll_{stat}_{label}"
                assert col in result.columns, f"Missing: {col}"

    #def test_custom_windows(self, sample_df):
    #    result = add_rolling_features(sample_df, windows=[30])
    #    assert "global_active_power_kw_roll_mean_0h" in result.columns


    def test_custom_windows(self, sample_df):
        result = add_rolling_features(sample_df, windows=[30])
        assert "global_active_power_kw_roll_mean_30m" in result.columns

    def test_rolling_mean_reasonable(self, sample_df):
        result = add_rolling_features(sample_df, windows=[60])
        roll_mean = result["global_active_power_kw_roll_mean_1h"].dropna()
        raw_mean = sample_df["global_active_power_kw"].mean()
        # Rolling mean should be in the same ballpark
        assert abs(roll_mean.mean() - raw_mean) < 1.0


class TestRateOfChange:
    def test_adds_diff_columns(self, sample_df):
        result = add_rate_of_change(sample_df)
        assert "global_active_power_kw_diff_1m" in result.columns
        assert "global_active_power_kw_diff_5m" in result.columns
        assert "global_active_power_kw_pct_change_1m" in result.columns

    def test_first_diff_is_nan(self, sample_df):
        result = add_rate_of_change(sample_df)
        assert pd.isna(result["global_active_power_kw_diff_1m"].iloc[0])

    def test_diff_value_correct(self, sample_df):
        result = add_rate_of_change(sample_df)
        expected = (
            sample_df["global_active_power_kw"].iloc[1]
            - sample_df["global_active_power_kw"].iloc[0]
        )
        assert result["global_active_power_kw_diff_1m"].iloc[1] == pytest.approx(expected)


class TestSubmeteringRatios:
    def test_adds_ratio_columns(self, sample_df):
        result = add_submetering_ratios(sample_df)
        expected = ["sub_total_wh", "total_wh_per_min", "metered_ratio",
                     "sub_1_share", "sub_2_share", "sub_3_share"]
        for col in expected:
            assert col in result.columns

    def test_shares_sum_to_one(self, sample_df):
        result = add_submetering_ratios(sample_df)
        share_sum = (
            result["sub_1_share"] + result["sub_2_share"] + result["sub_3_share"]
        )
        # Should be approximately 1.0 for rows where sub_total > 0
        valid = result["sub_total_wh"] > 0
        assert share_sum[valid].mean() == pytest.approx(1.0, abs=0.01)

    def test_metered_ratio_clipped(self, sample_df):
        result = add_submetering_ratios(sample_df)
        assert result["metered_ratio"].max() <= 1.5
        assert result["metered_ratio"].min() >= 0


class TestBuildFeatureMatrix:
    #def test_full_pipeline_runs(self, sample_df):
    #    result = build_feature_matrix(sample_df, drop_na=True)
    #    assert len(result) > 0
    #    assert len(result) < len(sample_df)  # Some rows dropped from warmup


    def test_full_pipeline_runs(self, sample_df):
        # Use smaller windows so 200-row sample survives dropna
        result = build_feature_matrix(sample_df, drop_na=False)
        result_clean = result.dropna(subset=["global_active_power_kw_roll_mean_1h"])
        assert len(result_clean) > 0
        assert len(result_clean) < len(sample_df)

    #def test_no_nans_when_drop_na(self, sample_df):
    #    result = build_feature_matrix(sample_df, drop_na=True)
    #    numeric = result.select_dtypes(include=[np.number])
    #    assert numeric.isna().sum().sum() == 0


    def test_no_nans_when_drop_na(self):
        # Need enough rows for the 24h window (1440 min)
        n = 1500
        idx = pd.date_range("2024-01-15", periods=n, freq="min")
        rng = np.random.default_rng(42)
        big_df = pd.DataFrame(
            {
                "global_active_power_kw": rng.uniform(0.5, 5.0, n),
                "global_reactive_power_kw": rng.uniform(0.05, 0.5, n),
                "voltage_v": rng.normal(230, 3, n),
                "global_intensity_a": rng.uniform(2, 22, n),
                "sub_metering_1_wh": rng.uniform(0, 10, n),
                "sub_metering_2_wh": rng.uniform(0, 8, n),
                "sub_metering_3_wh": rng.uniform(0, 15, n),
            },
            index=idx,
        )
        result = build_feature_matrix(big_df, drop_na=True)
        numeric = result.select_dtypes(include=[np.number])
        assert len(result) > 0
        assert numeric.isna().sum().sum() == 0

    def test_feature_count(self, sample_df):
        result = build_feature_matrix(sample_df, drop_na=False)
        # Should have significantly more columns than input
        assert len(result.columns) > len(sample_df.columns) + 20

    def test_preserves_index_type(self, sample_df):
        result = build_feature_matrix(sample_df)
        assert isinstance(result.index, pd.DatetimeIndex)
