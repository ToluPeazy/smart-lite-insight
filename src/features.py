"""Feature engineering pipeline for Smart-Lite Insight.

Transforms raw energy readings into ML-ready features. Each function
is pure (input DataFrame → output DataFrame) and composable.

Usage:
    from src.features import build_feature_matrix

    df_features = build_feature_matrix(df_raw)
"""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the timestamp index.

    Creates: hour, day_of_week, month, is_weekend, time_of_use,
    and cyclical sin/cos encodings for hour, day, and month.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with temporal features added.
    """
    out = df.copy()

    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)

    # Time-of-use periods
    conditions = [
        (out["hour"] >= 0) & (out["hour"] < 6),
        (out["hour"] >= 6) & (out["hour"] < 9),
        (out["hour"] >= 9) & (out["hour"] < 17),
        (out["hour"] >= 17) & (out["hour"] < 21),
        (out["hour"] >= 21),
    ]
    choices = ["off_peak", "morning_peak", "daytime", "evening_peak", "late_evening"]
    out["time_of_use"] = np.select(conditions, choices, default="unknown")

    # Cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "global_active_power_kw",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged values of the target column.

    Args:
        df: DataFrame with time-series data.
        target_col: Column to create lags for.
        lags: List of lag offsets in minutes (default: [1, 5, 15, 60]).

    Returns:
        DataFrame with lag columns added.
    """
    if lags is None:
        lags = [1, 5, 15, 60]

    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag_{lag}m"] = out[target_col].shift(lag)

    return out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "global_active_power_kw",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean, std, min, max for given window sizes.

    Args:
        df: DataFrame with time-series data.
        target_col: Column to compute rolling stats for.
        windows: Window sizes in minutes (default: [60, 360, 1440] = 1h, 6h, 24h).

    Returns:
        DataFrame with rolling statistic columns added.
    """
    if windows is None:
        windows = [60, 360, 1440]

    out = df.copy()
    for w in windows:
        label = f"{w // 60}h" if w >= 60 else f"{w}m"
        rolling = out[target_col].rolling(window=w, min_periods=w // 2)

        out[f"{target_col}_roll_mean_{label}"] = rolling.mean()
        out[f"{target_col}_roll_std_{label}"] = rolling.std()
        out[f"{target_col}_roll_min_{label}"] = rolling.min()
        out[f"{target_col}_roll_max_{label}"] = rolling.max()

    return out


def add_rate_of_change(
    df: pd.DataFrame,
    target_col: str = "global_active_power_kw",
) -> pd.DataFrame:
    """Add rate-of-change (first difference) features.

    Args:
        df: DataFrame with time-series data.
        target_col: Column to compute differences for.

    Returns:
        DataFrame with difference and percentage change columns.
    """
    out = df.copy()

    out[f"{target_col}_diff_1m"] = out[target_col].diff()
    out[f"{target_col}_diff_5m"] = out[target_col].diff(5)

    prev = out[target_col].shift(1)
    out[f"{target_col}_pct_change_1m"] = (
        out[target_col].diff() / prev.replace(0, np.nan)
    )

    return out


def add_submetering_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add sub-metering ratio features.

    Computes the proportion of total energy captured by each
    sub-meter and the overall metered ratio.

    Args:
        df: DataFrame with sub_metering columns.

    Returns:
        DataFrame with ratio columns added.
    """
    out = df.copy()

    out["sub_total_wh"] = (
        out["sub_metering_1_wh"]
        + out["sub_metering_2_wh"]
        + out["sub_metering_3_wh"]
    )

    out["total_wh_per_min"] = out["global_active_power_kw"] * 1000 / 60

    out["metered_ratio"] = (
        out["sub_total_wh"] / out["total_wh_per_min"].replace(0, np.nan)
    ).clip(0, 1.5)

    for i in [1, 2, 3]:
        out[f"sub_{i}_share"] = (
            out[f"sub_metering_{i}_wh"] / out["sub_total_wh"].replace(0, np.nan)
        ).fillna(0)

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "global_active_power_kw",
    drop_na: bool = True,
) -> pd.DataFrame:
    """Build the complete feature matrix from raw readings.

    Applies all feature engineering steps in sequence:
    temporal → lags → rolling → rate of change → sub-metering ratios.

    Args:
        df: Raw readings DataFrame with DatetimeIndex.
        target_col: Primary target column for lag/rolling features.
        drop_na: Whether to drop rows with NaN from window warmup.

    Returns:
        Feature-enriched DataFrame ready for ML.
    """
    out = df.copy()

    out = add_temporal_features(out)
    out = add_lag_features(out, target_col=target_col)
    out = add_rolling_features(out, target_col=target_col)
    out = add_rate_of_change(out, target_col=target_col)
    out = add_submetering_ratios(out)

    if drop_na:
        out = out.dropna()

    return out
