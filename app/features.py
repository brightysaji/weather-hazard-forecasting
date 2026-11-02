"""
Builds the exact same lag / rolling-average / calendar features used during training in
the experimentation notebooks (01/02 target + feature engineering). Keeping this logic
identical between training and serving is essential to avoid train/serve skew.
"""
import numpy as np
import pandas as pd

BASE_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
]

LAGS = [1, 2, 3]
ROLLING_WINDOWS = [3, 7]


def build_feature_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a daily weather dataframe (sorted ascending by date, with at least the
    BASE_FEATURES columns), returns a dataframe with lag, rolling-average and calendar
    features appended - one row per input row, with NaN for rows that don't have
    enough history yet (same convention as the training notebooks).
    """
    df = daily_df.sort_values("date").reset_index(drop=True).copy()

    for col in BASE_FEATURES:
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    for col in BASE_FEATURES:
        for window in ROLLING_WINDOWS:
            df[f"{col}_roll{window}"] = (
                df[col].shift(1).rolling(window=window, min_periods=window).mean()
            )

    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def all_feature_columns() -> list[str]:
    """Returns the full ordered feature list, matching what each model was trained on."""
    lag_features = [f"{col}_lag{lag}" for col in BASE_FEATURES for lag in LAGS]
    rolling_features = [f"{col}_roll{w}" for col in BASE_FEATURES for w in ROLLING_WINDOWS]
    calendar_features = ["month_sin", "month_cos"]
    return BASE_FEATURES + lag_features + rolling_features + calendar_features