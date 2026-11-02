"""
Loads the trained CCI and WHC model artefacts and exposes prediction functions that
handle fetching the required historical weather window, building features, and
running inference - matching the leakage-safe design used during training.
"""
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd

from app.features import all_feature_columns, build_feature_frame
from app.weather_client import WeatherFetchError, fetch_daily_weather_window

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

WHC_CLASS_NAMES = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk", 3: "Extreme Risk"}

# The largest lookback any single feature needs is the 7-day rolling window, which
# itself requires 7 days STRICTLY BEFORE the reference day (via shift(1) then
# rolling(7)) - so a reference day needs 7 days of history before it.
ROLLING_LOOKBACK_DAYS = 7

_cci_model = None
_cci_features = None
_whc_model = None
_whc_features = None


def load_models() -> None:
    """Loads both model artefacts into memory once, at API startup."""
    global _cci_model, _cci_features, _whc_model, _whc_features

    cci_dir = MODELS_DIR / "comfort_climate"
    whc_dir = MODELS_DIR / "weather_hazard"

    _cci_model = joblib.load(cci_dir / "cci_xgboost_model.pkl")
    _cci_features = joblib.load(cci_dir / "cci_feature_list.pkl")

    _whc_model = joblib.load(whc_dir / "whc_xgboost_model.pkl")
    _whc_features = joblib.load(whc_dir / "whc_feature_list.pkl")


def models_loaded() -> bool:
    return all(x is not None for x in (_cci_model, _cci_features, _whc_model, _whc_features))


def predict_cci(input_date: dt.date) -> dict[str, float]:
    """
    Predicts Climate Comfort Index for the 3 days following input_date.

    The trained model predicts CCI exactly 3 days after whatever reference day's
    features it is given. To produce day+1/day+2/day+3 from a single input_date, the
    model is applied 3 times, each time with features computed as of a different
    reference day (input_date-2, input_date-1, input_date respectively) - reusing the
    single trained model without needing 3 separate models.
    """
    # Need history back to the earliest reference day (input_date - 2) minus its own
    # 7-day rolling lookback, i.e. input_date - 2 - 7 = input_date - 9.
    lookback_days = 2 + ROLLING_LOOKBACK_DAYS
    daily_df = fetch_daily_weather_window(input_date, days_before=lookback_days)
    feature_df = build_feature_frame(daily_df)

    predictions = {}
    for offset, horizon in [(2, 1), (1, 2), (0, 3)]:
        reference_date = pd.Timestamp(input_date) - pd.Timedelta(days=offset)
        row = feature_df[feature_df["date"] == reference_date]

        if row.empty or row[_cci_features].isnull().any(axis=1).iloc[0]:
            raise WeatherFetchError(
                f"Insufficient historical weather data to compute features for "
                f"{reference_date.date()} (needed to predict {horizon} day(s) ahead)."
            )

        X = row[_cci_features]
        pred = float(_cci_model.predict(X)[0])
        target_date = (pd.Timestamp(input_date) + pd.Timedelta(days=horizon)).date().isoformat()
        predictions[target_date] = round(pred, 2)

    return predictions


def predict_whc(input_date: dt.date) -> dict[str, str]:
    """
    Predicts Weather Hazard Category exactly 7 days after input_date, using features
    computed directly from input_date's own (and preceding) weather.
    """
    daily_df = fetch_daily_weather_window(input_date, days_before=ROLLING_LOOKBACK_DAYS)
    feature_df = build_feature_frame(daily_df)

    reference_date = pd.Timestamp(input_date)
    row = feature_df[feature_df["date"] == reference_date]

    if row.empty or row[_whc_features].isnull().any(axis=1).iloc[0]:
        raise WeatherFetchError(
            f"Insufficient historical weather data to compute features for {input_date}."
        )

    X = row[_whc_features]
    pred_class = int(_whc_model.predict(X)[0])
    target_date = (pd.Timestamp(input_date) + pd.Timedelta(days=7)).date().isoformat()

    return {target_date: WHC_CLASS_NAMES[pred_class]}