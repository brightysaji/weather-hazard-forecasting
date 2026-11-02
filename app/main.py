import datetime as dt
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from app import model_service
from app.weather_client import WeatherFetchError

GITHUB_URL = "https://github.com/brightysaji/weather-hazard-forecasting"  


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_models()
    yield


app = FastAPI(
    title="Weather Intelligence API",
    description="Machine learning services for weather forecasting - Sydney, NSW",
    version="1.0.0",
    lifespan=lifespan,
)


def _parse_date(date_str: str) -> dt.date:
    try:
        return dt.date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.",
        )


@app.get("/")
def root():
    return {
        "project": "Weather Intelligence API",
        "description": "Machine learning services forecasting the Climate Comfort Index "
                        "and Weather Hazard Category for Sydney, NSW, using historical "
                        "weather data from the Open-Meteo API.",
        "version": "1.0.0",
        "github": GITHUB_URL,
        "endpoints": [
            "/",
            "/health",
            "/predict/index/comfort_climate",
            "/predict/category/weather_hazard",
            "/model-metadata",
        ],
    }


@app.get("/health")
def health():
    if not model_service.models_loaded():
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Models failed to load."},
        )
    return {"status": "healthy", "message": "Weather Intelligence API is running"}


@app.get("/predict/index/comfort_climate")
def predict_comfort_climate(date: str = Query(..., description="Input date, format YYYY-MM-DD")):
    input_date = _parse_date(date)

    if input_date > dt.date.today():
        raise HTTPException(status_code=400, detail="Input date cannot be in the future.")

    try:
        predictions = model_service.predict_cci(input_date)
    except WeatherFetchError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "input_date": input_date.isoformat(),
        "predictions": {"comfort_climate": predictions},
    }


@app.get("/predict/category/weather_hazard")
def predict_weather_hazard(date: str = Query(..., description="Input date, format YYYY-MM-DD")):
    input_date = _parse_date(date)

    if input_date > dt.date.today():
        raise HTTPException(status_code=400, detail="Input date cannot be in the future.")

    try:
        predictions = model_service.predict_whc(input_date)
    except WeatherFetchError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "input_date": input_date.isoformat(),
        "predictions": {"weather_hazard": predictions},
    }


@app.get("/model-metadata")
def model_metadata():
    return [
        {
            "target": "Climate Comfort Index",
            "prediction_type": "Regression",
            "algorithm": "XGBoost Regressor (hyperparameter-tuned)",
            "forecast_horizon": "1-3 Days",
            "features": model_service._cci_features if model_service.models_loaded() else [],
            "feature_importance_top5": {
                "temperature_2m_roll7": 0.092,
                "temperature_2m_roll3": 0.066,
                "temperature_2m_lag1": 0.065,
                "temperature_2m": 0.054,
                "month_cos": 0.040,
            },
            "performance_metrics": {
                "test_rmse": 10.16,
                "test_mae": 7.70,
                "test_r2": 0.249,
            },
        },
        {
            "target": "Weather Hazard Category",
            "prediction_type": "Multiclass Classification",
            "algorithm": "XGBoost Classifier (regularized, balanced class weighting)",
            "forecast_horizon": "7 Days",
            "features": model_service._whc_features if model_service.models_loaded() else [],
            "feature_importance_top5": {
                "temperature_2m": 0.044,
                "cloud_cover_lag3": 0.041,
                "cloud_cover_roll3": 0.040,
                "wind_speed_10m_lag3": 0.039,
                "relative_humidity_2m_lag1": 0.038,
            },
            "performance_metrics": {
                "test_macro_f1": 0.248,
                "test_weighted_f1": 0.443,
            },
        },
    ]