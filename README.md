# Weather Forecasting Service: ML Models Deployed as an API

Two machine learning models for Sydney, NSW weather, served as a Dockerised FastAPI service:

- **Climate Comfort Index (CCI):** XGBoost regression, a 0-100 comfort score for the next 3 days.
- **Weather Hazard Category (WHC):** XGBoost multiclass classification (Low / Moderate / High / Extreme Risk), predicted exactly 7 days ahead.

Data: 4,018 daily records (2015-2025) from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (CC BY 4.0).

## Results

| Model | Task | Metric |
|---|---|---|
| CCI | Regression | R² 0.25 |
| WHC | Multiclass classification | macro-F1 0.25 |

Both scores are modest, and the notebooks say so. Error analysis shows the hazard model missed 13 of 31 high-risk days, so it should not be used without human review for high-risk decisions.

## Method

- **Features:** 38 lag, rolling-window and cyclical calendar features built from daily weather.
- **Splits:** chronological train / validation / test splits to prevent leakage.
- **Models:** XGBoost for both tasks. The target-engineering helpers (`compute_cci`, `compute_whi`, `whi_to_class`, `attach_future_target`) come from a small Python package published to TestPyPI.
- **Serving:** the API fetches recent weather from Open-Meteo for the requested date, rebuilds the same features used in training, and returns predictions.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Overview |
| GET | `/health` | Health check |
| GET | `/predict/index/comfort_climate?date=YYYY-MM-DD` | CCI forecast, next 3 days |
| GET | `/predict/category/weather_hazard?date=YYYY-MM-DD` | WHC forecast, 7 days ahead |
| GET | `/model-metadata` | Model details, features and metrics |

## Repository layout

```
app/          FastAPI service (endpoints, feature engineering, model loading)
models/       Trained XGBoost models and feature lists
notebooks/    Data acquisition and the two experiment notebooks (CCI, WHC)
data/         Open-Meteo data used for the experiments
Dockerfile
pyproject.toml
```

## Run locally

With [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs`. With Docker:

```
docker build -t weather-intelligence-api .
docker run -p 8000:8000 weather-intelligence-api
```

## Limitations

- Open-Meteo's archive lags a few days behind real time, so requests for the most recent dates can fail with a 422 error.
- CCI predicts 3 days ahead from a given day's features; the day+1, day+2 and day+3 outputs reuse the same model with features from different reference days.
- Model scores are modest (R² 0.25, macro-F1 0.25); the aim of the project was a sound, leakage-safe pipeline and deployment, not a state-of-the-art forecast.
