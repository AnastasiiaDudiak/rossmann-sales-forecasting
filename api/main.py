
from fastapi import FastAPI

app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="API для навчання моделей і прогнозування продажів",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    """Перевірка працездатності API."""
    return {
        "status": "ok",
        "service": "Rossmann Sales Forecasting API"
    }


from .data_loader import load_data


@app.get("/data/info", tags=["Data"])
def get_data_info():
    """Інформація про навчальні та тестові дані Rossmann."""

    train, test = load_data()

    return {
        "train": {
            "rows": len(train),
            "columns": len(train.columns),
            "stores": int(train["Store"].nunique()),
            "date_min": str(train["Date"].min().date()),
            "date_max": str(train["Date"].max().date())
        },
        "test": {
            "rows": len(test),
            "columns": len(test.columns),
            "stores": int(test["Store"].nunique()),
            "date_min": str(test["Date"].min().date()),
            "date_max": str(test["Date"].max().date())
        },
        "column_types": {
            column: str(dtype)
            for column, dtype in train.dtypes.items()
        }
    }


import json
from pathlib import Path
from fastapi import HTTPException

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@app.get("/models", tags=["Models"])
def get_models():
    """Список усіх збережених моделей."""

    models = []

    if not MODELS_DIR.exists():
        return {
            "total": 0,
            "models": []
        }

    for model_dir in sorted(MODELS_DIR.iterdir()):
        metadata_path = model_dir / "metadata.json"

        if model_dir.is_dir() and metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            models.append({
                "name": metadata["model_name"],
                "type": metadata["model_type"],
                "created_at": metadata["created_at"],
                "validation_metrics": metadata["metrics"]["validation"]
            })

    return {
        "total": len(models),
        "models": models
    }


@app.get("/models/{name}", tags=["Models"])
def get_model_info(name: str):
    """Детальна інформація про збережену модель."""

    # Дозволяємо лише назви наявних папок моделей.
    # Забороняємо спеціальні символи та переходи між каталогами.
    if not name or not all(
        char.isalnum() and char.isascii() or char in "_-"
        for char in name
    ):
        raise HTTPException(
            status_code=400,
            detail="Некоректна назва моделі"
        )

    metadata_path = MODELS_DIR / name / "metadata.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Метадані моделі не знайдено"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata




@app.delete("/models/{name}", tags=["Models"])
def delete_model(name: str):
    """Безпечне видалення збереженої моделі."""
    import re
    import shutil

    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise HTTPException(
            status_code=400,
            detail="Некоректна назва моделі"
        )

    model_dir = MODELS_DIR / name
    metadata_path = model_dir / "metadata.json"

    if (
        not model_dir.is_dir()
        or model_dir.is_symlink()
        or not metadata_path.is_file()
    ):
        raise HTTPException(
            status_code=404,
            detail="Модель не знайдено"
        )

    shutil.rmtree(model_dir)

    from .model_service import get_model_type
    get_model_type.cache_clear()

    return {
        "message": "Модель успішно видалено",
        "model_name": name
    }

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .model_service import prepare_prediction_data, predict_sales


class PredictionRequest(BaseModel):
    model_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    store: int = Field(ge=1)
    date: date
    promo: int = Field(default=0, ge=0, le=1)
    school_holiday: int = Field(default=0, ge=0, le=1)
    state_holiday: str = "0"
    open: int = Field(default=1, ge=0, le=1)


@app.post("/predict", tags=["Prediction"])
def predict(request: PredictionRequest):
    """Прогноз продажів одного магазину на вибрану дату."""

    train, test = load_data()
    stores = pd.concat([train, test], ignore_index=True)

    store_data = stores[stores["Store"] == request.store]

    if store_data.empty:
        raise HTTPException(
            status_code=404,
            detail="Магазин не знайдено"
        )

    # Характеристики магазину беремо з наявних даних
    store_info = store_data.iloc[0]

    input_data = pd.DataFrame([{
        "Store": request.store,
        "Date": pd.Timestamp(request.date),
        "Promo": request.promo,
        "SchoolHoliday": request.school_holiday,
        "StateHoliday": request.state_holiday,
        "Open": request.open,
        "StoreType": store_info["StoreType"],
        "Assortment": store_info["Assortment"],
        "CompetitionDistance": store_info["CompetitionDistance"]
    }])

    try:
        prepared = prepare_prediction_data(
            request.model_name,
            input_data
        )

        prediction = predict_sales(
            request.model_name,
            prepared,
            features=[
                "Promo", "SchoolHoliday", "logMonthSales",
                "CompetitionDistance", "Store", "StateHoliday",
                "StoreType", "Assortment", "weekday",
                "monthday", "month"
            ]
        )

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    return {
        "model": request.model_name,
        "store": request.store,
        "date": request.date.isoformat(),
        "predicted_sales": round(float(prediction[0]), 2)
    }


from typing import List


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(
        min_length=1,
        max_length=1000
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Пакетне прогнозування продажів."""

    results = []

    for item in request.predictions:
        result = predict(item)
        results.append(result)

    return {
        "total": len(results),
        "predictions": results
    }


from datetime import timedelta


class ForecastRequest(BaseModel):
    model_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    stores: list[int] = Field(min_length=1, max_length=100)
    start_date: date
    end_date: date


@app.post("/forecast", tags=["Forecast"])
def forecast(request: ForecastRequest):
    """Прогнозування продажів за історичний період."""

    if request.start_date > request.end_date:
        raise HTTPException(
            status_code=422,
            detail="Початкова дата пізніша за кінцеву"
        )

    train, test = load_data()

    # Вибираємо магазини та часовий діапазон
    selected = test[
        test["Store"].isin(request.stores)
        & test["Date"].between(
            pd.Timestamp(request.start_date),
            pd.Timestamp(request.end_date)
        )
    ].copy()

    if selected.empty:
        raise HTTPException(
            status_code=404,
            detail="Для вибраних магазинів і дат немає даних"
        )

    # Прибираємо фактичні продажі перед прогнозуванням
    prediction_input = selected.drop(
        columns=["Sales", "Customers"],
        errors="ignore"
    )

    prepared = prepare_prediction_data(
        request.model_name,
        prediction_input
    )

    model_features = [
        "Promo", "SchoolHoliday", "logMonthSales",
        "CompetitionDistance", "Store", "StateHoliday",
        "StoreType", "Assortment", "weekday",
        "monthday", "month"
    ]

    predictions = predict_sales(
        request.model_name,
        prepared,
        model_features
    )

    result = selected[
        ["Store", "Date", "Sales"]
    ].copy()

    result["predicted_sales"] = predictions

    result = result.sort_values(["Store", "Date"])

    return {
        "model": request.model_name,
        "total": len(result),
        "metrics": calculate_forecast_metrics(
            result["Sales"],
            result["predicted_sales"]
        ),
        "forecasts": [
            {
                "store": int(row.Store),
                "date": row.Date.strftime("%Y-%m-%d"),
                "actual_sales": float(row.Sales),
                "predicted_sales": round(
                    float(row.predicted_sales), 2
                )
            }
            for row in result.itertuples(index=False)
        ]
    }


def calculate_forecast_metrics(actual, predicted):
    """Метрики якості прогнозування продажів."""

    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    total_sales = np.sum(np.abs(actual))
    wape = (
        np.sum(np.abs(actual - predicted))
        / total_sales * 100
        if total_sales > 0 else None
    )

    r2 = r2_score(actual, predicted) if len(actual) >= 2 else None

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "wape": round(float(wape), 2) if wape is not None else None,
        "r2": round(float(r2), 4) if r2 is not None else None
    }



class ScenarioForecastRequest(BaseModel):
    model_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    stores: list[int] = Field(min_length=1, max_length=100)
    start_date: date
    end_date: date
    promo: int = Field(default=0, ge=0, le=1)
    school_holiday: int = Field(default=0, ge=0, le=1)
    state_holiday: str = "0"
    open: int = Field(default=1, ge=0, le=1)


@app.post("/forecast/scenario", tags=["Forecast"])
def forecast_scenario(request: ScenarioForecastRequest):
    """Сценарний прогноз без використання фактичних продажів."""

    if request.start_date > request.end_date:
        raise HTTPException(
            status_code=422,
            detail="Некоректний діапазон дат"
        )

    dates = pd.date_range(
        request.start_date,
        request.end_date
    )

    if len(dates) > 366:
        raise HTTPException(
            status_code=422,
            detail="Максимальний період — 366 днів"
        )

    train, test = load_data()

    stores_data = pd.concat(
        [train, test],
        ignore_index=True
    )

    records = []

    for store_id in request.stores:
        store_rows = stores_data[
            stores_data["Store"] == store_id
        ]

        if store_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Магазин {store_id} не знайдено"
            )

        store_info = store_rows.iloc[0]

        for current_date in dates:
            records.append({
                "Store": store_id,
                "Date": current_date,
                "Promo": request.promo,
                "SchoolHoliday": request.school_holiday,
                "StateHoliday": request.state_holiday,
                "Open": request.open,
                "StoreType": store_info["StoreType"],
                "Assortment": store_info["Assortment"],
                "CompetitionDistance": store_info["CompetitionDistance"]
            })

    input_data = pd.DataFrame(records)

    prepared = prepare_prediction_data(
        request.model_name,
        input_data
    )

    model_features = [
        "Promo", "SchoolHoliday", "logMonthSales",
        "CompetitionDistance", "Store", "StateHoliday",
        "StoreType", "Assortment", "weekday",
        "monthday", "month"
    ]

    predictions = predict_sales(
        request.model_name,
        prepared,
        model_features
    )

    input_data["predicted_sales"] = predictions

    return {
        "model": request.model_name,
        "mode": "scenario",
        "total": len(input_data),
        "forecasts": [
            {
                "store": int(row.Store),
                "date": row.Date.strftime("%Y-%m-%d"),
                "predicted_sales": round(
                    float(row.predicted_sales), 2
                )
            }
            for row in input_data.itertuples(index=False)
        ]
    }


# Навчання моделей через API
from pydantic import BaseModel, Field
from typing import Optional
from .training_service import train_model


class TrainRequest(BaseModel):
    model_name: str = "random_forest"
    features: Optional[list[str]] = None
    parameters: dict = Field(default_factory=dict)


@app.post("/train", tags=["Training"])
def train_endpoint(request: TrainRequest):
    """Навчання та збереження нової моделі."""

    try:
        result = train_model(
            model_name=request.model_name,
            features=request.features,
            parameters=request.parameters
        )

        return {
            "status": "success",
            "message": "Модель успішно навчена",
            "model": result
        }

    except (ValueError, FileExistsError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
