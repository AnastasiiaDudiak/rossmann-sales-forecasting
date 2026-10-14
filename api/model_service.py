
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import tensorflow as tf

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SUPPORTED_MODELS = {
    "random_forest",
    "lightgbm",
    "keras_mlp"
}


@lru_cache(maxsize=3)

def get_model_type(model_name):
    """Визначення типу моделі за метаданими."""
    import json
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", model_name):
        raise ValueError("Некоректна назва моделі")

    metadata_path = MODELS_DIR / model_name / "metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Метадані моделі {model_name} не знайдено"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_type = metadata["model_type"]

    # Приводимо різні назви до єдиного формату
    model_types = {
        "RandomForestRegressor": "Random Forest",
        "Random Forest": "Random Forest",
        "LGBMRegressor": "LightGBM",
        "LightGBM": "LightGBM",
        "Keras MLP": "Keras MLP"
    }

    if model_type not in model_types:
        raise ValueError(
            f"Непідтримуваний тип моделі: {model_type}"
        )

    return model_types[model_type]


def load_model(model_name):
    """Завантаження навченої моделі."""

    model_dir = MODELS_DIR / model_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Модель {model_name} не знайдено"
        )

    if get_model_type(model_name) in {"Random Forest", "LightGBM"}:
        return {
            "model": joblib.load(
                model_dir / "model.joblib"
            ),
            "monthly_statistics": joblib.load(
                model_dir / "monthly_statistics.joblib"
            )
        }

    return {
        "model": tf.keras.models.load_model(
            str(model_dir / "model.keras")
        ),
        "preprocessor": joblib.load(
            model_dir / "preprocessor.joblib"
        ),
        "scaler": joblib.load(
            model_dir / "feature_scaler.joblib"
        ),
        "target_scaler": joblib.load(
            model_dir / "target_scaler.joblib"
        ),
        "monthly_statistics": joblib.load(
            model_dir / "monthly_statistics.joblib"
        )
    }


def predict_sales(model_name, data, features):
    """Прогноз продажів у початковій шкалі."""

    artifacts = load_model(model_name)

    if get_model_type(model_name) in {"Random Forest", "LightGBM"}:
        predictions_log = artifacts["model"].predict(
            data[features]
        )

    else:
        X = artifacts["preprocessor"].transform(
            data[features]
        )

        X = artifacts["scaler"].transform(X)

        predictions_scaled = artifacts["model"].predict(
            X,
            verbose=0
        )

        predictions_log = (
            artifacts["target_scaler"]
            .inverse_transform(predictions_scaled)
            .ravel()
        )

    predictions = np.maximum(
        np.expm1(predictions_log),
        0
    )

    # Закриті магазини не мають продажів
    predictions = np.where(
        data["Open"].to_numpy() == 0,
        0,
        predictions
    )

    return predictions


import pandas as pd

from .feature_engineering import (
    add_calendar_features,
    add_monthly_sales_feature
)


def prepare_prediction_data(model_name, data):
    """Підготовка вхідних даних для прогнозування."""

    artifacts = load_model(model_name)

    prepared = add_calendar_features(data)

    prepared = add_monthly_sales_feature(
        prepared,
        artifacts["monthly_statistics"]
    )

    required_features = [
        "Promo",
        "SchoolHoliday",
        "logMonthSales",
        "CompetitionDistance",
        "Store",
        "StateHoliday",
        "StoreType",
        "Assortment",
        "weekday",
        "monthday",
        "month",
        "Open"
    ]

    missing = [
        column for column in required_features
        if column not in prepared.columns
    ]

    if missing:
        raise ValueError(
            f"Відсутні необхідні колонки: {missing}"
        )

    return prepared
