
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import joblib

from .data_loader import load_data
from .feature_engineering import (
    add_calendar_features,
    calculate_monthly_statistics,
    add_monthly_sales_feature,
)
from .model_training import train_random_forest, train_lightgbm, calculate_metrics
from .mlp_training import train_mlp


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

DEFAULT_FEATURES = [
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
]


def train_model(model_name, features=None, parameters=None):
    if model_name not in {"random_forest", "lightgbm", "keras_mlp"}:
        raise ValueError("Непідтримуваний тип моделі")

    features = features or DEFAULT_FEATURES
    parameters = parameters or {}

    if not features or len(features) != len(set(features)):
        raise ValueError("Список ознак має бути непорожнім і без повторів")

    if not set(features).issubset(DEFAULT_FEATURES):
        raise ValueError("Передано недозволені ознаки")

    from uuid import uuid4

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    saved_name = (
        f"{model_name}_{timestamp}_{uuid4().hex[:6]}"
    )

    model_dir = MODELS_DIR / saved_name

    train, test = load_data()

    train = add_calendar_features(train)
    test = add_calendar_features(test)

    validation = test[
        (test["Date"] >= "2015-01-01")
        & (test["Date"] < "2015-04-01")
    ].copy()

    # Історичні статистики обчислюємо тільки на Train.
    statistics = calculate_monthly_statistics(train)

    train = train.sort_values(["Store", "Date"]).copy()
    historical_sales = train["Sales"].where(train["Open"] == 1)

    train["logMonthSales"] = (
        historical_sales.groupby(
            [train["Store"], train["month"]]
        )
        .transform(lambda x: x.shift(1).expanding().mean())
        .pipe(lambda x: __import__("numpy").log1p(x))
    )

    validation = add_monthly_sales_feature(validation, statistics)

    # Навчаємося й оцінюємося лише на відкритих магазинах.
    train = train[train["Open"] == 1].copy()
    validation = validation[validation["Open"] == 1].copy()

    start_time = time.perf_counter()

    history = None

    if model_name == "random_forest":
        model, metrics = train_random_forest(
            train,
            validation,
            features,
            parameters,
        )
    elif model_name == "lightgbm":
        model, metrics, history = train_lightgbm(
            train,
            validation,
            features,
            parameters,
        )
    else:
        mlp_artifacts = train_mlp(
            train,
            validation,
            features,
            parameters,
        )

        model = mlp_artifacts["model"]
        history = mlp_artifacts["history"]

        import numpy as np

        def evaluate_mlp(data):
            X = mlp_artifacts["preprocessor"].transform(
                data[features]
            )

            X = mlp_artifacts["scaler"].transform(X)

            predictions_scaled = model.predict(
                X,
                verbose=0
            )

            predictions_log = (
                mlp_artifacts["target_scaler"]
                .inverse_transform(predictions_scaled)
                .ravel()
            )

            y_true_log = np.log1p(
                data["Sales"].to_numpy()
            )

            return calculate_metrics(
                y_true_log,
                predictions_log
            )

        metrics = {
            "train": evaluate_mlp(train),
            "validation": evaluate_mlp(validation)
        }

    training_time = time.perf_counter() - start_time

    # Окрема назва захищає наявну модель.
    from uuid import uuid4

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    saved_name = (
        f"{model_name}_{timestamp}_{uuid4().hex[:6]}"
    )

    model_dir = MODELS_DIR / saved_name

    model_dir.mkdir(parents=True)

    if model_name == "keras_mlp":
        model.save(str(model_dir / "model.keras"))

        joblib.dump(
            mlp_artifacts["preprocessor"],
            model_dir / "preprocessor.joblib"
        )
        joblib.dump(
            mlp_artifacts["scaler"],
            model_dir / "feature_scaler.joblib"
        )
        joblib.dump(
            mlp_artifacts["target_scaler"],
            model_dir / "target_scaler.joblib"
        )
    else:
        joblib.dump(model, model_dir / "model.joblib")

    joblib.dump(
        statistics,
        model_dir / "monthly_statistics.joblib"
    )

    if history is not None:
        with open(
            model_dir / "training_history.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=2, ensure_ascii=False)

    metadata = {
        "model_name": saved_name,
        "model_type": {
            "random_forest": "Random Forest",
            "lightgbm": "LightGBM",
            "keras_mlp": "Keras MLP"
        }[model_name],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "parameters": parameters,
        "metrics": metrics,
        "training_time_seconds": training_time,
    }

    with open(model_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return metadata
