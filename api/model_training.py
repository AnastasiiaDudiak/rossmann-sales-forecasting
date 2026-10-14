
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .preprocessing import build_preprocessor


def build_random_forest(features, params=None):
    """Створення повного ML Pipeline."""

    params = params or {}

    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", build_preprocessor(features)),
        ("model", model)
    ])


def calculate_metrics(y_true_log, y_pred_log):
    """Метрики у логарифмічній та початковій шкалах."""

    y_true = np.expm1(np.asarray(y_true_log))
    y_pred = np.maximum(
        np.expm1(np.asarray(y_pred_log)),
        0
    )

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denominator = np.sum(np.abs(y_true))
    wape = (
        np.sum(np.abs(y_true - y_pred)) / denominator * 100
        if denominator > 0 else None
    )

    return {
        "log_rmse": float(np.sqrt(
            mean_squared_error(y_true_log, y_pred_log)
        )),
        "mae": float(mae),
        "rmse": float(rmse),
        "wape": float(wape) if wape is not None else None,
        "r2": float(r2_score(y_true, y_pred))
    }


def train_random_forest(
    train_df,
    validation_df,
    features,
    params=None
):
    """Навчання Random Forest та оцінювання на Validation."""

    pipeline = build_random_forest(features, params)

    y_train = np.log1p(train_df["Sales"])
    y_val = np.log1p(validation_df["Sales"])

    pipeline.fit(train_df[features], y_train)

    train_predictions = pipeline.predict(
        train_df[features]
    )

    val_predictions = pipeline.predict(
        validation_df[features]
    )

    metrics = {
        "train": calculate_metrics(
            y_train, train_predictions
        ),
        "validation": calculate_metrics(
            y_val, val_predictions
        )
    }

    return pipeline, metrics


def train_lightgbm(
    train_df,
    validation_df,
    features,
    params=None
):
    """Навчання LightGBM з Early Stopping."""

    import lightgbm as lgb

    params = params or {}

    # Обробка ознак навчається тільки на Train
    preprocessor = build_preprocessor(features)

    X_train = preprocessor.fit_transform(
        train_df[features]
    )

    X_val = preprocessor.transform(
        validation_df[features]
    )

    y_train = np.log1p(train_df["Sales"])
    y_val = np.log1p(validation_df["Sales"])

    model = lgb.LGBMRegressor(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.1),
        num_leaves=params.get("num_leaves", 100),
        max_depth=params.get("max_depth", 5),
        colsample_bytree=params.get("colsample_bytree", 0.7),
        random_state=42,
        verbosity=-1
    )

    # Зберігаємо історію навчання
    history = {}

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "validation"],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False
            ),
            lgb.record_evaluation(history)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    train_predictions = pipeline.predict(
        train_df[features]
    )

    val_predictions = pipeline.predict(
        validation_df[features]
    )

    metrics = {
        "train": calculate_metrics(
            y_train, train_predictions
        ),
        "validation": calculate_metrics(
            y_val, val_predictions
        )
    }

    return pipeline, metrics, history
