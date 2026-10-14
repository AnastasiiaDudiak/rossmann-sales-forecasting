
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class StoreMeanBaseline:
    """Baseline на основі середніх продажів магазинів."""

    def fit(self, train_df):
        opened = train_df[train_df["Open"] == 1].copy()

        if opened.empty:
            raise ValueError("Немає даних відкритих магазинів")

        self.store_means = opened.groupby("Store")["Sales"].mean()
        self.global_mean = opened["Sales"].mean()

        return self

    def predict(self, df):
        predictions = (
            df["Store"]
            .map(self.store_means)
            .fillna(self.global_mean)
            .to_numpy(dtype=float)
        )

        return np.where(
            df["Open"].to_numpy() == 0,
            0.0,
            predictions
        )


def evaluate_baseline(model, df):
    """Оцінювання Baseline у початковій шкалі продажів."""

    actual = df["Sales"].to_numpy(dtype=float)
    predicted = model.predict(df)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    denominator = np.sum(np.abs(actual))

    wape = (
        np.sum(np.abs(actual - predicted))
        / denominator * 100
        if denominator > 0
        else None
    )

    r2 = (
        r2_score(actual, predicted)
        if len(actual) >= 2
        else None
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "wape": float(wape) if wape is not None else None,
        "r2": float(r2) if r2 is not None else None
    }
