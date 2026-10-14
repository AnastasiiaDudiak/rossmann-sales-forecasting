
import numpy as np
import pandas as pd
import pytest

from api.baseline import StoreMeanBaseline, evaluate_baseline


@pytest.fixture
def train_data():
    return pd.DataFrame({
        "Store": [1, 1, 2, 2],
        "Open": [1, 1, 1, 0],
        "Sales": [100, 200, 300, 0]
    })


def test_store_mean(train_data):
    model = StoreMeanBaseline().fit(train_data)

    assert model.store_means[1] == 150
    assert model.store_means[2] == 300


def test_closed_store_prediction(train_data):
    model = StoreMeanBaseline().fit(train_data)

    data = pd.DataFrame({
        "Store": [1],
        "Open": [0]
    })

    prediction = model.predict(data)

    assert prediction[0] == 0


def test_unknown_store_prediction(train_data):
    model = StoreMeanBaseline().fit(train_data)

    data = pd.DataFrame({
        "Store": [999],
        "Open": [1]
    })

    prediction = model.predict(data)

    assert prediction[0] == pytest.approx(200)


def test_baseline_metrics(train_data):
    model = StoreMeanBaseline().fit(train_data)

    metrics = evaluate_baseline(model, train_data)

    assert set(metrics) == {"mae", "rmse", "wape", "r2"}
    assert all(
        np.isfinite(value)
        for value in metrics.values()
    )
