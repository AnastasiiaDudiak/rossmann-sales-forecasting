
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor
from .model_training import calculate_metrics


def train_mlp(
    train_df,
    validation_df,
    features,
    params=None
):
    """Навчання нейронної мережі Keras MLP."""

    params = params or {}

    tf.keras.utils.set_random_seed(42)

    # Попередня обробка ознак
    preprocessor = build_preprocessor(features)

    X_train = preprocessor.fit_transform(
        train_df[features]
    )

    X_val = preprocessor.transform(
        validation_df[features]
    )

    # Масштабування ознак
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Цільова змінна logSales
    y_train = np.log1p(train_df["Sales"]).to_numpy()
    y_val = np.log1p(validation_df["Sales"]).to_numpy()

    # Масштабування цільової змінної
    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(
        y_train.reshape(-1, 1)
    )

    y_val_scaled = target_scaler.transform(
        y_val.reshape(-1, 1)
    )

    # Архітектура нейронної мережі
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(
            params.get("hidden_1", 512),
            activation="relu"
        ),
        tf.keras.layers.Dropout(
            params.get("dropout", 0.1)
        ),
        tf.keras.layers.Dense(
            params.get("hidden_2", 256),
            activation="relu"
        ),
        tf.keras.layers.Dropout(
            params.get("dropout", 0.1)
        ),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params.get("learning_rate", 0.0001)
        ),
        loss="mae",
        metrics=["mae"]
    )

    history = model.fit(
        X_train,
        y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=params.get("epochs", 20),
        batch_size=params.get("batch_size", 512),
        verbose=1
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "history": history.history
    }
