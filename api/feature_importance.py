
import numpy as np
import pandas as pd


def calculate_feature_importance(pipeline, original_features):
    """Обчислення важливості початкових ознак моделі."""

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    if len(transformed_names) != len(importances):
        raise ValueError("Кількість ознак і значень важливості не збігається")

    # Сортуємо за довжиною, щоб правильно розпізнавати назви
    sorted_features = sorted(
        original_features,
        key=len,
        reverse=True
    )

    results = {feature: 0.0 for feature in original_features}

    for transformed_name, importance in zip(
        transformed_names, importances
    ):
        name = transformed_name.split("__", 1)[-1]

        for feature in sorted_features:
            if name == feature or name.startswith(feature + "_"):
                results[feature] += float(importance)
                break

    total = sum(results.values())

    if total > 0:
        results = {
            feature: value / total * 100
            for feature, value in results.items()
        }

    importance_df = pd.DataFrame({
        "Feature": list(results.keys()),
        "Importance (%)": list(results.values())
    })

    return importance_df.sort_values(
        "Importance (%)",
        ascending=False
    ).reset_index(drop=True)
