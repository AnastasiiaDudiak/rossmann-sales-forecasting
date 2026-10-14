
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "Promo",
    "SchoolHoliday",
    "logMonthSales",
    "CompetitionDistance"
]

CATEGORICAL_FEATURES = [
    "Store",
    "StateHoliday",
    "StoreType",
    "Assortment",
    "weekday",
    "monthday",
    "month"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor(features=None):
    """Створення обробника вибраних ознак."""

    if features is None:
        features = ALL_FEATURES.copy()

    unknown = set(features) - set(ALL_FEATURES)

    if unknown:
        raise ValueError(
            f"Невідомі ознаки: {sorted(unknown)}"
        )

    if not features:
        raise ValueError("Потрібно вибрати хоча б одну ознаку")

    numeric = [
        feature for feature in features
        if feature in NUMERIC_FEATURES
    ]

    categorical = [
        feature for feature in features
        if feature in CATEGORICAL_FEATURES
    ]

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric)
        )

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
