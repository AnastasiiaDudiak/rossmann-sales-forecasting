
import numpy as np
import pandas as pd


def add_calendar_features(df):
    """Створення календарних ознак."""
    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"])

    result["year"] = result["Date"].dt.year
    result["month"] = result["Date"].dt.month
    result["monthday"] = result["Date"].dt.day
    result["weekday"] = result["Date"].dt.dayofweek

    result["StateHoliday"] = (
        result["StateHoliday"].fillna("0").astype(str)
    )

    return result


def calculate_monthly_statistics(train):
    """Статистика лише за навчальними даними."""
    open_days = train[train["Open"] == 1].copy()

    statistics = (
        open_days.groupby(["Store", "month"])["Sales"]
        .mean()
        .reset_index(name="meanMonthSales")
    )

    return statistics


def add_monthly_sales_feature(df, statistics):
    """Додавання історичної ознаки logMonthSales."""
    result = df.drop(
        columns=["logMonthSales", "meanMonthSales"],
        errors="ignore"
    ).copy()

    result = result.merge(
        statistics,
        on=["Store", "month"],
        how="left",
        validate="many_to_one"
    )

    result["logMonthSales"] = np.log1p(
        result["meanMonthSales"]
    )

    return result.drop(columns=["meanMonthSales"])


def prepare_features(train, test):
    """Підготовка навчальних і тестових ознак."""
    train = add_calendar_features(train)
    test = add_calendar_features(test)

    statistics = calculate_monthly_statistics(train)

    # Для Train використовуємо лише попередні спостереження
    train = train.sort_values(["Store", "Date"]).copy()

    historical_sales = train["Sales"].where(train["Open"] == 1)

    train["logMonthSales"] = np.log1p(
        historical_sales.groupby(
            [train["Store"], train["month"]]
        ).transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # Для Test — статистику навчального періоду
    test = add_monthly_sales_feature(test, statistics)

    return train, test, statistics
