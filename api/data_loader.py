
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def load_data():
    """Завантаження навчальних і тестових даних Rossmann."""

    train_path = DATA_DIR / "rossm_train_df.csv"
    test_path = DATA_DIR / "rossm_test_df.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Не знайдено: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Не знайдено: {test_path}")

    train = pd.read_csv(
        train_path,
        parse_dates=["Date"],
        dtype={"StateHoliday": str}
    )

    test = pd.read_csv(
        test_path,
        parse_dates=["Date"],
        dtype={"StateHoliday": str}
    )

    return train, test
