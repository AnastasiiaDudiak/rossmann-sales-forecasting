
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def get_api_url():
    """Отримати адресу FastAPI з налаштувань."""

    # Змінна середовища має пріоритет
    api_url = os.getenv("API_URL")

    if api_url:
        return api_url.rstrip("/")

    # Локальний файл .env
    env_path = PROJECT_DIR / ".env"

    if env_path.exists():
        for line in env_path.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key, separator, value = line.partition("=")

            if separator and key.strip() == "API_URL":
                return value.strip().strip('"').strip("'").rstrip("/")

    return "http://127.0.0.1:8000"
