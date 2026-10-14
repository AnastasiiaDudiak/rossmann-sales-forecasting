from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Model Training | Rossmann",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Model Training")
st.markdown(
    "Налаштування та навчання моделей прогнозування "
    "продажів Rossmann."
)

st.divider()

# Вибір алгоритму
model_options = {
    "Random Forest": "random_forest",
    "LightGBM": "lightgbm",
    "Keras MLP": "keras_mlp"
}

selected_model = st.selectbox(
    "Оберіть алгоритм машинного навчання",
    options=list(model_options.keys())
)

model_name = model_options[selected_model]

# Вибір ознак
available_features = [
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
    "month"
]

st.subheader("Ознаки для навчання")

selected_features = st.multiselect(
    "Виберіть ознаки",
    options=available_features,
    default=available_features,
    help="Customers не використовується, щоб уникнути витоку даних."
)

st.divider()

# Налаштування гіперпараметрів
st.subheader("Гіперпараметри")

parameters = {}

if model_name == "random_forest":

    col1, col2, col3 = st.columns(3)

    with col1:
        parameters["n_estimators"] = st.number_input(
            "Кількість дерев",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

    with col2:
        parameters["max_depth"] = st.number_input(
            "Максимальна глибина",
            min_value=2,
            max_value=50,
            value=15
        )

    with col3:
        parameters["min_samples_leaf"] = st.number_input(
            "Мінімум записів у листку",
            min_value=1,
            max_value=50,
            value=1
        )

    parameters["random_state"] = 42

elif model_name == "lightgbm":

    col1, col2, col3 = st.columns(3)

    with col1:
        parameters["n_estimators"] = st.number_input(
            "Кількість дерев",
            min_value=10,
            max_value=2000,
            value=500,
            step=50
        )

    with col2:
        parameters["learning_rate"] = st.number_input(
            "Швидкість навчання",
            min_value=0.001,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.3f"
        )

    with col3:
        parameters["num_leaves"] = st.number_input(
            "Кількість листків",
            min_value=2,
            max_value=255,
            value=31
        )

    parameters["random_state"] = 42

else:

    col1, col2, col3 = st.columns(3)

    with col1:
        parameters["hidden_1"] = st.number_input(
            "Нейрони першого шару",
            min_value=16,
            max_value=1024,
            value=128,
            step=16
        )

        parameters["hidden_2"] = st.number_input(
            "Нейрони другого шару",
            min_value=8,
            max_value=512,
            value=64,
            step=8
        )

    with col2:
        parameters["epochs"] = st.number_input(
            "Кількість епох",
            min_value=1,
            max_value=200,
            value=20
        )

        parameters["batch_size"] = st.selectbox(
            "Розмір пакета",
            options=[32, 64, 128, 256, 512],
            index=4
        )

    with col3:
        parameters["learning_rate"] = st.number_input(
            "Швидкість навчання",
            min_value=0.00001,
            max_value=0.1,
            value=0.0001,
            format="%.5f"
        )

        parameters["dropout"] = st.slider(
            "Dropout",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05
        )

st.divider()

st.subheader("Поточна конфігурація")

col1, col2 = st.columns(2)

with col1:
    st.metric("Алгоритм", selected_model)

with col2:
    st.metric("Кількість ознак", len(selected_features))

st.markdown("#### Гіперпараметри")

parameter_labels = {
    "n_estimators": "Кількість дерев",
    "max_depth": "Максимальна глибина",
    "min_samples_leaf": "Мінімум записів у листку",
    "num_leaves": "Кількість листків",
    "hidden_1": "Нейрони першого шару",
    "hidden_2": "Нейрони другого шару",
    "epochs": "Кількість епох",
    "batch_size": "Розмір пакета",
    "learning_rate": "Швидкість навчання",
    "dropout": "Dropout",
    "random_state": "Random state"
}

parameter_rows = [
    {
        "Параметр": parameter_labels.get(key, key),
        "Значення": str(value)
    }
    for key, value in parameters.items()
]

st.dataframe(
    parameter_rows,
    use_container_width=True,
    hide_index=True
)

st.markdown("#### Вибрані ознаки")

if selected_features:
    st.write(", ".join(selected_features))
else:
    st.info("Ознаки ще не вибрано.")

if not selected_features:
    st.warning("Оберіть хоча б одну ознаку для навчання.")


# Запуск навчання через FastAPI
st.divider()
st.subheader("Запуск навчання")

from streamlit_app.config import get_api_url
API_URL = get_api_url()

if st.button(
    "🚀 Train Model",
    type="primary",
    disabled=not selected_features,
    use_container_width=True
):
    import requests

    payload = {
        "model_name": model_name,
        "features": selected_features,
        "parameters": parameters
    }

    try:
        with st.spinner("Навчання моделі виконується..."):
            response = requests.post(
                f"{API_URL}/train",
                json=payload,
                timeout=1800
            )

        if response.status_code == 200:
            result = response.json()["model"]

            st.success("Модель успішно навчена!")

            st.write("**Назва моделі:**", result["model_name"])
            st.write(
                "**Час навчання:**",
                round(result["training_time_seconds"], 2),
                "сек."
            )

            metrics = result.get("metrics", {})

            if "validation" in metrics:
                if "train" in metrics:
                    st.subheader("Метрики Train")

                    train_metrics = metrics["train"]

                    t1, t2, t3, t4 = st.columns(4)

                    t1.metric("MAE", f"{train_metrics['mae']:.2f}")
                    t2.metric("RMSE", f"{train_metrics['rmse']:.2f}")
                    t3.metric("WAPE", f"{train_metrics['wape']:.2f}%")
                    t4.metric("R²", f"{train_metrics['r2']:.4f}")

                    st.divider()

                st.subheader("Метрики Validation")

                val = metrics["validation"]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("MAE", f"{val['mae']:.2f}")
                col2.metric("RMSE", f"{val['rmse']:.2f}")
                col3.metric("WAPE", f"{val['wape']:.2f}%")
                col4.metric("R²", f"{val['r2']:.4f}")




        else:
            try:
                detail = response.json().get(
                    "detail", response.text
                )
            except ValueError:
                detail = response.text

            st.error(f"Помилка навчання: {detail}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Не вдалося підключитися до FastAPI. "
            "Перевір, чи запущений сервер на порту 8000."
        )

    except requests.exceptions.Timeout:
        st.error(
            "Час очікування відповіді API вичерпано. "
            "Перевір стан навчання на сервері."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Помилка запиту: {error}")


# Графіки процесу навчання моделей
st.divider()
st.subheader("Графіки процесу навчання")

st.caption(
    "Графіки показують зміну помилки моделі "
    "протягом навчання."
)

learning_curves = {
    "LightGBM": "lightgbm_learning_curve.html",
    "Keras MLP": "mlp_learning_curve.html"
}

selected_curve = st.selectbox(
    "Оберіть модель для перегляду графіка навчання",
    options=list(learning_curves.keys())
)

curve_path = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / learning_curves[selected_curve]
)

if curve_path.is_file():
    import streamlit.components.v1 as components

    components.html(
        curve_path.read_text(encoding="utf-8"),
        height=550,
        scrolling=True
    )
else:
    st.warning("Графік навчання для цієї моделі не знайдено.")
