from pathlib import Path

import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="Model Comparison | Rossmann",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Comparison")

st.markdown(
    "Порівняння моделей прогнозування продажів Rossmann "
    "за якістю прогнозування та параметрами навчання."
)

st.divider()

from streamlit_app.config import get_api_url
API_URL = get_api_url()


@st.cache_data(ttl=30)
def load_models():
    response = requests.get(
        f"{API_URL}/models",
        timeout=15
    )
    response.raise_for_status()
    return response.json()["models"]


@st.cache_data(ttl=30)
def load_model_details(name):
    response = requests.get(
        f"{API_URL}/models/{name}",
        timeout=15
    )
    response.raise_for_status()
    return response.json()


try:
    models = load_models()

except requests.exceptions.RequestException:
    st.error(
        "Не вдалося отримати список моделей. "
        "Перевір, чи працює FastAPI."
    )
    st.stop()


if not models:
    st.warning("Збережених моделей поки немає.")
    st.stop()


model_names = [model["name"] for model in models]

BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "baseline_metrics.json"
)

if BASELINE_PATH.is_file():
    model_names.append("Store Mean Baseline")

st.subheader("Вибір моделей")

selected_models = st.multiselect(
    "Оберіть моделі для порівняння",
    options=model_names,
    default=model_names[:3],
    help="Можна порівнювати різні алгоритми та конфігурації."
)

if not selected_models:
    st.info("Оберіть хоча б одну модель.")
    st.stop()


# Завантаження інформації про вибрані моделі
details = []

for name in selected_models:
    if name == "Store Mean Baseline":
        continue

    try:
        details.append(load_model_details(name))
    except requests.exceptions.RequestException:
        st.warning(f"Не вдалося завантажити модель: {name}")


if not details and "Store Mean Baseline" not in selected_models:
    st.stop()


st.divider()

period = st.selectbox(
    "Оберіть набір даних для оцінювання",
    options=["Validation", "Test", "Train"],
    index=0,
    help="Validation використовується для налаштування моделей, "
         "Test — для фінальної оцінки, Train — для аналізу навчання."
)

period_key = period.lower()

st.subheader(f"Порівняння метрик {period}")

rows = []

for model in details:
    metrics = model.get("metrics", {}).get(period_key)

    if metrics is None:
        st.warning(
            f"Для моделі {model['model_name']} "
            f"немає метрик {period}."
        )
        continue

    rows.append({
        "Модель": model["model_name"],
        "Тип": model["model_type"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "WAPE (%)": metrics["wape"],
        "R²": metrics["r2"],
        "Log RMSE": metrics["log_rmse"]
    })

# Додаємо Baseline для вибраного періоду
import json

if (
    "Store Mean Baseline" in selected_models
    and BASELINE_PATH.is_file()
):
    with open(BASELINE_PATH, "r", encoding="utf-8") as file:
        baseline_data = json.load(file)

    baseline_metrics = baseline_data["metrics"][period_key]

    rows.append({
        "Модель": "Store Mean Baseline",
        "Тип": "Baseline",
        "MAE": baseline_metrics["mae"],
        "RMSE": baseline_metrics["rmse"],
        "WAPE (%)": baseline_metrics["wape"],
        "R²": baseline_metrics["r2"],
        "Log RMSE": None
    })

comparison_df = pd.DataFrame(rows)

st.dataframe(
    comparison_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "MAE": st.column_config.NumberColumn(format="%.2f"),
        "RMSE": st.column_config.NumberColumn(format="%.2f"),
        "WAPE (%)": st.column_config.NumberColumn(format="%.2f"),
        "R²": st.column_config.NumberColumn(format="%.4f"),
        "Log RMSE": st.column_config.NumberColumn(format="%.4f")
    }
)

st.caption(
    "Для MAE, RMSE, WAPE та Log RMSE менше значення "
    "означає кращий результат. Для R² — більше."
)


# Графічне порівняння моделей
st.divider()
st.subheader("Графічне порівняння")

metric_options = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "WAPE (%)": "WAPE (%)",
    "R²": "R²",
    "Log RMSE": "Log RMSE"
}

selected_metric = st.selectbox(
    "Оберіть метрику для порівняння",
    options=list(metric_options.keys()),
    index=2
)

metric_column = metric_options[selected_metric]

chart_df = comparison_df[
    ["Модель", metric_column]
].sort_values(
    by=metric_column,
    ascending=(selected_metric != "R²")
)

import plotly.express as px

fig = px.bar(
    chart_df,
    x="Модель",
    y=metric_column,
    text=metric_column,
    title=f"Порівняння моделей за метрикою {selected_metric}"
)

fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Модель",
    yaxis_title=selected_metric,
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

best_model = chart_df.iloc[
    -1 if selected_metric == "R²" else 0
]

st.info(
    f"Найкраща модель за метрикою {selected_metric}: "
    f"{best_model['Модель']}"
)


# Перегляд конфігурації вибраної моделі
st.divider()
st.subheader("Конфігурація моделі")

selected_detail = st.selectbox(
    "Оберіть модель для перегляду параметрів",
    options=[model["model_name"] for model in details]
)

model_detail = next(
    model for model in details
    if model["model_name"] == selected_detail
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Алгоритм", model_detail["model_type"])

with col2:
    st.metric(
        "Кількість ознак",
        len(model_detail.get("features", []))
    )

with col3:
    training_time = model_detail.get("training_time_seconds")

    st.metric(
        "Час навчання",
        f"{training_time:.2f} с"
        if training_time is not None
        else "Немає даних"
    )

st.markdown("#### Гіперпараметри")

parameters = model_detail.get("parameters") or {}

if parameters:
    parameter_df = pd.DataFrame([
        {
            "Параметр": key,
            "Значення": str(value)
        }
        for key, value in parameters.items()
    ])

    st.dataframe(
        parameter_df,
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("Інформація про гіперпараметри відсутня.")

st.markdown("#### Ознаки моделі")

features = model_detail.get("features", [])

if features:
    st.write(", ".join(features))
else:
    st.info("Інформація про ознаки відсутня.")


# Аналіз важливості ознак
st.divider()
st.subheader("Feature Importance — важливість ознак")

st.caption(
    "Аналіз показує, які ознаки модель використовує "
    "найбільше. Значення не є причинним впливом на продажі."
)

import joblib
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
API_DIR = PROJECT_DIR / "api"

if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import importlib.util

feature_file = PROJECT_DIR / "api" / "feature_importance.py"

spec = importlib.util.spec_from_file_location(
    "rossmann_feature_importance",
    feature_file
)

feature_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_module)

calculate_feature_importance = (
    feature_module.calculate_feature_importance
)

tree_models = [
    model for model in details
    if model.get("model_type") in [
        "Random Forest",
        "RandomForestRegressor",
        "LightGBM",
        "LGBMRegressor"
    ]
]

if not tree_models:
    st.info(
        "Для аналізу важливості ознак виберіть "
        "Random Forest або LightGBM у списку моделей вище."
    )
else:
    importance_model_name = st.selectbox(
        "Оберіть модель для аналізу ознак",
        options=[
            model["model_name"] for model in tree_models
        ]
    )

    selected_info = next(
        model for model in tree_models
        if model["model_name"] == importance_model_name
    )

    model_path = (
        PROJECT_DIR
        / "models"
        / importance_model_name
        / "model.joblib"
    )

    if model_path.exists():
        pipeline = joblib.load(model_path)

        importance_df = calculate_feature_importance(
            pipeline,
            selected_info["features"]
        )

        fig_importance = px.bar(
            importance_df.sort_values("Importance (%)"),
            x="Importance (%)",
            y="Feature",
            orientation="h",
            text="Importance (%)",
            title="Важливість початкових ознак"
        )

        fig_importance.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig_importance.update_layout(
            height=520,
            xaxis_title="Відносна важливість (%)",
            yaxis_title="Ознака"
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

        top_feature = importance_df.iloc[0]

        st.info(
            f"Найважливіша ознака: "
            f"{top_feature['Feature']} "
            f"({top_feature['Importance (%)']:.2f}%)."
        )

    else:
        st.warning(
            "Файл навченої моделі не знайдено."
        )
