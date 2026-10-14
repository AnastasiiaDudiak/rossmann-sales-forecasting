from pathlib import Path

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

from streamlit_app.config import get_api_url
API_URL = get_api_url()

st.set_page_config(
    page_title="Forecasting | Rossmann",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting")
st.write(
    "Прогнозування продажів магазинів Rossmann "
    "за допомогою навчених моделей машинного навчання."
)

st.divider()

# Перевірка підключення до FastAPI
try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )
    health_response.raise_for_status()

except requests.RequestException:
    st.error(
        "Не вдалося підключитися до FastAPI. "
        "Перевірте, чи запущено сервер."
    )
    st.stop()

# Отримання збережених моделей
try:
    response = requests.get(
        f"{API_URL}/models",
        timeout=10
    )
    response.raise_for_status()

    models = response.json().get("models", [])

except (requests.RequestException, ValueError) as error:
    st.error(f"Помилка завантаження моделей: {error}")
    st.stop()

st.subheader("Налаштування прогнозування")

if not models:
    st.warning(
        "Немає збережених моделей. "
        "Спочатку навчіть модель на сторінці Model Training."
    )
    st.stop()

model_names = [model["name"] for model in models]

selected_model = st.selectbox(
    "Оберіть модель для прогнозування",
    options=model_names
)

selected_info = next(
    model for model in models
    if model["name"] == selected_model
)

st.caption(
    f"Алгоритм: {selected_info['type']}"
)

st.success("Моделі успішно завантажено з FastAPI.")


# Вибір магазину та періоду прогнозування
st.divider()
st.subheader("Магазин і період прогнозування")

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "rossm_test_df.csv"
)

@st.cache_data
def load_forecast_data():
    return pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"]
    )

forecast_data = load_forecast_data()

available_stores = sorted(
    forecast_data["Store"].dropna().unique().astype(int)
)

selected_store = st.selectbox(
    "Оберіть магазин",
    options=available_stores,
    format_func=lambda store: f"Магазин №{store}"
)

store_data = forecast_data[
    forecast_data["Store"] == selected_store
].copy()

min_date = store_data["Date"].min().date()
max_date = store_data["Date"].max().date()

selected_dates = st.date_input(
    "Оберіть період прогнозування",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    selected_data = store_data[
        store_data["Date"].dt.date.between(
            start_date,
            end_date
        )
    ]

    st.caption(
        f"Магазин №{selected_store} | "
        f"Період: {start_date} — {end_date} | "
        f"Кількість днів із даними: {len(selected_data)}"
    )
else:
    st.info("Оберіть початкову та кінцеву дати.")


# Побудова прогнозу через FastAPI
st.divider()
st.subheader("Результати прогнозування")

if st.button("📈 Побудувати прогноз", type="primary", use_container_width=True):

    if len(selected_dates) != 2:
        st.warning("Оберіть початкову та кінцеву дати.")
    else:
        request_data = {
            "model_name": selected_model,
            "stores": [int(selected_store)],
            "start_date": str(start_date),
            "end_date": str(end_date)
        }

        try:
            with st.spinner("Модель виконує прогнозування..."):
                response = requests.post(
                    f"{API_URL}/forecast",
                    json=request_data,
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()

            forecasts = pd.DataFrame(result["forecasts"])

            if forecasts.empty:
                st.warning("Для вибраного періоду немає прогнозів.")
            else:
                forecasts["date"] = pd.to_datetime(forecasts["date"])
                forecasts = forecasts.sort_values("date")

                st.success("Прогноз успішно побудовано!")

                st.subheader("Фактичні та прогнозовані продажі")

                import plotly.graph_objects as go

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=forecasts["date"],
                    y=forecasts["actual_sales"],
                    mode="lines+markers",
                    name="Фактичні продажі"
                ))

                fig.add_trace(go.Scatter(
                    x=forecasts["date"],
                    y=forecasts["predicted_sales"],
                    mode="lines+markers",
                    name="Прогноз моделі"
                ))

                fig.update_layout(
                    xaxis_title="Дата",
                    yaxis_title="Продажі",
                    hovermode="x unified",
                    height=450
                )

                st.plotly_chart(fig, use_container_width=True)

                metrics = result.get("metrics")

                if metrics:
                    st.subheader("Метрики точності прогнозу")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("MAE", f"{metrics['mae']:.2f}")
                    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
                    col3.metric("WAPE", f"{metrics['wape']:.2f}%")
                    col4.metric("R²", f"{metrics['r2']:.4f}")

                st.subheader("Дані прогнозування")

                st.dataframe(
                    forecasts,
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = forecasts.to_csv(index=False).encode("utf-8-sig")

                st.download_button(
                    label="📥 Завантажити прогноз (CSV)",
                    data=csv_data,
                    file_name=f"forecast_store_{selected_store}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except requests.exceptions.RequestException as error:
            st.error(f"Помилка запиту до FastAPI: {error}")
        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Помилка обробки результатів: {error}")


# Сценарний аналіз
st.divider()
st.subheader("Сценарний аналіз")
st.write(
    "Порівняйте прогноз продажів без акції та з акцією "
    "за однакових інших умов."
)

col1, col2 = st.columns(2)

with col1:
    scenario_open = st.selectbox(
        "Робота магазину",
        options=[1, 0],
        format_func=lambda value: (
            "Відкритий" if value == 1 else "Зачинений"
        )
    )

    scenario_school = st.selectbox(
        "Шкільні канікули",
        options=[0, 1],
        format_func=lambda value: (
            "Ні" if value == 0 else "Так"
        )
    )

with col2:
    scenario_holiday = st.selectbox(
        "Державне свято",
        options=["0", "a", "b", "c"],
        format_func=lambda value: (
            "Немає" if value == "0" else f"Тип {value}"
        )
    )

if st.button(
    "Порівняти сценарії",
    type="primary",
    use_container_width=True
):
    if len(selected_dates) != 2:
        st.warning("Оберіть початкову та кінцеву дати.")
    else:
        base_request = {
            "model_name": selected_model,
            "stores": [int(selected_store)],
            "start_date": str(start_date),
            "end_date": str(end_date),
            "school_holiday": scenario_school,
            "state_holiday": scenario_holiday,
            "open": scenario_open
        }

        try:
            with st.spinner("Розрахунок сценаріїв..."):
                responses = {}

                for promo_value in [0, 1]:
                    payload = {
                        **base_request,
                        "promo": promo_value
                    }

                    response = requests.post(
                        f"{API_URL}/forecast/scenario",
                        json=payload,
                        timeout=120
                    )
                    response.raise_for_status()

                    responses[promo_value] = pd.DataFrame(
                        response.json()["forecasts"]
                    )

            without_promo = responses[0].rename(
                columns={
                    "predicted_sales": "Без акції"
                }
            )

            with_promo = responses[1].rename(
                columns={
                    "predicted_sales": "З акцією"
                }
            )

            comparison = without_promo.merge(
                with_promo[
                    ["store", "date", "З акцією"]
                ],
                on=["store", "date"],
                how="inner",
                validate="one_to_one"
            )

            if comparison.empty:
                st.warning("Немає даних для порівняння.")
            else:
                comparison["date"] = pd.to_datetime(
                    comparison["date"]
                )

                comparison = comparison.sort_values("date")

                comparison["Різниця"] = (
                    comparison["З акцією"]
                    - comparison["Без акції"]
                )

                total_without = comparison["Без акції"].sum()
                total_with = comparison["З акцією"].sum()
                difference = total_with - total_without

                st.success("Сценарії успішно розраховано!")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Продажі без акції",
                    f"{total_without:,.0f}"
                )
                c2.metric(
                    "Продажі з акцією",
                    f"{total_with:,.0f}"
                )
                c3.metric(
                    "Різниця прогнозів",
                    f"{difference:+,.0f}"
                )

                import plotly.graph_objects as go

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=comparison["date"],
                    y=comparison["Без акції"],
                    mode="lines+markers",
                    name="Без акції"
                ))

                fig.add_trace(go.Scatter(
                    x=comparison["date"],
                    y=comparison["З акцією"],
                    mode="lines+markers",
                    name="З акцією"
                ))

                fig.update_layout(
                    title="Порівняння сценаріїв Promo",
                    xaxis_title="Дата",
                    yaxis_title="Прогнозовані продажі",
                    hovermode="x unified",
                    height=450
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = comparison.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "Завантажити порівняння (CSV)",
                    data=csv_data,
                    file_name="scenario_comparison.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.caption(
                    "Різниця показує зміну прогнозу моделі "
                    "за різних значень Promo. "
                    "Це не доводить причинний вплив акції."
                )

        except requests.exceptions.RequestException as error:
            st.error(f"Помилка FastAPI: {error}")
        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Помилка обробки даних: {error}")
