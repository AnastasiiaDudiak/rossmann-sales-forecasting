
import streamlit as st
import requests
from streamlit_app.config import get_api_url

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Rossmann Sales Forecasting")
st.subheader("Інтерактивна система прогнозування продажів")

st.markdown("""
Система призначена для дослідження історичних продажів
магазинів Rossmann, навчання моделей машинного навчання,
порівняння їхньої точності та прогнозування продажів.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Магазинів", "100")

with col2:
    st.metric("Train", "70 056 записів")

with col3:
    st.metric("Test", "21 200 записів")

st.divider()

st.subheader("Можливості системи")

st.markdown("""
**1. Data Exploration** — аналіз даних, статистика та графіки.

**2. Model Training** — навчання Random Forest, LightGBM і Keras MLP.

**3. Model Comparison** — порівняння результатів моделей.

**4. Forecasting** — прогнозування продажів і аналіз сценаріїв.
""")

st.divider()

st.subheader("Стан підключення до FastAPI")

try:
    response = requests.get(
        f"{get_api_url()}/health",
        timeout=5
    )
    response.raise_for_status()

    st.success("FastAPI працює. Підключення успішне!")

except requests.RequestException:
    st.error(
        "Немає підключення до FastAPI. "
        "Перевір, чи запущений сервер на порту 8000."
    )

st.caption(
    "Навчальний проєкт — Rossmann Sales Forecasting Demo Pipeline"
)
