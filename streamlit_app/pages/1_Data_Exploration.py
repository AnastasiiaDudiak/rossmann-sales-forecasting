
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Data Exploration | Rossmann",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@st.cache_data
def load_eda_data():
    train = pd.read_csv(
        DATA_DIR / "rossm_train_df.csv",
        parse_dates=["Date"]
    )

    test = pd.read_csv(
        DATA_DIR / "rossm_test_df.csv",
        parse_dates=["Date"]
    )

    train["Dataset"] = "Train"
    test["Dataset"] = "Test"

    return pd.concat([train, test], ignore_index=True)


st.title("📊 Data Exploration")
st.write("Дослідження історичних продажів магазинів Rossmann")

df = load_eda_data()

# Фільтри
st.sidebar.header("Фільтри EDA")

selected_stores = st.sidebar.multiselect(
    "Магазини",
    options=sorted(df["Store"].unique().tolist()),
    default=sorted(df["Store"].unique().tolist())
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Період",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered = df[df["Store"].isin(selected_stores)].copy()

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered = filtered[
        filtered["Date"].between(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date)
        )
    ]

# Основні показники
col1, col2, col3, col4 = st.columns(4)

col1.metric("Записів", f"{len(filtered):,}")
col2.metric("Магазинів", filtered["Store"].nunique())
col3.metric("Сумарні продажі", f"{filtered['Sales'].sum():,.0f}")
col4.metric("Середні продажі", f"{filtered['Sales'].mean():,.2f}")

st.divider()

st.subheader("Інформація про датасет")

st.write("Розмір повного набору:", df.shape)
st.write("Розмір після фільтрації:", filtered.shape)
st.write("Період даних:", min_date, "—", max_date)

st.subheader("Приклад даних")
st.dataframe(filtered.head(20), use_container_width=True)

st.subheader("Типи даних")
st.dataframe(
    pd.DataFrame({
        "Колонка": df.columns,
        "Тип даних": df.dtypes.astype(str).values
    }),
    use_container_width=True
)


# Розподіл продажів
import numpy as np
import plotly.express as px

st.divider()
st.subheader("Розподіл продажів")

# Для аналізу розподілу використовуємо лише відкриті магазини
open_data = filtered[filtered["Open"] == 1].copy()
open_data["logSales"] = np.log1p(open_data["Sales"])

if open_data.empty:
    st.warning("Немає даних про відкриті магазини для вибраних фільтрів.")
else:
    col1, col2 = st.columns(2)

    with col1:
        fig_sales = px.histogram(
            open_data,
            x="Sales",
            nbins=60,
            title="Розподіл щоденних продажів",
            labels={
                "Sales": "Продажі",
                "count": "Кількість днів"
            }
        )

        fig_sales.update_layout(
            xaxis_title="Продажі",
            yaxis_title="Кількість спостережень"
        )

        st.plotly_chart(fig_sales, use_container_width=True)

    with col2:
        fig_log = px.histogram(
            open_data,
            x="logSales",
            nbins=60,
            title="Розподіл logSales",
            labels={
                "logSales": "log(1 + Sales)"
            }
        )

        fig_log.update_layout(
            xaxis_title="log(1 + Sales)",
            yaxis_title="Кількість спостережень"
        )

        st.plotly_chart(fig_log, use_container_width=True)


# Динаміка продажів у часі
st.divider()
st.subheader("Динаміка продажів у часі")

period = st.radio(
    "Деталізація графіка",
    ["День", "Тиждень", "Місяць"],
    horizontal=True
)

sales_data = filtered.copy()
sales_data["Date"] = pd.to_datetime(sales_data["Date"])

if period == "День":
    sales_data["Period"] = sales_data["Date"]

elif period == "Тиждень":
    sales_data["Period"] = (
        sales_data["Date"]
        .dt.to_period("W")
        .dt.start_time
    )

else:
    sales_data["Period"] = (
        sales_data["Date"]
        .dt.to_period("M")
        .dt.start_time
    )

sales_by_period = (
    sales_data
    .groupby("Period", as_index=False)["Sales"]
    .sum()
    .sort_values("Period")
)

fig_time = px.line(
    sales_by_period,
    x="Period",
    y="Sales",
    title=f"Сумарні продажі за періодами: {period.lower()}",
    markers=True if period == "Місяць" else False
)

fig_time.update_layout(
    xaxis_title="Дата",
    yaxis_title="Сумарні продажі",
    hovermode="x unified"
)

st.plotly_chart(fig_time, use_container_width=True)


# Продажі окремого магазину
st.divider()
st.subheader("Динаміка продажів окремого магазину")

available_stores = sorted(filtered["Store"].unique().tolist())

if available_stores:
    selected_store = st.selectbox(
        "Оберіть магазин для аналізу",
        available_stores
    )

    store_data = (
        filtered[filtered["Store"] == selected_store]
        .sort_values("Date")
        .copy()
    )

    fig_store = px.line(
        store_data,
        x="Date",
        y="Sales",
        title=f"Щоденні продажі магазину №{selected_store}"
    )

    fig_store.update_layout(
        xaxis_title="Дата",
        yaxis_title="Продажі",
        hovermode="x unified"
    )

    st.plotly_chart(fig_store, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Сумарні продажі",
        f"{store_data['Sales'].sum():,.0f}"
    )

    col2.metric(
        "Середні щоденні продажі",
        f"{store_data['Sales'].mean():,.2f}"
    )

    col3.metric(
        "Кількість робочих днів",
        int((store_data["Open"] == 1).sum())
    )

else:
    st.info("Немає магазинів для вибраних фільтрів.")


# ТОП-10 магазинів за обсягом продажів
st.divider()
st.subheader("ТОП-10 магазинів за сумарними продажами")

# Беремо всі магазини, але тільки за дати з поточного фільтра
top_data = df[
    (df["Date"] >= filtered["Date"].min()) &
    (df["Date"] <= filtered["Date"].max())
].copy()

top_stores = (
    top_data.groupby("Store", as_index=False)["Sales"]
    .sum()
    .nlargest(10, "Sales")
    .sort_values("Sales")
)

top_stores["Store"] = (
    "Магазин №" + top_stores["Store"].astype(str)
)

fig_top = px.bar(
    top_stores,
    x="Sales",
    y="Store",
    orientation="h",
    text="Sales",
    title="10 магазинів із найбільшими продажами"
)

fig_top.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="inside",
    insidetextanchor="end"
)

fig_top.update_layout(
    xaxis_title="Сумарні продажі",
    yaxis_title="",
    yaxis_type="category",
    height=550,
    bargap=0.25,
    uniformtext_minsize=11,
    uniformtext_mode="hide"
)

st.plotly_chart(fig_top, use_container_width=True)


# --------------------------------------------------
# Продажі за днями тижня та вплив акцій Promo
# --------------------------------------------------

st.divider()
st.header("Продажі за днями тижня та вплив акцій")

# Використовуємо дані після фільтрації магазинів і дат
# Аналізуємо лише дні, коли магазини працювали
promo_df = filtered[
    (filtered["Open"] == 1) &
    (filtered["Sales"] > 0)
].copy()

weekday_names = {
    0: "Пн",
    1: "Вт",
    2: "Ср",
    3: "Чт",
    4: "Пт",
    5: "Сб",
    6: "Нд"
}

promo_df["День тижня"] = (
    promo_df["Date"].dt.dayofweek.map(weekday_names)
)

promo_df["Акція"] = promo_df["Promo"].map({
    0: "Без акції",
    1: "З акцією"
})

if promo_df.empty:
    st.info("Для вибраних фільтрів немає даних про продажі.")
else:
    fig = px.box(
        promo_df,
        x="День тижня",
        y="Sales",
        color="Акція",
        category_orders={
            "День тижня": [
                "Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"
            ],
            "Акція": ["Без акції", "З акцією"]
        },
        labels={
            "Sales": "Продажі",
            "День тижня": "День тижня"
        },
        title="Розподіл продажів за днями тижня та наявністю акції",
        points=False
    )

    fig.update_layout(
        height=500,
        boxmode="group",
        xaxis_title="День тижня",
        yaxis_title="Продажі"
    )

    st.plotly_chart(fig, use_container_width=True)



# --------------------------------------------------
# Аналіз продажів за типом магазину та асортиментом
# --------------------------------------------------

st.divider()
st.header("Продажі за характеристиками магазинів")

store_analysis = filtered[
    (filtered["Open"] == 1) &
    (filtered["Sales"] > 0)
].copy()

if store_analysis.empty:
    st.info("Немає даних для аналізу за вибраними фільтрами.")
else:
    col1, col2 = st.columns(2)

    # Середні продажі за типом магазину
    store_type_sales = (
        store_analysis
        .groupby("StoreType", as_index=False)["Sales"]
        .mean()
        .sort_values("Sales", ascending=False)
    )

    fig_type = px.bar(
        store_type_sales,
        x="StoreType",
        y="Sales",
        text="Sales",
        title="Середні продажі за типом магазину",
        labels={
            "StoreType": "Тип магазину",
            "Sales": "Середні продажі"
        }
    )

    fig_type.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    fig_type.update_layout(
        height=450,
        showlegend=False
    )

    with col1:
        st.plotly_chart(fig_type, use_container_width=True)

    # Середні продажі за категорією асортименту
    assortment_sales = (
        store_analysis
        .groupby("Assortment", as_index=False)["Sales"]
        .mean()
        .sort_values("Sales", ascending=False)
    )

    fig_assortment = px.bar(
        assortment_sales,
        x="Assortment",
        y="Sales",
        text="Sales",
        title="Середні продажі за асортиментом",
        labels={
            "Assortment": "Категорія асортименту",
            "Sales": "Середні продажі"
        }
    )

    fig_assortment.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    fig_assortment.update_layout(
        height=450,
        showlegend=False
    )

    with col2:
        st.plotly_chart(fig_assortment, use_container_width=True)



# --------------------------------------------------
# Сезонність продажів та вплив свят
# --------------------------------------------------

st.divider()
st.header("Сезонність продажів та святкові періоди")

season_df = filtered[
    (filtered["Open"] == 1) &
    (filtered["Sales"] > 0)
].copy()

if season_df.empty:
    st.info("Для вибраних фільтрів немає даних.")
else:
    # 1. Середні продажі за місяцями
    season_df["Місяць"] = season_df["Date"].dt.month

    monthly_sales = (
        season_df
        .groupby("Місяць", as_index=False)["Sales"]
        .mean()
        .sort_values("Місяць")
    )

    month_names = {
        1: "Січ", 2: "Лют", 3: "Бер", 4: "Кві",
        5: "Тра", 6: "Чер", 7: "Лип", 8: "Сер",
        9: "Вер", 10: "Жов", 11: "Лис", 12: "Гру"
    }

    monthly_sales["Назва місяця"] = (
        monthly_sales["Місяць"].map(month_names)
    )

    fig_month = px.line(
        monthly_sales,
        x="Назва місяця",
        y="Sales",
        markers=True,
        category_orders={
            "Назва місяця": list(month_names.values())
        },
        labels={
            "Назва місяця": "Місяць",
            "Sales": "Середні продажі"
        },
        title="Середні денні продажі за місяцями"
    )

    fig_month.update_layout(height=450)
    st.plotly_chart(fig_month, use_container_width=True)

    # 2. Державні свята
    holiday_names = {
        "0": "Звичайний день",
        "a": "Державне свято",
        "b": "Великоднє свято",
        "c": "Різдвяне свято"
    }

    season_df["Тип дня"] = (
        season_df["StateHoliday"]
        .astype(str)
        .map(holiday_names)
        .fillna("Інше")
    )

    state_sales = (
        season_df
        .groupby("Тип дня", as_index=False)
        .agg(
            average_sales=("Sales", "mean"),
            observations=("Sales", "size")
        )
    )

    # 3. Шкільні канікули
    season_df["Шкільні канікули"] = (
        season_df["SchoolHoliday"].map({
            0: "Ні",
            1: "Так"
        })
    )

    school_sales = (
        season_df
        .groupby("Шкільні канікули", as_index=False)
        .agg(
            average_sales=("Sales", "mean"),
            observations=("Sales", "size")
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_state = px.bar(
            state_sales,
            x="Тип дня",
            y="average_sales",
            text="average_sales",
            title="Продажі за типом святкового дня",
            labels={
                "Тип дня": "Тип дня",
                "average_sales": "Середні продажі"
            },
            custom_data=["observations"]
        )

        fig_state.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "Середні продажі: %{y:,.0f}<br>"
                "Спостережень: %{customdata[0]}<extra></extra>"
            )
        )

        fig_state.update_layout(
            height=480,
            xaxis_tickangle=-20
        )

        st.plotly_chart(fig_state, use_container_width=True)

    with col2:
        fig_school = px.bar(
            school_sales,
            x="Шкільні канікули",
            y="average_sales",
            text="average_sales",
            title="Продажі під час шкільних канікул",
            labels={
                "average_sales": "Середні продажі"
            },
            custom_data=["observations"]
        )

        fig_school.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "Середні продажі: %{y:,.0f}<br>"
                "Спостережень: %{customdata[0]}<extra></extra>"
            )
        )

        fig_school.update_layout(height=480)

        st.plotly_chart(fig_school, use_container_width=True)



# ============================================================
# Зв'язок продажів із покупцями та відстанню до конкурентів
# ============================================================

st.divider()
st.header("Фактори, пов'язані з продажами")

analysis_df = filtered[
    (filtered["Date"] >= pd.Timestamp(selected_dates[0])) &
    (filtered["Date"] <= pd.Timestamp(selected_dates[1])) &
    (filtered["Open"] == 1) &
    (filtered["Sales"] > 0)
].copy()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Продажі та кількість покупців")

    fig_customers = px.scatter(
        analysis_df,
        x="Customers",
        y="Sales",
        opacity=0.35,
        labels={
            "Customers": "Кількість покупців",
            "Sales": "Продажі"
        },
        title="Залежність продажів від кількості покупців"
    )

    fig_customers.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_customers,
        use_container_width=True
    )

with col2:
    st.subheader("Продажі та відстань до конкурентів")

    competition_df = analysis_df.dropna(
        subset=["CompetitionDistance"]
    )

    fig_competition = px.scatter(
        competition_df,
        x="CompetitionDistance",
        y="Sales",
        opacity=0.35,
        labels={
            "CompetitionDistance": "Відстань до конкурента",
            "Sales": "Продажі"
        },
        title="Продажі залежно від відстані до конкурента"
    )

    fig_competition.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_competition,
        use_container_width=True
    )

st.info(
    "Графіки показують зв'язки між показниками, "
    "але самі по собі не доводять причинно-наслідковий вплив. "
    "Кількість покупців використовується лише для EDA, "
    "оскільки до завершення торгового дня вона невідома."
)
