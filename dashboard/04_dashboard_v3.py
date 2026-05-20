# -*- coding: utf-8 -*-
"""
04_dashboard.py — Дашборд раннего предупреждения нестабильности воздушного движения (v3)

Запуск:
    streamlit run 04_dashboard.py

Файлы данных (в той же директории или укажите DATA_DIR):
    - dashboard_flight_summary.parquet  (~1.8 МБ, 29.8K рейсов)
    - dashboard_points_v3.parquet       (~3.2 ГБ, 62.6M точек)
    - events_v3.parquet                 (~2.6 МБ, 26K событий)
    - evaluation_report_v3.json         (опционально)

Зависимости:
    pip install streamlit pandas plotly pyarrow
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyarrow.parquet as pq
import os
import json
import time

# ===== КОНФИГУРАЦИЯ =====
DATA_DIR = "."

SUMMARY_FILE = os.path.join(DATA_DIR, "dashboard_flight_summary.parquet")
POINTS_FILE = os.path.join(DATA_DIR, "dashboard_points_v3.parquet")
EVENTS_FILE = os.path.join(DATA_DIR, "events_v3.parquet")
REPORT_FILE = os.path.join(DATA_DIR, "evaluation_report_v3.json")

# Пороги уровней риска (из 03.5 - калибровка по ECDF чистой calval-выборки)
RISK_LOW_MAX = 0.90       # < 0.90 = низкий уровень
RISK_MEDIUM_MAX = 0.99    # 0.90-0.99 = средний уровень, >= 0.99 = высокий уровень

# Цветовые схемы
RISK_LEVEL_COLORS = {
    "low": "#2ecc71",
    "medium": "#f39c12",
    "high": "#e74c3c",
    "unknown": "#95a5a6",
}

MAP_TRIAGE_ORDER = [
    "no_events",
    "event_low",
    "event_medium",
    "event_high",
    "event_top",
]

MAP_TRIAGE_LABELS = {
    "no_events": "Без событий",
    "event_low": "Низкий приоритет",
    "event_medium": "Средний приоритет",
    "event_high": "Высокий приоритет",
    "event_top": "Верхние 5%",
}

MAP_TRIAGE_COLORS = {
    "Без событий": "#2fbf71",
    "Низкий приоритет": "#9bdc65",
    "Средний приоритет": "#f3c64a",
    "Высокий приоритет": "#f28e2b",
    "Верхние 5%": "#d62728",
}

MODEL_COLORS = {
    "IF": "#FF6B6B",
    "HDBSCAN": "#4ECB71",
    "LSTM-AE": "#9B59B6",
}

CATEGORY_COLORS = {
    "potential_operational_anomaly": "#e74c3c",
    "mixed_or_derived_dynamics": "#f39c12",
    "likely_data_quality_artifact": "#3498db",
    "likely_feature_phase_artifact": "#9b59b6",
}

CATEGORY_LABELS = {
    "potential_operational_anomaly": "Потенциальная операционная аномалия",
    "mixed_or_derived_dynamics": "Смешанная / производная динамика",
    "likely_data_quality_artifact": "Артефакт данных",
    "likely_feature_phase_artifact": "Артефакт фазы/признаков",
}

PHASE_LABELS = {
    "unknown": "Неизвестная",
    "ground": "На земле",
    "takeoff": "Взлёт",
    "climb": "Набор высоты",
    "cruise": "Крейсер",
    "descent": "Снижение",
    "approach": "Заход",
    "landing": "Посадка",
}

SPLIT_LABELS = {
    "calibration_val": "Калибровка",
    "test": "Тест",
}


def add_map_triage_columns(map_df, priority_metric):
    """Добавляет UX-группы для карты: обзорная шкала приоритета вместо сырого хвоста риска."""
    result = map_df.copy()
    result["map_triage_key"] = "no_events"
    result["map_priority_pct"] = np.nan

    if priority_metric in result.columns:
        priority = pd.to_numeric(result[priority_metric], errors="coerce")
    else:
        priority = pd.Series(np.nan, index=result.index)

    if "n_events" in result.columns:
        has_events = pd.to_numeric(result["n_events"], errors="coerce").fillna(0) > 0
    else:
        # Если числа событий нет, считаем рейсы с непустой метрикой кандидатами в очередь.
        has_events = priority.notna()

    event_priority = priority[has_events]
    if len(event_priority) > 0:
        fallback = event_priority.median()
        if pd.isna(fallback):
            fallback = 0.0
        event_priority = event_priority.fillna(fallback)
        result.loc[has_events, "map_priority_pct"] = (
            event_priority.rank(pct=True, method="average") * 100
        )

        result.loc[has_events, "map_triage_key"] = "event_low"
        result.loc[
            has_events & (result["map_priority_pct"] >= 50),
            "map_triage_key",
        ] = "event_medium"
        result.loc[
            has_events & (result["map_priority_pct"] >= 80),
            "map_triage_key",
        ] = "event_high"
        result.loc[
            has_events & (result["map_priority_pct"] >= 95),
            "map_triage_key",
        ] = "event_top"

    result["map_triage_label"] = pd.Categorical(
        result["map_triage_key"].map(MAP_TRIAGE_LABELS),
        categories=[MAP_TRIAGE_LABELS[key] for key in MAP_TRIAGE_ORDER],
        ordered=True,
    )
    return result


def get_map_color_range(metric_values, color_metric):
    """Возвращает устойчивый диапазон цвета для сырой непрерывной шкалы."""
    metric_finite = metric_values.dropna()
    if len(metric_finite) == 0:
        return [0, 1.0]

    if color_metric == "n_events":
        low = 0.0
        high = float(metric_finite.quantile(0.95))
    elif color_metric == "max_event_risk":
        # У риска событий хвост очень плотный, поэтому оставляем только устойчивые квантили.
        low = float(metric_finite.quantile(0.05))
        high = float(metric_finite.quantile(0.99))
    else:
        # Для ranking_score и DQ это диагностическая шкала, а не статус безопасности.
        low = float(metric_finite.quantile(0.05))
        high = float(metric_finite.quantile(0.95))

    if not np.isfinite(low) or not np.isfinite(high):
        return [0, 1.0]
    if low == high:
        delta = 0.01 if high == 0 else abs(high) * 0.01
        low -= delta
        high += delta
    return [low, high]


def format_point_count(n_points):
    """Форматирует число точек для компактной подписи внутри сегмента."""
    if n_points >= 1_000_000:
        return f"{n_points / 1_000_000:.1f}M"
    if n_points >= 1_000:
        return f"{n_points / 1_000:.0f}K"
    return f"{n_points:,}"


def render_risk_level_distribution(eval_report):
    """Рисует KPI-блоки и компактную шкалу распределения точек по уровням риска."""
    dist = (
        eval_report
        .get("risk_score", {})
        .get("risk_level_distribution_all", {})
    )

    levels = ["low", "medium", "high"]
    if not all(level in dist for level in levels):
        st.info("В evaluation_report_v3.json нет risk_score.risk_level_distribution_all.")
        return

    level_labels = {
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    level_bounds = {
        "low": "risk < 0.90",
        "medium": "0.90 ≤ risk < 0.99",
        "high": "risk ≥ 0.99",
    }
    colors = {
        "low": "#27AE60",
        "medium": "#F39C12",
        "high": "#E74C3C",
    }

    counts = [int(dist[level]["n"]) for level in levels]
    pcts = [float(dist[level]["pct"]) for level in levels]
    if pcts and max(pcts) <= 1.0:
        pcts = [pct * 100 for pct in pcts]
    total = sum(counts)

    card_cols = st.columns(3)
    for card_col, level, n_points, pct in zip(card_cols, levels, counts, pcts):
        with card_col:
            st.markdown(
                f"""
                <div style="
                    border-left: 5px solid {colors[level]};
                    background: rgba(250, 250, 250, 0.75);
                    border-radius: 8px;
                    padding: 10px 12px 9px 12px;
                    min-height: 92px;
                ">
                    <div style="
                        color: #566573;
                        font-size: 12px;
                        font-weight: 700;
                        letter-spacing: 0;
                        text-transform: uppercase;
                    ">{level_labels[level]}</div>
                    <div style="
                        color: #1f2d3d;
                        font-size: 24px;
                        font-weight: 750;
                        line-height: 1.15;
                        margin-top: 4px;
                    ">{format_point_count(n_points)}</div>
                    <div style="
                        color: {colors[level]};
                        font-size: 18px;
                        font-weight: 700;
                        line-height: 1.15;
                    ">{pct:.1f}%</div>
                    <div style="
                        color: #7f8c8d;
                        font-size: 11px;
                        margin-top: 5px;
                        white-space: nowrap;
                    ">{level_bounds[level]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    fig = go.Figure()
    for level, n_points, pct in zip(levels, counts, pcts):
        fig.add_trace(go.Bar(
            y=["Все точки"],
            x=[n_points],
            orientation="h",
            name=f"{level_labels[level]} · {level_bounds[level]}",
            marker_color=colors[level],
            text=[""],
            textposition="none",
            hovertemplate=(
                f"{level}: {n_points:,} точек "
                f"({pct:.1f}%, {level_bounds[level]})"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showticklabels=False),
        height=120,
        margin=dict(l=10, r=10, t=8, b=58),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_annotation(
        text=f"На полной выборке мониторинга (calval + test, {total / 1_000_000:.1f} млн точек)",
        showarrow=False,
        x=0.5,
        y=1.25,
        xref="paper",
        yref="paper",
        font=dict(size=11, color="#7F8C8D"),
    )

    st.plotly_chart(fig, use_container_width=True)

# ===== НАСТРОЙКИ СТРАНИЦЫ =====
st.set_page_config(
    page_title="Мониторинг нестабильности ВД (v3)",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===== ЗАГРУЗКА ДАННЫХ =====
@st.cache_data
def load_flight_summary():
    """Загрузка сводки по рейсам (легковесная, всегда в памяти)."""
    df = pd.read_parquet(SUMMARY_FILE)
    return df


@st.cache_data
def load_events():
    """Загрузка таблицы событий (~2.6 МБ)."""
    df = pd.read_parquet(EVENTS_FILE)
    if "event_start_ts" in df.columns:
        df["event_start_ts"] = pd.to_datetime(df["event_start_ts"])
    if "event_end_ts" in df.columns:
        df["event_end_ts"] = pd.to_datetime(df["event_end_ts"])
    return df


@st.cache_data
def load_evaluation_report():
    """Загрузка JSON-отчёта (опционально)."""
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_flight_points(flight_id):
    """Загрузка поточечных данных одного рейса через filter (быстро благодаря row_group_size)."""
    fdata = pq.read_table(
        POINTS_FILE,
        filters=[("flight_id", "=", int(flight_id))]
    ).to_pandas()
    fdata["timestamp"] = pd.to_datetime(fdata["timestamp"])
    fdata = fdata.sort_values("timestamp").reset_index(drop=True)
    fdata["t_min"] = (
        (fdata["timestamp"] - fdata["timestamp"].min()).dt.total_seconds() / 60
    )
    return fdata


# ===== БОКОВАЯ ПАНЕЛЬ =====
def render_sidebar(df_summary):
    """Боковая панель с фильтрами рейсов."""
    st.sidebar.title("Фильтры рейсов")

    # Фильтр по подвыборке
    splits = sorted(df_summary["split"].dropna().unique().tolist())
    selected_splits = st.sidebar.multiselect(
        "Подвыборка",
        options=splits,
        default=splits,
        format_func=lambda x: SPLIT_LABELS.get(x, x),
    )

    # Фильтр по ranking_score
    st.sidebar.subheader("Ranking score")
    if "ranking_score" in df_summary.columns:
        rs_max_val = float(df_summary["ranking_score"].max())
        rs_min_val = float(df_summary["ranking_score"].min())
    else:
        rs_max_val = 1.0
        rs_min_val = 0.0

    rs_min, rs_max = st.sidebar.slider(
        "Диапазон",
        min_value=rs_min_val, max_value=rs_max_val,
        value=(rs_min_val, rs_max_val),
        step=0.01,
    )

    # Фильтр по числу событий
    n_evt_max = int(df_summary["n_events"].max()) if "n_events" in df_summary.columns else 0
    if n_evt_max > 0:
        ne_min, ne_max = st.sidebar.slider(
            "Число событий на рейс",
            min_value=0, max_value=n_evt_max,
            value=(0, n_evt_max),
        )
    else:
        ne_min, ne_max = 0, 0

    # Фильтр качества данных
    st.sidebar.subheader("Качество данных")
    only_clean = st.sidebar.checkbox(
        "Только чистые рейсы (dq_hard < 5%)",
        value=False,
        help="Скрывает рейсы с большой долей stale altitude / threshold violations",
    )

    # Метрика цвета для карты
    st.sidebar.subheader("Цвет точек на карте")
    map_color_mode_options = {
        "triage": "Группы приоритета",
        "raw": "Сырая метрика",
    }
    map_color_mode = st.sidebar.radio(
        "Режим",
        options=list(map_color_mode_options.keys()),
        format_func=lambda x: map_color_mode_options[x],
        index=0,
        horizontal=True,
        help="Группы приоритета лучше подходят для обзорной карты; сырая метрика оставлена для диагностики.",
    )

    map_color_options = {
        "ranking_score": "Ranking score (приоритет)",
        "max_event_risk": "Макс. риск события (NaN→серый)",
        "n_events": "Число событий",
        "dq_hard_share": "DQ hard %",
    }
    available_color = {k: v for k, v in map_color_options.items()
                       if k in df_summary.columns}
    default_color_metric = (
        "ranking_score" if "ranking_score" in available_color
        else next(iter(available_color))
    )
    map_color_metric = st.sidebar.selectbox(
        "Метрика приоритета",
        options=list(available_color.keys()),
        format_func=lambda x: available_color[x],
        index=list(available_color.keys()).index(default_color_metric),
        help="Изменение цветовой метрики для карты на вкладке «Обзор»",
    )
    st.session_state["map_color_mode"] = map_color_mode
    st.session_state["map_color_metric"] = map_color_metric

    # Сортировка
    st.sidebar.subheader("Сортировка")
    sort_options = {
        "ranking_score": "По ranking score",
        "max_event_risk": "По макс. риску события",
        "n_events": "По числу событий",
        "risk_p99": "По P99 risk",
        "duration_sec": "По длительности",
    }
    available_sort = {k: v for k, v in sort_options.items() if k in df_summary.columns}
    sort_by = st.sidebar.selectbox(
        "Параметр",
        options=list(available_sort.keys()),
        format_func=lambda x: available_sort[x],
        index=0,
    )
    sort_asc = st.sidebar.checkbox("По возрастанию", value=False)

    # Применение фильтров
    mask = df_summary["split"].isin(selected_splits)
    if "ranking_score" in df_summary.columns:
        mask &= (df_summary["ranking_score"] >= rs_min) & (df_summary["ranking_score"] <= rs_max)
    if "n_events" in df_summary.columns:
        mask &= (df_summary["n_events"] >= ne_min) & (df_summary["n_events"] <= ne_max)
    if only_clean and "dq_hard_share" in df_summary.columns:
        mask &= df_summary["dq_hard_share"] < 0.05

    filtered = df_summary[mask].sort_values(sort_by, ascending=sort_asc).reset_index(drop=True)

    st.sidebar.markdown(f"**{len(filtered):,}** / {len(df_summary):,} рейсов")

    # Подвал боковой панели
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Данные:** EUROCONTROL PRC 2024  \n"
        "**Модели:** IF, HDBSCAN/GLOSH, LSTM-AE  \n"
        f"**Рейсов:** {len(df_summary):,}"
    )

    return filtered


# ===== ВКЛАДКА 1: ОБЗОР =====
def render_overview(df_summary, df_filtered, df_events, eval_report):
    """Главная страница: KPI, карта, распределения, top flights."""

    # --- Ключевые метрики ---
    col1, col2, col3, col4, col5 = st.columns(5)

    n_total = len(df_summary)
    n_with_events = int((df_summary["n_events"] > 0).sum()) if "n_events" in df_summary.columns else 0
    n_total_events = len(df_events)
    n_high_evt = int((df_events["risk_max"] >= RISK_MEDIUM_MAX).sum()) if len(df_events) > 0 else 0
    median_rank = float(df_summary["ranking_score"].median()) if "ranking_score" in df_summary.columns else 0.0

    col1.metric("Всего рейсов", f"{n_total:,}")
    col2.metric(
        "С событиями",
        f"{n_with_events:,}",
        delta=f"{n_with_events / n_total * 100:.1f}%" if n_total > 0 else None,
        delta_color="off",
    )
    col3.metric("Всего событий", f"{n_total_events:,}")
    col4.metric(
        "События ≥ P99 (high)",
        f"{n_high_evt:,}",
        delta=f"{n_high_evt / n_total_events * 100:.1f}%" if n_total_events > 0 else None,
        delta_color="off",
    )
    col5.metric("Медианный ranking", f"{median_rank:.3f}")

    # --- Карта ---
    st.subheader("Карта рейсов")

    if "origin_lat" in df_filtered.columns and "origin_lon" in df_filtered.columns:
        map_df = df_filtered.dropna(subset=["origin_lat", "origin_lon"]).copy()
        # Размер: события + базовый размер
        map_df["map_size"] = map_df.get("n_events", 1).fillna(0) + 1

        # Метрика цвета (из боковой панели или по умолчанию)
        color_mode = st.session_state.get("map_color_mode", "triage")
        color_metric = st.session_state.get("map_color_metric", "ranking_score")
        if color_metric not in map_df.columns:
            color_metric = "ranking_score"

        if color_mode == "triage":
            map_df = add_map_triage_columns(map_df, color_metric)
            triage_counts = map_df["map_triage_label"].value_counts(sort=False)
            triage_caption = "  ·  ".join(
                f"{label}: {count:,}"
                for label, count in triage_counts.items()
                if count > 0
            )
            st.caption(
                f"**Цвет:** группы приоритета по {color_metric}  ·  "
                f"**Размер:** число событий  ·  {triage_caption}"
            )

            fig_map = px.scatter_mapbox(
                map_df,
                lat="origin_lat",
                lon="origin_lon",
                color="map_triage_label",
                size="map_size",
                size_max=20,
                color_discrete_map=MAP_TRIAGE_COLORS,
                category_orders={
                    "map_triage_label": [
                        MAP_TRIAGE_LABELS[key] for key in MAP_TRIAGE_ORDER
                    ]
                },
                hover_name="flight_id",
                custom_data=["flight_id"],
                hover_data={
                    "map_triage_label": True,
                    "map_priority_pct": ":.1f",
                    "ranking_score": ":.3f",
                    "max_event_risk": ":.3f",
                    "n_events": True,
                    "split": True,
                    "dq_hard_share": ":.3f",
                    "origin_lat": False,
                    "origin_lon": False,
                    "map_size": False,
                },
                labels={
                    "map_triage_label": "Группа",
                    "map_priority_pct": "Перцентиль приоритета",
                },
                zoom=3,
                height=500,
                mapbox_style="carto-positron",
            )
        else:
            # Сырая шкала нужна для диагностики, но на обзорной карте она часто сжимается в красный хвост.
            color_range = get_map_color_range(map_df[color_metric], color_metric)
            st.caption(
                f"**Цвет:** {color_metric}  ·  "
                f"**Размер:** число событий  ·  "
                f"диапазон шкалы: [{color_range[0]:.3f}, {color_range[1]:.3f}]"
            )

            fig_map = px.scatter_mapbox(
                map_df,
                lat="origin_lat",
                lon="origin_lon",
                color=color_metric,
                size="map_size",
                size_max=20,
                color_continuous_scale="RdYlGn_r",
                range_color=color_range,
                hover_name="flight_id",
                custom_data=["flight_id"],
                hover_data={
                    "ranking_score": ":.3f",
                    "max_event_risk": ":.3f",
                    "n_events": True,
                    "split": True,
                    "dq_hard_share": ":.3f",
                    "origin_lat": False,
                    "origin_lon": False,
                    "map_size": False,
                },
                zoom=3,
                height=500,
                mapbox_style="carto-positron",
            )
        fig_map.update_layout(margin=dict(l=0, r=0, t=20, b=0))

        event_obj = st.plotly_chart(
            fig_map,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="map_overview",
        )

        if event_obj.selection.points:
            selected_point = event_obj.selection.points[0]
            if selected_point.get("customdata"):
                fid = int(selected_point["customdata"][0])
            else:
                point_idx = selected_point["point_index"]
                fid = int(map_df.iloc[point_idx]["flight_id"])
            if fid != st.session_state.get("_prev_map_overview_select"):
                st.session_state["_prev_map_overview_select"] = fid
                st.session_state["map_selected_flight_id"] = fid
            st.info(f"Выбран рейс **{fid}** на карте. Перейдите на вкладку «Детализация рейса».")
    else:
        st.warning("Колонки origin_lat/origin_lon отсутствуют в summary.")

    # --- Распределения, строка 1 ---
    col_l1, col_r1 = st.columns(2)

    with col_l1:
        with st.container(border=True):
            st.subheader("Распределение точек по уровню риска")
            render_risk_level_distribution(eval_report)

    with col_r1:
        with st.container(border=True):
            st.subheader("Число событий на рейс (только с событиями)")
            if "n_events" in df_filtered.columns:
                with_evt = df_filtered[df_filtered["n_events"] > 0]
                if len(with_evt) > 0:
                    fig = px.histogram(
                        with_evt,
                        x="n_events",
                        nbins=30,
                        color_discrete_sequence=["#3498db"],
                        labels={"n_events": "События / рейс"},
                    )
                    fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Нет рейсов с событиями в текущих фильтрах.")

    # --- Распределения, строка 2: категории ---
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        with st.container(border=True):
            st.subheader("Категории событий")
            if len(df_events) > 0:
                cat_counts = df_events["category"].value_counts().reset_index()
                cat_counts.columns = ["category", "count"]
                cat_counts["label"] = cat_counts["category"].map(CATEGORY_LABELS).fillna(cat_counts["category"])

                fig = px.pie(
                    cat_counts,
                    values="count",
                    names="label",
                    color="category",
                    color_discrete_map=CATEGORY_COLORS,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        with st.container(border=True):
            st.subheader("События по фазам полёта")
            if len(df_events) > 0 and "phase_name" in df_events.columns:
                phase_counts = df_events["phase_name"].value_counts().reset_index()
                phase_counts.columns = ["phase", "count"]
                phase_counts["label"] = phase_counts["phase"].map(PHASE_LABELS).fillna(phase_counts["phase"])

                fig = px.bar(
                    phase_counts,
                    x="label",
                    y="count",
                    color_discrete_sequence=["#3498db"],
                    labels={"label": "Фаза", "count": "События"},
                )
                fig.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=80))
                fig.update_xaxes(title="", tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

    # --- Таблица топ рейсов ---
    st.subheader(f"Топ рейсов (top {min(50, len(df_filtered))})")

    display_cols_map = {
        "flight_id": "ID рейса",
        "split": "Split",
        "ranking_score": "Ranking",
        "max_event_risk": "Макс. риск события",
        "risk_p99": "P99 risk",
        "n_events": "События",
        "duration_sec": "Длит. (мин)",
        "dq_hard_share": "DQ hard %",
        "dq_soft_share": "DQ soft %",
        "feature_quality_share": "FQ %",
    }
    avail_cols = [c for c in display_cols_map if c in df_filtered.columns]
    display = df_filtered[avail_cols].head(50).copy()

    if "duration_sec" in display.columns:
        display["duration_sec"] = (display["duration_sec"] / 60).round(1)
    for col in ["ranking_score", "max_event_risk", "risk_p99"]:
        if col in display.columns:
            display[col] = display[col].round(4)
    for col in ["dq_hard_share", "dq_soft_share", "feature_quality_share"]:
        if col in display.columns:
            display[col] = (display[col] * 100).round(2)

    display = display.rename(columns=display_cols_map)
    st.dataframe(display, use_container_width=True, height=400)


# ===== ВКЛАДКА 2: ДЕТАЛИЗАЦИЯ РЕЙСА =====
def render_flight_detail(flight_id, df_summary, df_events):
    """Детальный анализ одного рейса."""

    # Загрузка точек
    progress = st.progress(0, text=f"Загрузка телеметрии рейса {flight_id}...")
    t0 = time.time()
    fdata = load_flight_points(flight_id)
    progress.empty()
    load_time = time.time() - t0

    if len(fdata) == 0:
        st.error(f"Данные для рейса {flight_id} не найдены.")
        return

    flight_info = df_summary[df_summary["flight_id"] == flight_id]
    if len(flight_info) == 0:
        st.error(f"Метаданные для рейса {flight_id} не найдены в summary.")
        return
    flight_info = flight_info.iloc[0]

    flight_events = df_events[df_events["flight_id"] == flight_id].sort_values("event_start_ts")

    # --- Ключевые метрики ---
    st.subheader(f"Рейс {flight_id}  ·  {SPLIT_LABELS.get(flight_info['split'], flight_info['split'])}")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Длительность", f"{flight_info['duration_sec'] / 60:.0f} мин")
    col2.metric("Точек", f"{len(fdata):,}")
    col3.metric(
        "Ranking",
        f"{flight_info.get('ranking_score', float('nan')):.3f}"
        if pd.notna(flight_info.get("ranking_score")) else "—",
    )

    max_evt_risk = flight_info.get("max_event_risk")
    col4.metric(
        "Макс. риск события",
        f"{max_evt_risk:.3f}" if pd.notna(max_evt_risk) else "—",
    )

    col5.metric("Событий", f"{int(flight_info.get('n_events', 0)):,}")
    col6.metric("Загрузка", f"{load_time:.2f} с")

    # Строка информации о качестве данных
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric(
        "DQ hard %",
        f"{flight_info.get('dq_hard_share', 0) * 100:.2f}%",
        help="Stale altitude / threshold violations / gaps"
    )
    col_d2.metric(
        "DQ soft %",
        f"{flight_info.get('dq_soft_share', 0) * 100:.2f}%",
        help="Suspicious derivative dynamics"
    )
    col_d3.metric(
        "Feature quality %",
        f"{flight_info.get('feature_quality_share', 0) * 100:.2f}%",
        help="Phase inconsistency / energy extreme"
    )
    col_d4.metric(
        "Чистых точек %",
        f"{flight_info.get('clean_point_share', 0) * 100:.2f}%"
    )

    # --- Карта траектории ---
    st.subheader("Траектория (цвет = risk_score)")

    fig_traj = px.scatter_mapbox(
        fdata,
        lat="latitude",
        lon="longitude",
        color="risk_score",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 1.0],
        hover_data={
            "altitude": ":.0f",
            "groundspeed": ":.0f",
            "risk_score": ":.3f",
            "phase_name": True,
            "latitude": False,
            "longitude": False,
        },
        zoom=4,
        height=400,
        mapbox_style="carto-positron",
    )
    fig_traj.update_traces(marker=dict(size=4))
    fig_traj.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_traj, use_container_width=True)

    # --- Многопанельная временная шкала ---
    st.subheader("Временная шкала")

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.17, 0.14, 0.14, 0.20, 0.20, 0.15],
        subplot_titles=(
            "Высота (фут)",
            "Путевая скорость (уз)",
            "Вертикальная скорость (фут/мин)",
            "Phase percentile моделей (P99 — горизонтальная линия)",
            "Ensemble и Risk score (пороги 0.90 и 0.99)",
            "DQ subflags (отметки point-level флагов)",
        ),
    )

    # Строка 1: высота
    fig.add_trace(go.Scatter(
        x=fdata["t_min"], y=fdata["altitude"],
        mode="lines", line=dict(color="#3498db", width=1),
        name="Высота", showlegend=False,
    ), row=1, col=1)

    # Строка 2: путевая скорость
    fig.add_trace(go.Scatter(
        x=fdata["t_min"], y=fdata["groundspeed"],
        mode="lines", line=dict(color="#2ecc71", width=1),
        name="Скорость", showlegend=False,
    ), row=2, col=1)

    # Строка 3: вертикальная скорость
    fig.add_trace(go.Scatter(
        x=fdata["t_min"], y=fdata["vertical_rate"],
        mode="lines", line=dict(color="#9b59b6", width=1),
        name="Верт. скорость", showlegend=False,
    ), row=3, col=1)

    # Строка 4: скоринги трёх моделей
    for col_name, label, color in [
        ("if_score_phase_pct", "IF", MODEL_COLORS["IF"]),
        ("hdb_score_phase_pct", "HDBSCAN", MODEL_COLORS["HDBSCAN"]),
        ("lstm_score_max_phase_pct", "LSTM-AE", MODEL_COLORS["LSTM-AE"]),
    ]:
        if col_name in fdata.columns:
            fig.add_trace(go.Scatter(
                x=fdata["t_min"], y=fdata[col_name],
                mode="lines", line=dict(color=color, width=1.3),
                name=label, opacity=0.85,
                legendgroup="models",
            ), row=4, col=1)

    # Линия P99
    fig.add_hline(y=0.99, line_dash="dash", line_color="red",
                  opacity=0.4, row=4, col=1)

    # Строка 5: ансамбль + risk_score
    fig.add_trace(go.Scatter(
        x=fdata["t_min"], y=fdata["ensemble_score"],
        mode="lines", line=dict(color="#34495e", width=1),
        name="Ensemble", legendgroup="risk",
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=fdata["t_min"], y=fdata["risk_score"],
        mode="lines", line=dict(color="#e74c3c", width=1.5),
        name="Risk score", legendgroup="risk",
        fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.15)",
    ), row=5, col=1)

    fig.add_hline(y=RISK_LOW_MAX, line_dash="dash",
                  line_color="#f39c12", opacity=0.5, row=5, col=1)
    fig.add_hline(y=RISK_MEDIUM_MAX, line_dash="dash",
                  line_color="#e74c3c", opacity=0.5, row=5, col=1)

    # Строка 6: подфлаги качества данных как маркерные ряды
    dq_subflags = [
        ("gap_flag", "gap > 5s", "#3498db"),
        ("dq_derivative_bad", "derivative", "#e67e22"),
        ("phase_inconsistent", "phase", "#9b59b6"),
        ("energy_deviation_extreme", "energy", "#e74c3c"),
    ]

    for i, (col, label, color) in enumerate(dq_subflags):
        if col in fdata.columns:
            bad = fdata[fdata[col] == 1]
            if len(bad) > 0:
                fig.add_trace(go.Scatter(
                    x=bad["t_min"], y=[i] * len(bad),
                    mode="markers",
                    marker=dict(color=color, size=6, symbol="square"),
                    name=label, legendgroup="dq",
                    showlegend=True,
                ), row=6, col=1)

    fig.update_yaxes(
        ticktext=[f[1] for f in dq_subflags],
        tickvals=list(range(len(dq_subflags))),
        range=[-0.5, len(dq_subflags) - 0.5],
        row=6, col=1,
    )

    # --- Подсветка событий вертикальными полосами на всех строках ---
    if len(flight_events) > 0:
        flight_start_ts = fdata["timestamp"].min()
        for _, evt in flight_events.iterrows():
            evt_start_t = (evt["event_start_ts"] - flight_start_ts).total_seconds() / 60
            evt_end_t = (evt["event_end_ts"] - flight_start_ts).total_seconds() / 60
            cat_color = CATEGORY_COLORS.get(evt["category"], "#95a5a6")
            for row_i in range(1, 7):
                fig.add_vrect(
                    x0=evt_start_t, x1=evt_end_t,
                    fillcolor=cat_color, opacity=0.12,
                    line_width=0, row=row_i, col=1,
                )

    fig.update_xaxes(title_text="Время (мин)", row=6, col=1)
    fig.update_layout(
        height=1100,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.03,
            xanchor="center", x=0.5,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    st.plotly_chart(fig, use_container_width=True)

    # Легенда для подсветки событий
    if len(flight_events) > 0:
        st.caption("**Цветные полосы — события (по категориям):**")
        cols = st.columns(4)
        cats_present = flight_events["category"].unique()
        for i, cat in enumerate(["potential_operational_anomaly", "mixed_or_derived_dynamics",
                                  "likely_data_quality_artifact", "likely_feature_phase_artifact"]):
            with cols[i]:
                color = CATEGORY_COLORS[cat]
                label = CATEGORY_LABELS[cat]
                marker = "■" if cat in cats_present else "□"
                opacity = "1.0" if cat in cats_present else "0.4"
                st.markdown(
                    f"<span style='color:{color};opacity:{opacity};font-size:1.2em;'>{marker}</span> "
                    f"<span style='opacity:{opacity};'>{label}</span>",
                    unsafe_allow_html=True
                )

    # --- Панель энергии ---
    st.subheader("Энергетические признаки")
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        if "energy_ratio" in fdata.columns:
            fig_er = go.Figure()
            fig_er.add_trace(go.Scatter(
                x=fdata["t_min"], y=fdata["energy_ratio"],
                mode="lines", line=dict(color="#e67e22", width=1),
                name="Energy ratio",
            ))
            fig_er.update_layout(
                title="Коэффициент энергии (KE / E_total)",
                xaxis_title="Время (мин)",
                yaxis_title="Коэффициент",
                height=300,
                margin=dict(l=50, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_er, use_container_width=True)

    with col_e2:
        if "energy_deviation" in fdata.columns:
            fig_ed = go.Figure()
            fig_ed.add_trace(go.Scatter(
                x=fdata["t_min"], y=fdata["energy_deviation"],
                mode="lines", line=dict(color="#c0392b", width=1),
                name="Energy deviation",
            ))
            fig_ed.add_hline(y=10, line_dash="dash", line_color="red", opacity=0.4)
            fig_ed.add_hline(y=-10, line_dash="dash", line_color="red", opacity=0.4)
            fig_ed.add_hrect(y0=-10, y1=10, fillcolor="green",
                            opacity=0.04, line_width=0)
            fig_ed.update_layout(
                title="Отклонение энергии от фазовой нормы",
                xaxis_title="Время (мин)",
                yaxis_title="Отклонение",
                height=300,
                margin=dict(l=50, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_ed, use_container_width=True)

    # --- Таблица событий для этого рейса ---
    if len(flight_events) > 0:
        st.subheader(f"События рейса ({len(flight_events)})")

        evt_cols_map = {
            "event_start_ts": "Начало",
            "duration_sec": "Длит. (с)",
            "phase_name": "Фаза",
            "category": "Категория",
            "risk_max": "Макс. risk",
            "ensemble_max": "Макс. ensemble",
            "if_max": "IF max",
            "hdb_max": "HDB max",
            "lstm_max": "LSTM max",
            "n_models_above_thresh": "Моделей ≥ P99",
            "dq_hard_share": "DQ hard %",
            "dq_soft_share": "DQ soft %",
            "feature_quality_share": "FQ %",
        }
        avail_evt_cols = [c for c in evt_cols_map if c in flight_events.columns]
        evt_display = flight_events[avail_evt_cols].copy()

        if "category" in evt_display.columns:
            evt_display["category"] = evt_display["category"].map(CATEGORY_LABELS).fillna(evt_display["category"])
        if "phase_name" in evt_display.columns:
            evt_display["phase_name"] = evt_display["phase_name"].map(PHASE_LABELS).fillna(evt_display["phase_name"])
        if "duration_sec" in evt_display.columns:
            evt_display["duration_sec"] = evt_display["duration_sec"].round(0).astype(int)

        for col in ["risk_max", "ensemble_max", "if_max", "hdb_max", "lstm_max"]:
            if col in evt_display.columns:
                evt_display[col] = evt_display[col].round(3)
        for col in ["dq_hard_share", "dq_soft_share", "feature_quality_share"]:
            if col in evt_display.columns:
                evt_display[col] = (evt_display[col] * 100).round(2)

        evt_display = evt_display.rename(columns=evt_cols_map)
        st.dataframe(evt_display, use_container_width=True, height=300)
    else:
        st.success("В этом рейсе не зафиксировано событий выше порога P99.")


# ===== ВКЛАДКА 3: СОБЫТИЯ =====
def render_events_tab(df_events):
    """Глобальная таблица событий с фильтрами и breakdown."""
    st.subheader("Все события")

    if len(df_events) == 0:
        st.warning("Таблица событий пуста.")
        return

    # --- Фильтры ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cat_opts = sorted(df_events["category"].unique().tolist())
        cat_filter = st.multiselect(
            "Категория",
            options=cat_opts,
            default=cat_opts,
            format_func=lambda x: CATEGORY_LABELS.get(x, x),
        )

    with col2:
        split_opts = sorted(df_events["split"].unique().tolist())
        split_filter = st.multiselect(
            "Подвыборка",
            options=split_opts,
            default=split_opts,
            format_func=lambda x: SPLIT_LABELS.get(x, x),
        )

    with col3:
        phase_opts = sorted(df_events["phase_name"].unique().tolist())
        phase_filter = st.multiselect(
            "Фаза",
            options=phase_opts,
            default=phase_opts,
            format_func=lambda x: PHASE_LABELS.get(x, x),
        )

    with col4:
        models_min = st.slider(
            "Мин. моделей ≥ P99",
            min_value=0, max_value=3, value=0,
            help="Сколько моделей независимо подтвердили событие (out of 3)",
        )

    # Применение фильтров
    mask = (
        df_events["category"].isin(cat_filter)
        & df_events["split"].isin(split_filter)
        & df_events["phase_name"].isin(phase_filter)
        & (df_events["n_models_above_thresh"] >= models_min)
    )
    filtered = df_events[mask].sort_values("risk_max", ascending=False).reset_index(drop=True)

    st.markdown(f"**{len(filtered):,}** / {len(df_events):,} событий")

    # --- Визуализации ---
    col_a, col_b = st.columns([1, 1.5])

    with col_a:
        with st.container(border=True):
            st.subheader("По категориям")
            cat_counts = filtered["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            cat_counts["label"] = cat_counts["category"].map(CATEGORY_LABELS).fillna(cat_counts["category"])

            fig = px.bar(
                cat_counts,
                x="count",
                y="label",
                orientation="h",
                color="category",
                color_discrete_map=CATEGORY_COLORS,
            )
            fig.update_layout(
                height=350,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=30),
            )
            fig.update_yaxes(title="")
            fig.update_xaxes(title="Событий")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        with st.container(border=True):
            st.subheader("Фаза × категория (heatmap)")
            crosstab = pd.crosstab(filtered["phase_name"], filtered["category"])
            if not crosstab.empty:
                fig = px.imshow(
                    crosstab.values,
                    x=[CATEGORY_LABELS.get(c, c)[:20] for c in crosstab.columns],
                    y=[PHASE_LABELS.get(p, p) for p in crosstab.index],
                    color_continuous_scale="Reds",
                    text_auto=True,
                    aspect="auto",
                )
                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=80),
                )
                fig.update_xaxes(tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)

    # --- Разбор согласия моделей ---
    with st.container(border=True):
        st.subheader("Согласие моделей на событиях")
        agreement = filtered["n_models_above_thresh"].value_counts().sort_index().reset_index()
        agreement.columns = ["n_models", "count"]
        agreement["label"] = agreement["n_models"].apply(
            lambda x: f"{x} модел{'ь' if x == 1 else ('и' if x in (2, 3, 4) else 'ей')}"
        )

        fig = px.bar(
            agreement,
            x="label",
            y="count",
            color="n_models",
            color_continuous_scale="RdYlGn",
            text="count",
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=40))
        fig.update_xaxes(title="Число моделей с phase_pct ≥ 0.99")
        fig.update_yaxes(title="События")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # --- Таблица событий ---
    st.subheader(f"Таблица событий (top {min(200, len(filtered))})")

    evt_cols_map = {
        "flight_id": "ID рейса",
        "split": "Split",
        "phase_name": "Фаза",
        "category": "Категория",
        "duration_sec": "Длит. (с)",
        "risk_max": "Макс. risk",
        "ensemble_max": "Макс. ensemble",
        "if_max": "IF",
        "hdb_max": "HDB",
        "lstm_max": "LSTM",
        "n_models_above_thresh": "Моделей",
        "dq_hard_share": "DQ hard %",
        "dq_soft_share": "DQ soft %",
        "feature_quality_share": "FQ %",
    }
    avail_evt_cols = [c for c in evt_cols_map if c in filtered.columns]
    display = filtered[avail_evt_cols].head(200).copy()

    if "category" in display.columns:
        display["category"] = display["category"].map(CATEGORY_LABELS).fillna(display["category"])
    if "phase_name" in display.columns:
        display["phase_name"] = display["phase_name"].map(PHASE_LABELS).fillna(display["phase_name"])
    if "split" in display.columns:
        display["split"] = display["split"].map(SPLIT_LABELS).fillna(display["split"])
    if "duration_sec" in display.columns:
        display["duration_sec"] = display["duration_sec"].round(0).astype(int)

    for col in ["risk_max", "ensemble_max", "if_max", "hdb_max", "lstm_max"]:
        if col in display.columns:
            display[col] = display[col].round(3)
    for col in ["dq_hard_share", "dq_soft_share", "feature_quality_share"]:
        if col in display.columns:
            display[col] = (display[col] * 100).round(1)

    # Сохраняем flight_id для возможного клика
    display_with_fid = display.rename(columns=evt_cols_map)

    event_obj = st.dataframe(
        display_with_fid,
        use_container_width=True,
        height=500,
        on_select="rerun",
        selection_mode="single-row",
        key="events_table",
    )

    if event_obj.selection.rows:
        row_idx = event_obj.selection.rows[0]
        fid = int(filtered.iloc[row_idx]["flight_id"])
        if fid != st.session_state.get("_prev_evt_select"):
            st.session_state["_prev_evt_select"] = fid
            st.session_state["events_selected_flight_id"] = fid
        st.info(f"Выбран рейс **{fid}** из событий. Перейдите на вкладку «Детализация рейса».")


# ===== ГЛАВНОЕ ПРИЛОЖЕНИЕ =====
def main():
    # Проверка файлов
    missing_files = []
    for path, label in [
        (SUMMARY_FILE, "dashboard_flight_summary.parquet"),
        (POINTS_FILE, "dashboard_points_v3.parquet"),
        (EVENTS_FILE, "events_v3.parquet"),
        (REPORT_FILE, "evaluation_report_v3.json"),
    ]:
        if not os.path.exists(path):
            missing_files.append((path, label))

    if missing_files:
        st.error("Не найдены следующие файлы:")
        for path, label in missing_files:
            st.code(f"{label} (ожидался: {path})")
        st.info("Поместите файлы в директорию скрипта или измените DATA_DIR.")
        return

    # Загрузка
    df_summary = load_flight_summary()
    df_events = load_events()
    eval_report = load_evaluation_report()

    # Заголовок
    st.title("Раннее предупреждение нестабильности воздушного движения")
    st.caption(
        "Система мониторинга на основе ансамбля моделей "
        "(Isolation Forest + HDBSCAN/GLOSH + LSTM-Autoencoder). "
        "Risk score — калиброванный относительный ранг "
        "относительно clean calibration baseline (P99 ≈ 0.97)."
    )

    # Боковая панель
    df_filtered = render_sidebar(df_summary)

    # Вкладки
    tab_overview, tab_detail, tab_events = st.tabs(
        ["📊 Обзор", "🛬 Детализация рейса", "⚠️ События"]
    )

    with tab_overview:
        render_overview(df_summary, df_filtered, df_events, eval_report)

    with tab_detail:
        if len(df_filtered) == 0:
            st.info("Нет рейсов, удовлетворяющих фильтрам. Измените настройки.")
        else:
            # Обработка заранее выбранного рейса
            preselected_id = None
            if "map_selected_flight_id" in st.session_state:
                preselected_id = st.session_state.pop("map_selected_flight_id")
            elif "events_selected_flight_id" in st.session_state:
                preselected_id = st.session_state.pop("events_selected_flight_id")

            if preselected_id is not None:
                st.session_state["flight_id_input"] = str(preselected_id)
                st.session_state["flight_select_mode_v3"] = "ID рейса"

            mode = st.radio(
                "Способ выбора рейса",
                ["Топ по ranking", "ID рейса"],
                horizontal=True,
                key="flight_select_mode_v3",
            )

            top_by_rank = df_filtered.head(20)["flight_id"].tolist()

            if mode == "Топ по ranking":
                opts = []
                for fid in top_by_rank:
                    row = df_summary[df_summary["flight_id"] == fid]
                    if len(row) == 0:
                        continue
                    row = row.iloc[0]
                    opts.append(
                        f"{fid}  ·  ranking={row.get('ranking_score', 0):.3f}  "
                        f"·  events={int(row.get('n_events', 0))}  "
                        f"·  {SPLIT_LABELS.get(row.get('split', ''), row.get('split', ''))}"
                    )
                if opts:
                    selected = st.selectbox("Рейс", opts, index=0)
                    selected_id = int(selected.split(" ")[0])
                    render_flight_detail(selected_id, df_summary, df_events)
                else:
                    st.warning("Нет рейсов в текущей выборке.")
            else:
                id_input = st.text_input(
                    "ID рейса",
                    value=str(df_filtered["flight_id"].iloc[0]),
                    key="flight_id_input",
                )
                try:
                    selected_id = int(id_input)
                    if selected_id not in df_summary["flight_id"].values:
                        st.warning(f"Рейс {selected_id} не найден в данных.")
                        return
                    render_flight_detail(selected_id, df_summary, df_events)
                except ValueError:
                    st.warning("Введите целое число.")

    with tab_events:
        render_events_tab(df_events)


if __name__ == "__main__":
    main()
