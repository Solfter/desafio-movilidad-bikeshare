"""Dashboard interactivo (Plotly Dash) de demanda de bicicletas.

Secciones:
- KPIs generales.
- Exploración: serie temporal (filtrable), heatmap hora×día, clima vs demanda.
- Modelo: predicho vs real e importancia de variables.
- Tipos de día: segmentación no supervisada (K-Means sobre perfiles horarios).
- Predicción what-if: consume la API ``/predict`` (con fallback al modelo local).

Ejecutar:
    uv run python -m bikeshare.dashboard.app
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, callback, dcc, html

from bikeshare import config
from bikeshare.etl.load import load_processed
from bikeshare.features import FEATURE_COLUMNS
from bikeshare.models import clustering
from bikeshare.models.predict import predict as local_predict

# ---------------------------------------------------------------------------
# Datos y artefactos (se cargan una vez)
# ---------------------------------------------------------------------------
DF = load_processed().sort_values("datetime").reset_index(drop=True)
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
CLIMA = {1: "Despejado", 2: "Nublado", 3: "Lluvia/nieve", 4: "Tormenta"}
CLUSTERS = clustering.compute_clusters(DF)

try:
    IMPORTANCES = pd.read_csv(config.IMPORTANCES_FILE)
except FileNotFoundError:
    IMPORTANCES = pd.DataFrame({"feature": [], "importance": []})

try:
    import json

    METRICS = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
except FileNotFoundError:
    METRICS = {"best_model": "n/d", "metrics": {}}

ACCENT = "#2b8a3e"
TEMPLATE = "plotly_white"

# Colores por tipo de día (misma paleta categórica del EDA / notebook 02)
COLORES_TIPO = {
    clustering.LABORAL: "#2b8a3e",
    clustering.NO_LABORAL: "#7048e8",
    clustering.LABORAL_CALIDO: "#2b8a3e",
    clustering.LABORAL_FRIO: "#1971c2",
    clustering.LABORAL_LLUVIOSO: "#e8590c",
}
TIPO_ORDEN = [
    clustering.LABORAL,
    clustering.LABORAL_CALIDO,
    clustering.LABORAL_FRIO,
    clustering.LABORAL_LLUVIOSO,
    clustering.NO_LABORAL,
]


# ---------------------------------------------------------------------------
# Componentes reutilizables
# ---------------------------------------------------------------------------
def kpi_card(title: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "0.8rem", "color": "#666"}),
            html.Div(value, style={"fontSize": "1.8rem", "fontWeight": 700, "color": ACCENT}),
            html.Div(subtitle, style={"fontSize": "0.75rem", "color": "#999"}),
        ],
        style={
            "background": "white",
            "borderRadius": "12px",
            "padding": "16px 20px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
            "flex": "1",
            "minWidth": "160px",
        },
    )


def kpi_row() -> html.Div:
    best = METRICS.get("best_model", "n/d")
    r2 = METRICS.get("metrics", {}).get(best, {}).get("r2")
    peak_hour = int(DF.groupby("hour")["cnt"].mean().idxmax())
    return html.Div(
        [
            kpi_card("Registros horarios", f"{len(DF):,}", "2011–2012, Washington DC"),
            kpi_card("Demanda media/hora", f"{DF['cnt'].mean():.0f}", "bicicletas"),
            kpi_card("Hora peak", f"{peak_hour}:00", "mayor demanda promedio"),
            kpi_card("Mejor modelo", best, f"R² = {r2:.3f}" if r2 else ""),
        ],
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "20px"},
    )


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_timeseries(dff: pd.DataFrame) -> go.Figure:
    daily = dff.set_index("datetime")["cnt"].resample("1D").sum().reset_index()
    fig = px.line(
        daily, x="datetime", y="cnt", template=TEMPLATE,
        labels={"datetime": "Fecha", "cnt": "Demanda diaria"},
    )
    fig.update_traces(line_color=ACCENT)
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), title="Demanda diaria total")
    return fig


def fig_heatmap(dff: pd.DataFrame) -> go.Figure:
    pivot = dff.pivot_table(index="hour", columns="dayofweek", values="cnt", aggfunc="mean")
    pivot = pivot.reindex(columns=range(7))
    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale="Viridis", template=TEMPLATE,
        labels=dict(x="Día", y="Hora", color="Demanda"),
    )
    fig.update_xaxes(tickmode="array", tickvals=list(range(7)), ticktext=DIAS)
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), title="Demanda media por hora y día")
    return fig


def fig_weather(dff: pd.DataFrame) -> go.Figure:
    sample = dff.sample(min(3000, len(dff)), random_state=42)
    fig = px.scatter(
        sample, x="temperature_2m", y="cnt", color="precipitation",
        color_continuous_scale="Blues_r", opacity=0.5, template=TEMPLATE,
        labels={"temperature_2m": "Temperatura (°C)", "cnt": "Demanda",
                "precipitation": "Precip (mm)"},
    )
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), title="Temperatura vs demanda")
    return fig


def fig_pred_vs_actual(hours: int = 168) -> go.Figure:
    tail = DF.tail(hours)
    pred = local_predict(tail[FEATURE_COLUMNS])
    fig = go.Figure()
    fig.add_scatter(x=tail["datetime"], y=tail["cnt"], name="Real", line=dict(color="#868e96"))
    fig.add_scatter(x=tail["datetime"], y=pred, name="Predicho", line=dict(color=ACCENT))
    fig.update_layout(
        template=TEMPLATE, margin=dict(l=40, r=20, t=30, b=40),
        title=f"Predicho vs real (últimas {hours} h)", legend=dict(orientation="h"),
    )
    return fig


def fig_importance() -> go.Figure:
    top = IMPORTANCES.head(12).sort_values("importance")
    fig = px.bar(
        top, x="importance", y="feature", orientation="h", template=TEMPLATE,
        labels={"importance": "Importancia", "feature": ""},
    )
    fig.update_traces(marker_color=ACCENT)
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40),
                      title="Importancia de variables (top 12)")
    return fig


def _tipos_presentes(k: int) -> list[str]:
    presentes = set(CLUSTERS.meta[f"tipo{k}"])
    return [t for t in TIPO_ORDEN if t in presentes]


def fig_cluster_profiles(k: int) -> go.Figure:
    tipo_col = f"tipo{k}"
    meta = CLUSTERS.meta
    fig = go.Figure()
    for nombre in _tipos_presentes(k):
        dias = meta.index[meta[tipo_col] == nombre]
        perfil = CLUSTERS.profiles.loc[dias].mean() * 100
        fig.add_scatter(x=list(perfil.index), y=perfil.values, name=nombre,
                        line=dict(color=COLORES_TIPO[nombre], width=2))
    fig.update_layout(
        template=TEMPLATE, margin=dict(l=40, r=20, t=30, b=40),
        title=f"Perfil horario promedio por cluster (k={k})",
        xaxis_title="Hora del día", yaxis_title="% de la demanda diaria",
        legend=dict(orientation="h"),
    )
    return fig


def fig_cluster_pca(k: int) -> go.Figure:
    tipo_col = f"tipo{k}"
    fig = px.scatter(
        CLUSTERS.pca, x="PC1", y="PC2", color=tipo_col,
        color_discrete_map=COLORES_TIPO, category_orders={tipo_col: TIPO_ORDEN},
        opacity=0.65, hover_data=["fecha"], template=TEMPLATE,
        labels={"PC1": f"PC1 ({CLUSTERS.var_pc1:.0f}% de la varianza)",
                "PC2": f"PC2 ({CLUSTERS.var_pc2:.0f}% de la varianza)"},
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40),
                      title="Días proyectados en 2 componentes principales (PCA)",
                      legend=dict(orientation="h"))
    return fig


def cluster_table(k: int) -> html.Table:
    resumen = clustering.cluster_summary(CLUSTERS.meta, f"tipo{k}")
    orden = {t: i for i, t in enumerate(TIPO_ORDEN)}
    resumen = resumen.sort_values(f"tipo{k}", key=lambda s: s.map(orden))
    cols = [
        (f"tipo{k}", "Tipo de día", lambda v: v),
        ("dias", "Días", lambda v: f"{v:.0f}"),
        ("pct_dia_laboral", "% laboral", lambda v: f"{v:.0%}"),
        ("pct_feriado", "% feriado", lambda v: f"{v:.0%}"),
        ("temp_media", "Temp. media (°C)", lambda v: f"{v:.1f}"),
        ("precip_total_media", "Precip. media (mm/día)", lambda v: f"{v:.1f}"),
        ("demanda_diaria_media", "Demanda diaria media", lambda v: f"{v:,.0f}"),
    ]
    th = {"textAlign": "left", "padding": "8px 12px", "borderBottom": "2px solid #dee2e6",
          "fontSize": "0.85rem", "color": "#666"}
    td = {"padding": "8px 12px", "borderBottom": "1px solid #eee", "fontSize": "0.9rem"}
    filas = []
    for _, row in resumen.iterrows():
        celdas = [
            html.Td(fmt(row[col]),
                    style={**td, "fontWeight": 600, "color": COLORES_TIPO[row[col]]}
                    if col == f"tipo{k}" else td)
            for col, _titulo, fmt in cols
        ]
        filas.append(html.Tr(celdas))
    return html.Table(
        [html.Thead(html.Tr([html.Th(t, style=th) for _c, t, _f in cols])),
         html.Tbody(filas)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__, title="Demanda de bicicletas")
server = app.server  # para despliegue WSGI (gunicorn/uvicorn workers)

_CARD = {"background": "white", "borderRadius": "12px", "padding": "12px",
         "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"}


def _slider(label, sid, mn, mx, val):
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem", "fontWeight": 600}),
            dcc.Slider(mn, mx, value=val, id=sid,
                       marks={mn: str(mn), mx: str(mx)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]
    )

app.layout = html.Div(
    [
        html.H1("🚲 Demanda de bicicletas — Dashboard", style={"color": ACCENT}),
        html.P(
            "Integra UCI Bike Sharing + clima real (Open-Meteo) + feriados. "
            "Modelo XGBoost para predecir la demanda horaria.",
            style={"color": "#555"},
        ),
        kpi_row(),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="📊 Exploración",
                    children=[
                        html.Div(
                            [
                                html.Label("Rango de fechas:"),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    min_date_allowed=DF["datetime"].min().date(),
                                    max_date_allowed=DF["datetime"].max().date(),
                                    start_date=DF["datetime"].min().date(),
                                    end_date=DF["datetime"].max().date(),
                                ),
                                html.Label("  Clima:", style={"marginLeft": "16px"}),
                                dcc.Dropdown(
                                    id="weather-filter",
                                    options=[{"label": v, "value": k} for k, v in CLIMA.items()],
                                    multi=True, placeholder="Todos", style={"width": "260px"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center", "gap": "8px",
                                   "flexWrap": "wrap", "margin": "12px 0"},
                        ),
                        dcc.Graph(id="ts-graph"),
                        html.Div(
                            [
                                html.Div(dcc.Graph(id="heatmap-graph"), style={"flex": "1"}),
                                html.Div(dcc.Graph(id="weather-graph"), style={"flex": "1"}),
                            ],
                            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="🤖 Modelo",
                    children=[
                        dcc.Graph(figure=fig_pred_vs_actual()),
                        dcc.Graph(figure=fig_importance()),
                    ],
                ),
                dcc.Tab(
                    label="🧩 Tipos de día",
                    children=[
                        html.Div(
                            [
                                html.P(
                                    "K-Means agrupa los días solo por la forma de su "
                                    "perfil horario (sin mirar el calendario): con k=2 "
                                    "redescubre la división laboral / no laboral y con "
                                    "k=4 aparecen además los días laborales de lluvia.",
                                    style={"color": "#555", "margin": "12px 0 8px"},
                                ),
                                html.Label("Granularidad:",
                                           style={"fontWeight": 600, "fontSize": "0.9rem"}),
                                dcc.RadioItems(
                                    id="cluster-k",
                                    options=[
                                        {"label": " k = 2 (laboral / no laboral)", "value": 2},
                                        {"label": " k = 4 (incluye clima)", "value": 4},
                                    ],
                                    value=2, inline=True,
                                    inputStyle={"marginLeft": "16px"},
                                ),
                            ],
                            style={"margin": "4px 0 8px"},
                        ),
                        html.Div(
                            [
                                html.Div(dcc.Graph(id="cluster-profile-graph"),
                                         style={"flex": "1", "minWidth": "420px"}),
                                html.Div(dcc.Graph(id="cluster-pca-graph"),
                                         style={"flex": "1", "minWidth": "420px"}),
                            ],
                            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                        ),
                        html.Div(id="cluster-summary",
                                 style={**_CARD, "marginTop": "16px", "overflowX": "auto"}),
                    ],
                ),
                dcc.Tab(
                    label="🔮 Predicción",
                    children=[
                        html.Div(
                            [
                                html.H3("Simula la demanda para una condición dada"),
                                html.Div(
                                    [
                                        _slider("Hora", "in-hour", 0, 23, 8),
                                        _slider("Mes", "in-month", 1, 12, 6),
                                        _slider("Día semana (0=Lun)", "in-weekday", 0, 6, 2),
                                        _slider("Temperatura (°C)", "in-temp", -5, 40, 24),
                                        _slider("Precipitación (mm)", "in-precip", 0, 20, 0),
                                        _slider("Humedad (%)", "in-hum", 0, 100, 55),
                                        _slider("Viento (km/h)", "in-wind", 0, 50, 12),
                                        _slider("Demanda reciente", "in-recent", 0, 600, 200),
                                    ],
                                    style={"display": "grid", "gap": "18px",
                                           "gridTemplateColumns":
                                               "repeat(auto-fit,minmax(240px,1fr))"},
                                ),
                                html.Button("Predecir", id="predict-btn", n_clicks=0,
                                            style={"marginTop": "16px", "padding": "10px 24px",
                                                   "background": ACCENT, "color": "white",
                                                   "border": "none", "borderRadius": "8px",
                                                   "fontSize": "1rem", "cursor": "pointer"}),
                                html.Div(id="predict-output", style={"marginTop": "20px",
                                                                     "fontSize": "1.4rem"}),
                            ],
                            style={**_CARD, "marginTop": "12px"},
                        ),
                    ],
                ),
            ]
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px",
           "fontFamily": "system-ui, sans-serif", "background": "#f5f6f8"},
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("ts-graph", "figure"),
    Output("heatmap-graph", "figure"),
    Output("weather-graph", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("weather-filter", "value"),
)
def update_exploration(start_date, end_date, weather):
    dff = DF
    if start_date:
        dff = dff[dff["datetime"] >= pd.Timestamp(start_date)]
    if end_date:
        dff = dff[dff["datetime"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if weather:
        dff = dff[dff["weathersit"].isin(weather)]
    if dff.empty:
        dff = DF.head(1)
    return fig_timeseries(dff), fig_heatmap(dff), fig_weather(dff)


@callback(
    Output("cluster-profile-graph", "figure"),
    Output("cluster-pca-graph", "figure"),
    Output("cluster-summary", "children"),
    Input("cluster-k", "value"),
)
def update_clusters(k):
    return fig_cluster_profiles(k), fig_cluster_pca(k), cluster_table(k)


@callback(
    Output("predict-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("in-hour", "value"), State("in-month", "value"), State("in-weekday", "value"),
    State("in-temp", "value"), State("in-precip", "value"), State("in-hum", "value"),
    State("in-wind", "value"), State("in-recent", "value"),
    prevent_initial_call=True,
)
def do_predict(_n, hour, month, weekday, temp, precip, hum, wind, recent):
    payload = {
        "hour": hour, "month": month, "weekday": weekday,
        "season": (month % 12 // 3) + 1, "yr": 1,
        "workingday": int(weekday < 5), "is_holiday": 0,
        "weathersit": 3 if precip > 1 else 1,
        "temperature_2m": temp, "precipitation": precip,
        "relative_humidity_2m": hum, "wind_speed_10m": wind, "recent_demand": recent,
    }
    # Intenta la API; si no está disponible, usa el modelo local.
    try:
        r = requests.post(f"{config.API_URL}/predict", json=payload, timeout=3)
        r.raise_for_status()
        demand = r.json()["predicted_demand"]
        fuente = "API"
    except requests.RequestException:
        from bikeshare.features import single_feature_row

        demand = round(float(local_predict(single_feature_row(**payload))[0]), 1)
        fuente = "modelo local"
    return html.Span(
        [html.B(f"Demanda estimada: {demand:.0f} bicicletas/hora "),
         html.Span(f"(vía {fuente})", style={"color": "#999", "fontSize": "0.9rem"})]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
